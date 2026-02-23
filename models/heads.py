import torch
import torch.nn as nn


def gumbel_sigmoid_sample(logits, tau=1.0):
    """
    Gumbel-Sigmoid straight-through estimator for binary patch selection.

    Compared to plain STE (identity backward):
    - Adds Gumbel noise for exploration (patches near boundary get a chance to flip)
    - Backward gradient is σ(1-σ)/τ — bounded, never causes gradient explosion
    - Gradient naturally attenuates for patches far from the decision boundary
      (high-confidence keep/drop get small gradient; uncertain patches get large gradient)
    - As τ → 0: approaches deterministic STE behavior
    - As τ → ∞: very soft, uniform gradient across all patches

    Args:
        logits: (B, N) patch importance logits
        tau: temperature scalar, annealed from tau_start → tau_end during training

    Returns:
        mask: (B, N) float, hard binary in forward / soft Gumbel-sigmoid in backward
    """
    # Gumbel noise via inverse CDF: -log(-log(U)), U ~ Uniform(0, 1)
    u = torch.zeros_like(logits).uniform_().clamp(1e-5, 1 - 1e-5)
    gumbel = -torch.log(-torch.log(u))

    # Soft mask: σ((logit + gumbel) / τ) — differentiable w.r.t. logits via autograd
    soft = torch.sigmoid((logits + gumbel) / tau)

    # Hard mask: threshold at 0.5 (equivalent to: perturbed logit > 0)
    hard = (soft > 0.5).float()

    # STE trick: hard forward, soft gradient in backward (no custom autograd needed)
    return hard - soft.detach() + soft


class StraightThroughEstimator(torch.autograd.Function):
    """
    Passes binary mask forward, passes gradient back to the logits unchanged.
    """
    @staticmethod
    def forward(ctx, input_logits):
        # Hard thresholding (e.g., > 0.5)
        return (input_logits > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Identity gradient
        return grad_output


class TopKStraightThrough(torch.autograd.Function):
    """
    Straight-Through Estimator for Top-K selection.
    Forward: returns binary mask with 1s at Top-K positions
    Backward: passes gradient through to original logits
    """
    @staticmethod
    def forward(ctx, logits, k):
        B, M = logits.shape
        k = min(k, M)  # Ensure k doesn't exceed number of patches

        # Get Top-K indices
        _, topk_indices = torch.topk(logits, k, dim=1)

        # Create binary mask
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_indices, 1.0)

        # Save for backward
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through unchanged (straight-through)
        return grad_output, None  # None for k (not differentiable)

class PatchMaskingHead(nn.Module):
    def __init__(self, input_dim, reduction_ratio=0.5, use_gumbel=False, tau_start=1.0):
        super().__init__()
        # Simple MLP to predict "keep" importance score for each patch
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1) # Output: Logit for "Importance"
        )
        self.reduction_ratio = reduction_ratio
        # Zero-init the final bias: removes the random kaiming bias that can
        # collapse all logits below the STE threshold at step 1.
        nn.init.zeros_(self.predictor[2].bias)
        # Learnable global offset: shifts ALL patch logits uniformly.
        # Initialized to 0; SparsityLoss converges it quickly to whatever
        # global shift brings kept_ratio → target_ratio, compensating for
        # the systematic negative skew from pretrained ViT feature structure.
        self.logit_offset = nn.Parameter(torch.zeros(1))

        # Gumbel-Sigmoid mode (replaces plain STE)
        self.use_gumbel = use_gumbel
        self._tau = tau_start  # Mutable temperature — annealed by trainer

    def set_tau(self, tau):
        """Update Gumbel temperature (called by trainer for annealing)."""
        self._tau = tau

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (B, N_patches, Dim) - EXCLUDING CLS token usually
        """
        # Predict per-patch importance logits + global offset
        logits = self.predictor(patch_embeddings).squeeze(-1) + self.logit_offset  # (B, N)

        if self.use_gumbel:
            # Gumbel-Sigmoid: bounded backward gradient, exploration via noise
            mask = gumbel_sigmoid_sample(logits, tau=self._tau)
        else:
            # Plain STE: identity backward (can cause gradient explosions with high FILIP weights)
            mask = StraightThroughEstimator.apply(logits)

        return mask, logits


class MaskHeadTopK(nn.Module):
    """
    Model D: Top-K Patch Selection Head.

    Instead of using a threshold to select patches, this head selects exactly
    the Top-K most important patches based on learned importance scores.

    Benefits:
    - Guaranteed sparsity: exactly K patches are selected
    - More efficient for local alignment: only K patches used as K/V
    - Reduced GPU memory during training and inference

    From thesis Section 14.1:
    - Computes importance logits for each patch
    - Selects Top-K patches using straight-through estimator
    - K determined by k_ratio * num_patches
    """
    def __init__(self, input_dim, k_ratio=0.25):
        """
        Args:
            input_dim: Dimension of patch embeddings (768 for ViT-B)
            k_ratio: Fraction of patches to keep (0.25 = keep 25% = 49 patches for 196 total)
        """
        super().__init__()
        self.k_ratio = k_ratio
        self._current_k_ratio = k_ratio  # Mutable ratio (for annealing)

        # Simple MLP to predict importance score for each patch
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (B, M, D) - patch embeddings excluding CLS token

        Returns:
            mask: (B, M) - binary mask with 1s at Top-K positions
            logits: (B, M) - importance logits (for auxiliary losses)
            topk_indices: (B, K) - indices of selected patches (for sparse operations)
        """
        B, M, D = patch_embeddings.shape

        # Compute importance logits
        logits = self.predictor(patch_embeddings).squeeze(-1)  # (B, M)

        # Compute K based on current (possibly annealed) ratio
        K = max(1, int(M * self._current_k_ratio))

        # Get binary mask using straight-through estimator
        mask = TopKStraightThrough.apply(logits, K)

        # Also return Top-K indices for sparse operations
        _, topk_indices = torch.topk(logits, K, dim=1)

        return mask, logits, topk_indices

    def set_k_ratio(self, ratio):
        """Update k_ratio (used for annealing during training)."""
        self._current_k_ratio = ratio

    def get_selected_patches(self, patch_embeddings, topk_indices):
        """
        Gather only the Top-K patches for efficient local alignment.

        Args:
            patch_embeddings: (B, M, D) - all patch embeddings
            topk_indices: (B, K) - indices of selected patches

        Returns:
            selected_patches: (B, K, D) - only the selected patches
        """
        B, M, D = patch_embeddings.shape
        K = topk_indices.shape[1]

        # Expand indices for gather operation
        indices_expanded = topk_indices.unsqueeze(-1).expand(B, K, D)

        # Gather selected patches
        selected_patches = torch.gather(patch_embeddings, dim=1, index=indices_expanded)

        return selected_patches


class PatchScorerMLP(nn.Module):
    """
    Model E: Lightweight MLP that scores patch importance for intra-ViT dropping.

    Applied to intermediate patch features (after drop_layer blocks).
    Returns only importance logits — actual sequence reduction is handled
    by the model's forward pass via index-based gathering.

    Unlike MaskHeadTopK (which produces a binary mask for weighted pooling),
    this scorer is used to reduce the ViT sequence length: the CLS token
    at inference is computed only from the K selected patches.
    """
    def __init__(self, input_dim, k_ratio=0.5):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )
        self._current_k_ratio = k_ratio

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (B, N, D) - intermediate patch features
        Returns:
            logits: (B, N) - importance logits (higher = more important)
        """
        return self.predictor(patch_embeddings).squeeze(-1)

    def set_k_ratio(self, ratio):
        self._current_k_ratio = ratio

    def get_k(self, n_patches):
        return max(1, int(n_patches * self._current_k_ratio))