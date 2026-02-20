import torch
import torch.nn as nn


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
    def __init__(self, input_dim, reduction_ratio=0.5):
        super().__init__()
        # Simple MLP to predict "keep" importance score for each patch
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1) # Output: Logit for "Importance"
        )
        self.reduction_ratio = reduction_ratio
        # Zero-init the final bias so logits start near 0 (~50% patches
        # selected) regardless of random seed. Default kaiming_uniform_ can
        # produce a strongly negative bias that drives all 196 logits below
        # the STE threshold (0) and collapses the mask to zero at step 1.
        nn.init.zeros_(self.predictor[2].bias)

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (B, N_patches, Dim) - EXCLUDING CLS token usually
        """
        # Predict importance logits
        logits = self.predictor(patch_embeddings).squeeze(-1) # (B, N)
        
        # Option A: Dynamic Keep based on ratio (Top-K)
        # Option B: Learned Threshold with STE (Simpler for "Bottleneck")
        
        # Here we use a simple STE on the logits for binary masking
        # In a real IB, you might penalize the number of 1s.
        
        mask = StraightThroughEstimator.apply(logits)

        # Alternatively: Gumbel Softmax for differentiable sampling
        # mask = F.gumbel_softmax(logits, hard=True)

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