import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, weight_i2t=0.5, weight_t2i=0.5):
        """
        Symmetric contrastive loss with configurable weights.

        Args:
            temperature: Temperature for similarity scaling (unused if logit_scale provided)
            weight_i2t: Weight for image-to-text loss (default 0.5)
            weight_t2i: Weight for text-to-image loss (default 0.5)
        """
        super().__init__()
        self.temperature = temperature
        self.weight_i2t = weight_i2t
        self.weight_t2i = weight_t2i
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, img_feat, text_feat, logit_scale):
        # Scale logic from CLIP (clamp to prevent overflow, exp(4.6) ≈ 100)
        logit_scale = torch.clamp(logit_scale, max=4.6).exp()

        # Cosine similarity
        logits_per_image = logit_scale * img_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()

        # Targets are diagonal (0, 1, 2...)
        batch_size = img_feat.shape[0]
        labels = torch.arange(batch_size, device=img_feat.device)

        loss_i2t = self.cross_entropy(logits_per_image, labels)
        loss_t2i = self.cross_entropy(logits_per_text, labels)

        return self.weight_i2t * loss_i2t + self.weight_t2i * loss_t2i

class SparsityLoss(nn.Module):
    """
    Encourages the mask to be sparse (Information Bottleneck).
    """
    def __init__(self, target_ratio=0.5):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, importance_logits):
        # Sigmoid to approximate the binary decision for loss calculation
        probs = torch.sigmoid(importance_logits)
        mean_activation = probs.mean()
        # L2 penalty towards target ratio
        return (mean_activation - self.target_ratio) ** 2


class LocalAlignmentLoss(nn.Module):
    """
    Local alignment loss between attention-aligned visual features and text tokens.

    Supports two modes:
    1. Multi-head mode (default): Uses pre-computed aligned features from LocalAlignModule
    2. Legacy single-head mode: Computes attention internally (backward compatible)

    Loss types:
    - "mse": MSE between aligned visual features and token embeddings (default, backward compatible)
    - "cosine": 1 - cosine_similarity, bounded in [0, 2], does not decay to zero

    Symmetry:
    - symmetric=False (default): text→image only (backward compatible)
    - symmetric=True: averages text→image and image→text directions
    """
    def __init__(self, temperature=1.0, use_multihead=True, loss_type="mse", symmetric=False):
        super().__init__()
        self.temperature = temperature
        self.use_multihead = use_multihead
        assert loss_type in ("mse", "cosine"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.symmetric = symmetric

    def _compute_loss(self, predicted, target, mask=None):
        """Compute per-element loss using the configured loss type.

        Args:
            predicted: (B, N, D)
            target: (B, N, D)
            mask: (B, N) or None — 1 for valid, 0 for padding

        Returns:
            scalar loss averaged over valid elements
        """
        if self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # (B, N)
            loss_per_elem = 1.0 - cos_sim
        else:
            loss_per_elem = F.mse_loss(predicted, target, reduction='none')  # (B, N, D)
            loss_per_elem = loss_per_elem.mean(dim=-1)  # (B, N)

        if mask is not None:
            mask_f = mask.float()
            return (loss_per_elem * mask_f).sum() / (mask_f.sum() + 1e-6)
        else:
            return loss_per_elem.mean()

    def forward(self, patch_features, token_features, attention_mask,
                aligned_features=None, attn_weights=None):
        """
        Args:
            patch_features: (B, M, D) - projected image patch embeddings (M=196 or fewer with masking)
            token_features: (B, L, D) - projected text token embeddings
            attention_mask: (B, L) - 1 for valid tokens, 0 for padding
            aligned_features: (B, L, D) - pre-computed aligned features (multi-head mode)
            attn_weights: (B, L, M) - attention weights (for logging/visualization)

        Returns:
            loss: scalar - local alignment loss
        """
        B, L, D = token_features.shape

        # ---- Forward direction: text → image ----
        if aligned_features is not None:
            v_aligned = aligned_features
        else:
            scale = D ** 0.5
            fwd_scores = torch.bmm(
                token_features, patch_features.transpose(1, 2)
            ) / (scale * self.temperature)
            fwd_weights = F.softmax(fwd_scores, dim=-1)
            v_aligned = torch.bmm(fwd_weights, patch_features)

        loss_fwd = self._compute_loss(v_aligned, token_features, mask=attention_mask)

        if not self.symmetric:
            return loss_fwd

        # ---- Reverse direction: image → text ----
        # Standard scaled dot-product attention (1/sqrt(D)), matching nn.MultiheadAttention
        scale = D ** 0.5
        rev_scores = torch.bmm(
            patch_features, token_features.transpose(1, 2)
        ) / scale  # (B, M, L)

        # Mask padding tokens so patches don't attend to them
        padding_mask = (~attention_mask.bool()).unsqueeze(1)  # (B, 1, L)
        rev_scores = rev_scores.masked_fill(padding_mask, float('-inf'))

        rev_weights = F.softmax(rev_scores, dim=-1)  # (B, M, L)
        t_aligned = torch.bmm(rev_weights, token_features)  # (B, M, D)

        loss_rev = self._compute_loss(t_aligned, patch_features)

        return 0.5 * loss_fwd + 0.5 * loss_rev


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between full and masked image-text similarities.

    Ensures that the masked embeddings maintain similar discriminative power
    as the full embeddings. From the Patch-IB framework:

    L_cons = (s_ii(z) - s̄_ii)²

    Where s_ii(z) is the similarity between masked image i and its text i,
    and s̄_ii is the similarity between full image i and its text i.
    """
    def __init__(self, include_negatives=False):
        """
        Args:
            include_negatives: If True, also penalize difference on negative pairs.
                              If False (default), only penalize positive pairs.
        """
        super().__init__()
        self.include_negatives = include_negatives

    def forward(self, img_emb_full, img_emb_masked, text_emb, logit_scale):
        """
        Args:
            img_emb_full: (B, D) - full image embeddings (no masking)
            img_emb_masked: (B, D) - masked image embeddings
            text_emb: (B, D) - text embeddings
            logit_scale: learnable temperature parameter

        Returns:
            loss: scalar - consistency loss
        """
        # Compute temperature-scaled similarities
        logit_scale = torch.clamp(logit_scale, max=4.6).exp()

        # Similarity matrices
        sim_full = logit_scale * img_emb_full @ text_emb.t()  # (B, B)
        sim_masked = logit_scale * img_emb_masked @ text_emb.t()  # (B, B)

        if self.include_negatives:
            # MSE over entire similarity matrix
            loss = F.mse_loss(sim_masked, sim_full)
        else:
            # MSE only on diagonal (positive pairs)
            pos_sim_full = sim_full.diag()  # (B,)
            pos_sim_masked = sim_masked.diag()  # (B,)
            loss = F.mse_loss(pos_sim_masked, pos_sim_full)

        return loss