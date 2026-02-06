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

    Loss: MSE between aligned visual features and token embeddings.
    """
    def __init__(self, temperature=1.0, use_multihead=True):
        super().__init__()
        self.temperature = temperature
        self.use_multihead = use_multihead

    def forward(self, patch_features, token_features, attention_mask,
                aligned_features=None, attn_weights=None):
        """
        Args:
            patch_features: (B, 196, D) - projected image patch embeddings
            token_features: (B, L, D) - projected text token embeddings
            attention_mask: (B, L) - 1 for valid tokens, 0 for padding
            aligned_features: (B, L, D) - pre-computed aligned features (multi-head mode)
            attn_weights: (B, L, 196) - attention weights (for logging/visualization)

        Returns:
            loss: scalar - masked MSE loss
        """
        B, L, D = token_features.shape

        if aligned_features is not None:
            # Multi-head mode: use pre-computed aligned features
            v_aligned = aligned_features
        else:
            # Legacy single-head mode: compute attention internally
            scale = D ** 0.5
            attention_scores = torch.bmm(
                token_features, patch_features.transpose(1, 2)
            ) / (scale * self.temperature)

            attention_weights = F.softmax(attention_scores, dim=-1)
            v_aligned = torch.bmm(attention_weights, patch_features)

        # MSE loss with masking
        # Only compute loss for non-padding tokens
        mse_per_token = F.mse_loss(v_aligned, token_features, reduction='none')  # (B, L, D)
        mse_per_token = mse_per_token.mean(dim=-1)  # (B, L) - average over embedding dim

        # Apply mask
        mask_expanded = attention_mask.float()  # (B, L)
        masked_mse = mse_per_token * mask_expanded

        # Average over valid tokens only
        num_valid_tokens = mask_expanded.sum() + 1e-6
        loss = masked_mse.sum() / num_valid_tokens

        return loss


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