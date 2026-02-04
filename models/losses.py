import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
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
        
        loss_i = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2

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
    Attention-based local alignment loss between image patches and text tokens.

    For each text token, computes attention-weighted sum of image patches,
    then measures MSE between aligned visual features and token embeddings.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, patch_features, token_features, attention_mask):
        """
        Args:
            patch_features: (B, 196, D) - projected image patch embeddings
            token_features: (B, L, D) - projected text token embeddings
            attention_mask: (B, L) - 1 for valid tokens, 0 for padding

        Returns:
            loss: scalar - masked MSE loss
        """
        B, L, D = token_features.shape

        # Step 1: Compute attention scores
        # (B, L, D) @ (B, D, 196) -> (B, L, 196)
        scale = D ** 0.5
        attention_scores = torch.bmm(
            token_features, patch_features.transpose(1, 2)
        ) / (scale * self.temperature)

        # Step 2: Softmax over patches dimension
        # (B, L, 196) -> attention distribution over patches for each token
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Step 3: Compute text-aligned visual features
        # (B, L, 196) @ (B, 196, D) -> (B, L, D)
        v_aligned = torch.bmm(attention_weights, patch_features)

        # Step 4: MSE loss with masking
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