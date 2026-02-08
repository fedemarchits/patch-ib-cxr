import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BiomedCLIPBackbone
from .heads import PatchMaskingHead, MaskHeadTopK


class LocalAlignModule(nn.Module):
    """
    Multi-head cross-attention for local alignment between patches and tokens.

    From thesis Section 11.1:
    - Query = text tokens (U)
    - Key = image patches (V)
    - Value = image patches (V)

    Produces attention-weighted visual features aligned with each text token.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # Optional final projection (as in thesis)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, token_features, patch_features, key_padding_mask=None):
        """
        Args:
            token_features: (B, L, D) - text token embeddings (queries)
            patch_features: (B, M, D) - image patch embeddings (keys & values)
            key_padding_mask: (B, M) - True for positions to mask (padding patches)

        Returns:
            aligned_features: (B, L, D) - attention-weighted visual features
            attention_weights: (B, L, M) - attention weights for visualization
        """
        # Multi-head cross-attention: Q=tokens, K=V=patches
        aligned_features, attention_weights = self.attn(
            query=token_features,
            key=patch_features,
            value=patch_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True  # Average across heads for visualization
        )

        # Final projection
        aligned_features = self.out_proj(aligned_features)

        return aligned_features, attention_weights


class ModelABaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomedCLIPBackbone(
            model_name=cfg['model']['vision_backbone'],
            device=cfg['training']['device']
        )

        # Dim check: ViT-Base usually 768
        self.embed_dim = 768
        self.proj_dim = 512  # CLIP projection dimension

        self.use_masking = cfg['model'].get('use_masking', False)
        self.use_local_alignment = cfg['model'].get('use_local_alignment', False)
        self.use_topk_masking = cfg['model'].get('use_topk_masking', False)
        self.k_ratio = cfg['model'].get('k_ratio', 0.25)

        # The Information Bottleneck Masker
        if self.use_masking:
            if self.use_topk_masking:
                # Model D: Top-K patch selection
                self.mask_head = MaskHeadTopK(self.embed_dim, k_ratio=self.k_ratio)
            else:
                # Model C: Threshold-based masking
                self.mask_head = PatchMaskingHead(self.embed_dim)

        # Projection layers for local alignment (768 -> 512)
        if self.use_local_alignment:
            self.patch_proj = nn.Linear(self.embed_dim, self.proj_dim)
            self.token_proj = nn.Linear(self.embed_dim, self.proj_dim)

            # Multi-head cross-attention module (thesis Section 11.1)
            n_heads = cfg['model'].get('local_alignment_n_heads', 4)
            self.local_align = LocalAlignModule(
                d_model=self.proj_dim,
                n_heads=n_heads,
                dropout=cfg['model'].get('local_alignment_dropout', 0.1)
            )

        # Projectors (if needed, though BiomedCLIP has them built-in)
        # We might need a projector after masking if we change the flow
        self.logit_scale = self.backbone.clip_model.logit_scale

        # Uncertainty weighting parameters (Kendall et al., 2018)
        # log_var = log(σ²), initialized to 0 means σ² = 1
        self.use_uncertainty_weighting = cfg['model'].get('use_uncertainty_weighting', False)
        if self.use_uncertainty_weighting:
            self.log_var_contrastive = nn.Parameter(torch.zeros(1))
            if self.use_local_alignment:
                self.log_var_local = nn.Parameter(torch.zeros(1))

    def _project_embedding(self, embedding):
        """Project embedding from 768 -> 512 using backbone's projection layer."""
        visual_model = self.backbone.clip_model.visual

        # CASE 1: Standard OpenAI CLIP (Matrix Multiplication)
        if hasattr(visual_model, 'proj') and visual_model.proj is not None:
            projection = visual_model.proj.to(embedding.dtype)
            return embedding @ projection

        # CASE 2: Timm/BiomedCLIP (Linear Layer)
        elif hasattr(visual_model, 'head') and visual_model.head is not None:
            return visual_model.head(embedding)

        else:
            print("CRITICAL ERROR: Could not find projection layer.")
            print(f"Available attributes: {dir(visual_model)}")
            raise AttributeError("Cannot project 768->512. Check logs.")

    def forward(self, images, text):
        # 1. Get Raw Features (Batch, 197, 768)
        features = self.backbone.encode_image_patches(images)

        importance_logits = None
        local_features = None  # Will hold (patch_feat, token_feat, attn_mask) if enabled
        img_emb_full = None    # Full embedding (for Patch-IB consistency loss)
        topk_indices = None    # For Model D: indices of selected patches

        # Extract CLS and patches
        cls_token = features[:, 0, :]      # (B, 768)
        patch_tokens = features[:, 1:, :]  # (B, 196, 768)

        if self.use_masking:
            # ============ PATCH-IB: Dual-Path Forward ============
            # Path 1: Full embedding (no masking) - for L_NCE_full and consistency
            img_emb_full = self._project_embedding(cls_token)
            img_emb_full = F.normalize(img_emb_full, p=2, dim=-1)

            # Path 2: Masked embedding - for L_NCE_mask
            # 2a. Generate Mask (The "Bottleneck")
            if self.use_topk_masking:
                # Model D: Top-K selection returns (mask, logits, topk_indices)
                mask, importance_logits, topk_indices = self.mask_head(patch_tokens)
            else:
                # Model C: Threshold-based returns (mask, logits)
                mask, importance_logits = self.mask_head(patch_tokens)

            # 2b. Apply Mask
            masked_patches = patch_tokens * mask.unsqueeze(-1)

            # 2c. Global Pooling of Masked Patches
            mask_sum = mask.sum(dim=1, keepdim=True) + 1e-6
            img_embedding_768 = masked_patches.sum(dim=1) / mask_sum

            # 2d. Project masked embedding
            img_embedding = self._project_embedding(img_embedding_768)
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)
            # =====================================================
        else:
            # Model A/B: Take CLS token -> Shape: (Batch, 768)
            img_embedding = self._project_embedding(cls_token)
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)

        # Encode Text (pooled embedding for global loss)
        text_embedding = self.backbone.encode_text(text)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        # Local alignment features (if enabled)
        if self.use_local_alignment:
            # Get token-level embeddings
            token_embeddings, attention_mask = self.backbone.encode_text_tokens(text)

            # For Model D with Top-K: use only selected patches for efficiency
            if self.use_topk_masking and topk_indices is not None:
                # Get only the Top-K selected patches (B, K, 768)
                selected_patches = self.mask_head.get_selected_patches(patch_tokens, topk_indices)
                # Project selected patches to shared space (768 -> 512)
                patch_features = self.patch_proj(selected_patches)  # (B, K, 512)
            else:
                # Model C or no masking: use all patches
                patch_features = self.patch_proj(patch_tokens)  # (B, 196, 512)

            token_features = self.token_proj(token_embeddings)  # (B, L, 512)

            # Normalize for stable attention computation
            patch_features = F.normalize(patch_features, dim=-1)
            token_features = F.normalize(token_features, dim=-1)

            # Multi-head cross-attention: align patches to tokens
            # Query=tokens, Key/Value=patches (Top-K patches for Model D)
            aligned_features, attn_weights = self.local_align(
                token_features, patch_features
            )

            # local_features now includes aligned features for loss computation
            # Format: (patch_feat, token_feat, attention_mask, aligned_feat, attn_weights)
            local_features = (patch_features, token_features, attention_mask, aligned_features, attn_weights)

        # Return format:
        # - img_embedding: masked if use_masking else full (main embedding for retrieval)
        # - text_embedding: text embedding
        # - importance_logits: mask logits (for sparsity loss)
        # - local_features: (patch_feat, token_feat, attn_mask) or None
        # - img_emb_full: full embedding if use_masking else None (for consistency loss)
        return img_embedding, text_embedding, importance_logits, local_features, img_emb_full