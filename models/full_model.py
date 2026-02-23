import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BiomedCLIPBackbone
from .heads import PatchMaskingHead, MaskHeadTopK, PatchScorerMLP, TopKStraightThrough
from .cross_attention import BidirectionalCrossAttention


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
                # Model C: Threshold-based masking (STE or Gumbel-Sigmoid)
                use_gumbel = cfg['model'].get('use_gumbel', False)
                tau_start = cfg['model'].get('gumbel_tau_start', 1.0)
                self.mask_head = PatchMaskingHead(
                    self.embed_dim, use_gumbel=use_gumbel, tau_start=tau_start
                )

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

        # Mid-Fusion Cross-Attention
        self.use_mid_fusion = cfg['model'].get('use_mid_fusion', False)
        if self.use_mid_fusion:
            fusion_layers = cfg['model'].get('mid_fusion_layers', [4, 8, 12])
            fusion_n_heads = cfg['model'].get('mid_fusion_n_heads', 12)
            # Convert 1-indexed layer numbers to 0-indexed
            self.mid_fusion_layer_indices = [l - 1 for l in fusion_layers]
            self.mid_fusion_modules = nn.ModuleList([
                BidirectionalCrossAttention(
                    d_model=self.embed_dim,
                    n_heads=fusion_n_heads,
                    dropout=cfg['model'].get('mid_fusion_dropout', 0.1),
                )
                for _ in fusion_layers
            ])

            # Per-module projection layers for mid-fusion local loss (768 -> 512)
            # Separate projections per fusion point since representations at
            # different depths live in different subspaces
            mid_fusion_loss_weights = cfg['model'].get('mid_fusion_loss_weights', None)
            self.use_mid_fusion_local_loss = mid_fusion_loss_weights is not None
            if self.use_mid_fusion_local_loss:
                self.mid_fusion_patch_projs = nn.ModuleList([
                    nn.Linear(self.embed_dim, self.proj_dim) for _ in fusion_layers
                ])
                self.mid_fusion_token_projs = nn.ModuleList([
                    nn.Linear(self.embed_dim, self.proj_dim) for _ in fusion_layers
                ])

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

        # Visualization flag: set to True to store cross-attention weights
        self.store_attention_weights = False
        self._mid_fusion_attn_weights = None

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

    def _forward_mid_fusion(self, images, text):
        """
        Layer-by-layer lockstep forward pass with bidirectional cross-attention
        injected at specified layers. Returns ViT features (with norm applied)
        and BERT hidden states (raw, before pooling).
        """
        visual = self.backbone.clip_model.visual
        text_encoder = self.backbone.clip_model.text

        # --- ViT embedding ---
        vit_trunk = visual.trunk
        x_vit = vit_trunk.patch_embed(images)
        x_vit = vit_trunk._pos_embed(x_vit)
        x_vit = vit_trunk.patch_drop(x_vit)
        x_vit = vit_trunk.norm_pre(x_vit)

        # --- BERT embedding ---
        attention_mask = (text != 0).long()
        x_bert = text_encoder.transformer.embeddings(input_ids=text)

        # Extended attention mask for BERT layers: (B, 1, 1, L)
        # 0.0 for real tokens, large negative for padding
        extended_mask = attention_mask[:, None, None, :].to(dtype=x_bert.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(x_bert.dtype).min

        # Padding mask for cross-attention: True = padding (ignored)
        bert_padding_mask = (attention_mask == 0)

        # --- Lockstep layer execution ---
        vit_blocks = vit_trunk.blocks
        bert_layers = text_encoder.transformer.encoder.layer
        fusion_idx = 0
        mid_fusion_intermediates = []
        mid_fusion_attentions = [] if self.store_attention_weights else None

        for layer_idx in range(12):
            x_vit = vit_blocks[layer_idx](x_vit)
            x_bert = bert_layers[layer_idx](x_bert, attention_mask=extended_mask)[0]

            if layer_idx in self.mid_fusion_layer_indices:
                # Collect BEFORE cross-attention so the local loss measures
                # independent encoder alignment (cross-attention can't shortcut)
                if self.use_mid_fusion_local_loss:
                    mid_fusion_intermediates.append(
                        (x_vit[:, 1:, :], x_bert)  # patches (no CLS), tokens
                    )
                if self.store_attention_weights:
                    x_vit, x_bert, v2t_w, t2v_w = self.mid_fusion_modules[fusion_idx](
                        x_vit, x_bert, bert_padding_mask, return_attention=True
                    )
                    mid_fusion_attentions.append((v2t_w, t2v_w))
                else:
                    x_vit, x_bert = self.mid_fusion_modules[fusion_idx](
                        x_vit, x_bert, bert_padding_mask
                    )
                fusion_idx += 1

        # Store attention weights for visualization
        self._mid_fusion_attn_weights = mid_fusion_attentions

        # --- ViT final norm ---
        x_vit = vit_trunk.norm(x_vit)

        return x_vit, x_bert, attention_mask, mid_fusion_intermediates

    def _pool_and_project_text(self, hidden_states):
        """
        Pool BERT hidden states (CLS token) and project to 512-d,
        replicating the monolithic encode_text path.
        BiomedCLIP uses ClsLastHiddenStatePooler (just CLS extraction)
        followed by an MLP projection (768 -> 640 -> 512).
        """
        text_encoder = self.backbone.clip_model.text
        # CLS token pooling (position 0)
        pooled = hidden_states[:, 0, :]
        # text.proj: Sequential(Linear 768->640, GELU, Linear 640->512)
        projected = text_encoder.proj(pooled)
        return projected

    def forward(self, images, text):
        importance_logits = None
        local_features = None
        img_emb_full = None
        topk_indices = None

        if self.use_mid_fusion:
            # ============ MID-FUSION PATH — Dual Contrastive (ALBEF-style) ============
            # 1. Lockstep forward with cross-attention
            vit_features, bert_hidden, attention_mask, mid_intermediates = \
                self._forward_mid_fusion(images, text)

            # 2. Fused embeddings (primary contrastive loss — trains cross-attention + backbone)
            cls_token = vit_features[:, 0, :]
            img_embedding = self._project_embedding(cls_token)
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)

            text_embedding = self._pool_and_project_text(bert_hidden)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)

            # 3. Independent embeddings (second contrastive loss — ensures retrieval works)
            #    Backbone weights are shared so both losses update them.
            features_ind = self.backbone.encode_image_patches(images)
            cls_ind = features_ind[:, 0, :]
            img_emb_ind = self._project_embedding(cls_ind)
            img_emb_ind = F.normalize(img_emb_ind, p=2, dim=-1)

            txt_emb_ind = self.backbone.encode_text(text)
            txt_emb_ind = F.normalize(txt_emb_ind, p=2, dim=-1)

            # 4. Patch-IB masking (Model C): override img_embedding with masked pool
            if self.use_masking:
                patch_tokens = vit_features[:, 1:, :]  # (B, 196, 768)
                if self.use_topk_masking:
                    mask, importance_logits, topk_indices = self.mask_head(patch_tokens)
                else:
                    mask, importance_logits = self.mask_head(patch_tokens)

                # Save fused CLS as reference for consistency loss
                fused_full_emb = img_embedding  # already CLS → projected → normalized

                # Masked embedding: pool selected patches → project → normalize
                masked_patches = patch_tokens * mask.unsqueeze(-1)
                mask_sum = mask.sum(dim=1, keepdim=True) + 1e-6
                img_embedding_768 = masked_patches.sum(dim=1) / mask_sum
                img_embedding = self._project_embedding(img_embedding_768)
                img_embedding = F.normalize(img_embedding, p=2, dim=-1)

                # 3-tuple signals Model C + mid-fusion to trainer
                img_emb_full = (fused_full_emb, img_emb_ind, txt_emb_ind)
            else:
                # Standard Model B: 2-tuple
                img_emb_full = (img_emb_ind, txt_emb_ind)

            # 5. Mid-fusion local loss (optional, from lockstep intermediates)
            if self.use_mid_fusion_local_loss and mid_intermediates:
                local_features = []
                for k, (patches_768, tokens_768) in enumerate(mid_intermediates):
                    pf = F.normalize(self.mid_fusion_patch_projs[k](patches_768), dim=-1)
                    tf = F.normalize(self.mid_fusion_token_projs[k](tokens_768), dim=-1)
                    local_features.append((pf, tf, attention_mask))
            # =====================================================================
        else:
            # ============ STANDARD PATH (Models A/B/C/D) ============
            # 1. Get Raw Features (Batch, 197, 768)
            features = self.backbone.encode_image_patches(images)

            # Extract CLS and patches
            cls_token = features[:, 0, :]      # (B, 768)
            patch_tokens = features[:, 1:, :]  # (B, 196, 768)

            if self.use_masking:
                # ============ PATCH-IB: Dual-Path Forward ============
                # Path 1: Full embedding (no masking) - for L_NCE_full and consistency
                img_emb_full = self._project_embedding(cls_token)
                img_emb_full = F.normalize(img_emb_full, p=2, dim=-1)

                # Path 2: Masked embedding - for L_NCE_mask
                if self.use_topk_masking:
                    mask, importance_logits, topk_indices = self.mask_head(patch_tokens)
                else:
                    mask, importance_logits = self.mask_head(patch_tokens)

                masked_patches = patch_tokens * mask.unsqueeze(-1)
                mask_sum = mask.sum(dim=1, keepdim=True) + 1e-6
                img_embedding_768 = masked_patches.sum(dim=1) / mask_sum

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
                token_embeddings, attention_mask = self.backbone.encode_text_tokens(text)

                if self.use_topk_masking and topk_indices is not None:
                    selected_patches = self.mask_head.get_selected_patches(patch_tokens, topk_indices)
                    patch_features = self.patch_proj(selected_patches)
                else:
                    patch_features = self.patch_proj(patch_tokens)

                token_features = self.token_proj(token_embeddings)

                patch_features = F.normalize(patch_features, dim=-1)
                token_features = F.normalize(token_features, dim=-1)

                aligned_features, attn_weights = self.local_align(
                    token_features, patch_features
                )
                local_features = (patch_features, token_features, attention_mask,
                                  aligned_features, attn_weights)
            # =======================================================

        return img_embedding, text_embedding, importance_logits, local_features, img_emb_full

    @torch.no_grad()
    def encode_independent(self, images, text):
        """
        Encode image and text independently through the backbone, bypassing
        mid-fusion cross-attention. This produces unimodal embeddings suitable
        for retrieval evaluation (no information leakage between modalities).

        The backbone weights are still shaped by mid-fusion training via
        gradient flow, so the representations reflect training quality.

        Returns:
            img_emb: (B, 512) L2-normalized image embeddings
            txt_emb: (B, 512) L2-normalized text embeddings
        """
        # Image: backbone patches -> CLS -> project -> normalize
        features = self.backbone.encode_image_patches(images)
        cls_token = features[:, 0, :]
        img_emb = self._project_embedding(cls_token)
        img_emb = F.normalize(img_emb, p=2, dim=-1)

        # Text: backbone encode -> normalize
        txt_emb = self.backbone.encode_text(text)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        return img_emb, txt_emb


class ModelE(nn.Module):
    """
    Model E: Intra-ViT Patch Dropping.

    Patches are scored by a lightweight MLP at an intermediate ViT layer
    (default: layer 6 of 12) and the lowest-scoring patches are dropped.
    The CLS token and K selected patches continue through the upper ViT layers.

    Key advantage over Model C/D:
    - No train/inference mismatch: encode_independent uses the SAME mid-drop forward.
    - Single contrastive loss on the final CLS token — no competing gradients.
    - The CLS token is shaped by only the semantically important patches.

    Gradient to scorer: TopKStraightThrough STE applied to patch features
    before gathering ensures the scorer MLP receives gradients from the
    contrastive loss even though Top-K selection is non-differentiable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomedCLIPBackbone(
            model_name=cfg['model']['vision_backbone'],
            device=cfg['training']['device']
        )
        self.embed_dim = 768
        self.proj_dim = 512

        self.drop_layer = cfg['model'].get('drop_layer', 6)

        # Lightweight MLP scorer applied at intermediate ViT layer
        k_ratio = cfg['model'].get('k_ratio', 0.5)
        self.scorer = PatchScorerMLP(self.embed_dim, k_ratio=k_ratio)

        # Optional FILIP local loss on selected patches (post upper-block processing)
        self.use_filip_local = cfg['model'].get('use_filip_local', False)
        if self.use_filip_local:
            self.patch_proj = nn.Linear(self.embed_dim, self.proj_dim)
            self.token_proj = nn.Linear(self.embed_dim, self.proj_dim)

        self.logit_scale = self.backbone.clip_model.logit_scale

    def _project_embedding(self, embedding):
        visual_model = self.backbone.clip_model.visual
        if hasattr(visual_model, 'proj') and visual_model.proj is not None:
            return embedding @ visual_model.proj.to(embedding.dtype)
        elif hasattr(visual_model, 'head') and visual_model.head is not None:
            return visual_model.head(embedding)
        else:
            raise AttributeError("Cannot project 768->512. Check model architecture.")

    def _forward_with_mid_drop(self, images):
        """
        Layer-by-layer ViT forward with patch dropping at self.drop_layer.

        Phase 1: All 197 tokens (CLS + 196 patches) through blocks 0..drop_layer-1.
        Scoring: PatchScorerMLP assigns importance to each of the 196 patch tokens.
        Drop:    Top-K patches kept; rest discarded. Sequence becomes K+1 tokens.
        Phase 2: CLS + K selected patches through blocks drop_layer..11.

        STE trick (training only): TopKStraightThrough multiplied into patch features
        before gathering so the scorer receives gradients from the contrastive loss.

        Returns:
            features:          (B, K+1, D) — final normed ViT features, reduced sequence
            topk_indices:      (B, K)      — which patch positions were kept
            importance_logits: (B, 196)    — raw scorer output (for sparsity loss + logging)
        """
        vit_trunk = self.backbone.clip_model.visual.trunk

        # ViT embedding (identical to _forward_mid_fusion in ModelABaseline)
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        # Phase 1: full sequence through first drop_layer blocks
        for i in range(self.drop_layer):
            x = vit_trunk.blocks[i](x)

        # Score intermediate patch features (exclude CLS at index 0)
        patch_features = x[:, 1:, :]          # (B, 196, D)
        importance_logits = self.scorer(patch_features)  # (B, 196)
        N, D = patch_features.shape[1], patch_features.shape[2]
        k = self.scorer.get_k(N)

        if self.training:
            # STE: TopKStraightThrough gives a binary mask with gradients flowing
            # back to importance_logits → scorer MLP via the identity backward.
            mask_ste = TopKStraightThrough.apply(importance_logits, k)  # (B, N)
            patch_features_for_select = patch_features * mask_ste.unsqueeze(-1)
        else:
            # No gradient needed at eval time — use raw patch features directly.
            patch_features_for_select = patch_features

        # Top-K indices for actual sequence reduction
        _, topk_idx = torch.topk(importance_logits, k, dim=1)
        topk_idx_sorted = topk_idx.sort(dim=1).values  # (B, K) sorted for stability

        # Gather K selected patches (at TopK positions, mask_ste=1 so values unchanged)
        selected = patch_features_for_select.gather(
            1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D)
        )

        # Reduced sequence: CLS token + K selected patches
        x = torch.cat([x[:, :1, :], selected], dim=1)  # (B, K+1, D)

        # Phase 2: remaining blocks on reduced sequence
        for i in range(self.drop_layer, len(vit_trunk.blocks)):
            x = vit_trunk.blocks[i](x)

        x = vit_trunk.norm(x)

        return x, topk_idx_sorted, importance_logits

    def forward(self, images, text):
        local_features = None

        # Image: intra-ViT dropping forward
        vit_features, topk_indices, importance_logits = self._forward_with_mid_drop(images)

        # CLS token from reduced sequence (attended only to selected patches)
        cls_token = vit_features[:, 0, :]
        img_emb = self._project_embedding(cls_token)
        img_emb = F.normalize(img_emb, p=2, dim=-1)

        # Text: standard independent encoding
        txt_emb = self.backbone.encode_text(text)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        # Optional FILIP local loss on selected patches after upper-block processing.
        # vit_features[:, 1:, :] are the K patches refined by blocks drop_layer..11.
        if self.use_filip_local:
            selected_patch_feats = vit_features[:, 1:, :]  # (B, K, D)
            token_embeddings, attention_mask = self.backbone.encode_text_tokens(text)

            patch_f = F.normalize(self.patch_proj(selected_patch_feats), dim=-1)
            token_f = F.normalize(self.token_proj(token_embeddings), dim=-1)
            # Return as single-element list to use mid-fusion FILIP path in trainer
            local_features = [(patch_f, token_f, attention_mask)]

        # img_emb_full=None: no dual-path, no consistency loss, no competing gradients
        return img_emb, txt_emb, importance_logits, local_features, None

    @torch.no_grad()
    def encode_independent(self, images, text):
        """
        Evaluation encoding — uses the SAME mid-drop forward as training.
        No train/inference mismatch: CLS is computed on the same K selected patches.
        """
        vit_features, _, _ = self._forward_with_mid_drop(images)

        cls_token = vit_features[:, 0, :]
        img_emb = self._project_embedding(cls_token)
        img_emb = F.normalize(img_emb, p=2, dim=-1)

        txt_emb = self.backbone.encode_text(text)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        return img_emb, txt_emb