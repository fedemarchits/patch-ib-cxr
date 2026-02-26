import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BiomedCLIPBackbone
from .heads import (PatchMaskingHead, MaskHeadTopK, PatchScorerMLP,
                    TopKStraightThrough, StraightThroughEstimator, gumbel_sigmoid_sample)
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

            # FILIP mode: skip cross-attention module (max-cosine matching, no cross-attn)
            self.use_local_filip = cfg['model'].get('local_alignment_loss_type', 'mse') == 'filip'

            if not self.use_local_filip:
                # Multi-head cross-attention module (thesis Section 11.1)
                n_heads = cfg['model'].get('local_alignment_n_heads', 4)
                self.local_align = LocalAlignModule(
                    d_model=self.proj_dim,
                    n_heads=n_heads,
                    dropout=cfg['model'].get('local_alignment_dropout', 0.1)
                )

        # Multi-scale alignment probes (no injection — deep supervision only).
        # Extracts patch features at specified intermediate ViT layers and computes
        # FILIP against full BERT token features. The ViT forward pass is unmodified
        # (no cross-modal residual injection), so the global embedding space is not
        # contaminated by text. This is purely auxiliary supervision at multiple depths.
        self.use_multiscale_probes = cfg['model'].get('use_multiscale_probes', False)
        if self.use_multiscale_probes:
            probe_layers = cfg['model'].get('probe_layers', [4, 8])
            # Convert 1-indexed layer numbers to 0-indexed block indices
            self.probe_layer_indices = [l - 1 for l in probe_layers]
            # Separate projection heads per probe depth:
            # different ViT depths live in different subspaces
            self.probe_patch_projs = nn.ModuleList([
                nn.Linear(self.embed_dim, self.proj_dim) for _ in probe_layers
            ])
            self.probe_token_projs = nn.ModuleList([
                nn.Linear(self.embed_dim, self.proj_dim) for _ in probe_layers
            ])

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

    def _forward_with_probes(self, images):
        """
        Layer-by-layer ViT forward that snapshots patch features at probe layers.
        No cross-modal injection — the ViT sees only the image, exactly as in Model A.
        Probe features are used exclusively for FILIP supervision (auxiliary losses).

        Returns:
            final_features:  (B, 197, D) — final normed ViT features (CLS + patches)
            probe_features:  list of (B, 196, D) — patch snapshots at each probe layer
        """
        vit_trunk = self.backbone.clip_model.visual.trunk
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        probe_set = set(self.probe_layer_indices)
        probe_features = []

        for i, block in enumerate(vit_trunk.blocks):
            x = block(x)
            if i in probe_set:
                # Snapshot patch tokens (exclude CLS at index 0); clone to
                # detach from inplace ops in subsequent blocks
                probe_features.append(x[:, 1:, :].clone())

        x = vit_trunk.norm(x)
        return x, probe_features

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

        if self.use_multiscale_probes:
            # ============ MULTI-SCALE PROBE PATH ============
            # Pure visual ViT forward (same as Model A) + FILIP at intermediate layers.
            # No cross-modal injection — text never modifies image features.
            features, probe_feats = self._forward_with_probes(images)

            cls_token = features[:, 0, :]
            img_embedding = self._project_embedding(cls_token)
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)

            text_embedding = self.backbone.encode_text(text)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)

            # Build FILIP supervision list: one entry per probe layer
            # Trainer will average the losses across entries (is_filip=True path)
            token_embeddings, attention_mask = self.backbone.encode_text_tokens(text)
            local_features = []
            for k, patch_feats in enumerate(probe_feats):
                pf = F.normalize(self.probe_patch_projs[k](patch_feats), dim=-1)
                tf = F.normalize(self.probe_token_projs[k](token_embeddings), dim=-1)
                local_features.append((pf, tf, attention_mask))

            # Optional: also add final post-ViT FILIP as an additional probe
            if self.use_local_alignment:
                patch_tokens = features[:, 1:, :]
                pf_final = F.normalize(self.patch_proj(patch_tokens), dim=-1)
                tf_final = F.normalize(self.token_proj(token_embeddings), dim=-1)
                local_features.append((pf_final, tf_final, attention_mask))
            # ================================================

        elif self.use_mid_fusion:
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
                    # Gather selected patches directly (no STE mask multiplication).
                    # Gradient from FILIP flows to backbone (patch_tokens) but NOT to
                    # importance_logits — this prevents logit mean explosion caused by
                    # one-sided STE gradient (only selected patches receive FILIP signal).
                    # MaskHead gradient comes from: masked contrastive mean-pool STE path
                    # (which gives balanced gradient to ALL patches) + sparsity loss.
                    selected_patches = self.mask_head.get_selected_patches(patch_tokens, topk_indices)
                    patch_features = self.patch_proj(selected_patches)
                else:
                    patch_features = self.patch_proj(patch_tokens)

                token_features = self.token_proj(token_embeddings)

                patch_features = F.normalize(patch_features, dim=-1)
                token_features = F.normalize(token_features, dim=-1)

                if self.use_local_filip:
                    # Return as list to use the mid-fusion FILIP path in trainer
                    local_features = [(patch_features, token_features, attention_mask)]
                else:
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


class ModelF(nn.Module):
    """
    Model F: Text-Conditioned FILIP-Scored Intra-ViT Patch Dropping.

    Improves Model E by replacing the text-agnostic PatchScorerMLP with a
    text-conditioned FILIP scorer. Patches at drop_layer are ranked by their
    max cosine similarity to text tokens (probe projections W_p_probe / W_t_probe),
    so the drop decision is query-specific rather than globally biased.

    Two complementary training signals:
      1. Probe FILIP loss at drop_layer on ALL 196 patches (before dropping):
         - Trains W_p_probe, W_t_probe, and backbone blocks 0..drop_layer-1
         - "Make text-relevant patches stand out in FILIP space"
         - ALL patches receive gradient — not just kept ones
      2. Final FILIP loss on K selected patches after upper-block processing:
         - Trains W_p_final, W_t_final, and upper backbone blocks
         - "Kept patches remain aligned with text after further processing"

    Gradient to drop decision (via STE):
      contrastive_loss → CLS → upper blocks → kept patches → STE
        → FILIP scores → W_p_probe → intermediate patch feats → blocks 0..drop_layer-1

    Both signals push the backbone to produce features where text-relevant
    patches naturally score high — the scoring criterion and the training
    objective are jointly optimised.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomedCLIPBackbone(
            model_name=cfg['model']['vision_backbone'],
            device=cfg['training']['device']
        )
        self.embed_dim = 768
        self.proj_dim  = 512

        self.drop_layer = cfg['model'].get('drop_layer', 6)

        k_ratio = cfg['model'].get('k_ratio', 0.5)
        self._current_k_ratio = k_ratio

        # Learnable projections for FILIP scoring at drop_layer (768 → 512).
        # Intermediate ViT features live in a different subspace than final
        # features, so these are kept separate from patch_proj / token_proj.
        self.probe_patch_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.probe_token_proj = nn.Linear(self.embed_dim, self.proj_dim)

        # Optional final FILIP loss on K selected patches after upper blocks.
        # Separate projections because upper-block features differ from drop_layer.
        self.use_filip_final = cfg['model'].get('use_filip_final', True)
        if self.use_filip_final:
            self.patch_proj = nn.Linear(self.embed_dim, self.proj_dim)
            self.token_proj = nn.Linear(self.embed_dim, self.proj_dim)

        self.logit_scale = self.backbone.clip_model.logit_scale

    # ─────────────────────── helpers ────────────────────────────────────────

    def set_k_ratio(self, ratio):
        """Update k_ratio for annealing (called by trainer)."""
        self._current_k_ratio = ratio

    def get_k(self, n_patches):
        return max(1, int(n_patches * self._current_k_ratio))

    def _forward_full_vit_with_probe(self, images):
        """
        Full ViT forward (all 12 blocks, all 196 patches) with a snapshot of the
        intermediate state at drop_layer.

        Returns:
            final_features: (B, 197, D)  post-ViT normed features (CLS + patches)
            probe_feats:    (B, 196, D)  patch features at drop_layer (for FILIP scoring)
            state_at_drop:  (B, 197, D)  pre-norm state at drop_layer
                            (starting point for the separate dropped Phase 2 path)
        """
        vit_trunk = self.backbone.clip_model.visual.trunk
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        state_at_drop = None
        probe_feats   = None
        for i, block in enumerate(vit_trunk.blocks):
            x = block(x)
            if i == self.drop_layer - 1:
                # Clone both: the loop will keep updating x through the remaining blocks.
                probe_feats   = x[:, 1:, :].clone()   # (B, 196, D)
                state_at_drop = x.clone()              # (B, 197, D) — for dropped Phase 2

        x = vit_trunk.norm(x)
        return x, probe_feats, state_at_drop

    def _project_embedding(self, embedding):
        visual_model = self.backbone.clip_model.visual
        if hasattr(visual_model, 'proj') and visual_model.proj is not None:
            return embedding @ visual_model.proj.to(embedding.dtype)
        elif hasattr(visual_model, 'head') and visual_model.head is not None:
            return visual_model.head(embedding)
        else:
            raise AttributeError("Cannot project 768->512. Check model architecture.")

    def _compute_filip_scores(self, patch_feats_768, token_feats_768, attn_mask):
        """
        Per-patch FILIP importance: score_i = max_{l: valid} cos(W_p_probe*p_i, W_t_probe*t_l)

        Args:
            patch_feats_768: (B, 196, 768) intermediate patch features at drop_layer
            token_feats_768: (B, L,   768) BERT token features (raw hidden states)
            attn_mask:       (B, L)        1 = real token, 0 = padding

        Returns:
            scores: (B, 196)  per-patch FILIP scores in [-1, 1]
        """
        pf = F.normalize(self.probe_patch_proj(patch_feats_768), dim=-1)  # (B, 196, 512)
        tf = F.normalize(self.probe_token_proj(token_feats_768), dim=-1)  # (B, L,   512)

        sim = torch.bmm(pf, tf.transpose(1, 2))        # (B, 196, L)

        # Mask padding so they cannot inflate the max
        pad = (attn_mask == 0).unsqueeze(1)             # (B, 1, L)
        sim = sim.masked_fill(pad, torch.finfo(sim.dtype).min)

        return sim.max(dim=-1).values                   # (B, 196)

    # ─────────────────────── forward helpers ────────────────────────────────

    def _forward_with_filip_drop(self, images, token_feats, attn_mask):
        """
        Layer-by-layer ViT forward with text-conditioned FILIP patch dropping.

        Phase 1: All 197 tokens (CLS + 196 patches) through blocks 0..drop_layer-1.
        Score:   FILIP max-cosine scores via probe projections W_p_probe / W_t_probe.
        Drop:    Top-K patches kept; rest discarded. Sequence → K+1 tokens.
        Phase 2: CLS + K selected patches through blocks drop_layer..11.

        STE trick (training): TopKStraightThrough multiplied into patch features
        before gathering so the scorer receives gradients from the upper losses.

        Returns:
            features:          (B, K+1, D) final normed ViT features
            topk_idx_sorted:   (B, K)      kept patch positions (original indices)
            filip_scores:      (B, 196)    FILIP scores at drop_layer (for logging)
            probe_patch_feats: (B, 196, D) intermediate patch features (for probe loss)
        """
        vit_trunk = self.backbone.clip_model.visual.trunk

        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        # Phase 1: full sequence through first drop_layer blocks
        for i in range(self.drop_layer):
            x = vit_trunk.blocks[i](x)

        # Extract intermediate patch features and clone to isolate from phase 2
        patch_feats = x[:, 1:, :].clone()                          # (B, 196, D)

        # Text-conditioned FILIP scoring
        filip_scores = self._compute_filip_scores(patch_feats, token_feats, attn_mask)

        N, D = patch_feats.shape[1], patch_feats.shape[2]
        k = self.get_k(N)

        if self.training:
            # STE: forward = binary TopK mask, backward = identity gradient
            # This propagates contrastive gradients back through FILIP scores
            # → probe_patch_proj → patch_feats → ViT blocks 0..drop_layer-1
            mask_ste = TopKStraightThrough.apply(filip_scores, k)   # (B, N)
            patch_feats_for_select = patch_feats * mask_ste.unsqueeze(-1)
        else:
            patch_feats_for_select = patch_feats

        # Top-K indices sorted for attention stability in phase 2
        _, topk_idx = torch.topk(filip_scores, k, dim=1)
        topk_idx_sorted = topk_idx.sort(dim=1).values               # (B, K)

        # Gather K selected patches
        selected = patch_feats_for_select.gather(
            1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D)
        )

        # Reduced sequence: CLS token + K selected patches
        x = torch.cat([x[:, :1, :], selected], dim=1)              # (B, K+1, D)

        # Phase 2: remaining blocks on reduced sequence
        for i in range(self.drop_layer, len(vit_trunk.blocks)):
            x = vit_trunk.blocks[i](x)

        x = vit_trunk.norm(x)

        return x, topk_idx_sorted, filip_scores, patch_feats

    def get_intermediate_patch_scores(self, images, text):
        """
        Compute text-conditioned FILIP scores at drop_layer on all 196 patches.
        Used by deletion/insertion test and grounding visualisation.
        Runs only Phase 1 of the ViT (no dropping, no upper blocks).
        """
        token_feats, attn_mask = self.backbone.encode_text_tokens(text)

        vit_trunk = self.backbone.clip_model.visual.trunk
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        for i in range(self.drop_layer):
            x = vit_trunk.blocks[i](x)

        patch_feats = x[:, 1:, :]
        return self._compute_filip_scores(patch_feats, token_feats, attn_mask)

    # ─────────────────────── main forward ───────────────────────────────────

    def forward(self, images, text):
        # Text encoding (needed for FILIP scoring at drop_layer)
        token_feats, attn_mask = self.backbone.encode_text_tokens(text)

        # ── Single Phase 1 + Full Phase 2 (text-independent path) ─────────
        # Runs all 12 blocks on all 196 patches; also captures the state at
        # drop_layer so we can branch off a separate dropped Phase 2 below.
        vit_full, probe_patch_feats, state_at_drop = \
            self._forward_full_vit_with_probe(images)

        # Primary image embedding — text-independent full CLS.
        # This is the MAIN contrastive signal and what encode_independent()
        # returns for retrieval evaluation.
        full_cls = vit_full[:, 0, :]
        img_emb = F.normalize(self._project_embedding(full_cls), p=2, dim=-1)

        # Text global embedding
        txt_emb = F.normalize(self.backbone.encode_text(text), dim=-1)

        # ── FILIP scoring at drop_layer ────────────────────────────────────
        filip_scores = self._compute_filip_scores(probe_patch_feats, token_feats, attn_mask)

        # ── Dropped Phase 2 (text-conditioned auxiliary path) ─────────────
        # Reuses state_at_drop from Phase 1, then runs Phase 2 on only the
        # K FILIP-selected patches.  The resulting CLS is used as an AUXILIARY
        # contrastive signal (returned via img_emb_full) to train the scorer.
        N, D = probe_patch_feats.shape[1], probe_patch_feats.shape[2]
        k = self.get_k(N)

        if self.training:
            mask_ste = TopKStraightThrough.apply(filip_scores, k)
            patch_feats_for_select = state_at_drop[:, 1:, :] * mask_ste.unsqueeze(-1)
        else:
            patch_feats_for_select = state_at_drop[:, 1:, :]

        _, topk_idx = torch.topk(filip_scores, k, dim=1)
        topk_idx_sorted = topk_idx.sort(dim=1).values
        selected = patch_feats_for_select.gather(
            1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D)
        )
        x_dropped = torch.cat([state_at_drop[:, :1, :], selected], dim=1)  # (B, K+1, D)

        vit_trunk = self.backbone.clip_model.visual.trunk
        for i in range(self.drop_layer, len(vit_trunk.blocks)):
            x_dropped = vit_trunk.blocks[i](x_dropped)
        x_dropped = vit_trunk.norm(x_dropped)
        dropped_cls = x_dropped[:, 0, :]
        dropped_img_emb = F.normalize(self._project_embedding(dropped_cls), p=2, dim=-1)

        # ── Local FILIP losses ─────────────────────────────────────────────
        local_features = []

        #   [0] Probe FILIP on ALL 196 patches at drop_layer — primary scorer signal.
        pf_probe = F.normalize(self.probe_patch_proj(probe_patch_feats), dim=-1)
        tf_probe = F.normalize(self.probe_token_proj(token_feats), dim=-1)
        local_features.append((pf_probe, tf_probe, attn_mask))

        #   [1] Final FILIP on K selected patches after dropped Phase 2.
        if self.use_filip_final:
            selected_feats = x_dropped[:, 1:, :]              # (B, K, D)
            pf_final = F.normalize(self.patch_proj(selected_feats), dim=-1)
            tf_final = F.normalize(self.token_proj(token_feats), dim=-1)
            local_features.append((pf_final, tf_final, attn_mask))

        # img_emb_full = (dropped_img_emb, txt_emb):
        #   Trainer adds InfoNCE(dropped_img_emb, txt_emb) × contrastive_full_weight
        #   as an auxiliary loss that trains the FILIP scorer end-to-end.
        #   Retrieval metrics use encode_independent() (full CLS), not this tuple.
        return img_emb, txt_emb, filip_scores, local_features, (dropped_img_emb, txt_emb)

    @torch.no_grad()
    def encode_independent(self, images, text):
        """
        Evaluation encoding — text-independent full ViT forward (all 196 patches).

        The FILIP-drop forward used during training produces a text-conditioned
        image embedding (the CLS attends only to patches selected for the paired
        text), which violates retrieval evaluation independence and inflates R@K.
        Using the full 12-block forward gives a genuinely text-independent image
        embedding suitable for cross-modal retrieval benchmarks.
        """
        vit_trunk = self.backbone.clip_model.visual.trunk
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)
        for block in vit_trunk.blocks:
            x = block(x)
        x = vit_trunk.norm(x)
        cls_token = x[:, 0, :]
        img_emb = F.normalize(self._project_embedding(cls_token), p=2, dim=-1)
        txt_emb = F.normalize(self.backbone.encode_text(text), dim=-1)
        return img_emb, txt_emb


class ModelFAdaptive(nn.Module):
    """
    Model F Adaptive: Text-conditioned FILIP scoring + adaptive K via STE + sparsity.

    Unlike Model F (fixed TopK budget), this model learns HOW MANY patches to keep
    per image-text pair by using a soft threshold on FILIP scores instead of TopK.

    Two competing forces determine the effective K:
      - Probe FILIP loss: pushes text-relevant patch scores positive  (keep them)
      - Sparsity loss:    pushes the mean activation toward target_ratio (be selective)

    The threshold at score=0 has a natural interpretation:
      "keep patches that are positively correlated with the text, discard the rest."

    Architecture:
      1. Full ViT forward (all 12 blocks, all 196 patches).
      2. At drop_layer: snapshot intermediate features → compute FILIP scores
         using probe projections W_p_probe / W_t_probe (text-conditioned).
      3. STE on temperature-scaled scores → binary mask (variable K per image/query).
         Gumbel-Sigmoid optionally replaces STE for training stability.
      4. Masked mean-pool of POST-ViT patch features × mask → project → contrastive.
      5. Probe FILIP loss on ALL 196 patches at drop_layer (key training signal).
      6. Sparsity loss: (sigmoid(scaled_scores).mean() - target_ratio)^2.

    Comparison with Model F (TopK):
      Model F:         fixed K, true intra-ViT sequence reduction.
      ModelFAdaptive:  variable K per query, full ViT, masked mean-pool.
      Variable K means: "cardiomegaly" → few cardiac patches kept;
                        "bilateral infiltrates" → patches across both lungs kept.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomedCLIPBackbone(
            model_name=cfg['model']['vision_backbone'],
            device=cfg['training']['device']
        )
        self.embed_dim = 768
        self.proj_dim  = 512

        self.drop_layer = cfg['model'].get('drop_layer', 6)

        # Temperature for STE binarisation: scales FILIP scores ∈ [-1,1] → [-T, T].
        # Larger T makes the selection more binary (sigmoid closer to {0, 1}).
        # T=5 → sigmoid(±1) = 0.007 / 0.993   (very binary)
        # T=1 → sigmoid(±1) = 0.27  / 0.73    (soft, noisy selection)
        self.drop_temperature = cfg['model'].get('drop_temperature', 5.0)

        # Optional Gumbel-Sigmoid instead of plain STE (softer, gradient-smoothing)
        self.use_gumbel = cfg['model'].get('use_gumbel', False)
        self._tau = cfg['model'].get('gumbel_tau_start', 1.0)

        # Probe projections for FILIP scoring at drop_layer (768 → 512)
        self.probe_patch_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.probe_token_proj = nn.Linear(self.embed_dim, self.proj_dim)

        self.logit_scale = self.backbone.clip_model.logit_scale

    # ─────────────────────── helpers ────────────────────────────────────────

    def set_tau(self, tau):
        """Update Gumbel temperature (called by trainer for annealing)."""
        self._tau = tau

    def _project_embedding(self, embedding):
        visual_model = self.backbone.clip_model.visual
        if hasattr(visual_model, 'proj') and visual_model.proj is not None:
            return embedding @ visual_model.proj.to(embedding.dtype)
        elif hasattr(visual_model, 'head') and visual_model.head is not None:
            return visual_model.head(embedding)
        else:
            raise AttributeError("Cannot project 768->512. Check model architecture.")

    def _compute_filip_scores(self, patch_feats_768, token_feats_768, attn_mask):
        """
        Per-patch FILIP importance: score_i = max_{l: valid} cos(W_p_probe*p_i, W_t_probe*t_l)
Identical to ModelF._compute_filip_scores.
        """
        pf = F.normalize(self.probe_patch_proj(patch_feats_768), dim=-1)
        tf = F.normalize(self.probe_token_proj(token_feats_768), dim=-1)
        sim = torch.bmm(pf, tf.transpose(1, 2))
        pad = (attn_mask == 0).unsqueeze(1)
        sim = sim.masked_fill(pad, torch.finfo(sim.dtype).min)
        return sim.max(dim=-1).values                               # (B, 196)

    # ─────────────────────── forward helpers ────────────────────────────────

    def _forward_full_vit_with_probe(self, images):
        """
        Full ViT forward (all 12 blocks) with intermediate feature snapshot
        at drop_layer (after running drop_layer blocks, same position as Model F/E).

        Returns:
            final_features: (B, 197, D)  post-ViT normed features
            probe_feats:    (B, 196, D)  intermediate patch features at drop_layer
        """
        vit_trunk = self.backbone.clip_model.visual.trunk

        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)

        probe_feats = None
        for i, block in enumerate(vit_trunk.blocks):
            x = block(x)
            if i == self.drop_layer - 1:
                # Snapshot after running exactly drop_layer blocks (consistent with Model F/E)
                probe_feats = x[:, 1:, :].clone()                  # (B, 196, D)

        x = vit_trunk.norm(x)
        return x, probe_feats

    def get_intermediate_patch_scores(self, images, text):
        """
        Text-conditioned FILIP scores at drop_layer on all 196 patches.
        Used by deletion/insertion test and grounding visualisation.
        Returns unscaled FILIP scores (before temperature multiplication).
        """
        token_feats, attn_mask = self.backbone.encode_text_tokens(text)
        _, probe_feats = self._forward_full_vit_with_probe(images)
        return self._compute_filip_scores(probe_feats, token_feats, attn_mask)

    # ─────────────────────── main forward ───────────────────────────────────

    def forward(self, images, text):
        # Text encoding (needed for FILIP scoring at drop_layer)
        token_feats, attn_mask = self.backbone.encode_text_tokens(text)

        # Full ViT forward + intermediate snapshot
        vit_features, probe_feats = self._forward_full_vit_with_probe(images)

        # Text-conditioned FILIP scores at drop_layer
        filip_scores  = self._compute_filip_scores(probe_feats, token_feats, attn_mask)

        # Scale for STE binarisation and sparsity loss.
        # SparsityLoss receives scaled_scores: (sigmoid(s).mean() - target_ratio)^2.
        # With T=5: sigmoid(±1)=0.007/0.993 → strong gradient, near-binary selection.
        scaled_scores = self.drop_temperature * filip_scores        # (B, 196)

        # Binary mask: variable K per image-text pair
        if self.use_gumbel:
            # Gumbel-Sigmoid: soft exploration, bounded gradients
            mask = gumbel_sigmoid_sample(scaled_scores, tau=self._tau)
        else:
            # Plain STE: threshold at 0 → keep patches with filip_score > 0
            mask = StraightThroughEstimator.apply(scaled_scores)

        # ── Masked mean-pool (text-conditioned auxiliary embedding) ───────
        patch_feats   = vit_features[:, 1:, :]                     # (B, 196, D)
        masked_feats  = patch_feats * mask.unsqueeze(-1)            # (B, 196, D)
        mask_sum      = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        img_emb_768   = masked_feats.sum(dim=1) / mask_sum          # (B, D)
        aux_img_emb   = F.normalize(self._project_embedding(img_emb_768), p=2, dim=-1)

        # ── Primary image embedding: text-independent full CLS ─────────────
        # The full CLS (all 196 patches, all 12 blocks) is used for the main
        # contrastive loss and at evaluation.  This avoids the data leakage
        # caused by text-conditioned masked mean-pool in retrieval evaluation.
        full_cls = vit_features[:, 0, :]
        img_emb  = F.normalize(self._project_embedding(full_cls), p=2, dim=-1)

        txt_emb = F.normalize(self.backbone.encode_text(text), dim=-1)

        # Probe FILIP loss: all 196 patches at drop_layer.
        # Direct gradient to probe_patch_proj and ViT blocks 0..drop_layer-1.
        pf_probe = F.normalize(self.probe_patch_proj(probe_feats), dim=-1)
        tf_probe = F.normalize(self.probe_token_proj(token_feats), dim=-1)
        local_features = [(pf_probe, tf_probe, attn_mask)]

        # img_emb_full = (aux_img_emb, txt_emb):
        #   Trainer adds InfoNCE(aux_img_emb, txt_emb) × contrastive_full_weight
        #   as an auxiliary signal that trains the FILIP scorer + mask head.
        #   scaled_scores are passed as importance_logits for:
        #     - SparsityLoss: (sigmoid(s).mean() - target_ratio)^2
        #     - Trainer mask logging (kept ratio = fraction where scores > 0)
        #     - Grounding vis (monotone w.r.t. raw FILIP scores)
        return img_emb, txt_emb, scaled_scores, local_features, (aux_img_emb, txt_emb)

    @torch.no_grad()
    def encode_independent(self, images, text):
        """
        Evaluation encoding — full ViT forward, CLS token (text-independent).

        The masked-mean-pool used during training requires text to compute the
        FILIP mask, making the image embedding text-conditioned and violating
        retrieval evaluation independence. The CLS token from the full 12-block
        forward provides a text-independent image representation.
        """
        vit_features, _ = self._forward_full_vit_with_probe(images)
        cls_token = vit_features[:, 0, :]
        img_emb = F.normalize(self._project_embedding(cls_token), p=2, dim=-1)
        txt_emb = F.normalize(self.backbone.encode_text(text), dim=-1)
        return img_emb, txt_emb