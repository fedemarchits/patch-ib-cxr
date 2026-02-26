"""
Deletion / Insertion faithfulness test — Approach 1 (feature-space masking).

For each image-text pair:
  1. Get patch importance scores z from the model (model-specific strategy).
  2. Sort patches by z descending (most important first).
  3. Deletion curve:  start from all 196 patches, progressively remove top-k
                      patches and recompute mean-pool similarity with text.
  4. Insertion curve: start from 0 patches, progressively add top-k patches
                      and recompute mean-pool similarity with text.
  5. Report AUC of both curves + full curve arrays.

Score strategy per model type:
  - Models C / D  (mask_head present):      explicit importance_logits from MaskHead
  - Model E       (scorer + drop_layer):    importance_logits from intra-ViT scorer
  - Model B FILIP (patch_proj / token_proj): max token-patch cosine similarity (FILIP)
  - Model A / fallback:                      per-patch cosine similarity with text emb
                                             (each patch projected to 512, dot with txt)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Score extraction
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_patch_scores(model, images, text):
    """
    Compute per-patch importance scores and return all ingredients for the curves.

    Returns:
        patch_feats : (B, 196, 768)  final post-ViT patch features
        scores      : (B, 196)       importance scores (higher = more important)
        txt_emb     : (B, 512)       L2-normalised text embedding
    """
    # Final ViT patch features — same for all models (196 patches, dim 768)
    all_tokens = model.backbone.encode_image_patches(images)   # (B, 197, 768)
    patch_feats = all_tokens[:, 1:, :]                         # (B, 196, 768)

    # Text embedding (normalised, 512-d)
    txt_emb = F.normalize(model.backbone.encode_text(text), dim=-1)  # (B, 512)

    # ── Models C / D: explicit MaskHead importance logits ──
    if hasattr(model, 'mask_head'):
        out = model.mask_head(patch_feats)
        scores = out[1]   # index 1 = importance_logits for both TopK and STE heads

    # ── Model F: text-conditioned FILIP scorer at intermediate ViT layer ──
    elif hasattr(model, 'probe_patch_proj') and hasattr(model, 'drop_layer'):
        # Use intermediate features (not post-ViT) — that's where the drop happens.
        # patch_feats above is post-ViT (used for the deletion/insertion curves).
        # Scores come from drop_layer features projected via probe projections.
        scores = model.get_intermediate_patch_scores(images, text)  # (B, 196)

    # ── Model E: lightweight scorer on intermediate ViT features ──
    elif hasattr(model, 'scorer') and hasattr(model, 'drop_layer'):
        vit_trunk = model.backbone.clip_model.visual.trunk
        x = vit_trunk.patch_embed(images)
        x = vit_trunk._pos_embed(x)
        x = vit_trunk.patch_drop(x)
        x = vit_trunk.norm_pre(x)
        for i in range(model.drop_layer):
            x = vit_trunk.blocks[i](x)
        intermediate_patches = x[:, 1:, :]           # (B, 196, 768)
        scores = model.scorer(intermediate_patches)   # (B, 196)

    # ── Models B with FILIP: max token-patch cosine similarity ──
    elif hasattr(model, 'patch_proj') and hasattr(model, 'token_proj'):
        token_emb, attn_mask = model.backbone.encode_text_tokens(text)
        pf = F.normalize(model.patch_proj(patch_feats), dim=-1)   # (B, 196, 512)
        tf = F.normalize(model.token_proj(token_emb), dim=-1)     # (B,   L, 512)
        sim = torch.bmm(pf, tf.transpose(1, 2))                   # (B, 196,   L)
        # Mask padding tokens so they don't inflate the max
        pad = (attn_mask == 0).unsqueeze(1)                        # (B,   1,   L)
        sim = sim.masked_fill(pad, -1e9)
        scores = sim.max(dim=-1).values                            # (B, 196)

    # ── Model A / fallback: per-patch projection-space similarity with text ──
    else:
        scores = _projection_proxy_scores(model, patch_feats, txt_emb)

    return patch_feats, scores, txt_emb


def _projection_proxy_scores(model, patch_feats, txt_emb):
    """
    Project each patch independently to 512-d, compute cosine similarity
    with the (already normalised) text embedding.
    Gives a natural text-relevance ranking for Model A.
    """
    B, M, D = patch_feats.shape
    flat = patch_feats.reshape(B * M, D)
    projected = model._project_embedding(flat).reshape(B, M, -1)   # (B, 196, 512)
    projected = F.normalize(projected, dim=-1)
    scores = (projected * txt_emb.unsqueeze(1)).sum(-1)             # (B, 196)
    return scores


# ─────────────────────────────────────────────────────────────
# Curve computation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_curves(patch_feats, scores, txt_emb, model, n_steps=20):
    """
    Compute deletion and insertion curves for a batch.

    Args:
        patch_feats : (B, 196, 768)
        scores      : (B, 196)       importance scores
        txt_emb     : (B, 512)       normalised text embeddings
        model       : needed for _project_embedding (768 → 512)
        n_steps     : number of evaluation steps (default 20 = every 5%)

    Returns:
        deletion  : (B, n_steps+1)  cosine similarity at each deletion step
        insertion : (B, n_steps+1)  cosine similarity at each insertion step
        fractions : list[float]     x-axis values (fraction of patches kept)
    """
    B, M, D = patch_feats.shape

    # Sort patches: index 0 = most important
    sorted_idx = torch.argsort(scores, descending=True, dim=1)          # (B, M)
    sorted_feats = torch.gather(
        patch_feats, 1,
        sorted_idx.unsqueeze(-1).expand(-1, -1, D)
    )                                                                    # (B, M, D)

    # Evenly spaced evaluation points: 0, M/n_steps, 2*M/n_steps, ..., M
    steps = [int(round(i * M / n_steps)) for i in range(n_steps + 1)]
    fractions = [k / M for k in steps]

    deletion_sims  = []
    insertion_sims = []

    for k in steps:
        # ── Deletion: remove the top-k most important patches ──
        if k == M:
            # All important patches removed — no signal left
            del_sim = torch.zeros(B, device=patch_feats.device)
        else:
            remaining = sorted_feats[:, k:, :]          # (B, M-k, D)
            emb = remaining.mean(dim=1)                  # (B, D)
            emb = F.normalize(model._project_embedding(emb), dim=-1)
            del_sim = (emb * txt_emb).sum(-1)            # (B,)
        deletion_sims.append(del_sim)

        # ── Insertion: keep only the top-k most important patches ──
        if k == 0:
            # No patches yet — no signal
            ins_sim = torch.zeros(B, device=patch_feats.device)
        else:
            kept = sorted_feats[:, :k, :]               # (B, k, D)
            emb = kept.mean(dim=1)                       # (B, D)
            emb = F.normalize(model._project_embedding(emb), dim=-1)
            ins_sim = (emb * txt_emb).sum(-1)            # (B,)
        insertion_sims.append(ins_sim)

    deletion  = torch.stack(deletion_sims,  dim=1).cpu().numpy()  # (B, n_steps+1)
    insertion = torch.stack(insertion_sims, dim=1).cpu().numpy()  # (B, n_steps+1)

    return deletion, insertion, fractions


# ─────────────────────────────────────────────────────────────
# Full test runner
# ─────────────────────────────────────────────────────────────

def run_deletion_insertion_test(model, dataloader, device, n_steps=20):
    """
    Run the deletion/insertion test over the full dataloader.

    Returns a dict:
        deletion_curve  : (n_steps+1,)  mean cosine similarity per step
        insertion_curve : (n_steps+1,)  mean cosine similarity per step
        deletion_auc    : float         area under deletion curve  (lower is better:
                                        fast drop = important patches removed first)
        insertion_auc   : float         area under insertion curve (higher is better:
                                        fast rise = important patches added first)
        fractions       : list[float]   x-axis (fraction of patches kept/removed)
    """
    model.eval()
    all_deletion  = []
    all_insertion = []
    fractions     = None

    for batch in tqdm(dataloader, desc="Deletion/Insertion test"):
        if batch is None:
            continue
        images = batch[0].to(device)
        text   = batch[1].to(device)

        patch_feats, scores, txt_emb = get_patch_scores(model, images, text)
        deletion, insertion, fractions = compute_curves(
            patch_feats, scores, txt_emb, model, n_steps=n_steps
        )
        all_deletion.append(deletion)
        all_insertion.append(insertion)

    all_deletion  = np.concatenate(all_deletion,  axis=0)   # (N_test, n_steps+1)
    all_insertion = np.concatenate(all_insertion, axis=0)

    mean_deletion  = all_deletion.mean(axis=0)
    mean_insertion = all_insertion.mean(axis=0)

    # AUC via trapezoidal rule on [0, 1] x-axis
    deletion_auc  = float(np.trapz(mean_deletion,  fractions))
    insertion_auc = float(np.trapz(mean_insertion, fractions))

    print(f"\nDeletion/Insertion results:")
    print(f"  Deletion  AUC: {deletion_auc:.4f}  (lower = faster drop = more faithful)")
    print(f"  Insertion AUC: {insertion_auc:.4f}  (higher = faster rise = more faithful)")

    return {
        'deletion_curve':  mean_deletion.tolist(),
        'insertion_curve': mean_insertion.tolist(),
        'deletion_auc':    deletion_auc,
        'insertion_auc':   insertion_auc,
        'fractions':       fractions,
    }
