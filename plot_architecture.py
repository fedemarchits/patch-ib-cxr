"""
Architecture diagram for the Patch-IB CXR model (thesis figure).
Generates a clean, publication-ready block diagram using matplotlib.

Usage:
    python plot_architecture.py                  # Model B (default)
    python plot_architecture.py --variant C      # Model C (with Patch-IB masking)
    python plot_architecture.py --variant D      # Model D (with Top-K selection)
    python plot_architecture.py --variant all    # All variants side by side
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import argparse
import os

# ─── Color Palette (thesis-friendly, dark background) ────────────────────────
BG_COLOR = '#0d1117'
BOX_VISION = '#1f6feb'       # Blue - vision pathway
BOX_TEXT = '#3fb950'          # Green - text pathway
BOX_CROSS = '#a371f7'        # Purple - cross-modal
BOX_LOSS = '#f0883e'         # Orange - losses
BOX_MASK = '#f85149'         # Red - masking / bottleneck
BOX_INPUT = '#8b949e'        # Gray - inputs
ARROW_COLOR = '#c9d1d9'
TEXT_COLOR = '#c9d1d9'
DIM_TEXT = '#8b949e'
BORDER_COLOR = '#30363d'


def draw_box(ax, x, y, w, h, label, color, fontsize=9, sublabel=None, alpha=0.85, bold=True):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor='white',
        linewidth=1.2,
        alpha=alpha,
        zorder=3
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight=weight,
            color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.15, sublabel,
                ha='center', va='center', fontsize=7, fontweight='normal',
                color='#d0d0d0', zorder=4, style='italic')
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, style='->', lw=1.5, ls='-'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, connectionstyle='arc3,rad=0'),
                zorder=2)


def draw_curved_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, rad=0.3, lw=1.5, ls='-'):
    """Draw a curved arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)


def draw_dim_label(ax, x, y, text, fontsize=7):
    """Draw a dimension annotation."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=DIM_TEXT, zorder=4, family='monospace')


def draw_brace_region(ax, x, y, w, h, label, color, alpha=0.08):
    """Draw a shaded background region with a label."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=alpha,
        linestyle='--',
        zorder=0
    )
    ax.add_patch(box)
    ax.text(x + w/2 - 0.1, y + h/2 - 0.12, label,
            ha='right', va='top', fontsize=7, color=color, alpha=0.7,
            fontweight='bold', zorder=1)


def plot_model_b(ax, title="Model B: BiomedCLIP + Local Alignment"):
    """Draw Model B architecture (no masking, with local alignment)."""

    # ── Layout constants ──
    vx = -2.0    # Vision column x
    tx = 2.0     # Text column x
    cx = 0.0     # Center (cross-modal)
    bw = 2.2     # Box width
    bh = 0.45    # Box height

    # ── Background regions ──
    draw_brace_region(ax, vx, 4.5, 2.8, 6.2, "Vision Encoder", BOX_VISION)
    draw_brace_region(ax, tx, 4.5, 2.8, 6.2, "Text Encoder", BOX_TEXT)
    draw_brace_region(ax, cx, -0.5, 6.5, 2.8, "Cross-Modal Interaction", BOX_CROSS)

    # ═══ INPUTS ═══
    draw_box(ax, vx, 7.8, bw, bh, "Chest X-Ray", BOX_INPUT, sublabel="224 × 224 × 3")
    draw_box(ax, tx, 7.8, bw, bh, "Radiology Report", BOX_INPUT, sublabel="Findings + Impression")

    # ═══ BACKBONE ENCODERS ═══
    draw_box(ax, vx, 6.8, bw, bh+0.1, "ViT-B/16", BOX_VISION, fontsize=10, sublabel="12 Transformer layers")
    draw_box(ax, tx, 6.8, bw, bh+0.1, "PubMedBERT", BOX_TEXT, fontsize=10, sublabel="12 Transformer layers")

    # Arrows: input → backbone
    draw_arrow(ax, vx, 7.8 - bh/2, vx, 6.8 + (bh+0.1)/2)
    draw_arrow(ax, tx, 7.8 - bh/2, tx, 6.8 + (bh+0.1)/2)

    # ═══ VISION OUTPUTS ═══
    draw_box(ax, vx - 0.7, 5.6, 1.3, bh, "[CLS]", BOX_VISION, sublabel="768-d")
    draw_box(ax, vx + 0.7, 5.6, 1.3, bh, "Patches", BOX_VISION, sublabel="196 × 768")

    # Arrows: backbone → outputs (split)
    draw_arrow(ax, vx - 0.3, 6.8 - (bh+0.1)/2, vx - 0.7, 5.6 + bh/2)
    draw_arrow(ax, vx + 0.3, 6.8 - (bh+0.1)/2, vx + 0.7, 5.6 + bh/2)

    # ═══ TEXT OUTPUTS ═══
    draw_box(ax, tx - 0.7, 5.6, 1.3, bh, "Pooled", BOX_TEXT, sublabel="512-d")
    draw_box(ax, tx + 0.7, 5.6, 1.3, bh, "Tokens", BOX_TEXT, sublabel="L × 768")

    # Arrows: backbone → outputs (split)
    draw_arrow(ax, tx - 0.3, 6.8 - (bh+0.1)/2, tx - 0.7, 5.6 + bh/2)
    draw_arrow(ax, tx + 0.3, 6.8 - (bh+0.1)/2, tx + 0.7, 5.6 + bh/2)

    # ═══ GLOBAL PATH (CLS → projection → embedding) ═══
    draw_box(ax, vx - 0.7, 4.6, 1.6, bh, "Proj 768→512", BOX_VISION, fontsize=8, alpha=0.7)
    draw_arrow(ax, vx - 0.7, 5.6 - bh/2, vx - 0.7, 4.6 + bh/2)

    draw_box(ax, vx - 0.7, 3.7, 1.6, bh, "L2 Normalize", BOX_VISION, fontsize=8, alpha=0.7)
    draw_arrow(ax, vx - 0.7, 4.6 - bh/2, vx - 0.7, 3.7 + bh/2)

    # Text pooled is already 512-d, just L2 norm
    draw_box(ax, tx - 0.7, 4.2, 1.6, bh, "L2 Normalize", BOX_TEXT, fontsize=8, alpha=0.7)
    draw_arrow(ax, tx - 0.7, 5.6 - bh/2, tx - 0.7, 4.2 + bh/2)

    # ═══ LOCAL PATH (Patches → proj → features) ═══
    draw_box(ax, vx + 0.7, 4.6, 1.6, bh, "Proj 768→512", BOX_VISION, fontsize=8, alpha=0.7)
    draw_arrow(ax, vx + 0.7, 5.6 - bh/2, vx + 0.7, 4.6 + bh/2)

    draw_box(ax, vx + 0.7, 3.7, 1.6, bh, "L2 Normalize", BOX_VISION, fontsize=8, alpha=0.7)
    draw_arrow(ax, vx + 0.7, 4.6 - bh/2, vx + 0.7, 3.7 + bh/2)

    draw_box(ax, tx + 0.7, 4.6, 1.6, bh, "Proj 768→512", BOX_TEXT, fontsize=8, alpha=0.7)
    draw_arrow(ax, tx + 0.7, 5.6 - bh/2, tx + 0.7, 4.6 + bh/2)

    draw_box(ax, tx + 0.7, 3.7, 1.6, bh, "L2 Normalize", BOX_TEXT, fontsize=8, alpha=0.7)
    draw_arrow(ax, tx + 0.7, 4.6 - bh/2, tx + 0.7, 3.7 + bh/2)

    # Dimension labels
    draw_dim_label(ax, vx + 0.7, 3.2, "196 × 512")
    draw_dim_label(ax, tx + 0.7, 3.2, "L × 512")

    # ═══ EMBEDDINGS ═══
    draw_dim_label(ax, vx - 0.7, 3.25, "img_emb (512)")
    draw_dim_label(ax, tx - 0.7, 3.75, "txt_emb (512)")

    # ═══ CROSS-ATTENTION MODULE ═══
    draw_box(ax, cx, 0.8, 3.8, 0.65, "Multi-Head Cross-Attention", BOX_CROSS, fontsize=10,
             sublabel="Q = text tokens, K/V = image patches")

    # Arrows: patches → cross-attention
    draw_curved_arrow(ax, vx + 0.7, 3.7 - bh/2, cx - 0.5, 0.8 + 0.65/2, color=BOX_VISION, rad=0.2)
    # Arrows: tokens → cross-attention
    draw_curved_arrow(ax, tx + 0.7, 3.7 - bh/2, cx + 0.5, 0.8 + 0.65/2, color=BOX_TEXT, rad=-0.2)

    # ═══ ALIGNED FEATURES ═══
    draw_box(ax, cx, -0.1, 2.5, bh, "Aligned Features", BOX_CROSS, sublabel="L × 512")
    draw_arrow(ax, cx, 0.8 - 0.65/2, cx, -0.1 + bh/2, color=BOX_CROSS)

    # ═══ SYMMETRIC DIRECTION ═══
    draw_box(ax, cx, -0.9, 2.5, bh, "Symmetric Reverse", BOX_CROSS, fontsize=8,
             sublabel="Q = patches, K/V = tokens", alpha=0.6)
    draw_arrow(ax, cx, -0.1 - bh/2, cx, -0.9 + bh/2, color=BOX_CROSS, ls='--')

    # ═══ LOSSES ═══
    loss_y = -2.5
    draw_box(ax, -2.0, loss_y, 2.5, 0.65, "Contrastive Loss", BOX_LOSS, fontsize=10,
             sublabel="InfoNCE (symmetric)")
    draw_box(ax, 2.0, loss_y, 2.5, 0.65, "Local Alignment Loss", BOX_LOSS, fontsize=10,
             sublabel="Cosine Distance (symmetric)")

    # Arrows: embeddings → contrastive loss
    draw_curved_arrow(ax, vx - 0.7, 3.25 - 0.15, -2.0, loss_y + 0.65/2, color=BOX_LOSS, rad=0.3)
    draw_curved_arrow(ax, tx - 0.7, 3.75 - 0.15, -2.0, loss_y + 0.65/2, color=BOX_LOSS, rad=-0.3)

    # Arrows: aligned features + tokens → local loss
    draw_curved_arrow(ax, cx, -0.9 - bh/2, 2.0, loss_y + 0.65/2, color=BOX_LOSS, rad=-0.15)

    # ═══ TOTAL LOSS ═══
    draw_box(ax, cx, -3.8, 4.5, 0.55, "L = L_contrastive + w_local · L_local", BOX_LOSS,
             fontsize=10, alpha=0.7)
    draw_arrow(ax, -2.0, loss_y - 0.65/2, cx - 0.5, -3.8 + 0.55/2, color=BOX_LOSS)
    draw_arrow(ax, 2.0, loss_y - 0.65/2, cx + 0.5, -3.8 + 0.55/2, color=BOX_LOSS)

    ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=15)


def plot_model_cd(ax, variant="C"):
    """Draw Model C or D architecture (with Patch-IB masking + local alignment)."""
    is_d = variant.upper() == "D"
    mask_label = "Top-K Selection" if is_d else "Threshold Masking"
    mask_sub = f"STE + k_ratio annealing" if is_d else "STE + sigmoid threshold"
    title = f"Model {variant.upper()}: Patch-IB + Local Alignment"

    # ── Layout ──
    vx = -2.2
    tx = 2.2
    cx = 0.0
    bw = 2.0
    bh = 0.42

    # ── Background ──
    draw_brace_region(ax, vx, 5.2, 3.0, 5.0, "Vision Encoder", BOX_VISION)
    draw_brace_region(ax, tx, 5.2, 3.0, 5.0, "Text Encoder", BOX_TEXT)
    draw_brace_region(ax, cx - 0.2, 0.7, 5.5, 2.2, "Cross-Modal", BOX_CROSS)
    draw_brace_region(ax, vx + 0.2, 2.3, 3.0, 1.8, "Information Bottleneck", BOX_MASK)

    # ═══ INPUTS ═══
    draw_box(ax, vx, 7.8, bw, bh, "Chest X-Ray", BOX_INPUT, sublabel="224 × 224")
    draw_box(ax, tx, 7.8, bw, bh, "Report", BOX_INPUT, sublabel="Findings + Impression")

    # ═══ BACKBONES ═══
    draw_box(ax, vx, 7.0, bw, bh+0.05, "ViT-B/16", BOX_VISION, fontsize=10)
    draw_box(ax, tx, 7.0, bw, bh+0.05, "PubMedBERT", BOX_TEXT, fontsize=10)
    draw_arrow(ax, vx, 7.8 - bh/2, vx, 7.0 + (bh+0.05)/2)
    draw_arrow(ax, tx, 7.8 - bh/2, tx, 7.0 + (bh+0.05)/2)

    # ═══ VISION OUTPUTS ═══
    draw_box(ax, vx - 0.7, 6.0, 1.2, bh, "[CLS]", BOX_VISION, fontsize=8, sublabel="768")
    draw_box(ax, vx + 0.7, 6.0, 1.2, bh, "Patches", BOX_VISION, fontsize=8, sublabel="196×768")
    draw_arrow(ax, vx - 0.3, 7.0 - (bh+0.05)/2, vx - 0.7, 6.0 + bh/2)
    draw_arrow(ax, vx + 0.3, 7.0 - (bh+0.05)/2, vx + 0.7, 6.0 + bh/2)

    # ═══ TEXT OUTPUTS ═══
    draw_box(ax, tx - 0.7, 6.0, 1.2, bh, "Pooled", BOX_TEXT, fontsize=8, sublabel="512")
    draw_box(ax, tx + 0.7, 6.0, 1.2, bh, "Tokens", BOX_TEXT, fontsize=8, sublabel="L×768")
    draw_arrow(ax, tx - 0.3, 7.0 - (bh+0.05)/2, tx - 0.7, 6.0 + bh/2)
    draw_arrow(ax, tx + 0.3, 7.0 - (bh+0.05)/2, tx + 0.7, 6.0 + bh/2)

    # ═══ FULL PATH (CLS → projection → full embedding) ═══
    draw_box(ax, vx - 0.7, 5.1, 1.5, bh, "Proj → L2", BOX_VISION, fontsize=7, alpha=0.7)
    draw_arrow(ax, vx - 0.7, 6.0 - bh/2, vx - 0.7, 5.1 + bh/2)
    draw_dim_label(ax, vx - 0.7, 4.7, "full_emb (512)")

    # ═══ TEXT GLOBAL ═══
    draw_box(ax, tx - 0.7, 5.1, 1.5, bh, "L2 Norm", BOX_TEXT, fontsize=7, alpha=0.7)
    draw_arrow(ax, tx - 0.7, 6.0 - bh/2, tx - 0.7, 5.1 + bh/2)
    draw_dim_label(ax, tx - 0.7, 4.7, "txt_emb (512)")

    # ═══ MASKING HEAD (Information Bottleneck) ═══
    draw_box(ax, vx + 0.7, 5.1, 1.5, bh, "Importance MLP", BOX_MASK, fontsize=7,
             sublabel="768→192→1")
    draw_arrow(ax, vx + 0.7, 6.0 - bh/2, vx + 0.7, 5.1 + bh/2)

    draw_box(ax, vx + 0.7, 4.2, 1.8, bh+0.05, mask_label, BOX_MASK, fontsize=8,
             sublabel=mask_sub)
    draw_arrow(ax, vx + 0.7, 5.1 - bh/2, vx + 0.7, 4.2 + (bh+0.05)/2, color=BOX_MASK)

    # Masked patches → masked embedding
    draw_box(ax, vx - 0.1, 3.2, 2.0, bh, "Mask & Pool", BOX_MASK, fontsize=8, alpha=0.7,
             sublabel="Σ(patch × mask) / Σmask")
    draw_arrow(ax, vx + 0.7, 4.2 - (bh+0.05)/2, vx - 0.1, 3.2 + bh/2, color=BOX_MASK)

    draw_box(ax, vx - 0.1, 2.3, 1.8, bh, "Proj → L2", BOX_MASK, fontsize=7, alpha=0.7)
    draw_arrow(ax, vx - 0.1, 3.2 - bh/2, vx - 0.1, 2.3 + bh/2, color=BOX_MASK)
    draw_dim_label(ax, vx - 0.1, 1.85, "mask_emb (512)")

    # ═══ LOCAL PATH ═══
    # Patches (or selected patches for Model D) → projection
    patch_label = "Top-K Patches" if is_d else "All Patches"
    draw_box(ax, tx + 0.7, 5.1, 1.5, bh, "Proj 768→512", BOX_TEXT, fontsize=7, alpha=0.7)
    draw_arrow(ax, tx + 0.7, 6.0 - bh/2, tx + 0.7, 5.1 + bh/2)

    draw_box(ax, tx + 0.7, 4.3, 1.5, bh, "L2 Norm", BOX_TEXT, fontsize=7, alpha=0.7)
    draw_arrow(ax, tx + 0.7, 5.1 - bh/2, tx + 0.7, 4.3 + bh/2)

    # Also project patches for local alignment
    if is_d:
        # Arrow from masking head to patch proj (selected patches only)
        draw_curved_arrow(ax, vx + 0.7, 4.2 - (bh+0.05)/2, cx - 0.3, 1.8 + 0.6/2,
                         color=BOX_MASK, rad=0.3, ls='--')
        draw_dim_label(ax, cx - 1.5, 2.6, f"K×512")
    else:
        draw_curved_arrow(ax, vx + 0.7, 5.1 - bh/2, cx - 0.3, 1.8 + 0.6/2,
                         color=BOX_VISION, rad=0.3, ls='--')

    # ═══ CROSS-ATTENTION ═══
    draw_box(ax, cx, 1.0, 3.5, 0.55, "Multi-Head Cross-Attention", BOX_CROSS, fontsize=9,
             sublabel="Q=tokens, K/V=patches")
    draw_curved_arrow(ax, tx + 0.7, 4.3 - bh/2, cx + 0.5, 1.0 + 0.55/2,
                     color=BOX_TEXT, rad=-0.2)
    # Patches to cross-attn
    # Patch features (left side) go to cross attention
    draw_box(ax, cx - 1.5, 1.8, 1.5, 0.4, "Patch Proj", BOX_VISION, fontsize=7, alpha=0.6,
             sublabel="768→512, L2")
    draw_arrow(ax, cx - 1.5, 1.8 - 0.4/2, cx - 0.5, 1.0 + 0.55/2, color=BOX_VISION)

    draw_box(ax, cx, 0.1, 2.2, bh, "Aligned Features", BOX_CROSS, fontsize=8, sublabel="L×512")
    draw_arrow(ax, cx, 1.0 - 0.55/2, cx, 0.1 + bh/2, color=BOX_CROSS)

    # ═══ LOSSES ═══
    loss_y = -1.5
    lw = 2.0
    lh = 0.55

    draw_box(ax, -3.0, loss_y, lw, lh, "L_NCE_full", BOX_LOSS, fontsize=8,
             sublabel="Full contrastive")
    draw_box(ax, -1.0, loss_y, lw, lh, "L_NCE_mask", BOX_LOSS, fontsize=8,
             sublabel="Masked contrastive")
    draw_box(ax, 1.0, loss_y, lw, lh, "L_local", BOX_LOSS, fontsize=8,
             sublabel="Cosine alignment")
    draw_box(ax, 3.0, loss_y, lw, lh, "L_sparse", BOX_LOSS, fontsize=8,
             sublabel="Sparsity penalty")

    # Arrows to losses
    draw_curved_arrow(ax, vx - 0.7, 4.7 - 0.1, -3.0, loss_y + lh/2, color=BOX_LOSS, rad=0.4)
    draw_curved_arrow(ax, vx - 0.1, 1.85 - 0.1, -1.0, loss_y + lh/2, color=BOX_LOSS, rad=0.3)
    draw_arrow(ax, cx, 0.1 - bh/2, 1.0, loss_y + lh/2, color=BOX_LOSS)

    # Sparsity from importance logits
    draw_curved_arrow(ax, vx + 0.7, 4.2 - (bh+0.05)/2, 3.0, loss_y + lh/2,
                     color=BOX_LOSS, rad=-0.4)

    # Consistency loss
    draw_box(ax, cx, loss_y - 1.0, lw, lh, "L_consistency", BOX_LOSS, fontsize=8,
             sublabel="full ↔ masked similarity", alpha=0.7)
    draw_curved_arrow(ax, -3.0, loss_y - lh/2, cx - 0.3, loss_y - 1.0 + lh/2,
                     color=BOX_LOSS, rad=0.15, ls='--')
    draw_curved_arrow(ax, -1.0, loss_y - lh/2, cx + 0.3, loss_y - 1.0 + lh/2,
                     color=BOX_LOSS, rad=-0.15, ls='--')

    # Total loss
    draw_box(ax, cx, loss_y - 2.2, 5.0, 0.55, "L = L_NCE_full + L_NCE_mask + w_local·L_local + w_sp·L_sparse + L_cons",
             BOX_LOSS, fontsize=8, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=15)


def create_figure(variant="B"):
    """Create and save the architecture figure."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR,
        'text.color': TEXT_COLOR,
        'font.family': 'sans-serif',
    })

    if variant.upper() == "ALL":
        fig, axes = plt.subplots(1, 3, figsize=(36, 14))
        for a in axes:
            a.set_xlim(-5, 5)
            a.set_ylim(-5, 9)
            a.set_aspect('equal')
            a.axis('off')
        plot_model_b(axes[0], title="Model B")
        plot_model_cd(axes[1], variant="C")
        plot_model_cd(axes[2], variant="D")
        fig.suptitle("Patch-IB CXR: Model Variants", fontsize=18, fontweight='bold',
                     color=TEXT_COLOR, y=0.98)
        filename = "architecture_all_variants.png"
    elif variant.upper() in ("C", "D"):
        fig, ax = plt.subplots(1, 1, figsize=(12, 14))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4.5, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        plot_model_cd(ax, variant=variant.upper())
        filename = f"architecture_model_{variant.lower()}.png"
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        plot_model_b(ax)
        filename = "architecture_model_b.png"

    os.makedirs("imgs", exist_ok=True)
    filepath = os.path.join("imgs", filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model architecture diagram.")
    parser.add_argument("--variant", type=str, default="B",
                        choices=["B", "C", "D", "all"],
                        help="Model variant to plot (default: B)")
    args = parser.parse_args()
    create_figure(args.variant)
