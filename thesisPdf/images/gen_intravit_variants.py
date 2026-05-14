import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#f9f9f9')

pink   = '#E84393'
blue   = '#4A90D9'
orange = '#F5A623'
grey   = '#cccccc'
dark   = '#333333'
light  = '#f9f9f9'
green  = '#5DB85D'

def rounded_box(ax, x, y, w, h, color, text, fontsize=9, textcolor='white', alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3,
        alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4, alpha=alpha)

def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)

def setup_ax(ax, title, subtitle):
    ax.set_facecolor(light)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', color=dark, pad=8)
    ax.text(0.5, 0.965, subtitle, ha='center', va='top',
            fontsize=9, color='#888888', style='italic')

def vit_stack(ax, x, y_bottom, y_top, label, color=blue):
    """Draw a stack of ViT layers as a tall rectangle with label."""
    h = y_top - y_bottom
    rect = mpatches.FancyBboxPatch(
        (x - 0.10, y_bottom), 0.20, h,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85
    )
    ax.add_patch(rect)
    ax.text(x, (y_bottom + y_top) / 2, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)

def token_row(ax, y, n_tokens, color, alpha=1.0, label=None):
    """Draw a row of small token squares."""
    xs = np.linspace(0.18, 0.82, n_tokens)
    w, h = 0.055, 0.045
    for x in xs:
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='white', linewidth=0.8,
            zorder=3, alpha=alpha
        )
        ax.add_patch(rect)
    if label:
        ax.text(0.5, y - h/2 - 0.04, label, ha='center', va='top',
                fontsize=8, color='#666666', style='italic')

# ── 1. AgnosticDrop ────────────────────────────────────────────────
ax = axes[0, 0]
setup_ax(ax, 'AgnosticDrop', 'Text-agnostic MLP scorer — ablation baseline')

rounded_box(ax, 0.50, 0.92, 0.28, 0.08, '#555555', 'X-ray (196 tokens)', fontsize=8)
arrow(ax, 0.50, 0.88, 0.50, 0.82)

vit_stack(ax, 0.50, 0.72, 0.82, 'ViT\nlayers 1–6', color=blue)
arrow(ax, 0.50, 0.72, 0.50, 0.65)

# MLP scorer (no text input)
rounded_box(ax, 0.50, 0.59, 0.30, 0.08, green, 'MLP Scorer', fontsize=9)
ax.text(0.50, 0.52, '(image features only)', ha='center', va='top',
        fontsize=8, color='#666666', style='italic')
arrow(ax, 0.50, 0.55, 0.50, 0.48)

# Hard drop box
rounded_box(ax, 0.50, 0.42, 0.26, 0.08, dark, 'Hard Drop (Top-K)', fontsize=8)

# Tokens: kept vs dropped
token_row(ax, 0.33, 5, pink, label='K kept')
token_row(ax, 0.33, 3, grey, alpha=0.35)
ax.text(0.84, 0.33, '+ dropped', ha='left', va='center', fontsize=7, color='#aaaaaa')

arrow(ax, 0.50, 0.38, 0.50, 0.36)
arrow(ax, 0.50, 0.30, 0.50, 0.23)

vit_stack(ax, 0.50, 0.13, 0.23, 'ViT\nlayers 7–12', color=blue)
arrow(ax, 0.50, 0.13, 0.50, 0.07)
rounded_box(ax, 0.50, 0.04, 0.28, 0.06, blue, 'CLS → NCE loss', fontsize=8)

# No text input marker
ax.text(0.85, 0.59, 'No text\nguide', ha='center', va='center',
        fontsize=8, color='#aaaaaa', style='italic')
ax.plot([0.72, 0.84], [0.59, 0.59], color='#aaaaaa', lw=1.2, linestyle='--')

# ── 2. FILIPDrop ───────────────────────────────────────────────────
ax = axes[0, 1]
setup_ax(ax, 'FILIPDrop', 'FILIP-scored hard drop at one layer')

rounded_box(ax, 0.28, 0.92, 0.26, 0.08, '#555555', 'X-ray (196)', fontsize=8)
rounded_box(ax, 0.78, 0.92, 0.26, 0.08, '#555555', 'Report', fontsize=8)
arrow(ax, 0.28, 0.88, 0.28, 0.82)
arrow(ax, 0.78, 0.88, 0.78, 0.82)

vit_stack(ax, 0.28, 0.72, 0.82, 'ViT\n1–6', color=blue)
rounded_box(ax, 0.78, 0.77, 0.22, 0.09, orange, 'BERT', fontsize=10)

arrow(ax, 0.28, 0.72, 0.28, 0.65)

# FILIP scoring
rounded_box(ax, 0.28, 0.59, 0.26, 0.08, pink, 'FILIP Score', fontsize=9)
# Arrow from BERT to FILIP scorer
ax.annotate('', xy=(0.41, 0.59), xytext=(0.67, 0.73),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.5,
                            connectionstyle='arc3,rad=0.2'), zorder=2)
ax.text(0.56, 0.68, 'FILIP', ha='center', fontsize=8, color=orange)

arrow(ax, 0.28, 0.55, 0.28, 0.48)
rounded_box(ax, 0.28, 0.42, 0.26, 0.08, dark, 'Hard Drop (Top-K)', fontsize=8)

token_row(ax, 0.33, 5, pink, label='K kept')
token_row(ax, 0.33, 3, grey, alpha=0.35)
ax.text(0.84, 0.33, '+ dropped', ha='left', va='center', fontsize=7, color='#aaaaaa')

arrow(ax, 0.28, 0.38, 0.28, 0.30)
arrow(ax, 0.28, 0.23, 0.28, 0.16)

vit_stack(ax, 0.28, 0.06, 0.16, 'ViT 7–12', color=blue)
arrow(ax, 0.28, 0.06, 0.28, 0.02)
rounded_box(ax, 0.28, 0.00, 0.26, 0.05, blue, 'CLS → NCE', fontsize=7)

ax.text(0.28, 0.27, 'layer 6', ha='center', va='center',
        fontsize=8, color=pink, fontweight='bold')

# ── 3. SoftGate ────────────────────────────────────────────────────
ax = axes[1, 0]
setup_ax(ax, 'SoftGate', 'Sigmoid re-weighting — no hard drop')

rounded_box(ax, 0.28, 0.92, 0.26, 0.08, '#555555', 'X-ray (196)', fontsize=8)
rounded_box(ax, 0.78, 0.92, 0.26, 0.08, '#555555', 'Report', fontsize=8)
arrow(ax, 0.28, 0.88, 0.28, 0.82)
arrow(ax, 0.78, 0.88, 0.78, 0.82)

vit_stack(ax, 0.28, 0.72, 0.82, 'ViT\n1–6', color=blue)
rounded_box(ax, 0.78, 0.77, 0.22, 0.09, orange, 'BERT', fontsize=10)

arrow(ax, 0.28, 0.72, 0.28, 0.65)
rounded_box(ax, 0.28, 0.59, 0.26, 0.08, pink, 'FILIP Score', fontsize=9)
ax.annotate('', xy=(0.41, 0.59), xytext=(0.67, 0.73),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.5,
                            connectionstyle='arc3,rad=0.2'), zorder=2)

arrow(ax, 0.28, 0.55, 0.28, 0.48)

# Sigmoid gate
rounded_box(ax, 0.28, 0.42, 0.30, 0.08, pink, 'Sigmoid Gate  sigma(T*s)', fontsize=8)
arrow(ax, 0.28, 0.38, 0.28, 0.31)

# All tokens pass, varying opacity
xs_tok = np.linspace(0.10, 0.80, 8)
alphas = [0.95, 0.80, 0.65, 0.55, 0.40, 0.30, 0.22, 0.15]
w, h = 0.055, 0.045
for x, a in zip(xs_tok, alphas):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, 0.26 - h/2), w, h,
        boxstyle="round,pad=0.005",
        facecolor=pink, edgecolor='white', linewidth=0.8,
        zorder=3, alpha=a
    )
    ax.add_patch(rect)
ax.text(0.45, 0.19, 'all 196 pass — re-weighted', ha='center', va='top',
        fontsize=8, color='#666666', style='italic')

arrow(ax, 0.28, 0.23, 0.28, 0.16)
vit_stack(ax, 0.28, 0.06, 0.16, 'ViT 7–12', color=blue)
arrow(ax, 0.28, 0.06, 0.28, 0.02)
rounded_box(ax, 0.28, 0.00, 0.26, 0.05, blue, 'CLS → NCE', fontsize=7)

# ── 4. HierDrop ────────────────────────────────────────────────────
ax = axes[1, 1]
setup_ax(ax, 'HierDrop', 'Two-stage hierarchical drop at L1 and L2')

rounded_box(ax, 0.28, 0.92, 0.26, 0.08, '#555555', 'X-ray (196)', fontsize=8)
rounded_box(ax, 0.78, 0.92, 0.26, 0.08, '#555555', 'Report', fontsize=8)
arrow(ax, 0.28, 0.88, 0.28, 0.82)
arrow(ax, 0.78, 0.88, 0.78, 0.82)
rounded_box(ax, 0.78, 0.77, 0.22, 0.09, orange, 'BERT', fontsize=10)

vit_stack(ax, 0.28, 0.71, 0.82, 'ViT 1–4', color=blue)
arrow(ax, 0.28, 0.71, 0.28, 0.64)

# First drop at L1=4
rounded_box(ax, 0.28, 0.58, 0.26, 0.08, pink, 'FILIP + Drop', fontsize=9)
ax.annotate('', xy=(0.41, 0.58), xytext=(0.67, 0.73),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.4,
                            connectionstyle='arc3,rad=0.2'), zorder=2)
ax.text(0.20, 0.56, 'L1=4', ha='right', va='center',
        fontsize=8, color=pink, fontweight='bold')

token_row(ax, 0.50, 6, pink, label='~120 kept')
arrow(ax, 0.28, 0.54, 0.28, 0.53)
arrow(ax, 0.28, 0.47, 0.28, 0.40)

vit_stack(ax, 0.28, 0.30, 0.40, 'ViT 5–9', color=blue)
arrow(ax, 0.28, 0.30, 0.28, 0.23)

# Second drop at L2=9
rounded_box(ax, 0.28, 0.17, 0.26, 0.08, pink, 'FILIP + Drop', fontsize=9)
ax.annotate('', xy=(0.41, 0.17), xytext=(0.67, 0.73),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.4,
                            connectionstyle='arc3,rad=-0.25'), zorder=2)
ax.text(0.20, 0.15, 'L2=9', ha='right', va='center',
        fontsize=8, color=pink, fontweight='bold')

token_row(ax, 0.09, 4, pink, label='~50 kept')
arrow(ax, 0.28, 0.13, 0.28, 0.11)
arrow(ax, 0.28, 0.05, 0.28, 0.02)
vit_stack(ax, 0.28, -0.02, 0.05, 'ViT 10–12', color=blue)
ax.set_ylim(-0.06, 1)

plt.tight_layout(pad=2.5)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/intravit_variants.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
