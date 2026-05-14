import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#f9f9f9')

pink   = '#E84393'
blue   = '#4A90D9'
orange = '#F5A623'
grey   = '#cccccc'
dark   = '#333333'
light  = '#f9f9f9'

def rounded_box(ax, x, y, w, h, color, text, fontsize=10, textcolor='white'):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)

def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)

def setup_ax(ax, title, subtitle):
    ax.set_facecolor(light)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', color=dark, pad=10)
    ax.text(0.5, 0.94, subtitle, ha='center', va='top',
            fontsize=9, color='#888888', style='italic')

# ── 1. LocalAlign ─────────────────────────────────────────────────
ax = axes[0]
setup_ax(ax, 'LocalAlign', 'Align patches with text — no selection')

# X-ray input
rounded_box(ax, 0.18, 0.75, 0.22, 0.10, '#555555', 'X-ray', fontsize=9)
# Report input
rounded_box(ax, 0.78, 0.75, 0.26, 0.10, '#555555', 'Report', fontsize=9)

# ViT block
rounded_box(ax, 0.18, 0.55, 0.22, 0.10, blue, 'ViT', fontsize=11)
# BERT block
rounded_box(ax, 0.78, 0.55, 0.22, 0.10, orange, 'BERT', fontsize=11)

arrow(ax, 0.18, 0.70, 0.18, 0.60)
arrow(ax, 0.78, 0.70, 0.78, 0.60)

# 196 patches output
rounded_box(ax, 0.18, 0.37, 0.28, 0.09, grey, '196 patches', fontsize=9, textcolor=dark)
# text tokens
rounded_box(ax, 0.78, 0.37, 0.28, 0.09, grey, 'text tokens', fontsize=9, textcolor=dark)

arrow(ax, 0.18, 0.50, 0.18, 0.42)
arrow(ax, 0.78, 0.50, 0.78, 0.42)

# FILIP alignment arrow between patches and tokens
ax.annotate('', xy=(0.64, 0.37), xytext=(0.32, 0.37),
            arrowprops=dict(arrowstyle='<->', color=pink, lw=2.0,
                            connectionstyle='arc3,rad=-0.3'), zorder=2)
ax.text(0.50, 0.25, 'FILIP alignment', ha='center', va='top',
        fontsize=9, color=pink, fontweight='bold')

# CLS → contrastive loss
rounded_box(ax, 0.18, 0.13, 0.22, 0.09, blue, 'CLS', fontsize=10)
arrow(ax, 0.18, 0.32, 0.18, 0.18)
ax.text(0.50, 0.10, '← no patch dropped →', ha='center', va='center',
        fontsize=8, color='#888888', style='italic')

# ── 2. MaskSTE ────────────────────────────────────────────────────
ax = axes[1]
setup_ax(ax, 'MaskSTE', 'Learn a binary mask — threshold-based')

rounded_box(ax, 0.18, 0.88, 0.22, 0.09, '#555555', 'X-ray', fontsize=9)
rounded_box(ax, 0.78, 0.88, 0.26, 0.09, '#555555', 'Report', fontsize=9)
rounded_box(ax, 0.18, 0.73, 0.22, 0.09, blue, 'ViT', fontsize=11)
rounded_box(ax, 0.78, 0.73, 0.22, 0.09, orange, 'BERT', fontsize=11)

arrow(ax, 0.18, 0.83, 0.18, 0.78)
arrow(ax, 0.78, 0.83, 0.78, 0.78)
arrow(ax, 0.18, 0.68, 0.18, 0.61)

# 196 patches
rounded_box(ax, 0.18, 0.55, 0.28, 0.09, grey, '196 patches', fontsize=9, textcolor=dark)

# Mask head
rounded_box(ax, 0.50, 0.55, 0.22, 0.09, pink, 'Mask Head', fontsize=9)
arrow(ax, 0.32, 0.55, 0.39, 0.55)

# STE gate
rounded_box(ax, 0.50, 0.39, 0.22, 0.09, pink, 'STE Gate', fontsize=9)
arrow(ax, 0.50, 0.50, 0.50, 0.44)

# FILIP from BERT to mask head
ax.annotate('', xy=(0.61, 0.55), xytext=(0.78, 0.68),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.5,
                            connectionstyle='arc3,rad=0.2'), zorder=2)
ax.text(0.72, 0.61, 'FILIP', ha='center', fontsize=8, color=orange)

# Selected patches
rounded_box(ax, 0.20, 0.25, 0.10, 0.08, pink, '', fontsize=8)
rounded_box(ax, 0.32, 0.25, 0.10, 0.08, grey, '', fontsize=8)
rounded_box(ax, 0.44, 0.25, 0.10, 0.08, pink, '', fontsize=8)
rounded_box(ax, 0.56, 0.25, 0.10, 0.08, grey, '', fontsize=8)
rounded_box(ax, 0.68, 0.25, 0.10, 0.08, pink, '', fontsize=8)
ax.text(0.44, 0.16, 'variable patches kept', ha='center',
        fontsize=8, color='#666666', style='italic')

arrow(ax, 0.50, 0.34, 0.44, 0.29)

# mean pool → CLS
rounded_box(ax, 0.44, 0.07, 0.26, 0.08, blue, 'mean-pool → NCE loss', fontsize=8)
arrow(ax, 0.44, 0.21, 0.44, 0.11)

# ── 3. MaskTopK ───────────────────────────────────────────────────
ax = axes[2]
setup_ax(ax, 'MaskTopK', 'Deterministic Top-K — fixed budget')

rounded_box(ax, 0.18, 0.88, 0.22, 0.09, '#555555', 'X-ray', fontsize=9)
rounded_box(ax, 0.78, 0.88, 0.26, 0.09, '#555555', 'Report', fontsize=9)
rounded_box(ax, 0.18, 0.73, 0.22, 0.09, blue, 'ViT', fontsize=11)
rounded_box(ax, 0.78, 0.73, 0.22, 0.09, orange, 'BERT', fontsize=11)

arrow(ax, 0.18, 0.83, 0.18, 0.78)
arrow(ax, 0.78, 0.83, 0.78, 0.78)
arrow(ax, 0.18, 0.68, 0.18, 0.61)

rounded_box(ax, 0.18, 0.55, 0.28, 0.09, grey, '196 patches', fontsize=9, textcolor=dark)

rounded_box(ax, 0.50, 0.55, 0.22, 0.09, pink, 'Mask Head', fontsize=9)
arrow(ax, 0.32, 0.55, 0.39, 0.55)

rounded_box(ax, 0.50, 0.39, 0.22, 0.09, pink, 'Top-K Select', fontsize=9)
arrow(ax, 0.50, 0.50, 0.50, 0.44)

ax.annotate('', xy=(0.61, 0.55), xytext=(0.78, 0.68),
            arrowprops=dict(arrowstyle='->', color=orange, lw=1.5,
                            connectionstyle='arc3,rad=0.2'), zorder=2)
ax.text(0.72, 0.61, 'FILIP', ha='center', fontsize=8, color=orange)

# Exactly K patches — show K kept, rest dropped
for i, x in enumerate([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]):
    color = pink if i < 4 else grey
    rounded_box(ax, x, 0.25, 0.08, 0.08, color, '', fontsize=8)

# vertical cut line
ax.axvline(x=0.645, ymin=0.17, ymax=0.33,
           color=dark, lw=2, linestyle='-', zorder=5)
ax.text(0.645, 0.32, ' K', ha='left', va='top',
        fontsize=10, fontweight='bold', color=dark)
ax.text(0.50, 0.16, 'always exactly K patches kept', ha='center',
        fontsize=8, color='#666666', style='italic')

arrow(ax, 0.50, 0.34, 0.40, 0.29)

rounded_box(ax, 0.40, 0.07, 0.26, 0.08, blue, 'mean-pool → NCE loss', fontsize=8)
arrow(ax, 0.40, 0.21, 0.40, 0.11)

plt.tight_layout(pad=2.5)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/postvit_variants.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
