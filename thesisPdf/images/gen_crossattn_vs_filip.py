import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('#f9f9f9')

patch_color = '#444444'
token_colors = ['#E84393', '#4A90D9', '#F5A623']
patch_labels = ['P1', 'P2', 'P3', 'P4', 'P5']
token_labels = ['T1', 'T2', 'T3']

patch_y = [0.85, 0.68, 0.50, 0.33, 0.16]
token_y = [0.75, 0.50, 0.25]

box_w = 0.18
box_h = 0.10

def draw_boxes(ax, x, ys, labels, colors):
    for y, label, color in zip(ys, labels, colors):
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='white', linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=4)

# ── LEFT: Cross-attention ──────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#f9f9f9')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Cross-Attention', fontsize=14, fontweight='bold',
             color='#333333', pad=12)

px, tx = 0.22, 0.78

# Patches (grey)
draw_boxes(ax, px, patch_y, patch_labels,
           ['#888888'] * len(patch_labels))
# Tokens (colored)
draw_boxes(ax, tx, token_y, token_labels, token_colors)

# All-to-all arrows with varying alpha (weighted blend)
np.random.seed(7)
for py in patch_y:
    weights = np.random.dirichlet(np.ones(3))   # random weights sum to 1
    for ty, w in zip(token_y, weights):
        ax.annotate('', xy=(tx - box_w/2, ty),
                    xytext=(px + box_w/2, py),
                    arrowprops=dict(
                        arrowstyle='->', color='#888888',
                        lw=0.6 + w * 3.5,          # thickness = weight
                        alpha=0.25 + w * 0.75
                    ), zorder=2)

ax.text(0.5, 0.04, 'Every patch contributes to every token\n— weighted average, can collapse',
        ha='center', va='bottom', fontsize=9, color='#666666',
        style='italic')

# ── RIGHT: FILIP ───────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#f9f9f9')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('FILIP', fontsize=14, fontweight='bold',
             color='#333333', pad=12)

# Each patch maps to exactly one token (best match)
best_match = [0, 2, 1, 0, 2]   # index into token_y

patch_colors_filip = [token_colors[b] for b in best_match]
draw_boxes(ax, px, patch_y, patch_labels, patch_colors_filip)
draw_boxes(ax, tx, token_y, token_labels, token_colors)

for py, bm in zip(patch_y, best_match):
    ty = token_y[bm]
    color = token_colors[bm]
    ax.annotate('', xy=(tx - box_w/2, ty),
                xytext=(px + box_w/2, py),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=2.5, alpha=0.9
                ), zorder=2)

ax.text(0.5, 0.04, 'Each patch commits to its best-matching token\n— one bold arrow, no collapse',
        ha='center', va='bottom', fontsize=9, color='#666666',
        style='italic')

plt.tight_layout(pad=2.0)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/crossattn_vs_filip.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
