import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('#f9f9f9')

scores = [0.82, 0.61, 0.45, 0.38, 0.29, 0.21, 0.13, 0.07]
n = len(scores)
labels = [f'P{i+1}' for i in range(n)]

kept_color   = '#E84393'
dropped_color = '#cccccc'
bg = '#f9f9f9'

patch_w = 0.07
gap = 0.105
start_x = 0.04

def patch_centers(n, start, gap):
    return [start + i * gap for i in range(n)]

xs = patch_centers(n, start_x, gap)

# Temperature parameter for sigmoid
T = 10.0
threshold = 0.35

def sigmoid(s, T, theta):
    return 1 / (1 + np.exp(-T * (s - theta)))

# ── LEFT: Hard Top-K ──────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(bg)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Hard Drop (Top-K)', fontsize=13, fontweight='bold',
             color='#333333', pad=12)

K = 4
sorted_idx = np.argsort(scores)[::-1]
kept_set = set(sorted_idx[:K])

bar_y = 0.28
for i, (x, score, label) in enumerate(zip(xs, scores, labels)):
    kept = i in kept_set
    color = kept_color if kept else dropped_color
    bar_h = score * 0.50

    rect = mpatches.FancyBboxPatch(
        (x - patch_w/2, bar_y), patch_w, bar_h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white', linewidth=1, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, bar_y - 0.06, label, ha='center', va='top',
            fontsize=9, color='#555555')

    if not kept:
        ax.text(x, bar_y + bar_h + 0.04, 'x', ha='center', va='bottom',
                fontsize=11, color='#aaaaaa', fontweight='bold')
    else:
        # Show weight = 1
        ax.text(x, bar_y + bar_h + 0.04, 'w=1', ha='center', va='bottom',
                fontsize=7, color=kept_color)

# Vertical cut line
ax.annotate('', xy=(xs[K] - patch_w/2 - 0.005, 0.85),
            xytext=(xs[K] - patch_w/2 - 0.005, 0.26),
            arrowprops=dict(arrowstyle='-', color='#333333', lw=2))
ax.text(xs[K] - patch_w/2 - 0.02, 0.56, 'K', ha='right', va='center',
        fontsize=12, fontweight='bold', color='#333333')

ax.text(0.5, 0.08,
        'Dropped tokens never reach upper layers\n— binary, gradient blocked',
        ha='center', va='bottom', fontsize=9, color='#666666', style='italic')
ax.text(0.5, 0.93, f'K={K} kept (w=1), rest removed (w=0)',
        ha='center', va='top', fontsize=9, color='#E84393')

# ── RIGHT: Soft Sigmoid Gate ──────────────────────────────────────
ax = axes[1]
ax.set_facecolor(bg)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Soft Sigmoid Gate', fontsize=13, fontweight='bold',
             color='#333333', pad=12)

bar_y = 0.28
weights = [sigmoid(s, T, threshold) for s in scores]

for x, score, label, w in zip(xs, scores, labels, weights):
    bar_h = score * 0.50
    # Color blended by weight
    alpha = 0.25 + w * 0.75
    color = kept_color

    rect = mpatches.FancyBboxPatch(
        (x - patch_w/2, bar_y), patch_w, bar_h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white', linewidth=1,
        alpha=alpha, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, bar_y - 0.06, label, ha='center', va='top',
            fontsize=9, color='#555555')
    # Show sigmoid weight above bar
    ax.text(x, bar_y + bar_h + 0.04, f'{w:.2f}', ha='center', va='bottom',
            fontsize=7, color=kept_color,
            alpha=0.5 + w * 0.5)

# Sigmoid curve inset (top right)
sx = np.linspace(0, 1, 100)
sy = sigmoid(sx, T, threshold)
inset_x0, inset_y0, inset_w, inset_h = 0.60, 0.55, 0.35, 0.30
inset_ax = ax.inset_axes([inset_x0, inset_y0, inset_w, inset_h],
                          transform=ax.transAxes)
inset_ax.plot(sx, sy, color=kept_color, lw=2)
inset_ax.axvline(x=threshold, color='#F5A623', lw=1.5, linestyle='--')
inset_ax.set_xlim(0, 1)
inset_ax.set_ylim(0, 1)
inset_ax.set_xticks([0, threshold, 1])
inset_ax.set_xticklabels(['0', r'$\theta$', '1'], fontsize=7)
inset_ax.set_yticks([0, 0.5, 1])
inset_ax.set_yticklabels(['0', '0.5', '1'], fontsize=7)
inset_ax.set_facecolor('#f0f0f0')
inset_ax.tick_params(colors='#888888', labelsize=7)
inset_ax.set_title(r'$\sigma(T \cdot (s_i - \theta))$', fontsize=8,
                   color='#333333', pad=3)
for spine in inset_ax.spines.values():
    spine.set_color('#cccccc')

ax.text(0.5, 0.08,
        'All tokens pass — re-weighted by sigmoid score\n— soft, differentiable, gradient flows',
        ha='center', va='bottom', fontsize=9, color='#666666', style='italic')
ax.text(0.5, 0.93, 'All 8 kept, scaled by w = sigma(s)',
        ha='center', va='top', fontsize=9, color='#E84393')

plt.tight_layout(pad=2.0)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/softgate_viz.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
