import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('#f9f9f9')

# Patch importance scores (same for both diagrams)
scores = [0.82, 0.61, 0.45, 0.38, 0.29, 0.21, 0.13, 0.07]
n = len(scores)
labels = [f'P{i+1}' for i in range(n)]

kept_color   = '#E84393'
dropped_color = '#cccccc'
bg = '#f9f9f9'

patch_w = 0.07
patch_h = 0.18
gap = 0.105
start_x = 0.04

def patch_centers(n, start, gap):
    return [start + i * gap for i in range(n)]

xs = patch_centers(n, start_x, gap)

# ── LEFT: STE (threshold-based, variable K) ───────────────────────
ax = axes[0]
ax.set_facecolor(bg)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('STE — Threshold-based', fontsize=13, fontweight='bold',
             color='#333333', pad=12)

threshold = 0.35   # patches above this survive

# Draw threshold line
thresh_x = xs[0] - gap * 0.3
thresh_x2 = xs[-1] + gap * 0.3
thresh_y = 0.62
ax.axhline(y=thresh_y, xmin=0.03, xmax=0.97,
           color='#F5A623', lw=2, linestyle='--', zorder=2)
ax.text(0.97, thresh_y + 0.02, 'threshold', ha='right', va='bottom',
        fontsize=9, color='#F5A623')

for x, score, label in zip(xs, scores, labels):
    kept = score >= threshold
    color = kept_color if kept else dropped_color
    # bar height proportional to score
    bar_h = score * 0.55
    bar_y = 0.28

    rect = mpatches.FancyBboxPatch(
        (x - patch_w/2, bar_y), patch_w, bar_h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white', linewidth=1, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, bar_y - 0.06, label, ha='center', va='top',
            fontsize=9, color='#555555')

    if not kept:
        # X mark
        ax.text(x, bar_y + bar_h + 0.04, '✕', ha='center', va='bottom',
                fontsize=11, color='#aaaaaa')

ax.text(0.5, 0.08,
        'Variable number of patches survive\n— depends on score distribution',
        ha='center', va='bottom', fontsize=9, color='#666666', style='italic')

ax.text(0.5, 0.93, '5 kept, 3 dropped  (varies per image)',
        ha='center', va='top', fontsize=9, color='#E84393')

# ── RIGHT: Top-K (fixed budget) ───────────────────────────────────
ax = axes[1]
ax.set_facecolor(bg)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Top-K — Fixed budget', fontsize=13, fontweight='bold',
             color='#333333', pad=12)

K = 4   # always keep top 4

sorted_idx = np.argsort(scores)[::-1]
kept_set = set(sorted_idx[:K])

for i, (x, score, label) in enumerate(zip(xs, scores, labels)):
    kept = i in kept_set
    color = kept_color if kept else dropped_color
    bar_h = score * 0.55
    bar_y = 0.28

    rect = mpatches.FancyBboxPatch(
        (x - patch_w/2, bar_y), patch_w, bar_h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white', linewidth=1, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, bar_y - 0.06, label, ha='center', va='top',
            fontsize=9, color='#555555')

    if not kept:
        ax.text(x, bar_y + bar_h + 0.04, '✕', ha='center', va='bottom',
                fontsize=11, color='#aaaaaa')

# Hard cut line after K-th patch
cut_x_idx = sorted(list(kept_set))[-1]
# draw a vertical bracket after the K-th ranked patch
# rank patches by score for visual bracket
ranked_xs = [xs[i] for i in sorted_idx]
cut_after_x = ranked_xs[K-1] + patch_w/2 + 0.01
# actually just annotate
ax.annotate('', xy=(xs[K] - patch_w/2 - 0.005, 0.85),
            xytext=(xs[K] - patch_w/2 - 0.005, 0.26),
            arrowprops=dict(arrowstyle='-', color='#333333', lw=2))
ax.text(xs[K] - patch_w/2 - 0.02, 0.56, 'K', ha='right', va='center',
        fontsize=12, fontweight='bold', color='#333333')

ax.text(0.5, 0.08,
        'Exactly K patches always survive\n— constant budget, stable training',
        ha='center', va='bottom', fontsize=9, color='#666666', style='italic')

ax.text(0.5, 0.93, f'Always exactly {K} kept  (fixed)',
        ha='center', va='top', fontsize=9, color='#E84393')

plt.tight_layout(pad=2.0)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/ste_vs_topk.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
