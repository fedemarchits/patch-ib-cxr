import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# x axis: fraction of patches removed/added (0 to 1)
x = np.linspace(0, 1, 50)

# Deletion curve: starts high, drops fast when important patches are removed
deletion = 0.72 * np.exp(-4.5 * x) + 0.05

# Insertion curve: starts low, rises fast as important patches are added
insertion = 0.70 * (1 - np.exp(-4.0 * x)) + 0.02

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(x, deletion, color='#E84393', linewidth=2.8, label='Deletion ↓')
ax.plot(x, insertion, color='#4A90D9', linewidth=2.8, label='Insertion ↑')

# Fill under curves lightly
ax.fill_between(x, deletion, alpha=0.08, color='#E84393')
ax.fill_between(x, insertion, alpha=0.08, color='#4A90D9')

# Axis labels
ax.set_xlabel('Fraction of patches', fontsize=12, color='#333333')
ax.set_ylabel('Image–Text Similarity', fontsize=12, color='#333333')

# Clean style
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(colors='#666666', labelsize=10)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('#f9f9f9')

# Legend
ax.legend(fontsize=11, frameon=False, loc='center right')

# Annotations
ax.annotate('Important patches\nremoved first',
            xy=(0.35, deletion[17]), xytext=(0.45, 0.55),
            arrowprops=dict(arrowstyle='->', color='#E84393', lw=1.5),
            fontsize=9, color='#E84393')

ax.annotate('Important patches\nadded first',
            xy=(0.25, insertion[12]), xytext=(0.38, 0.18),
            arrowprops=dict(arrowstyle='->', color='#4A90D9', lw=1.5),
            fontsize=9, color='#4A90D9')

plt.tight_layout()
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/del_ins_curve.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
