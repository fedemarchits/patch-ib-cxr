import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Define cluster centers and colors
clusters = [
    ((1.0, 4.5), '#E84393', 'Effusion'),
    ((4.0, 5.5), '#4A90D9', 'Pneumonia'),
    ((6.5, 3.5), '#F5A623', 'Edema'),
    ((2.5, 1.5), '#7ED321', 'Atelectasis'),
    ((5.5, 1.0), '#9B59B6', 'Pneumothorax'),
]

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

for (cx, cy), color, label in clusters:
    n = np.random.randint(35, 55)
    x = np.random.normal(cx, 0.45, n)
    y = np.random.normal(cy, 0.45, n)
    ax.scatter(x, y, c=color, s=28, alpha=0.85, edgecolors='none', label=label)

# Legend outside bottom
ax.legend(fontsize=9, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, -0.04), ncol=3, handletextpad=0.3)

ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 7.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/clustering_viz.png',
            dpi=180, bbox_inches='tight')
print("Saved.")
