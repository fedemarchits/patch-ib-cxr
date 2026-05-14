import matplotlib.pyplot as plt
import numpy as np

rows = [
    # (label, [I2T R@1, T2I R@1, R@10 avg, NMI, Purity, Del AUC, Ins AUC], group)
    ('GlobalNCE',                                [17.93, 17.75, 56.9,  0.1410, 0.4351, None,  None ], 0),
    ('AgnosticDrop (text-agnostic MLP, layer 6)',[15.18, 14.65, 50.3,  0.1437, 0.4269, 0.458, 0.470], 1),
    ('FILIPDrop-layer6 (FILIP drop@6, k=0.5)',   [17.96, 18.17, 58.0,  0.1523, 0.4645, 0.205, 0.595], 2),
    ('FILIPDrop-layer9 (FILIP drop@9, k=0.5)',   [17.49, 17.96, 57.6,  0.1542, 0.4486, 0.183, 0.678], 2),
    ('FILIPDrop-Adaptive (layer 6, adaptive)',   [17.86, 17.28, 57.2,  0.1435, 0.4428, 0.032, 0.485], 2),
    ('HierDrop (2-stage: layer4→layer9)',         [17.38, 17.02, 57.4,  0.1460, 0.4327, 0.253, 0.622], 3),
    ('SoftGate (soft sigmoid gate, layer 6)',     [16.96, 16.96, 55.6,  0.1540, 0.4530, 0.299, 0.681], 3),
]

col_headers = ['I2T R@1', 'T2I R@1', 'R@10 avg', 'NMI', 'Purity', 'Del AUC↓', 'Ins AUC↑']
higher_is_better = [True, True, True, True, True, False, True]

n_rows = len(rows)
n_cols = len(col_headers)

def rank_col(col_idx):
    vals = [(i, rows[i][1][col_idx]) for i in range(n_rows)
            if rows[i][1][col_idx] is not None]
    hib = higher_is_better[col_idx]
    vals.sort(key=lambda x: x[1], reverse=hib)
    best   = vals[0][0] if len(vals) > 0 else None
    second = vals[1][0] if len(vals) > 1 else None
    return best, second

best_map   = {}
second_map = {}
for c in range(n_cols):
    b, s = rank_col(c)
    if b is not None: best_map[(b, c)]   = True
    if s is not None: second_map[(s, c)] = True

def fmt(val, c):
    if val is None: return '—'
    if c < 2:    return f'{val:.2f}%'
    elif c == 2: return f'{val:.1f}%'
    elif c < 5:  return f'{val:.4f}'
    else:        return f'{val:.3f}'

# ── Layout ─────────────────────────────────────────────────────────
col_w  = [3.4, 0.85, 0.85, 0.85, 0.80, 0.80, 0.88, 0.88]
row_h  = 0.30
hdr_h  = 0.38
fig_w  = sum(col_w) + 0.1
fig_h  = n_rows * row_h + hdr_h + 0.2

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

total_w = sum(col_w)
total_h = n_rows * row_h + hdr_h
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)

def hline(y, lw=0.8, color='black'):
    ax.plot([0, total_w], [y, y], color=color, lw=lw, solid_capstyle='butt')

def row_bottom(r): return total_h - hdr_h - (r + 1) * row_h
def row_mid(r):    return row_bottom(r) + row_h / 2

# ── Header ─────────────────────────────────────────────────────────
hline(total_h, lw=1.5)
hdr_mid = total_h - hdr_h / 2
x = col_w[0]
for ch, cw in zip(col_headers, col_w[1:]):
    ax.text(x + cw/2, hdr_mid, ch,
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='black')
    x += cw
hline(total_h - hdr_h, lw=0.9)

# ── Data rows ──────────────────────────────────────────────────────
group_sep_after = {0, 1, 4}

for r, (label, vals, grp) in enumerate(rows):
    ym = row_mid(r)
    ax.text(0.12, ym, label,
            ha='left', va='center', fontsize=9, color='black')

    x = col_w[0]
    for c, (val, cw) in enumerate(zip(vals, col_w[1:])):
        txt = fmt(val, c)
        fw  = 'bold' if (r, c) in best_map else 'normal'
        ax.text(x + cw/2, ym, txt,
                ha='center', va='center', fontsize=9, fontweight=fw, color='black')

        if (r, c) in second_map and val is not None:
            tx  = x + cw/2
            uly = ym - row_h * 0.34
            ax.plot([tx - cw*0.33, tx + cw*0.33], [uly, uly],
                    color='black', lw=0.7)
        x += cw

    if r in group_sep_after:
        ax.plot([0, total_w], [row_bottom(r), row_bottom(r)],
                color='black', lw=0.4)

hline(row_bottom(n_rows - 1), lw=1.5)

plt.tight_layout(pad=0.1)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/results_table_intravit.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print("Saved.")
