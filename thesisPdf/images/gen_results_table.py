import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── All models & data ──────────────────────────────────────────────
# cols: I2T R@1, T2I R@1, R@10 avg, NMI, Purity, Del AUC, Ins AUC
# None = n/a
rows = [
    # (model_label, [values], group)
    # cols: I2T R@1, T2I R@1, R@10 avg, NMI, Purity, Del AUC, Ins AUC
    ('GlobalNCE',                              [17.93, 17.75, 56.9,  0.1410, 0.4351, None,  None ], 0),
    ('LocalAlign-5 (FILIP mid-fusion)',        [18.28, 19.12, 58.6,  0.1495, 0.4501, 0.368, 0.463], 1),
    ('MaskTopK-1 (k=0.4, post-ViT)',           [18.33, 18.33, 58.5,  0.1184, 0.4342, 0.424, 0.423], 1),
    ('AgnosticDrop (text-agnostic, layer 6)',  [15.18, 14.65, 50.3,  0.1437, 0.4269, 0.458, 0.470], 2),
    ('FILIPDrop-layer6 (FILIP drop@6, k=0.5)',[17.96, 18.17, 58.0,  0.1523, 0.4645, 0.205, 0.595], 2),
    ('FILIPDrop-layer9 (FILIP drop@9, k=0.5)',[17.49, 17.96, 57.6,  0.1542, 0.4486, 0.183, 0.678], 2),
    ('SoftGate (soft sigmoid gate, layer 6)',  [16.96, 16.96, 55.6,  0.1540, 0.4530, 0.299, 0.681], 2),
    ('FILIPDrop-Adaptive (layer 6, adaptive)', [17.86, 17.28, 57.2,  0.1435, 0.4428, 0.032, 0.485], 2),
]

col_headers = ['I2T R@1', 'T2I R@1', 'R@10 avg', 'NMI', 'Purity', 'Del AUC↓', 'Ins AUC↑']
higher_is_better = [True, True, True, True, True, False, True]

n_rows = len(rows)
n_cols = len(col_headers)

# ── Find best and 2nd-best per column (excluding None) ─────────────
def rank_col(col_idx):
    vals = [(i, rows[i][1][col_idx]) for i in range(n_rows)
            if rows[i][1][col_idx] is not None]
    hib = higher_is_better[col_idx]
    vals.sort(key=lambda x: x[1], reverse=hib)
    best = vals[0][0] if vals else None
    second = vals[1][0] if len(vals) > 1 else None
    return best, second

best_map = {}    # (row, col) -> 'bold'
second_map = {}  # (row, col) -> 'underline'
for c in range(n_cols):
    b, s = rank_col(c)
    if b is not None: best_map[(b, c)] = True
    if s is not None: second_map[(s, c)] = True

# ── Format cell ────────────────────────────────────────────────────
def fmt(val, c):
    if val is None:
        return '—'
    if c == 0:   # I2T R@1 %
        return f'{val:.2f}%'
    elif c == 1:  # T2I R@1 %
        return f'{val:.2f}%'
    elif c == 2:  # R@10 avg %
        return f'{val:.1f}%'
    elif c <= 4:  # NMI, Purity
        return f'{val:.4f}'
    else:         # Del/Ins AUC
        return f'{val:.3f}'

# ── Figure setup ───────────────────────────────────────────────────
fs = 7.2          # base fontsize
row_h = 0.22      # inches per row
hdr_h = 0.32
col_w = [3.0, 0.82, 0.82, 0.82, 0.78, 0.78, 0.82, 0.82]  # model + 7 data cols
fig_w = sum(col_w) + 0.1
fig_h = n_rows * row_h + hdr_h + 0.25

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

total_w = sum(col_w)
total_h = n_rows * row_h + hdr_h

ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)

def hline(y, lw=0.7, ls='-', color='black'):
    ax.plot([0, total_w], [y, y], color=color, lw=lw, linestyle=ls,
            solid_capstyle='butt', transform=ax.transData)

def row_bottom(r):
    """y of bottom edge of row r (0=first data row from top)."""
    return total_h - hdr_h - (r + 1) * row_h

def row_mid(r):
    return row_bottom(r) + row_h / 2

# ── Header ─────────────────────────────────────────────────────────
hline(total_h, lw=1.5)
hdr_mid = total_h - hdr_h / 2
x = col_w[0]
for c, (ch, cw) in enumerate(zip(col_headers, col_w[1:])):
    ax.text(x + cw/2, hdr_mid, ch,
            ha='center', va='center', fontsize=fs, fontweight='bold', color='black')
    x += cw
hline(total_h - hdr_h, lw=0.9)

# ── Data rows ──────────────────────────────────────────────────────
group_sep_after = {0, 2, 3}   # draw thin rule AFTER these row indices

for r, (label, vals, grp) in enumerate(rows):
    ym = row_mid(r)

    # model label
    ax.text(0.08, ym, label,
            ha='left', va='center', fontsize=fs, color='black')

    x = col_w[0]
    for c, (val, cw) in enumerate(zip(vals, col_w[1:])):
        txt = fmt(val, c)
        is_best   = (r, c) in best_map
        is_second = (r, c) in second_map

        fw = 'bold' if is_best else 'normal'
        txt_obj = ax.text(x + cw/2, ym, txt,
                          ha='center', va='center',
                          fontsize=fs, fontweight=fw, color='black')

        # underline for second-best: draw a short line below the text
        if is_second and val is not None:
            txt_x = x + cw/2
            ul_y = ym - row_h * 0.34
            ul_half = cw * 0.33
            ax.plot([txt_x - ul_half, txt_x + ul_half], [ul_y, ul_y],
                    color='black', lw=0.7, transform=ax.transData)

        x += cw

    # group separator
    if r in group_sep_after:
        yline = row_bottom(r)
        ax.plot([0, total_w], [yline, yline],
                color='black', lw=0.4, linestyle='-', transform=ax.transData)

# ── Bottom rule ────────────────────────────────────────────────────
hline(row_bottom(n_rows - 1), lw=1.5)

plt.tight_layout(pad=0.1)
plt.savefig('/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/results_table.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print("Saved.")
