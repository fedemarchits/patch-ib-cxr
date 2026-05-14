"""
Generate cxr_patches_zoom figure WITHOUT CXR background.
Patches are drawn as grey boxes on white background.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLS, ROWS = 14, 14
N = COLS * ROWS  # 196

# Highlighted region: rows 4-7 (0-indexed 3-6), cols 5-8 (0-indexed 4-7)
HL_ROW0, HL_ROW1 = 3, 6   # inclusive, 0-indexed
HL_COL0, HL_COL1 = 4, 7   # inclusive, 0-indexed

ORANGE = "#E8541A"
PATCH_LIGHT = "#CCCCCC"   # normal patch fill (light grey)
PATCH_HL    = "#999999"   # highlighted patch fill (slightly darker)
EDGE_COLOR  = "white"

def patch_idx(r, c):
    return r * COLS + c + 1   # 1-indexed

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="white")

# Left axis: full grid
ax_grid = fig.add_axes([0.0, 0.0, 0.60, 1.0])
ax_grid.set_xlim(0, COLS)
ax_grid.set_ylim(0, ROWS)
ax_grid.set_aspect("equal")
ax_grid.axis("off")
ax_grid.set_facecolor("white")

# Draw each patch
for r in range(ROWS):
    for c in range(COLS):
        row_plot = ROWS - 1 - r   # flip so row 0 is at top
        is_hl = (HL_ROW0 <= r <= HL_ROW1) and (HL_COL0 <= c <= HL_COL1)
        fc = PATCH_HL if is_hl else PATCH_LIGHT
        rect = mpatches.Rectangle(
            (c, row_plot), 1, 1,
            linewidth=0.8, edgecolor=EDGE_COLOR, facecolor=fc
        )
        ax_grid.add_patch(rect)
        idx = patch_idx(r, c)
        ax_grid.text(
            c + 0.5, row_plot + 0.5, str(idx),
            ha="center", va="center",
            fontsize=5.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="#333333",
                      ec="none", alpha=0.75)
        )

# Orange rectangle around highlighted region (in plot coords)
hl_x = HL_COL0
hl_y = ROWS - 1 - HL_ROW1   # bottom of highlighted area in plot coords
hl_w = (HL_COL1 - HL_COL0 + 1)
hl_h = (HL_ROW1 - HL_ROW0 + 1)
rect_hl = mpatches.Rectangle(
    (hl_x, hl_y), hl_w, hl_h,
    linewidth=2.5, edgecolor=ORANGE, facecolor="none"
)
ax_grid.add_patch(rect_hl)

# ── Right axis: zoom inset ───────────────────────────────────────────────────
ZOOM_COLS = HL_COL1 - HL_COL0 + 1  # 4
ZOOM_ROWS = HL_ROW1 - HL_ROW0 + 1  # 4

ax_zoom = fig.add_axes([0.62, 0.10, 0.36, 0.78])
ax_zoom.set_xlim(0, ZOOM_COLS)
ax_zoom.set_ylim(0, ZOOM_ROWS)
ax_zoom.set_aspect("equal")
ax_zoom.axis("off")
ax_zoom.set_facecolor("white")

# Orange border for zoom panel
for spine in ["top", "bottom", "left", "right"]:
    ax_zoom.spines[spine].set_visible(True)
    ax_zoom.spines[spine].set_color(ORANGE)
    ax_zoom.spines[spine].set_linewidth(3)
ax_zoom.axis("on")
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])

# Draw zoomed patches
for zi, r in enumerate(range(HL_ROW0, HL_ROW1 + 1)):
    for zj, c in enumerate(range(HL_COL0, HL_COL1 + 1)):
        row_plot = ZOOM_ROWS - 1 - zi
        idx = patch_idx(r, c)
        rect = mpatches.Rectangle(
            (zj, row_plot), 1, 1,
            linewidth=1.5, edgecolor="white", facecolor=PATCH_HL
        )
        ax_zoom.add_patch(rect)
        ax_zoom.text(
            zj + 0.5, row_plot + 0.5, str(idx),
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#333333",
                      ec="none", alpha=0.80)
        )

# "zoom" label
ax_zoom.text(
    ZOOM_COLS / 2, ZOOM_ROWS + 0.08, "zoom",
    ha="center", va="bottom",
    fontsize=14, fontweight="bold", color=ORANGE,
    transform=ax_zoom.transData
)

# ── Dashed connector lines from grid to zoom ────────────────────────────────
# We draw them in figure coordinates
from matplotlib.patches import ConnectionPatch

# Top-right corner of orange box in grid  →  top-left of zoom panel
# Bottom-right corner  →  bottom-left of zoom panel
corners_grid = [
    (hl_x + hl_w, hl_y + hl_h),   # top-right  (data coords of ax_grid)
    (hl_x + hl_w, hl_y),           # bottom-right
]
corners_zoom = [
    (0, ZOOM_ROWS),                 # top-left   (data coords of ax_zoom)
    (0, 0),                         # bottom-left
]

for (gx, gy), (zx, zy) in zip(corners_grid, corners_zoom):
    con = ConnectionPatch(
        xyA=(gx, gy), coordsA=ax_grid.transData,
        xyB=(zx, zy), coordsB=ax_zoom.transData,
        color=ORANGE, lw=1.5, linestyle="--",
        arrowstyle="-", zorder=10
    )
    fig.add_artist(con)

# ── Save ─────────────────────────────────────────────────────────────────────
out = "/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/cxr_patches_zoom_nobg.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
