import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C_BLOCK   = "#D6E4F7"   # ViT block fill
C_BLOCK_E = "#1876D2"   # ViT block edge
C_SCORE   = "#FFF3CD"   # scorer box fill
C_SCORE_E = "#FBA418"   # scorer box edge
C_DROP    = "#F8D7DA"   # drop / gate box fill
C_DROP_E  = "#D10000"   # drop / gate box edge
C_TEXT    = "#E8F5E9"   # text-token box fill
C_TEXT_E  = "#0A9E4D"   # text-token box edge
C_ARROW   = "#394144"   # generic arrow
C_TARRW   = "#0A9E4D"   # text-branch arrow
C_LOSS    = "#EDE7F6"   # loss annotation fill
C_LOSS_E  = "#4D4DFF"   # loss annotation edge
BG        = "#F8F9FA"   # panel background

FONT = "DejaVu Sans"

def box(ax, x, y, w, h, label, sublabel=None,
        fc=C_BLOCK, ec=C_BLOCK_E, fontsize=9, lw=1.5, radius=0.04):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2 + (0.06 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            fontfamily=FONT, zorder=4, color="#1a1a1a")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.10, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, fontfamily=FONT, zorder=4,
                color="#555555", style="italic")

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def token_label(ax, x, y, text, color="#394144", fontsize=8):
    ax.text(x, y, text, ha="center", va="bottom",
            fontsize=fontsize, fontfamily=FONT,
            color=color, fontstyle="italic")

def loss_tag(ax, x, y, text, fc=C_LOSS, ec=C_LOSS_E):
    t = ax.text(x, y, text, ha="center", va="center",
                fontsize=7.5, fontfamily=FONT, color="#4D4DFF",
                bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=1.2),
                zorder=5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("white")
titles = [
    "(a) AgnosticDrop",
    "(b) FILIPDrop",
    "(c) HierDrop",
    "(d) SoftGate",
]

for ax, title in zip(axes.flat, titles):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.text(5, 5.25, title, ha="center", va="center",
            fontsize=12, fontweight="bold", fontfamily=FONT, color="#1a1a1a")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (a) AGNOSTICDROP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = axes[0, 0]

# image input
box(ax, 0.1, 2.0, 0.9, 1.0, "CXR\nImage", fc="#FAFAFA", ec="#888888", fontsize=8)
arrow(ax, 1.0, 2.5, 1.4, 2.5)

# blocks 1-6
box(ax, 1.4, 2.0, 1.8, 1.0, "Blocks 1–6", sublabel="197 tokens", fc=C_BLOCK, ec=C_BLOCK_E)
arrow(ax, 3.2, 2.5, 3.6, 2.5)

# scorer
box(ax, 3.6, 1.8, 1.5, 1.4, "PatchScorer\nMLP", sublabel="768→64→1\n(visual only)", fc=C_SCORE, ec=C_SCORE_E, fontsize=8)

# scores → drop
arrow(ax, 5.1, 2.5, 5.5, 2.5)
box(ax, 5.5, 2.0, 1.3, 1.0, "Top-K Drop\nkeep 98 ✂", fc=C_DROP, ec=C_DROP_E, fontsize=8)

# No text branch indicator
ax.text(4.35, 1.4, "✗  no text", ha="center", va="center",
        fontsize=8, color="#D10000", fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFF0F0", ec="#D10000", lw=1))

# blocks 7-12
arrow(ax, 6.8, 2.5, 7.2, 2.5)
box(ax, 7.2, 2.0, 1.8, 1.0, "Blocks 7–12", sublabel="99 tokens", fc=C_BLOCK, ec=C_BLOCK_E)

# CLS output
arrow(ax, 9.0, 2.5, 9.4, 2.5)
ax.text(9.5, 2.5, "CLS", ha="left", va="center",
        fontsize=8, fontweight="bold", fontfamily=FONT, color="#1876D2")

# token count labels
token_label(ax, 2.3, 3.05, "197 tokens", color="#1876D2")
token_label(ax, 8.1, 3.05, "99 tokens", color="#D10000")

# loss annotation
loss_tag(ax, 4.35, 4.6, "InfoNCE loss on CLS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (b) FILIPDROP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = axes[0, 1]

# image input
box(ax, 0.1, 2.8, 0.9, 1.0, "CXR\nImage", fc="#FAFAFA", ec="#888888", fontsize=8)
arrow(ax, 1.0, 3.3, 1.4, 3.3)

# text input
box(ax, 0.1, 0.8, 0.9, 1.0, "Text\nTokens", fc=C_TEXT, ec=C_TEXT_E, fontsize=8)

# blocks 1-L
box(ax, 1.4, 2.8, 1.6, 1.0, "Blocks 1–L", sublabel="197 tokens", fc=C_BLOCK, ec=C_BLOCK_E)
arrow(ax, 3.0, 3.3, 3.4, 3.3)

# FILIP scorer
box(ax, 3.4, 2.6, 1.7, 1.4, "FILIP Scorer", sublabel="sᵢ = max_j cos(Wₚxᵢ,Wₜtⱼ)", fc=C_SCORE, ec=C_SCORE_E, fontsize=8)

# text branch arrow
arrow(ax, 1.0, 1.3, 4.25, 2.6, color=C_TARRW, lw=1.5)
ax.text(2.3, 2.1, "Wₜ probe\n(768→512)", ha="center", va="center",
        fontsize=7, color=C_TEXT_E, fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_TEXT, ec=C_TEXT_E, lw=1))

# probe loss
loss_tag(ax, 4.25, 4.6, "Probe FILIP loss")

# drop
arrow(ax, 5.1, 3.3, 5.5, 3.3)
box(ax, 5.5, 2.8, 1.3, 1.0, "Top-K Drop\nkeep K ✂", fc=C_DROP, ec=C_DROP_E, fontsize=8)

# blocks L+1-12
arrow(ax, 6.8, 3.3, 7.2, 3.3)
box(ax, 7.2, 2.8, 1.8, 1.0, "Blocks L+1–12", sublabel="K+1 tokens", fc=C_BLOCK, ec=C_BLOCK_E, fontsize=8)

# final loss
loss_tag(ax, 8.1, 4.6, "Final FILIP loss")

# CLS output
arrow(ax, 9.0, 3.3, 9.4, 3.3)
ax.text(9.5, 3.3, "CLS", ha="left", va="center",
        fontsize=8, fontweight="bold", fontfamily=FONT, color="#1876D2")

# InfoNCE
loss_tag(ax, 5.0, 0.5, "InfoNCE loss on CLS")

# token count labels
token_label(ax, 2.2, 3.85, "197 tokens", color="#1876D2")
token_label(ax, 8.1, 3.85, "K+1 tokens", color="#D10000")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (c) HIERDROP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = axes[1, 0]

# image input
box(ax, 0.1, 2.0, 0.9, 1.0, "CXR\nImage", fc="#FAFAFA", ec="#888888", fontsize=8)
arrow(ax, 1.0, 2.5, 1.3, 2.5)

# text
box(ax, 0.1, 0.3, 0.9, 0.9, "Text\nTokens", fc=C_TEXT, ec=C_TEXT_E, fontsize=7.5)

# blocks 1-4
box(ax, 1.3, 2.0, 1.3, 1.0, "Blocks\n1–4", sublabel="197 tok.", fc=C_BLOCK, ec=C_BLOCK_E, fontsize=8)

# stage 1 scorer
arrow(ax, 2.6, 2.5, 3.0, 2.5)
box(ax, 3.0, 1.7, 1.3, 1.6, "FILIP\nStage 1", sublabel="layer 4\nkeep 147\n(75%) ✂", fc=C_SCORE, ec=C_SCORE_E, fontsize=7.5)
# text arrow to stage 1
arrow(ax, 1.0, 0.75, 3.65, 1.7, color=C_TARRW, lw=1.4)
loss_tag(ax, 3.65, 4.7, "Probe loss 1")

# blocks 5-9
arrow(ax, 4.3, 2.5, 4.7, 2.5)
box(ax, 4.7, 2.0, 1.3, 1.0, "Blocks\n5–9", sublabel="147 tok.", fc=C_BLOCK, ec=C_BLOCK_E, fontsize=8)

# stage 2 scorer
arrow(ax, 6.0, 2.5, 6.4, 2.5)
box(ax, 6.4, 1.7, 1.3, 1.6, "FILIP\nStage 2", sublabel="layer 9\nkeep 98\n(50%) ✂", fc=C_SCORE, ec=C_SCORE_E, fontsize=7.5)
# text arrow to stage 2
arrow(ax, 1.0, 0.75, 7.05, 1.7, color=C_TARRW, lw=1.4)
loss_tag(ax, 7.05, 4.7, "Probe loss 2")

# blocks 10-12
arrow(ax, 7.7, 2.5, 8.1, 2.5)
box(ax, 8.1, 2.0, 1.3, 1.0, "Blocks\n10–12", sublabel="98 tok.", fc=C_BLOCK, ec=C_BLOCK_E, fontsize=8)

# CLS
arrow(ax, 9.4, 2.5, 9.7, 2.5)
ax.text(9.75, 2.5, "CLS", ha="left", va="center",
        fontsize=7.5, fontweight="bold", fontfamily=FONT, color="#1876D2")

# final loss
loss_tag(ax, 8.75, 4.7, "Final FILIP\n+ InfoNCE")

# irreversibility note
ax.text(5.0, 0.35, "⚠  patches dropped at Stage 1 cannot be recovered at Stage 2",
        ha="center", va="center", fontsize=7.5, color="#D10000",
        fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF0F0", ec="#D10000", lw=1))

# token labels
token_label(ax, 1.95, 3.05, "197", color="#1876D2")
token_label(ax, 5.35, 3.05, "147", color="#FBA418")
token_label(ax, 8.75, 3.05, "98", color="#D10000")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (d) SOFTGATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = axes[1, 1]

# image input
box(ax, 0.1, 2.8, 0.9, 1.0, "CXR\nImage", fc="#FAFAFA", ec="#888888", fontsize=8)
arrow(ax, 1.0, 3.3, 1.4, 3.3)

# text
box(ax, 0.1, 0.8, 0.9, 1.0, "Text\nTokens", fc=C_TEXT, ec=C_TEXT_E, fontsize=8)

# blocks 1-6
box(ax, 1.4, 2.8, 1.6, 1.0, "Blocks 1–6", sublabel="197 tokens", fc=C_BLOCK, ec=C_BLOCK_E)
arrow(ax, 3.0, 3.3, 3.4, 3.3)

# FILIP scorer
box(ax, 3.4, 2.6, 1.7, 1.4, "FILIP Scorer", sublabel="sᵢ = max_j cos(Wₚxᵢ,Wₜtⱼ)", fc=C_SCORE, ec=C_SCORE_E, fontsize=8)

# text arrow
arrow(ax, 1.0, 1.3, 4.25, 2.6, color=C_TARRW, lw=1.5)
ax.text(2.3, 2.1, "Wₜ probe\n(768→512)", ha="center", va="center",
        fontsize=7, color=C_TEXT_E, fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_TEXT, ec=C_TEXT_E, lw=1))

# sigmoid gate box  (NOT drop)
arrow(ax, 5.1, 3.3, 5.5, 3.3)
box(ax, 5.5, 2.8, 1.5, 1.0, "Sigmoid Gate\ngᵢ = σ(5·sᵢ)", sublabel="x̃ᵢ = xᵢ × gᵢ", fc="#E3F2FD", ec="#1876D2", fontsize=7.5)

# NO drop label
ax.text(6.25, 4.05, "no drop  ✓\n197 tokens kept", ha="center", va="center",
        fontsize=7.5, color="#0A9E4D", fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.2", fc="#E8F5E9", ec="#0A9E4D", lw=1.2))

# blocks 7-12
arrow(ax, 7.0, 3.3, 7.4, 3.3)
box(ax, 7.4, 2.8, 1.8, 1.0, "Blocks 7–12", sublabel="197 weighted\ntokens", fc=C_BLOCK, ec=C_BLOCK_E, fontsize=8)

# CLS output
arrow(ax, 9.2, 3.3, 9.5, 3.3)
ax.text(9.55, 3.3, "CLS", ha="left", va="center",
        fontsize=8, fontweight="bold", fontfamily=FONT, color="#1876D2")

# losses
loss_tag(ax, 4.25, 4.7, "Probe FILIP loss")
loss_tag(ax, 8.3, 4.7, "InfoNCE + gated\nFILIP loss")

# token count labels
token_label(ax, 2.2, 3.85, "197 tokens", color="#1876D2")
token_label(ax, 8.3, 3.85, "197 tokens", color="#0A9E4D")

# gradient note
ax.text(5.0, 0.5, "∇ flows continuously through σ  (no STE needed)",
        ha="center", va="center", fontsize=7.5, color="#4D4DFF",
        fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.25", fc=C_LOSS, ec=C_LOSS_E, lw=1))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.tight_layout(pad=1.5)

out = "/Users/federicomarchi/Desktop/Thesis/patch-ib-cxr/thesisPdf/images/intravit_variants.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
