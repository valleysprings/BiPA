"""Draft of Figure 2 (fig:overview): one round of BACH on a skill S.

Panels follow Section 4 of the rewritten methodology:
  (a) measure S against the seed cache on the fit set D (Sec. 4.2-4.3),
  (b) plan on the decomposition, zero extra runs (Sec. 4.4),
  (c) one certified round on fresh tasks T (Sec. 4.5).
Palette, fonts, box and arrow shapes copy render_taming_teaser.py /
render_bach_method.py; blue marks BACH's steps, as in Algorithm 1.
The canvas is in points (1 data unit = 1 pt at 7.0 x 2.8 in). All numbers
are illustrative and chosen to be mutually consistent (see the values
printed at the end).
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="bach-zo-mpl-"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "svg.hashsalt": "bach-method-v2",
        "pdf.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Rectangle)


STEM = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/paper/assets/old/bach_method_search")
FIG_W_IN, FIG_H_IN = 7.0, 2.8
WIDTH, HEIGHT = FIG_W_IN * 72, FIG_H_IN * 72  # 504 x 201.6 pt
MIN_FONT = 6.5

# Palette of Figures 1 and 2 (dataviz-validated hues), plus tints for blocks.
INK = "#253443"
MUTED = "#5E6B77"
LINE = "#CFD6DC"
PANEL = "#F7F9FA"
GATE = "#EAEFF3"
WHITE = "#FFFFFF"
BLUE = "#2A78D6"        # BACH's steps (the blue of Algorithm 1)
BLUE_TXT = "#1D5FAF"    # same hue, darker for small text
BLUE_TINT = "#E4EEFA"
SKILL = "#E0662F"       # skill text
TOOL = "#169A6C"        # tools and their outputs
CTX_BAR = "#B9C0C6"     # history / other context
TEXT_FILL = "#F9DDCF"   # seed block
TEAL_FILL = "#D5EFE5"   # block added by the optimizer
CHECK, CROSS = "✓", "✗"

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
ax.axis("off")
fig.patch.set_facecolor(WHITE)
labels = []
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def text(x, y, label, *, size=7.0, color=INK, weight="normal", style="normal",
         ha="center", va="center", zorder=8, within=None, rotation=0):
    artist = ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                     fontstyle=style, ha=ha, va=va, zorder=zorder,
                     rotation=rotation)
    labels.append((artist, within))
    return artist


def text_width(label, size, weight="normal", style="normal"):
    probe = ax.text(0, 0, label, fontsize=size, fontweight=weight,
                    fontstyle=style)
    bb = probe.get_window_extent(renderer)
    probe.remove()
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x1 - x0


def rbox(x, y, w, h, *, fill=WHITE, edge=LINE, lw=.6, radius=2.0, zorder=3,
         ls="-"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fill, edgecolor=edge, linewidth=lw,
                           linestyle=ls, zorder=zorder)
    ax.add_patch(patch)
    return patch


def arrow(p0, p1, *, color=INK, lw=.6, scale=4.0, zorder=7, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 mutation_scale=scale, linewidth=lw,
                                 color=color, shrinkA=0, shrinkB=0,
                                 zorder=zorder))


def polyline_arrow(pts, *, color=INK, lw=.7):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=7,
                solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, lw=lw, scale=4.5)


def hexagon(x, y, w, h, *, fill=GATE, edge=INK, lw=.7, tip=4.0, zorder=3):
    ax.add_patch(Polygon([(x, y + h / 2), (x + tip, y), (x + w - tip, y),
                          (x + w, y + h / 2), (x + w - tip, y + h),
                          (x + tip, y + h)], closed=True, facecolor=fill,
                         edgecolor=edge, linewidth=lw, joinstyle="round",
                         zorder=zorder))


def cylinder(x, y, w, h, eh=4.5, fill=GATE, edge=INK, lw=.7):
    ax.add_patch(Rectangle((x, y + eh / 2), w, h - eh, facecolor=fill,
                           edgecolor="none", zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + eh / 2), w, eh, facecolor=fill,
                         edgecolor=edge, linewidth=lw, zorder=3))
    for xx in (x, x + w):
        ax.plot([xx, xx], [y + eh / 2, y + h - eh / 2], color=edge, lw=lw,
                zorder=4)
    ax.add_patch(Ellipse((x + w / 2, y + h - eh / 2), w, eh, facecolor=WHITE,
                         edgecolor=edge, linewidth=lw, zorder=5))
    ax.plot([x + 2, x + w - 2], [y + h / 2 - .5] * 2, color=MUTED, lw=.5,
            zorder=5)


def dashed_v(x, y0, y1, *, color=INK, lw=.8, zorder=6):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, dashes=(2.2, 1.6),
            zorder=zorder)


# ---------------------------------------------------------------------------
# Illustrative numbers, kept consistent with Section 4.
# ---------------------------------------------------------------------------
KAPPA = math.log(1.1)                     # budget, eps = 0.1
ELL = [-.22, -.06, .05, .12, .18, .24, .29, .35, .42, .55, .83]  # l_i on D
LHAT = sum(ELL) / len(ELL)                # 0.25
DELTA = [("skill", SKILL, .02), ("KB search", TOOL, .12), ("read", TOOL, .01),
         ("shell", TOOL, .04), ("history", CTX_BAR, .06)]   # sums to L-hat
GAMMA = LHAT - KAPPA                      # 0.155
# blocks of S: (label, added?, tokens, lowered text price, lowered behavior price)
BLOCKS = [("goal", False, 60, .004, 0.0),
          ("tool use", False, 90, .005, 0.0),
          ("search KB", True, 110, .007, .097),
          ("re-check", True, 45, .003, .052),
          ("confirm all", True, 160, .010, .038),
          ("reply style", False, 40, .002, 0.0)]
PLAN = {"search KB", "re-check"}          # J*: fewest tokens covering gamma
CAP = sum(t for _, a, t, *_ in BLOCKS if a) / 2      # capacity c
# scoped call constraints priced by replay: (label, s, se); s- = 0.5 s - se
LEVERS = [("KB search ≤ 1/task", .74, .09),
          ("KB search, no repeat", .48, .07),
          ("shell output ≤ 500", .30, .05)]
PHI0, S_MIN = .5, .24
# certified round: clipped paired log ratios on fresh tasks, bound at checkpoints
B = math.log(3)
ZETA = [-.62, -.15, -.21, -.48, -.33, -.05, -.71, -.26, -.40, .12, -.55, -.19,
        -.09, -.44, -B, -.31, .18, -.37, -.52, -.08, -.29, -.66, .02, -.35]
U_T = [(6, .58), (12, .17), (18, .01), (24, -.08)]
EPS_MIN, ALPHA, M_Q, DQ = .03, .03, .10, -.03

# ---------------------------------------------------------------------------
# Panels.
# ---------------------------------------------------------------------------
PANEL_TOP, PANEL_BOTTOM = HEIGHT - 2.0, 17.0
GAP = 6.0
PW = [154.0, 210.0, 124.0]
PX = [2.0, 2.0 + PW[0] + GAP, 2.0 + PW[0] + PW[1] + 2 * GAP]
GAPX = [PX[0] + PW[0] + GAP / 2, PX[1] + PW[1] + GAP / 2]
Y_TITLE, Y_SUB = PANEL_TOP - 10.0, PANEL_TOP - 20.5
TITLES = ["(a) Measure against the seed", "(b) Price and propose",
          "(c) Screen and keep"]
SUBS = [r"recorded episodes on a task batch $\mathcal{I}$",
        "block prices, credits, four moves",
        r"fresh batch $\mathcal{B}_t$, $\mathcal{V}$, archive"]
panels = []
for px, pw, title, sub in zip(PX, PW, TITLES, SUBS):
    box = (px, PANEL_BOTTOM, px + pw, PANEL_TOP)
    panels.append(box)
    rbox(px, PANEL_BOTTOM, pw, PANEL_TOP - PANEL_BOTTOM, fill=PANEL,
         edge=INK, lw=.8, radius=4, zorder=0)
    text(px + 7, Y_TITLE, title, size=8.5, weight="bold", ha="left",
         within=box)
    text(px + 7, Y_SUB, sub, size=7, style="italic", color=MUTED, ha="left",
         within=box)

# ===========================================================================
# (a) Measure against the seed
# ===========================================================================
px, pw, box = PX[0], PW[0], panels[0]
L0 = px + 7
# inputs: recorded episodes of S <-> seed cache, on the same tasks
Y_IN = 163.5
chip_lab = r"$S$ on $\mathcal{I}$"
cw = text_width(chip_lab, 7) + 7
rbox(L0, Y_IN - 6.5, cw, 13, fill=WHITE, edge=INK, lw=.8, radius=2.5,
     zorder=4)
text(L0 + cw / 2, Y_IN - .2, chip_lab, within=box)
CYW, CYH = 13.0, 14.0
CYX = L0 + cw + 40.0
cylinder(CYX, Y_IN - CYH / 2, CYW, CYH, eh=4.0)
text(CYX + CYW + 3.5, Y_IN, r"seed cache $c^0$", ha="left", within=box)
ax.add_patch(FancyArrowPatch((L0 + cw + 2.0, Y_IN - 2.0), (CYX - 2.0, Y_IN - 2.0),
                             arrowstyle="<|-|>", mutation_scale=3.8, lw=.7,
                             color=BLUE, shrinkA=0, shrinkB=0, zorder=5))
text((L0 + cw + CYX) / 2, Y_IN + 3.6, "same tasks", size=6.5,
     color=BLUE_TXT, style="italic", within=box)

# per-task log ratio strip
Y_LAB, Y_DOT, Y_AX = 151.0, 143.0, 137.0
text(L0, Y_LAB, r"per task: $\ell_i=\log\,c_i(S)/c_i^0$", ha="left",
     within=box)
DX0, DX1, DV0, DV1 = L0 + 4, px + pw - 9, -.3, .9
xd = lambda v: DX0 + (v - DV0) * (DX1 - DX0) / (DV1 - DV0)
ax.plot([DX0, DX1], [Y_AX, Y_AX], color=MUTED, lw=.6, zorder=2)
ax.plot([xd(0)] * 2, [Y_AX, Y_DOT + 5], color=LINE, lw=.6, zorder=1)
for k, v in enumerate(ELL):
    ax.add_patch(Circle((xd(v), Y_DOT + (1.6 if k % 2 else -1.6)), 1.8,
                        facecolor=MUTED, edgecolor=WHITE, linewidth=.4,
                        zorder=5))
ax.add_patch(Polygon([(xd(LHAT) - 3.2, Y_AX), (xd(LHAT), Y_AX + 3.2),
                      (xd(LHAT) + 3.2, Y_AX), (xd(LHAT), Y_AX - 3.2)],
                     closed=True, facecolor=BLUE, edgecolor=WHITE,
                     linewidth=.4, zorder=6))
text(xd(0), Y_AX - 7.0, "0", size=6.5, color=MUTED, within=box)
text(xd(LHAT) + 4.0, Y_AX - 7.0, rf"mean $\widehat L={LHAT:.2f}$", size=7,
     color=BLUE_TXT, ha="left", within=box)
ax.plot([xd(LHAT), xd(LHAT) + 3.0], [Y_AX - 3.4, Y_AX - 6.0], color=BLUE,
        lw=.5, zorder=5)

# split over components (exact on each task): bars from zero, then the total
Y_HEAD = 118.0
text(L0, Y_HEAD, r"split over components $g$: $\delta_g$", ha="left",
     within=box)
ax.plot([L0, px + pw - 7], [Y_HEAD - 5.5, Y_HEAD - 5.5], color=LINE, lw=.5,
        zorder=1)
BZ, BSC = L0 + 43.0, 290.0
PITCH = 7.5
for k, (name, col, v) in enumerate(DELTA):
    y = Y_HEAD - 13.0 - PITCH * k
    text(L0, y, name, size=7, color=MUTED, ha="left", within=box)
    ax.add_patch(Rectangle((BZ, y - 2.8), v * BSC, 5.6, facecolor=col,
                           edgecolor="none", zorder=4))
    text(BZ + v * BSC + 2.5, y, f"+{v:.2f}", size=6.5, ha="left",
         within=box)
Y_TOT = Y_HEAD - 13.0 - PITCH * len(DELTA) - 3.0
ax.plot([BZ, BZ], [Y_HEAD - 8.5, Y_TOT - 4.5], color=INK, lw=.6, zorder=5)
ax.plot([L0, BZ + LHAT * BSC + 22], [Y_TOT + 5.6] * 2, color=LINE, lw=.5,
        zorder=1)
text(L0, Y_TOT, r"total $\widehat L$", ha="left", within=box)
x = BZ
for k, (name, col, v) in enumerate(DELTA):
    ax.add_patch(Rectangle((x, Y_TOT - 3.5), v * BSC, 7.0, facecolor=col,
                           edgecolor=WHITE, linewidth=.5, zorder=4))
    x += v * BSC
text(x + 2.5, Y_TOT, f"{LHAT:.2f}", size=7, ha="left", within=box)
XK = BZ + KAPPA * BSC
dashed_v(XK, Y_TOT - 8.0, Y_TOT + 7.0)
text(XK - 2.5, Y_TOT - 7.5, r"$\kappa$", ha="right", within=box)
# the gap gamma, from kappa to L-hat
YG = Y_TOT - 9.0
ax.plot([XK, x], [YG, YG], color=BLUE, lw=.9, zorder=6)
for xx in (XK, x):
    ax.plot([xx, xx], [YG - 2.0, YG + 2.0], color=BLUE, lw=.9, zorder=6)
text((XK + x) / 2, YG - 9.5, rf"gap $\gamma=\widehat L-\kappa={GAMMA:.3f}$",
     color=BLUE_TXT, within=box)
Y_NOTE1, Y_NOTE2 = 34.5, 24.2

text(L0, Y_NOTE2 + 3.0, r"$\gamma$ prices the deletion move (d)", size=7,
     color=MUTED, ha="left", within=box)

# ===========================================================================
# (b) Price and propose
# ===========================================================================
px, pw, box = PX[1], PW[1], panels[1]
L1 = px + 7
R1 = px + pw - 7.0
CHIP_W, BH, BP = 41.0, 8.0, 9.0
BAR0 = L1 + CHIP_W + 4.0
PSC = 430.0                                  # pt per unit of lowered price
CR0 = BAR0 + 62.0                            # credit column
CSC = 900.0
CRED = {"goal": .05, "tool use": .04, "search KB": .002, "re-check": .012,
        "confirm all": .03, "reply style": .02}
Y_BH = 168.0
text(L1, Y_BH, r"block of $S$", ha="left", within=box)
text(BAR0, Y_BH, r"price $p_j^-$", ha="left", within=box)
text(CR0, Y_BH, r"credit $q_j$", ha="left", within=box)
ROWS_TOP = Y_BH - 10.0
row_y = [ROWS_TOP - BP * k for k in range(len(BLOCKS))]
ratio = {lab: (pt + pb) / CRED[lab] for lab, _, _, pt, pb in BLOCKS}
worst = max(ratio, key=ratio.get)
for (lab, added, tok, pt, pb), y in zip(BLOCKS, row_y):
    removed = lab in PLAN
    ax.add_patch(FancyBboxPatch((L1, y - BH / 2), CHIP_W, BH,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=TEAL_FILL if added else TEXT_FILL,
                                edgecolor=TOOL if added else SKILL,
                                linewidth=.8, linestyle="--" if removed else "-",
                                zorder=4))
    text(L1 + CHIP_W / 2, y - .2, lab, size=7, within=box)
    ax.add_patch(Rectangle((BAR0, y - 2.4), pt * PSC, 4.8, facecolor=SKILL,
                           edgecolor="none", zorder=4))
    if pb:
        ax.add_patch(Rectangle((BAR0 + pt * PSC, y - 2.4), pb * PSC, 4.8,
                               facecolor=TOOL, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((CR0, y - 2.4), CRED[lab] * CSC, 4.8,
                           facecolor=BLUE, edgecolor="none", zorder=4))
    if lab == worst:
        text(CR0 + 12.0, y - .2, r"$\leftarrow$ target of (a)", size=6.5,
             color=BLUE_TXT, ha="left", within=box)
ax.plot([L1, R1], [row_y[-1] - 6.0] * 2, color=LINE, lw=.5, zorder=1)
Y_LEDG_BOT = row_y[-1] - 6.0

# the four moves
MW, MH = 94.0, 24.0
MX = [L1, L1 + MW + 8.0]
MY = [Y_LEDG_BOT - 8.0 - MH, Y_LEDG_BOT - 8.0 - 2 * MH - 7.0]
MOVES = [("c", "analyst step", "SkillOpt analyst step", MX[0], MY[0]),
         ("a", "rewrite one block", "highest price per credit", MX[1], MY[0]),
         ("d", "delete (knapsack)", r"cover $\gamma$, fewest tokens", MX[0], MY[1]),
         ("b", "recombine", "knapsack of old blocks", MX[1], MY[1])]
for tag, name, sub_, mx, my in MOVES:
    rbox(mx, my, MW, MH, fill=WHITE, edge=BLUE, lw=.9, radius=3, zorder=3)
    ax.add_patch(Circle((mx + 8.0, my + MH / 2), 5.0, facecolor=BLUE,
                        edgecolor="none", zorder=4))
    text(mx + 8.0, my + MH / 2 - .2, tag, size=7, color=WHITE, weight="bold",
         within=box)
    text(mx + 16.0, my + MH / 2 + 4.8, name, size=7, weight="bold",
         ha="left", within=box)
    text(mx + 16.0, my + MH / 2 - 4.8, sub_, size=6.5, color=MUTED,
         ha="left", within=box)
arrow(((L1 + R1) / 2, Y_LEDG_BOT + .5), ((L1 + R1) / 2, MY[0] + MH + .6),
      color=INK, lw=.8, scale=4.5)
# child
CHW, CHH = 112.0, 15.0
CHX = (L1 + R1) / 2 - CHW / 2
CHY = PANEL_BOTTOM + 5.0
rbox(CHX, CHY, CHW, CHH, fill=BLUE_TINT, edge=BLUE, lw=.9, radius=3, zorder=3)
text(CHX + CHW / 2, CHY + CHH / 2 - .2, r"child $C$ of incumbent $P$",
     size=7, weight="bold", color=BLUE_TXT, within=box)
for mx in MX:
    arrow((mx + MW / 2, MY[1] - .5 if mx == MX[0] or True else 0),
          (mx + MW / 2, CHY + CHH + .6), color=INK, lw=.8, scale=4.5)
for mx in MX:   # row-1 moves feed the child through the gap between rows
    pass
CH_MID = CHY + CHH / 2

# ===========================================================================
# (c) Screen and keep
# ===========================================================================
px, pw, box = PX[2], PW[2], panels[2]
L2 = px + 7
R2 = px + pw - 7
SY0, SY1 = 134.0, 169.0
rbox(L2, SY0, R2 - L2, SY1 - SY0, fill=WHITE, edge=INK, lw=.7, radius=3,
     zorder=3)
text(L2 + 4, SY1 - 6.5, r"$C$ vs parent $P$ on $B_t$", size=7, weight="bold",
     ha="left", within=box)
text(L2 + 4, SY1 - 16.5, r"success step: $\Delta Q>0$", size=6.5, ha="left",
     within=box)
text(L2 + 4, SY1 - 27.0, r"workload step: $\Delta Q\geq-0.10$", size=6.5,
     ha="left", within=box)
EV_Y = 122.0
rbox(L2 + 18.0, EV_Y - 6.0, R2 - L2 - 36.0, 12.0, fill=GATE, edge=INK, lw=.6,
     radius=2.5, zorder=3)
text((L2 + R2) / 2, EV_Y - .2, r"evaluate on $V$", size=7, within=box)
arrow(((L2 + R2) / 2, SY0 - .5), ((L2 + R2) / 2, EV_Y + 6.6), color=INK,
      lw=.8, scale=4.5)
text((L2 + R2) / 2 + 3.0, (SY0 + EV_Y + 6) / 2, "pass", size=6.5,
     color=BLUE_TXT, ha="left", within=box)
# (Q, L) plane
QX0, QX1 = L2 + 11.0, R2 - 1.0
QY0, QY1 = 54.0, 106.0
LV0, LV1, QV0, QV1 = -.02, .50, 38.0, 54.0
xp = lambda v: QX0 + (v - LV0) * (QX1 - QX0) / (LV1 - LV0)
yp = lambda v: QY0 + (v - QV0) * (QY1 - QY0) / (QV1 - QV0)
HMAX = math.log(1.5)
rbox(QX0, QY0, QX1 - QX0, QY1 - QY0, fill=WHITE, edge=LINE, lw=.5, radius=1,
     zorder=1)
ax.add_patch(Rectangle((xp(HMAX), QY0), QX1 - xp(HMAX), QY1 - QY0,
                       facecolor=GATE, edgecolor="none", zorder=1.5))
arrow(((L2 + R2) / 2, EV_Y - 6.6), ((L2 + R2) / 2, QY1 + 5.0), color=INK,
      lw=.8, scale=4.5)
dashed_v(xp(KAPPA), QY0 - 1.0, QY1 + 1.0)
dashed_v(xp(HMAX), QY0 - 1.0, QY1 + 1.0, color=BLUE)
text(xp(KAPPA) - 1.5, QY1 - 4.0, r"$\kappa$", ha="right", within=box)
text(xp(HMAX) - 1.5, QY1 - 4.0, r"$h_{\max}$", ha="right", color=BLUE_TXT,
     within=box)
ARCH = [(0.0, 41.0), (.05, 43.0), (.085, 45.0), (.16, 47.0), (.30, 50.0),
        (.38, 52.0)]
DOM = [(.12, 42.0), (.21, 45.0), (.33, 47.0), (.04, 40.0)]
for (a_, b_), (c_, d_) in zip(ARCH[:-1], ARCH[1:]):
    ax.plot([xp(a_), xp(c_), xp(c_)], [yp(b_), yp(b_), yp(d_)], color=BLUE,
            lw=.8, zorder=4)
for l_, q_ in DOM:
    ax.add_patch(Circle((xp(l_), yp(q_)), 1.7, facecolor=WHITE,
                        edgecolor=MUTED, linewidth=.6, zorder=5))
RET = ARCH[2]
for l_, q_ in ARCH:
    if (l_, q_) == RET:
        continue
    ax.add_patch(Circle((xp(l_), yp(q_)), 1.9, facecolor=BLUE,
                        edgecolor=WHITE, linewidth=.4, zorder=6))
ax.add_patch(Polygon([(xp(RET[0]) + 4.2 * math.cos(math.pi / 2 + k * 2 * math.pi / 5),
                       yp(RET[1]) + 4.2 * math.sin(math.pi / 2 + k * 2 * math.pi / 5))
                      if k % 2 == 0 else
                      (xp(RET[0]) + 1.8 * math.cos(math.pi / 2 + k * 2 * math.pi / 5),
                       yp(RET[1]) + 1.8 * math.sin(math.pi / 2 + k * 2 * math.pi / 5))
                      for k in range(10)][:0] or
                     [(xp(RET[0]) + (4.4 if i % 2 == 0 else 1.9) *
                       math.cos(math.pi / 2 + i * math.pi / 5),
                       yp(RET[1]) + (4.4 if i % 2 == 0 else 1.9) *
                       math.sin(math.pi / 2 + i * math.pi / 5))
                      for i in range(10)],
                     closed=True, facecolor=SKILL, edgecolor=WHITE,
                     linewidth=.4, zorder=7))
text((QX0 + QX1) / 2, QY0 - 6.0, r"$\widehat L$ on $V$", size=7, within=box)
text(L2 + 2.0, (QY0 + QY1) / 2, r"$Q$", size=7, within=box)
text(xp(RET[0]) + 5.5, yp(RET[1]) - 5.0, "final skill", size=6.5, color=SKILL,
     weight="bold", ha="left", within=box)
text(xp(.12), yp(53.0), "archive", size=6.5, color=BLUE_TXT, ha="left",
     within=box)
text(L2, 32.0, r"final skill: best $Q$ with $\widehat L\leq\kappa$,", size=6.5,
     color=MUTED, ha="left", within=box)
text(L2, 24.5, r"else the seed $S_0$", size=6.5, color=MUTED, ha="left",
     within=box)

# cross-panel flow: gamma (a) -> plan (b); C (b) -> round (c)
polyline_arrow([(px_ := PX[0] + PW[0] - 7.0, YG), (GAPX[0], YG),
                (GAPX[0], MY[1] + MH / 2), (MX[0] - 1.0, MY[1] + MH / 2)])
polyline_arrow([(CHX + CHW, CH_MID), (GAPX[1], CH_MID),
                (GAPX[1], (SY0 + SY1) / 2), (PX[2] + 3.5, (SY0 + SY1) / 2)])
# (the gamma bracket line runs on to the panel edge)
ax.plot([x_end := BZ + LHAT * BSC, PX[0] + PW[0] - 7.0], [YG, YG],
        color=INK, lw=.7, zorder=6)

# ---------------------------------------------------------------------------
# Legend (one row) and the illustrative-values note.
# ---------------------------------------------------------------------------
Y_LEG = 8.0


def leg_sq(color):
    def draw(x, y):
        rbox(x, y - 3.5, 7, 7, fill=color, edge=WHITE, lw=.4, radius=1.2,
             zorder=6)
        return 7
    return draw


def leg_chip(fill, edge):
    def draw(x, y):
        rbox(x, y - 3.5, 13, 7, fill=fill, edge=edge, lw=.8, radius=1.4,
             zorder=6)
        return 13
    return draw


def leg_blue(x, y):
    ax.plot([x, x + 12], [y, y], color=BLUE, lw=1.4, zorder=6,
            solid_capstyle="butt")
    return 12


def leg_dash(x, y):
    dashed_v(x + 2.0, y - 4.5, y + 4.5)
    return 4


ROW = [(leg_sq(SKILL), "skill text"),
       (leg_sq(TOOL), "tools"),
       (leg_sq(CTX_BAR), "history"),
       (leg_chip(TEXT_FILL, SKILL), "seed block"),
       (leg_chip(TEAL_FILL, TOOL), "added block"),
       (leg_blue, "BACH step"),
       (leg_dash, "budget / barrier")]
GAP_ICON, GAP_ITEM = 3.0, 9.0
ICON_W = {0: 7, 1: 7, 2: 7, 3: 13, 4: 13, 5: 12, 6: 4}
NOTE = "Values illustrative"
note_w = text_width(NOTE, 6.5, style="italic")
widths = [ICON_W[k] + GAP_ICON + text_width(lab, 7) for k, (_, lab)
          in enumerate(ROW)]
total = sum(widths) + GAP_ITEM * (len(ROW) - 1)
x = 4.0
for (fn, lab), w in zip(ROW, widths):
    iw = fn(x, Y_LEG)
    text(x + iw + GAP_ICON, Y_LEG, lab, size=7, ha="left")
    x += w + GAP_ITEM
text(WIDTH - 3.0, Y_LEG, NOTE, size=6.5, style="italic", color=MUTED,
     ha="right")

# ---------------------------------------------------------------------------
# Layout report: off-canvas, escaping-panel and overlapping labels; values.
# ---------------------------------------------------------------------------
fig.canvas.draw()
canvas = fig.bbox
boxes, problems = [], []
for label, within in labels:
    extent = label.get_window_extent(renderer)
    name = label.get_text()
    if label.get_fontsize() < MIN_FONT:
        problems.append(f"font below {MIN_FONT}: {name!r}")
    if not (extent.x0 >= canvas.x0 and extent.x1 <= canvas.x1
            and extent.y0 >= canvas.y0 and extent.y1 <= canvas.y1):
        problems.append(f"off canvas: {name!r}")
    if within is not None:
        (x0, y0), (x1, y1) = inv.transform([(extent.x0, extent.y0),
                                            (extent.x1, extent.y1)])
        if not (x0 >= within[0] + 2 and x1 <= within[2] - 2
                and y0 >= within[1] + 2 and y1 <= within[3] - 2):
            problems.append(f"escapes its panel: {name!r}")
    boxes.append((name, extent))
for i, (ta, ea) in enumerate(boxes):
    for tb, eb in boxes[i + 1:]:
        if ea.overlaps(eb):
            problems.append(f"overlap: {ta!r} / {tb!r}")

METADATA = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}
for ext, dpi in (("pdf", 72), ("png", 200), ("svg", 72)):
    fig.savefig(STEM.with_suffix("." + ext), dpi=dpi, facecolor=WHITE,
                metadata=METADATA[ext])
plt.close(fig)

print("layout problems:" if problems else "layout: no problems")
for p in problems:
    print("  " + p)
