"""The teaser figure (Figure 1): one AutoBench test task that the seed and SkillOpt's skill both solve under DeepSeek, each
drawn as its episode graph, and the four-term split of the pair's change in log input tokens as a waterfall.

The episodes are each side's first test run (aux.tokens.split reads them). A panel's turns are dots, its tool
calls arrows from a turn to the tool group it called, and its context edges, drawn where a turn reads tokens of a group,
are as wide as those tokens. Panel (c) is the split of aux.tokens.split on this one pair of episodes, each step a
factor exp(term) on a log axis from the seed's input tokens to the skill's."""
import math

import numpy as np

from src.analysis import settings as S
from aux.tokens.split import four_terms, side
from src.utils import fig as style
from src.utils.logger import md_table, record

BENCH, TASK, SKILL = "automationbench", "finance-vendor-payment-approval", "skillopt"
GROUPS = ["skill", "bash", "read", "other"]
COL = dict(skill=style.COMPONENT["skill"], bash=style.CALL["bash"], read=style.CALL["read"], other=style.CALL["other"])
TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
FIG_W, FIG_H = style.TEXT_W, 1.9
PANEL_TOP, PANEL_BOTTOM, C_BOTTOM, FOOTER_Y = 1.88, 0.25, 0.02, 0.115
DOT_Y, BOX_Y, STAT_Y = 1.53, 0.76, 0.43
BOX_W, BOX_H, BOX_GAP = 0.2, 0.15, 0.36


def group(comp):
    if comp == "skill":
        return "skill"
    if comp.startswith("tool:bash"):
        return "bash"
    return "read" if comp.startswith("tool:read") or comp in ("tool:grep", "tool:glob") else "other"


def panel_data(method):
    e, m = side(S.DEEPSEEK, BENCH, method)[0][0][TASK]
    if e.success is not True:
        raise SystemExit(f"{TASK} is not solved under {method}")
    tokens = np.zeros((e.R, len(GROUPS)))
    for j, comp in enumerate(e.comps):
        tokens[:, GROUPS.index(group(comp))] += e.Xrg[:, j]
    return dict(R=e.R, X=e.X, N=e.N, success=e.success, Xrg=tokens, metrics=m, calls=sorted({(c[0], group(c[1])) for c in e.calls}),
                ncalls={g: sum(group(c[1]) == g for c in e.calls) for g in GROUPS}, tokens={g: float(tokens[:, j].sum()) for j, g in enumerate(GROUPS)})


def panel(ax, x, w, title, bottom=PANEL_BOTTOM):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, bottom), w, PANEL_TOP - bottom, boxstyle="round,pad=0,rounding_size=0.06", fc=style.PANEL, ec=style.PANEL_EDGE,
                                lw=0.5, zorder=0))
    ax.text(x + 0.08, PANEL_TOP - 0.1, title, fontsize=8, fontweight="bold", ha="left", va="center", color=style.INK)


def text_width(ax, text, size, weight="normal"):
    """Width in inches of a text run (one data unit is one inch)."""
    probe = ax.text(0, 0, text, fontsize=size, fontweight=weight)
    width = probe.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.figure.dpi
    probe.remove()
    return width


def rich_line(ax, cx, y, pieces, size):
    """A centred line of (text, weight, colour) runs."""
    widths = [text_width(ax, text, size, weight) for text, weight, _ in pieces]
    x = cx - sum(widths) / 2
    for (text, weight, colour), w in zip(pieces, widths):
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=colour, ha="left", va="center")
        x += w


def draw_episode(ax, p, px, pw, spacing, title):
    """Turn dots on top, component boxes below, call arrows turn -> tool box, context edges sized by tokens read."""
    from matplotlib.patches import FancyBboxPatch
    panel(ax, px, pw, title)
    R, X, cx = p["R"], p["Xrg"], px + pw / 2
    dx = [cx + spacing * (r - (R - 1) / 2) for r in range(R)]
    bx = {g: cx + BOX_GAP * (j - 1.5) for j, g in enumerate(GROUPS)}
    box_top = BOX_Y + BOX_H / 2
    for g in ("skill", "bash", "read"):
        for r in range(R):
            if X[r][GROUPS.index(g)] > 0:
                ax.plot([bx[g], dx[r]], [box_top, DOT_Y - 0.03], color=COL[g], lw=max(X[r][GROUPS.index(g)] / 9000, 0.15), alpha=0.32,
                        solid_capstyle="butt", zorder=1)
    count, seen = {g: sum(h == g for _, h in p["calls"]) for g in GROUPS}, dict.fromkeys(GROUPS, 0)
    for turn, g in p["calls"]:
        off = (seen[g] - (count[g] - 1) / 2) * min(0.16 / max(count[g], 1), 0.03)
        seen[g] += 1
        ax.annotate("", xy=(bx[g] + off, box_top + 0.004), xytext=(dx[turn], DOT_Y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", lw=0.3, color=style.MID, mutation_scale=3, shrinkA=0, shrinkB=0), zorder=3)
    ax.plot([dx[0], dx[-1]], [DOT_Y, DOT_Y], color=style.INK, lw=0.4, zorder=2)
    ax.plot(dx, [DOT_Y] * R, "o", ms=3.0, color=style.INK, mec="white", mew=0.3, zorder=4)
    for g in GROUPS:
        ax.add_patch(FancyBboxPatch((bx[g] - BOX_W / 2, BOX_Y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.025", fc=COL[g],
                                    ec="none", zorder=5))
        if g in ("bash", "read"):
            ax.text(bx[g], BOX_Y, str(p["ncalls"][g]), color="white", fontsize=7, fontweight="bold", ha="center", va="center", zorder=6)
    dot, ink = (" · ", "normal", style.MUTED), style.INK
    rich_line(ax, cx, STAT_Y, [("pass" if p["success"] else "fail", "normal", ink), dot, (str(p["R"]), "bold", ink), (" turns", "normal", ink), dot,
                               (str(p["N"]), "bold", ink), (" calls", "normal", ink), dot, (f"{p['X'] / 1e3:.0f}k", "bold", ink),
                               (" tokens", "normal", ink)], 7.5)


def arrow(ax, left, right, y, h, colour, pointing_right):
    """A bar from left to right with its head at the end it points to."""
    from matplotlib.patches import Polygon, Rectangle
    tip, base = (right, right - 0.05) if pointing_right else (left, left + 0.05)
    ax.add_patch(Rectangle((left if pointing_right else base, y - h / 2), right - left - 0.05, h, fc=colour, ec="none", zorder=3))
    ax.add_patch(Polygon([(base, y - h * 0.75), (tip, y), (base, y + h * 0.75)], closed=True, fc=colour, ec="none", zorder=3))


def draw_waterfall(ax, seed, learned, terms, px, pw):
    """Panel (c): from the seed's input tokens to the skill's on a log axis, one step exp(term) per term."""
    from matplotlib.patches import Rectangle
    panel(ax, px, pw, "(c) Decomposition", bottom=C_BOTTOM)
    factors = [math.exp(terms[k]) for k in TERMS]
    levels = [seed]
    for f in factors:
        levels.append(levels[-1] * f)
    x_lo, x_hi, label_right = 250e3, 800e3, px + 0.74
    plot_left, plot_right = label_right + 0.06, px + pw - 0.38

    def xpos(v):
        return plot_left + (plot_right - plot_left) * math.log(v / x_lo) / math.log(x_hi / x_lo)
    rows = [("seed", x_lo, seed, style.MUTED, f"{seed / 1e3:.0f}k", True)]
    rows += [(TERM_LABEL[k], a, b, COL["skill"] if k == "skill_text" else style.MID, f"×{f:.2f}", False)
             for k, f, a, b in zip(TERMS, factors, levels, levels[1:])]
    rows.append(("learned", x_lo, learned, style.MUTED, f"{learned / 1e3:.0f}k (×{learned / seed:.2f})", True))
    bar_h = 0.12
    ys = [1.56 - 0.225 * i for i in range(len(rows))]
    ticks, y_axis = [300e3, 400e3, 600e3, 800e3], ys[-1] - 0.15
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, ys[0] + bar_h / 2 + 0.03], color=style.GRID, lw=0.4, zorder=1)
    for i, (label, a, b, colour, value, total) in enumerate(rows):
        y, left, right = ys[i], xpos(min(a, b)), xpos(max(a, b))
        ax.text(label_right, y, label, fontsize=7.5, fontweight="bold" if total else "normal", ha="right", va="center", color=style.INK)
        if total:
            ax.add_patch(Rectangle((left, y - bar_h / 2), right - left, bar_h, fc=colour, ec="none", zorder=3))
        else:
            arrow(ax, left, right, y, bar_h, colour, b >= a)
        shrink = b < a and not total
        ax.text(left - 0.04 if shrink else right + 0.04, y, value, fontsize=7.5, fontweight="bold" if total else "normal",
                ha="right" if shrink else "left", va="center", color=style.INK)
        if i + 1 < len(rows):
            ax.plot([xpos(b), xpos(b)], [y - bar_h / 2, ys[i + 1] + bar_h / 2], color=style.MID, lw=0.4, zorder=2)
    bx, top, bottom = xpos(seed) - 0.05, ys[1] + bar_h / 2, ys[2] - bar_h / 2
    ax.plot([bx, bx], [bottom, top], color=style.MUTED, lw=0.6, zorder=2)
    for yy in (top, bottom):
        ax.plot([bx, bx + 0.03], [yy, yy], color=style.MUTED, lw=0.6, zorder=2)
    ax.text(bx - 0.03, (top + bottom) / 2, f"turns\n×{factors[0] * factors[1]:.2f}", fontsize=7, ha="right", va="center", color=style.MUTED, linespacing=1.0)
    ax.plot([plot_left, plot_right], [y_axis, y_axis], color=style.MID, lw=0.5, zorder=2)
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, y_axis - 0.025], color=style.MID, lw=0.5)
        ax.text(xpos(v), y_axis - 0.085, f"{v / 1e3:.0f}k", fontsize=7, ha="center", va="center", color=style.MUTED)


def draw_legend(ax, cx, y):
    from matplotlib.patches import FancyBboxPatch
    items = [("skill text", COL["skill"]), ("bash output", COL["bash"]), ("read / grep output", COL["read"]), ("other context", COL["other"])]
    widths = [0.15 + text_width(ax, label, 7.5) for label, _ in items]
    x = cx - (sum(widths) + 0.2 * (len(items) - 1)) / 2
    for (label, colour), w in zip(items, widths):
        ax.add_patch(FancyBboxPatch((x, y - 0.035), 0.1, 0.07, boxstyle="round,pad=0,rounding_size=0.015", fc=colour, ec="none"))
        ax.text(x + 0.15, y, label, fontsize=7.5, ha="left", va="center", color=style.INK)
        x += w + 0.2


def main():
    p = {"seed": panel_data("native-seed"), SKILL: panel_data(SKILL)}
    a, b = p["seed"]["metrics"], p[SKILL]["metrics"]
    (terms, dX), s_f, r_mean = four_terms([(a, b)]), (a["sF"] + b["sF"]) / 2, (a["R"] + b["R"]) / 2
    terms = {k: float(v[0]) for k, v in terms.items()}
    rows = [[k, p[k]["R"], p[k]["N"], f"{p[k]['X']:.0f}", p[k]["success"], " / ".join(f"{g} {p[k]['tokens'][g]:.0f}" for g in GROUPS),
             " / ".join(f"{g} {p[k]['ncalls'][g]}" for g in GROUPS)] for k in p]
    term_rows = [[TERM_LABEL[k], f"{terms[k]:+.4f}", f"{math.exp(terms[k]):.3f}"] for k in TERMS]
    term_rows += [["sum of the four terms", f"{sum(terms.values()):+.4f}", f"{math.exp(sum(terms.values())):.3f}"],
                  [f"total (measured), log(X_{S.ARM_NAME[SKILL]} / X_seed)", f"{dX[0]:+.4f}", f"{math.exp(dX[0]):.3f}"]]
    metric_rows = [[q] + [f"{m[q]:.4f}" if q in ("sF", "eR") else f"{m[q]:.1f}" if q in ("F", "h") else f"{m[q]:.0f}" for m in (a, b)]
                   for q in ("R", "N", "X", "F", "h", "sF", "eR")]
    numbers = "\n\n".join([
        f"AutoBench test task `{TASK}` ({S.MODEL_NAME[S.DEEPSEEK]}).",
        md_table(["panel", "turns R", "tool calls N", "input tokens X", "success", "input tokens by component", "calls by group"], rows),
        f"Panel (c), the four-term split of this pair (log input tokens, {S.ARM_NAME[SKILL]} minus seed); the waterfall steps are the factors exp(term):",
        md_table(["term", "value (log)", "factor"], term_rows),
        f"Midpoint prefix share s_F = {s_f:.4f}, turns R = {r_mean:.1f}, e_R = {1 + (1 - s_f) * r_mean / (r_mean - 1):.4f}.",
        md_table(["quantity", "seed", S.ARM_NAME[SKILL]], metric_rows)])
    record(__spec__.name, "teaser", numbers, S.runs_text([(S.DEEPSEEK, BENCH, "native-seed"), (S.DEEPSEEK, BENCH, SKILL)]))
    plt = style.pyplot(style.TEXT)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, FIG_W), ylim=(0, FIG_H))
    ax.axis("off")
    width_a, width_b = 1.76, 2.0
    spacing = (width_b - 0.3) / (max(q["R"] for q in p.values()) - 1)
    x_a = 0.02
    x_b = x_a + width_a + 0.08
    x_c = x_b + width_b + 0.08
    draw_episode(ax, p["seed"], x_a, width_a, spacing, "(a) Hand-written seed")
    draw_episode(ax, p[SKILL], x_b, width_b, spacing, f"(b) Learned skill ({S.ARM_NAME[SKILL]})")
    draw_waterfall(ax, p["seed"]["X"], p[SKILL]["X"], terms, x_c, FIG_W - 0.02 - x_c)
    draw_legend(ax, (x_a + x_b + width_b) / 2, FOOTER_Y)
    style.save_figure(fig, "teaser2")


if __name__ == "__main__":
    main()
