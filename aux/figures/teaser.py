"""The teaser figure: one AutoBench test task that the seed and SkillOpt's skill both solve, each drawn as its episode graph, and the
four-term split of the pair's change in log input tokens.
  python -m aux.figures.teaser compute   -> teaser.json in the analysis cache
  python -m aux.figures.teaser draw      -> mech_teaser2.{pdf,svg}"""
import json
import math
import sys

import numpy as np

from src.utils import dataloader as C
from src.utils.fig import save_figure

BENCH, MODEL, TASK, SKILLOPT_RUN = "automationbench", "deepseek-flash", "finance-vendor-payment-approval", "native-20261008-https"
GROUPS = ["skill", "bash", "read", "other"]
COL = dict(skill="#D55E00", bash="#009E73", read="#0072B2", other="#BBBBBB")
TERMS = ["more_actions", "more_requests", "longer_content", "longer_skill"]
TERM_LABEL = {"more_actions": "more actions", "more_requests": "turn-splitting", "longer_content": "content per turn",
              "longer_skill": "skill text"}
NEUTRAL = "#8C8C8C"
TERM_COLOR = {"more_actions": NEUTRAL, "more_requests": NEUTRAL, "longer_content": NEUTRAL, "longer_skill": COL["skill"]}


def group(c):
    if c == "skill":
        return "skill"
    if c.startswith("tool:bash"):
        return "bash"
    return "read" if c.startswith("tool:read") or c in ("tool:grep", "tool:glob") else "other"


def compute():
    """Both episodes of the task by component group, and the four terms of the pair (the per-task split of the attribution)."""
    from src.analysis.episodes import metrics
    from src.graph import episode as W
    seed_run = C.current_run("native-seed", MODEL, BENCH)
    so_run = C.DS / "skillopt" / BENCH / SKILLOPT_RUN
    files = {"seed": C.eval_file(seed_run, "test", C.vid(C.SEED_TEXT)),
             "skillopt": C.eval_file(so_run, "test", C.vid(C.final_skill("skillopt", so_run)))}
    eps = {key: {e.task: e for e in W.load(path)}[TASK] for key, path in files.items()}
    panels = {}
    for key, e in eps.items():
        if e.success is not True:
            raise SystemExit(f"{TASK} is not solved under {key}")
        tokens = np.zeros((e.R, 4))
        for j, comp in enumerate(e.comps):
            tokens[:, GROUPS.index(group(comp))] += e.Xrg[:, j]
        panels[key] = dict(R=e.R, X=e.X, N=e.N, success=e.success, Xrg=tokens.tolist(),
                           calls=sorted({(c[0], group(c[1])) for c in e.calls}),
                           ncalls={g: sum(1 for c in e.calls if group(c[1]) == g) for g in GROUPS},
                           tokens_by_group={g: float(tokens[:, j].sum()) for j, g in enumerate(GROUPS)})
    a, b = metrics(eps["seed"]), metrics(eps["skillopt"])
    d = {k: math.log(b[k] / a[k]) for k in ("R", "N", "h", "X", "F")}
    s_f = (a["sF"] + b["sF"]) / 2
    r_mean = (a["R"] + b["R"]) / 2
    k = 1 + (1 - s_f) * r_mean / (r_mean - 1)
    terms = dict(more_actions=k * d["N"], more_requests=-k * (d["N"] - d["R"]), longer_content=(1 - s_f) * d["h"],
                 longer_skill=s_f * d["F"])
    terms |= {"segments_sum": sum(terms[t] for t in TERMS), "dlogX": d["X"]}
    C.write_json("teaser", dict(task=TASK, files={k: str(v) for k, v in files.items()},
                                runs=dict(seed=seed_run.name, skillopt=so_run.name), panels=panels, terms=terms,
                                metrics=dict(seed=a, skillopt=b), k=k, s_f=s_f, r_mean=r_mean, logs=d))
    print(json.dumps({"terms": terms, "k": k, "ratio": b["X"] / a["X"]}, indent=1))


INK, MUTED, TOTAL = "#222222", "#5A5A5A", "#555555"
FIG_W, FIG_H = 6.3, 1.9
PANEL_TOP, PANEL_BOTTOM, C_BOTTOM, FOOTER_Y = 1.88, 0.25, 0.02, 0.115
DOT_Y, BOX_Y, STAT_Y = 1.53, 0.76, 0.43
BOX_W, BOX_H, BOX_GAP = 0.2, 0.15, 0.36


def panel(ax, x, w, title, bottom=PANEL_BOTTOM):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, bottom), w, PANEL_TOP - bottom, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#F7F8FA", ec="#C9CED6", lw=0.5, zorder=0))
    ax.text(x + 0.08, PANEL_TOP - 0.1, title, fontsize=8, fontweight="bold", ha="left", va="center", color=INK)


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
    R, X, cx = p["R"], np.array(p["Xrg"]), px + pw / 2
    dx = [cx + spacing * (r - (R - 1) / 2) for r in range(R)]
    bx = {g: cx + BOX_GAP * (j - 1.5) for j, g in enumerate(GROUPS)}
    box_top = BOX_Y + BOX_H / 2
    for g in ("skill", "bash", "read"):
        for r in range(R):
            if X[r][GROUPS.index(g)] > 0:
                ax.plot([bx[g], dx[r]], [box_top, DOT_Y - 0.03], color=COL[g], lw=max(X[r][GROUPS.index(g)] * (1.0 / 9000), 0.15),
                        alpha=0.32, solid_capstyle="butt", zorder=1)
    count = {g: sum(1 for _, h in p["calls"] if h == g) for g in GROUPS}
    seen = dict.fromkeys(GROUPS, 0)
    for turn, g in p["calls"]:
        off = (seen[g] - (count[g] - 1) / 2) * min(0.16 / max(count[g], 1), 0.03)
        seen[g] += 1
        ax.annotate("", xy=(bx[g] + off, box_top + 0.004), xytext=(dx[turn], DOT_Y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", lw=0.3, color="#9A9A9A", mutation_scale=3, shrinkA=0, shrinkB=0), zorder=3)
    ax.plot([dx[0], dx[-1]], [DOT_Y, DOT_Y], color=INK, lw=0.4, zorder=2)
    ax.plot(dx, [DOT_Y] * R, "o", ms=3.0, color=INK, mec="white", mew=0.3, zorder=4)
    for g in GROUPS:
        ax.add_patch(FancyBboxPatch((bx[g] - BOX_W / 2, BOX_Y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=COL[g], ec="none", zorder=5))
        if g in ("bash", "read"):
            ax.text(bx[g], BOX_Y, str(p["ncalls"][g]), color="white", fontsize=7, fontweight="bold", ha="center", va="center", zorder=6)
    dot = (" · ", "normal", MUTED)
    rich_line(ax, cx, STAT_Y, [("pass" if p["success"] else "fail", "normal", INK), dot, (str(p["R"]), "bold", INK),
                               (" turns", "normal", INK), dot, (str(p["N"]), "bold", INK), (" calls", "normal", INK), dot,
                               (f"{p['X'] / 1e3:.0f}k", "bold", INK), (" tokens", "normal", INK)], 7.5)


def arrow(ax, left, right, y, h, colour, pointing_right):
    """A bar from left to right with its head at the end it points to."""
    from matplotlib.patches import Polygon, Rectangle
    tip, base = (right, right - 0.05) if pointing_right else (left, left + 0.05)
    ax.add_patch(Rectangle((left if pointing_right else base, y - h / 2), right - left - 0.05, h, fc=colour, ec="none", zorder=3))
    ax.add_patch(Polygon([(base, y - h * 0.75), (tip, y), (base, y + h * 0.75)], closed=True, fc=colour, ec="none", zorder=3))


def draw_waterfall(ax, d, px, pw):
    """Panel (c): from the seed's input tokens to the skill's on a log axis, one step exp(term) per term."""
    from matplotlib.patches import Rectangle
    panel(ax, px, pw, "(c) Decomposition", bottom=C_BOTTOM)
    seed, learned = d["panels"]["seed"]["X"], d["panels"]["skillopt"]["X"]
    factors = [math.exp(d["terms"][k]) for k in TERMS]
    levels = [seed]
    for f in factors:
        levels.append(levels[-1] * f)
    x_lo, x_hi, label_right = 250e3, 800e3, px + 0.74
    plot_left, plot_right = label_right + 0.06, px + pw - 0.38

    def xpos(v):
        return plot_left + (plot_right - plot_left) * math.log(v / x_lo) / math.log(x_hi / x_lo)
    rows = [("seed", x_lo, seed, TOTAL, f"{seed / 1e3:.0f}k", True)]
    rows += [(TERM_LABEL[k], a, b, TERM_COLOR[k], f"×{f:.2f}", False) for k, f, a, b in zip(TERMS, factors, levels, levels[1:])]
    rows.append(("learned", x_lo, learned, TOTAL, f"{learned / 1e3:.0f}k (×{learned / seed:.2f})", True))
    bar_h = 0.12
    ys = [1.56 - 0.225 * i for i in range(len(rows))]
    ticks, y_axis = [300e3, 400e3, 600e3, 800e3], ys[-1] - 0.15
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, ys[0] + bar_h / 2 + 0.03], color="#E2E5EA", lw=0.4, zorder=1)
    for i, (label, a, b, colour, value, total) in enumerate(rows):
        y, left, right = ys[i], xpos(min(a, b)), xpos(max(a, b))
        ax.text(label_right, y, label, fontsize=7.5, fontweight="bold" if total else "normal", ha="right", va="center", color=INK)
        if total:
            ax.add_patch(Rectangle((left, y - bar_h / 2), right - left, bar_h, fc=colour, ec="none", zorder=3))
        else:
            arrow(ax, left, right, y, bar_h, colour, b >= a)
        ax.text(left - 0.04 if b < a and not total else right + 0.04, y, value, fontsize=7.5, fontweight="bold" if total else "normal",
                ha="right" if b < a and not total else "left", va="center", color=INK)
        if i + 1 < len(rows):
            ax.plot([xpos(b), xpos(b)], [y - bar_h / 2, ys[i + 1] + bar_h / 2], color="#8A8A8A", lw=0.4, zorder=2)
    bx, top, bottom = xpos(seed) - 0.05, ys[1] + bar_h / 2, ys[2] - bar_h / 2
    ax.plot([bx, bx], [bottom, top], color=MUTED, lw=0.6, zorder=2)
    for yy in (top, bottom):
        ax.plot([bx, bx + 0.03], [yy, yy], color=MUTED, lw=0.6, zorder=2)
    ax.text(bx - 0.03, (top + bottom) / 2, f"turns\n×{factors[0] * factors[1]:.2f}", fontsize=7, ha="right", va="center",
            color=MUTED, linespacing=1.0)
    ax.plot([plot_left, plot_right], [y_axis, y_axis], color="#8A8A8A", lw=0.5, zorder=2)
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, y_axis - 0.025], color="#8A8A8A", lw=0.5)
        ax.text(xpos(v), y_axis - 0.085, f"{v / 1e3:.0f}k", fontsize=7, ha="center", va="center", color=MUTED)
    return dict(factors=dict(zip(TERMS, factors)), turns_factor=factors[0] * factors[1], seed_X=seed, learned_X=learned,
                ratio=learned / seed)


def draw_legend(ax, cx, y):
    from matplotlib.patches import FancyBboxPatch
    items = [("skill text", COL["skill"]), ("bash output", COL["bash"]), ("read / grep output", COL["read"]), ("other context", COL["other"])]
    widths = [0.1 + 0.05 + text_width(ax, label, 7.5) for label, _ in items]
    x = cx - (sum(widths) + 0.2 * (len(items) - 1)) / 2
    for (label, colour), w in zip(items, widths):
        ax.add_patch(FancyBboxPatch((x, y - 0.035), 0.1, 0.07, boxstyle="round,pad=0,rounding_size=0.015", fc=colour, ec="none"))
        ax.text(x + 0.1 + 0.05, y, label, fontsize=7.5, ha="left", va="center", color=INK)
        x += w + 0.2


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    d = json.loads((C.CACHE / "teaser.json").read_text())
    P = d["panels"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    width_a, width_b = 1.76, 2.0
    spacing = (width_b - 0.3) / (max(p["R"] for p in P.values()) - 1)
    x_a = 0.02
    x_b = x_a + width_a + 0.08
    x_c = x_b + width_b + 0.08
    draw_episode(ax, P["seed"], x_a, width_a, spacing, "(a) Hand-written seed")
    draw_episode(ax, P["skillopt"], x_b, width_b, spacing, "(b) Success-only optimizer (SkillOpt)")
    numbers = draw_waterfall(ax, d, x_c, FIG_W - 0.02 - x_c)
    draw_legend(ax, (x_a + x_b + width_b) / 2, FOOTER_Y)
    save_figure(fig, "teaser2")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    {"compute": compute, "draw": draw}[sys.argv[1]]()
