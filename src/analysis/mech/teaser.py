"""mech_teaser: Figure 1 of paper 1, casestudy/teaser.py without the BACH panel and with a narrow attribution panel.
One AutoBench test task (casestudy.teaser.PIN, finance-vendor-payment-approval) that the seed skill and SkillOpt's skill (native-20261008-https) both solve, each drawn as its real
episode (turns -> tool calls -> component boxes, exposure edges = input tokens of a component re-sent in later turns), plus panel (c): the four-term attribution of the change in log
input tokens of this one task pair (more actions, turn-splitting, content per turn, skill text, measured total) with exactly the per-task computation of mechanism/appc.py compute_attr.
  .venv/bin/python -m src.analysis.mech.teaser compute      -> assets/cache/analysis/mech/teaser.json
  .venv/bin/python -m src.analysis.mech.teaser draw                -> paper/assets/fig/mech/mech_teaser.pdf/.svg and data/mech/mech_teaser.md"""
import json
import math
import sys
import numpy as np
from src.utils import fig, logger, dataloader
from src.utils.fig import save_figure
# constants copied from bach-harness/src/analysis/casestudy/teaser.py (the casestudy package is not part of bipa)
BENCH, MODEL = "automationbench", "deepseek-flash"
GROUPS = ["skill", "bash", "read", "other"]
PIN = "finance-vendor-payment-approval"  # the task of the drafted fig:teaser; pinned so the seed / SkillOpt panels stay the episodes the caption quotes


def group(c):
    if c == "skill": return "skill"
    if c.startswith("tool:bash"): return "bash"
    if c.startswith("tool:read") or c in ("tool:grep", "tool:glob"): return "read"
    return "other"


COL = dict(skill="#D55E00", bash="#009E73", read="#0072B2", other="#BBBBBB")
NAME = dict(skill="skill", bash="bash", read="read / grep", other="other context")

SKILLOPT_RUN = "native-20261008-https"
TERMS = ["more_actions", "more_requests", "longer_content", "longer_skill"]
TERM_LABEL = {"more_actions": "more actions", "more_requests": "turn-splitting", "longer_content": "content per turn",
              "longer_skill": "skill text", "dlogX": "total (measured)"}
TERM_COLOR = {"more_actions": "#0072B2", "more_requests": "#E69F00", "longer_content": "#CC79A7", "longer_skill": "#D55E00", "dlogX": "#222222"}


def compute():
    """Load the seed and SkillOpt episodes of the pinned task, record per-panel numbers and the four-term attribution of the pair."""
    from src.utils import dataloader as C
    from src.analysis.mechanism.slopes import metrics
    from src.graph import episode as W
    seed_run = C.current_run("native-seed", MODEL, BENCH)
    so_run = C.DS / "skillopt" / BENCH / SKILLOPT_RUN
    files = {"seed": C.eval_file(seed_run, "test", C.vid(C.SEED_TEXT)),
             "skillopt": C.eval_file(so_run, "test", C.vid(C.final_skill("skillopt", so_run)))}
    episodes = {}
    for key, path in files.items():
        episodes[key] = {e.task: e for e in W.load(path)}
    task = PIN
    for key in episodes:
        # the pin must stay solved by both skills
        if episodes[key][task].success is not True:
            raise SystemExit(f"pinned task {task} is not solved under {key}")
    panels = {}
    for key in files:
        e = episodes[key][task]
        tokens = np.zeros((e.R, 4))
        for j, comp in enumerate(e.comps):
            tokens[:, GROUPS.index(group(comp))] += e.Xrg[:, j]
        calls = sorted({(c[0], group(c[1])) for c in e.calls})
        panels[key] = dict(R=e.R, X=e.X, N=e.N, success=e.success, Xrg=tokens.tolist(), calls=calls,
                           ncalls={g: sum(1 for c in e.calls if group(c[1]) == g) for g in GROUPS},
                           tokens_by_group={g: float(tokens[:, j].sum()) for j, g in enumerate(GROUPS)})
    # four-term attribution of the pair: the per-task body of appc.compute_attr
    a = metrics(episodes["seed"][task])
    b = metrics(episodes["skillopt"][task])
    d = {k: math.log(b[k] / a[k]) for k in ("R", "N", "h", "X", "F")}
    d_m = d["N"] - d["R"]
    s_f = (a["sF"] + b["sF"]) / 2
    s_h = 1 - s_f
    r_mean = (a["R"] + b["R"]) / 2
    k = 1 + s_h * r_mean / (r_mean - 1)
    terms = dict(more_actions=k * d["N"], more_requests=-k * d_m, longer_content=s_h * d["h"], longer_skill=s_f * d["F"])
    terms["segments_sum"] = sum(terms[t] for t in TERMS)
    terms["dlogX"] = d["X"]
    out = dict(task=task, files={k: str(v) for k, v in files.items()}, runs=dict(seed=seed_run.name, skillopt=so_run.name),
               panels=panels, terms=terms, metrics=dict(seed=a, skillopt=b), k=k, s_f=s_f, r_mean=r_mean, logs=d)
    dataloader.write_json("teaser", out)
    print(json.dumps({"terms": terms, "k": k, "panels": {p: {x: v for x, v in q.items() if x in ("R", "N", "X", "success")} for p, q in panels.items()}}, indent=1))


# drawing constants: Times-like serif (TrueType, so the pdf embeds Type 42), ink colours, light panels
FONT = "Liberation Serif"
INK = "#222222"
MUTED = "#5A5A5A"
PANEL_FC = "#F7F8FA"
PANEL_EC = "#C9CED6"
TOTAL_COLOR = "#555555"
FIG_W, FIG_H = 6.3, 1.9
PANEL_TOP, PANEL_BOTTOM = 1.88, 0.25
FOOTER_Y = 0.115
DOT_Y = 1.53
BOX_Y = 0.76
STAT_Y = 0.43
BOX_W, BOX_H, BOX_GAP = 0.2, 0.15, 0.36
EDGE_PT_PER_TOKEN = 1.0 / 9000


def panel(ax, x, w, title):
    """Light rounded panel spanning PANEL_BOTTOM..PANEL_TOP with its bold 8 pt title at the top left."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x, PANEL_BOTTOM), w, PANEL_TOP - PANEL_BOTTOM, boxstyle="round,pad=0,rounding_size=0.06",
                         fc=PANEL_FC, ec=PANEL_EC, lw=0.5, zorder=0)
    ax.add_patch(box)
    ax.text(x + 0.08, PANEL_TOP - 0.1, title, fontsize=8, fontweight="bold", ha="left", va="center", color=INK)


def text_width(ax, text, size, weight="normal"):
    """Width in inches of one text run (the axes maps one data unit to one inch)."""
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(0, 0, text, fontsize=size, fontweight=weight)
    width = probe.get_window_extent(renderer).width / ax.figure.dpi
    probe.remove()
    return width


def rich_line(ax, cx, y, pieces, size):
    """One centred text line built from (text, weight, colour) runs, so numbers can be bold inside a regular line."""
    widths = [text_width(ax, text, size, weight) for text, weight, _ in pieces]
    x = cx - sum(widths) / 2
    for (text, weight, colour), w in zip(pieces, widths):
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=colour, ha="left", va="center")
        x += w


def draw_episode(ax, p, px, pw, spacing, title):
    """One episode as its turn-component graph: turn dots on top (one horizontal scale for both panels), component boxes below,
    call arrows turn -> tool box, coloured context edges box -> turn with width proportional to the component's tokens read in that turn,
    and one stat line (outcome, turns, calls, input tokens)."""
    from matplotlib.patches import FancyBboxPatch
    panel(ax, px, pw, title)
    R = p["R"]
    X = np.array(p["Xrg"])
    cx = px + pw / 2
    # turn dots: fixed spacing per turn, centred in the panel, so the longer episode is visibly wider
    dx = [cx + spacing * (r - (R - 1) / 2) for r in range(R)]
    # component boxes: the same spacing in both panels, centred under the turns
    bx = {g: cx + BOX_GAP * (j - 1.5) for j, g in enumerate(GROUPS)}
    box_top = BOX_Y + BOX_H / 2
    # context edges: only where the turn reads tokens of that component
    for g in ("skill", "bash", "read"):
        j = GROUPS.index(g)
        for r in range(R):
            tokens = X[r][j]
            if tokens <= 0:
                continue
            width = max(tokens * EDGE_PT_PER_TOKEN, 0.15)
            ax.plot([bx[g], dx[r]], [box_top, DOT_Y - 0.03], color=COL[g], lw=width, alpha=0.32, solid_capstyle="butt", zorder=1)
    # call arrows: one per (turn, tool group) pair with at least one call, heads fanned slightly over the box top
    count = {g: sum(1 for _, h in p["calls"] if h == g) for g in GROUPS}
    seen = {g: 0 for g in GROUPS}
    for turn, g in p["calls"]:
        off = (seen[g] - (count[g] - 1) / 2) * min(0.16 / max(count[g], 1), 0.03)
        seen[g] += 1
        ax.annotate("", xy=(bx[g] + off, box_top + 0.004), xytext=(dx[turn], DOT_Y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", lw=0.3, color="#9A9A9A", mutation_scale=3, shrinkA=0, shrinkB=0), zorder=3)
    # turns in time order
    ax.plot([dx[0], dx[-1]], [DOT_Y, DOT_Y], color=INK, lw=0.4, zorder=2)
    ax.plot(dx, [DOT_Y] * R, "o", ms=3.0, color=INK, mec="white", mew=0.3, zorder=4)
    # component boxes; tool boxes carry their call count
    for g in GROUPS:
        box = FancyBboxPatch((bx[g] - BOX_W / 2, BOX_Y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.025",
                             fc=COL[g], ec="none", zorder=5)
        ax.add_patch(box)
        if g in ("bash", "read"):
            ax.text(bx[g], BOX_Y, str(p["ncalls"][g]), color="white", fontsize=7, fontweight="bold", ha="center", va="center", zorder=6)
    outcome = "pass" if p["success"] else "fail"
    pieces = [(outcome, "normal", INK), (" · ", "normal", MUTED), (str(p["R"]), "bold", INK), (" turns", "normal", INK),
              (" · ", "normal", MUTED), (str(p["N"]), "bold", INK), (" calls", "normal", INK),
              (" · ", "normal", MUTED), (f"{p['X'] / 1e3:.0f}k", "bold", INK), (" tokens", "normal", INK)]
    rich_line(ax, cx, STAT_Y, pieces, 7.5)


def draw_waterfall(ax, d, px, pw):
    """Panel (c): multiplicative waterfall on a log token axis from the seed's input tokens to the learned skill's, one step per term,
    factor = exp(term) of the four-term split of this task pair; thin connectors join consecutive steps."""
    from matplotlib.patches import Polygon, Rectangle
    panel(ax, px, pw, "(c) Decomposition")
    P = d["panels"]
    t = d["terms"]
    seed = P["seed"]["X"]
    learned = P["skillopt"]["X"]
    factors = [math.exp(t[k]) for k in TERMS]
    levels = [seed]
    for f in factors:
        levels.append(levels[-1] * f)
    # log axis: bars of the two totals start at x_lo; the steps' lengths are their log factors
    x_lo, x_hi = 250e3, 800e3
    label_right = px + 0.74
    plot_left = label_right + 0.06
    plot_right = px + pw - 0.38

    def xpos(v):
        return plot_left + (plot_right - plot_left) * math.log(v / x_lo) / math.log(x_hi / x_lo)

    rows = [("seed", x_lo, seed, TOTAL_COLOR, f"{seed / 1e3:.0f}k", True)]
    for k, f, a, b in zip(TERMS, factors, levels[:-1], levels[1:]):
        rows.append((TERM_LABEL[k], a, b, TERM_COLOR[k], f"\u00d7{f:.2f}", False))
    rows.append(("learned", x_lo, learned, TOTAL_COLOR, f"{learned / 1e3:.0f}k (\u00d7{learned / seed:.2f})", True))
    y_first, y_step, bar_h = 1.55, 0.172, 0.11
    ys = [y_first - y_step * i for i in range(len(rows))]
    # faint gridlines at the ticks
    ticks = [300e3, 400e3, 600e3, 800e3]
    y_axis = ys[-1] - 0.13
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, ys[0] + bar_h / 2 + 0.03], color="#E2E5EA", lw=0.4, zorder=1)
    head = 0.05   # arrow-head length (in): every step points from its start level to its end level
    for i, (label, a, b, colour, value, total) in enumerate(rows):
        y = ys[i]
        left, right = xpos(min(a, b)), xpos(max(a, b))
        ax.text(label_right, y, label, fontsize=7.5, fontweight="bold" if total else "normal", ha="right", va="center", color=INK)
        if total:
            # the two totals are plain bars from the axis start
            ax.add_patch(Rectangle((left, y - bar_h / 2), right - left, bar_h, fc=colour, ec="none", zorder=3))
            ax.text(right + 0.04, y, value, fontsize=7.5, fontweight="bold", ha="left", va="center", color=INK)
        elif b >= a:
            # an increase: body, then a head pointing right; the factor sits right of the head
            ax.add_patch(Rectangle((left, y - bar_h / 2), right - left - head, bar_h, fc=colour, ec="none", zorder=3))
            ax.add_patch(Polygon([(right - head, y - bar_h * 0.75), (right, y), (right - head, y + bar_h * 0.75)], closed=True, fc=colour, ec="none", zorder=3))
            ax.text(right + 0.04, y, value, fontsize=7.5, ha="left", va="center", color=INK)
        else:
            # a decrease: the head points left, toward the lower level, and the factor sits left of the head
            ax.add_patch(Rectangle((left + head, y - bar_h / 2), right - left - head, bar_h, fc=colour, ec="none", zorder=3))
            ax.add_patch(Polygon([(left + head, y - bar_h * 0.75), (left, y), (left + head, y + bar_h * 0.75)], closed=True, fc=colour, ec="none", zorder=3))
            ax.text(left - 0.04, y, value, fontsize=7.5, ha="right", va="center", color=INK)
        # connector: from this bar's end level down to the next bar
        if i + 1 < len(rows):
            end = b
            ax.plot([xpos(end), xpos(end)], [y - bar_h / 2, ys[i + 1] + bar_h / 2], color="#8A8A8A", lw=0.4, zorder=2)
    # bracket over the two terms that add turns (more actions, turn-splitting): their product is the price of the extra turns
    turns_factor = factors[0] * factors[1]
    bx = xpos(seed) - 0.05
    top, bottom = ys[1] + bar_h / 2, ys[2] - bar_h / 2
    ax.plot([bx, bx], [bottom, top], color=MUTED, lw=0.6, zorder=2)
    for yy in (top, bottom):
        ax.plot([bx, bx + 0.03], [yy, yy], color=MUTED, lw=0.6, zorder=2)
    ax.text(bx - 0.03, (top + bottom) / 2, f"turns\n\u00d7{turns_factor:.2f}", fontsize=7, ha="right", va="center", color=MUTED, linespacing=1.0)
    # log token axis
    ax.plot([plot_left, plot_right], [y_axis, y_axis], color="#8A8A8A", lw=0.5, zorder=2)
    for v in ticks:
        ax.plot([xpos(v), xpos(v)], [y_axis, y_axis - 0.025], color="#8A8A8A", lw=0.5)
        ax.text(xpos(v), y_axis - 0.085, f"{v / 1e3:.0f}k", fontsize=7, ha="center", va="center", color=MUTED)


def draw_legend(ax, cx, y):
    """One-row legend of the component colours, centred at cx."""
    from matplotlib.patches import FancyBboxPatch
    items = [("skill text", COL["skill"]), ("bash output", COL["bash"]), ("read / grep output", COL["read"]), ("other context", COL["other"])]
    swatch, pad, gap = 0.1, 0.05, 0.2
    widths = [swatch + pad + text_width(ax, label, 7.5) for label, _ in items]
    x = cx - (sum(widths) + gap * (len(items) - 1)) / 2
    for (label, colour), w in zip(items, widths):
        ax.add_patch(FancyBboxPatch((x, y - 0.035), swatch, 0.07, boxstyle="round,pad=0,rounding_size=0.015", fc=colour, ec="none"))
        ax.text(x + swatch + pad, y, label, fontsize=7.5, ha="left", va="center", color=INK)
        x += w + gap


def draw():
    """Draw mech_teaser and write the record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42})
    d = json.loads((dataloader.CACHE / "teaser.json").read_text())
    P = d["panels"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    margin, gap = 0.02, 0.08
    width_a, width_b = 1.76, 2.0
    # one horizontal scale: the longer episode (b) fills its panel, (a) uses the same spacing per turn
    spacing = (width_b - 0.3) / (max(p["R"] for p in P.values()) - 1)
    x_a = margin
    x_b = x_a + width_a + gap
    x_c = x_b + width_b + gap
    width_c = FIG_W - margin - x_c
    draw_episode(ax, P["seed"], x_a, width_a, spacing, "(a) Hand-written seed")
    draw_episode(ax, P["skillopt"], x_b, width_b, spacing, "(b) Success-only optimizer (SkillOpt)")
    draw_waterfall(ax, d, x_c, width_c)
    draw_legend(ax, (x_a + x_b + width_b) / 2, FOOTER_Y)
    ratio = P["skillopt"]["X"] / P["seed"]["X"]
    ax.text(x_c + width_c / 2, FOOTER_Y, f"Same pass, \u00d7{ratio:.2f} tokens \u2014 paid in extra turns.", fontsize=8, fontweight="bold",
            ha="center", va="center", color=INK)
    save_figure(fig, "teaser")
    record(d)


def record(d):
    """Audit record with the per-panel numbers and the four terms."""
    P = d["panels"]
    rows = []
    for key in ("seed", "skillopt"):
        p = P[key]
        rows.append([key, p["R"], p["N"], f"{p['X']:.0f}", str(p["success"]),
                     " / ".join(f"{g} {p['tokens_by_group'][g]:.0f}" for g in GROUPS), " / ".join(f"{g} {n}" for g, n in p["ncalls"].items())])
    table = f"AutoBench test task `{d['task']}` (DeepSeek-V4.1-Flash, OpenCode).\n\n"
    table += logger.md_table(["panel", "turns R", "tool calls N", "input tokens X", "success", "input tokens by component", "calls by group"], rows)
    t = d["terms"]
    trows = [[TERM_LABEL[k], f"{t[k]:+.4f}", f"{math.exp(t[k]):.3f}"] for k in TERMS]
    trows.append(["sum of the four terms", f"{t['segments_sum']:+.4f}", f"{math.exp(t['segments_sum']):.3f}"])
    trows.append(["total (measured), log(X_SkillOpt / X_seed)", f"{t['dlogX']:+.4f}", f"{math.exp(t['dlogX']):.3f}"])
    table += ("\n\nPanel (c), four-term attribution of this pair (log input tokens, SkillOpt minus seed); the waterfall steps are the factors exp(term)\n\n"
              + logger.md_table(["term", "value (log)", "factor"], trows))
    a, b = d["metrics"]["seed"], d["metrics"]["skillopt"]
    mrows = [[k, f"{a[k]:.4g}", f"{b[k]:.4g}"] for k in ("R", "N", "X", "F", "h", "sF", "eR")]
    table += (f"\n\nPer-episode inputs of the decomposition (slopes_checks.metrics): ratio SkillOpt/seed of X = {P['skillopt']['X'] / P['seed']['X']:.3f}; "
              f"k = 1 + s_H R/(R-1) with the pair-averaged s_F = {d['s_f']:.4f}, R = {d['r_mean']:.1f}: k = {d['k']:.4f}.\n\n")
    table += logger.md_table(["quantity", "seed", "SkillOpt"], mrows)
    table += "\n\nFiles\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in d["files"].items())
    logger.write_record(
        "teaser",
        "Figure 1 of paper 1: one AutoBench test task solved by the seed skill and by SkillOpt's skill, each as its real episode graph (turns, tool calls, exposure of tool outputs in later turns), and panel (c): the four-term attribution of this pair as a multiplicative waterfall on a log token axis from the seed's input tokens to SkillOpt's (step factor = exp(term)).",
        f"DeepSeek-V4.1-Flash in OpenCode, AutoBench test split. Seed: native-seed/automationbench/{d['runs']['seed']} (the run-0 test episode). SkillOpt: skillopt/automationbench/{d['runs']['skillopt']} (its own test evaluation). Task pinned to casestudy.teaser.PIN `{d['task']}`; both episodes pass. No BACH episode.",
        "Panel numbers are the single test episode of each skill (groups skill, bash, read / grep, other context as in casestudy/teaser.py; turn spacing shared by (a) and (b); context edge width 1/9000 pt per token, minimum 0.15 pt, drawn only where the turn reads tokens of the component; one call arrow per (turn, tool group) pair). Panel (c) is the per-task body of mechanism/appc.py compute_attr: terms = k*dlogN, -k*(dlogN-dlogR), s_H*dlog h, s_F*dlog F with k = 1 + s_H R/(R-1) and s_F, R averaged over the two episodes; the measured total is log(X_SkillOpt/X_seed).",
        table,
        ".venv/bin/python -m src.analysis.mech.teaser compute\n.venv/bin/python -m src.analysis.mech.teaser draw",
        "Numbers: assets/cache/analysis/mech/teaser.json.")


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute()
    else:
        draw()
