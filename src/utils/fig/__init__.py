"""The paper's one figure style: fonts, text sizes, widths, colours, markers, the matplotlib settings and the save function.

Every figure draws with these. A colour or marker a figure needs and this module lacks is added here, never in the figure.
A figure keeps its own size, layout and text sizes (pass them to pyplot); everything else comes from here. Each colour has
one meaning across the paper: the optimizers keep the colours of the main-text figures, the token components and call
kinds the Okabe-Ito hues of the teaser and of Figure 5."""
from src import layout
from src.utils.dataloader import CACHE

FIG = layout.OUT / "fig/mech"

FONT = "Liberation Serif"
# text sizes in points; TINY is the smallest size the paper prints
TEXT, SMALL, TINY, TITLE, HEAD = 7.5, 7.0, 6.5, 8.0, 8.5
# widths in inches: the text block (455.24 pt) and one column of the two-column body
TEXT_W, COL_W = 6.3, 219.08 / 72
PREVIEW_DPI = 200

# greys, darkest first, and the named greys of the teaser panels and the dev / test figure
INK, DARK, MUTED, MID, LIGHT, FAINT, GRID = "#222222", "#444444", "#5A5A5A", "#8C8C8C", "#BBBBBB", "#D9D9D9", "#E2E5EA"
PANEL, PANEL_EDGE = "#F7F8FA", "#C9CED6"
SLATE = {"bar": "#D5DAE0", "tick": "#5F6873", "note": "#6B7480"}

# arms and the learned skills pooled
ARM = {"native-noseed": LIGHT, "native-seed": INK, "gepa": "#2A9D4B", "skillopt": "#1F6FB5", "skilladam": "#E08A1E"}
LEARNED = "#882255"
MARKER = {"native-noseed": "o", "native-seed": "o", "gepa": "^", "skillopt": "o", "skilladam": "s"}
# models: DeepSeek a filled circle, Luna a hollow diamond
MODEL_MARKER = {"deepseek-flash": dict(marker="o", filled=True), "gpt-6-luna": dict(marker="D", filled=False)}
# benchmarks: markers where benchmarks share a panel; colours of the turn-slope figure; names in text and on ticks
BENCH_MARKER = {"tau3-bench": "o", "automationbench": "s", "terminal-bench-2-1": "^"}
BENCH_COLOR = {"tau3-bench": "#0072B2", "automationbench": "#D55E00", "terminal-bench-2-1": "#009E73"}
BENCH_LABEL = {"tau3-bench": "τ³-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
BENCH_TICK = {"tau3-bench": "τ³", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
# what an input token is spent on, and what a call does
COMPONENT = {"prompt": LIGHT, "skill": "#D55E00", "tools": "#009E73", "messages": "#56B4E9"}
CALL = {"bash": "#009E73", "read": "#0072B2", "read_line": "#3F7FB5", "read_light": "#7FA7CC", "other": LIGHT,
        "write": "#C0392B", "none": FAINT}
# rule categories of the added text
RULE = {"verify": "#6A3D9A", "serialize": "#A6761D", "ask_stop": "#969696", "domain": "#C9B99A", "format": "#ECE6D6",
        "other": "#D4D4D4"}
# dev and test panels, kept and rejected edits
SPLIT = {"dev": "#5B7DB1", "test": "#C0504D"}
EDIT = {"kept": "#4A4A4A", "rejected": "#A9A9A9"}

RC = {"font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
      "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}


def pyplot(size=TEXT, title=None, label=None, tick=None, legend=None):
    """matplotlib.pyplot with the Agg backend and the paper's settings, from matplotlib's defaults (nothing carries over
    from a figure drawn before). size is the figure's base text size; title, label, tick and legend default to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcdefaults()
    plt.rcParams.update(RC | {"font.size": size, "axes.titlesize": title or size, "figure.titlesize": title or size,
                              "axes.labelsize": label or size, "xtick.labelsize": tick or size, "ytick.labelsize": tick or size,
                              "legend.fontsize": legend or size, "legend.title_fontsize": legend or size})
    return plt


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def check(fig, width=None):
    """[(problem, text, ...)] for every visible text smaller than TINY as printed, outside the canvas, or overlapping another;
    width is the printed width in inches (default: the figure's own width, i.e. printed unscaled)."""
    import matplotlib
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = (width or fig.get_figwidth()) / fig.get_figwidth()
    bad, boxes = [], []
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        if t.get_fontsize() * scale < TINY - 1e-6:
            bad.append(("small", t.get_text(), round(t.get_fontsize() * scale, 2)))
        bb = t.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or bb.y1 > fig.bbox.height + 0.5:
            bad.append(("clipped", t.get_text()))
        boxes.append((t.get_text(), bb))
    bad += [("overlap", a, b) for i, (a, p) in enumerate(boxes) for b, q in boxes[i + 1:] if p.overlaps(q)]
    return bad


def save_figure(fig, name, width=None):
    """OUT/fig/mech/mech_<name>.pdf and .svg (text kept as text), a png preview in the analysis cache, and a printed report
    of the layout problems check() finds (reported, never fatal)."""
    for problem in check(fig, width):
        print(f"mech_{name}: layout:", *problem)
    import matplotlib
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({k: RC[k] for k in ("svg.fonttype", "pdf.fonttype", "ps.fonttype")}):
        fig.savefig(FIG / f"mech_{name}.svg")
        fig.savefig(FIG / f"mech_{name}.pdf")
    fig.savefig(CACHE / f"mech_{name}.png", dpi=PREVIEW_DPI)
