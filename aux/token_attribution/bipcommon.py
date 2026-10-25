"""Shared helpers of the bipartite-view figures.

Episode graphs come from src.graph.episode.load (parse cache under $BIPA_ASSETS/cache/workload-parse, filled on a miss).
Figures and numbers go to OUT/token_attribution.
"""
from src import layout
from src.graph import episode as W

OUT = layout.OUT / "token_attribution"

# figure style
FONT = "Liberation Serif"
INK = "#222222"
MUTED = "#5A5A5A"
PANEL_FC = "#F7F8FA"
PANEL_EC = "#C9CED6"
GRID = "#E2E5EA"
# the four terms of the input-token split
TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
TERM_COLOR = dict(more_actions="#0072B2", turn_splitting="#E69F00", content="#CC79A7", skill_text="#D55E00")
TOTAL_COLOR = "#222222"
BENCH_NAME = {"tau3-bench": r"$\tau^3$-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2",
              "tau3": r"$\tau^3$-bench", "AutoB": "AutoBench", "TB2": "TB2"}
MODEL_NAME = {"deepseek-flash": "DeepSeek-V4.1-Flash", "gpt-6-luna": "GPT-6-Luna"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "stix", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    return plt


def save(fig, name):
    """pdf (TrueType fonts), svg (editable text) and a 200 dpi png preview in OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)


def load_cached(eval_path):
    """Episodes of one DeepSeek evaluation.json, through the shared parse cache."""
    return W.load(eval_path)


def group_of(comp):
    """Coarse component group of a graph column: rest of prompt, skill text, tool outputs, conversation (messages so far)."""
    if comp == "skill":
        return "skill"
    if comp in ("system", "tool_schemas", "task"):
        return "prompt"
    if comp == "history":
        return "messages"
    if comp.startswith("tool:"):
        return "tools"
    return "other"
