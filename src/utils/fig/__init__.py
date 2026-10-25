"""Figure output: pdf and editable svg in OUT/fig/mech, png preview in the analysis cache."""
from src import layout
from src.utils.dataloader import CACHE

FIG = layout.OUT / "fig/mech"


def save_figure(fig, name):
    import matplotlib
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(FIG / f"mech_{name}.svg")
        fig.savefig(FIG / f"mech_{name}.pdf")
    fig.savefig(CACHE / f"mech_{name}.png", dpi=200)
