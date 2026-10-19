"""Figure output for the analyses: paper/assets/fig/mech target paths and save_figure (pdf + editable svg, png preview into the mech cache)."""
from pathlib import Path

from src.utils.dataloader import H, CACHE

FIG = H / "paper/assets/fig/mech"
# paper/assets was regrouped on 2026-10-12 (fig/mech, fig/method, data/mech, data/method); the existing BACH-paper files are looked up in either layout
OLD_FIG_DIRS = [FIG, FIG.parent]


def save_figure(fig, name):
    """Write paper/assets/fig/mech/mech_<name>.pdf and .svg (svg.fonttype none, text stays editable) and a png preview into the mech cache."""
    import matplotlib
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # TrueType (Type 42) fonts in pdf/ps so the PDFs embed without font-type warnings
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(FIG / f"mech_{name}.svg")
        fig.savefig(FIG / f"mech_{name}.pdf")
    fig.savefig(CACHE / f"mech_{name}.png", dpi=200)
