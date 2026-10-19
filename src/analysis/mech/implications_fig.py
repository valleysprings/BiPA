"""Paper 1 (mechanism paper), Section 5: the evidence figure behind the design principles for skill optimizers.

Role in the argument: each principle of Section 5 rests on a replay of the optimizers' own logs or test episodes. This module only lays out
four panels that the analysis modules already draw, one per principle:
  (a) keep only real gains      selrules.panel_bars   per acceptance rule, how often it keeps an identical candidate and a true gain
  (b) price the behavior        designev.panel_predict R^2 of an edit's change in input tokens on its change in skill length, turns, actions
  (c) batch independent calls   batching.panel         token tax as recorded and with the seed's calls per turn restored
  (d) cost-aware proposals      designev.panel_gate    change in input tokens of rejected and accepted proposals, and a cost-aware gate
The numbers live in the records mech_selrules.md, mech_designev.md and mech_batching.md.

Command (from bipa):  python3 -m src.analysis.mech.implications_fig  -> fig/mech/mech_implications.{pdf,svg}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.mech import batching
from src.utils import fig, dataloader
from src.analysis.mech import designev
from src.analysis.mech import selrules

TITLES = ["(a) What each rule keeps", "(b) What explains token change", "(c) Seed's calls per turn", "(d) Kept vs. rejected"]


def legend_selrules(ax):
    """Panel (a) legend above its bars: which candidate a bar counts."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="0.72", label="unchanged edit"), Patch(facecolor="#0072B2", label="+2 dev tasks")]
    ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.8, frameon=False, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0.0)


def main():
    """Load the three caches and draw the 1x4 figure at full text width."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.6})
    sel = json.loads((dataloader.CACHE / "selrules.json").read_text())
    dev = json.loads((dataloader.CACHE / "designev.json").read_text())
    bat = json.loads((dataloader.CACHE / "batching.json").read_text())

    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.72), gridspec_kw=dict(width_ratios=[1.35, 1, 0.9, 0.85]))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.235, top=0.875, wspace=0.62)

    # (a) the four acceptance rules on the fitted seed task model
    selrules.panel_bars(axes[0], sel)
    legend_selrules(axes[0])
    # (b) what predicts an edit's change in input tokens
    designev.panel_predict(axes[1], dev)
    # (c) the tax with the seed's calls per turn restored
    batching.panel(axes[2], bat)
    # (d) rejected vs accepted proposals and the cost-aware gate
    designev.panel_gate(axes[3], dev)

    for ax, title in zip(axes, TITLES):
        ax.set_title(title, fontsize=7, loc="left", pad=3, fontweight="bold")
    fig.save_figure(fig, "implications")


if __name__ == "__main__":
    main()
