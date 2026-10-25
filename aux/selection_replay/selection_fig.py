"""The earlier mech_selection figure: the announced dev gain of SkillOpt's and GEPA's rules under no effect, from selection.json.

Run: python -m aux.selection_replay.selection_fig   -> fig/mech/mech_selection.{pdf,svg}
"""
import json

import numpy as np

from aux.noise_and_domains import appb
from aux.selection_replay.selection import BENCHES
from src.utils import dataloader
from src.utils.fig import save_figure


def style():
    """matplotlib with TrueType fonts in the pdf (the ACL check rejects Type 3)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    return plt


def draw_selection(result):
    """mech_selection, full width (6.3 in): two rows of three benchmark panels, the distribution of the announced dev gain under no effect for
    SkillOpt's rule (filled bars) and GEPA's rule (hatched outline bars), fitted prior; (a) baseline drawn like the candidates, (b) baseline at
    the run's own seed draw; the gain each Table 1 run announced as a thick dark line with a marker on top (solid, triangle: SkillOpt; dashed,
    diamond: GEPA).  Readable in grayscale; every text >= 7 pt."""
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    look = {"SkillOpt": {"offset": -0.2, "face": appb.OC["SkillOpt"], "hatch": None, "line": "-", "marker": "v"},
            "GEPA": {"offset": 0.2, "face": "white", "hatch": "////", "line": "--", "marker": "D"}}
    rows = (("hist_random_baseline", "(a) random seed draw as baseline"), ("hist_observed_baseline", "(b) baseline = the run's own seed draw"))
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.83, wspace=0.3, hspace=0.85)
    for row, (key, title) in enumerate(rows):
        for ax, (bench, disp) in zip(axs[row], BENCHES):
            r = result[bench]
            g = r["priors"]["fitted"]["rules"]
            hists = {rule: np.array(g[rule][key], dtype=float) / sum(g[rule][key]) for rule in look}
            top = max(max(int(np.nonzero(h > 0.002)[0].max()) for h in hists.values()), max(r["observed"][k]["gain"] for k in look)) + 1
            ymax = max(h[:top + 1].max() for h in hists.values())
            for rule, lk in look.items():
                xs = np.arange(top + 1) + lk["offset"]
                ax.bar(xs, hists[rule][:top + 1], width=0.38, color=lk["face"], edgecolor=appb.OC[rule], linewidth=0.8, hatch=lk["hatch"])
                # the gain this optimizer's Table 1 run announced: a thick dark line with a marker above the bars
                x_obs = r["observed"][rule]["gain"] + lk["offset"]
                ax.plot([x_obs, x_obs], [0, ymax * 1.08], color="0.1", lw=1.8, ls=lk["line"], zorder=5)
                ax.plot([x_obs], [ymax * 1.12], marker=lk["marker"], color="0.1", markersize=4.5, zorder=6, clip_on=False)
            label = f"{disp}  (K = {g['SkillOpt']['K']} / {g['GEPA']['K']})"
            if row == 1:
                label = f"{disp}  (baseline {r['observed']['SkillOpt']['baseline']}/{r['dev_tasks']})"
            ax.set_title(label, fontsize=8, pad=8)
            ax.set_ylim(0, ymax * 1.18)
            ax.set_xlabel("announced dev gain (tasks)", fontsize=7.5)
            ax.set_ylabel("probability", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # gains are whole tasks
            ax.spines[["top", "right"]].set_visible(False)
        axs[row][1].annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    handles = [Patch(facecolor=look["SkillOpt"]["face"], edgecolor=appb.OC["SkillOpt"], label="SkillOpt rule, no effect"),
               Patch(facecolor="white", edgecolor=appb.OC["GEPA"], hatch="////", label="GEPA rule, no effect"),
               Line2D([], [], color="0.1", lw=1.8, ls="-", marker="v", markersize=4.5, label="SkillOpt run's gain"),
               Line2D([], [], color="0.1", lw=1.8, ls="--", marker="D", markersize=4, label="GEPA run's gain")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.53, 1.0))
    save_figure(fig, "selection")


if __name__ == "__main__":
    draw_selection(json.load(open(dataloader.CACHE / "selection.json")))
