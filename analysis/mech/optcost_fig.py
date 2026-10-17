"""mech_optcost: the optimizer's own bill (appendix, replaces the table).
Per optimizer run (rows grouped by benchmark) two horizontal bars on one log axis in episodes: filled = the bill in seed test episodes
(nominal input tokens of the optimization / median input tokens of one seed test episode), hollow = break-even (episodes of use after which the
per-episode token tax exceeds the bill). Reads assets/cache/analysis/mech/optcost.json (nominal values); no .venv needed.
  python3 -m analysis.mech.optcost_fig"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.mech import common as M

BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}


def fmt(v):
    """Compact count label: 1,975 -> 2.0k style above 1000."""
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def draw():
    data = json.loads((M.CACHE / "optcost.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.16, top=0.985)
    pitch, bh, gap = 1.0, 0.42, 0.4
    y, yticks, ylabels, rows = 0.0, [], [], []
    for bkey, disp in BENCH:
        first = y
        for o in OPTS:
            m = data[bkey]["methods"][o]
            bill, be = m["seed_episodes_nominal"], m["breakeven_nominal"]
            c = OC[o]
            ax.barh(y - 0.25, bill, height=bh, color=c, lw=0, zorder=3)
            ax.barh(y + 0.25, be, height=bh, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
            ax.text(bill * 1.08, y - 0.25, fmt(bill), va="center_baseline", ha="left", fontsize=7, zorder=4)
            ax.text(be * 1.08, y + 0.25, fmt(be), va="center_baseline", ha="left", fontsize=7, zorder=4)
            yticks.append(y)
            ylabels.append(o)
            rows.append((disp, o, bill, be))
            y += pitch
        y += gap
    ax.set_xscale("log")
    ax.set_xlim(300, 60000)
    ax.set_ylim(y - gap - 0.3, -0.55)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("episodes (log scale)")
    # benchmark names left of the optimizer names
    for i, (bkey, disp) in enumerate(BENCH):
        ymid = yticks[3 * i + 1]
        ax.annotate(disp, xy=(0.01, ymid), xycoords=("figure fraction", "data"), ha="left", va="center", fontweight="bold", fontsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, fc="0.35", lw=0), plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.35", lw=0.8)]
    ax.legend(h, ["bill (seed test episodes)", "break-even (episodes of use)"], loc="center right", frameon=False,
              fontsize=7, handlelength=1.2, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.56))
    M.save_figure(fig, "optcost")
    return rows


def main():
    rows = draw()
    table = M.md_table(["benchmark", "optimizer", "bill (seed test episodes)", "break-even (episodes)"],
                       [(b, o, f"{x:.0f}", f"{e:.0f}") for b, o, x, e in rows])
    M.write_record("optcost", "The optimizer's own bill per run: nominal optimization input tokens in seed test episodes (filled bar) and the break-even number of episodes of use after which the per-episode token tax exceeds the bill (hollow bar), log axis.",
                   "Nine optimizer runs (GEPA, SkillOpt, SkillAdam x tau3, AutoBench, TB2), native-20261008-https.",
                   "assets/cache/analysis/mech/optcost.json, fields seed_episodes_nominal and breakeven_nominal.", table,
                   "python3 -m analysis.mech.optcost_fig")


if __name__ == "__main__":
    main()
