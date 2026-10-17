"""mech_price: appendix figure replacing the table "The token tax under cached-input pricing".
Data: assets/cache/analysis/mech/price3.json (written by analysis.mech.price). For each learned skill (3 optimizers x 3 benchmarks), price ratio to the seed at the
main weighting (cached 0.1, output 4): per-task geometric mean (filled marker) and ratio of totals (hollow marker); a line spans the min..max of both ratios over
the nine weightings; right column = cached share of the skill's input. Run: python3 -m src.analysis.mech.price_fig"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.analysis.mech import common as M

BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
OPT = ["GEPA", "SkillOpt", "SkillAdam"]
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
OM = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}


def main():
    d = json.loads((M.CACHE / "price3.json").read_text())["results"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.25, top=0.99)
    y = 0.0
    rows, ticks, tl = [], [], []
    for bench, disp in BENCH:
        ax.text(-0.30, y, disp, transform=ax.get_yaxis_transform(), ha="left", va="center", fontweight="bold", fontsize=7)
        y += 1
        for o in OPT:
            r = d[bench][o]
            lo, hi = min(r["geo_min"], r["sum_min"]), max(r["geo_max"], r["sum_max"])
            # thin guide from the seed (1) to the skill, then the range over nine weightings as a thick bar
            ax.plot([1.0, r["geo"]], [y, y], color=OC[o], lw=0.5, alpha=0.5, zorder=1)
            ax.plot([lo, hi], [y, y], color=OC[o], lw=3.2, alpha=0.28, solid_capstyle="butt", zorder=2)
            ax.plot(r["geo"], y, marker=OM[o], ms=4.5, color=OC[o], ls="", zorder=4, mew=0.8)
            ax.plot(r["sum"], y, marker=OM[o], ms=4.5, mfc="white", mec=OC[o], ls="", zorder=4, mew=0.9)
            ax.text(1.03, y, f"{100 * r['cached_share']:.1f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7)
            ticks.append(y); tl.append(o)
            rows.append([disp, o, r["geo"], r["sum"], lo, hi, r["cached_share"]])
            y += 1
        y += 0.4
    ax.axvline(1.0, color="0.2", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(1.0, -1.0, "seed = 1", ha="center", va="center", fontsize=7, color="0.2")
    ax.set_ylim(y - 0.3, -1.5)
    ax.set_xlim(0.97, 1.36)
    ax.set_yticks(ticks); ax.set_yticklabels(tl)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(1.03, -0.55, "cached\nshare (%)", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, linespacing=0.95)
    h = [plt.Line2D([], [], marker="o", ls="", color="0.3", ms=4.5), plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4.5, mew=0.9),
         plt.Line2D([], [], color="0.3", lw=3.2, alpha=0.3)]
    fig.legend(h, ["per task", "total", "range, nine weightings"], loc="lower center", bbox_to_anchor=(0.53, -0.005), ncol=3, frameon=False,
               fontsize=7, handlelength=1.3, columnspacing=1.2, handletextpad=0.3)
    ax.annotate("price ratio to the seed (cached 0.1, output 4)", xy=(0.5, -0.1), xycoords="axes fraction", ha="center", va="top", fontsize=7)
    M.save_figure(fig, "price")
    table = M.md_table(["benchmark", "skill", "per-task ratio", "total ratio", "min over 9", "max over 9", "cached share"],
                       [[a, b, f"{c:.3f}", f"{e:.3f}", f"{f:.3f}", f"{g:.3f}", f"{h_:.3f}"] for a, b, c, e, f, g, h_ in rows])
    M.write_record("price", "Price ratio of each learned skill to the seed at cached weight 0.1 and output weight 4, per task (geometric mean) and in total, with the min-max range of both over the nine weightings, and the cached share of input.",
                   "Test runs of Table 1 (price3.json, see analysis.mech.price).", "Ratios as in price3.json (geo, sum, geo_min/max, sum_min/max); range = min of the two mins, max of the two maxes.", table,
                   "Figure 4.5 x 2.5 in.", "python3 -m src.analysis.mech.price_fig")
    print(table)


if __name__ == "__main__":
    main()
