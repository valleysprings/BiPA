"""The per-domain figure: test success (avg@3) per task domain of tau3-bench and AutoBench for No skill (hollow grey), Seed (black), GEPA, SkillOpt
and SkillAdam on DeepSeek-V4.1-Flash, the two benchmarks side by side; every learned skill more than 7.3 points from the seed carries its difference.
Reads domains.json from python -m aux.noise_and_domains.domains (its Luna rows are not drawn).
Run:  python -m aux.noise_and_domains.domains_fig   -> fig/mech/mech_domains_wide.{pdf,svg}, data/mech/mech_domains_wide_numbers.json
"""
import json

from src.utils import dataloader, logger
from src.utils.fig import save_figure

FIG_ROWS = ["No skill", "Seed skill", "GEPA", "SkillOpt", "SkillAdam"]
FIG_LABEL = {"Seed skill": "Seed"}
FIG_STYLE = {"No skill": dict(m="o", fc="white", ec="0.5"), "Seed skill": dict(m="o", fc="k", ec="k"),
             "GEPA": dict(m="^", fc="#009E73", ec="#009E73"), "SkillOpt": dict(m="o", fc="#0072B2", ec="#0072B2"),
             "SkillAdam": dict(m="s", fc="#D55E00", ec="#D55E00")}
FIG_PANELS = [("tau3-bench", r"$\tau^3$-bench", ["airline", "retail", "telecom", "banking_knowledge"], ["airline", "retail", "telecom", "banking"]),
              ("automationbench", "AutoBench", ["finance", "hr", "marketing", "operations", "sales", "support"],
               ["finance", "HR", "marketing", "operations", "sales", "support"])]
LEARNED = ["GEPA", "SkillOpt", "SkillAdam"]
THRESH = 7.3
TITLES = {"tau3-bench": r"$\tau^3$-bench: all learned skills within 7.3 points of the seed",
          "automationbench": "AutoBench: only SkillOpt on sales lies further, 13.3 points below the seed"}


def outliers(d):
    """Every (bench, domain, skill, learned - seed) with |difference| > 7.3, unrounded; plus the largest one below it."""
    out, below = [], []
    for bk, _, keys, _ in FIG_PANELS:
        for k in keys:
            for r in LEARNED:
                diff = d[bk][r][k]["q"] - d[bk]["Seed skill"][k]["q"]
                (out if abs(diff) > THRESH else below).append((bk, k, r, round(diff, 3)))
    return out, max(below, key=lambda t: abs(t[3]))


def draw(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.7), gridspec_kw=dict(width_ratios=[4, 6], wspace=0.07), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.755)
    offs = [(i - 2) * 0.13 for i in range(5)]
    for ax, (bk, disp, keys, names) in zip(axes, FIG_PANELS):
        for j, k in enumerate(keys):
            for r, o in zip(FIG_ROWS, offs):
                s = FIG_STYLE[r]
                ax.scatter([j + o], [d[bk][r][k]["q"]], s=13, marker=s["m"], facecolor=s["fc"], edgecolor=s["ec"], lw=0.8, zorder=3)
        for (b, k, r, diff) in out:
            if b == bk:
                j = keys.index(k)
                ax.annotate(f"{diff:+.1f}".replace("-", "−"), (j + offs[FIG_ROWS.index(r)], d[bk][r][k]["q"]), xytext=(0, -5), textcoords="offset points",
                            ha="center", va="top", fontsize=6.8, color=FIG_STYLE[r]["ec"])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 50, 100])
        ax.set_title(TITLES[bk], loc="left", fontsize=7.5, pad=4)
        ax.grid(axis="y", color="#E2E5EA", lw=0.4); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=2)
    axes[0].set_ylabel("test success (%)", labelpad=2)
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker=FIG_STYLE[r]["m"], mfc=FIG_STYLE[r]["fc"], mec=FIG_STYLE[r]["ec"], ms=4.2, mew=0.8) for r in FIG_ROWS]
    fig.legend(h, [FIG_LABEL.get(r, r) for r in FIG_ROWS], loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=5,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.4, handletextpad=0.3)
    save_figure(fig, "domains_wide")


def main():
    d = json.load(open(dataloader.CACHE / "domains.json"))
    out, nearest = outliers(d)
    assert len(out) == 1 and out[0][:3] == ("automationbench", "sales", "SkillOpt"), out
    assert round(out[0][3], 1) == -13.3, out   # the AutoBench title quotes it
    draw(d, out)
    nums = {bk: {k: {FIG_LABEL.get(r, r): round(d[bk][r][k]["q"], 1) for r in FIG_ROWS} for k in keys} for bk, _, keys, _ in FIG_PANELS}
    rec = dict(numbers=nums, beyond_7_3=out, largest_within=nearest, titles=TITLES)
    logger.DATA.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(logger.DATA / "mech_domains_wide_numbers.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
