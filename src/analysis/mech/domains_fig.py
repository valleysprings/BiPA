"""mech_domains: test avg@3 success per task domain for No skill, Seed, GEPA, SkillOpt, SkillAdam (tau3-bench and AutoBench panels).
One marker per row and domain, side by side inside each domain slot; the grey tick on banking_knowledge is the published best pass^1 of tau-Knowledge (25.52).
Reads assets/cache/analysis/mech/domains.json; no .venv needed.
  python3 -m src.analysis.mech.domains_fig"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import fig, logger, dataloader
ROWS = ["No skill", "Seed skill", "GEPA", "SkillOpt", "SkillAdam"]
LABEL = {"Seed skill": "Seed"}
STYLE = {"No skill": dict(m="o", fc="white", ec="0.5"), "Seed skill": dict(m="o", fc="k", ec="k"),
         "GEPA": dict(m="^", fc="#009E73", ec="#009E73"), "SkillOpt": dict(m="o", fc="#0072B2", ec="#0072B2"),
         "SkillAdam": dict(m="s", fc="#D55E00", ec="#D55E00")}
PANELS = [("tau3-bench", r"$\tau^3$-bench", ["airline", "retail", "telecom", "banking_knowledge"], ["airline", "retail", "telecom", "banking"]),
          ("automationbench", "AutoBench", ["finance", "hr", "marketing", "operations", "sales", "support"], ["finance", "HR", "marketing", "ops", "sales", "support"])]
PUBLISHED = 25.5


def draw():
    d = json.loads((dataloader.CACHE / "domains.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), gridspec_kw=dict(width_ratios=[4, 6], wspace=0.08), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.17, top=0.80)
    step = 0.15
    offs = [(i - 2) * step for i in range(5)]
    numbers = []
    for ax, (bk, disp, keys, names) in zip(axes, PANELS):
        for j, k in enumerate(keys):
            for r, o in zip(ROWS, offs):
                q = d[bk][r][k]["q"]
                s = STYLE[r]
                ax.scatter([j + o], [q], s=14, marker=s["m"], facecolor=s["fc"], edgecolor=s["ec"], lw=0.9, zorder=3)
                numbers.append((disp, k, r, q))
            if k == "banking_knowledge":
                ax.plot([j - 0.42, j + 0.42], [PUBLISHED] * 2, color="0.3", lw=1.3, zorder=2, solid_capstyle="butt")
                ax.text(j, PUBLISHED - 3, "published best", ha="center", va="top", fontsize=7, color="0.25")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(0, 105)
        ax.set_xlabel(disp)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("test success (%)")
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker=STYLE[r]["m"], mfc=STYLE[r]["fc"], mec=STYLE[r]["ec"], ms=4.5, mew=0.9) for r in ROWS]
    h.append(plt.Line2D([], [], color="0.3", lw=1.3))
    fig.legend(h, [LABEL.get(r, r) for r in ROWS] + [r"best published (25.5%)"], loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=6,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.save_figure(fig, "domains")
    heads = ["bench", "domain"] + [LABEL.get(r, r) for r in ROWS]
    tab = []
    for bk, disp, keys, _ in PANELS:
        for k in keys:
            tab.append([bk, k] + [f"{d[bk][r][k]['q']:.1f}" for r in ROWS])
    logger.write_record("domains_fig", "Test avg@3 success (%) per domain for No skill (hollow grey), Seed (black), GEPA, SkillOpt, SkillAdam; grey tick = best published tau-Knowledge pass^1 (25.52) on the banking domain.",
                   "Same runs as Table 1 (see mech_domains.md tables from mech.domains); DeepSeek-V4.1-Flash, OpenCode.",
                   "domains.json q: success among verified tasks per run, averaged over runs, per domain.",
                   logger.md_table(heads, tab), "python3 -m src.analysis.mech.domains_fig")
    return numbers


def draw_col():
    """Single-column variant (3.25 x 2.5 in): tau3-bench panel above AutoBench panel, shared legend on top in two rows."""
    d = json.loads((dataloader.CACHE / "domains.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.3))
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.075, top=0.835)
    offs = [(i - 2) * 0.15 for i in range(5)]
    for ax, (bk, disp, keys, names) in zip(axes, PANELS):
        for j, k in enumerate(keys):
            for r, o in zip(ROWS, offs):
                s = STYLE[r]
                ax.scatter([j + o], [d[bk][r][k]["q"]], s=12, marker=s["m"], facecolor=s["fc"], edgecolor=s["ec"], lw=0.8, zorder=3)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 50, 100])
        ax.text(0.01, 0.97, disp, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=2)
    fig.text(0.02, 0.455, "test success (%)", rotation=90, ha="center", va="center", fontsize=7)
    h = [plt.Line2D([], [], ls="", marker=STYLE[r]["m"], mfc=STYLE[r]["fc"], mec=STYLE[r]["ec"], ms=4.5, mew=0.9) for r in ROWS]
    # the published tau-Banking score is a setup check, not part of this figure's comparison; it stays in Appendix A only
    fig.legend(h, [LABEL.get(r, r) for r in ROWS], loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=3,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.0, handletextpad=0.3, labelspacing=0.25)
    fig.save_figure(fig, "domains_col")


if __name__ == "__main__":
    draw()
    draw_col()
