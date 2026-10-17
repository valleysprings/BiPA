"""mech_devtest: the dev gain each optimizer reported against the test difference over three runs (nine learned skills, three benchmarks).
Dumbbell chart: hollow marker = reported dev gain, filled marker = test avg@3 difference vs the seed, arrow dev -> test, grey tick = gain that the
selection rule alone reports under no effect (fitted prior), grey band = +/- detectable difference of a paired avg@3 comparison.
SkillOpt and GEPA report the returned skill's gain on the whole dev set (observed baseline for the tick); SkillAdam reports the sum of its
accepted patches' gains, each on that iteration's dev subset (10 tasks, all 9 on TB2); its rows have no tick, since its patches lose to their
parent more often than identical patches would (its selection-alone replay is in mech_selection.md).
Reads stats.json (_dev), selection.json and the two audit records; no .venv needed.
  python3 -m src.analysis.mech.devtest_fig"""
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.analysis.mech import common as M

BENCH = [("tau3-bench", r"$\tau^3$-bench", "tau3"), ("automationbench", "AutoBench", "AutoB"), ("terminal-bench-2-1", "TB2", "TB2")]
OPTS = ["GEPA", "SkillOpt"]          # optimizers that score the returned skill on the whole dev set
ROWS = ["GEPA", "SkillOpt", "SkillAdam"]
OC = {"SkillOpt": "#0072B2", "GEPA": "#009E73", "SkillAdam": "#D55E00"}
OM = {"SkillOpt": "o", "GEPA": "^", "SkillAdam": "s"}


def null_gain_tasks():
    """E[gain] under no effect, fitted prior, observed baseline, from mech_selection.md (tasks)."""
    out = {}
    for line in M.DATA.joinpath("mech_selection.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[1] == "fitted" and c[2] in OPTS:
            out[(c[0], c[2])] = float(c[7])
    return out


def mde_points():
    """MDE avg@3 vs avg@3 (pts) per benchmark from mech_stats.md section d."""
    out = {}
    for line in M.DATA.joinpath("mech_stats.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        found = re.match(r"[\d.]+ \(n=", c[7]) if len(c) == 9 else None
        if found and c[0] in ("tau3", "AutoB", "TB2"):
            out[c[0]] = float(re.match(r"[\d.]+", c[7]).group())
    return out


def draw():
    stats = json.loads((M.CACHE / "stats.json").read_text())
    dev = {(r["bench"], r["method"]): r for r in stats["_dev"]}
    null = null_gain_tasks()
    mde = mde_points()
    sel = json.loads((M.CACHE / "selection.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.155, top=0.82)
    step, gap = 0.68, 1.1
    ylabels, yticks, rows = [], [], []
    y = 0.0
    xlo, xhi = -14, 44
    for bkey, disp, short in BENCH:
        y0 = y
        ax.add_patch(Rectangle((-mde[short], y0 - 0.42), 2 * mde[short], 2.35, fc="0.5", alpha=0.15, lw=0, zorder=0))
        ax.text(xhi - 0.3, y0 - 0.38, disp, ha="right", va="center", fontweight="bold", fontsize=7)
        y = y0 + 0.2
        for o in ROWS:
            if o == "SkillAdam":
                # SkillAdam reports the summed gains of its accepted patches, each on that iteration's dev subset; its patches lose to
                # their parent more often than identical ones would, so its selection-alone replay (mech_selection.md) gets no tick here
                d = sel[bkey]["observed"]["SkillAdam"]["reported_gain_points"]
                t = stats[bkey]["methods"]["SkillAdam"]["success"]["dQ_table1"]
                nl = None
            else:
                r = dev[(bkey, o)]
                d, t = r["dev_gain_pts"], r["test_dQ_avg3"]
                nl = null[(bkey, o)] / r["dev_size"] * 100
            col, mk = OC[o], OM[o]
            if nl is not None:
                ax.plot([nl, nl], [y - 0.22, y + 0.22], color="0.35", lw=1.6, zorder=2, solid_capstyle="butt")
            ax.annotate("", xy=(t, y), xytext=(d, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=3.5, shrinkB=3.5, mutation_scale=6), zorder=3)
            ax.scatter([d], [y], s=26, marker=mk, facecolor="white", edgecolor=col, lw=1.0, zorder=4)
            ax.scatter([t], [y], s=26, marker=mk, color=col, lw=0, zorder=5)
            rows.append((disp, o, d, t, nl, mde[short]))
            yticks.append(y)
            ylabels.append(o)
            y += step
        y = y0 + 3 * step + gap - 0.6
    ax.axvline(0, color="k", lw=0.6, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(y - gap + 0.35, -0.65)
    ax.set_xlabel("success difference to the seed (points)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.25", ms=4.5),
         plt.Line2D([], [], marker="o", ls="", color="0.25", ms=4.5),
         plt.Line2D([], [], color="0.35", lw=1.6),
         Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.15, lw=0)]
    lab = ["reported dev gain", "test difference (3 runs)", "dev gain from selection alone", "detectable difference"]
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=2, frameon=False, fontsize=7, handlelength=1.3,
               columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)
    M.save_figure(fig, "devtest")
    table = M.md_table(["bench", "optimizer", "dev gain (pts)", "test difference (pts)", "selection-alone dev gain (pts)", "detectable difference (pts)"],
                       [[a, b, f"{c:+.1f}", f"{d:+.1f}", "-" if e is None else f"{e:+.1f}", f"{f:.1f}"] for a, b, c, d, e, f in
                        [(r[0].replace("$\\tau^3$", "tau3"), *r[1:]) for r in rows]])
    M.write_record("devtest", "Reported dev gain (hollow) against the test avg@3 difference vs the seed (filled) for the nine learned skills; SkillOpt and GEPA report the returned skill's gain on the whole dev set, SkillAdam the sum of its accepted patches' gains, each on that iteration's dev subset (10 tasks; all 9 on TB2). Grey tick = expected dev gain when selection alone acts (fitted prior, observed baseline, tasks / dev size; SkillOpt and GEPA only, SkillAdam's replay is in mech_selection.md); band = +/- detectable difference of a paired avg@3 comparison.",
                   "Table 1 runs (native-20261008-https) and test reps, DeepSeek-V4.1-Flash; dev sizes 30 / 30 / 9.",
                   "stats.json key _dev (dev_gain_pts, test_dQ_avg3); E[gain] column of mech_selection.md divided by dev size; selection.json observed.SkillAdam.reported_gain_points; MDE avg@3 vs avg@3 of mech_stats.md section d.",
                   table, "python3 -m src.analysis.mech.devtest_fig")
    return rows


if __name__ == "__main__":
    for r in draw():
        print(r)
