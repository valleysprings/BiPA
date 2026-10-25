"""The devtest and paired figures from stats.json, selection.json and mech_selection.md.

  python -m aux.noise_and_domains.stats_figs devtest   -> fig/mech/mech_devtest.{pdf,svg}, data/mech/mech_devtest.md
  python -m aux.noise_and_domains.stats_figs paired    -> fig/mech/mech_paired.{pdf,svg}, data/mech/mech_paired.md
"""
import json
import sys

from src.utils import dataloader, logger
from src.utils.fig import save_figure

FIG_OC = {"SkillOpt": "#0072B2", "GEPA": "#009E73", "SkillAdam": "#D55E00"}
FIG_MARK = {"SkillOpt": "o", "GEPA": "^", "SkillAdam": "s"}
DEVTEST_BENCH = [("tau3-bench", r"$\tau^3$-bench", "tau3"), ("automationbench", "AutoBench", "AutoB"), ("terminal-bench-2-1", "TB2", "TB2")]
DEVTEST_OPTS = ["GEPA", "SkillOpt"]          # optimizers that score the returned skill on the whole dev set
DEVTEST_ROWS = ["GEPA", "SkillOpt", "SkillAdam"]
PAIRED_BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
PAIRED_METHODS = ["GEPA", "SkillOpt", "SkillAdam"]


def null_gain_tasks():
    """E[gain] under no effect, fitted prior, observed baseline, from mech_selection.md (tasks)."""
    out = {}
    for line in logger.DATA.joinpath("mech_selection.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[1] == "fitted" and c[2] in DEVTEST_OPTS:
            out[(c[0], c[2])] = float(c[7])
    return out


def devtest():
    """mech_devtest: the dev gain each optimizer reported against the test difference over three runs (nine learned skills, three
    benchmarks).  One row per skill: hollow marker = reported dev gain, filled marker = test avg@3 difference vs the seed with its 95% paired
    bootstrap interval over test tasks (section b, dQ_ci), dotted line joining the two, grey tick = expected dev gain the selection rule
    reports from the seed draw the run started at when every candidate is identical to the seed (fitted task model).  SkillOpt and GEPA
    report the returned skill's gain on the whole dev set; SkillAdam reports the sum of its accepted patches' gains, each on that iteration's
    dev subset (10 tasks, all 9 on TB2); its rows have no tick (its summed replay is in mech_selection.md).  No significance band: a
    difference is judged by its interval.  Reads stats.json (_dev and section b), selection.json and mech_selection.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = json.loads((dataloader.CACHE / "stats.json").read_text())
    dev = {(r["bench"], r["method"]): r for r in stats["_dev"]}
    null = null_gain_tasks()
    sel = json.loads((dataloader.CACHE / "selection.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.155, top=0.84)
    step, gap = 0.68, 1.1
    ylabels, yticks, rows = [], [], []
    y = 0.0
    xlo, xhi = -25, 44
    for bkey, disp, short in DEVTEST_BENCH:
        y0 = y
        ax.text(xhi - 0.3, y0 - 0.38, disp, ha="right", va="center", fontweight="bold", fontsize=7)
        y = y0 + 0.2
        for o in DEVTEST_ROWS:
            success = stats[bkey]["methods"][o]["success"]
            lo, hi = success["dQ_ci"]
            if o == "SkillAdam":
                # summed gains of its accepted patches, each on that iteration's dev subset; no tick (replay in mech_selection.md)
                d = sel[bkey]["observed"]["SkillAdam"]["reported_gain_points"]
                t = success["dQ_table1"]
                nl = None
            else:
                r = dev[(bkey, o)]
                d, t = r["dev_gain_pts"], r["test_dQ_avg3"]
                nl = null[(bkey, o)] / r["dev_size"] * 100
            col, mk = FIG_OC[o], FIG_MARK[o]
            if nl is not None:
                ax.plot([nl, nl], [y - 0.22, y + 0.22], color="0.35", lw=1.6, zorder=2, solid_capstyle="butt")
            ax.plot([t, d], [y, y], color=col, lw=0.7, ls=(0, (1, 1.5)), alpha=0.8, zorder=1)
            ax.plot([lo, hi], [y, y], color=col, lw=1.3, zorder=3, solid_capstyle="butt")
            for xe in (lo, hi):
                ax.plot([xe, xe], [y - 0.11, y + 0.11], color=col, lw=1.0, zorder=3)
            ax.scatter([d], [y], s=26, marker=mk, facecolor="white", edgecolor=col, lw=1.0, zorder=4)
            ax.scatter([t], [y], s=26, marker=mk, color=col, edgecolor="white", lw=0.6, zorder=5)
            rows.append((disp, o, d, t, lo, hi, nl))
            yticks.append(y)
            ylabels.append(o)
            y += step
        y = y0 + 3 * step + gap - 0.6
    ax.axvline(0, color="k", lw=0.6, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(y - gap + 0.35, -0.65)
    ax.set_xlabel("success difference to the seed (points)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.25", ms=4.5),
         plt.Line2D([], [], marker="o", ls="-", color="0.25", ms=4.5, lw=1.3),
         plt.Line2D([], [], color="0.35", lw=1.6)]
    lab = ["reported dev gain", "test difference, 95% interval", "dev gain from selection alone"]
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=2, frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)
    save_figure(fig, "devtest")
    table = logger.md_table(["bench", "optimizer", "dev gain (pts)", "test difference (pts)", "95% interval (pts)", "selection-alone dev gain (pts)"],
                       [[a, b, f"{c:+.1f}", f"{d:+.1f}", f"[{lo:+.1f}, {hi:+.1f}]", "-" if e is None else f"{e:+.1f}"] for a, b, c, d, lo, hi, e in
                        [(r[0].replace("$\\tau^3$", "tau3"), *r[1:]) for r in rows]])
    logger.write_record("devtest", "Reported dev gain (hollow) against the test avg@3 difference vs the seed (filled) with its 95% paired bootstrap interval over test tasks, for the nine learned skills; SkillOpt and GEPA report the returned skill's gain on the whole dev set, SkillAdam the sum of its accepted patches' gains, each on that iteration's dev subset (10 tasks; all 9 on TB2). Grey tick = expected dev gain the selection rule reports from the seed draw the run started at when every candidate is identical to the seed (fitted task model, tasks / dev size; SkillOpt and GEPA only, SkillAdam's replay is in mech_selection.md). No significance band.",
                   "Table 1 runs (native-20261008-https) and test reps, DeepSeek-V4.1-Flash; dev sizes 30 / 30 / 9.",
                   "stats.json key _dev (dev_gain_pts, test_dQ_avg3) and section b dQ_ci (10,000 paired bootstrap resamples over test tasks, seed 20261012); E[gain] column of mech_selection.md divided by dev size; selection.json observed.SkillAdam.reported_gain_points.",
                   table, "python -m aux.noise_and_domains.stats_figs devtest")
    return rows


def paired_collect():
    """One dict per (benchmark, method) with the plotted numbers."""
    stats = json.loads((dataloader.CACHE / "stats.json").read_text())
    rows = []
    for bench, disp in PAIRED_BENCH:
        for name in PAIRED_METHODS:
            m = stats[bench]["methods"][name]
            x = m["ratios"]["X"]
            sv = m["survivor"]
            rows.append({"bench": disp, "method": name, "n": x["n"], "rho": x["rho_geo"], "total": x["rho_sum"], "share": x["share_gt1"],
                         "solve": sv["both_solve_all"]["X"]["rho_geo"], "n_solve": sv["both_solve_all"]["X"]["n"],
                         "fail": sv["both_fail_all"]["X"]["rho_geo"], "n_fail": sv["both_fail_all"]["X"]["n"]})
    return rows


def paired_draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    rows = paired_collect()
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    fig.subplots_adjust(left=0.17, right=0.87, bottom=0.12, top=0.88)
    XMAX = 1.62
    ys, y = [], 0.0
    for g in range(3):
        y += 1.0
        for _ in PAIRED_METHODS:
            ys.append(y)
            y += 1.0
    header_y = [ys[3 * g] - 1.0 for g in range(3)]
    ax.set_xlim(0.97, XMAX)
    ax.set_ylim(y - 0.4, -0.2)
    ax.axvline(1.0, color="0.15", lw=1.4, zorder=1)
    ax.text(1.0, -0.35, "seed = 1", ha="center", va="bottom", fontsize=7, fontweight="bold", color="0.15")
    for yy in header_y:
        ax.axhspan(yy - 0.45, yy + 0.45, color="0.93", zorder=0, lw=0)
    for r, yy in zip(rows, ys):
        c, mk = FIG_OC[r["method"]], FIG_MARK[r["method"]]
        ax.plot([1.0, r["rho"]], [yy, yy], color=c, lw=0.9, alpha=0.35, zorder=2)
        ax.scatter(r["rho"], yy - 0.1, s=34, marker=mk, color=c, lw=0, zorder=5)
        ax.scatter(r["total"], yy - 0.1, s=34, marker=mk, facecolor="white", edgecolor=c, lw=1.0, zorder=6)
        ax.scatter(r["solve"], yy + 0.25, s=12, marker="D", color="0.35", lw=0, zorder=4)
        if r["fail"] <= XMAX - 0.02:
            ax.scatter(r["fail"], yy + 0.25, s=18, marker="x", color="0.35", lw=0.9, zorder=4)
            if r["n_fail"] < 5:
                ax.text(r["fail"] + 0.012, yy + 0.25, f"n={r['n_fail']}", fontsize=7, color="0.35", va="center", ha="left")
        else:
            ax.annotate("", xy=(XMAX - 0.005, yy + 0.25), xytext=(XMAX - 0.07, yy + 0.25),
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9, mutation_scale=7), zorder=4)
            ax.text(XMAX - 0.075, yy + 0.25, f"{r['fail']:.2f} (n={r['n_fail']})", fontsize=7, color="0.35", va="center", ha="right")
        ax.text(1.02, yy, f"{100 * r['share']:.0f}%", transform=ax.get_yaxis_transform(), fontsize=7, va="center", ha="left")
    for g, (_, disp) in enumerate(PAIRED_BENCH):
        ax.text(-0.02, header_y[g], disp, transform=ax.get_yaxis_transform(), fontsize=7, fontweight="bold", ha="right", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["method"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    ax.set_xlabel("input-token ratio to the seed skill")
    ax.text(1.02, 1.005, "share $>1$", transform=ax.transAxes, fontsize=7, va="bottom", ha="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color="0.3", ms=4, label=r"$\rho$ (per-task geometric mean)"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4, label="total tokens"),
               plt.Line2D([], [], marker="D", ls="", color="0.25", ms=3.5, label="both solve (lower)"),
               plt.Line2D([], [], marker="x", ls="", color="0.25", ms=4, label="both fail (lower)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), frameon=False, ncol=4, fontsize=7, handletextpad=0.1, columnspacing=0.8)
    return fig, rows


def paired():
    """mech_paired: appendix figure replacing the table "The token tax under other aggregations".  For each of the nine learned skills
    (GEPA, SkillOpt, SkillAdam x three benchmarks) one row on a log axis of input-token ratio to the seed skill (line at 1): rho (per-task
    geometric mean, filled marker), ratio of total input tokens (hollow), rho on tasks both sides solve in all runs (diamond), rho on tasks
    both fail in all runs (x, with its task count when n < 5).  The share of tasks above 1 is a text column in the right margin."""
    fig, rows = paired_draw()
    save_figure(fig, "paired")
    head = ["Bench", "Method", "n", "rho (geo)", "total tokens", "both solve (n)", "both fail (n)", "share > 1"]
    table = logger.md_table(head, [[r["bench"], r["method"], r["n"], f"{r['rho']:.3f}", f"{r['total']:.3f}", f"{r['solve']:.3f} ({r['n_solve']})",
                                f"{r['fail']:.3f} ({r['n_fail']})", f"{100 * r['share']:.0f}%"] for r in rows])
    logger.write_record("paired", "Input-token ratio to the seed skill of the nine learned skills under four aggregations (geometric mean of per-task ratios, ratio of totals, "
                   "tasks both solve in every run, tasks both fail in every run) and the share of tasks above 1. Size 4.5 x 3.3 in; both-fail values beyond 1.6 are drawn as arrows with their value.",
                   "Same runs and task sets as Table 1 (see mech_stats).", "As in mech_stats section 5a and 5c; the x marker is annotated with its task count when n < 5.",
                   table, "python -m aux.noise_and_domains.stats_figs paired")
    return rows


if __name__ == "__main__":
    if sys.argv[1:] not in (["devtest"], ["paired"]):
        sys.exit("usage: python -m aux.noise_and_domains.stats_figs devtest|paired")
    {"devtest": devtest, "paired": paired}[sys.argv[1]]()
