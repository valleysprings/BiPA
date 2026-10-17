"""mech_paired: appendix figure replacing the table "The token tax under other aggregations".
For each of the nine learned skills (GEPA, SkillOpt, SkillAdam x three benchmarks) one row on a log axis of input-token ratio to the seed skill (line at 1):
rho (per-task geometric mean, filled marker), ratio of total input tokens (hollow), rho on tasks both sides solve in all runs (diamond), rho on tasks both
fail in all runs (x, with its task count when n < 5). The share of tasks above 1 is a text column in the right margin.
Data: assets/cache/analysis/mech/stats.json (written by analysis.mech.stats).
  python3 -m src.analysis.mech.paired_fig"""
import json
import numpy as np
from src.analysis.mech import common as M

BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
METHODS = ["GEPA", "SkillOpt", "SkillAdam"]
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
MARK = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}


def collect():
    """One dict per (benchmark, method) with the plotted numbers."""
    stats = json.loads((M.CACHE / "stats.json").read_text())
    rows = []
    for bench, disp in BENCH:
        for name in METHODS:
            m = stats[bench]["methods"][name]
            x = m["ratios"]["X"]
            sv = m["survivor"]
            rows.append({"bench": disp, "method": name, "n": x["n"], "rho": x["rho_geo"], "total": x["rho_sum"], "share": x["share_gt1"],
                         "solve": sv["both_solve_all"]["X"]["rho_geo"], "n_solve": sv["both_solve_all"]["X"]["n"],
                         "fail": sv["both_fail_all"]["X"]["rho_geo"], "n_fail": sv["both_fail_all"]["X"]["n"]})
    return rows


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    rows = collect()
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    fig.subplots_adjust(left=0.17, right=0.87, bottom=0.12, top=0.88)
    XMAX = 1.62
    ys, y = [], 0.0
    for g in range(3):
        y += 1.0
        for _ in METHODS:
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
        c, mk = OC[r["method"]], MARK[r["method"]]
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
    for g, (_, disp) in enumerate(BENCH):
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


def main():
    fig, rows = draw()
    M.save_figure(fig, "paired")
    head = ["Bench", "Method", "n", "rho (geo)", "total tokens", "both solve (n)", "both fail (n)", "share > 1"]
    table = M.md_table(head, [[r["bench"], r["method"], r["n"], f"{r['rho']:.3f}", f"{r['total']:.3f}", f"{r['solve']:.3f} ({r['n_solve']})",
                                f"{r['fail']:.3f} ({r['n_fail']})", f"{100 * r['share']:.0f}%"] for r in rows])
    M.write_record("paired", "Input-token ratio to the seed skill of the nine learned skills under four aggregations (geometric mean of per-task ratios, ratio of totals, "
                   "tasks both solve in every run, tasks both fail in every run) and the share of tasks above 1. Appendix figure, 4.5 x 3.3 in; both-fail values beyond 1.6 are drawn as arrows with their value.",
                   "Same runs and task sets as Table 1 (see mech_stats).", "As in mech_stats section 5a and 5c; the x marker is annotated with its task count when n < 5.",
                   table, "python3 -m src.analysis.mech.paired_fig")
    return rows


if __name__ == "__main__":
    for r in main():
        print(r)
