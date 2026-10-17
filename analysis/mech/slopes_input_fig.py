"""mech_slopes_input: first row of mech_slopes for the methodology section (single column, 3.25 x 1.35 in).
Per benchmark, each test episode's input tokens against its turns, both relative to the task's geometric mean (log-log), the fitted within-task slope and the dashed k = 1 line.
Data: assets/cache/analysis/mech/slopes.json written by analysis.mech.slopes compute (loaded here, not recomputed).
  python3 -m analysis.mech.slopes_input_fig"""
import json
import numpy as np
from analysis.mech import common as M

BENCH = [("tau3-bench", r"$\tau^3$-bench", "#0072B2"), ("automationbench", "AutoBench", "#D55E00"), ("terminal-bench-2-1", "TB2", "#009E73")]


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
    c = json.loads((M.CACHE / "slopes.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(3.25, 1.35), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.88, wspace=0.3)
    rng = np.random.default_rng(0)
    xx = np.exp(np.array([-0.6, 0.6]))
    reg = {b: c["reg"][b]["input"] for b, _, _ in BENCH}
    pts = {b: (np.array(c["pts"][b]["input"][0]), np.array(c["pts"][b]["input"][1])) for b, _, _ in BENCH}
    # y range from the bulk of the episodes (0.5th to 99.5th percentile) and the fitted lines, so a few outlying episodes do not leave half the axes empty
    data_y = np.concatenate([pts[b][1] for b in pts])
    line_y = np.concatenate([np.log(xx) * max(reg[b]["slope"], 1.0) for b in pts])
    lo = np.exp(min(np.percentile(data_y, 0.5), line_y.min()))
    hi = np.exp(max(np.percentile(data_y, 99.5), line_y.max()))
    span = hi / lo
    lo, hi = lo / span ** 0.06, hi * span ** 0.06
    ticks = [t for t in (0.5, 1, 2, 4) if lo <= t <= hi]
    for ax, (b, disp, col) in zip(axs, BENCH):
        xs, ys = pts[b]
        sel = rng.choice(len(xs), min(len(xs), 250), replace=False)
        ax.scatter(np.exp(xs[sel]), np.exp(ys[sel]), s=2, color=col, alpha=0.3, lw=0, rasterized=True)
        ax.plot(xx, xx, color="k", ls="--", lw=0.6)
        ax.plot(xx, xx ** reg[b]["slope"], color=col, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.exp(-0.7), np.exp(0.7))
        ax.set_ylim(lo, hi)
        ax.set_xticks([0.5, 1, 2])
        ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_yticks(ticks)
        ax.minorticks_off()
        ax.set_title(disp, fontweight="bold", pad=2)
        ax.text(0.05, 0.95, f"slope {reg[b]['slope']:.2f}", transform=ax.transAxes, ha="left", va="top", color=col)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_yticklabels([f"{t:g}" for t in ticks])
    axs[0].set_ylabel("input tokens / task mean")
    axs[1].set_xlabel("turns / task mean")
    M.save_figure(fig, "slopes_input")
    rows = [[d, f"{reg[b]['slope']:.3f}", f"{reg[b]['se']:.3f}", reg[b]["n"], reg[b]["n_tasks"]] for b, d, _ in BENCH]
    M.write_record("slopes_input", "First row of mech_slopes (input tokens vs. turns, both relative to the task's mean, log-log) as a 3.25 x 1.35 in single-column figure; dashed line k = 1.",
                   "As mech_slopes (seed run-0 and GEPA/SkillOpt/SkillAdam test episodes, DeepSeek-V4.1-Flash).", "Within-task fixed-effects slope, as mech_slopes; 250 points per panel are drawn at random (seed 0).",
                   M.md_table(["Benchmark", "slope k", "SE", "episodes", "tasks"], rows), "python3 -m analysis.mech.slopes_input_fig")
    print(rows)


if __name__ == "__main__":
    draw()
