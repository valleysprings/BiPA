"""Figure cummain (main text, one column): the input tokens a task has read so far, seed and SkillOpt, DeepSeek-V4.1-Flash, three benchmarks.
Same quantity and construction as fig_cumabs.py (per task cumulative tokens by component, carried forward after the episode ends, averaged over tasks; SkillOpt over the tasks it shares with the seed;
dashed line: the seed's total over the same tasks; label: SkillOpt's total input tokens relative to the seed's, full episodes).
Run:  python -m aux.token_attribution.fig_cum_main
"""
import json

import numpy as np

from aux.token_attribution import fig_cumabs as M
from aux.token_attribution.bipcommon import setup_mpl, save, INK, MUTED, GRID, BENCH_NAME

COLS = [("deepseek-flash", "tau3-bench"), ("deepseek-flash", "automationbench"), ("deepseek-flash", "terminal-bench-2-1")]


def main():
    M.load()
    plt = setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(3.35, 2.75), sharex="col")
    cells = [M.cell(m, b) for m, b in COLS]
    res = {}
    for ci, ((model, bench), c) in enumerate(zip(COLS, cells)):
        o = c["opts"]["skillopt"]
        ymax = max(c["seed"].sum(1).max(), o["learned"].sum(1).max(), o["seed"].sum(1).max()) * 1.05 / 1000
        for ri in range(2):
            ax = axes[ri, ci]
            r = c["rmax"]
            x = np.arange(1, r + 1)
            curve = c["seed"] if ri == 0 else o["learned"]
            bottom = np.zeros(r)
            for j, g in enumerate(M.STACK):
                v = curve[:, j] / 1000.0
                ax.fill_between(x, bottom, bottom + v, step="mid", color=M.COLOR[g], lw=0, alpha=0.95)
                bottom = bottom + v
            if ri == 1:
                ax.step(x, o["seed"].sum(1) / 1000.0, where="mid", color=INK, lw=0.8, ls=(0, (3, 2)))
                extra = (o["learned_total"].sum() - o["seed_total"].sum()) / o["seed_total"].sum() * 100
                ax.text(0.05, 0.93, "%+.0f%%" % extra, transform=ax.transAxes, fontsize=6.6, va="top", ha="left", color="#1F6FB5", fontweight="bold")
                res[bench] = dict(extra_pct=round(float(extra), 1), tasks=int(o["n"]), seed_k=round(float(o["seed_total"].sum() / 1000), 1), skillopt_k=round(float(o["learned_total"].sum() / 1000), 1),
                                  extra_k_by_component={g: round(float((o["learned_total"][j] - o["seed_total"][j]) / 1000), 1) for j, g in enumerate(M.STACK)})
            ax.set_xlim(0.5, r + 0.5); ax.set_ylim(0, ymax)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
            ax.tick_params(labelsize=5.8, pad=1.2, length=2)
            if ri == 0:
                ax.set_title(BENCH_NAME[bench], fontsize=7, fontweight="bold", loc="left", pad=2)
            if ri == 1:
                ax.set_xlabel("turn", fontsize=6.5, labelpad=1)
            if ci == 0:
                ax.set_ylabel(["Seed", "SkillOpt"][ri], fontsize=7, fontweight="bold", color=[INK, "#1F6FB5"][ri], labelpad=1)
    fig.text(0.004, 0.55, "input tokens read so far per task, thousands", rotation=90, va="center", fontsize=5.8, color=MUTED)
    handles = [Patch(color=M.COLOR[g], label=M.LABEL[g]) for g in M.STACK] + [Line2D([0], [0], color=INK, lw=0.8, ls=(0, (3, 2)), label="seed, same tasks")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.54, 0.0))
    fig.tight_layout(rect=(0.03, 0.12, 1, 1), w_pad=0.5, h_pad=0.5)
    save(fig, "bip_cum_main")
    json.dump(res, open(M.OUT / "bip_cum_main_numbers.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
