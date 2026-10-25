"""Draw the four-term heatmap and write mech_attribution3.md from attribution.json (written by the attribution compute step)."""
import json
import math

import numpy as np

from src.utils import dataloader, logger
from src.utils.fig import save_figure

TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
PREFIX = ["system", "tool_schemas", "task", "skill"]


def draw_heatmap(cells):
    """Heatmap in the style of appC_attribution: rows = skills grouped by optimizer (benchmark labels), columns = four terms + measured total, RdBu_r centred at 0."""
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    head = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]
    bshort = {"tau3-bench": r"$\tau^3$", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
    keys = TERMS + ["total"]
    M = np.array([[c["A"][k] for k in keys] for c in cells])
    methods = [c["method"] for c in cells]
    bounds = [i for i in range(1, len(cells)) if methods[i] != methods[i - 1]]
    groups = [(m, sum(i for i, x in enumerate(methods) if x == m) / methods.count(m)) for m in dict.fromkeys(methods)]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    fig.subplots_adjust(left=0.21, right=0.86, bottom=0.01, top=0.88)
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.62 * v else "k", fontweight="bold" if j == 4 else "normal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(head, fontsize=6.4, linespacing=0.95)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2)
    # second level of row labels: benchmark names right-aligned against the cells
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([bshort[c["bench"]] for c in cells], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cells), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y in [b - 0.5 for b in bounds]:
        ax.plot([-0.5, 4.5], [y, y], color="k", lw=0.9, clip_on=False)
    # first level: optimizer name once per group, bold, far left
    for name, yc in groups:
        ax.text(-0.33, yc, name, transform=ax.get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=7, fontweight="bold", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, ticks=[-0.2, 0, 0.2])
    cb.ax.set_yticklabels(["-0.2", "0", "+0.2"], fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("change in log input tokens", fontsize=7)
    save_figure(fig, "attribution3")


def f(x, nd=3):
    return f"{x:+.{nd}f}"


def record(cells):
    """OUT/data/mech/mech_attribution3.md."""
    rowsA = []
    for c in cells:
        a = c["A"]
        rowsA.append([c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f(a[k]) for k in TERMS] + [f(a["total"]), f(c["table1_log_rho_usable"]), f(a["residual_mean"]), f"{a['residual_max_abs']:.3f}", TERM_LABEL[a["largest"]]])
    rowsB = []
    for c in cells:
        for b in c["B"]:
            rowsB.append([c["method"], c["short"], b["run"]] + [f(b[k]) for k in TERMS] + [f(b["total"]), f(b["residual_mean"]), TERM_LABEL[b["largest"]]])
    rowsS = [[c["method"], c["short"], " / ".join(TERM_LABEL[b["largest"]] for b in c["B"]), "yes" if c["same_largest_all_runs"] else "no"] for c in cells]
    rowsD = [[c["method"], c["short"], c["n_table1"] - c["n"], "; ".join(f"{k}: {v}" for k, v in c["dropped"].items() if v) or "-"] for c in cells]
    rowsC = [[c["method"], c["short"]] + [f"{c['C'][p]['delta']:+.0f} (seed {c['C'][p]['seed']:.0f})" for p in PREFIX] for c in cells]
    rowsT = [[c["method"], c["short"], f"{c['table1_rho_full']:.3f}", f"{math.exp(c['table1_log_rho_usable']):.3f}", f"{math.exp(c['A']['total']):.3f}"] for c in cells]
    what = ("Four-term split of the change in log input tokens of each baseline skill against the seed, on the same data as Table 1: per task, R, N, X, F are averaged over the three test runs of each side "
            "before the split, so the total is Table 1's log rho_trj (mean over tasks of log(Xbar_skill / Xbar_seed)). Terms per task: more actions e_R dlogN, turn-splitting -e_R dlog(N/R), "
            "content per turn s_H dlogh, skill text s_F dlogF (midpoint shares, exact e_R = 1 + s_H R/(R-1)). The residual is total minus the sum of the four terms.")
    runs = "Skills: native-20261008-https (rep 0) + test-reps rep 1/2; seed: native-seed native-20261003-trace-final + rep-1, rep-2; DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1."
    agg = ("Tasks: Table 1's common task set restricted to tasks with a verified outcome and N, X in all three runs of both sides and a usable graph episode in all six runs (R>=2, an executed call, positive history). "
           "A) means over tasks of the per-task terms on 3-run means. B) the same terms on single run pairs (skill rep k vs seed rep k). C) mean over tasks of the per-request tokens of each fixed-prefix column, "
           "skill minus seed (per-task mean over the three runs).")
    table = ("### A. 3-run means (log units; n = tasks used / Table 1 common tasks)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "actions", "turn-split", "content", "skill text", "total", "Table 1 log rho (same tasks)", "resid mean", "resid max |.|", "largest"], rowsA)
             + "\n\n### B. Per run pair (skill rep k vs seed rep k)\n\n"
             + logger.md_table(["Skill", "Bench", "run", "actions", "turn-split", "content", "skill text", "total", "resid mean", "largest"], rowsB)
             + "\n\nStability of the largest term (runs 0 / 1 / 2):\n\n" + logger.md_table(["Skill", "Bench", "largest per run", "same in all three (and in A)"], rowsS)
             + "\n\n### C. Fixed-prefix columns, change per request in tokens (skill minus seed)\n\n" + logger.md_table(["Skill", "Bench", "system", "tool_schemas", "task", "skill"], rowsC)
             + "\n\n### Table 1 consistency (rho_trj)\n\n" + logger.md_table(["Skill", "Bench", "Table 1 rho (full common set)", "rho on used tasks (Table 1 rows)", "exp(total) of A"], rowsT)
             + "\n\n### Dropped tasks (Table 1 common set minus used)\n\n" + logger.md_table(["Skill", "Bench", "dropped", "reason"], rowsD))
    logger.write_record("attribution3", what, runs, agg, table,
                        "python -m src.analysis.attribution compute\npython -m aux.token_attribution.attribution_draw")


def main():
    cells = json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"]
    draw_heatmap(cells)
    record(cells)


if __name__ == "__main__":
    main()
