"""Figure bip_cum: the input tokens a task has read so far, turn by turn and component by component, for the seed and for each learned skill.

Per task, the cumulative tokens read from each component up to turn r (an episode that has ended keeps its total), averaged over the task's runs, then over tasks.  Row 1: the seed (all its tasks).
Rows 2-4: GEPA, SkillOpt, SkillAdam, over the tasks that have runs under both the seed and the optimizer; the dashed line is the seed's total over the same tasks.  The label gives the total
extra input tokens of the learned skill relative to the seed's total (full episodes, not cut at the last turn shown).  Turns shown: up to the 95th percentile of the episode length.
Columns: DeepSeek tau3 / AutoBench / TB2, Luna tau3 / AutoBench (Luna TB2: every optimizer returned the seed).

Run:  python -m aux.token_attribution.fig_cumabs
"""
import json
import pickle
from collections import defaultdict

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403

D = None
STACK = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
OPTS = [("gepa", "GEPA", "#2A9D4B"), ("skillopt", "SkillOpt", "#1F6FB5"), ("skilladam", "SkillAdam", "#E08A1E")]
COLS = [("deepseek-flash", "tau3-bench"), ("deepseek-flash", "automationbench"), ("deepseek-flash", "terminal-bench-2-1"),
        ("gpt-6-luna", "tau3-bench"), ("gpt-6-luna", "automationbench")]


def cum_by_task(eps, rmax):
    per = defaultdict(list)
    for e in eps:
        C = np.cumsum(e["M"][:rmax], axis=0)
        if C.shape[0] < rmax:
            C = np.vstack([C, np.repeat(C[-1:], rmax - C.shape[0], axis=0)])
        per[e["task"]].append(C)
    return {t: np.mean(v, axis=0) for t, v in per.items()}


def cell(model, bench):
    seed = D[(model, bench, "seed")]
    out = dict(seed=None, opts={})
    allR = [e["R"] for e in seed] + [e["R"] for o, _, _ in OPTS for e in D.get((model, bench, o), [])]
    rmax = int(np.ceil(np.percentile(allR, 95)))
    cs = cum_by_task(seed, rmax)
    fs = cum_by_task(seed, 10 ** 3)
    out["rmax"] = rmax
    out["seed"] = np.mean(list(cs.values()), axis=0)
    out["seed_total"] = np.mean([v[-1] for v in fs.values()], axis=0)
    out["seed_tasks"] = len(cs)
    for opt, _, _ in OPTS:
        L = D.get((model, bench, opt))
        if not L:
            continue
        cl, fl = cum_by_task(L, rmax), cum_by_task(L, 10 ** 3)
        tasks = sorted(set(cs) & set(cl))
        out["opts"][opt] = dict(
            learned=np.mean([cl[t] for t in tasks], axis=0), seed=np.mean([cs[t] for t in tasks], axis=0), n=len(tasks),
            learned_total=np.mean([fl[t][-1] for t in tasks], axis=0), seed_total=np.mean([fs[t][-1] for t in tasks], axis=0))
    return out


def load():
    """Read the inputs this figure draws from."""
    global D
    D = pickle.loads((OUT / "bipdata.pkl").read_bytes())["episodes"]


def main():
    load()
    plt = setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.1, 6.3))
    gs = GridSpec(4, 6, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1], left=0.1, right=0.99, top=0.9, bottom=0.095, hspace=0.18, wspace=0.36)
    gcol = [0, 1, 2, 4, 5]
    cells = [cell(m, b) for m, b in COLS]
    ymax = []
    for c in cells:
        m = c["seed"].sum(1).max()
        for o in c["opts"].values():
            m = max(m, o["learned"].sum(1).max(), o["seed"].sum(1).max())
        ymax.append(m * 1.06 / 1000)
    res = {}
    for ci, ((model, bench), c) in enumerate(zip(COLS, cells)):
        for ri in range(4):
            ax = fig.add_subplot(gs[ri, gcol[ci]])
            r = c["rmax"]
            x = np.arange(1, r + 1)
            if ri == 0:
                curve, ref, name, col = c["seed"], None, "Seed", INK
            else:
                opt, name, col = OPTS[ri - 1]
                if opt not in c["opts"]:
                    ax.axis("off")
                    if ri == 1:
                        ax.text(0.5, 0.5, "every optimizer\nreturned the seed", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                o = c["opts"][opt]
                curve, ref = o["learned"], o["seed"]
            bottom = np.zeros(r)
            for j, g in enumerate(STACK):
                v = curve[:, j] / 1000.0
                ax.fill_between(x, bottom, bottom + v, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                bottom = bottom + v
            if ref is not None:
                ax.step(x, ref.sum(1) / 1000.0, where="mid", color=INK, lw=0.9, ls=(0, (3, 2)))
                o = c["opts"][OPTS[ri - 1][0]]
                extra = (o["learned_total"].sum() - o["seed_total"].sum()) / o["seed_total"].sum() * 100
                ax.text(0.04, 0.93, "%+.0f%% tokens" % extra, transform=ax.transAxes, fontsize=6.4, va="top", ha="left", color=OPTS[ri - 1][2], fontweight="bold")
                res["%s|%s|%s" % (model, bench, OPTS[ri - 1][0])] = dict(
                    turns_shown=int(r), tasks=int(o["n"]), seed_tokens_per_task_k=round(float(o["seed_total"].sum() / 1000), 1),
                    learned_tokens_per_task_k=round(float(o["learned_total"].sum() / 1000), 1), extra_pct=round(float(extra), 1),
                    extra_k_by_component={g: round(float((o["learned_total"][j] - o["seed_total"][j]) / 1000), 1) for j, g in enumerate(STACK)})
            else:
                res["%s|%s|seed" % (model, bench)] = dict(turns_shown=int(r), tasks=int(c["seed_tasks"]), seed_tokens_per_task_k=round(float(c["seed_total"].sum() / 1000), 1),
                                                           by_component_k={g: round(float(c["seed_total"][j] / 1000), 1) for j, g in enumerate(STACK)})
            ax.set_xlim(0.5, r + 0.5)
            ax.set_ylim(0, ymax[ci])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.3, pad=1.5)
            if ri < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("turn", fontsize=7, labelpad=1)
            if ci in (0, 3):
                ax.set_ylabel(name, fontsize=8, fontweight="bold", color=col, labelpad=2)
            if ri == 0:
                ax.set_title(BENCH_NAME[bench], fontsize=8, fontweight="bold", loc="left", pad=3)
    fig.text(0.1, 0.955, "DeepSeek-V4.1-Flash", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.74, 0.955, "GPT-6-Luna", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.012, 0.5, "input tokens read so far per task, thousands", rotation=90, va="center", fontsize=7, color=MUTED)
    handles = [Patch(color=COLOR[g], label=LABEL[g]) for g in STACK] + [Line2D([0], [0], color=INK, lw=0.9, ls=(0, (3, 2)), label="seed, same tasks")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0), handlelength=1.4, columnspacing=1.4)
    save(fig, "bip_cum")
    json.dump(res, open(OUT / "bip_cum_numbers.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
