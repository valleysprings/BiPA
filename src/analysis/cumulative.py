"""Figure 5: input tokens a task has read up to each turn, by prompt component, seed against SkillOpt (DeepSeek).

Per task the cumulative tokens of each component group up to turn r (an ended episode keeps its total), averaged over
the task's test runs, then over tasks; turns up to the 95th percentile of the episode lengths of the seed and the three
optimizers. Writes <cache>/analysis/main/bipcum_main.json and fig/mech/mech_bipcum_main2.
"""
from collections import defaultdict

import numpy as np

from src.analysis.attribution import run_files
from src.graph import episode as W
from src.utils import dataloader as C
from src.utils.fig import save_figure

BENCHES = {"tau3-bench": "τ³-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
OPTS = ("gepa", "skillopt", "skilladam")
GROUPS = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
GROUP = {"system": "prompt", "tool_schemas": "prompt", "task": "prompt", "skill": "skill", "history": "messages"}
INK, MUTED, GRID, SKILLOPT = "#222222", "#5A5A5A", "#E2E5EA", "#1F6FB5"


def matrices(files):
    """[(task, R x 4 tokens per turn by group)] over the runs; one episode per task and run."""
    out = []
    for path in files.values():
        for e in {e.task: e for e in W.load(path)}.values():
            M = np.zeros((e.R, len(GROUPS)))
            for j, comp in enumerate(e.comps):
                M[:, GROUPS.index("tools" if comp.startswith("tool:") else GROUP[comp])] += e.Xrg[:, j]
            out.append((e.task, M))
    return out


def per_task(eps, rmax):
    """{task: run mean of the cumulative tokens up to rmax turns}, {task: run mean of the totals}."""
    cum, tot = defaultdict(list), defaultdict(list)
    for task, M in eps:
        c = np.cumsum(M, axis=0)
        tot[task].append(c[-1])
        c = c[:rmax]
        cum[task].append(np.vstack([c, np.repeat(c[-1:], rmax - len(c), axis=0)]))
    return {t: np.mean(v, axis=0) for t, v in cum.items()}, {t: np.mean(v, axis=0) for t, v in tot.items()}


def cell(bench):
    seed = matrices(run_files("skillopt", bench)[1])
    opts = {m: matrices(run_files(m, bench)[0]) for m in OPTS}
    rmax = int(np.ceil(np.percentile([M.shape[0] for _, M in seed] + [M.shape[0] for m in OPTS for _, M in opts[m]], 95)))
    (seed_cum, seed_tot), (own_cum, own_tot) = per_task(seed, rmax), per_task(opts["skillopt"], rmax)
    tasks = sorted(set(seed_cum) & set(own_cum))
    mean = lambda d: np.mean([d[t] for t in tasks], axis=0)  # noqa: E731
    return dict(rmax=rmax, seed=np.mean(list(seed_cum.values()), axis=0), learned=mean(own_cum), seed_same=mean(seed_cum),
                learned_total=mean(own_tot), seed_total=mean(seed_tot), n=len(tasks))


def numbers(c):
    extra = (c["learned_total"].sum() - c["seed_total"].sum()) / c["seed_total"].sum() * 100
    return dict(extra_pct=round(float(extra), 1), tasks=c["n"], seed_k=round(float(c["seed_total"].sum() / 1000), 1),
                skillopt_k=round(float(c["learned_total"].sum() / 1000), 1),
                extra_k_by_component={g: round(float((c["learned_total"][j] - c["seed_total"][j]) / 1000), 1) for j, g in enumerate(GROUPS)},
                printed_label="%+.0f%%" % extra, turns_shown=c["rmax"])


def draw(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator
    fs = 7.0
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    fig, axes = plt.subplots(2, 3, figsize=(219.08 / 72, 2.65), sharex="col")
    for ci, (bench, c) in enumerate(cells.items()):
        ymax = max(c["seed"].sum(1).max(), c["learned"].sum(1).max(), c["seed_same"].sum(1).max()) * 1.05 / 1000
        x = np.arange(1, c["rmax"] + 1)
        for ri, curve in enumerate((c["seed"], c["learned"])):
            ax = axes[ri, ci]
            bottom = np.zeros(c["rmax"])
            for j, g in enumerate(GROUPS):
                ax.fill_between(x, bottom, bottom + curve[:, j] / 1000.0, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                bottom = bottom + curve[:, j] / 1000.0
            if ri == 1:
                ax.step(x, c["seed_same"].sum(1) / 1000.0, where="mid", color=INK, lw=0.8, ls=(0, (3, 2)))
                ax.text(0.05, 0.95, numbers(c)["printed_label"], transform=ax.transAxes, fontsize=fs, va="top", ha="left",
                        color=SKILLOPT, fontweight="bold")
            ax.set_xlim(0.5, c["rmax"] + 0.5)
            ax.set_ylim(0, ymax)
            ticks = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, ymax) if 0 <= t <= ymax]
            unit = (lambda t: "%gM" % (t / 1000.0)) if max(ticks) >= 1000 else (lambda t: "%gk" % t)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter(["0" if t == 0 else unit(t) for t in ticks]))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5, pad=1.2, length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            if ri == 0:
                ax.set_title(BENCHES[bench], fontsize=fs + 0.5, fontweight="bold", loc="left", pad=2)
            else:
                ax.set_xlabel("turn", fontsize=fs, labelpad=1)
            if ci == 0:
                ax.set_ylabel(["Seed", "SkillOpt"][ri], fontsize=fs + 0.5, fontweight="bold", color=[INK, SKILLOPT][ri], labelpad=1)
    fig.text(0.006, 0.585, "input tokens read so far per task", rotation=90, va="center", ha="left", fontsize=fs, color=MUTED)
    handles = [Patch(color=COLOR[g], label=LABEL[g]) for g in GROUPS]
    handles.append(Line2D([0], [0], color=INK, lw=0.8, ls=(0, (3, 2)), label="seed, same tasks"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=0.8,
               handletextpad=0.4, labelspacing=0.25, bbox_to_anchor=(0.53, 0.0))
    fig.tight_layout(rect=(0.04, 0.145, 1, 1), w_pad=0.3, h_pad=0.4)
    save_figure(fig, "bipcum_main2")


def main():
    cells = {bench: cell(bench) for bench in BENCHES}
    C.write_json("bipcum_main", {bench: numbers(c) for bench, c in cells.items()})
    for bench, c in cells.items():
        print(bench, numbers(c))
    draw(cells)


if __name__ == "__main__":
    main()
