"""The priorfit figure: how often the seed solves the tasks of each benchmark, counted and as the fitted Beta prior expects.

Per benchmark, the share of tasks the seed solves in none of its runs, in some but not all, and in all of them, counted on the dev tasks (solid; TB2
has two runs per dev task, the others three) and on the test tasks (hatched; three runs each), and the share the prior fitted to dev and test tasks
together expects (diamonds; beta-binomial, taking each task's own number of runs). The executions are those of the selection task model
(selection.seed_executions).
Run:  python -m aux.selection_replay.fig_priorfit   -> fig/mech/mech_priorfit.{pdf,svg}, data/mech/mech_priorfit_numbers.json
"""
import json, math

from aux.selection_replay import selection as S
from aux.selection_replay.style import BENCH_NAME, GRID, INK, MUTED, setup_mpl
from src.utils import logger
from src.utils.fig import save_figure

BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
CATS = ["none", "some", "all"]


def bb(k, n, a, b):
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
                    - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))


def cat(v):
    k, n = sum(v), len(v)
    return 0 if k == 0 else (2 if k == n else 1)


def shares(ex):
    c = [0, 0, 0]
    for v in ex.values():
        c[cat(v)] += 1
    return c


def expected(ex, a, b):
    e = [0.0, 0.0, 0.0]
    for v in ex.values():
        n = len(v)
        p0, pn = bb(0, n, a, b), bb(n, n, a, b)
        e[0] += p0; e[2] += pn; e[1] += 1 - p0 - pn
    return e


def main():
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
    res = {}
    for ax, bench in zip(axes, BENCH):
        dev, test, _ = S.seed_executions(bench)
        a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        dc, tc = shares(dev), shares(test)
        de, te = expected(dev, a, b), expected(test, a, b)
        nd, nt = len(dev), len(test)
        xs = [0, 1, 2]
        w = 0.36
        ax.bar([x - w / 2 for x in xs], [100 * c / nd for c in dc], width=w, color="#5B7DB1", edgecolor="#5B7DB1", lw=0.8, label="dev tasks", zorder=3)
        ax.bar([x + w / 2 for x in xs], [100 * c / nt for c in tc], width=w, color="white", edgecolor="#C0504D", hatch="////", lw=0.9, label="test tasks", zorder=3)
        ax.scatter([x - w / 2 for x in xs], [100 * e / nd for e in de], marker="D", s=13, color=INK, zorder=5, label="fitted prior")
        ax.scatter([x + w / 2 for x in xs], [100 * e / nt for e in te], marker="D", s=13, color=INK, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["in no run", "in some runs", "in all runs"], fontsize=7)
        ax.set_xlabel("task solved", fontsize=7.5, labelpad=2)
        ax.set_title("%s   (a, b) = (%.2f, %.2f)" % (BENCH_NAME[bench], a, b), fontsize=7.6, loc="left", pad=3, fontweight="bold")
        runs = sorted({len(v) for v in dev.values()})
        ax.text(0.03, 0.93, "dev: %d tasks, %s runs each\ntest: %d tasks, 3 runs each" % (nd, "/".join(map(str, runs)), nt), transform=ax.transAxes,
                ha="left", va="top", fontsize=5.8, color=MUTED, linespacing=1.15)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, pad=1.5)
        res[bench] = dict(a=round(a, 3), b=round(b, 3), dev_tasks=nd, dev_runs_per_task=runs, dev_counts=dict(zip(CATS, dc)),
                          dev_expected=dict(zip(CATS, [round(x, 1) for x in de])), test_tasks=nt, test_counts=dict(zip(CATS, tc)),
                          test_expected=dict(zip(CATS, [round(x, 1) for x in te])))
    axes[0].set_ylabel("share of tasks, %", fontsize=7.5, labelpad=2)
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.4, bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, "priorfit")
    logger.DATA.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(logger.DATA / "mech_priorfit_numbers.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
