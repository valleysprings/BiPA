"""Every test run of the seed and of each learned skill, both models: its success and its input tokens against the seed's
runs, beside the seed's own spread from run to run (Figure mech_reruns3).

A run's success is the share of its tasks with a verified outcome that it solves; the mean of a cell's three runs is Table
1's Q. A run's input-token ratio is the geometric mean, over its tasks with a verified outcome and input tokens, of its input
tokens on a task over the geometric mean of the seed's runs on that task. Input tokens are summed over the distinct requests
of an episode that report them, also when the episode's token capture is incomplete. A skill's bar is the mean of its runs'
success and the geometric mean of their ratios. The seed's spread is its highest run's success minus its lowest, as printed
(to 0.1 point), on each benchmark with a learned skill; a skill is within it when its mean differs from the seed's mean by no
more than the spread. A skill that returned the seed has no runs of its own and is left out."""
import json, math

import numpy as np

from src.analysis import settings as S
from src.utils import fig as F, logger

BENCHES = ("terminal-bench-2-1", "tau3-bench", "automationbench")  # the figure's column order


def tokens(episode):
    """Input tokens of an episode, summed over its distinct requests that report them (None if none does)."""
    counts = [r["input_tokens"] for r in {r["request_id"]: r for r in episode.get("requests", [])}.values()
              if r.get("input_tokens") is not None]
    return sum(counts) if counts else None


def runs(model, bench, method):
    """[{task: (solved, input tokens)}] per test run, over the tasks with a verified outcome."""
    return [{t: (e["success"], tokens(e)) for t, e in eps.items() if S.known(e)}
            for eps in S.cell_episodes(model, bench, method).values()]


def cell(model, bench):
    """{method: {"success": [per run], "ratio": [per run]}} of the seed and the learned skills of one model and benchmark."""
    seed = runs(model, bench, "native-seed")
    logs = {}
    for run in seed:
        for t, (_, x) in run.items():
            if x:
                logs.setdefault(t, []).append(math.log(x))
    ref = {t: np.mean(v) for t, v in logs.items()}
    out = {}
    for method in ("native-seed",) + S.OPTIMIZERS:
        if method != "native-seed" and S.returned_seed(model, bench, method):
            continue
        rs = seed if method == "native-seed" else runs(model, bench, method)
        out[method] = {"success": [100 * float(np.mean([s for s, _ in r.values()])) for r in rs],
                       "ratio": [float(np.exp(np.mean([math.log(x) - ref[t] for t, (_, x) in r.items() if x and t in ref]))) for r in rs]}
    return out


def gmean(values):
    return float(np.exp(np.mean(np.log(values))))


def summary(cells):
    """Per model: the seed's spread per benchmark, and how many learned skills lie within it and read more than every seed run."""
    out = {}
    for model, by_bench in cells.items():
        within = above = n = 0
        spread = {}
        for bench, c in by_bench.items():
            seed = c["native-seed"]
            for method, v in c.items():
                if method != "native-seed":
                    spread[bench] = round(max(seed["success"]), 1) - round(min(seed["success"]), 1)
                    n += 1
                    within += abs(np.mean(v["success"]) - np.mean(seed["success"])) <= spread[bench] + 1e-9
                    above += min(v["ratio"]) > max(seed["ratio"])
        out[model] = {"spread": spread, "within": within, "above": above, "learned": n}
    return out


def numbers(cells, summ):
    rows = [[f"{S.MODEL_SHORT[m]} {S.BENCH_SHORT[b]} {S.ARM_NAME[k]}", ", ".join(f"{x:.1f}" for x in v["success"]), f"{np.mean(v['success']):.1f}",
             ", ".join(f"{x:.3f}" for x in v["ratio"]), f"{gmean(v['ratio']):.3f}"]
            for m, by_bench in cells.items() for b, c in by_bench.items() for k, v in c.items()]
    out = [logger.md_table(["cell", "success per run (%)", "mean", "input tokens / seed per run", "bar"], rows), ""]
    for model, s in summ.items():
        out.append(f"{S.MODEL_NAME[model]}: seed spread (points) " + ", ".join(f"{S.BENCH_SHORT[b]} {v:.1f}" for b, v in s["spread"].items())
                   + f"; learned skills within the seed's spread: {s['within']} of {s['learned']}; whose every run reads more input tokens"
                   + f" than the seed's highest run: {s['above']} of {s['learned']}.")
    return "\n".join(out)


def panel(ax, plt, c, row):
    """Row 0: success; row 1: the input-token ratio, on a log axis."""
    from matplotlib.ticker import MaxNLocator
    key, names = ("success", "ratio")[row], list(c)
    seed = c["native-seed"][key]
    ax.axhspan(min(seed), max(seed), color=F.MID, alpha=0.16, lw=0, zorder=0)
    ax.axhline(np.mean(seed) if row == 0 else gmean(seed), color=F.MID, lw=0.8, zorder=1)
    if row == 1:
        ax.axhline(1.0, color=F.INK, lw=0.5, ls=(0, (3, 2)), zorder=1)
    every = []
    for i, m in enumerate(names):
        vals = c[m][key]
        every += vals
        xs = i + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.2
        ax.scatter(xs, vals, s=20, marker=F.MARKER[m], color=F.ARM[m], zorder=4, lw=0)
        bar = np.mean(vals) if row == 0 else gmean(vals)
        ax.plot([i - 0.32, i + 0.32], [bar, bar], color=F.ARM[m], lw=1.6, zorder=3, solid_capstyle="butt")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([S.ARM_NAME[m] if m != "native-seed" else "seed" for m in names] if row == 1 else [""] * len(names),
                       rotation=25 if row == 1 else 0, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0 if row == 0 else 2.5, pad=2, labelsize=7.5)
    ax.set_xlim(-0.6, max(len(names), 4) - 0.4)
    F.despine(ax)
    if row == 0:
        lo, hi = min(every), max(every)
        pad = max(1.5, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    else:
        ax.set_yscale("log")
        lo, hi = min(every + [1.0]) / 1.05, max(every + [1.0]) * 1.05
        ax.set_ylim(lo, hi)
        ticks = [t / 10 for t in range(8, 18) if lo <= t / 10 <= hi]
        ax.set_yticks(ticks if len(ticks) <= 5 else ticks[::2] if ticks[0] * 10 % 2 == 0 else ticks[1::2])
        ax.set_yticklabels([f"{t:.1f}x" for t in ax.get_yticks()])
        ax.minorticks_off()
    if len(names) == 1 and row == 0:
        ax.text(0.97, 0.5, "all three optimizers\nreturned the seed:\nno skill rows", transform=ax.transAxes, ha="right", va="center",
                fontsize=7, color=F.DARK, linespacing=1.25, bbox=dict(fc="white", ec="none", alpha=0.9, pad=2), zorder=5)


def draw(cells):
    plt = F.pyplot(7.5, title=8, tick=7, legend=7.5)
    fig, A = plt.subplots(5, 3, figsize=(6.3, 5.4), gridspec_kw=dict(height_ratios=[1, 1, 0.42, 1, 1], hspace=0.2, wspace=0.3,
                                                                     left=0.085, right=0.995, bottom=0.075, top=0.9))
    for ax in A[2]:
        ax.set_visible(False)
    axs = np.array([A[0], A[1], A[3], A[4]])
    for g, (model, by_bench) in enumerate(cells.items()):
        for j, bench in enumerate(BENCHES):
            for row in (0, 1):
                panel(axs[2 * g + row, j], plt, by_bench[bench], row)
            axs[2 * g, j].set_title(F.BENCH_LABEL[bench], loc="left", fontweight="bold", pad=3)
        axs[2 * g, 0].set_ylabel("test success (%)")
        axs[2 * g + 1, 0].set_ylabel("input tokens / seed")
        top = axs[2 * g, 0].get_position().y1
        fig.text(0.085, top + 0.035, S.MODEL_NAME[model], fontsize=F.HEAD, fontweight="bold", ha="left", va="bottom")
        fig.text(0.995, top + 0.035, "input-token row: bars and grey line are geometric means", fontsize=7.5, color=F.DARK,
                 ha="right", va="bottom")
        if g:
            fig.add_artist(plt.Line2D([0.02, 0.995], [top + 0.057] * 2, color=F.LIGHT, lw=0.5, transform=fig.transFigure))
    handles = [plt.Line2D([], [], marker="o", ls="", color=F.ARM["native-seed"], ms=4.5), plt.Rectangle((0, 0), 1, 1, fc=F.MID, alpha=0.16, lw=0),
               plt.Line2D([], [], ls="-", color=F.MID, lw=1.2)]
    handles += [plt.Line2D([], [], marker=F.MARKER[m], ls="", color=F.ARM[m], ms=4.5) for m in S.OPTIMIZERS]
    handles += [plt.Line2D([], [], ls="-", color=F.DARK, lw=1.6)]
    labels = ["seed run", "seed: range of its 3 runs", "seed mean", "GEPA run", "SkillOpt run", "SkillAdam run", "mean of a skill's runs"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False, ncol=7, handlelength=1.2,
               columnspacing=1.0, handletextpad=0.4, fontsize=7)
    F.save_figure(fig, "reruns3", width=F.TEXT_W)


def main():
    cells = {model: {bench: cell(model, bench) for bench in BENCHES} for model in (S.DEEPSEEK, S.LUNA)}
    (S.cache("success") / "reruns.json").write_text(json.dumps(cells, indent=1))
    summ = summary(cells)
    used = [(m, b, k) for m, by_bench in cells.items() for b, c in by_bench.items() for k in c]
    logger.record(__spec__.name, "reruns", numbers(cells, summ), S.runs_text(used))
    draw(cells)


if __name__ == "__main__":
    main()
