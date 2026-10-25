"""Figure devtest: what optimizing moves on dev and what it moves on test.  Aggregated (a) and per task (b).

(a) Per learned skill (3 optimizers x 3 benchmarks, DeepSeek-V4.1-Flash): x = candidates the optimizer scored on dev (SkillAdam: attempts), hollow = the dev gain it reported
    (points), filled = the test difference to the seed (avg@3, Table 1 runs, points; values from mech_devtest.md).  Thin line = the gap between the two.
(b) Per task, one run each side, taken from the optimizer's own run (it evaluated the seed and the returned skill on dev and on test): the share of tasks the seed fails and the
    returned skill solves (gained, up) and the share the seed solves and the returned skill fails (lost, down), on dev and on test.  Grey ticks: the share a second run of the seed itself
    flips each way (expected over pairs of its runs: mean over tasks of k(n-k)/(n(n-1))), i.e. what run-to-run noise alone does.
    SkillAdam is left out of (b): it scores 10-task subsets, never the whole dev panel with a returned skill.  Luna on TB2 returned the seed.

Run:  python -m aux.selection_replay.fig_devtest   (after collect)
"""
import json, os, pickle
import numpy as np
from aux.selection_replay.style import FONT, INK, MUTED, GRID, BENCH_NAME, setup_mpl  # noqa
from src.analysis import selection as S

from src import layout

OUT = str(layout.OUT / "selection_replay")
OPTCOL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5", "skilladam": "#E08A1E"}
OPTNAME = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
MARK = {"tau3-bench": "o", "automationbench": "s", "terminal-bench-2-1": "^"}
# (dev gain, test difference avg@3) in points, from mech_devtest.md; K = candidates scored on dev (SkillAdam: attempts), selection.observed
A = {
    ("tau3-bench", "gepa"): (7, 16.7, -0.9), ("tau3-bench", "skillopt"): (4, 13.3, 2.7), ("tau3-bench", "skilladam"): (30, 40.0, -1.3),
    ("automationbench", "gepa"): (7, 3.3, -0.8), ("automationbench", "skillopt"): (4, 6.7, -3.3), ("automationbench", "skilladam"): (25, 20.0, -0.3),
    ("terminal-bench-2-1", "gepa"): (8, 11.1, -1.9), ("terminal-bench-2-1", "skillopt"): (5, 11.1, -11.1), ("terminal-bench-2-1", "skilladam"): (21, 22.2, -7.4),
}


def seed_noise(bench):
    """(dev, test) share of tasks a second run of the seed flips in one direction (gained = lost in expectation)."""
    dev_files, test_files = S.seed_files(bench)
    out = []
    for files in (dev_files, test_files):
        ex = S.executions(files)
        out.append(float(np.mean([k * (n - k) / (n * (n - 1)) for k, n in ((sum(v), len(v)) for v in ex.values()) if n > 1])) * 100)
    return out


def flips(a, b):
    t = [k for k in a if k in b]
    g = sum(1 for k in t if a[k] == 0 and b[k] == 1)
    l = sum(1 for k in t if a[k] == 1 and b[k] == 0)
    return g, l, len(t)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    data = [d for d in pickle.load(open(OUT + "/devtest_data.pkl", "rb")) if "error" not in d]
    data = [d for d in data if not (d["dev_seed"] == d["dev_final"] or d["final"] == d["seed"])]
    noise = {b: seed_noise(b) for b in BENCH}
    fig = plt.figure(figsize=(7.1, 3.35))
    ax = fig.add_axes([0.07, 0.23, 0.30, 0.66])
    for (bench, opt), (K, dev, test) in A.items():
        c = OPTCOL[opt]
        ax.plot([K, K], [test, dev], color=c, lw=0.8, alpha=0.55, zorder=1)
        ax.scatter([K], [dev], marker=MARK[bench], s=26, facecolor=c, edgecolor=c, lw=0.6, zorder=3)
        ax.scatter([K], [test], marker=MARK[bench], s=26, facecolor="white", edgecolor=c, lw=1.2, zorder=4)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xscale("log")
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([4, 8, 16, 30])
    ax.set_xticklabels(["4", "8", "16", "30"])
    ax.set_xlim(3.2, 38)
    ax.set_ylim(-14, 44)
    ax.set_xlabel("candidates scored on dev", fontsize=7.5, labelpad=2)
    ax.set_ylabel("points vs the seed", fontsize=7.5, labelpad=2)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.set_title("(a) Aggregated: reported dev gain (solid)\nand test difference (hollow)", fontsize=8, loc="left", pad=4, fontweight="bold")
    # per-task panel
    bx = fig.add_axes([0.47, 0.23, 0.52, 0.66])
    order = []
    for model in ("deepseek-flash", "gpt-6-luna"):
        for opt in ("skillopt", "gepa"):
            for bench in BENCH:
                for d in data:
                    if d["model"] == model and d["opt"] == opt and d["bench"] == bench:
                        order.append(d)
    xt, xl, numbers = [], [], []
    grp = {}
    x = 0.0
    group_start = {}
    prev_model = None
    for d in order:
        if prev_model is not None and d["model"] != prev_model:
            x += 0.9
        prev_model = d["model"]
        gd, ld, nd = flips(d["dev_seed"], d["dev_final"])
        gt, lt, nt = flips(d["test_seed"], d["test_final"])
        c = OPTCOL[d["opt"]]
        for off, (g, l, n, ref, split) in ((-0.2, (gd, ld, nd, noise[d["bench"]][0], "dev")), (0.2, (gt, lt, nt, noise[d["bench"]][1], "test"))):
            fc = c if split == "dev" else "white"
            bx.bar(x + off, 100 * g / n, width=0.34, color=fc, edgecolor=c, lw=0.9, zorder=3, hatch="////" if split == "test" else None)
            bx.bar(x + off, -100 * l / n, width=0.34, color=fc, edgecolor=c, lw=0.9, zorder=3, hatch="////" if split == "test" else None)
            bx.plot([x + off - 0.19, x + off + 0.19], [ref, ref], color=INK, lw=0.8, zorder=5)
            bx.plot([x + off - 0.19, x + off + 0.19], [-ref, -ref], color=INK, lw=0.8, zorder=5)
        numbers.append(dict(model=d["model"], opt=d["opt"], bench=d["bench"], dev=dict(gained=gd, lost=ld, n=nd), test=dict(gained=gt, lost=lt, n=nt)))
        xt.append(x)
        xl.append({"tau3-bench": r"$\tau^3$", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}[d["bench"]])
        grp.setdefault((d["model"], d["opt"]), []).append(x)
        x += 1.0
    bx.axhline(0, color=INK, lw=0.6)
    bx.set_xticks(xt)
    bx.set_xticklabels(xl, fontsize=6.8)
    bx.tick_params(axis="x", pad=1.5, length=0)
    bx.tick_params(axis="y", labelsize=7, pad=1.5)
    bx.set_ylabel("share of tasks, %", fontsize=7.5, labelpad=2)
    bx.set_xlim(-0.7, x - 0.3)
    bx.set_ylim(-16, 25)
    bx.grid(axis="y", color=GRID, lw=0.4)
    bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.set_title("(b) Per task: newly solved (up) and newly failed (down), dev and test", fontsize=8, loc="left", pad=4, fontweight="bold")
    for (m, o), xs in grp.items():
        bx.text(np.mean(xs), -0.13, OPTNAME[o], transform=bx.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, fontweight="bold", color=OPTCOL[o])
    # model labels
    n_ds = sum(1 for d in order if d["model"] == "deepseek-flash")
    bx.text((xt[0] + xt[n_ds - 1]) / 2, 22.5, "DeepSeek-V4.1-Flash", ha="center", fontsize=7.2, color=MUTED)
    bx.text((xt[n_ds] + xt[-1]) / 2, 22.5, "GPT-6-Luna", ha="center", fontsize=7.2, color=MUTED)
    h = [Patch(facecolor=OPTCOL["gepa"], edgecolor=OPTCOL["gepa"], label="GEPA"), Patch(facecolor=OPTCOL["skillopt"], edgecolor=OPTCOL["skillopt"], label="SkillOpt"),
         Patch(facecolor=OPTCOL["skilladam"], edgecolor=OPTCOL["skilladam"], label="SkillAdam (a only)"),
         Patch(facecolor="#777777", edgecolor="#777777", label="dev (solid)"), Patch(facecolor="white", edgecolor="#777777", hatch="////", label="test (hollow, hatched)"),
         Line2D([0], [0], color=INK, lw=0.9, label="a second run of the seed"),
         Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", lw=0, markersize=4, label=r"$\tau^3$"),
         Line2D([0], [0], marker="s", color="#777777", markerfacecolor="white", lw=0, markersize=4, label="AutoBench"),
         Line2D([0], [0], marker="^", color="#777777", markerfacecolor="white", lw=0, markersize=4, label="TB2")]
    fig.legend(handles=h, loc="lower center", ncol=9, frameon=False, fontsize=6.3, handlelength=1.3, columnspacing=0.9, bbox_to_anchor=(0.52, 0.0))
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devtest.{ext}", **kw)
    json.dump(dict(per_task=numbers, seed_noise_pct_dev_test={b: noise[b] for b in BENCH}, a=[dict(bench=b, opt=o, K=k, dev_gain=dv, test_diff=t) for (b, o), (k, dv, t) in A.items()]),
              open(OUT + "/devtest_numbers.json", "w"), indent=1)
    for n in numbers:
        print(n)
    print(noise)


if __name__ == "__main__":
    main()
