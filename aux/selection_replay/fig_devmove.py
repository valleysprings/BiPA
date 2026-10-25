"""Figure devmove: what optimizing moves on dev and what it moves on test.

(a) Aggregated: for each benchmark, the success of the seed and of the returned skill on the dev tasks and on the test tasks, one run each side, from the optimizer's own run (SkillOpt, GEPA; DeepSeek solid
    lines, GPT-6-Luna dashed).  Lines run from the seed to the returned skill; on dev they rise, on test they stay.
(b) Per task: share of tasks the seed fails and the returned skill solves (up) and the share it solves and the returned skill fails (down), on dev and on test; black ticks: what a second run of the seed
    itself flips in each direction (expected over pairs of its runs).
SkillAdam is left out: it scores 10-task subsets and has no whole-panel dev evaluation of a returned skill.  Luna on TB2 returned the seed.
Run:  python -m aux.selection_replay.fig_devmove   (after collect)
"""
import json, os, pickle
import numpy as np
from aux.selection_replay.style import FONT, INK, MUTED, GRID, BENCH_NAME, setup_mpl  # noqa
from src.analysis import selection as S

from src import layout

OUT = str(layout.OUT / "selection_replay")
OPTCOL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5"}
OPTNAME = {"gepa": "GEPA", "skillopt": "SkillOpt"}
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]


def seed_noise(bench):
    dev_files, test_files = S.seed_files(bench)
    out = []
    for files in (dev_files, test_files):
        ex = S.executions(files)
        out.append(float(np.mean([k * (n - k) / (n * (n - 1)) for k, n in ((sum(v), len(v)) for v in ex.values()) if n > 1])) * 100)
    return out


def flips(a, b):
    t = [k for k in a if k in b]
    return (sum(1 for k in t if a[k] == 0 and b[k] == 1), sum(1 for k in t if a[k] == 1 and b[k] == 0), len(t))


def rate(a, b):
    t = [k for k in a if k in b]
    return 100 * sum(a[k] for k in t) / len(t), 100 * sum(b[k] for k in t) / len(t)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec
    data = [d for d in pickle.load(open(OUT + "/devtest_data.pkl", "rb")) if "error" not in d]
    data = [d for d in data if not (d["dev_seed"] == d["dev_final"] or d["final"] == d["seed"])]
    noise = {b: seed_noise(b) for b in BENCH}
    fig = plt.figure(figsize=(7.1, 4.55))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], left=0.075, right=0.99, top=0.92, bottom=0.2, hspace=0.62, wspace=0.28)
    levels = {}
    for ci, bench in enumerate(BENCH):
        ax = fig.add_subplot(gs[0, ci])
        ys = []
        for d in [x for x in data if x["bench"] == bench]:
            ds, dl = rate(d["dev_seed"], d["dev_final"])
            ts, tl = rate(d["test_seed"], d["test_final"])
            c = OPTCOL[d["opt"]]
            ls = "-" if d["model"] == "deepseek-flash" else (0, (3, 2))
            ax.plot([0, 1], [ds, dl], color=c, ls=ls, lw=1.5, marker="o", ms=3.6, zorder=3)
            ax.plot([2.3, 3.3], [ts, tl], color=c, ls=ls, lw=1.5, marker="o", ms=3.6, zorder=3)
            ys += [ds, dl, ts, tl]
            levels["%s|%s|%s" % (d["model"], d["opt"], bench)] = dict(dev_seed=round(ds, 1), dev_returned=round(dl, 1), test_seed=round(ts, 1), test_returned=round(tl, 1))
        lo, hi = (np.floor(min(ys) / 10) * 10 - 5, np.ceil(max(ys) / 10) * 10 + 5) if ys else (0, 100)
        ax.set_ylim(max(lo, 0), min(hi, 100))
        ax.set_xlim(-0.35, 3.65)
        ax.axvline(1.65, color=GRID, lw=0.8)
        ax.set_xticks([0, 1, 2.3, 3.3])
        ax.set_xticklabels(["seed", "returned", "seed", "returned"], fontsize=6.4)
        ax.text(0.5, 1.02, "dev tasks", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)
        ax.text(2.8, 1.02, "test tasks", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)
        ax.set_title(BENCH_NAME[bench], fontsize=8, fontweight="bold", loc="left", pad=15)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.8, pad=1.5, length=2)
        if ci == 0:
            ax.set_ylabel("success, %", fontsize=7.5, labelpad=2)
        if bench == "terminal-bench-2-1":
            ax.text(0.97, 0.04, "Luna returned the seed", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=MUTED)
    fig.text(0.075, 0.975, "(a) Aggregated: success of the seed and of the returned skill", fontsize=8, fontweight="bold", ha="left", va="top")
    # (b) per task
    bx = fig.add_subplot(gs[1, :])
    order = [d for model in ("deepseek-flash", "gpt-6-luna") for opt in ("skillopt", "gepa") for bench in BENCH for d in data if d["model"] == model and d["opt"] == opt and d["bench"] == bench]
    xt, xl, numbers, grp = [], [], [], {}
    x, prev = 0.0, None
    for d in order:
        if prev is not None and d["model"] != prev:
            x += 0.9
        prev = d["model"]
        gd, ld, nd = flips(d["dev_seed"], d["dev_final"])
        gt, lt, nt = flips(d["test_seed"], d["test_final"])
        c = OPTCOL[d["opt"]]
        for off, (g, l, n, ref, split) in ((-0.2, (gd, ld, nd, noise[d["bench"]][0], "dev")), (0.2, (gt, lt, nt, noise[d["bench"]][1], "test"))):
            fc = c if split == "dev" else "white"
            h = "////" if split == "test" else None
            bx.bar(x + off, 100 * g / n, width=0.34, color=fc, edgecolor=c, lw=0.9, zorder=3, hatch=h)
            bx.bar(x + off, -100 * l / n, width=0.34, color=fc, edgecolor=c, lw=0.9, zorder=3, hatch=h)
            bx.plot([x + off - 0.19, x + off + 0.19], [ref, ref], color=INK, lw=0.8, zorder=5)
            bx.plot([x + off - 0.19, x + off + 0.19], [-ref, -ref], color=INK, lw=0.8, zorder=5)
        numbers.append(dict(model=d["model"], opt=d["opt"], bench=d["bench"], dev=dict(gained=gd, lost=ld, n=nd), test=dict(gained=gt, lost=lt, n=nt)))
        xt.append(x); xl.append({"tau3-bench": r"$\tau^3$", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}[d["bench"]])
        grp.setdefault((d["model"], d["opt"]), []).append(x)
        x += 1.0
    bx.axhline(0, color=INK, lw=0.6)
    bx.set_xticks(xt); bx.set_xticklabels(xl, fontsize=6.8); bx.tick_params(axis="x", pad=1.5, length=0); bx.tick_params(axis="y", labelsize=7, pad=1.5)
    bx.set_ylabel("share of tasks, %", fontsize=7.5, labelpad=2)
    bx.set_xlim(-0.7, x - 0.3); bx.set_ylim(-16, 25)
    bx.grid(axis="y", color=GRID, lw=0.4); bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    for (m, o), xs in grp.items():
        bx.text(np.mean(xs), -0.13, OPTNAME[o], transform=bx.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, fontweight="bold", color=OPTCOL[o])
    n_ds = sum(1 for d in order if d["model"] == "deepseek-flash")
    bx.text((xt[0] + xt[n_ds - 1]) / 2, 22.5, "DeepSeek-V4.1-Flash", ha="center", fontsize=7.2, color=MUTED)
    bx.text((xt[n_ds] + xt[-1]) / 2, 22.5, "GPT-6-Luna", ha="center", fontsize=7.2, color=MUTED)
    bx.set_title("(b) Per task: newly solved (up) and newly failed (down), dev and test", fontsize=8, loc="left", pad=4, fontweight="bold")
    h = [Patch(facecolor=OPTCOL["gepa"], edgecolor=OPTCOL["gepa"], label="GEPA"), Patch(facecolor=OPTCOL["skillopt"], edgecolor=OPTCOL["skillopt"], label="SkillOpt"),
         Line2D([0], [0], color="#555555", lw=1.5, label="DeepSeek"), Line2D([0], [0], color="#555555", lw=1.5, ls=(0, (3, 2)), label="GPT-6-Luna"),
         Patch(facecolor="#777777", edgecolor="#777777", label="dev (solid)"), Patch(facecolor="white", edgecolor="#777777", hatch="////", label="test (hatched)"),
         Line2D([0], [0], color=INK, lw=0.9, label="a second run of the seed")]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=6.4, handlelength=1.5, columnspacing=1.0, bbox_to_anchor=(0.52, 0.0))
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devmove.{ext}", **kw)
    json.dump(dict(levels=levels, per_task=numbers, seed_noise_pct_dev_test=noise), open(OUT + "/devmove_numbers.json", "w"), indent=1)
    for k, v in levels.items():
        print(k, v)


if __name__ == "__main__":
    main()
