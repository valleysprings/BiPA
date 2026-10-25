"""The devpertask figure: what optimizing moves on dev and what it moves on test, task by task.

Per optimizer run (SkillOpt, GEPA; both models), the share of tasks the seed fails and the returned skill solves (up) and the share it solves and the
returned skill fails (down), on dev and on test, one run each side from the optimizer's own run; black ticks: what a second run of the seed itself
flips in each direction (expected over pairs of its executions, selection.seed_executions). SkillAdam is left out: it scores 10-task subsets and has no whole-panel dev evaluation
of a returned skill. Luna on TB2 returned the seed.
Run:  python -m aux.selection_replay.fig_devpertask   (after collect)  -> fig/mech/mech_devpertask.{pdf,svg}, data/mech/mech_devpertask_numbers.json
"""
import json, pickle
import numpy as np
from aux.selection_replay.selection import seed_executions
from aux.selection_replay.style import INK, MUTED, GRID, setup_mpl

from src import layout
from src.utils import logger
from src.utils.fig import save_figure

OUT = str(layout.OUT / "selection_replay")
OPTCOL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5"}
OPTNAME = {"gepa": "GEPA", "skillopt": "SkillOpt"}
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]


def seed_noise(bench):
    out = []
    for ex in seed_executions(bench)[:2]:
        out.append(float(np.mean([k * (n - k) / (n * (n - 1)) for k, n in ((sum(v), len(v)) for v in ex.values()) if n > 1])) * 100)
    return out


def flips(a, b):
    t = [k for k in a if k in b]
    return (sum(1 for k in t if a[k] == 0 and b[k] == 1), sum(1 for k in t if a[k] == 1 and b[k] == 0), len(t))


def main():
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec
    data = [d for d in pickle.load(open(OUT + "/devtest_data.pkl", "rb")) if "error" not in d]
    data = [d for d in data if not (d["dev_seed"] == d["dev_final"] or d["final"] == d["seed"])]
    noise = {b: seed_noise(b) for b in BENCH}
    fig = plt.figure(figsize=(7.1, 2.75))
    gs = GridSpec(1, 1, figure=fig, left=0.075, right=0.99, top=0.88, bottom=0.27)
    bx = fig.add_subplot(gs[0, 0])
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
    bx.set_title("Per task: newly solved (up) and newly failed (down), dev and test", fontsize=8, loc="left", pad=4, fontweight="bold")
    h = [Patch(facecolor=OPTCOL["gepa"], edgecolor=OPTCOL["gepa"], label="GEPA"), Patch(facecolor=OPTCOL["skillopt"], edgecolor=OPTCOL["skillopt"], label="SkillOpt"),
         Patch(facecolor="#777777", edgecolor="#777777", label="dev (solid)"), Patch(facecolor="white", edgecolor="#777777", hatch="////", label="test (hatched)"),
         Line2D([0], [0], color=INK, lw=0.9, label="a second run of the seed")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.4, handlelength=1.5, columnspacing=1.0, bbox_to_anchor=(0.52, 0.0))
    save_figure(fig, "devpertask")
    logger.DATA.mkdir(parents=True, exist_ok=True)
    json.dump(dict(per_task=numbers, seed_noise_pct_dev_test=noise), open(logger.DATA / "mech_devpertask_numbers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
