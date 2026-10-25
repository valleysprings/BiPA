"""Figure dev2test (devdyn and devchance in one figure): why the dev gains the optimizers report do not show up on test.  Two panels, DeepSeek-V4.1-Flash.

(a) Reported dev gain against what selection alone reports.  For the six SkillOpt and GEPA runs: dot = the dev gain the run reported (points over its starting score); grey bar = the 5 to 95 percent
    range, tick = the mean, of the gain the best of k copies of the seed report when k = the candidates the run scored on dev (selection.py replay, the same rule and baseline as the run).
    SkillAdam is left out of the panel (its gain is a sum of margins over accepted patches, on another scale; its replay is selection.skilladam_reported_gain).
(b) Test difference against the dev gain, all nine skills.  Filled = the returned skill, test difference to the seed (avg@3, Table 1 runs).  Hollow, joined by a line from the origin = the skills along the
    tau3 path (SkillOpt after rounds 1, 2 and the returned one, GEPA after candidates 1, 2 and the returned one), one test pass each on the updated model against a rerun of the seed on it
    (path_test.json of aux.path_test).  Dashed diagonal: a gain that transfers fully.

Run:  python -m aux.selection_replay.fig_dev2test   (after aux.path_test.collect_path_test)
"""
import json, os
import numpy as np
from aux.selection_replay.style import INK, MUTED, GRID, BENCH_NAME, setup_mpl  # noqa
from aux.selection_replay import fig_devdyn as D  # curve(), END, COL, NAME, MARK
from aux.selection_replay import fig_chance as C  # chance()

from src import layout

OUT = str(layout.OUT / "selection_replay")
PATHTEST = str(layout.OUT / "path_test/path_test.json")
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
SHORT = {"tau3-bench": r"$\tau^3$-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}


def main():
    os.makedirs(OUT, exist_ok=True)
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    ch = {b: C.chance(b) for b in BENCH}
    path = json.load(open(PATHTEST))
    fig = plt.figure(figsize=(6.3, 2.85))
    # ------------------------------------------------ (a)
    ax = fig.add_axes([0.075, 0.34, 0.375, 0.52])
    x, ticks, ticklabels, grp, rows = 0.0, [], [], [], []
    for bench in BENCH:
        xs = []
        for opt in ("skillopt", "gepa"):
            dev, _ = D.END[(bench, opt)]
            K = ch[bench]["obs"]["SkillOpt" if opt == "skillopt" else "GEPA"]["K"]
            m, lo, hi = ch[bench]["best_of_k"][min(K, 8)]
            ax.add_patch(Rectangle((x - 0.27, lo), 0.54, hi - lo, facecolor="#D3D8DE", edgecolor="none", zorder=1))
            ax.plot([x - 0.27, x + 0.27], [m, m], color="#6B7480", lw=1.1, zorder=2)
            ax.scatter([x], [dev], s=34, marker="o", facecolor=D.COL[opt], edgecolor="white", lw=0.7, zorder=4)
            rows.append(dict(bench=bench, opt=opt, K=K, reported=dev, chance_mean=round(m, 1), chance_5=round(lo, 1), chance_95=round(hi, 1)))
            ticks.append(x); ticklabels.append(D.NAME[opt]); xs.append(x)
            x += 1.0
        grp.append((bench, float(np.mean(xs))))
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=6.6)
    ax.tick_params(axis="x", length=0, pad=2)
    for bench, gx in grp:
        ax.text(gx, -0.20, SHORT[bench], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK)
    ax.set_xlim(-0.7, x - 1.0 + 0.7 - 0.8)
    ax.set_ylim(-1, 26)
    ax.set_ylabel("dev gain, points over the start", fontsize=7.2, labelpad=2)
    ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=7, pad=1.5)
    ax.set_title("(a) Reported dev gains match what selection alone reports", fontsize=7.5, loc="left", pad=5, fontweight="bold")
    # ------------------------------------------------ (b)
    bx = fig.add_axes([0.595, 0.34, 0.395, 0.52])
    bx.plot([0, 14], [0, 14], color="#8A9099", lw=0.9, ls=(0, (4, 2)), zorder=1)
    bx.text(14.8, 12.6, "gain transfers in full", fontsize=6.6, color="#6B7480", ha="left", va="center")
    bx.text(42.5, 9.2, "nine skills: slope 0.02,\ncorrelation 0.05", fontsize=6.6, color="#6B7480", ha="right", va="center", linespacing=1.15)
    bx.axhline(0, color=INK, lw=0.6, zorder=1)
    final = []
    for (bench, opt), (dev, test) in D.END.items():
        bx.scatter([dev], [test], marker=D.MARK[bench], s=34, facecolor=D.COL[opt], edgecolor="white", lw=0.7, zorder=5)
        final.append(dict(bench=bench, opt=opt, dev=dev, test=test))
    for opt in ("skillopt", "gepa"):
        pts = sorted((k, g) for k, g, role in path.get("deepseek-flash|%s|tau3-bench" % opt, []) if k is not None)
        if not pts:
            continue
        _, y = D.curve(opt, "tau3-bench")
        xy = [(0.0, 0.0)] + [(float(y[k]), g) for k, g in pts]
        bx.plot([a for a, _ in xy], [b for _, b in xy], color=D.COL[opt], lw=0.9, alpha=0.75, zorder=3)
        bx.scatter([a for a, _ in xy[1:]], [b for _, b in xy[1:]], marker="o", s=26, facecolor="white", edgecolor=D.COL[opt], lw=1.2, zorder=6)
        final.append(dict(path=opt, points=[(round(a, 1), round(b, 1)) for a, b in xy[1:]]))
    bx.set_xlim(-1.5, 43)
    bx.set_ylim(-13, 14)
    bx.set_xlabel("dev gain the optimizer reported, points", fontsize=7.2, labelpad=2)
    bx.set_ylabel("test difference to the seed, points", fontsize=7.2, labelpad=2)
    bx.grid(axis="y", color=GRID, lw=0.4); bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.tick_params(labelsize=7, pad=1.5)
    bx.set_title("(b) Test does not follow the dev gain", fontsize=7.5, loc="left", pad=5, fontweight="bold")
    # ------------------------------------------------ legends
    ha = [Line2D([0], [0], marker="o", color="#555555", lw=0, markersize=5, label="reported dev gain"),
          Patch(facecolor="#D3D8DE", label="5 to 95% of what copies of the seed report"),
          Line2D([0], [0], color="#6B7480", lw=1.1, label="mean of those")]
    fig.legend(handles=ha, loc="lower left", ncol=1, frameon=False, fontsize=6.6, handlelength=1.3, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.055, -0.005))
    hb = [Line2D([0], [0], marker="o", color=D.COL["skillopt"], lw=0, markersize=5, label="SkillOpt"),
          Line2D([0], [0], marker="o", color="#555555", lw=0, markersize=4.5, label=r"$\tau^3$"),
          Line2D([0], [0], marker="o", color=D.COL["gepa"], lw=0, markersize=5, label="GEPA"),
          Line2D([0], [0], marker="s", color="#555555", lw=0, markersize=4.5, label="AutoBench"),
          Line2D([0], [0], marker="o", color=D.COL["skilladam"], lw=0, markersize=5, label="SkillAdam"),
          Line2D([0], [0], marker="^", color="#555555", lw=0, markersize=4.5, label="TB2"),
          Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0.9, markersize=4.5, label=r"along the $\tau^3$ path")]
    fig.legend(handles=hb, loc="lower left", ncol=4, frameon=False, fontsize=6.6, handlelength=1.3, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.545, -0.005))
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/dev2test.{ext}", **kw)
    json.dump(dict(panel_a=rows, panel_b=final), open(OUT + "/dev2test_numbers.json", "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
