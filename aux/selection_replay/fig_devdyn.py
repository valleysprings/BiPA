"""Figures devdyn and devtransfer, one column each.

devdyn: along each optimizer's run, the dev gain it has reported so far (points over its starting score) against the share of its candidates scored; at the end of the run, the test
difference of the returned skill to the seed (avg@3, hollow).  SkillOpt: best dev score after each round (rounds.json); GEPA: best of val_scores; SkillAdam: running sum of the margins of its accepted patches
(10-task subsets), against its cumulative attempts.  DeepSeek-V4.1-Flash, Table 1 runs.
devtransfer: for each of the nine learned skills, the reported dev gain against the test difference to the seed; the diagonal is perfect transfer; the line is the least-squares fit.
Test differences and dev gains of the endpoints from mech_devtest.md.
Run:  python -m aux.selection_replay.fig_devdyn
"""
import json, os
import numpy as np
from aux.selection_replay.style import INK, MUTED, GRID, BENCH_NAME, setup_mpl  # noqa

from src import layout

EXE = str(layout.EXECUTION / "deepseek-flash")
RES = str(layout.RESULTS / "deepseek-flash")
OUT = str(layout.OUT / "selection_replay")
RUN = "native-20261008-https"
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
COL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5", "skilladam": "#E08A1E"}
NAME = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
MARK = {"tau3-bench": "o", "automationbench": "s", "terminal-bench-2-1": "^"}
# (dev gain reported, test difference avg@3) in points, mech_devtest.md
END = {("tau3-bench", "gepa"): (16.7, -0.9), ("tau3-bench", "skillopt"): (13.3, 2.7), ("tau3-bench", "skilladam"): (40.0, -1.3),
       ("automationbench", "gepa"): (3.3, -0.8), ("automationbench", "skillopt"): (6.7, -3.3), ("automationbench", "skilladam"): (20.0, -0.3),
       ("terminal-bench-2-1", "gepa"): (11.1, -1.9), ("terminal-bench-2-1", "skillopt"): (11.1, -11.1), ("terminal-bench-2-1", "skilladam"): (22.2, -7.4)}


def curve(opt, bench):
    """(x in % of candidates scored, dev gain so far in points) from the run's own record, starting at (0, 0)."""
    s = json.load(open(f"{EXE}/{opt}/{bench}/{RUN}/summary.json"))["extra"]
    if opt == "skillopt":
        r = json.load(open(f"{RES}/skillopt/{bench}/rounds.json"))["rounds"]
        base = s["baseline_selection_hard"]
        best = [100 * (x["scores"]["best_score"] - base) for x in r if x.get("scores")]
        ys = [0.0] + best
        xs = np.linspace(0, 100, len(ys))
    elif opt == "gepa":
        v = s["val_scores"]
        ys = [0.0] + [100 * (max(v[:i + 1]) - v[0]) for i in range(1, len(v))]
        xs = np.linspace(0, 100, len(ys))
    else:
        it = [i for i in s["iterations"] if isinstance(i.get("baseline"), (int, float)) and isinstance(i.get("candidate"), (int, float))]
        gain, att, ys, xs = 0.0, 0, [0.0], [0.0]
        total = sum(i["attempts"] for i in it)
        for i in it:
            att += i["attempts"]
            if i["accepted"]:
                gain += 100 * (i["candidate"] - i["baseline"])
            ys.append(gain)
            xs.append(100 * att / total)
        xs = np.array(xs)
    return np.array(xs), np.array(ys)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    # ---------------- devdyn
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.55), sharex=True)
    for ax, bench in zip(axes, BENCH):
        for oi, opt in enumerate(("gepa", "skillopt", "skilladam")):
            x, y = curve(opt, bench)
            dev, test = END[(bench, opt)]
            xm = 104 + 4.2 * oi
            ax.step(x, y, where="post", color=COL[opt], lw=1.5, zorder=3)
            ax.scatter([xm], [test], marker="o", s=22, facecolor="white", edgecolor=COL[opt], lw=1.3, zorder=5, clip_on=False)
            ax.plot([100, xm], [y[-1], test], color=COL[opt], lw=0.7, ls=":", zorder=2, clip_on=False)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(-14, 44)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, pad=1.5)
        ax.set_title(BENCH_NAME[bench], fontsize=8, loc="left", pad=2, fontweight="bold")
        ax.set_ylabel("points vs the seed", fontsize=7, labelpad=1)
    axes[-1].set_xlabel("share of the optimizer's candidates scored, %", fontsize=7.5, labelpad=2)
    axes[0].text(1.0, 1.28, "dev gain so far (line), test difference (hollow)", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=6.4, color=MUTED)
    h = [Line2D([0], [0], color=COL[o], lw=1.5, label=NAME[o]) for o in ("gepa", "skillopt", "skilladam")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.8, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.52, 0.0))
    fig.tight_layout(rect=(0, 0.04, 0.93, 1), h_pad=0.6)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devdyn.{ext}", **kw)
    # ---------------- devtransfer
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    xs, ys = [], []
    for (bench, opt), (dev, test) in END.items():
        ax.scatter([dev], [test], marker=MARK[bench], s=34, facecolor=COL[opt], edgecolor="white", lw=0.6, zorder=4)
        xs.append(dev); ys.append(test)
    xs, ys = np.array(xs), np.array(ys)
    slope, icpt = np.polyfit(xs, ys, 1)
    xx = np.array([0, 42])
    ax.plot(xx, xx, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.text(30.5, 33.2, "perfect transfer", fontsize=6.4, color=MUTED, ha="center", rotation=36)
    ax.plot(xx, slope * xx + icpt, color=INK, lw=1.1, zorder=2)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(0, 44); ax.set_ylim(-14, 44)
    ax.set_xlabel("reported dev gain, points", fontsize=7.5, labelpad=2)
    ax.set_ylabel("test difference to the seed, points", fontsize=7.5, labelpad=2)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, pad=1.5)
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.text(0.97, 0.06, "fitted slope %.2f, correlation %.2f" % (slope, r), transform=ax.transAxes, ha="right", fontsize=6.6, color=INK)
    h = [Line2D([0], [0], marker="o", color=COL[o], lw=0, markersize=5, label=NAME[o]) for o in ("gepa", "skillopt", "skilladam")] + \
        [Line2D([0], [0], marker=MARK[b], color="#777777", lw=0, markersize=4.5, markerfacecolor="#777777", label=n) for b, n in (("tau3-bench", r"$\tau^3$"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2"))]
    ax.legend(handles=h, loc="upper left", ncol=2, frameon=False, fontsize=6.3, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devtransfer.{ext}", **kw)
    json.dump(dict(slope=round(float(slope), 3), intercept=round(float(icpt), 2), corr=round(r, 3), n=len(xs),
                   transfer_ratio={f"{b}|{o}": round(t / d, 2) for (b, o), (d, t) in END.items()}), open(OUT + "/devtransfer_numbers.json", "w"), indent=1)
    print("slope %.3f intercept %.2f corr %.3f" % (slope, icpt, r))
    for opt in ("skillopt", "gepa", "skilladam"):
        for b in BENCH:
            x, y = curve(opt, b); print(opt, b[:5], np.round(y, 1).tolist())


if __name__ == "__main__":
    main()
