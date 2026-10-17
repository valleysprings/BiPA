"""mech_calls: share of AutoBench turns with 0, 1 and 2+ executed calls, seed vs learned skill, per optimizer (stacked 100% bars);
calls per turn (seed -> skill) printed at the right. Reads assets/cache/analysis/mech/callsperturn.json (cell 'AutoB').
  python3 -m analysis.mech.calls_fig"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.mech import common as M

OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
COLORS = ["#D9D9D9", "#9ECAE1", "#2171B5"]
LABELS = ["0 calls", "1 call", "2+ calls"]


def draw():
    cells = {c["method"]: c for c in json.loads((M.CACHE / "callsperturn.json").read_text())["cells"] if c["short"] == "AutoB"}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7})
    fig, ax = plt.subplots(figsize=(4.0, 1.6))
    fig.subplots_adjust(left=0.23, right=0.79, bottom=0.04, top=0.84)
    y, ys, numbers = 0.0, [], []
    for o in OPTS:
        ytop = y
        for kind, lab in (("seed", "seed"), ("skill", "skill")):
            c = cells[o][kind]
            sh = [c["share0"], c["share1"], c["share2p"]]
            left = 0.0
            for s, col in zip(sh, COLORS):
                ax.barh(y, s * 100, left=left, height=0.8, color=col, lw=0)
                if s >= 0.12:
                    ax.text(left + s * 50, y, f"{s * 100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if col == COLORS[2] else "black")
                left += s * 100
            ax.text(-1.5, y, lab, ha="right", va="center", fontsize=7)
            numbers.append((o, kind, round(sh[0] * 100, 1), round(sh[1] * 100, 1), round(sh[2] * 100, 1), round(c["m"], 2), c["turns"]))
            y += 1
        ax.text(102, ytop + 0.5, f"{cells[o]['seed']['m']:.2f} → {cells[o]['skill']['m']:.2f}", ha="left", va="center", fontsize=7)
        ax.text(-15, ytop + 0.5, o, ha="right", va="center", fontsize=7, fontweight="bold")
        y += 0.55
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.55, -0.6)
    ax.axis("off")
    ax.text(102, -0.6, "calls / turn", ha="left", va="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, LABELS, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.45, 1.0),
               handlelength=1.0, columnspacing=1.5)
    M.save_figure(fig, "calls")
    return numbers


def main():
    numbers = draw()
    table = M.md_table(["optimizer", "skill", "0 calls %", "1 call %", "2+ calls %", "calls/turn", "turns"], numbers)
    M.write_record("calls", "Share of AutoBench turns with 0, 1, 2+ executed calls, seed vs learned skill; calls per turn at the right.",
                   "deepseek-flash, AutoBench, each optimizer's own seed and learned-skill AutoBench runs (callsperturn.json).", "Pooled over episodes and turns.", table,
                   "python3 -m analysis.mech.calls_fig")


if __name__ == "__main__":
    main()
