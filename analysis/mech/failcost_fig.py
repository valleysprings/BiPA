"""Appendix figure mech_failcost: the four-term input-token tax of each learned skill on tasks both skill and seed solve 3/3 and tasks both fail 3/3.
Role: replaces the table "The tax on solved and failed tasks". Reads assets/cache/analysis/mech/failcost.json (failcost.py compute).
For each of the nine (optimizer, benchmark) skills two horizontal stacked bars (upper: both solve, lower: both fail) of the four terms of dlog X
(positive right, negative left); task count n at the bar end; black tick at the total.
Command (from bipa):  python3 -m analysis.mech.failcost_fig"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.mech import common as MC

TERMS = [("more_actions", "more actions", "#0072B2"), ("turn_splitting", "turn-splitting", "#E69F00"),
         ("content", "content per turn", "#CC79A7"), ("skill_text", "skill text", "#D55E00")]
BENCH = [("tau3", r"$\tau^3$-bench"), ("AutoB", "AutoBench"), ("TB2", "TB2")]
METHODS = ["GEPA", "SkillOpt", "SkillAdam"]
BAR, DY, PITCH = 0.38, 0.22, 1.0


def main():
    data = json.loads((MC.CACHE / "failcost.json").read_text())
    cells = {(c["method"], c["short"]): c for c in data["cells"]}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    rows, numbers = [], []
    y = 0.0
    for bi, (short, bname) in enumerate(BENCH):
        top = y
        for m in METHODS:
            for sub, dy in (("solve", -DY), ("fail", DY)):
                s = cells[(m, short)]["subsets"][sub]
                pos = neg = 0.0
                for key, _, col in TERMS:
                    v = s["terms"][key]
                    left = pos if v >= 0 else neg
                    ax.barh(y + dy, v, left=left, height=BAR, color=col, linewidth=0)
                    if v >= 0:
                        pos += v
                    else:
                        neg += v
                ax.plot([s["total"]] * 2, [y + dy - BAR / 2 - 0.03, y + dy + BAR / 2 + 0.03], color="#333333", lw=1.1, solid_capstyle="butt")
                ax.text(max(pos, s["total"]) + 0.015, y + dy, f"n={s['n']}", va="center", ha="left", fontsize=7)
                numbers.append((m, short, sub, s["n"], {k: round(s["terms"][k], 3) for k, _, _ in TERMS}, round(s["total"], 3)))
            rows.append((y, m))
            y += PITCH
        ax.text(-0.235, (top + y - PITCH) / 2, bname, rotation=90, ha="center", va="center", fontsize=7, fontweight="bold",
                transform=ax.get_yaxis_transform(), clip_on=False)
        if bi < 2:
            ax.axhline(y - PITCH / 2, color="#BBBBBB", lw=0.5, xmin=0, xmax=1)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_xlim(-0.25, 0.9)
    ax.set_xlabel(r"change in $\log$ input tokens, skill vs. seed")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    h = [Patch(color=c) for _, _, c in TERMS] + [Line2D([0], [0], color="#333333", lw=1.1)]
    lab = [l for _, l, _ in TERMS] + ["total"]
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=5, frameon=False, fontsize=7, handlelength=1.0,
               columnspacing=1.0, handletextpad=0.4)
    fig.text(0.55, 0.905, "upper bar of each pair: tasks both solve 3/3; lower bar: tasks both fail 3/3", ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.86, bottom=0.14)
    MC.save_figure(fig, "failcost")
    rec = MC.md_table(["skill", "bench", "subset", "n", "terms", "total"], numbers)
    MC.write_record("failcost_fig", "Four-term input-token tax per skill on tasks both skill and seed solve 3/3 (upper bar) and both fail 3/3 (lower bar); n at bar end, tick = total.",
                    "deepseek-flash; nine learned skills vs. seed, 3 runs each (see failcost.py).", "3-run per-task means, mean over tasks; terms of attribution.py.",
                    rec, "python3 -m analysis.mech.failcost_fig")
    return numbers


if __name__ == "__main__":
    main()
