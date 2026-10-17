"""Paper 1 figure mech_runs: the four terms of the log input-token ratio on each of the three single-run pairs of the nine learned skills.
One thin stacked bar per run pair (positive terms right of 0, negative left), black tick = measured total, white-ringed dot on the largest term.
Data: assets/cache/analysis/mech/attribution.json cells[*].B.
Draw: python3 -m analysis.mech.runs_fig  -> fig/mech/mech_runs.{pdf,svg}, data/mech/mech_runs.md"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.mech import common as MC

OPT = [("GEPA", "GEPA", "#009E73"), ("SkillOpt", "SkillOpt", "#0072B2"), ("SkillAdam", "SkillAdam", "#D55E00")]
BENCH = [("$\\tau^3$-bench", "tau3-bench"), ("AutoBench", "automationbench"), ("TB2", "terminal-bench-2-1")]
TERMS = [("more_actions", "more actions", "#0072B2"), ("turn_splitting", "turn-splitting", "#E69F00"),
         ("content", "content per turn", "#CC79A7"), ("skill_text", "skill text", "#D55E00")]


def draw():
    d = json.load(open(MC.CACHE / "attribution.json"))
    cell = {(c["method"], c["bench"]): c for c in d["cells"]}
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ypos, ylab, ogrp, rows = [], [], [], []
    y = 0.0
    for oname, okey, ocol in OPT:
        y0 = y
        for bdisp, bkey in BENCH:
            c = cell[(okey, bkey)]
            ys = []
            for b in c["B"]:
                ys.append(y)
                left_pos = left_neg = 0.0
                for key, _, col in TERMS:
                    v = b[key]
                    if v >= 0:
                        ax.barh(y, v, left=left_pos, height=0.8, color=col, linewidth=0)
                        centre = left_pos + v / 2
                        left_pos += v
                    else:
                        ax.barh(y, v, left=left_neg, height=0.8, color=col, linewidth=0)
                        centre = left_neg + v / 2
                        left_neg += v
                    if key == b["largest"]:
                        ax.plot(centre, y, "o", ms=2.6, mfc="white", mec="black", mew=0.5, zorder=4)
                ax.plot([b["total"]] * 2, [y - 0.5, y + 0.5], color="black", lw=1.0, zorder=5, solid_capstyle="butt")
                rows.append((oname, bdisp, b["run"], [b[k] for k, _, _ in TERMS], b["total"], b["largest"]))
                y += 1
            ypos.append(sum(ys) / 3)
            ylab.append(bdisp)
            y += 0.55
        ogrp.append((oname, ocol, y0, y - 0.55 - 1))
        y += 0.45
    ax.axvline(0, color="#555", lw=0.5, zorder=0)
    ax.set_ylim(y - 0.9, -0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.5)
    ax.set_xlim(-0.2, 0.6)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    ax.set_xlabel("change in log input tokens")
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    for oname, ocol, ya, yb in ogrp:
        ax.annotate(oname, xy=(0, (ya + yb) / 2), xycoords=("axes fraction", "data"), xytext=(-52, 0), textcoords="offset points",
                    ha="center", va="center", rotation=90, fontweight="bold", color=ocol, annotation_clip=False)
    handles = [Patch(facecolor=c, label=l) for _, l, c in TERMS]
    handles += [Line2D([], [], color="black", lw=1.0, label="measured total"),
                Line2D([], [], marker="o", ls="", ms=2.6, mfc="white", mec="black", mew=0.5, label="largest term")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, handlelength=1.0, handletextpad=0.4,
               columnspacing=0.9, bbox_to_anchor=(0.54, 1.0))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.925, bottom=0.17)
    MC.save_figure(fig, "runs")
    head = ["optimizer", "benchmark", "run"] + [t[1] for t in TERMS] + ["total", "largest"]
    md = MC.md_table(head, [[o, b, r] + [f"{v:+.3f}" for v in t] + [f"{tot:+.3f}", lg] for o, b, r, t, tot, lg in rows])
    MC.write_record("runs", "The four terms (log input tokens) on each of the three single-run pairs, per learned skill; bar = stacked terms (positive right, negative left), tick = measured total, dot = largest term.",
                    "native-20261008-https, deepseek-flash; GEPA, SkillOpt, SkillAdam x tau3 / AutoBench / TB2; run pairs 0, 1, 2 (attribution.json cells[*].B).",
                    "None; values per single run pair as stored by analysis.mech.attribution.", md,
                    commands="python3 -m analysis.mech.runs_fig")


if __name__ == "__main__":
    draw()
