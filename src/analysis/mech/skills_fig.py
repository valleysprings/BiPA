"""Paper 1 (mechanism paper), Section 4.3: one figure, one row per learned skill, that puts where its token tax goes next to what its optimizer wrote.

Role in the argument: Findings 2 and 4 are read on the same nine skills. Side by side they show the link the text draws, e.g. the AutoBench skills whose tax is
turn-splitting carry the serialize rules, and SkillOpt's tau3-bench skill, four times the seed's length, is the one whose skill-text term is large.
  (a) four-term split of each skill's change in log input tokens against the seed (attribution.json, as mech_attribution3) and the measured total
  (b) rules added to the seed by coded category (skilltext.json, as mech_rules), with the verify and serialize counts and the skill's words relative to the seed
Rows: GEPA, SkillOpt, SkillAdam x tau3-bench, AutoBench, TB2, the order of Table 1. The numbers are those of the records mech_attribution3.md and mech_rules.md.

Command (from bipa):  python3 -m src.analysis.mech.skills_fig  -> fig/mech/mech_skills.{pdf,svg}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.utils import fig, dataloader
from src.analysis.mech import rules_fig as RF

OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
BENCHES = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
TERMS = ["more_actions", "turn_splitting", "content", "skill_text", "total"]
TERM_HEAD = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]
GAP = 0.45   # vertical gap between optimizer groups, in rows


def row_positions():
    """y position of each (optimizer, benchmark) row, with a gap between optimizer groups."""
    rows = []
    y = 0.0
    for opt in OPTS:
        for bench, label in BENCHES:
            rows.append((opt, bench, label, y))
            y += 1
        y += GAP
    return rows, y - GAP


def draw_split(ax, rows, cells):
    """(a) Heatmap of the four terms and the measured total, drawn as rectangles so that its rows share the y positions of panel (b)."""
    by_key = {(c["method"], c["bench"]): c["A"] for c in cells}
    values = np.array([[by_key[(opt, bench)][t] for t in TERMS] for opt, bench, _, _ in rows])
    vmax = np.abs(values).max()
    norm = matplotlib.colors.Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdBu_r")
    for (opt, bench, label, y), line in zip(rows, values):
        for j, v in enumerate(line):
            ax.add_patch(plt.Rectangle((j, y - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            # white text on the darkest cells, bold for the measured total
            color = "white" if abs(v) > 0.62 * vmax else "k"
            weight = "bold" if j == len(TERMS) - 1 else "normal"
            ax.text(j + 0.5, y, f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00"), ha="center", va="center", fontsize=6.5, color=color, fontweight=weight)
    ax.set_xlim(0, len(TERMS))
    for j, head in enumerate(TERM_HEAD):
        ax.text(j + 0.5, -0.65, head, ha="center", va="bottom", fontsize=6.3, linespacing=0.95)
    ax.set_xticks([])
    ax.set_yticks([y for _, _, _, y in rows])
    ax.set_yticklabels([label for _, _, label, _ in rows], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    # optimizer name once per group, left of the benchmark labels
    for k, opt in enumerate(OPTS):
        mid = rows[3 * k + 1][3]
        ax.annotate(opt, xy=(0, mid), xycoords=("axes fraction", "data"), xytext=(-40, 0), textcoords="offset points", ha="center", va="center",
                    rotation=90, fontsize=7, fontweight="bold", color=RF.OPT_COLOR[opt], annotation_clip=False)


def draw_rules(ax, rows, text):
    """(b) Stacked bars of the rules each skill adds by category, with verify and serialize counts and words relative to the seed in margin columns."""
    keys = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}
    margin = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    columns = {"verify": 1.12, "serialize": 1.36, "limit": 1.59, "words": 1.84}
    for opt, bench, _, y in rows:
        key = f"{keys[opt]}|{bench}"
        counts = text["counts"][key]
        left = 0
        for cat in RF.ORDER:
            n = counts.get(cat, 0)
            # skip empty categories so no zero-width segment draws an edge
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.72, color=RF.COLOR[cat], edgecolor="white", linewidth=0.4)
            left += n
        ratio = text["sizes"][key]["words_ratio"]
        ax.text(columns["verify"], y, str(counts.get("verify", 0)), transform=margin, ha="center", va="center", color=RF.COLOR["verify"], fontweight="bold", fontsize=7)
        ax.text(columns["serialize"], y, str(counts.get("serialize", 0)), transform=margin, ha="center", va="center", color=RF.COLOR["serialize"], fontweight="bold", fontsize=7)
        ax.text(columns["limit"], y, "0", transform=margin, ha="center", va="center", color="#444", fontsize=7)
        ax.text(columns["words"], y, f"×{ratio:.1f}", transform=margin, ha="center", va="center", fontsize=7)
    for name, head, color in [("verify", "verify", RF.COLOR["verify"]), ("serialize", "one at\na time", RF.COLOR["serialize"]),
                              ("limit", "limit\nread", "#444"), ("words", "length\n\u00d7 seed", "black")]:
        ax.text(columns[name], -0.65, head, transform=margin, ha="center", va="bottom", color=color, fontsize=6.3, linespacing=0.95, clip_on=False)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 35, 70, 105])
    ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
    ax.set_xlabel("rules added", fontsize=7, labelpad=1)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main():
    """Load the two caches and draw the full-width figure."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})
    cells = json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"]
    text = json.loads((dataloader.CACHE / "skilltext.json").read_text())
    rows, bottom = row_positions()

    fig = plt.figure(figsize=(6.75, 2.05))
    ax_a = fig.add_axes([0.105, 0.115, 0.36, 0.70])
    ax_b = fig.add_axes([0.545, 0.115, 0.235, 0.70])
    for ax in (ax_a, ax_b):
        ax.set_ylim(bottom + 0.5, -0.5)
    draw_split(ax_a, rows, cells)
    draw_rules(ax_b, rows, text)
    fig.text(0.105, 0.985, "(a) Where the token tax goes", fontsize=7.5, fontweight="bold", va="top")
    fig.text(0.545, 0.985, "(b) What the optimizers write", fontsize=7.5, fontweight="bold", va="top")
    # the serialize category reads "one at a time" in this figure, as its column head
    labels = dict(RF.LABEL, serialize="one at a time")
    handles = [Patch(facecolor=RF.COLOR[c], label=labels[c]) for c in RF.ORDER]
    # the SkillAdam rows are short, so the legend sits in the empty lower right of panel (b)
    ax_b.legend(handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=6.3, handlelength=0.9, handletextpad=0.3,
                columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1)
    fig.save_figure(fig, "skills")


if __name__ == "__main__":
    main()
