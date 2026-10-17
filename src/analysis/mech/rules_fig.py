"""Paper 1 figure mech_rules: what the optimizers write. One stacked bar per learned skill (3 optimizers x 3 benchmarks); segments are the coded rule
categories of the units the optimizer added to the seed (analysis.mech.skilltext coded_units); text right of each bar = skill words relative to the seed.
Verify and serialize carry the colour; limit-reading has zero rules in all nine skills and is shown as a right-margin column.
Draw: python3 -m src.analysis.mech.rules_fig  -> fig/mech/mech_rules.{pdf,svg}, data/mech/mech_rules.md"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.analysis.mech import common as MC

OPT = [("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam")]
BENCH = [("$\\tau^3$-bench", "tau3-bench"), ("AutoBench", "automationbench"), ("TB2", "terminal-bench-2-1")]
ORDER = ["verify", "serialize", "ask_stop", "domain", "format", "other", "parallelize"]
LABEL = {"verify": "verify", "serialize": "serialize", "ask_stop": "ask or stop", "domain": "domain facts",
         "format": "format", "other": "other", "parallelize": "parallelize"}
COLOR = {"verify": "#c0392b", "serialize": "#2e6fb7", "ask_stop": "#bdbdbd", "domain": "#d9d9d9",
         "format": "#e9e4d8", "other": "#f1f1f1", "parallelize": "#7a7a7a"}
OPT_COLOR = {"GEPA": "#2ca02c", "SkillOpt": "#1f77b4", "SkillAdam": "#ff7f0e"}
OPT_MARK = {"GEPA": "^", "SkillOpt": "o", "SkillAdam": "s"}


def draw():
    d = json.load(open(MC.CACHE / "skilltext.json"))
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "mathtext.default": "regular"})
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    rows, ypos, y = [], [], 0.0
    for oi, (od, ok) in enumerate(OPT):
        for bd, bk in BENCH:
            rows.append((od, bd, f"{ok}|{bk}"))
            ypos.append(y)
            y += 1
        y += 0.45
    table = []
    # right-margin columns in axes-fraction x: verify count, serialize count, limit-reading count, words vs seed
    cols = dict(verify=1.17, serialize=1.50, limit=1.82, words=2.17)
    tr = None
    tx = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for (od, bd, key), y in zip(rows, ypos):
        left = 0
        counts = d["counts"][key]
        for cat in ORDER:
            n = counts.get(cat, 0)
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.72, color=COLOR[cat], edgecolor="white", linewidth=0.4)
            left += n
        ratio = d["sizes"][key]["words_ratio"]
        ax.text(cols["verify"], y, str(counts.get("verify", 0)), transform=tx, ha="center", va="center", color=COLOR["verify"], fontweight="bold")
        ax.text(cols["serialize"], y, str(counts.get("serialize", 0)), transform=tx, ha="center", va="center", color=COLOR["serialize"], fontweight="bold")
        ax.text(cols["limit"], y, "0", transform=tx, ha="center", va="center", color="#444")
        ax.text(cols["words"], y, f"\u00d7{ratio:.1f}", transform=tx, ha="center", va="center")
        table.append((od, bd, [counts.get(c, 0) for c in ORDER], ratio))
    ax.set_ylim(y + 0.55, -0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 35, 70, 105])
    ax.set_xlabel("rules added")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for od, _ in OPT:
        mid = ypos[3 * [o for o, _ in OPT].index(od) + 1]
        ax.annotate(od, xy=(0, mid), xycoords=("axes fraction", "data"), xytext=(-47, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=7, fontweight="bold", color=OPT_COLOR[od], rotation=90, annotation_clip=False)
    for name, txt, colr in [("verify", "verify", COLOR["verify"]), ("serialize", "serial-\nize", COLOR["serialize"]),
                            ("limit", "limit\nread", "#444"), ("words", "words/\nseed", "black")]:
        ax.text(cols[name], -0.7, txt, transform=tx, ha="center", va="bottom", color=colr, linespacing=0.95, clip_on=False)
    handles = [Patch(facecolor=COLOR[c], label=LABEL[c]) for c in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7, handlelength=1, handletextpad=0.4,
               columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.265, right=0.575, top=0.79, bottom=0.13)
    MC.save_figure(fig, "rules")
    head = ["optimizer", "benchmark"] + ORDER + ["limit_read", "words / seed"]
    md = MC.md_table(head, [[o, b] + c + [0, f"{r:.2f}"] for o, b, c, r in table])
    MC.write_record("rules", "Rules the optimizers add to the seed skill, by coded category (stacked bars, one per learned skill); text = words of the learned skill / words of the seed (580). limit reading = 0 in all nine skills.",
                    "native-20261008-https, deepseek-flash; GEPA, SkillOpt, SkillAdam x tau3 / AutoBench / TB2.",
                    "Count of coded units per final category (skilltext.json coded_units).", md,
                    commands="python3 -m src.analysis.mech.rules_fig")


if __name__ == "__main__":
    draw()
