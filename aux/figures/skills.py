"""The learned-skills figure (DeepSeek): (a) the four-term split of each skill's change in log input tokens, (b) the rules it adds by category.

Rows GEPA, SkillOpt, SkillAdam x tau3, AutoBench, TB2. (a) reads attribution.json (python -m src.analysis.attribution compute), (b) reads
skilltext.json (python -m aux.rule_coding.skilltext compute); 'parallelize' is folded into 'other'. Writes <cache>/analysis/main/skills2.json
and fig/mech/mech_skills2, drawn at its printed width (0.82 of the text width).
"""
import json

import numpy as np

from src.utils import dataloader as C
from src.utils.fig import save_figure

PRINT_PT = 0.82 * 455.24
FIG_W, FIG_H = PRINT_PT / 72, 2.66
FS = 7.0
OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
OPT_KEY = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}
BENCHES = [("tau3-bench", "τ³-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
TERMS = ["more_actions", "turn_splitting", "content", "skill_text", "total"]
TERM_HEAD = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]
ORDER = ["verify", "serialize", "ask_stop", "domain", "format", "other"]
LABEL = {"verify": "verify", "serialize": "one at a time", "ask_stop": "ask or stop", "domain": "domain facts", "format": "format", "other": "other"}
COLOR = {"verify": "#6a3d9a", "serialize": "#a6761d", "ask_stop": "#969696", "domain": "#c9b99a", "format": "#ece6d6", "other": "#d4d4d4"}
OPT_COLOR = {"GEPA": "#2ca02c", "SkillOpt": "#1f77b4", "SkillAdam": "#ff7f0e"}
GAP = 0.45


def rows():
    out, y = [], 0.0
    for opt in OPTS:
        for bench, label in BENCHES:
            out.append((opt, bench, label, y))
            y += 1
        y += GAP
    return out


def draw(cells, text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": FS,
                         "axes.linewidth": 0.6})
    R = rows()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y0, h = 0.47, 1.8
    ax_a = fig.add_axes([0.67 / FIG_W, y0 / FIG_H, 1.86 / FIG_W, h / FIG_H])
    ax_b = fig.add_axes([2.8 / FIG_W, y0 / FIG_H, 1.0 / FIG_W, h / FIG_H])
    for ax in (ax_a, ax_b):
        ax.set_ylim(R[-1][3] + 0.6, -0.5)

    # (a) four-term split
    by_key = {(c["method"], c["bench"]): c["A"] for c in cells}
    values = np.array([[by_key[(opt, bench)][t] for t in TERMS] for opt, bench, _, _ in R])
    vmax = np.abs(values).max()
    norm, cmap = matplotlib.colors.Normalize(-vmax, vmax), plt.get_cmap("RdBu_r")
    for (_, _, _, y), line in zip(R, values):
        for j, v in enumerate(line):
            ax_a.add_patch(Rectangle((j, y - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            ax_a.text(j + 0.5, y, f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00"), ha="center", va="center", fontsize=FS,
                      color="white" if abs(v) > 0.62 * vmax else "k", fontweight="bold" if j == len(TERMS) - 1 else "normal")
    ax_a.set_xlim(0, len(TERMS))
    for j, head in enumerate(TERM_HEAD):
        ax_a.text(j + 0.5, -0.62, head, ha="center", va="bottom", fontsize=FS, linespacing=0.95)
    ax_a.set_xticks([])
    ax_a.set_yticks([y for _, _, _, y in R])
    ax_a.set_yticklabels([label for _, _, label, _ in R], fontsize=FS)
    ax_a.tick_params(axis="y", length=0, pad=2)
    for side in ax_a.spines.values():
        side.set_visible(False)
    for k, opt in enumerate(OPTS):
        ax_a.annotate(opt, xy=(0, R[3 * k + 1][3]), xycoords=("axes fraction", "data"), xytext=(-41, 0), textcoords="offset points",
                      ha="center", va="center", rotation=90, fontsize=FS + 0.5, fontweight="bold", color=OPT_COLOR[opt], annotation_clip=False)

    # (b) rules added by category, with margin columns
    margin = matplotlib.transforms.blended_transform_factory(ax_b.transAxes, ax_b.transData)
    columns = {"verify": 1.2, "serialize": 1.56, "limit": 1.9, "words": 2.24}
    numbers = []
    for (opt, bench, _, y), line in zip(R, values):
        key = f"{OPT_KEY[opt]}|{bench}"
        counts = dict(text["counts"][key])
        counts["other"] = counts.get("other", 0) + counts.get("parallelize", 0)
        left = 0
        for cat in ORDER:
            if counts.get(cat, 0):
                ax_b.barh(y, counts[cat], left=left, height=0.72, color=COLOR[cat], edgecolor="white", linewidth=0.4)
                left += counts[cat]
        ratio = text["sizes"][key]["words_ratio"]
        for name, value, colour, weight in (("verify", counts.get("verify", 0), COLOR["verify"], "bold"),
                                            ("serialize", counts.get("serialize", 0), COLOR["serialize"], "bold"),
                                            ("limit", counts.get("limit_read", 0), "#444", "normal"), ("words", f"×{ratio:.1f}", "k", "normal")):
            ax_b.text(columns[name], y, str(value), transform=margin, ha="center", va="center", color=colour, fontweight=weight, fontsize=FS)
        numbers.append(dict(optimizer=opt, benchmark=bench, terms=dict(zip(TERMS, map(float, line))),
                            rules_by_category={c: counts.get(c, 0) for c in ORDER}, rules_total=int(left),
                            limit_read=counts.get("limit_read", 0), length_ratio=ratio))
    for name, head, colour in [("verify", "verify", COLOR["verify"]), ("serialize", "one at\na time", COLOR["serialize"]),
                               ("limit", "limit\nread", "#444"), ("words", "length\n× seed", "black")]:
        ax_b.text(columns[name], -0.62, head, transform=margin, ha="center", va="bottom", color=colour, fontsize=FS, linespacing=0.95,
                  clip_on=False)
    ax_b.set_xlim(0, 105)
    ax_b.set_xticks([0, 35, 70, 105])
    ax_b.tick_params(axis="x", labelsize=FS, length=2, pad=1)
    ax_b.set_xlabel("rules added", fontsize=FS, labelpad=1)
    ax_b.set_yticks([])
    for side in ("top", "right", "left"):
        ax_b.spines[side].set_visible(False)

    title_y = (y0 + h + 0.29) / FIG_H
    fig.text(0.67 / FIG_W, title_y, "(a) Where the extra input tokens go", fontsize=7.5, fontweight="bold", va="bottom")
    fig.text(2.8 / FIG_W, title_y, "(b) What the optimizers write", fontsize=7.5, fontweight="bold", va="bottom")
    fig.legend(handles=[Patch(facecolor=COLOR[c], label=LABEL[c]) for c in ORDER], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6,
               frameon=False, fontsize=FS, handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    save_figure(fig, "skills2")
    return dict(colour_scale_max_abs=float(vmax), rows=numbers)


def main():
    cells = json.loads((C.CACHE / "attribution.json").read_text())["cells"]
    text = json.loads((C.CACHE / "skilltext.json").read_text())
    numbers = draw(cells, text)
    C.write_json("skills2", numbers)
    for n in numbers["rows"]:
        print(n["optimizer"], n["benchmark"], {k: round(v, 2) for k, v in n["terms"].items()}, n["rules_total"],
              n["rules_by_category"]["verify"], n["rules_by_category"]["serialize"], round(n["length_ratio"], 1))


if __name__ == "__main__":
    main()
