"""Rule-coding figures (mech_rules, mech_skills) and the by-hand review listing of the coded units.

Run:  python -m aux.rule_coding.figures rules|skills|review
"""
import argparse
import json

from src.utils import fig, logger
from src.utils import dataloader as C


def review():
    """Print every coded unit with its first-pass and final category (the by-hand check)."""
    from src.analysis.skilltext import BENCHES, OPTS, code_cell, final_texts
    texts = final_texts()
    for opt, _ in OPTS:
        for bench, _ in BENCHES:
            run, text = texts[(opt, bench)]
            coded, structure = code_cell(opt, bench, text)
            print(f"##### {opt} {bench} units={len(coded)} structure={structure}")
            for c in coded:
                mark = "*" if c["first_pass"] != c["category"] else " "
                print(f"[{c['idx']}]{mark}{c['category']:11s} | {c['text'][:200]}")


OPT = [("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam")]
BENCH = [("$\\tau^3$-bench", "tau3-bench"), ("AutoBench", "automationbench"), ("TB2", "terminal-bench-2-1")]
ORDER = ["verify", "serialize", "ask_stop", "domain", "format", "other", "parallelize"]
LABEL = {"verify": "verify", "serialize": "serialize", "ask_stop": "ask or stop", "domain": "domain facts",
         "format": "format", "other": "other", "parallelize": "parallelize"}
COLOR = {"verify": "#c0392b", "serialize": "#2e6fb7", "ask_stop": "#bdbdbd", "domain": "#d9d9d9",
         "format": "#e9e4d8", "other": "#f1f1f1", "parallelize": "#7a7a7a"}
OPT_COLOR = {"GEPA": "#2ca02c", "SkillOpt": "#1f77b4", "SkillAdam": "#ff7f0e"}
OPT_MARK = {"GEPA": "^", "SkillOpt": "o", "SkillAdam": "s"}


def rules_draw():
    """mech_rules: rules each learned skill adds to the seed, by coded category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    d = json.load(open(C.CACHE / "skilltext.json"))
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "mathtext.default": "regular"})
    f, ax = plt.subplots(figsize=(3.25, 2.8))
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
    f.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7, handlelength=1, handletextpad=0.4,
             columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    f.subplots_adjust(left=0.265, right=0.575, top=0.79, bottom=0.13)
    fig.save_figure(f, "rules")
    head = ["optimizer", "benchmark"] + ORDER + ["limit_read", "words / seed"]
    md = logger.md_table(head, [[o, b] + c + [0, f"{r:.2f}"] for o, b, c, r in table])
    logger.write_record("rules", "Rules the optimizers add to the seed skill, by coded category (stacked bars, one per learned skill); text = words of the learned skill / words of the seed (580). limit reading = 0 in all nine skills.",
                    "native-20261008-https, deepseek-flash; GEPA, SkillOpt, SkillAdam x tau3 / AutoBench / TB2.",
                    "Count of coded units per final category (skilltext.json coded_units).", md,
                    commands="python -m aux.rule_coding.figures rules")


SKILLS_OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
SKILLS_BENCHES = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
SKILLS_TERMS = ["more_actions", "turn_splitting", "content", "skill_text", "total"]
SKILLS_TERM_HEAD = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]
GAP = 0.45   # vertical gap between optimizer groups, in rows


def skills_row_positions():
    """y position of each (optimizer, benchmark) row, with a gap between optimizer groups."""
    rows = []
    y = 0.0
    for opt in SKILLS_OPTS:
        for bench, label in SKILLS_BENCHES:
            rows.append((opt, bench, label, y))
            y += 1
        y += GAP
    return rows, y - GAP


def skills_draw_split(ax, rows, cells):
    """(a) Heatmap of the four terms and the measured total, drawn as rectangles so that its rows share the y positions of panel (b)."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    by_key = {(c["method"], c["bench"]): c["A"] for c in cells}
    values = np.array([[by_key[(opt, bench)][t] for t in SKILLS_TERMS] for opt, bench, _, _ in rows])
    vmax = np.abs(values).max()
    norm = matplotlib.colors.Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdBu_r")
    for (opt, bench, label, y), line in zip(rows, values):
        for j, v in enumerate(line):
            ax.add_patch(plt.Rectangle((j, y - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            # white text on the darkest cells, bold for the measured total
            color = "white" if abs(v) > 0.62 * vmax else "k"
            weight = "bold" if j == len(SKILLS_TERMS) - 1 else "normal"
            ax.text(j + 0.5, y, f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00"), ha="center", va="center", fontsize=6.5, color=color, fontweight=weight)
    ax.set_xlim(0, len(SKILLS_TERMS))
    for j, head in enumerate(SKILLS_TERM_HEAD):
        ax.text(j + 0.5, -0.65, head, ha="center", va="bottom", fontsize=6.3, linespacing=0.95)
    ax.set_xticks([])
    ax.set_yticks([y for _, _, _, y in rows])
    ax.set_yticklabels([label for _, _, label, _ in rows], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    # optimizer name once per group, left of the benchmark labels
    for k, opt in enumerate(SKILLS_OPTS):
        mid = rows[3 * k + 1][3]
        ax.annotate(opt, xy=(0, mid), xycoords=("axes fraction", "data"), xytext=(-40, 0), textcoords="offset points", ha="center", va="center",
                    rotation=90, fontsize=7, fontweight="bold", color=OPT_COLOR[opt], annotation_clip=False)


def skills_draw_rules(ax, rows, text):
    """(b) Stacked bars of the rules each skill adds by category, with verify and serialize counts and words relative to the seed in margin columns."""
    import matplotlib
    keys = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}
    margin = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    columns = {"verify": 1.12, "serialize": 1.36, "limit": 1.59, "words": 1.84}
    for opt, bench, _, y in rows:
        key = f"{keys[opt]}|{bench}"
        counts = text["counts"][key]
        left = 0
        for cat in ORDER:
            n = counts.get(cat, 0)
            # skip empty categories so no zero-width segment draws an edge
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.72, color=COLOR[cat], edgecolor="white", linewidth=0.4)
            left += n
        ratio = text["sizes"][key]["words_ratio"]
        ax.text(columns["verify"], y, str(counts.get("verify", 0)), transform=margin, ha="center", va="center", color=COLOR["verify"], fontweight="bold", fontsize=7)
        ax.text(columns["serialize"], y, str(counts.get("serialize", 0)), transform=margin, ha="center", va="center", color=COLOR["serialize"], fontweight="bold", fontsize=7)
        ax.text(columns["limit"], y, "0", transform=margin, ha="center", va="center", color="#444", fontsize=7)
        ax.text(columns["words"], y, f"×{ratio:.1f}", transform=margin, ha="center", va="center", fontsize=7)
    for name, head, color in [("verify", "verify", COLOR["verify"]), ("serialize", "one at\na time", COLOR["serialize"]),
                              ("limit", "limit\nread", "#444"), ("words", "length\n\u00d7 seed", "black")]:
        ax.text(columns[name], -0.65, head, transform=margin, ha="center", va="bottom", color=color, fontsize=6.3, linespacing=0.95, clip_on=False)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 35, 70, 105])
    ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
    ax.set_xlabel("rules added", fontsize=7, labelpad=1)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def skills_draw():
    """Load the two caches and draw the full-width figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})
    cells = json.loads((C.CACHE / "attribution.json").read_text())["cells"]
    text = json.loads((C.CACHE / "skilltext.json").read_text())
    rows, bottom = skills_row_positions()

    f = plt.figure(figsize=(6.75, 2.05))
    ax_a = f.add_axes([0.105, 0.115, 0.36, 0.70])
    ax_b = f.add_axes([0.545, 0.115, 0.235, 0.70])
    for ax in (ax_a, ax_b):
        ax.set_ylim(bottom + 0.5, -0.5)
    skills_draw_split(ax_a, rows, cells)
    skills_draw_rules(ax_b, rows, text)
    f.text(0.105, 0.985, "(a) Where the extra input tokens go", fontsize=7.5, fontweight="bold", va="top")
    f.text(0.545, 0.985, "(b) What the optimizers write", fontsize=7.5, fontweight="bold", va="top")
    # the serialize category reads "one at a time" in this figure, as its column head
    labels = dict(LABEL, serialize="one at a time")
    handles = [Patch(facecolor=COLOR[c], label=labels[c]) for c in ORDER]
    # the SkillAdam rows are short, so the legend sits in the empty lower right of panel (b)
    ax_b.legend(handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=6.3, handlelength=0.9, handletextpad=0.3,
                columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1)
    fig.save_figure(f, "skills")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["rules", "skills", "review"])
    {"rules": rules_draw, "skills": skills_draw, "review": review}[ap.parse_args().cmd]()


if __name__ == "__main__":
    main()
