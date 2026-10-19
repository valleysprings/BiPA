"""mech_edges: what an edit says and what it changes. For each of the ten keyword groups of the 59 canonical dev edits, the difference (edits whose added text
contains the group minus edits without it, unweighted mean over edits) in the actions term and in the turn-splitting term of the change in log input tokens,
(the 95% bootstrap intervals over edits are kept in the record, not drawn). Groups sorted by the actions difference; labels carry the with/without edit counts.
Reads skilltext.json (emp_baseline_check.category_presence); no .venv needed.
  python3 -m src.analysis.mech.edges_fig"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import fig, logger, dataloader
GROUPS = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
# plain names of the keyword groups (patterns in edges_build.CATEGORY_PATTERNS); "one at a time" as in the rules figure
NAME = {"verify": "verify", "reread": "re-read state", "enumerate": "check every item", "test": "run tests", "format": "output format",
        "business": "domain rules", "scope_stop": "stop or budget", "economize": "save calls or reads", "serialize": "one at a time", "other": "other"}
TERMS = [("actions", "more actions", "#0072B2", "o"), ("turnsplit", "turn-splitting", "#E69F00", "s")]


def draw():
    pres = json.load(open(dataloader.CACHE / "skilltext.json"))["emp_baseline_check"]["category_presence"]
    holm = {(t["category"], t["term"]): t for t in json.load(open(dataloader.CACHE / "edges_holm.json"))["tests"]}
    term_name = {"actions": "more actions", "turnsplit": "turn-splitting"}
    # rows sorted by their largest difference, biggest effect on top
    rows = sorted(GROUPS, key=lambda g: -max(pres[g]["contrast"]["actions"][0], pres[g]["contrast"]["turnsplit"][0]))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"], "font.size": 7,
                         "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    off = {"actions": -0.17, "turnsplit": 0.17}
    ax.axvline(0, color="black", lw=1.2, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.92", lw=0.5)
    table = []
    for i, g in enumerate(rows):
        line = [NAME.get(g, g), pres[g]["edges_with"], pres[g]["edges_without"]]
        for key, _, col, mk in TERMS:
            d, lo, hi = pres[g]["contrast"][key]
            y = i + off[key]
            h = holm[(g, term_name[key])]
            thin = h["without"] < 5 and h["survives"]  # survives, but the comparison group is tiny
            if thin:
                ax.scatter([d], [y], s=20, marker=mk, facecolor="white", edgecolor="0.55", lw=0.9, zorder=3)
                ax.annotate("only 2 edits without", (d, y), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                            fontsize=6.5, color="0.4")
            elif h["survives"]:
                ax.scatter([d], [y], s=30, marker=mk, color=col, edgecolor="black", lw=0.6, zorder=4)
                ax.annotate(f"{d:+.2f}", (d, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=7, weight="bold")
            else:
                ax.scatter([d], [y], s=16, marker=mk, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
            line += [d, lo, hi]
        table.append(line)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NAME.get(g, g)}  {pres[g]['edges_with']}/{pres[g]['edges_without']}" for g in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.2, 0.26)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xticklabels(["$-$0.2", "$-$0.1", "0", "0.1", "0.2"])
    ax.set_xlabel("edits with the group minus edits without, change in log input tokens")
    ax.spines[["top", "right", "left"]].set_visible(False)
    h = [plt.Line2D([], [], marker=mk, ls="none", color=col, ms=4.5, mec="black", mew=0.5) for _, _, col, mk in TERMS]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", ms=4.5, mew=0.9))
    fig.legend(h, ["more actions", "turn-splitting", "hollow: not reliable after correcting for 20 comparisons"],
               loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.24, right=0.985, top=0.9, bottom=0.17)
    fig.save_figure(fig, "edges")
    body = logger.md_table(["group", "with", "without", "d actions", "[95%]", "d turn-splitting", "[95%]"],
                      [[r[0], r[1], r[2], f"{r[3]*100:+.1f}", f"[{r[4]*100:.1f}, {r[5]*100:.1f}]", f"{r[6]*100:+.1f}", f"[{r[7]*100:.1f}, {r[8]*100:.1f}]"] for r in table])
    logger.write_record("edges_fig", "Per keyword group of the added text, the with-minus-without difference of the actions and turn-splitting terms of the change in log input tokens per edit (59 dev edits), with 95% bootstrap intervals over edits. Replaces the appendix table of edit categories.",
                   "59 canonical dev edges of native-20261008-https (SkillAdam x TB2 excluded); an edit is 'with' a group when its added text holds >= 150 characters of the group's keywords.",
                   "Unweighted mean over edits, with minus without; 95% bootstrap over edits (3000 resamples, each side within itself). Values x100 log points in this table; the figure axis is in log units.",
                   body, "python3 -m src.analysis.mech.edges_fig")
    return table


if __name__ == "__main__":
    for r in draw():
        print(r)
