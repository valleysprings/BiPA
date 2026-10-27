"""The figures of the edits: what the design principles rest on (mech_implications4) and what the learned DeepSeek skills
cost and contain (mech_skills2).

implications4: (a) edits kept under the four acceptance rules of mech_selrules, an unchanged candidate and one that truly
solves PLOT_GAIN more dev tasks (bar: mean over the benchmarks, line: range); (b) the share of an edit's (and a learned
skill's) change in log input tokens explained by skill length, turns and tool calls (mech_designev); (c) mean log rho per
benchmark as recorded and recomputed at the seed's calls per turn (mech_batching); (d) each edit's change in log input
tokens against its parent, rejected and kept, with the mean. skills2: (a) the four-term split of each skill's change in log
input tokens (mech_attribution3), (b) the rules it adds by category (mech_skilltext), parallelize counted under other.

They read the caches of aux.edits.gate (selrules.json, designev.json), aux.calls (batching.json), aux.edits.rules
(skilltext.json) and aux.tokens.split (attribution.json)."""
import json

import numpy as np

from src.analysis import settings as S
from src.utils.fig import (ARM, BENCH_LABEL, DARK, EDIT, INK, MID, RULE, TEXT, TEXT_W, despine, pyplot,
                           save_figure)

SIZE = 7.0
RULES = ("better", "critical", "confirm", "mean3")
RULE_LABELS = ("(1) one win\n(SkillOpt,\nSkillAdam)", "(2) one win\n≥ the critical\nvalue", "(3) two wins\nin a row", "(4) higher\n3-run total")
SEL_BENCHES = ("terminal-bench-2-1", "tau3-bench", "automationbench")
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
TITLES = ("(a) Rule (3), a repeated win, keeps few unchanged candidates", "(b) Turns, not skill length, explain an edit's token change",
          "(c) At the seed's calls per turn, AutoBench's log ρ halves", "(d) Kept and rejected edits add tokens alike")
TITLE_DX, TITLE_DY = (0.065, 0.13, 0.065, 0.065), 0.07
GAP = 0.45  # between the optimizers of skills2
SKILLS_W = 0.82 * 455.24 / 72  # skills2 is printed at 0.82 of the text width (455.24 pt)
SHOWN, PLAIN = ARM["skillopt"], EDIT["rejected"]  # the quantity a panel is about, against the grey it is compared with
TERMS = ("more_actions", "turn_splitting", "content", "skill_text", "total")
TERM_HEAD = ("more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total")
CATEGORIES = ("verify", "serialize", "ask_stop", "domain", "format", "other")
CATEGORY_LABEL = {"verify": "verify", "serialize": "serialize", "ask_stop": "ask or stop", "domain": "domain facts", "format": "format", "other": "other"}


def load(folder, name):
    return json.loads((S.cache(folder) / f"{name}.json").read_text())


def legend_above(ax, handles):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles), frameon=False, handlelength=0.9,
              handleheight=0.8, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.15, borderpad=0.0)


def rules_panel(ax, data, Patch):
    gain, width = str(data["plot_gain"]), 0.38
    for i, key in enumerate(RULES):
        for offset, values, colour in ((-width / 2, [data[b]["null"][key] for b in SEL_BENCHES], PLAIN),
                                       (width / 2, [data[b]["power"][gain][key] for b in SEL_BENCHES], SHOWN)):
            mean = 100 * np.mean(values)
            ax.bar(i + offset, mean, width=width, color=colour, edgecolor="none")
            # the range over the benchmarks, capped so it reads as lowest to highest
            ax.errorbar(i + offset, mean, yerr=[[mean - 100 * min(values)], [100 * max(values) - mean]], fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2, capthick=0.9)
    ax.set_xticks(range(len(RULES)), RULE_LABELS, linespacing=0.95)
    ax.set(xlim=(-0.6, len(RULES) - 0.4), ylim=(0, 102), yticks=[0, 25, 50, 75, 100])
    ax.set_ylabel("candidates accepted (%)", labelpad=1)
    ax.tick_params(axis="x", length=0, pad=2)
    legend_above(ax, [Patch(facecolor=PLAIN, label="unchanged candidate"),
                      Patch(facecolor=SHOWN, label=f"truly solves {gain} more dev tasks")])


def explained_panel(ax, data):
    """The edits only: the learned skills' R^2 (data["test_level"]) stays in mech_designev."""
    edge = data["edge_level"]
    values = 100 * np.array([edge["turns"]["r2"], edge["actions"]["r2"], edge["skill length"]["r2"]])
    colours, ypos = [SHOWN, PLAIN, PLAIN], np.array([2.0, 1.0, 0.0])
    ax.barh(ypos, values, color=colours, height=0.6)
    for y, v, colour in zip(ypos, values, colours):
        ax.text(v + 2, y, f"{v:.0f}%", va="center", color=SHOWN if colour == SHOWN else INK)
    ax.set(ylim=(-0.6, 2.6), xlim=(0, 109))
    ax.set_yticks(ypos, ["turns", "tool calls", "skill length"])
    for label, colour in zip(ax.get_yticklabels(), colours):
        label.set_color(SHOWN if colour == SHOWN else INK)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("variance of the token change explained, each on its own (%)", labelpad=1)
    ax.tick_params(axis="y", length=0)


def batching_panel(ax, data, Patch):
    groups = [[c for c in data["cells"] if c["bench"] == b] for b in BENCHES] + [data["cells"]]
    x, width = np.arange(len(groups)), 0.38
    recorded, packed = ([np.mean([c[key] for c in g]) for g in groups] for key in ("log_rho", "log_rho_cf"))
    ax.bar(x - width / 2, recorded, width, color=PLAIN)
    ax.bar(x + width / 2, packed, width, color=SHOWN)
    j = BENCHES.index("automationbench")  # the two numbers the text quotes
    for dx, v, colour in ((-width / 2, recorded[j], INK), (width / 2, packed[j], SHOWN)):
        ax.text(j + dx, v + 0.004, f"{v:.2f}", ha="center", va="bottom", color=colour)
    ax.axhline(0, color=DARK, lw=0.5)
    ax.set_xticks(x, [BENCH_LABEL[b] for b in BENCHES] + ["all nine"])
    ax.set_ylabel(r"$\log\rho$ (input tokens vs. seed)", labelpad=1)
    ax.set(ylim=(0, 0.22), yticks=[0, 0.05, 0.1, 0.15, 0.2])
    ax.tick_params(axis="x", length=0, pad=2)
    legend_above(ax, [Patch(facecolor=PLAIN, label="as recorded"), Patch(facecolor=SHOWN, label="recomputed at the seed's calls per turn")])


def edits_panel(ax, data):
    rng = S.rng("edits", "figure jitter")
    for i, status in enumerate(("rejected", "accepted")):
        g = data["status"][status]
        values = 100 * np.array(g["values"])
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(values)), values, s=9, color=EDIT[status.replace("accepted", "kept")], alpha=0.85,
                   linewidths=0, zorder=2)
        ax.hlines(100 * g["mean"], i - 0.32, i + 0.32, color=INK, lw=1.3, zorder=3)
        ax.text(i, 104, f"{100 * g['share_above']:.0f}% add tokens", ha="center", va="bottom")
    ax.axhline(0, color=MID, lw=0.5, ls=(0, (3, 2)), zorder=1)  # thin, grey, dashed: not read as a mean bar
    ax.set_xticks([0, 1], [f"rejected ({data['status']['rejected']['n']} edits)", f"kept ({data['status']['accepted']['n']} edits)"])
    ax.set(xlim=(-0.6, 1.6), ylim=(-60, 118), yticks=[-50, 0, 50, 100])
    ax.spines["left"].set_bounds(-60, 100)
    ax.set_ylabel("change in log input tokens\nvs. parent (×100)", labelpad=2.5, linespacing=0.95)
    ax.tick_params(axis="x", length=0, pad=2)


def implications():
    plt = pyplot(SIZE)
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 3.4))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.9, wspace=0.42, hspace=0.8)
    rules_panel(axes[0, 0], load("edits", "selrules"), Patch)
    explained_panel(axes[0, 1], load("edits", "designev"))
    batching_panel(axes[1, 0], load("calls", "batching"), Patch)
    edits_panel(axes[1, 1], load("edits", "designev"))
    for ax, title, dx in zip(axes.flat, TITLES, TITLE_DX):
        despine(ax)
        box = ax.get_position()
        fig.text(box.x0 - dx, box.y1 + TITLE_DY, title, fontsize=TEXT, fontweight="bold", ha="left", va="bottom")
    save_figure(fig, "implications4")


def skills():
    plt = pyplot(SIZE)
    import matplotlib
    from matplotlib.patches import Patch, Rectangle
    width, height, y0, h = SKILLS_W, 2.96, 0.6, 1.8
    rows = [(o, b, k * (len(BENCHES) + GAP) + j) for k, o in enumerate(S.OPTIMIZERS) for j, b in enumerate(BENCHES)]
    terms = {(c["method"], c["bench"]): c["A"] for c in load("main", "attribution")["cells"]}
    text = {(r["method"], r["bench"]): r for r in load("edits", "skilltext") if r["model"] == S.DEEPSEEK}
    fig = plt.figure(figsize=(width, height))
    ax_a = fig.add_axes([0.67 / width, y0 / height, 1.86 / width, h / height])
    ax_b = fig.add_axes([2.8 / width, y0 / height, 1.0 / width, h / height])
    for ax in (ax_a, ax_b):
        ax.set_ylim(rows[-1][2] + 0.6, -0.5)
    values = np.array([[terms[(S.ARM_NAME[o], b)][t] for t in TERMS] for o, b, _ in rows])
    vmax = np.abs(values).max()
    norm, cmap = matplotlib.colors.Normalize(-vmax, vmax), plt.get_cmap("RdBu_r")
    for (_, _, y), line in zip(rows, values):
        for j, v in enumerate(line):
            ax_a.add_patch(Rectangle((j, y - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            ax_a.text(j + 0.5, y, f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00"), ha="center", va="center",
                      color="white" if abs(v) > 0.62 * vmax else INK, fontweight="bold" if j == len(TERMS) - 1 else "normal")
    ax_a.set_xlim(0, len(TERMS))
    for j, head in enumerate(TERM_HEAD):
        ax_a.text(j + 0.5, -0.62, head, ha="center", va="bottom", linespacing=0.95)
    # the first two terms are the turn terms the text sums: a bracket over their heads (figure fractions, so any dpi)
    to_fig = ax_a.transData + fig.transFigure.inverted()
    (x_left, y_head), (x_right, _) = to_fig.transform([(0.08, -0.62), (1.92, -0.62)])
    low, high = y_head + 16 / 72 / height, y_head + 19 / 72 / height
    fig.add_artist(matplotlib.lines.Line2D([x_left, x_left, x_right, x_right], [low, high, high, low], transform=fig.transFigure,
                                           color=INK, lw=0.6))
    fig.text((x_left + x_right) / 2, high + 1 / 72 / height, "turn terms", ha="center", va="bottom")
    ax_a.set_xticks([])
    ax_a.set_yticks([y for _, _, y in rows], [BENCH_LABEL[b] for _, b, _ in rows])
    ax_a.tick_params(axis="y", length=0, pad=2)
    for side in ax_a.spines.values():
        side.set_visible(False)
    for k, o in enumerate(S.OPTIMIZERS):
        ax_a.annotate(S.ARM_NAME[o], xy=(0, rows[3 * k + 1][2]), xycoords=("axes fraction", "data"), xytext=(-41, 0), textcoords="offset points",
                      ha="center", va="center", rotation=90, fontsize=TEXT, fontweight="bold", color=ARM[o], annotation_clip=False)
    margin = matplotlib.transforms.blended_transform_factory(ax_b.transAxes, ax_b.transData)
    columns = {"verify": 1.2, "serialize": 1.56, "limit": 1.9, "words": 2.24}
    for o, b, y in rows:
        r = text[(o, b)]
        counts = dict(r["counts"], other=r["counts"].get("other", 0) + r["counts"].get("parallelize", 0))
        left = 0
        for cat in CATEGORIES:
            if counts.get(cat):
                ax_b.barh(y, counts[cat], left=left, height=0.72, color=RULE[cat], edgecolor="white", linewidth=0.4)
                left += counts[cat]
        for name, value, colour, weight in (("verify", counts.get("verify", 0), RULE["verify"], "bold"),
                                            ("serialize", counts.get("serialize", 0), RULE["serialize"], "bold"),
                                            ("limit", counts.get("limit_read", 0), DARK, "normal"), ("words", f"×{r['words_ratio']:.1f}", INK, "normal")):
            ax_b.text(columns[name], y, str(value), transform=margin, ha="center", va="center", color=colour, fontweight=weight)
    for name, head, colour in (("verify", "verify", RULE["verify"]), ("serialize", "serialize", RULE["serialize"]),
                               ("limit", "limit\nreading", DARK), ("words", "length\n× seed", INK)):
        ax_b.text(columns[name], -0.62, head, transform=margin, ha="center", va="bottom", color=colour, linespacing=0.95, clip_on=False)
    ax_b.set(xlim=(0, 105), xticks=[0, 35, 70, 105], yticks=[])
    ax_b.tick_params(axis="x", pad=1)
    ax_b.set_xlabel("rules added", labelpad=1)
    despine(ax_b)
    ax_b.spines["left"].set_visible(False)
    title_y = (y0 + h + 0.46) / height
    fig.text(0.67 / width, title_y, "(a) Where the extra input tokens go", fontsize=TEXT, fontweight="bold", va="bottom")
    fig.text(2.8 / width, title_y, "(b) What the optimizers write", fontsize=TEXT, fontweight="bold", va="bottom")
    # the legend decodes (b) only, so it sits under (b): two rows of three, filled column by column, read row-wise in stacking order
    order = [CATEGORIES[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=[Patch(facecolor=RULE[c], label=CATEGORY_LABEL[c]) for c in order], loc="lower left", bbox_to_anchor=(2.8 / width, 0.0),
               ncol=3, frameon=False, handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    save_figure(fig, "skills2")


if __name__ == "__main__":
    implications()
    skills()
