"""The three-word-seed figure (mech_scratch2): test success from each seed, and every candidate's dev score against its length.

(a, b) Test success of the three-word seed, the skills learned from it, the hand-written seed and the skills learned from
that; open circles are three test runs with a bar at their mean, filled diamonds single test runs (on tau3-bench the
three-word seed was tested once in each optimizer run). (c, d) Every candidate each run scored on the dev tasks against its
length in words; filled is a rewrite of the seed, open a rewrite of a rewrite, the ringed one the returned skill. Numbers
come from aux.three_word_seed.collect; a candidate's small horizontal jitter is drawn from settings.rng."""
import numpy as np

from aux.three_word_seed import collect as C
from src.analysis import settings as S
from src.utils import fig as F

COLOR = {"SkillOpt": F.ARM["skillopt"], "GEPA": F.ARM["gepa"], "seed": F.INK}
MARK = {"SkillOpt": F.MARKER["skillopt"], "GEPA": F.MARKER["gepa"]}
GROUPS = [("from three words", "Started from the three-word seed"), ("from the hand-written seed", "Started from the hand-written seed")]
ROWS = ["seed", "SkillOpt", "GEPA"]
HEAD_Y, ROW_Y = [0.0, 4.25], [[1.0, 2.0, 3.0], [5.25, 6.25, 7.25]]


def test_runs(bench):
    """{group: {row: [test success per run]}} and the hand-written seed's mean."""
    runs = {name: C.success(fs) for name, fs in C.agents(bench).items()}
    seed3 = [x for name, v in runs.items() if name.startswith("three-word seed") for x in v]
    out = {"from three words": {"seed": seed3, "SkillOpt": runs["SkillOpt from three words"], "GEPA": runs["GEPA from three words"]},
           "from the hand-written seed": {"seed": runs["hand-written seed"], "SkillOpt": runs["SkillOpt from the hand-written seed"],
                                          "GEPA": runs["GEPA from the hand-written seed"]}}
    return out, float(np.mean(runs["hand-written seed"]))


def test_panel(ax, runs, hand, title, xl):
    xq = xl[1]
    ax.axvline(hand, color=F.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(xq, HEAD_Y[0], "%", fontsize=7, color=F.MUTED, ha="right", va="center")
    for (group, _), ys in zip(GROUPS, ROW_Y):
        for row, y in zip(ROWS, ys):
            vals, col = runs[group][row], COLOR[row]
            if len(vals) == 3:
                ax.plot(vals, [y] * 3, "o", ms=3.4, mfc="white", mec=col, mew=0.8, zorder=3, alpha=0.95)
                ax.plot([np.mean(vals)] * 2, [y - 0.3, y + 0.3], color=col, lw=2.0, zorder=4, solid_capstyle="butt")
            else:
                ax.plot(vals, [y] * len(vals), "D", ms=3.4, mfc=col, mec=col, mew=0.8, zorder=4)
            txt = ", ".join(f"{v:.1f}" for v in vals) if len(vals) == 2 else f"{np.mean(vals):.1f}"
            ax.text(xq, y, txt, va="center", ha="right", fontsize=7, color=col)
    ax.set_xlim(*xl)
    ax.set_ylim(8.35, -0.6)
    ax.set_yticks(ROW_Y[0] + ROW_Y[1])
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    ax.axhline(HEAD_Y[1] - 0.5, color=F.GRID, lw=0.7)
    ax.set_xlabel("test success (%)")
    ax.set_xticks([t for t in ax.get_xticks() if xl[0] <= t <= xq - 0.17 * (xl[1] - xl[0])])
    ax.grid(axis="x", color=F.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, loc="left", color=F.INK, pad=6)
    ax.text(hand, 8.3, " hand-written seed", fontsize=6.8, color=F.MUTED, va="bottom", ha="left")


def candidate_panel(fig, ax, bench, title, rng):
    from matplotlib.transforms import offset_copy
    seed_dev = C.dev_solved(C.runs(bench)["skillopt"], C.seed(bench))[0]
    gepa_seed, gepa = C.gepa_candidates(bench)
    if gepa_seed != seed_dev:
        raise ValueError(f"{bench}: the two runs scored the seed {seed_dev} and {gepa_seed} on dev")
    seed_words = C.words(C.seed(bench))
    ax.axhline(seed_dev, color=F.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.plot([seed_words], [seed_dev], "s", ms=4.5, color=COLOR["seed"], zorder=5)
    ax.annotate("Solve the task.", (seed_words, seed_dev), xytext=(4, 5), textcoords="offset points", fontsize=6.8, color=F.INK, style="italic")
    for o, cands in (("SkillOpt", C.skillopt_candidates(bench)), ("GEPA", gepa)):
        ret = next(c for c in cands if c["returned"])
        for c in cands:
            # a candidate with the returned skill's dev score at almost its length is drawn 7 pt to the right, y kept true
            tie = not c["returned"] and c["dev"] == ret["dev"] and abs(np.log(c["words"] / ret["words"])) < 0.08
            jit = 0.0 if c["returned"] or tie else 0.13 * (rng.random() - 0.5)
            dx = 7.0 if tie else 0.0
            if c["tag"] == "last round":
                if tie:
                    ax.annotate(f"last round, {c['words']} words, same dev score", (c["words"], c["dev"]), xytext=(dx / 2 + 6, 6),
                                textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=COLOR[o])
                else:
                    ax.annotate(f"last round, {c['words']} words", (c["words"], c["dev"] + jit), xytext=(-6, 0),
                                textcoords="offset points", ha="right", va="center", fontsize=6.8, color=COLOR[o])
            if c["returned"]:
                ax.plot(c["words"], c["dev"], MARK[o], ms=6.5, mfc=COLOR[o], mec=F.INK, mew=0.9, zorder=8)
                xt, ha, va = ((-6, 0), "right", "center") if o == "SkillOpt" else ((6, 0), "left", "center") if bench == C.BENCHES[0] \
                    else ((-4, -5), "right", "top")
                ax.annotate(f"{c['words']} words", (c["words"], c["dev"]), xytext=xt, textcoords="offset points", ha=ha, va=va,
                            fontsize=6.8, color=COLOR[o])
            else:
                ax.plot(c["words"], c["dev"] + jit, MARK[o], ms=4.3, mfc=COLOR[o] if c["depth"] == 1 else "white", mec=COLOR[o], mew=0.9,
                        zorder=7.5 if o == "SkillOpt" else 7, alpha=0.95, transform=offset_copy(ax.transData, fig=fig, x=dx, y=0, units="points"))
    ax.set_xscale("log")
    ax.set_xlim(8, 2600)
    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.set_xticklabels(["10", "30", "100", "300", "1k"])
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("skill length (words)")
    ax.grid(color=F.GRID, lw=0.5)
    ax.set_axisbelow(True)
    F.despine(ax)
    ax.set_title(title, loc="left", color=F.INK, pad=4)
    ax.text(2500, seed_dev + 0.12, "seed", fontsize=6.8, color=F.MUTED, va="bottom", ha="right")


def main():
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory
    plt = F.pyplot(7.5, tick=7)
    rng = S.rng("three_word_seed", "jitter")
    fig = plt.figure(figsize=(6.3, 4.75))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.905, bottom=0.088, hspace=0.50, wspace=0.10, height_ratios=[1.12, 1])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = (fig.add_subplot(g) for g in gs[1, :].subgridspec(1, 2, wspace=0.28))
    autob, tau3 = C.BENCHES
    for ax, bench, title, xl in ((axA, autob, "(a) AutoBench: from three words, both end where\nthe runs from the hand-written seed end", (15, 55)),
                                 (axB, tau3, "(b) τ³-bench: the three-word seed is already close to\nthe hand-written seed, and optimizing adds nothing",
                                  (50, 84))):
        test_panel(ax, *test_runs(bench), title, xl)
    axA.set_yticklabels(ROWS + ROWS)
    for label, row in zip(axA.get_yticklabels(), ROWS + ROWS):
        label.set_color(COLOR[row])
    axB.set_yticklabels([])
    tr = blended_transform_factory(fig.transFigure, axA.transData)
    for (_, head), y in zip(GROUPS, HEAD_Y):
        axA.text(0.012, y, head, transform=tr, fontsize=7.5, color=F.INK, ha="left", va="center", fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=6, clip_on=False)
    for ax, bench, title in ((axC, autob, "(c) AutoBench dev: the returned skill is one\nrewrite of the seed; rewriting it again scores lower"),
                             (axD, tau3, "(d) τ³-bench dev: the returned skill is one\nrewrite of the seed; rewriting it again scores no higher")):
        candidate_panel(fig, ax, bench, title, rng)
    axC.set_ylabel("dev tasks solved (of 30)")
    axC.set_ylim(11.5, 20.5)
    axC.set_yticks([12, 14, 16, 18, 20])
    axD.set_ylim(13.5, 23.5)
    axD.set_yticks([14, 16, 18, 20, 22])
    handles = [Line2D([], [], ls="", marker=MARK[o], ms=4.3, mfc=COLOR[o], mec=COLOR[o], label=o) for o in ("SkillOpt", "GEPA")]
    handles += [Line2D([], [], ls="", marker="o", ms=4.3, mfc=F.MID, mec=F.MID, label="rewrite of the seed"),
                Line2D([], [], ls="", marker="o", ms=4.3, mfc="white", mec=F.MID, label="rewrite of a rewrite"),
                Line2D([], [], ls="", marker="o", ms=6.5, mfc=F.MID, mec=F.INK, mew=0.9, label="returned skill")]
    axC.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    F.save_figure(fig, "scratch2", width=F.TEXT_W)


if __name__ == "__main__":
    main()
