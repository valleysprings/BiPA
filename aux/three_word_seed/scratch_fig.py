"""The three-word-seed figure (mech_scratch2): GPT-6-Luna runs of SkillOpt and GEPA started from "Solve the task.".

(a, b) test success of the three-word seed, the skills learned from it, the hand-written seed and the skills learned from that;
       open circles = three test runs with a bar at the mean, filled diamonds = single test runs (tau3: the three-word seed
       was tested once in each optimizer run).
(c, d) every candidate each run scored on the 30 dev tasks against its length in words (front matter included); filled = a
       rewrite of the seed, open = a rewrite of a rewrite.
Inputs: numbers.json of python -m aux.three_word_seed.make_numbers (test success, seed dev score), the tau3 repetitions of
the hand-written-seed runs (aux.headline_numbers.transfer.cell_data), and the runs' candidate logs (SkillOpt history.json and
steps/, GEPA candidates.json, run_log.txt, run_log.json). The drawn numbers go to OUT/data/mech/mech_scratch2_numbers.json.
Run:  python -m aux.three_word_seed.scratch_fig
"""
import json
import re

import numpy as np

from aux.figures.layoutcheck import check
from aux.headline_numbers.transfer import cell_data
from src import layout
from src.utils.fig import save_figure
from src.utils.logger.paperout import DATA

EXE = layout.EXECUTION / "gpt-6-luna"
NUMBERS = layout.OUT / "three_word_seed/numbers.json"
INK, MUTED, GRID = "#222222", "#5A5A5A", "#E2E5EA"
C = {"SkillOpt": "#0072B2", "GEPA": "#009E73", "seed": "#222222"}
MK = {"SkillOpt": "o", "GEPA": "^"}
GROUPS = [("three_word", "Started from the three-word seed"), ("hand_written", "Started from the hand-written seed")]
ROWS = ["seed", "SkillOpt", "GEPA"]
HEAD_Y = [0.0, 4.25]
ROW_Y = [[1.0, 2.0, 3.0], [5.25, 6.25, 7.25]]
# tau3 test repetitions of the runs from the hand-written seed, as in the three-word-seed record
HAND_TAU3 = {"skillopt": [62.7, 64.0, 62.7], "gepa": [62.7, 68.0, 65.3]}


def words(text):
    return len(text.split())


def skillopt_candidates(bench):
    """(seed words, every SkillOpt candidate with dev score, words, depth, accepted, returned) of the run from three words."""
    run = EXE / "skillopt" / bench / "fromscratch"
    r = run / "skillopt"
    out, depth = [], 0
    for x in json.loads((r / "history.json").read_text()):
        text = (r / "steps" / f"step_{x['step']:04d}" / "candidate_skill.md").read_text()
        out.append(dict(dev=round(30 * x["selection_hard"]), words=words(text), depth=depth + 1, accepted=x["action"].startswith("accept"),
                        returned=(x["step"] == x["best_step"] and x["action"] == "accept_new_best"), tag=f"step {x['step']}"))
        if x["action"].startswith("accept"):
            depth += 1
    best = [c for c in out if c["returned"]]
    assert len(best) == 1 and best[0]["words"] == words((run / "final/SKILL.md").read_text()), bench
    # the skill after the last round, scored on dev as final_selection_hard
    extra = json.loads((run / "summary.json").read_text())["extra"]
    last = (r / "skills/skill_v0004.md").read_text()
    out.append(dict(dev=round(30 * extra["final_selection_hard"]), words=words(last), depth=depth + 1, accepted=False, returned=False,
                    tag="last iterate"))
    return words((r / "seed.md").read_text()), out


def gepa_candidates(bench):
    """(seed words, seed dev score, every GEPA candidate with dev score, words, depth, returned, parent)."""
    run = EXE / "gepa" / bench / "fromscratch"
    g = run / "gepa"
    cand = json.loads((g / "candidates.json").read_text())
    score, last = {}, None
    for line in (g / "run_log.txt").read_text().splitlines():
        if m := re.match(r"Iteration \d+: Base program full valset score: ([0-9.]+)", line):
            score[0] = float(m.group(1))
        if m := re.match(r"Iteration \d+: Valset score for new program: ([0-9.]+)", line):
            last = float(m.group(1))
        if m := re.match(r"Iteration \d+: New program candidate index: (\d+)", line):
            i = int(m.group(1))
            assert i not in score
            score[i], last = last, None
    parent = {e["new_program_idx"]: e["tasks"][0]["parent_idx"] for e in json.loads((g / "run_log.json").read_text())
              if e.get("new_program_idx") is not None}
    depth = {0: 0}
    for i in range(1, len(cand)):
        depth[i] = depth[parent[i]] + 1
    best = max(range(1, len(cand)), key=lambda i: (score[i], -i))
    assert cand[best]["current_candidate"].strip() == (run / "final/SKILL.md").read_text().strip(), bench
    out = [dict(dev=round(30 * score[i]), words=words(cand[i]["current_candidate"]), depth=depth[i], returned=(i == best), tag=f"cand {i}",
                parent=parent[i]) for i in range(1, len(cand))]
    return words(cand[0]["current_candidate"]), round(30 * score[0]), out


def inputs():
    if not NUMBERS.is_file():
        raise SystemExit(f"{NUMBERS} missing: run the aux.three_word_seed steps up to make_numbers")
    num = json.loads(NUMBERS.read_text())
    ab, tau = num["a_test"]["automationbench"]["rep_pct"], num["a_test"]["tau3-bench"]
    hand = {arm: [cell_data(arm, "native-20261012", "tau3-bench")["reps"][k]["Q"] for k in range(3)] for arm in HAND_TAU3}
    for arm, want in HAND_TAU3.items():
        assert all(abs(a - b) < 0.05 for a, b in zip(hand[arm], want)), (arm, hand[arm])
    test = {"automationbench": {"three_word": {"seed": ab["seed3"], "SkillOpt": ab["skillopt_fs"], "GEPA": ab["gepa_fs"]},
                                "hand_written": {"seed": ab["hand_seed"], "SkillOpt": ab["skillopt_native"], "GEPA": ab["gepa_native"]},
                                "hand_seed_mean": sum(ab["hand_seed"]) / 3},
            "tau3-bench": {"three_word": {"seed": [tau["skillopt"]["seed_pct"], tau["gepa"]["seed_pct"]],
                                          "SkillOpt": [tau["skillopt"]["returned_pct"]], "GEPA": [tau["gepa"]["returned_pct"]]},
                           "hand_written": {"seed": tau["hand_seed_rep_pct"], "SkillOpt": hand["skillopt"], "GEPA": hand["gepa"]},
                           "hand_seed_mean": tau["hand_seed_avg3_pct"]}}
    cands = {}
    for bench in ("automationbench", "tau3-bench"):
        sw, so = skillopt_candidates(bench)
        gw, gseed, go = gepa_candidates(bench)
        seed_dev = num["d_dev"]["collect"][f"skillopt|{bench}"]["seed_dev_solved"][0]
        assert sw == gw and seed_dev == gseed, (bench, sw, gw, seed_dev, gseed)
        cands[bench] = dict(seed_words=sw, seed_dev=seed_dev, SkillOpt=so, GEPA=go)
    return test, cands


def test_panel(fig, ax, T, title, xl, nums):
    span, xq, hand = xl[1] - xl[0], xl[1], T["hand_seed_mean"]
    ax.axvline(hand, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(xq, HEAD_Y[0], "%", fontsize=7, color=MUTED, ha="right", va="center")
    for (gkey, _), ys in zip(GROUPS, ROW_Y):
        nums[gkey] = {}
        for row, y in zip(ROWS, ys):
            vals, col = list(T[gkey][row]), C[row]
            m = float(np.mean(vals))
            if len(vals) == 3:
                ax.plot(vals, [y] * 3, "o", ms=3.4, mfc="white", mec=col, mew=0.8, zorder=3, alpha=0.95)
                ax.plot([m, m], [y - 0.3, y + 0.3], color=col, lw=2.0, zorder=4, solid_capstyle="butt")
                txt = f"{m:.1f}"
            else:
                ax.plot(vals, [y] * len(vals), "D", ms=3.4, mfc=col, mec=col, mew=0.8, zorder=4)
                txt = f"{m:.1f}" if len(vals) == 1 else f"{vals[0]:.1f}, {vals[1]:.1f}"
            ax.text(xq, y, txt, va="center", ha="right", fontsize=7, color=col)
            nums[gkey][row] = dict(runs=vals, shown=txt)
    ax.set_xlim(*xl)
    ax.set_ylim(8.35, -0.6)
    ax.set_yticks(ROW_Y[0] + ROW_Y[1])
    ax.tick_params(axis="y", length=0)
    ax.axhline(HEAD_Y[1] - 0.5, color=GRID, lw=0.7)
    ax.set_xlabel("test success (%)")
    ax.set_xticks([t for t in ax.get_xticks() if xl[0] <= t <= xq - 0.17 * span])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=6)
    ax.text(hand, 8.3, " hand-written seed", fontsize=6.8, color=MUTED, va="bottom", ha="left")
    nums["hand_seed_line"] = hand


def candidate_panel(fig, ax, bench, d, title, offset_copy):
    ax.axhline(d["seed_dev"], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.plot([d["seed_words"]], [d["seed_dev"]], "s", ms=4.5, color=C["seed"], zorder=5)
    ax.annotate("Solve the task.", (d["seed_words"], d["seed_dev"]), xytext=(4, 5), textcoords="offset points", fontsize=6.8, color=INK,
                style="italic")
    rng = np.random.default_rng(3)
    for o in ("SkillOpt", "GEPA"):
        ret = [c for c in d[o] if c["returned"]][0]
        for c in d[o]:
            jit = 0.13 * (rng.random() - 0.5) if not c["returned"] else 0.0
            # a candidate with the returned skill's dev score at almost its length is drawn 7 pt to the right, y kept true
            tie = (not c["returned"]) and c["dev"] == ret["dev"] and abs(np.log(c["words"] / ret["words"])) < 0.08
            dx = 7.0 if tie else 0.0
            if tie:
                jit = 0.0
            tr = offset_copy(ax.transData, fig=fig, x=dx, y=0, units="points")
            if c.get("tag") == "last iterate":
                if tie:
                    ax.annotate(f"last round, {c['words']} words, same dev score", (c["words"], c["dev"]), xytext=(dx / 2 + 6, 6),
                                textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=C[o])
                else:
                    ax.annotate(f"last round, {c['words']} words", (c["words"], c["dev"] + jit), xytext=(-6, 0),
                                textcoords="offset points", ha="right", va="center", fontsize=6.8, color=C[o])
            if c["returned"]:
                ax.plot(c["words"], c["dev"] + jit, MK[o], ms=6.5, mfc=C[o], mec=INK, mew=0.9, zorder=8)
                if o == "SkillOpt":
                    xt, ha, va = (-6, 0), "right", "center"
                elif bench == "automationbench":
                    xt, ha, va = (6, 0), "left", "center"
                else:
                    xt, ha, va = (-4, -5), "right", "top"
                ax.annotate(f"{c['words']} words", (c["words"], c["dev"]), xytext=xt, textcoords="offset points", ha=ha, va=va, fontsize=6.8,
                            color=C[o])
            else:
                ax.plot(c["words"], c["dev"] + jit, MK[o], ms=4.3, mfc=(C[o] if c["depth"] == 1 else "white"), mec=C[o], mew=0.9,
                        zorder=(7.5 if o == "SkillOpt" else 7), alpha=0.95, transform=tr)
    ax.set_xscale("log")
    ax.set_xlim(8, 2600)
    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.set_xticklabels(["10", "30", "100", "300", "1k"])
    ax.minorticks_off()
    ax.set_xlabel("skill length (words)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=4)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory, offset_copy
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5})
    test, cands = inputs()
    nums = {"test_pct": {}, "candidates": cands}
    fig = plt.figure(figsize=(6.3, 4.75))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.905, bottom=0.088, hspace=0.50, wspace=0.10, height_ratios=[1.12, 1])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gs2 = gs[1, :].subgridspec(1, 2, wspace=0.28)
    axC, axD = fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])
    for ax, b, title, xl in ((axA, "automationbench", "(a) AutoBench: from three words, both end where\nthe runs from the hand-written seed end", (15, 55)),
                             (axB, "tau3-bench", "(b) τ³-bench: the three-word seed is already close to\nthe hand-written seed, and optimizing adds nothing",
                              (50, 84))):
        nums["test_pct"][b] = {}
        test_panel(fig, ax, test[b], title, xl, nums["test_pct"][b])
    axA.set_yticklabels(ROWS + ROWS)
    for lab, row in zip(axA.get_yticklabels(), ROWS + ROWS):
        lab.set_color(C[row])
    axB.set_yticklabels([])
    trA = blended_transform_factory(fig.transFigure, axA.transData)
    for (_, head), y in zip(GROUPS, HEAD_Y):
        axA.text(0.012, y, head, transform=trA, fontsize=7.5, color=INK, ha="left", va="center", fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=6, clip_on=False)
    for ax, b, title in ((axC, "automationbench", "(c) AutoBench dev: the returned skill is one\nrewrite of the seed; rewriting it again scores lower"),
                         (axD, "tau3-bench", "(d) τ³-bench dev: the returned skill is one\nrewrite of the seed; rewriting it again scores no higher")):
        candidate_panel(fig, ax, b, cands[b], title, offset_copy)
    axC.set_ylabel("dev tasks solved (of 30)")
    axC.set_ylim(11.5, 20.5)
    axC.set_yticks([12, 14, 16, 18, 20])
    axD.set_ylim(13.5, 23.5)
    axD.set_yticks([14, 16, 18, 20, 22])
    axC.text(2500, 13.12, "seed", fontsize=6.8, color=MUTED, va="bottom", ha="right")
    axD.text(2500, 19.12, "seed", fontsize=6.8, color=MUTED, va="bottom", ha="right")
    handles = [Line2D([], [], ls="", marker="o", ms=4.3, mfc=C["SkillOpt"], mec=C["SkillOpt"], label="SkillOpt"),
               Line2D([], [], ls="", marker="^", ms=4.3, mfc=C["GEPA"], mec=C["GEPA"], label="GEPA"),
               Line2D([], [], ls="", marker="o", ms=4.3, mfc="#888888", mec="#888888", label="rewrite of the seed"),
               Line2D([], [], ls="", marker="o", ms=4.3, mfc="white", mec="#888888", label="rewrite of a rewrite"),
               Line2D([], [], ls="", marker="o", ms=6.5, mfc="#888888", mec=INK, mew=0.9, label="returned skill")]
    axC.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3, ncol=1)
    assert not check(fig), "layout problems above"
    save_figure(fig, "scratch2")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "mech_scratch2_numbers.json").write_text(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
