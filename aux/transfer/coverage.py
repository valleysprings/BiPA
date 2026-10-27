"""Coverage and agreement: the seed lets the agent solve tasks it never solves without a skill, while the learned skills gain
about as many tasks over the seed as they lose; where a learned skill gains, a task's episodes agree more (record mech_gains,
figure mech_gains2).

Pairs and contrasts: every model-benchmark pair with a learned skill (settings.learned_pairs), the seed against no skill and
each learned skill of the pair against the seed; a skill that returned the seed is left out. The test runs are the main
table's, three per side.

Coverage (panel a): over the tasks verified in all three runs on both sides, the tasks that only one side solves in at least
one run, summed over the pairs and, for the learned skills, over the optimizers; an exact two-sided sign test of the totals.

Agreement (panel b): over the tasks with three verified runs that each took a counted turn, on both sides, a task's episodes
agree more when the mean pairwise similarity of their lists of executed tool names (one minus the Levenshtein distance over
the longer list's length) is higher than under the reference. A task's success went up, stayed the same or went down with
the number of its runs that the side solved. The Mantel-Haenszel odds ratio of agreeing more, up against same, is stratified
by the reference's successes, with the Robins-Breslow-Greenland interval; its p against one, and the comparison of the
learned skills' odds ratio with the seed's, take the normal approximation on the log scale with that standard error."""
import math

import numpy as np

from src.analysis import settings as S
from src.utils import fig
from src.utils.logger import md_table, record

ORDER = ("terminal-bench-2-1", "tau3-bench", "automationbench")  # the main table's order
SEED, NOSKILL = "native-seed", "native-noseed"
MOVES = (("up", "success\nrose"), ("same", "success\nunchanged"), ("down", "success\nfell"))
WORDS = "zero one two three four five six seven eight nine ten".split()


def similarity(a, b):
    """One minus the Levenshtein distance of two lists over the longer length (1 for two empty lists)."""
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1.0 - prev[-1] / len(a)


def runs(model, bench, method):
    """{repetition: {task: success (None unless verified), counted turns, executed tool names}} of a cell."""
    out = {}
    for k, eps in S.cell_episodes(model, bench, method).items():
        out[k] = {}
        for t, e in eps.items():
            reqs = [r for r in {r["request_id"]: r for r in e.get("requests", [])}.values() if r.get("input_tokens") is not None]
            out[k][t] = dict(success=e["success"] if S.known(e) else None, turns=len(reqs),
                             tools=[c["tool"] for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False])
    return out


def spread(reps):
    """{task: (successes, divergence)} over the tasks with three verified runs that each took a turn; divergence is one minus
    the mean pairwise similarity of the runs' tool lists."""
    if len(reps) != 3:
        return {}
    out = {}
    for t in sorted(set.intersection(*map(set, reps.values()))):
        rs = [reps[k][t] for k in sorted(reps)]
        if all(r["success"] is not None and r["turns"] > 0 for r in rs):
            sims = [similarity(rs[i]["tools"], rs[j]["tools"]) for i, j in ((0, 1), (0, 2), (1, 2))]
            out[t] = (sum(r["success"] is True for r in rs), 1 - np.mean(sims))
    return out


def contrast(own, ref):
    """((tasks only own solves, tasks only the reference solves), [(reference successes, own successes, agrees more)])."""
    (wo, so), (wr, sr) = own, ref
    tasks = set(wo) & set(wr)
    cover = (sum(wo[t] >= 1 and wr[t] == 0 for t in tasks), sum(wo[t] == 0 and wr[t] >= 1 for t in tasks))
    return cover, [(sr[t][0], so[t][0], so[t][1] < sr[t][1]) for t in so if t in sr]


def compute():
    """[{model, bench, contrasts: {"seed" or optimizer: contrast}}] over the learned pairs."""
    learned = S.learned()
    pairs = []
    for model in S.MODELS:
        for bench in ORDER:
            skills = [o for m, b, o in learned if (m, b) == (model, bench)]
            if skills:
                sides = {a: runs(model, bench, a) for a in (NOSKILL, SEED, *skills)}
                sides = {a: (S.wins(r), spread(r)) for a, r in sides.items()}
                contrasts = {"seed": contrast(sides[SEED], sides[NOSKILL])} | {o: contrast(sides[o], sides[SEED]) for o in skills}
                pairs.append(dict(model=model, bench=bench, contrasts=contrasts))
    return pairs


def sign_p(a, b):
    """Exact two-sided sign test of a against b discordant tasks."""
    return min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))


def odds_ratio(triples):
    """(odds ratio, standard error of its log): Mantel-Haenszel, agreeing more when success went up against when it stayed the
    same, strata the reference's successes; Robins-Breslow-Greenland variance."""
    r_ = s_ = pr = psqr = qs = 0.0
    for level in range(4):
        up = [a for r, o, a in triples if r == level and o > r]
        same = [a for r, o, a in triples if r == level and o == r]
        if up and same:
            a, b, c, d = sum(up), len(up) - sum(up), sum(same), len(same) - sum(same)
            n = a + b + c + d
            r, s, p, q = a * d / n, b * c / n, (a + d) / n, (b + c) / n
            r_, s_, pr, psqr, qs = r_ + r, s_ + s, pr + p * r, psqr + p * s + q * r, qs + q * s
    return r_ / s_, math.sqrt(pr / (2 * r_ * r_) + psqr / (2 * r_ * s_) + qs / (2 * s_ * s_))


def moves(triples):
    """{up, same, down: (agree more, tasks, share %)}."""
    pick = {"up": lambda r, o: o > r, "same": lambda r, o: o == r, "down": lambda r, o: o < r}
    out = {}
    for k, keep in pick.items():
        sel = [a for r, o, a in triples if keep(r, o)]
        out[k] = (sum(sel), len(sel), 100 * sum(sel) / len(sel) if sel else float("nan"))
    return out


def label(pair):
    return f"{S.MODEL_SHORT[pair['model']]} {S.BENCH_SHORT[pair['bench']]}"


def draw(pairs, totals, p_seed, agree):
    """(a) tasks only one side solves (up: the newer side, down: the reference), per pair and summed; (b) share of tasks whose
    episodes agree more, by how success moved."""
    plt = fig.pyplot(7, title=7.8, label=7.5, tick=7.2, legend=7.2)
    num = 6.8
    figure, (axa, axb) = plt.subplots(1, 2, figsize=(fig.TEXT_W, 2.35), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.2))
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.255, top=0.905)
    xs = list(range(len(pairs))) + [len(pairs) + 0.5]
    groups = [(label(p), p["seed"], p["learned"]) for p in pairs] + [("total", *totals)]
    width, gap_total = 0.38, 0.42  # a wider gap in the total group, so the p label clears the next bar
    for i, (x, (_, s, l)) in enumerate(zip(xs, groups)):
        gap = gap_total if i == len(pairs) else 0.04
        for j, ((own, ref), color) in enumerate(((s, fig.ARM[SEED]), (l, fig.LEARNED))):
            xx = x + (j - 0.5) * (width + gap)
            axa.bar(xx, own, width=width, color=color, edgecolor="white", linewidth=0.4)
            axa.bar(xx, -ref, width=width, color=color, alpha=0.38, edgecolor="white", linewidth=0.4)
            axa.text(xx, own + 1.5, f"{own}", ha="center", va="bottom", fontsize=num)
            axa.text(xx, -ref - 1.5, f"{ref}", ha="center", va="top", fontsize=num)
    axa.text(xs[-1] - 0.5 * (width + gap_total), totals[0][0] + 11, f"$p$={p_seed:.3f}" if p_seed < 0.01 else f"$p$={p_seed:.2f}",
             ha="center", va="bottom", fontsize=num)
    axa.axhline(0, color=fig.DARK, lw=0.6)
    axa.axvline((xs[-2] + xs[-1]) / 2, color=fig.LIGHT, lw=0.6, ls=":")
    axa.set_xticks(xs)
    axa.set_xticklabels([fig.BENCH_TICK[p["bench"]] for p in pairs] + ["total"])
    axa.tick_params(axis="x", length=0, pad=2)
    spans = [(m, [x for x, p in zip(xs, pairs) if p["model"] == m]) for m in S.MODELS]
    spans = [(S.MODEL_SHORT[m], x[0], x[-1]) for m, x in spans if x] + [(f"all {WORDS[len(pairs)]} pairs", xs[-1], xs[-1])]
    for name, x0, x1 in spans:
        if x1 > x0:
            axa.plot([x0 - 0.35, x1 + 0.35], [-0.13, -0.13], transform=axa.get_xaxis_transform(), color=fig.MID, lw=0.5,
                     clip_on=False)
        axa.text((x0 + x1) / 2, -0.16, name, transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=7.2)
    axa.set_ylim(-88, 82)
    axa.set_yticks([-60, -30, 0, 30, 60])
    axa.set_yticklabels(["60", "30", "0", "30", "60"])
    axa.set_ylabel("tasks (learned skills:\nsummed over optimizers)", labelpad=1, linespacing=1.1)
    axa.text(-0.58, 81, "gained: covered only by the skill", ha="left", va="top", fontsize=num, color=fig.DARK)
    axa.text(-0.58, -87, "lost: covered only by the reference", ha="left", va="bottom", fontsize=num, color=fig.DARK)
    axa.set_xlim(-0.62, xs[-1] + 0.7)
    axa.set_title("(a) tasks covered by one side only", loc="left")
    sides = (("seed", fig.ARM[SEED]), ("learned", fig.LEARNED))
    for j, (name, color) in enumerate(sides):
        for g, (k, _) in enumerate(MOVES):
            agreeing, n, share = agree[name][k]
            x = g + (j - 0.5) * 0.36
            axb.bar(x, share, width=0.32, color=color, edgecolor="white", linewidth=0.4)
            axb.text(x, 3, f"{agreeing}/{n}", ha="center", va="bottom", fontsize=num, color="white", rotation=90)
    axb.axhline(50, color=fig.DARK, lw=0.6, ls=":")
    axb.set_xticks(range(len(MOVES)))
    axb.set_xticklabels([m[1] for m in MOVES])
    axb.tick_params(axis="x", length=0, pad=2)
    axb.set_xlim(-0.5, 2.5)
    axb.set_ylim(0, 100)
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_ylabel("share of tasks (%)", labelpad=1)
    axb.set_title("(b) episodes agree more than under the reference", loc="left")
    for ax in (axa, axb):
        fig.despine(ax)
        ax.tick_params(length=2, pad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fig.ARM[SEED], label="seed over no skill"),
               plt.Rectangle((0, 0), 1, 1, facecolor=fig.LEARNED, label="learned skills over seed")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.02), ncol=2, frameon=False, handlelength=0.9,
                  handletextpad=0.3, columnspacing=1.2)
    fig.save_figure(figure, "gains2")


def main():
    pairs = compute()
    for p in pairs:
        c = p["contrasts"]
        p["seed"] = c["seed"][0]
        p["learned"] = tuple(sum(c[o][0][i] for o in S.OPTIMIZERS if o in c) for i in (0, 1))
    totals = [tuple(sum(p[side][i] for p in pairs) for i in (0, 1)) for side in ("seed", "learned")]
    triples = {"seed": [x for p in pairs for x in p["contrasts"]["seed"][1]],
               "learned": [x for p in pairs for o in S.OPTIMIZERS if o in p["contrasts"] for x in p["contrasts"][o][1]]}
    agree = {k: moves(v) for k, v in triples.items()}
    odds = {k: odds_ratio(v) for k, v in triples.items()}
    draw(pairs, totals, sign_p(*totals[0]), agree)

    pair = lambda v: f"{v[0]} / {v[1]}"
    rows_a = [[label(p), pair(p["seed"]), *(pair(p["contrasts"][o][0]) if o in p["contrasts"] else "-" for o in S.OPTIMIZERS),
               pair(p["learned"])] for p in pairs]
    by_opt = [tuple(sum(p["contrasts"][o][0][i] for p in pairs if o in p["contrasts"]) for i in (0, 1)) for o in S.OPTIMIZERS]
    rows_a.append(["total", pair(totals[0]), *map(pair, by_opt), pair(totals[1])])
    share = lambda m: f"{m[0]}/{m[1]} = {m[2]:.1f}%" if m[1] else "0/0"
    interval = lambda o: f"{o[0]:.2f} [{o[0] * math.exp(-S.Z95 * o[1]):.2f}, {o[0] * math.exp(S.Z95 * o[1]):.2f}]"
    p_one = lambda o: math.erfc(abs(math.log(o[0])) / o[1] / math.sqrt(2))
    names = {"seed": "seed over no skill", "learned": "learned skills over seed"}
    rows_b = [[names[k], *(share(agree[k][m]) for m, _ in MOVES), interval(odds[k]), f"{p_one(odds[k]):.3f}"] for k in names]
    for p in pairs:
        c = p["contrasts"]
        per = {"seed": c["seed"][1], "learned": [x for o in S.OPTIMIZERS if o in c for x in c[o][1]]}
        rows_b += [[f"{label(p)}: {names[k]}", *(share(moves(per[k])[m]) for m, _ in MOVES), "", ""] for k in names]
    (ol, sl), (os_, ss) = odds["learned"], odds["seed"]
    z = (math.log(ol) - math.log(os_)) / math.hypot(sl, ss)
    full = lambda o: f"{o[0]:.3f} [{o[0] * math.exp(-S.Z95 * o[1]):.3f}, {o[0] * math.exp(S.Z95 * o[1]):.3f}]"
    two_three = {k: [a for r, o, a in v if (r, o) == (2, 3)] for k, v in triples.items()}
    numbers = "\n".join([
        f"Tasks that only one side solves in at least one of its three runs, gained / lost, over {len(pairs)} pairs.", "",
        md_table(["pair", "seed over no skill", *(f"{S.ARM_NAME[o]} over seed" for o in S.OPTIMIZERS),
                  "learned skills over seed"], rows_a), "",
        f"Exact two-sided sign test of the totals: seed over no skill {totals[0][0]} vs {totals[0][1]}, "
        f"p = {sign_p(*totals[0]):.4f}; learned skills over seed {totals[1][0]} vs {totals[1][1]}, p = {sign_p(*totals[1]):.3f}.", "",
        "Tasks whose episodes agree more than under the reference, by how success moved; odds ratio of agreeing more, up against "
        "same.", "",
        md_table(["contrast", "success up", "same", "down", "odds ratio, up vs same [95% interval]", "p against 1"], rows_b), "",
        f"Learned skills' odds ratio {full(odds['learned'])} against the seed's {full(odds['seed'])}: ratio {ol / os_:.2f}, "
        f"z = {z:.2f} on the log scale, two-sided p = {math.erfc(abs(z) / math.sqrt(2)):.3f}.",
        "Tasks going from two to three solved runs whose episodes agree more: " + ", ".join(
            f"{k} {sum(v)}/{len(v)} = {100 * np.mean(v):.1f}%"
            for k, v in (("learned skills", two_three["learned"]), ("seed", two_three["seed"]))) + "."])
    cells = [(p["model"], p["bench"], m) for p in pairs
             for m in (NOSKILL, SEED, *(o for o in S.OPTIMIZERS if o in p["contrasts"]))]
    record(__spec__.name, "gains", numbers, S.runs_text(cells))


if __name__ == "__main__":
    main()
