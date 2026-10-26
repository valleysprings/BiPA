"""GPT-6-Luna's SkillOpt and GEPA started from the three-word seed "Solve the task.": test success against the runs from
the hand-written seed, the dev gain against selection alone, the skills' length, cost, writing behaviour and rules, and
which tasks the learned skills cover (Appendix C.7).

Test success per run and avg@3 are Table 1's; a paired difference is the mean, over the tasks verified in all three runs of
both skills, of the per-task difference in success. The replay draws the dev tasks' success probabilities N_DRAWS times from
the fitted task model of aux.selection.taskmodel, fitted to the three-word seed's distinct dev and test executions in these
runs, and applies each optimizer's rule to K candidates identical to the seed; it reports P(gain >= the observed gain) with
the baseline fixed at its observed score or drawn like a candidate. One draw of probabilities per benchmark serves both
optimizers and both baselines. rho is Table 1's input-token ratio: on AutoBench against the three-word seed's three runs, on
tau3-bench against the seed's test run inside the same optimizer run. Writing behaviour counts the executed calls of every
recorded turn of all episodes (aux.calls.classify): the share of writes in a turn with another write, and the share of
episodes with a write whose last write is followed by a read in a later turn. Coverage is pass@3 (settings.wins): a skill
covers a task it solves in at least one of its three runs."""
import json

import numpy as np

from aux.calls.classify import classify, turn_calls
from aux.selection import taskmodel as TM
from aux.three_word_seed import collect as C
from src.analysis import settings as S
from src.analysis.common import accumulate, geo
from src.utils import logger

AUTOB, TAU3 = C.BENCHES
SEED3, HAND, NONE = "three-word seed", "hand-written seed", "no skill"
FROM3 = {o: f"{S.ARM_NAME[o]} from three words" for o in C.OPTIMIZERS}
FROMHAND = {o: f"{S.ARM_NAME[o]} from the hand-written seed" for o in S.OPTIMIZERS}


def rho(files, ref):
    means = [accumulate([{(t, 0): r for t, r in rows.items()} for rows in S.rows(fs, S.LUNA).values()]) for fs in (files, ref)]
    return geo(*means, turns=True)[1]["rho_trj"]


def replay(bench, sel):
    """{optimizer: (P with the baseline fixed, P with it drawn)} of reaching the observed dev gain by selection alone."""
    runs, seed = C.runs(bench).values(), C.seed(bench)
    dev = TM.executions([f for r in runs for f in C.dev_files(r, seed)])
    test = TM.executions([f for r in runs for fs in S.run_files(r, seed).values() for f in fs])
    rng = S.rng("three_word_seed", "replay", bench)
    p = TM.task_probabilities(dev, TM.priors(dev, test)[0]["fitted"], len(dev), S.N_DRAWS, rng)
    out = {}
    for o, s in sel.items():
        gain = max(s["returned"] - s["baseline"], 1)
        out[o] = tuple(float((TM.best_of(p, s["K"], base, rng) >= gain).mean()) for base in (s["baseline"], TM.outcomes(p, rng)))
    return out


def shared_episodes(bench):
    """(identical, common): the three-word seed's verified test episodes that its two optimizer runs share (one cached episode)."""
    sig = [{e["task_id"]: TM.signature(e) for fs in S.run_files(C.runs(bench)[o], C.seed(bench)).values() for f in fs
            for e in json.loads(f.read_text())["episodes"] if TM.verified(e)} for o in C.OPTIMIZERS]
    common = set(sig[0]) & set(sig[1])
    return sum(sig[0][t] == sig[1][t] for t in common), len(common)


def behaviour(files):
    """(episodes, writes sharing a turn with another write %, episodes whose last write is followed by a later read %)."""
    eps = writes = shared = with_write = read_after = 0
    for by_task in S.episodes(files).values():
        for e in by_task.values():
            turns = [[classify(tool, args) for tool, args in turn] for turn in turn_calls(e)]
            w = [turn.count("W") for turn in turns]
            eps, writes, shared = eps + bool(turns), writes + sum(w), shared + sum(x for x in w if x >= 2)
            if any(w):
                last = max(i for i, x in enumerate(w) if x)
                with_write, read_after = with_write + 1, read_after + any("R" in turn for turn in turns[last + 1:])
    return eps, f"{100 * shared / writes:.1f}", f"{100 * read_after / with_write:.1f}"


def coverage(w):
    """The coverage audit on AutoBench, w = {agent: settings.wins}: (rows per optimizer, prose)."""
    tasks = sorted(set.intersection(*map(set, w.values())))
    cover = lambda a, b: "{} / {}".format(sum(w[a][t] and not w[b][t] for t in tasks), sum(w[b][t] and not w[a][t] for t in tasks))  # noqa: E731
    known = [t for t in tasks if w[NONE][t] or w[HAND][t]]
    rest = [t for t in tasks if t not in known]
    points = lambda a, sub: 100 * sum(w[a][t] - w[SEED3][t] for t in sub) / 3 / len(tasks)  # noqa: E731
    middle = [t for t in tasks if w[SEED3][t] in (1, 2)]
    Q = lambda a: 100 * sum(w[a][t] for t in tasks) / 3 / len(tasks)  # noqa: E731
    rows = [["paired difference from the three-word seed (points)"] + [f"{points(FROM3[o], tasks):+.1f}" for o in C.OPTIMIZERS],
            ["paired difference from the hand-written seed (points)"]
            + [f"{100 * sum(w[FROM3[o]][t] - w[HAND][t] for t in tasks) / 3 / len(tasks):+.1f}" for o in C.OPTIMIZERS],
            ["covers / loses against the three-word seed"] + [cover(FROM3[o], SEED3) for o in C.OPTIMIZERS],
            ["of those it covers, solved by no skill or the hand-written seed"]
            + [sum(w[FROM3[o]][t] and not w[SEED3][t] for t in known) for o in C.OPTIMIZERS],
            [f"points from the {len(known)} tasks no skill or the hand-written seed solves / the other {len(rest)}"]
            + [f"{points(FROM3[o], known):.1f} / {points(FROM3[o], rest):.1f}" for o in C.OPTIMIZERS],
            [f"of the {len(middle)} tasks the three-word seed solves in one or two runs, solved in all three"]
            + [sum(w[FROM3[o]][t] == 3 for t in middle) for o in C.OPTIMIZERS],
            ["from the hand-written seed: covers / loses against it"] + [cover(FROMHAND[o], HAND) for o in C.OPTIMIZERS],
            ["tasks covered by its run from one seed only"]
            + [sum(bool(w[FROM3[o]][t]) != bool(w[FROMHAND[o]][t]) for t in tasks) for o in C.OPTIMIZERS]]
    learned = [*FROM3.values(), *FROMHAND.values()]
    new = [t for t in tasks if any(w[a][t] for a in learned) and not any(w[a][t] for a in (NONE, SEED3, HAND))]
    prose = (f"Three-word seed against no skill: {Q(SEED3):.1f} against {Q(NONE):.1f}, covers / loses {cover(SEED3, NONE)}. Tasks covered by some learned skill and by "
             f"none of the runs of no skill and the two seeds: {len(new)}; the most runs of the {len(learned)} learned skills that "
             f"solve one of them: {max(sum(w[a][t] for a in learned) for t in new)} of {3 * len(learned)}.")
    return rows, prose


def section(bench):
    agents = C.agents(bench)
    runs = C.runs(bench)
    test = {name: C.success(fs) for name, fs in agents.items() if name not in (NONE, FROMHAND["skilladam"])}
    out = [f"### {S.BENCH_SHORT[bench]}", "", logger.md_table(["agent", "test success per run (%)", "mean"],
                                                             [[n, ", ".join(f"{x:.1f}" for x in v), f"{np.mean(v):.1f}"] for n, v in test.items()]), ""]
    sel = {o: C.selection(bench, o) for o in C.OPTIMIZERS}
    pr = replay(bench, sel)
    out += [logger.md_table(["optimizer", "K", "dev baseline", "dev returned", "dev gain (points)", "P(gain >= observed), baseline fixed",
                             "P(gain >= observed), baseline drawn"],
                            [[S.ARM_NAME[o], s["K"], s["baseline"], s["returned"], f"{100 * (s['returned'] - s['baseline']) / s['n_dev']:+.1f}",
                              f"{pr[o][0]:.3f}", f"{pr[o][1]:.3f}"] for o, s in sel.items()]), ""]
    ref = {o: agents[SEED3] if bench == AUTOB else S.run_files(runs[o], C.seed(bench)) for o in C.OPTIMIZERS}
    rows = [[FROM3[o], C.words(C.returned(runs[o])), f"{rho(agents[FROM3[o]], ref[o]):.3f}",
             "{} of {}".format(sum(lv in ("S2", "S3") for _, lv in C.labelled(bench, o)), len(C.labelled(bench, o)))] for o in C.OPTIMIZERS]
    if bench == AUTOB:
        rows += [[SEED3, C.words(C.seed(bench)), "", ""], [HAND, C.words(S.skill(S.LUNA, bench, "native-seed")), f"{rho(agents[HAND], agents[SEED3]):.3f}", ""]]
    out += [logger.md_table(["skill", "words", "rho against the three-word seed", "sentences particular to the benchmark (S2, S3)"], rows), ""]
    if bench == TAU3:
        same = shared_episodes(bench)
        return out + ["Returned skill minus the three-word seed's test run in the same optimizer run (points): "
                      + ", ".join(f"{S.ARM_NAME[o]} {np.mean(test[FROM3[o]]) - np.mean(test[f'{SEED3} ({S.ARM_NAME[o]} run)']):+.1f}"
                                  for o in C.OPTIMIZERS) + f". The two test runs of the three-word seed share {same[0]} of {same[1]} cached episodes.", ""]
    rows, prose = coverage({name: S.wins(S.rows(fs, S.LUNA)) for name, fs in agents.items()})
    return out + [logger.md_table(["agent", "episodes", "writes sharing a turn with another write (%)", "last write followed by a later read (%)"],
                                  [[n, *behaviour(agents[n])] for n in (SEED3, HAND, *FROM3.values())]), "",
                  logger.md_table(["AutoBench", *(S.ARM_NAME[o] for o in C.OPTIMIZERS)], rows), "", prose, ""]


def main():
    lines = [x for bench in C.BENCHES for x in section(bench)]
    cells = [(S.LUNA, b, m) for b in C.BENCHES for m in S.ARMS]
    logger.record(__spec__.name, "fromscratch", "\n".join(lines), S.runs_text(cells, [r for b in C.BENCHES for r in C.runs(b).values()]))


if __name__ == "__main__":
    main()
