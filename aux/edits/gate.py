"""What the optimizers' gate keeps: kept against rejected edits, what sets an edit's cost, how often four acceptance rules
keep an unchanged candidate, and whether success or cost separates a candidate from its parent.

Edits are the dev records of the edge table (aux.edits.lineage), one per DeepSeek edit scored on dev. An edit's cost is the
change in mean log input tokens of the child against its parent on the dev tasks both ran (dlogX), split into more actions,
turn-splitting, content per turn and skill text; dQ is the change in dev success in points. Group means are unweighted over
edits or weighted by the edit's shared tasks (n_pair); an interval resamples edits, accepted and rejected each within
itself. Holm: an edit carries a keyword tag of the text it adds when it adds at least MIN_CHARS characters of it; with-minus-
without differences on the more-actions and turn-splitting terms, two-sided p = 2 x the smaller resampled tail mass at zero
with the +1 correction, Holm's step-down over every contrast at ALPHA.

What sets an edit's cost: R^2 of a simple regression and Spearman's correlation of dlogX on the change in log skill
characters, log turns or log tool calls over the edits; for the learned skills, log rho of input tokens (the attribution
total) on the log ratio of the skill's words to the seed's and on log rho of turns (mech_compaction). The cost-aware gate
accepts an edit iff dQ >= 0 and dlogX <= 0; the Mann-Whitney test compares kept and rejected dlogX.

Acceptance rules, on the task model of the seed's dev tasks (aux.selection.taskmodel, fitted prior, DeepSeek): one
candidate against its parent, unchanged or truly solving g more dev tasks, on fresh single runs drawn from one draw of the
task probabilities: (a) better on one run, (b) a gain of at least the critical value, the smallest net difference two
identical single runs reach in at most 5% of draws, (c) better, then better again on a second run, (d) a higher total over
three runs. A logged accepted edit's gain in tasks is dQ x n_pair / 100 against the critical value of its comparison size.

Tie-break: per dev-scored edit of each model, on the dev tasks both versions ran (the largest shared panel), the success
difference D (tasks) with paired SE sqrt(n) sd and the mean per-task log token ratio L with paired SE sd / sqrt(n); a change
is beyond noise when |z| >= Z95, and for success also when |D| reaches the critical value of the nearest comparison size
(the DeepSeek values serve both models). The seed-vs-seed noise floor pairs the seed's distinct full-panel dev draws in the
seed dev pool. Pool rule (runs that scored every version on their full dev panel): among the versions within one critical
value of the best, P returns the one with the lowest mean log dev input tokens; Z returns the anchor (the seed if eligible,
else the optimizer's own pick) unless an eligible version is cheaper with token z <= -Z95. Sequential gate (SkillOpt's
chain, SkillAdam's iterations on dev subsets): against the current skill, accept iff D reaches the critical value, or D is
above minus it and the candidate is cheaper (P: L < 0; Z: token z <= -Z95); a comparison is skipped when the current skill
was not scored on that subset. A replay of selection on logged proposals, not a rerun; test values are Table 1's."""
import functools, json, math, sys
from pathlib import Path

import numpy as np
from scipy import stats

from aux.edits import lineage as L
from aux.selection import taskmodel as TM
from src.analysis import settings as S
from src.analysis import table1 as T
from src.utils.dataloader import BENCHES, SEED_TEXT
from src.utils.logger import md_table, record

TERMS = ("dlogX", "actions", "turnsplit", "content", "text", "resid", "dQ")
SCOPES = ("gepa", "skillopt", "skilladam")
MIN_CHARS = 150
ALPHA = 0.05
HOLM_TERMS = (("actions", "more actions"), ("turnsplit", "turn-splitting"))
RULES = (("better", "(a) better on one run"), ("critical", "(b) gain >= one-sided 0.05 critical value"),
         ("confirm", "(c) better, then better again"), ("mean3", "(d) mean of 3 runs"))
PLOT_GAIN = 2
SEED = L.version(SEED_TEXT)
ROW = {m: n for n, m in T.ROWS}
SHORT = {"tau3-bench": "tau3", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
pct = lambda x: "n/a" if x is None else f"{100 * x:+.1f}"  # noqa: E731
span = lambda lo, hi: "n/a" if lo is None else f"[{100 * lo:.1f}, {100 * hi:.1f}]"  # noqa: E731
frac = lambda p: f"{p[0]}/{p[1]} ({100 * p[0] / p[1]:.0f}%)"  # noqa: E731


def save(name, obj):
    (S.cache("edits") / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def edits_line(edges, excluded):
    return (f"{len(edges)} DeepSeek edits scored on dev ({len(excluded)} SkillAdam terminal-bench-2-1 dev edits left out: their parents "
            f"are ambiguous)")


# ---- kept against rejected edits

def edges(rows, excluded):
    def mean(group, idx, term, weighted):
        v, w = np.array([e[term] for e in group], float)[idx], np.array([e["n_pair"] for e in group], float)[idx]
        return (v * w).sum(-1) / w.sum(-1) if weighted else v.mean(-1)

    groups, diffs = {}, {}
    for scope in ("all", *SCOPES):
        members = [e for e in rows if scope in ("all", e["opt"])]
        split = {s: [e for e in members if e["accepted"] == (s == "accepted")] for s in ("accepted", "rejected")}
        draws = {s: S.boot_index(len(g), "edits", "edges", scope, s) if len(g) > 1 else None for s, g in split.items()}
        for s, g in split.items():
            groups[(scope, s)] = {"n": len(g), "pairs": sum(e["n_pair"] for e in g),
                                  **{t + w * "_w": float(mean(g, np.arange(len(g)), t, w)) for t in TERMS for w in (0, 1) if g},
                                  "ci": S.interval(mean(g, draws[s], "dlogX", 0)) if draws[s] is not None else (None, None)}
        a, r = split["accepted"], split["rejected"]
        if a and r:
            diffs[scope] = {(t, w): (float(mean(a, np.arange(len(a)), t, w) - mean(r, np.arange(len(r)), t, w)),
                                     *S.interval(mean(a, draws["accepted"], t, w) - mean(r, draws["rejected"], t, w)))
                            for t in TERMS for w in (0, 1)}
    head = ["scope", "status", "edges", "pairs", "dlogX", "[95%]", "actions", "turn-split", "content", "text", "resid", "dQ (pts)"]
    terms = lambda g, w: [pct(g["dlogX" + w]), span(*g["ci"]) if not w else "", *(pct(g[t + w]) for t in TERMS[1:6]), f"{g['dQ' + w]:+.1f}"]  # noqa: E731
    unweighted = [[sc, s, g["n"], g["pairs"], *terms(g, "")] if g["n"] else [sc, s, 0, 0] + ["-"] * 8 for (sc, s), g in groups.items()]
    weighted = [[sc, s, g["n"], g["pairs"], *terms(g, "_w")] for (sc, s), g in groups.items() if g["n"]]
    cell = lambda p, lo, hi, dq: f"{p:+.1f} [{lo:.1f}, {hi:.1f}]" if dq else f"{pct(p)} {span(lo, hi)}"  # noqa: E731
    difference = [[sc, label] + [cell(*d[(t, w)], t == "dQ") for t in ("dlogX", "actions", "turnsplit", "content", "text", "dQ")]
                  for sc, d in diffs.items() for label, w in (("edges", 0), ("pairs", 1))]
    none = [f"{sc.replace('gepa', 'GEPA')} has no rejected dev edge in the table (all {groups[(sc, 'accepted')]['n']} are accepted), so its "
            "rows have no comparison." for sc in SCOPES if not groups[(sc, "rejected")]["n"]]
    numbers = "\n\n".join([
        f"The {edits_line(rows, excluded)}. Mean change per edge against its parent, dev panel, log points x100 (dQ in success points); "
        "[95%] is the bootstrap 95% interval of the mean over edges. Unweighted over edges:", md_table(head, unweighted), "Task-pair-weighted (weights n_pair):", md_table(head, weighted),
        f"Accepted minus rejected (bootstrap 95% over edges, {S.N_BOOT:,} resamples, accepted and rejected resampled separately):",
        md_table(["scope", "weighting", "d dlogX", "d actions", "d turn-split", "d content", "d text", "d dQ (pts)"], difference), *none])
    record(__spec__.name, "edges", numbers, S.runs_text(cells()),
           what="Whether rejected candidates also read more than their parents (the editor adds the input-token tax and success-only "
                "selection lets it through) or whether accepted ones read more than rejected ones (selection ratchets cost). Each edge "
                "is a scored candidate against its parent on the same dev tasks.")


def holm(rows, excluded):
    tests = []
    for tag in (*L.TAGS, "other"):
        has, without = [e for e in rows if e["mass_add"][tag] >= MIN_CHARS], [e for e in rows if e["mass_add"][tag] < MIN_CHARS]
        if not 1 < len(has) < len(rows) - 1:
            continue
        ih, iw = S.boot_index(len(has), "edits", "holm", tag, "with"), S.boot_index(len(without), "edits", "holm", tag, "without")
        for term, label in HOLM_TERMS:
            a, b = np.array([e[term] for e in has]), np.array([e[term] for e in without])
            reps = a[ih].mean(1) - b[iw].mean(1)
            tail = min(np.sum(reps <= 0) + 1, np.sum(reps >= 0) + 1) / (len(reps) + 1)
            tests.append({"tag": tag, "term": label, "n": (len(has), len(without)), "diff": a.mean() - b.mean(), "ci": S.interval(reps),
                          "p": min(1.0, 2 * tail)})
    order, running = np.argsort([t["p"] for t in tests]), 0.0
    for rank, k in enumerate(order):
        running = max(running, (len(tests) - rank) * tests[k]["p"])
        tests[k]["holm"] = min(1.0, running)
    rows_ = [[t["tag"], t["term"], f"{t['n'][0]}/{t['n'][1]}", f"{100 * t['diff']:+.1f}", span(*t["ci"]), f"{t['p']:.4f}",
              f"{t['holm']:.4f}", "yes" if t["ci"][0] > 0 or t["ci"][1] < 0 else "no", "yes" if t["holm"] < ALPHA else "no"]
             for t in sorted(tests, key=lambda t: t["p"])]
    numbers = "\n\n".join([
        f"The {edits_line(rows, excluded)}; an edit carries a tag when it adds at least {MIN_CHARS} characters of it. {S.N_BOOT:,} resamples "
        f"per tag (settings.rng, seed {S.SEED}); Holm over the {len(tests)} tests at alpha {ALPHA}; differences in log points x 100.",
        md_table(["category", "term", "edges with/without", "difference", "95% bootstrap interval", "p", f"p (Holm, {len(tests)} tests)",
                  "unadjusted interval excludes 0", f"survives Holm at {ALPHA}"], rows_)])
    record(__spec__.name, "edges_holm", numbers, S.runs_text(cells()),
           what=f"Whether the keyword-tag contrasts on the per-edge split of input tokens (with-minus-without the tag, on the actions and "
                f"turn-splitting terms) survive Holm's adjustment for the {len(tests)} tests.")


# ---- what sets an edit's cost; the cost-aware gate; does the editor remove text

def fit(x, y):
    reg, rho = stats.linregress(x, y), stats.spearmanr(x, y)
    return dict(n=len(x), r2=float(reg.rvalue ** 2), slope=float(reg.slope), spearman=float(rho.statistic), p=float(rho.pvalue))


def designev(rows, excluded):
    dlogx = np.array([e["dlogX"] for e in rows])
    edge = {name: fit(x, dlogx) for name, x in (("skill length", np.log([e["len_c"] / e["len_p"] for e in rows])),
                                                ("turns", np.array([e["dlogR"] for e in rows])), ("actions", np.array([e["dlogN"] for e in rows])))}
    attribution = json.loads((S.cache("main") / "attribution.json").read_text())["cells"]
    turns = {(c["method"], c["bench"]): c["rho_turns"] for c in json.loads((S.cache("tokens") / "compaction.json").read_text())["cells"]}
    ratio = {(S.ARM_NAME[r["method"]], r["bench"]): r["words_ratio"] for r in json.loads((S.cache("edits") / "skilltext.json").read_text())
             if r["model"] == S.DEEPSEEK}
    log_rho = np.array([c["A"]["total"] for c in attribution])
    skill = {"skill length": fit(np.log([ratio[(c["method"], c["bench"])] for c in attribution]), log_rho),
             "turns": fit(np.log([turns[(c["method"], c["bench"])] for c in attribution]), log_rho)}
    kept, rejected = [e for e in rows if e["accepted"]], [e for e in rows if not e["accepted"]]
    status = {s: dict(n=len(g), mean=float(np.mean([e["dlogX"] for e in g])), median=float(np.median([e["dlogX"] for e in g])),
                      share_above=float(np.mean([e["dlogX"] > 0 for e in g])), values=[e["dlogX"] for e in g])
              for s, g in (("accepted", kept), ("rejected", rejected))}
    passing = [e for e in rows if e["dQ"] >= 0 and e["dlogX"] <= 0]
    removing, shortening, adding = ([e for e in rows if e["rem_chars"] > 0], [e for e in rows if e["len_c"] < e["len_p"]],
                                    [e for e in rows if e["rem_chars"] == 0])
    save("designev", {"edge_level": edge, "test_level": skill, "status": status})
    f = lambda x, nd=2: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
    mean = lambda g: float(np.mean([e["dlogX"] for e in g])) if g else None  # noqa: E731
    above = lambda g: sum(e["dlogX"] > 0 for e in g)  # noqa: E731
    by_opt = {o: [e for e in rows if e["opt"] == o] for o in SCOPES}
    table_a = md_table(["level", "predictor", "n", "R^2", "Spearman", "p", "slope"],
                       [[label, name, v["n"], f(v["r2"]), f(v["spearman"]), f(v["p"], 3), f(v["slope"])]
                        for label, fits in ((f"dev edges ({len(rows)}), dlogX on", edge), (f"{len(attribution)} skills, log rho on", skill))
                        for name, v in fits.items()])
    table_b = md_table(["optimizer", "proposals", "dlogX > 0", "share"],
                       [[S.ARM_NAME[o], len(g), above(g), f(above(g) / len(g))] for o, g in by_opt.items()] + [["all", len(rows), above(rows), f(above(rows) / len(rows))]])
    table_k = md_table(["status", "n", "mean dlogX x100", "median x100", "share > 0"],
                       [[s, g["n"], f(100 * g["mean"], 1), f(100 * g["median"], 1), f(g["share_above"])] for s, g in status.items()])
    gate = (f"Gate accept iff dQ >= 0 and dlogX <= 0: {len(passing)} of {len(rows)} proposals pass (min dlogX over all proposals = "
            f"{f(100 * dlogx.min(), 1)} x100). It keeps {sum(e['accepted'] for e in passing)} of {len(kept)} accepted edges and would accept "
            f"{sum(not e['accepted'] for e in passing)} of {len(rejected)} rejected ones; mean dlogX of the kept set = {f(100 * mean(passing), 1) if passing else 'n/a'} "
            f"x100. {sum(e['dQ'] >= 0 for e in rows)} proposals have dQ >= 0; of the {len(rows) - above(rows)} that do not raise input tokens, "
            f"{sum(e['dQ'] < 0 for e in rows if e['dlogX'] <= 0)} lose dev success. Mann-Whitney p (accepted vs rejected dlogX) = "
            f"{f(stats.mannwhitneyu(status['accepted']['values'], status['rejected']['values']).pvalue, 3)}.")
    passed = md_table(["optimizer", "child", "accepted", "dlogX x100", "dQ"],
                      [[f"{S.ARM_NAME[e['opt']]} {e['bench']}", e["child"], e["accepted"], f(100 * e["dlogX"], 1), f(e["dQ"], 1)] for e in passing])
    table_c = md_table(["optimizer", "edges", "remove text", "shorten skill", "chars added", "chars removed"],
                       [[S.ARM_NAME[o] if o != "all" else o, len(g), sum(e["rem_chars"] > 0 for e in g), sum(e["len_c"] < e["len_p"] for e in g),
                         sum(e["add_chars"] for e in g), sum(e["rem_chars"] for e in g)] for o, g in [*by_opt.items(), ("all", rows)]])
    numbers = "\n\n".join([
        f"The {edits_line(rows, excluded)}, and the {len(attribution)} learned DeepSeek skills (log rho as Table 1 counts it; turns from "
        f"mech_compaction). Gate: accept iff dQ >= 0 and dlogX <= 0, counted over all {len(rows)} edits. No intervals: the sample is "
        f"{len(rows)} edits and {len(attribution)} skills.", "### A. what predicts dlogX", table_a,
        "### B. proposals and the cost-aware gate", table_b, table_k,
        "Mean token change of a kept and of a rejected edit, exp(mean dlogX) - 1: "
        + ", ".join(f"{s} {100 * math.expm1(g['mean']):.1f}%" for s, g in status.items()) + ".", gate, passed,
        "### C. operators: does the editor ever remove text?", table_c,
        f"Mean dlogX x100: edges that remove text ({len(removing)}) = {f(100 * mean(removing), 1) if removing else 'n/a'}; edges that shorten the "
        f"skill ({len(shortening)}) = {f(100 * mean(shortening), 1) if shortening else 'n/a'}; pure additions ({len(adding)}) = "
        f"{f(100 * mean(adding), 1) if adding else 'n/a'}."])
    record(__spec__.name, "designev", numbers, S.runs_text(cells()),
           what="What sets an edit's cost (skill length, turns or tool calls), whether a cost-aware gate would find cost-free edits among "
                "the logged ones, and whether the editors remove text: a counterfactual on logged proposals, not a new run.")


# ---- acceptance rules on the task model

@functools.lru_cache(maxsize=None)
def task_model(bench):
    """(the seed's dev executions, fitted prior) of DeepSeek on a benchmark."""
    dev, test, _ = TM.seed_executions(S.DEEPSEEK, bench)
    return dev, TM.priors(dev, test)[0]["fitted"]


@functools.lru_cache(maxsize=None)
def critical(bench, size):
    """The one-sided 0.05 critical net difference of a single-run comparison of `size` dev tasks under no effect."""
    dev, prior = task_model(bench)
    rng = S.rng("edits", "critical value", bench, size)
    return TM.critical_value(TM.task_probabilities(dev, prior, size, S.N_DRAWS, rng), rng)


def critical_by_size(rows):
    return {b: {n: critical(b, n) for n in sorted({e["n_pair"] for e in rows if e["bench"] == b})} for b in BENCHES}


def selrules(rows, excluded):
    result, sizes = {}, critical_by_size(rows)
    for bench in T.TEX_BENCHES:
        dev, prior = task_model(bench)
        rng = S.rng("edits", "acceptance rules", bench)
        p = TM.task_probabilities(dev, prior, len(dev), S.N_DRAWS, rng)
        c = critical(bench, len(dev))
        result[bench] = {"dev": len(dev), "critical": c, "null": TM.acceptance(p, p, c, rng),
                         "power": {str(g): TM.acceptance(p, TM.raised(p, g), c, rng) for g in S.GAINS}}
    save("selrules", {**result, "plot_gain": PLOT_GAIN})
    kept = [e for e in rows if e["accepted"]]
    refused = np.array([e["dQ"] * e["n_pair"] / 100 < sizes[e["bench"]][e["n_pair"]] for e in kept])
    dlogx, weight = 100 * np.array([e["dlogX"] for e in kept]), np.array([e["n_pair"] for e in kept], float)
    table = md_table(["benchmark", "rule", "false acceptance %"] + [f"accepted at true gain {g} tasks %" for g in S.GAINS],
                     [[b, label, f"{100 * r['null'][k]:.1f}"] + [f"{100 * r['power'][str(g)][k]:.1f}" for g in S.GAINS]
                      for b, r in result.items() for k, label in RULES])
    groups = [[name, int(m.sum()), f"{dlogx[m].mean():.1f}", f"{np.average(dlogx[m], weights=weight[m]):.1f}"]
              for name, m in (("refused", refused), ("survive", ~refused)) if m.any()]
    numbers = "\n".join([
        f"Task model: the seed's dev and test executions of Table 1's DeepSeek runs, {S.N_DRAWS} draws of the task probabilities per "
        f"benchmark, true gains of g = {', '.join(map(str, S.GAINS))} dev tasks (the figure shows g = {PLOT_GAIN}); rule (d) sums 3 runs. "
        f"Logged edits: the {edits_line(rows, excluded)}, {len(kept)} accepted. Rule (b) applied to the {len(kept)} accepted ones: a gain in "
        "tasks (dQ x n_pair / 100) of at least the one-sided 0.05 critical value of an n_pair-task comparison under no effect.", "", table, "",
        "One-sided 0.05 critical net difference of a single-run comparison under no effect (fitted prior):", "",
        md_table(["benchmark", "dev tasks", "critical value (tasks)", "critical value by comparison size (tasks)"],
                 [[b, r["dev"], r["critical"], json.dumps({str(n): c for n, c in sizes[b].items()})] for b, r in result.items()]), "",
        "Rules: " + "; ".join(label for _, label in RULES) + ".", "",
        f"Logged accepted edges refused by rule (b): {int(refused.sum())} of {len(kept)} (survive {int((~refused).sum())}).", "",
        md_table(["benchmark", "accepted edges", "refused by (b)"],
                 [[b, sum(e["bench"] == b for e in kept), sum(f for e, f in zip(kept, refused) if e["bench"] == b)] for b in T.TEX_BENCHES]), "",
        md_table(["group", "edges", "mean dlogX (x100)", "n_pair-weighted dlogX (x100)"], groups)])
    record(__spec__.name, "selrules", numbers, S.runs_text(cells()),
           what="How often four acceptance rules accept one candidate against its parent: a candidate identical to the parent (false "
                "acceptance) and one that truly solves g more dev tasks in expectation; and how many logged accepted dev edits a "
                "critical-value gate (rule b) would have refused, with the token cost they carried.")


# ---- tie-break: what separates a candidate from its parent

@functools.lru_cache(maxsize=None)
def panel(path, model):
    """{task: (success 0/1 or None, input tokens or None)} of the first repetition of one evaluation, by Table 1's reader."""
    d, read = json.loads(Path(path).read_text()), T.row_reader(model)
    eps = {}
    for e in d["episodes"]:
        eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    return {t["task_id"]: (int(bool(e["success"])) if e and TM.verified(e) else None, read(e)["X"] or None)
            for t in d["panel"] if t.get("repetition", 0) == 0 for e in [eps.get((t["task_id"], 0), {})]}


def compare(a, b):
    """Paired success and token statistics of b against a on their shared tasks."""
    ty = [t for t in a if t in b and a[t][0] is not None and b[t][0] is not None]
    d = np.array([b[t][0] - a[t][0] for t in ty], float)
    lr = np.array([math.log(b[t][1] / a[t][1]) for t in a if t in b and a[t][1] and b[t][1]], float)
    D, se_d = float(d.sum()), float(math.sqrt(len(d)) * d.std(ddof=1)) if len(d) > 1 else float("nan")
    Lm, se_l = float(lr.mean()) if len(lr) else float("nan"), float(lr.std(ddof=1) / math.sqrt(len(lr))) if len(lr) > 1 else float("nan")
    return {"n_y": len(ty), "D": D, "se_D": se_d, "z_D": D / se_d if se_d > 0 else (0.0 if D == 0 else math.copysign(math.inf, D)),
            "n_x": len(lr), "L": Lm, "se_L": se_l, "z_L": Lm / se_l if se_l > 0 else float("nan"),
            "sd_l": float(lr.std(ddof=1)) if len(lr) > 1 else float("nan"), "solved": (int(sum(a[t][0] for t in ty)), int(sum(b[t][0] for t in ty)))}


def edge_stats(model, crit):
    table = {(e["opt"], e["bench"], e["parent"], e["child"], e["panel"]): e for e in L.dev_edges()} if model == S.DEEPSEEK else None
    rows = []
    for method, bench, e, tp, tc, index in L.scored_edges(model):
        panels = [k for k in L.shared_panels(index, tp, tc, "dev") if (method, bench) != L.EXCLUDED
                  and (table is None or (method, bench, e["parent"], e["child"], k[1]) in table)]
        if panels:
            key = max(panels, key=lambda k: len(panel(index[L.version(tp)][k], model)))
            st = compare(panel(index[L.version(tp)][key], model), panel(index[L.version(tc)][key], model))
            rows.append(dict(st, opt=method, bench=bench, parent=e["parent"], child=e["child"], accepted=e["accepted"],
                             crit=crit(bench, st["n_y"]), from_seed=L.version(tp) == SEED,
                             logged=table[(method, bench, e["parent"], e["child"], key[1])] if table else None))
    return rows


def summary(rows, z=S.Z95):
    share = lambda f, rs: (sum(map(f, rs)), len(rs))  # noqa: E731
    groups = {"all": rows, "accepted": [r for r in rows if r["accepted"]], "rejected": [r for r in rows if not r["accepted"]],
              **{m: [r for r in rows if r["opt"] == m] for m in L.METHODS}}
    return {g: {"edges": len(rs), "success_z": share(lambda r: abs(r["z_D"]) >= z, rs), "success_crit": share(lambda r: abs(r["D"]) >= r["crit"], rs),
                "tokens_z": share(lambda r: abs(r["z_L"]) >= z, rs), "up": sum(r["z_L"] >= z for r in rs), "down": sum(r["z_L"] <= -z for r in rs),
                **{f"median_{k}": float(np.nanmedian([f(r) for r in rs])) for k, f in (
                    ("zD", lambda r: abs(r["z_D"])), ("zL", lambda r: abs(r["z_L"])), ("seD", lambda r: r["se_D"]),
                    ("seL", lambda r: r["se_L"]), ("L", lambda r: abs(r["L"])))},
                "tokens_when_success_not": share(lambda r: abs(r["z_L"]) >= z, [r for r in rs if abs(r["D"]) < r["crit"]])}
            for g, rs in groups.items() if rs}


def noise_floor():
    """Per benchmark the seed's distinct full-panel dev draws in DeepSeek's seed dev pool, compared in pairs."""
    out = {}
    for bench in BENCHES:
        n_dev, files = len(task_model(bench)[0]), TM.dev_files(S.DEEPSEEK, bench)
        draws = {}
        for f in files:
            eps = json.loads(Path(f).read_text())["episodes"]
            if len(eps) == n_dev:
                draws.setdefault(frozenset((e["task_id"], TM.signature(e)) for e in eps if TM.verified(e)), f)
        evals = [(TM.run_name(f, S.DEEPSEEK), panel(f, S.DEEPSEEK)) for f in draws.values()]
        evals += [(runs[0], {t: (o, x) for t, o, x in got}) for runs, got in TM.kept_dev(bench, files).items()]
        out[bench] = [dict(compare(a, b), files=(na, nb)) for i, (na, a) in enumerate(evals) for nb, b in evals[i + 1:]]
    return out


def pool_rule(bench, key, pool, evs, returned, crit):
    tasks = sorted(set.intersection(*[{t for t in ev if ev[t][1] and ev[t][0] is not None} for ev in evs.values()]))
    solved = {v: sum(y for y, _ in evs[v].values() if y is not None) for v in pool}
    logx = {v: float(np.mean([math.log(evs[v][t][1]) for t in tasks])) for v in pool}
    c, best = crit(bench, len(evs[pool[0]])), max(solved.values())
    eligible = [v for v in pool if best - solved[v] < c]
    anchor = SEED if SEED in eligible else returned
    cheaper = [v for v in eligible if v != anchor and compare(evs[anchor], evs[v])["z_L"] <= -S.Z95]
    pick = min(eligible, key=lambda v: (round(logx[v], 12), pool.index(v)))
    return {"crit": c, "best": best, "solved": solved, "logx": logx, "eligible": eligible, "pool": pick,
            "pool_z": min(cheaper, key=lambda v: logx[v]) if cheaper else anchor, "pick_z": compare(evs[anchor], evs[pick])["z_L"]}


def sequential(steps, index, model, crit, z):
    """The skill a gate returns over [(candidate version, panel key or None)], starting from the seed."""
    current = SEED
    for cand, key in steps:
        keys = [key] if key else [k for k in set(index.get(current, {})) & set(index.get(cand, {})) if k[0] == "dev"]
        if not keys or (key and (key not in index.get(current, {}) or key not in index.get(cand, {}))):
            continue
        st = compare(panel(index[current][keys[0]], model), panel(index[cand][keys[0]], model))
        c = crit(st["n_y"])
        if st["D"] >= c or (st["D"] > -c and (st["z_L"] <= -S.Z95 if z else st["L"] < 0)):
            current = cand
    return current


def replay(model, crit):
    out = {}
    for method in L.METHODS:
        for bench in BENCHES:
            if not S.dev_gated(model, bench, method):
                continue
            nodes, edges_, index = L.lineage(model, bench, method)
            names = {**{L.version(t): n for n, t in nodes.items()}, SEED: "seed"}
            returned = L.version(S.skill(model, bench, method))
            count = {}
            for k in (k for v in index.values() for k in v if k[0] == "dev"):
                count[k] = count.get(k, 0) + 1
            key = max(count, key=count.get)

            def order(v, n=""):
                n = names.get(v, "")
                return (-1, 0) if v == SEED else (0, int(n[1:])) if n[:1] in ("c", "s") and n[1:].isdigit() else (1, v)
            pool = sorted([v for v in index if key in index[v]], key=order)
            evs = {v: panel(index[v][key], model) for v in pool}
            cell, picks = {"returned": returned, "names": names, "key": key}, {}
            if len(evs[pool[0]]) == len(task_model_of(model, bench)) and len(pool) > 2:
                cell["pool"] = pr = pool_rule(bench, key, pool, evs, returned, crit)
                pr["order"] = pool
                picks.update(pool=pr["pool"], pool_z=pr["pool_z"])
            if method == "skillopt" or (method == "skilladam" and "pool" not in cell):
                steps = [(L.version(nodes[e["child"]]), key if method == "skillopt" else None) for e in edges_]
                picks.update(seq=sequential(steps, index, model, lambda n: crit(bench, n), False),
                             seq_z=sequential(steps, index, model, lambda n: crit(bench, n), True))
            for rule, v in picks.items():
                if "pool" in cell and v in evs and returned in evs:
                    dev = {"returned": compare(evs[returned], evs[v]), "seed": compare(evs[SEED], evs[v]) if SEED in evs else None}
                else:
                    shared = lambda x: [k for k in set(index.get(v, {})) & set(index.get(x, {})) if k[0] == "dev"]  # noqa: E731
                    dev = {name: compare(panel(index[x][shared(x)[0]], model), panel(index[v][shared(x)[0]], model)) if shared(x) and v != x else None
                           for name, x in (("returned", returned), ("seed", SEED))}
                picks[rule] = {"pick": v, **dev}
            out[(method, bench)] = dict(cell, picks=picks)
    return out


@functools.lru_cache(maxsize=None)
def task_model_of(model, bench):
    return TM.seed_executions(model, bench)[0]


def gate_counts(rows, z):
    cheaper = (lambda r: r["z_L"] <= -S.Z95) if z else (lambda r: r["L"] < 0)
    passes = lambda r: r["D"] >= r["crit"] or (r["D"] > -r["crit"] and cheaper(r))  # noqa: E731
    acc, rej = [r for r in rows if r["accepted"]], [r for r in rows if not r["accepted"]]
    kept, refused = [r for r in acc if passes(r)], [r for r in acc if not passes(r)]
    by = "; ".join(f"{S.ARM_NAME[m]} {sum(not passes(r) for r in acc if r['opt'] == m)}/{sum(r['opt'] == m for r in acc)} refused, "
                   f"{sum(passes(r) for r in rej if r['opt'] == m)}/{sum(r['opt'] == m for r in rej)} admitted" for m in L.METHODS)
    mean = lambda g: f"{np.mean([r['L'] for r in g]):+.3f}" if g else "n/a"  # noqa: E731
    return (f"- Edge gate {'Z' if z else 'P'}{f' (cheaper: token z <= -{S.Z95})' if z else ''}: refuses {len(refused)} of {len(acc)} accepted edges, admits {sum(map(passes, rej))} of {len(rej)} "
            f"rejected ({by}). Mean L of refused accepted edges {mean(refused)}, of kept {mean(kept)}.")


def tiebreak(rows):
    sizes = critical_by_size(rows)
    crit = lambda bench, n: sizes[bench][min(sizes[bench], key=lambda k: abs(k - n))]  # noqa: E731
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    q = lambda m, b, method: table1[f"{m}|{b}"][ROW[method]]  # noqa: E731
    models = (S.DEEPSEEK, S.LUNA)
    edge, replays, noise = {m: edge_stats(m, crit) for m in models}, {m: replay(m, crit) for m in models}, noise_floor()
    summ = {m: summary(edge[m]) for m in models}
    ds = edge[S.DEEPSEEK]
    sd = {b: float(np.median([r["sd_l"] for r in ds if r["bench"] == b])) for b in BENCHES}
    crit_text = "; ".join(f"{SHORT[b]} " + ", ".join(f"{n} tasks: {c}" for n, c in sorted(sizes[b].items(), reverse=True)) for b in BENCHES)
    same_d = sum(abs(r["D"] - r["logged"]["dQ"] * r["logged"]["n_pair"] / 100) < 1e-9 for r in ds)
    corr = np.corrcoef([r["L"] for r in ds], [r["logged"]["dlogX"] for r in ds])[0, 1]
    subset = sorted({r["n_y"] for r in ds if r["opt"] == "skilladam"})
    lines = [f"Edits: {len(ds)} DeepSeek (the dev edits of the edge table) and {len(edge[S.LUNA])} {S.MODEL_SHORT[S.LUNA]} (every dev-scored edit of its "
             f"dev-gated runs, SkillAdam on Terminal-Bench 2.1 left out); SkillOpt's chain is replayed on the full dev panel, SkillAdam's "
             f"iterations on their {', '.join(map(str, subset))}-task dev subsets. Critical values (rule b of mech_selrules, the one-sided 0.05 "
             f"critical value of the DeepSeek task model; nearest comparison size, for both models): {crit_text}. "
             f"D equals the edge table's dQ x n_pair / 100 on {same_d} of {len(ds)} DeepSeek edges; L correlates {corr:.3f} with its dlogX "
             f"(dlogX keeps only the tasks where the turn model applies). Separable = |z| >= {S.Z95}. Test values are Table 1's (avg@3).", "",
             "## (a) What separates a candidate from its parent on dev", ""]
    lines += [md_table(["model", "edges", "n", "success |z|>=1.96", "success |D|>=crit", "tokens |z|>=1.96", "tokens up / down", "median |z| success",
                        "median |z| tokens", "median SE of D (tasks)", "median SE of L", "median |L|", "tokens separable when success is not (|D|<crit)"],
                       [[m, g, s["edges"], frac(s["success_z"]), frac(s["success_crit"]), frac(s["tokens_z"]), f"{s['up']} / {s['down']}",
                         f"{s['median_zD']:.2f}", f"{s['median_zL']:.2f}", f"{s['median_seD']:.2f}", f"{s['median_seL']:.3f}", f"{s['median_L']:.3f}",
                         frac(s["tokens_when_success_not"])] for m in models for g, s in summ[m].items()]), ""]
    within = [r for r in ds if not r["from_seed"]]
    both = sum(abs(r["z_D"]) >= S.Z95 and abs(r["z_L"]) >= S.Z95 for r in within)
    lines += [f"DeepSeek edges whose parent is not the seed (the seed's dev episodes come from an earlier run): tokens separable "
              f"{sum(abs(r['z_L']) >= S.Z95 for r in within)}/{len(within)}, success {sum(abs(r['z_D']) >= S.Z95 for r in within)}/{len(within)} (|z|) "
              f"and {sum(abs(r['D']) >= r['crit'] for r in within)}/{len(within)} (critical value). {both} of them have both separable.", "",
              "Edges where success is separable:", ""]
    lines += [f"- {m} {S.ARM_NAME[r['opt']]} {SHORT[r['bench']]} {r['parent'][:12]} -> {r['child'][:12]}: D {int(r['D']):+d}/{r['n_y']} (crit {r['crit']}, "
              f"z {r['z_D']:+.1f}), token ratio {math.exp(r['L']):.2f} (z {r['z_L']:+.1f}), accepted={r['accepted']}"
              for m in models for r in edge[m] if abs(r["z_D"]) >= S.Z95 or abs(r["D"]) >= r["crit"]]
    pairs = [p for ps in noise.values() for p in ps]
    apart = [p for p in pairs if abs(p["z_L"]) >= S.Z95]
    lines += ["", "### Seed-vs-seed noise floor (DeepSeek dev)", "",
              md_table(["bench", "seed draws", "solved", "tasks", "SD per-task log ratio", "mean log ratio", "SE", "z"],
                       [[SHORT[b], " vs ".join(p["files"]), f"{p['solved'][0]} vs {p['solved'][1]}", p["n_x"], f"{p['sd_l']:.2f}", f"{p['L']:+.3f}",
                         f"{p['se_L']:.3f}", f"{p['z_L']:+.1f}"] for b, ps in noise.items() for p in ps]), "",
              "Per-task SD of the log token ratio, seed vs seed: " + ", ".join(f"{SHORT[b]} {np.mean([p['sd_l'] for p in ps]):.2f}" + (" (one pair)" if len(ps) == 1 else "")
                                                                               for b, ps in noise.items() if ps)
              + f"; median over the {len(ds)} edges: " + ", ".join(f"{SHORT[b]} {v:.2f}" for b, v in sd.items()) + ". Mean paired SE of a seed pair: "
              + ", ".join(f"{SHORT[b]} {np.mean([p['se_L'] for p in ps]):.2f} on {ps[0]['n_x']} tasks" for b, ps in noise.items() if ps)
              + f". {len(apart)} of {len(pairs)} seed pairs is separable on tokens ("
              + ", ".join(f"{SHORT[b]}, {p['L']:+.2f}" for b, ps in noise.items() for p in ps if abs(p["z_L"]) >= S.Z95)
              + f"); the others are within {max(abs(p['z_L']) for p in pairs if abs(p['z_L']) < S.Z95):.1f} SE. "
              f"{S.MODEL_SHORT[S.LUNA]} has one seed execution per dev task, so no {S.MODEL_SHORT[S.LUNA]} seed-vs-seed pair exists.", ""]
    for m in models:
        lines += [f"## (b) Tie-break replay, {m}", ""]
        rows_ = []
        for (method, bench), c in replays[m].items():
            ret, name = q(m, bench, method), lambda v: c["names"].get(v, v)  # noqa: E731
            for rule, r in c["picks"].items():
                same = r["pick"] == c["returned"]
                test = (f"seed: Q {q(m, bench, 'native-seed')['Q']:.1f} vs returned {ret['Q']:.1f} (rho {ret['rho_trj']:.2f})" if r["pick"] == SEED else
                        f"= returned (Q {ret['Q']:.1f}, rho {ret['rho_trj']:.2f})" if same else "none (dev only)")
                dv, ds_ = r["returned"], r["seed"]
                rows_.append([f"{S.ARM_NAME[method]} {SHORT[bench]}", name(c["returned"]), rule, name(r["pick"]), "yes" if same else "no",
                              f"{int(dv['D']):+d}/{dv['n_y']}" if dv else "-", f"{math.exp(dv['L']):.2f}" if dv and dv["L"] == dv["L"] else "-",
                              f"{dv['z_L']:+.1f}" if dv and dv["z_L"] == dv["z_L"] else "-",
                              "0" if r["pick"] == SEED else f"{int(ds_['D']):+d}/{ds_['n_y']}" if ds_ else "-",
                              "1.00" if r["pick"] == SEED else f"{math.exp(ds_['L']):.2f}" if ds_ and ds_["L"] == ds_["L"] else "-", test])
        lines += [md_table(["run", "returned", "rule", "rule returns", "same", "dev D vs returned", "dev tokens vs returned", "z", "dev D vs seed",
                            "dev tokens vs seed", "test (Table 1)"], rows_), "", gate_counts(edge[m], False), gate_counts(edge[m], True), "",
                  "Pools (dev solved on the full panel; token change vs the seed in log points; * = eligible; uN = SkillOpt slow-update skill; "
                  "a bare hash = a version the run scored on dev that is in no lineage file, a re-proposal kept in the pool because the run scored it):", ""]
        for (method, bench), c in replays[m].items():
            if p := c.get("pool"):
                s0 = p["logx"].get(SEED, 0.0)
                lines.append(f"- {method}|{bench} (crit {p['crit']}, best {p['best']}): " + ", ".join(
                    f"{c['names'].get(v, v[:6])} {p['solved'][v]} {p['logx'][v] - s0:+.2f}{'*' if v in p['eligible'] else ''}" for v in p["order"]))
        lines.append("")
    lines += ["## (c) Verdict", ""]
    for m in models:
        cells_ = [(method, bench, c) for (method, bench), c in replays[m].items() if c["returned"] != SEED]
        z = {(method, bench): c["picks"].get("pool_z", c["picks"].get("seq_z"))["pick"] for method, bench, c in cells_}
        seed = [(method, bench) for method, bench, _ in cells_ if z[(method, bench)] == SEED]
        dq = [q(m, b, o)["Q"] - q(m, b, "native-seed")["Q"] for o, b in seed]
        rho = [q(m, b, o)["rho_trj"] for o, b in seed]
        other = [f"{S.ARM_NAME[o]} {SHORT[b]} {c['names'].get(z[(o, b)], z[(o, b)])}" for o, b, c in cells_ if z[(o, b)] != SEED]
        lines.append(f"- Z rule, {m}: returns the seed in {len(seed)} of the {len(cells_)} runs that returned a learned skill ("
                     + ", ".join(f"{S.ARM_NAME[o]} {SHORT[b]}" for o, b in seed) + "). On test (Table 1) the skills those runs returned differ from "
                     f"the seed by {', '.join(f'{x:+.1f}' for x in dq)} points (mean {np.mean(dq):+.1f}) at rho {min(rho):.2f}-{max(rho):.2f}, so the rule's "
                     f"choice reads {100 * (1 - 1 / min(rho)):.1f}-{100 * (1 - 1 / max(rho)):.1f}% fewer input tokens than the returned skill. "
                     + (f"Elsewhere it returns {', '.join(other)}. " if other else "")
                     + f"Tokens separate {summ[m]['all']['tokens_z'][0]} of {summ[m]['all']['edges']} edges and success "
                     f"{summ[m]['all']['success_z'][0]} of {summ[m]['all']['edges']}.")
    admitted = [r for r in ds if -r["crit"] < r["D"] < r["crit"] and r["L"] < 0]
    least = min(admitted, key=lambda r: -r["L"])
    p_z = [abs(c["pool"]["pick_z"]) for m in models for c in replays[m].values() if "pool" in c and c["pool"]["pool"] != c["pool"]["pool_z"]]
    lines += [f"- P (point-estimate 'cheaper') admits {len(admitted)} DeepSeek edges on cost alone, the least of them {S.ARM_NAME[least['opt']]} "
              f"{SHORT[least['bench']]} {least['child']} for a {-100 * math.expm1(least['L']):.1f}% token difference; where its pool pick differs "
              f"from the Z rule's, the token z of the pick against the anchor is at most {max(p_z, default=0):.1f} in absolute value.",
              f"- Cost separates {summ[S.DEEPSEEK]['all']['tokens_z'][0]} of {len(ds)} DeepSeek edits and success {summ[S.DEEPSEEK]['all']['success_z'][0]} "
              f"of them; under {S.MODEL_SHORT[S.LUNA]} cost separates {summ[S.LUNA]['all']['tokens_z'][0]} of {len(edge[S.LUNA])} and success "
              f"{summ[S.LUNA]['all']['success_z'][0]} of {len(edge[S.LUNA])}.", ""]
    groups = [("within run, all", within), ("within run, accepted", [r for r in within if r["accepted"]]),
              ("within run, rejected", [r for r in within if not r["accepted"]]),
              ("seed parent (across runs), accepted", [r for r in ds if r["from_seed"] and r["accepted"]]), ("all accepted", [r for r in ds if r["accepted"]])]
    count = {g: [len(es), sum(abs(e["z_L"]) >= S.Z95 for e in es), sum(e["z_L"] >= S.Z95 for e in es), sum(abs(e["z_D"]) >= S.Z95 for e in es),
                 sum(e["D"] >= e["crit"] for e in es)] for g, es in groups}
    wa, sa, aa = count["within run, accepted"], count["seed parent (across runs), accepted"], count["all accepted"]
    lines += ["## (d) Within-run counts for the accepted edges", "",
              f"DeepSeek edges; within run = the parent is not the seed (the restriction of the {count['within run, all'][1]}/{count['within run, all'][0]} "
              "line above).", "",
              md_table(["DeepSeek edges", "n", "tokens |z|>=1.96", "tokens up (z>=1.96)", "success |z|>=1.96", "success D >= crit"], [[g] + count[g] for g, _ in groups]), "",
              f"Of the {wa[0]} within-run accepted edits, {wa[2]} raised tokens beyond noise and {wa[4]} raised success beyond its critical value; "
              f"the all-accepted {aa[2]}/{aa[0]} and {aa[4]}/{aa[0]} include {sa[2]} and {sa[4]} seed-parent edges compared across runs (run-date "
              f"drift; TB2 seed-vs-seed pair z={', '.join(format(p['z_L'], '+.1f') for p in noise['terminal-bench-2-1'])})."]
    record(__spec__.name, "tiebreak", "\n".join(lines), S.runs_text(cells(models)),
           what="Whether success or cost separates a candidate from its parent on the same dev evidence, and what a cost tie-break would "
                "have returned from the optimizers' own candidates.")


def cells(models=(S.DEEPSEEK,)):
    return [(m, b, o) for m in models for b in BENCHES for o in L.METHODS if S.dev_gated(m, b, o)]


def main(steps):
    rows = L.dev_edges()
    excluded = [r for r in json.loads((S.cache("edits") / L.TABLE).read_text()) if r["split"] == "dev" and L.excluded(r)]
    for name in steps:
        {"edges": lambda: edges(rows, excluded), "holm": lambda: holm(rows, excluded), "designev": lambda: designev(rows, excluded),
         "selrules": lambda: selrules(rows, excluded), "tiebreak": lambda: tiebreak(rows)}[name]()


if __name__ == "__main__":
    main(sys.argv[1:] or ["edges", "holm", "designev", "selrules", "tiebreak"])
