"""Four-term split of each learned skill's change in log input tokens against the seed, on Table 1's test runs.

compute  -> <cache>/main/attribution.json: per DeepSeek cell (A) the split on 3-run task means, (B) per run pair, (C) the
            per-request change of each prefix column, and Table 1's rho on the same tasks
attr     -> <cache>/appc/attr.json: the split on one run per side (each side's first test run), read by the rule coding

graphs() and split() also serve the token analyses in aux/tokens: the episode graphs of a cell's runs for either model,
and a cell's usable tasks with the 3-run means of both sides."""
import functools, json, math, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.analysis import settings as S
from src.analysis.episodes import metrics
from src.graph import episode as W
from src.utils import dataloader as C

MODEL = S.DEEPSEEK
OPT = [("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"), ("GEPA", "gepa")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
PREFIX = list(W.FIXED)
DROPPED = ["no verified outcome / N, X in all six runs", "no graph episode in some run",
           "episode without usable metrics (R<2, no call, no history)", "no positive history on the 3-run mean"]
ATTR_NAMES = dict(more_actions="more_actions", turn_splitting="more_requests", content="longer_content", skill_text="longer_skill")


def run_files(method, bench, model=MODEL):
    """({rep: [evaluation.json]} of the skill, the same for the seed), as Table 1 reads them."""
    return S.cell_files(model, bench, method), S.cell_files(model, bench, "native-seed")


def _luna_graph(ep):
    requests = [r for r in ep["requests"] if r["input_tokens"] is not None]
    return W._episode((dict(ep, requests=requests), "")) if requests else None


def graphs(files, model=MODEL):
    """{rep: {task: Episode}} of a cell's test runs. DeepSeek: the parse of each file (src.graph.episode.load), the later
    files of a repetition filling only tasks without an outcome. GPT-6-Luna: the episodes Table 1 merges that have a verified
    outcome, rebuilt without their provider-failed requests (no usage, retried), so R, N and X are Table 1's Luna counts."""
    if model != S.LUNA:
        out = {}
        for k, fs in sorted(files.items()):
            got = {}
            for f in fs:
                got.update({t: e for t, e in {e.task: e for e in W.load(f)}.items() if t not in got or got[t].success is None})
            out[k] = got
        return out
    items = [(k, t, e) for k, rep in S.episodes(files).items() for t, e in rep.items() if S.known(e)]
    with ProcessPoolExecutor(24) as pool:
        built = list(pool.map(_luna_graph, [e for _, _, e in items], chunksize=4))
    out = {k: {} for k in files}
    for (k, t, _), g in zip(items, built):
        if g is not None:
            out[k][t] = g
    return out


def load_runs(files, model=MODEL):
    """({rep: {task: (Episode, metrics or None)}}, {rep: {task: Table 1 row}}) of {rep: [evaluation.json]}."""
    eps = {k: {t: (e, metrics(e)) for t, e in rep.items()} for k, rep in graphs(files, model).items()}
    return eps, S.rows(files, model)


@functools.lru_cache(maxsize=None)
def side(model, bench, method):
    """load_runs of one row of Table 1 (an optimizer that returned the seed reads the Seed row's runs)."""
    return load_runs(S.cell_files(model, bench, method), model)


def verified(row):
    return bool(row) and None not in (row["success"], row["N"], row["X"])


def means(rows):
    """{task: mean N and X over the runs with a verified outcome and both counts}: Table 1's per-task means."""
    per = {}
    for rep in rows.values():
        for t, r in rep.items():
            if verified(r):
                per.setdefault(t, []).append(r)
    return {t: {k: float(np.mean([r[k] for r in v])) for k in ("N", "X")} for t, v in per.items()}


def common(ref, own):
    """Table 1's common task set: both sides averaged, mean N and X above 0."""
    return [t for t in sorted(ref) if t in own and min(ref[t]["N"], ref[t]["X"], own[t]["N"], own[t]["X"]) > 0]


def mean_metrics(per_run):
    """R, N, X, F averaged over runs, h from X = R F + R (R - 1) / 2 h; None without positive history."""
    R, N, X, F = (float(np.mean([m[k] for m in per_run])) for k in "RNXF")
    if X - R * F <= 0 or R <= 1:
        return None
    return dict(R=R, N=N, X=X, F=F, h=(X - R * F) / (R * (R - 1) / 2), sF=R * F / X)


def four_terms(pairs):
    """Per-task terms over [(seed metrics, skill metrics)] with midpoint shares and the exact e_R; also the measured dlogX."""
    def d(key):
        return np.array([math.log(b[key] / a[key]) for a, b in pairs])
    dR, dN, dh, dX, dF = d("R"), d("N"), d("h"), d("X"), d("F")
    sF = np.array([(a["sF"] + b["sF"]) / 2 for a, b in pairs])
    Rm = np.array([(a["R"] + b["R"]) / 2 for a, b in pairs])
    eR = 1 + (1 - sF) * Rm / (Rm - 1)
    return dict(more_actions=eR * dN, turn_splitting=-eR * (dN - dR), content=(1 - sF) * dh, skill_text=sF * dF), dX


def summarize(terms, dX):
    """Task means of the terms and of the total, the residual, and the largest term."""
    out = {k: float(v.mean()) for k, v in terms.items()} | {"total": float(dX.mean())}
    resid = dX - sum(terms.values())
    out |= dict(residual_mean=float(resid.mean()), residual_max_abs=float(np.abs(resid).max()))
    return out | {"largest": max(TERMS, key=lambda k: abs(out[k]))}


def split(model, bench, method):
    """One cell: Table 1's common tasks, and the usable ones with the 3-run means of both sides (seed, skill) and the runs
    averaged. DeepSeek needs a verified outcome, N and X and a usable graph episode in all six runs; GPT-6-Luna averages the
    runs Table 1 averages and drops a task when one of them has no usable graph episode."""
    own, seed = side(model, bench, method), side(model, bench, "native-seed")
    reps = sorted(own[1])
    seed_means, own_means = means(seed[1]), means(own[1])
    tasks, usable, dropped = common(seed_means, own_means), {}, dict.fromkeys(DROPPED, 0)
    for t in tasks:
        runs = [[r for r in sorted(rows) if verified(rows[r].get(t))] for _, rows in (seed, own)]
        if model == MODEL and any(rs != reps for rs in runs):
            dropped[DROPPED[0]] += 1
        elif not all(t in eps.get(r, {}) for (eps, _), rs in zip((seed, own), runs) for r in rs):
            dropped[DROPPED[1]] += 1
        elif not all(eps[r][t][1] for (eps, _), rs in zip((seed, own), runs) for r in rs):
            dropped[DROPPED[2]] += 1
        elif None in (pair := tuple(mean_metrics([eps[r][t][1] for r in rs]) for (eps, _), rs in zip((seed, own), runs))):
            dropped[DROPPED[3]] += 1
        else:
            usable[t] = dict(pair=pair, runs=runs)
    log_rho = {t: math.log(own_means[t]["X"] / seed_means[t]["X"]) for t in tasks}
    return dict(own=own[0], seed=seed[0], rows=(seed[1], own[1]), reps=reps, common=tasks, usable=usable, dropped=dropped, log_rho=log_rho)


def prefix_columns(c):
    """Per prefix column, the mean over tasks of the per-request tokens (mean over the runs), skill minus seed, and the seed's."""
    def level(eps, runs, task, col):
        return float(np.mean([e.Xrg[:, e.comps.index(col)].mean() if col in e.comps else 0.0 for e in (eps[r][task][0] for r in runs)]))
    rows = {col: [(level(c["own"], u["runs"][1], t, col), level(c["seed"], u["runs"][0], t, col)) for t, u in c["usable"].items()]
            for col in PREFIX}
    return {col: dict(delta=float(np.mean([a - b for a, b in v])), seed=float(np.mean([b for _, b in v]))) for col, v in rows.items()}


def cell(disp, method, bench, short):
    c = split(MODEL, bench, method)
    assert c["reps"] == [0, 1, 2], (method, bench, c["reps"])
    pairs = [u["pair"] for u in c["usable"].values()]
    A = summarize(*four_terms(pairs))
    B = [dict(run=r, **summarize(*four_terms([(c["seed"][r][t][1], c["own"][r][t][1]) for t in c["usable"]]))) for r in c["reps"]]
    full, used = (float(np.mean([c["log_rho"][t] for t in ts])) for ts in (c["common"], c["usable"]))
    print(disp, bench, "n", len(pairs), "of", len(c["common"]), {k: round(v, 3) for k, v in A.items() if k != "largest"}, A["largest"])
    return dict(method=disp, bench=bench, short=short, run=S.run(MODEL, bench, method).name, n=len(pairs), n_table1=len(c["common"]),
                dropped=c["dropped"], A=A, B=B, same_largest_all_runs=len({b["largest"] for b in B} | {A["largest"]}) == 1,
                table1_log_rho_usable=used, table1_log_rho_full=full, table1_rho_full=math.exp(full), total_gap_to_table1=A["total"] - used,
                C=prefix_columns(c))


def compute():
    C.write_json("attribution", dict(model=MODEL, cells=[cell(d, m, b, s) for d, m in OPT for b, s in BENCH]))


def attr():
    """The four-term split on one run per side (each side's first test run), per DeepSeek skill."""
    skills = []
    for disp, method in OPT:
        for bench, _ in BENCH:
            own_files, seed_files = ({0: files[0]} for files in run_files(method, bench))
            (own, own_rows), (seed, seed_rows) = load_runs(own_files), load_runs(seed_files)
            pairs = [(seed[0][t][1], m) for t, (_, m) in own[0].items() if m and seed[0].get(t, (0, None))[1]
                     and verified(r := own_rows[0].get(t)) and verified(s := seed_rows[0].get(t)) and min(r["N"], r["X"], s["N"], s["X"]) > 0]
            terms, dX = four_terms(pairs)
            seg = {ATTR_NAMES[k]: float(v.mean()) for k, v in terms.items()}
            skills.append(dict(model=MODEL, bench=bench, method=disp, n=len(pairs), run=S.run(MODEL, bench, method).name, **seg,
                               segments_sum=sum(seg.values()), dlogX=float(dX.mean()), ratio=C.gmean(np.exp(dX))))
    (S.cache("appc") / "attr.json").write_text(json.dumps(dict(skills=skills), indent=1))


if __name__ == "__main__":
    {"compute": compute, "attr": attr}[sys.argv[1]]()
