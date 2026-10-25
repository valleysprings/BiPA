"""Four-term split of each skill's change in log input tokens against the seed (DeepSeek, Table 1's test runs).

compute  -> <cache>/analysis/main/attribution.json: per cell (A) the split on 3-run task means, (B) per run pair,
            (C) the per-request change of each prefix column, and the Table 1 rho on the same tasks
attr     -> <cache>/analysis/appc/attr.json: the split on one run per side (the https reruns), read by skilltext
"""
import json
import math
import sys

import numpy as np

from src.analysis.common import mean_rows, rep_files, rows_of
from src.analysis.episodes import episodes, metrics
from src.graph import episode as W
from src.utils import dataloader as C

MODEL = "deepseek-flash"
OPT = [("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"), ("GEPA", "gepa")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
PREFIX = list(W.FIXED)
DROPPED = ["no verified outcome / N, X in all six runs", "no graph episode in some run",
           "episode without usable metrics (R<2, no call, no history)", "no positive history on the 3-run mean"]
# attr: the seed's first run and the https reruns of the optimizers
SEED_RUN, ATTR_RUN = "native-20261003-trace-final", "native-20261008-https"
ATTR_NAMES = dict(more_actions="more_actions", turn_splitting="more_requests", content="longer_content", skill_text="longer_skill")


def run_files(method, bench):
    """({rep: evaluation.json} of the skill, the same for the seed), as Table 1 picks them."""
    seed_run = C.current_run("native-seed", MODEL, bench)
    seed = rep_files([seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"], C.vid(C.SEED_TEXT))
    run = C.current_run(method, MODEL, bench)
    own = rep_files([run, C.EX / MODEL / "test-reps" / f"{bench}__{method}__{run.name}"], C.vid(C.final_skill(method, run)))
    return own, seed


def load_runs(files):
    """{rep: {task: (Episode, metrics)}} and {rep: {task: Table 1 row}}."""
    eps = {rep: {e.task: (e, metrics(e)) for e in W.load(path)} for rep, path in sorted(files.items())}
    rows = {rep: {t: r for (t, _), r in rows_of(path).items()} for rep, path in sorted(files.items())}
    return eps, rows


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
    sH = 1 - sF
    Rm = np.array([(a["R"] + b["R"]) / 2 for a, b in pairs])
    eR = 1 + sH * Rm / (Rm - 1)
    return dict(more_actions=eR * dN, turn_splitting=-eR * (dN - dR), content=sH * dh, skill_text=sF * dF), dX


def summarize(terms, dX):
    """Task means of the terms and of the total, the residual, and the largest term."""
    out = {k: float(v.mean()) for k, v in terms.items()}
    out["total"] = float(dX.mean())
    resid = dX - sum(terms.values())
    out["residual_mean"] = float(resid.mean())
    out["residual_max_abs"] = float(np.abs(resid).max())
    out["largest"] = max(TERMS, key=lambda k: abs(out[k]))
    return out


def prefix_columns(own_eps, seed_eps, tasks):
    """Per prefix column, the mean over tasks of the per-request tokens (3-run mean), skill minus seed, and the seed's level."""
    def level(eps, task, col):
        return float(np.mean([e.Xrg[:, e.comps.index(col)].mean() if col in e.comps else 0.0
                              for e in (eps[rep][task][0] for rep in eps)]))
    return {col: dict(delta=float(np.mean([level(own_eps, t, col) - level(seed_eps, t, col) for t in tasks])),
                      seed=float(np.mean([level(seed_eps, t, col) for t in tasks]))) for col in PREFIX}


def usable_tasks(tasks, own, seed, reps):
    """The tasks with outcome, N, X, a usable episode in all six runs and positive 3-run history; drop counts by reason."""
    (own_eps, own_rows), (seed_eps, seed_rows) = own, seed
    dropped, per_side = dict.fromkeys(DROPPED, 0), {}
    for t in tasks:
        if not all(rows[r].get(t) and None not in (rows[r][t]["success"], rows[r][t]["N"], rows[r][t]["X"])
                   for rows in (own_rows, seed_rows) for r in reps):
            dropped[DROPPED[0]] += 1
        elif not all(t in eps[r] for eps in (own_eps, seed_eps) for r in reps):
            dropped[DROPPED[1]] += 1
        elif not all(eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
            dropped[DROPPED[2]] += 1
        elif None in (a := mean_metrics([seed_eps[r][t][1] for r in reps]), b := mean_metrics([own_eps[r][t][1] for r in reps])):
            dropped[DROPPED[3]] += 1
        else:
            per_side[t] = (a, b)
    return per_side, dropped


def cell(disp, method, bench, short):
    own_files, seed_files = run_files(method, bench)
    own, seed = load_runs(own_files), load_runs(seed_files)
    reps = sorted(own_files)
    assert reps == [0, 1, 2] and sorted(seed_files) == [0, 1, 2], (method, bench, reps)
    own_mean, seed_mean = mean_rows(own_files)[1], mean_rows(seed_files)[1]

    def log_rho(t):
        return math.log(own_mean[(t, 0)]["X"] / seed_mean[(t, 0)]["X"])
    t1_tasks = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean
                and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
    t1_log_rho = float(np.mean([log_rho(t) for t in t1_tasks]))
    per_side, dropped = usable_tasks(t1_tasks, own, seed, reps)
    usable = list(per_side)
    A = summarize(*four_terms(list(per_side.values())))
    B = [dict(run=r, **summarize(*four_terms([(seed[0][r][t][1], own[0][r][t][1]) for t in usable]))) for r in reps]
    log_rho_usable = float(np.mean([log_rho(t) for t in usable]))
    print(disp, bench, "n", len(usable), "of", len(t1_tasks), {k: round(v, 3) for k, v in A.items() if k != "largest"}, A["largest"])
    return dict(method=disp, bench=bench, short=short, run=C.current_run(method, MODEL, bench).name, n=len(usable),
                n_table1=len(t1_tasks), dropped=dropped, A=A, B=B, same_largest_all_runs=len({b["largest"] for b in B} | {A["largest"]}) == 1,
                table1_log_rho_usable=log_rho_usable, table1_log_rho_full=t1_log_rho, table1_rho_full=math.exp(t1_log_rho),
                total_gap_to_table1=A["total"] - log_rho_usable, C=prefix_columns(own[0], seed[0], usable))


def compute():
    C.write_json("attribution", dict(model=MODEL, cells=[cell(d, m, b, s) for d, m in OPT for b, s in BENCH]))


def attr():
    """The four-term split on one run per side (seed's first run against the https rerun), per skill."""
    skills = []
    seed_text = C.vid(C.SEED_TEXT)
    for disp, method in OPT:
        for bench, _ in BENCH:
            seed_run = C.EX / MODEL / "native-seed" / bench / SEED_RUN
            run = C.EX / MODEL / method / bench / ATTR_RUN
            version = C.vid(C.final_skill(method, run))
            seed_rows, own_rows = C.evaluation(seed_run, "test", seed_text)[0], C.evaluation(run, "test", version)[0]
            common = {t for (t, _), r in own_rows.items() if (s := seed_rows.get((t, 0))) and None not in (r["success"], s["success"])
                      and all(x and x > 0 for x in (r["N"], r["X"], s["N"], s["X"]))}
            S, O = episodes(C.eval_file(seed_run, "test", seed_text)), episodes(C.eval_file(run, "test", version))
            pairs = [(a, b) for t, e in O.items() if t in S and (a := metrics(S[t])) and (b := metrics(e)) and t in common]
            terms, dX = four_terms(pairs)
            seg = {ATTR_NAMES[k]: float(v.mean()) for k, v in terms.items()}
            skills.append(dict(model=MODEL, bench=bench, method=disp, n=len(pairs), run=run.name, **seg,
                               segments_sum=sum(seg.values()), dlogX=float(dX.mean()), ratio=C.gmean(np.exp(dX))))
    (C.out_dir("appc") / "attr.json").write_text(json.dumps(dict(skills=skills), indent=1))


if __name__ == "__main__":
    {"compute": compute, "attr": attr}[sys.argv[1]]()
