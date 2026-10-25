#!/usr/bin/env python
"""Paths of score, cost and content along every optimization run (GEPA, SkillOpt, SkillAdam; DeepSeek-V4.1-Flash and GPT-6 Luna).

Reads only OUT/skill_paths/paths/*.json (written by extract_{gepa,skillopt,skilladam}.py) and inputs/seed_inputs.json (written by
prep_inputs.py: the seed's per-task dev executions that the paper's selection replay uses), plus three read-only caches of the paper for checks
(assets/cache/analysis/main/selection.json, selection_luna.json, skilltext.json) and the paper's code skilltext.py for the keyword pass (copied verbatim
below, checked against it).  Writes only under OUT/skill_paths/: analysis/paths_analysis.json, fig/*.pdf (+ png for inspection), mech_paths.md.

Regenerate (idempotent, ~5-10 min, seeded):  python -m aux.skill_paths.analyze_paths [--draws N] [--no-figs]

Questions (tracking the optimization path):
  Q1 when do gains arrive (dev success of every scored candidate, best-so-far, share of the final gain at the first scored candidate, candidates scored
     after the best; against the paper's null: the optimizer's own rule replayed on exact copies of the seed, selection.py);
  Q2 cost along the path (mean input tokens per dev episode of every scored candidate relative to the seed/incumbent; kept vs rejected; relation to dev
     success; whether cost keeps rising after dev stops improving);
  Q3 content along the path (every added unit coded into the paper's eight categories by the paper's keyword pass; cumulative composition by step);
  Q4 skill length against dev score.

What each optimizer's dev number is (read from the optimizers' code and the extractors' notes, not guessed):
  SkillOpt  every candidate is scored once on the WHOLE dev panel; accepted iff its score is strictly above the stored score of the current skill
            (current = best, so best-so-far is the running maximum).  Returned = first candidate reaching the maximum, else the seed.
  GEPA      a child is first compared with its parent on an 8-task TRAIN minibatch (strict sum improvement); only accepted children are scored on the
            whole dev panel and ALL of them enter the pool (no dev gate); children rejected on the minibatch are never scored on dev (dev = null).
            Returned = argmax of the dev mean over the pool, ties to the lowest index (the seed first).  K = number of pool entrants (the paper's K).
  SkillAdam NO whole-panel dev score exists.  Each iteration rolls the incumbent and the single candidate out on one fresh random 10-task (TB2: 9) draw of
            the dev panel; accepted iff candidate - incumbent > 0 on that draw.  Its path quantity is the paired difference per iteration and the cumulative
            announced gain (sum of the accepted differences, points of the draw) = the paper's "reported gain".
"""
import sys

sys.dont_write_bytecode = True

import argparse
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from harness_bench.eval import version_from_files

from src import layout

SC = layout.OUT / "skill_paths"
PATHS = SC / "paths"
INPUTS = SC / "inputs/seed_inputs.json"
FIG = SC / "fig"
OUT_JSON = SC / "analysis/paths_analysis.json"
OUT_MD = SC / "mech_paths.md"
CACHE = layout.CACHE / "analysis/main"
SEED_HAND = layout.SEED_FILE
SEED_MIN = layout.ASSETS / "skills/seed-minimal/SKILL.md"

SEED = 20261023
MODERATE = 4.0     # concentration of the moderate prior (selection.py MODERATE)
MODELS = ["deepseek-flash", "gpt-6-luna"]
MODEL_NAME = {"deepseek-flash": "DeepSeek", "gpt-6-luna": "Luna"}
OPTS = ["gepa", "skillopt", "skilladam"]
OPT_NAME = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
BENCHES = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
BENCH_NAME = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
OC = {"gepa": "#009E73", "skillopt": "#0072B2", "skilladam": "#D55E00"}
MK = {"gepa": "^", "skillopt": "o", "skilladam": "s"}


# ------------------------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------------------------------------------------------
def vid(text):
    """harness-bench version id of a one-file skill."""
    return version_from_files({"SKILL.md": text})["version_id"]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f2(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def rank(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rank(x), rank(y))


def sign_test_p(k, n):
    """Exact two-sided binomial sign test p-value (H0: p = 1/2) for k successes of n non-tied comparisons."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def disp_run(r):
    tag = {"hand-written": "", "near-empty": " (near-empty seed)"}[r["seed"]]
    return f"{MODEL_NAME[r['model']]} {BENCH_NAME[r['bench']]}{tag}"


# ------------------------------------------------------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------------------------------------------------------
def load_runs():
    runs = []
    for f in sorted(PATHS.glob("*.json")):
        d = json.load(open(f))
        runs.append({"key": f.stem, "d": d, "model": d["model"], "opt": d["optimizer"], "bench": d["benchmark"], "run_id": d["run_id"],
                     "seed": d["seed"], "partial": d["partial"], "steps": d["steps"]})
    order = lambda r: (MODELS.index(r["model"]), 0 if r["seed"] == "hand-written" else 1, OPTS.index(r["opt"]), BENCHES.index(r["bench"]))
    runs.sort(key=order)
    for i, r in enumerate(runs):
        r["idx"] = i
    return runs


def text_of(r):
    return {s["candidate_id"]: s["skill_text"] for s in r["steps"]}


# ------------------------------------------------------------------------------------------------------------------------------------------------
# score paths
# ------------------------------------------------------------------------------------------------------------------------------------------------
def panel_path(r):
    """SkillOpt / GEPA: every dev-scored candidate in the order produced, on the whole dev panel, against the run's seed draw b."""
    d = r["d"]
    n, b, tok0 = d["dev_panel_size"], d["seed_dev"]["n_success"], d["seed_dev"]["mean_input_tokens_per_episode"]
    assert d["seed_dev"]["n"] == n, (r["key"], "seed dev is not the whole panel")
    rows, run_max, best_k, prev_kept_max = [], b, 0, b
    for s in r["steps"][1:]:
        if not s["dev"]:
            continue
        assert s["dev"]["n"] == n, (r["key"], s["step"])
        k = len(rows) + 1
        S = s["dev"]["n_success"]
        improved = S > run_max          # strict: ties go to the earlier candidate and to the seed (GEPA: lowest index; SkillOpt: strict >)
        if improved:
            run_max, best_k = S, k
        rows.append({"k": k, "step": s["step"], "id": s["candidate_id"], "parent": s["parent_id"], "kept": s["kept"], "S": S, "unk": s["dev"]["n_unknown"],
                     "gain": S - b, "G": max(0, run_max - b), "improved": improved, "words": s["skill_words"], "tokens": s["dev"]["mean_input_tokens_per_episode"],
                     "rho": s["dev"]["mean_input_tokens_per_episode"] / tok0, "kept_rule_ok": None})
    if r["opt"] == "skillopt":   # SkillOpt's own rule: kept iff strictly above the current (= running max) score
        cur = b
        for row in rows:
            row["kept_rule_ok"] = bool(row["kept"]) == (row["S"] > cur)
            cur = max(cur, row["S"]) if row["kept"] else cur
    best_id = rows[best_k - 1]["id"] if best_k else d["steps"][0]["candidate_id"]
    return {"n": n, "b": b, "tok0": tok0, "rows": rows, "K": len(rows), "k_best": best_k, "G_final": max(0, run_max - b), "best_id": best_id,
            "step_best": rows[best_k - 1]["step"] if best_k else 0}


def sa_path(r):
    """SkillAdam: one entry per iteration; paired candidate-minus-incumbent difference on that iteration's fresh validation draw."""
    d = r["d"]
    rows, cum = [], 0.0
    for i, s in enumerate(r["steps"][1:], start=1):
        g, ex = s["gate"]["detail"], s["extra"]
        n_draw = g["baseline"]["n"]
        live = bool(ex["candidate_exists"]) and not bool(ex["identical_to_parent"])
        gain_t = g["gain_tasks"]
        before = cum
        if s["kept"]:
            cum += 100.0 * gain_t / n_draw
        cand_tok = s["dev"]["mean_input_tokens_per_episode"] if s["dev"] else None
        base_tok = g["baseline"]["dev"]["mean_input_tokens_per_episode"]
        rows.append({"k": i, "step": s["step"], "id": s["candidate_id"], "parent": s["parent_id"], "kept": s["kept"], "live": live, "mode": g["patch_mode"],
                     "n_draw": n_draw, "base": g["baseline"]["n_success"], "cand": g["candidate"]["n_success"], "gain_t": gain_t,
                     "gain_pts": 100.0 * gain_t / n_draw, "cum_before": before, "cum": cum, "words": s["skill_words"], "edit_budget": g.get("edit_budget"),
                     "tok_cand": cand_tok, "tok_base": base_tok, "case_ids": g["validation_case_ids"], "scored": s["dev"] is not None,
                     "rho": (cand_tok / base_tok) if (cand_tok and base_tok) else None})
    kept_steps = [x["k"] for x in rows if x["kept"]]
    return {"T": len(rows), "rows": rows, "G_final_pts": cum, "accepted": len(kept_steps), "k_first_accept": kept_steps[0] if kept_steps else 0,
            "k_last_accept": kept_steps[-1] if kept_steps else 0, "n_live": sum(x["live"] for x in rows), "n_scored": sum(x["scored"] for x in rows)}


# ------------------------------------------------------------------------------------------------------------------------------------------------
# the paper's null: the optimizer's own rule replayed on exact copies of the seed (selection.py), as a path
# ------------------------------------------------------------------------------------------------------------------------------------------------
def seed_model(r, inputs):
    """(dev {task: [0/1 ...]}, {prior name: (a, b)}, primary prior name, description)."""
    if r["seed"] == "hand-written":
        inp = inputs[r["model"]][r["bench"]]
        a, b = inp["prior_fitted"]
        m = inp["mean_success"]
        return inp["dev"], {"fitted": (a, b), "moderate": (MODERATE * m, MODERATE * (1 - m))}, "fitted", f"fitted (a={a:.2f}, b={b:.2f}); moderate m={m:.3f}"
    ne = inputs["gpt-6-luna-nearempty"]["automationbench"]
    m, md = ne["pooled_mean_success_dev_and_test"], ne["dev_mean_success"]
    plug = {t: sum(v) / len(v) for t, v in ne["all_dev"].items()}
    return (ne["skillopt"]["dev"], {"moderate": (MODERATE * m, MODERATE * (1 - m)), "moderate_devm": (MODERATE * md, MODERATE * (1 - md)),
                                    "null_plugin": ("plugin", plug)}, "moderate",
            f"ASSUMED moderate prior Beta(4m,4(1-m)); m={m:.3f} (seed dev+test pooled, 150 executions) / {md:.3f} (seed dev only); null plug-in: per-task rate = "
            f"share of successes over all {ne['all_dev_n_files']} dev evaluations of the two fromscratch runs (22-23 executions per task)")


def task_prob_draws(dev, prior, draws, rng):
    tasks = sorted(dev)
    a, b = prior
    pa = np.array([a + sum(dev[t]) for t in tasks])
    pb = np.array([b + len(dev[t]) - sum(dev[t]) for t in tasks])
    return rng.beta(pa, pb, size=(draws, len(tasks)))


def sim_best_so_far(p, K, b, rng, chunk=20000):
    """(draws, K) int16: G_k = max(0, max_{j<=k} S_j - b) for K candidates identical to the seed (fresh outcomes), b fixed at the run's seed draw."""
    D, T = p.shape
    out = np.empty((D, K), dtype=np.int16)
    for s in range(0, D, chunk):
        pc = p[s:s + chunk]
        sc = (rng.random((pc.shape[0], K, T), dtype=np.float32) < pc[:, None, :]).sum(axis=2)
        out[s:s + chunk] = np.maximum.accumulate(sc, axis=1)
    return np.maximum(out - b, 0)


def sim_skilladam(p_dev, batch, budgets, rng):
    """Port of selection.skilladam_reported_gain returning the cumulative reported gain (points) after every iteration: per iteration a random
    subset of `batch` dev tasks; the current skill's outcomes are cached per task (drawn the first time the task is used); `budget` candidate
    patches with fresh outcomes (budget 0 = no live candidate: nothing can be accepted); the first strictly better candidate is accepted and
    becomes the current skill with its own outcomes on that subset (every other task unrun).  Returns ((draws, T) float32, accepts per draw)."""
    D, n_dev = p_dev.shape
    rows = np.arange(D)[:, None]
    current = np.full((D, n_dev), -1, dtype=np.int8)
    total = np.zeros(D)
    accepts = np.zeros(D, dtype=int)
    traj = np.zeros((D, len(budgets)), dtype=np.float32)
    for t, budget in enumerate(budgets):
        if budget > 0:
            pick = np.argpartition(rng.random((D, n_dev)), batch - 1, axis=1)[:, :batch]
            p = p_dev[rows, pick]
            seen = current[rows, pick]
            fresh = (rng.random(p.shape) < p).astype(np.int8)
            seen = np.where(seen < 0, fresh, seen)
            current[rows, pick] = seen
            base = seen.sum(axis=1)
            patches = (rng.random((D, budget, batch)) < p[:, None, :]).astype(np.int8)
            better = patches.sum(axis=2) > base[:, None]
            accepted = better.any(axis=1)
            first = better.argmax(axis=1)
            winner = patches[np.arange(D), first]
            total += np.where(accepted, 100 * (winner.sum(axis=1) - base) / batch, 0)
            accepts += accepted
            current[accepted] = -1
            current[rows[accepted], pick[accepted]] = winner[accepted]
        traj[:, t] = total
    return traj, accepts


def qdisc(a, q):
    """Discrete quantile: the smallest value v with P(X <= v) >= q."""
    return np.quantile(a, q, axis=0, method="inverted_cdf")


def band(G, to_pts):
    """Null band of a (draws, K) array: mean, 5/50/95% quantiles (in points)."""
    return {"mean": (G.mean(axis=0) * to_pts).tolist(), "q05": (qdisc(G, 0.05) * to_pts).tolist(), "q50": (qdisc(G, 0.5) * to_pts).tolist(),
            "q95": (qdisc(G, 0.95) * to_pts).tolist()}


def p_ge(a, obs):
    return float((a >= obs - 1e-9).mean())


def share_stats(G, obs_G):
    """Null reference for 'the share of the final gain reached by the first scored candidate': among draws with G_K > 0, mean of G_1/G_K and the
    probability that the first maximum is already reached at k = 1 (G_1 = G_K); and the observed value."""
    GK, G1 = G[:, -1], G[:, 0]
    pos = GK > 0
    return {"p_final_positive": float(pos.mean()), "mean_share1_given_positive": float((G1[pos] / GK[pos]).mean()) if pos.any() else float("nan"),
            "p_first_is_best_given_positive": float((G1[pos] == GK[pos]).mean()) if pos.any() else float("nan")}


def run_nulls(runs, inputs, draws):
    """Replay for every run.  Returns {run key: {...}} with bands (points), P-values and the raw final arrays for pooling (kept in `_arrays`)."""
    sel = json.load(open(CACHE / "selection.json"))
    sel_luna = json.load(open(CACHE / "selection_luna.json"))
    out, arrays = {}, {}
    for r in runs:
        dev, priors, primary, desc = seed_model(r, inputs)
        res = {"model_desc": desc, "primary": primary, "priors": {}}
        for j, (pname, prior) in enumerate(priors.items()):
            rng = np.random.default_rng([SEED, r["idx"], j])
            if prior[0] == "plugin":     # the paper's prior-free Luna model: fixed per-task rates, no posterior draw
                tasks = sorted(prior[1])
                p = np.tile(np.array([prior[1][t] for t in tasks]), (draws, 1))
            else:
                p = task_prob_draws(dev, prior, draws, rng)
            if r["opt"] in ("skillopt", "gepa"):
                P = r["path"]
                G = sim_best_so_far(p, P["K"], P["b"], rng)
                to_pts = 100.0 / P["n"]
                obs_G = [x["G"] for x in P["rows"]]
                res["priors"][pname] = {"band_pts": band(G, to_pts), "prior": (list(prior) if prior[0] != "plugin" else "plug-in"), "E_G_final_tasks": float(G[:, -1].mean()),
                                        "P_final_ge_obs": p_ge(G[:, -1], P["G_final"]), "P_final_ge_max_obs_1": p_ge(G[:, -1], max(P["G_final"], 1)),
                                        "pointwise_P_ge_obs": [p_ge(G[:, k], obs_G[k]) for k in range(P["K"])],
                                        "pointwise_P_ge_max_obs_1": [p_ge(G[:, k], max(obs_G[k], 1)) for k in range(P["K"])],
                                        "share": share_stats(G, P["G_final"]), "to_pts": to_pts}
                arrays[(r["key"], pname)] = (G.astype(np.int8), to_pts)
            else:
                P = r["path"]
                batch = P["rows"][0]["n_draw"]
                live_budgets = [1 if x["live"] else 0 for x in P["rows"]]
                ref = (sel[r["bench"]]["observed"]["SkillAdam"] if r["model"] == "deepseek-flash" else sel_luna[r["bench"]]["observed"]["SkillAdam"])
                paper_budgets = ref["budgets"]
                assert len(paper_budgets) == P["T"] and ref["batch"] == batch, (r["key"], len(paper_budgets), P["T"], ref["batch"], batch)
                variants = {}
                for vname, budgets in (("one_live_candidate", live_budgets), ("paper_replay", paper_budgets)):
                    traj, acc = sim_skilladam(p, batch, budgets, np.random.default_rng([SEED, r["idx"], j, 7, len(vname)]))
                    obs = P["G_final_pts"]
                    variants[vname] = {"band_pts": {"mean": traj.mean(axis=0).tolist(), "q05": qdisc(traj, 0.05).tolist(), "q50": qdisc(traj, 0.5).tolist(),
                                                    "q95": qdisc(traj, 0.95).tolist()}, "E_final_pts": float(traj[:, -1].mean()), "E_accepts": float(acc.mean()),
                                       "P_final_ge_obs": p_ge(traj[:, -1], obs) if obs > 0 else None, "P_final_gt_0": float((traj[:, -1] > 0).mean()),
                                       "pointwise_P_ge_obs": [p_ge(traj[:, k], x["cum"]) if x["cum"] > 0 else None for k, x in enumerate(P["rows"])],
                                       "budgets": budgets}
                    arrays[(r["key"], pname, vname)] = (traj, 1.0)
                res["priors"][pname] = {"prior": (list(prior) if prior[0] != "plugin" else "plug-in"), "variants": variants}
        out[r["key"]] = res
    return out, arrays


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Q1: when do gains arrive
# ------------------------------------------------------------------------------------------------------------------------------------------------
def q1_run(r, null):
    P = r["path"]
    pr = null["priors"][null["primary"]]
    if r["opt"] in ("skillopt", "gepa"):
        rows = P["rows"]
        first_imp = next((x["k"] for x in rows if x["improved"]), 0)
        gf = P["G_final"]
        return {"kind": "panel", "n_dev": P["n"], "seed_dev": P["b"], "K": P["K"], "scores": [x["S"] for x in rows], "steps": [x["step"] for x in rows],
                "kept": [x["kept"] for x in rows], "G_final_tasks": gf, "G_final_pts": 100.0 * gf / P["n"], "k_best": P["k_best"], "step_best": P["step_best"],
                "id_best": P["best_id"], "k_first_improvement": first_imp, "share_at_first_scored": (rows[0]["G"] / gf if gf > 0 else None),
                "n_after_best": P["K"] - P["k_best"], "best_so_far_tasks": [x["G"] for x in rows],
                "P_final_primary": pr["P_final_ge_max_obs_1"], "P_final_primary_note": ("P(G >= obs)" if gf > 0 else "obs gain 0: P(G >= 1) = chance selection alone keeps one"),
                "P_final_by_prior": {k: v["P_final_ge_max_obs_1"] for k, v in null["priors"].items()},
                "E_G_final_by_prior_tasks": {k: v["E_G_final_tasks"] for k, v in null["priors"].items()},
                "share_null": pr["share"], "pointwise_P": pr["pointwise_P_ge_max_obs_1"],
                "n_pointwise_above_q95": int(sum(1 for k, x in enumerate(rows) if x["G"] > pr["band_pts"]["q95"][k] * P["n"] / 100.0 + 1e-9)),
                "n_unknown_dev": sum(x["unk"] for x in rows)}
    rows = P["rows"]
    v1 = pr["variants"]["one_live_candidate"]
    v2 = pr["variants"]["paper_replay"]
    obs = P["G_final_pts"]
    return {"kind": "draws", "T": P["T"], "n_live": P["n_live"], "n_scored": P["n_scored"], "accepted": P["accepted"], "k_first_accept": P["k_first_accept"],
            "k_last_accept": P["k_last_accept"], "G_final_pts": obs, "accepted_gains_tasks": [x["gain_t"] for x in rows if x["kept"]],
            "n_after_last_accept": P["T"] - P["k_last_accept"], "cum_pts": [x["cum"] for x in rows], "gain_pts": [x["gain_pts"] for x in rows],
            "kept": [x["kept"] for x in rows], "live": [x["live"] for x in rows],
            "P_final_one_live": v1["P_final_ge_obs"] if obs > 0 else v1["P_final_gt_0"], "P_final_paper_replay": v2["P_final_ge_obs"] if obs > 0 else v2["P_final_gt_0"],
            "P_note": "P(cumulative gain >= observed)" if obs > 0 else "observed 0: P(gain > 0)",
            "P_by_prior": {k: {"one_live": (v["variants"]["one_live_candidate"]["P_final_ge_obs"] if obs > 0 else v["variants"]["one_live_candidate"]["P_final_gt_0"]),
                               "paper_replay": (v["variants"]["paper_replay"]["P_final_ge_obs"] if obs > 0 else v["variants"]["paper_replay"]["P_final_gt_0"])}
                           for k, v in null["priors"].items()},
            "E_final_one_live": v1["E_final_pts"], "E_final_paper_replay": v2["E_final_pts"], "E_accepts_one_live": v1["E_accepts"],
            "n_above_q95_one_live": int(sum(1 for k, x in enumerate(rows) if x["cum"] > v1["band_pts"]["q95"][k] + 1e-3))}


def q1_first_stats(runs, q1):
    """Among runs with a positive final dev gain: how many already have their best at the first scored candidate, against the replay's expectation
    (sum over those runs of P(first scored candidate is already the final best | final gain > 0)); mean share of the final gain at the first scored
    candidate against the replay's mean share (conditional on a positive final gain)."""
    out = {}
    for opt in ("skillopt", "gepa"):
        for gname, cond in (("hand-written", lambda r: r["seed"] == "hand-written"), ("near-empty", lambda r: r["seed"] == "near-empty")):
            rs = [r for r in runs if r["opt"] == opt and cond(r)]
            pos = [r for r in rs if q1[r["key"]]["G_final_tasks"] > 0]
            if not rs:
                continue
            out[f"{opt}|{gname}"] = {"n_runs": len(rs), "n_positive": len(pos),
                                     "obs_first_is_best": int(sum(1 for r in pos if q1[r["key"]]["share_at_first_scored"] == 1.0)),
                                     "exp_first_is_best": float(sum(q1[r["key"]]["share_null"]["p_first_is_best_given_positive"] for r in pos)),
                                     "obs_mean_share1": float(np.mean([q1[r["key"]]["share_at_first_scored"] for r in pos])) if pos else None,
                                     "null_mean_share1": float(np.mean([q1[r["key"]]["share_null"]["mean_share1_given_positive"] for r in pos])) if pos else None,
                                     "median_step_best": float(np.median([q1[r["key"]]["step_best"] for r in pos])) if pos else None,
                                     "median_k_best_over_K": float(np.median([q1[r["key"]]["k_best"] / q1[r["key"]]["K"] for r in pos])) if pos else None,
                                     "null_median_k_best_over_K": 0.5}
    return out


def pooled_stats(runs, arrays, nulls):
    """Pooled over the hand-written-seed runs of one optimizer (group = model or both): the mean final gain in points against the replay's distribution
    of the same mean (independent draws per run), and the pooled mean best-so-far curve with its null band."""
    out = {}
    hand = [r for r in runs if r["seed"] == "hand-written"]
    groups = {"DeepSeek": [r for r in hand if r["model"] == "deepseek-flash"], "Luna": [r for r in hand if r["model"] == "gpt-6-luna"], "both": hand}
    for opt in OPTS:
        for gname, members in groups.items():
            rs = [r for r in members if r["opt"] == opt]
            for pname in ("fitted", "moderate"):
                if opt == "skilladam":
                    for vname in ("one_live_candidate", "paper_replay"):
                        arr = [arrays[(r["key"], pname, vname)][0][:, -1] for r in rs]
                        obs = float(np.mean([r["path"]["G_final_pts"] for r in rs]))
                        null = np.mean(arr, axis=0)
                        out[f"{opt}|{gname}|{pname}|{vname}"] = {"n_runs": len(rs), "obs_mean_pts": obs, "null_mean_pts": float(null.mean()), "q05": float(qdisc(null, 0.05)),
                                                                  "q95": float(qdisc(null, 0.95)), "P_ge_obs": p_ge(null, obs)}
                else:
                    arr = [arrays[(r["key"], pname)][0][:, -1] * arrays[(r["key"], pname)][1] for r in rs]
                    obs = float(np.mean([100.0 * r["path"]["G_final"] / r["path"]["n"] for r in rs]))
                    null = np.mean(arr, axis=0)
                    out[f"{opt}|{gname}|{pname}"] = {"n_runs": len(rs), "obs_mean_pts": obs, "null_mean_pts": float(null.mean()), "q05": float(qdisc(null, 0.05)),
                                                      "q95": float(qdisc(null, 0.95)), "P_ge_obs": p_ge(null, obs)}
    # pooled curves (fitted prior), runs that still have a k-th scored candidate; shown while at least two runs contribute
    curves = {}
    for opt in OPTS:
        for gname in ("DeepSeek", "Luna"):
            rs = [r for r in groups[gname] if r["opt"] == opt]
            if opt == "skilladam":
                lens = [r["path"]["T"] for r in rs]
                obs_curve = lambda r: [x["cum"] for x in r["path"]["rows"]]
                arr_of = lambda r: arrays[(r["key"], "fitted", "one_live_candidate")][0]
            else:
                lens = [r["path"]["K"] for r in rs]
                obs_curve = lambda r: [100.0 * x["G"] / r["path"]["n"] for x in r["path"]["rows"]]
                arr_of = lambda r: arrays[(r["key"], "fitted")][0] * arrays[(r["key"], "fitted")][1]
            pts = []
            for k in range(1, max(lens) + 1):
                m = [r for r, L in zip(rs, lens) if L >= k]
                if len(m) < 2:
                    break
                nullk = np.mean([arr_of(r)[:, k - 1] for r in m], axis=0)
                pts.append({"k": k, "n_runs": len(m), "obs_mean": float(np.mean([obs_curve(r)[k - 1] for r in m])), "null_mean": float(nullk.mean()),
                            "q05": float(qdisc(nullk, 0.05)), "q95": float(qdisc(nullk, 0.95))})
            curves[f"{opt}|{gname}"] = pts
    out["curves"] = curves
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Q2: cost along the path
# ------------------------------------------------------------------------------------------------------------------------------------------------
def seed_task_tokens(inputs, model, bench):
    """Seed's mean input tokens per dev episode, per task (mean over its distinct executions), for the task-matched ratio of SkillAdam."""
    inp = inputs[model][bench]["dev_tokens"]
    return {t: float(np.mean(v)) for t, v in inp.items() if v}


def cost_rows(r, inputs):
    P = r["path"]
    out = []
    if r["opt"] in ("skillopt", "gepa"):
        for x in P["rows"]:
            out.append({"k": x["k"], "step": x["step"], "id": x["id"], "kept": bool(x["kept"]), "rho": x["rho"], "dsucc_pts": 100.0 * x["gain"] / P["n"],
                        "words": x["words"], "after_best": x["k"] > P["k_best"], "at_or_before_best": x["k"] <= P["k_best"] and P["k_best"] > 0})
        return out
    seed_tok = seed_task_tokens(inputs, r["model"], r["bench"]) if r["seed"] == "hand-written" else None
    for x in P["rows"]:
        if not x["scored"]:
            continue
        matched = inc_matched = None
        if seed_tok is not None:
            ids = [t for t in x["case_ids"] if t in seed_tok]
            if len(ids) == len(x["case_ids"]):
                ref = float(np.mean([seed_tok[t] for t in ids]))
                matched, inc_matched = x["tok_cand"] / ref, x["tok_base"] / ref
        out.append({"k": x["k"], "step": x["step"], "id": x["id"], "kept": bool(x["kept"]), "rho": x["rho"], "dsucc_pts": x["gain_pts"], "live": x["live"],
                    "rho_seed_matched": matched, "rho_incumbent_seed_matched": inc_matched, "words": x["words"], "after_best": x["k"] > P["k_last_accept"],
                    "at_or_before_best": x["k"] <= P["k_last_accept"] and P["k_last_accept"] > 0})
    return out


def gm(v):
    v = [x for x in v if x is not None and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else None


def boot_diff(a, b, rng, nb=10000):
    """Bootstrap 95% interval of mean(a) - mean(b) (resampling a and b separately)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(nb)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def within_run_perm_corr(groups, rng, nperm=10000):
    """Pooled within-run Spearman-type correlation between log rho and dev change, with a within-run permutation p (exploratory; candidates of a run are
    not independent).  groups = list of (x, y) arrays, one per run."""
    def stat(gs):
        xs, ys = [], []
        for x, y in gs:
            if len(x) >= 2:
                rx, ry = rank(x), rank(y)
                xs.append(rx - rx.mean())
                ys.append(ry - ry.mean())
        if not xs:
            return float("nan")
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        return pearson(xs, ys)
    obs = stat(groups)
    if math.isnan(obs):
        return obs, float("nan"), 0
    cnt = 0
    for _ in range(nperm):
        perm = [(x, rng.permutation(y)) for x, y in groups]
        if abs(stat(perm)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (nperm + 1), sum(len(x) for x, _ in groups)


def q2_stats(runs, inputs):
    rng = np.random.default_rng([SEED, 99])
    per_run, allrows = {}, {}
    for r in runs:
        rows = cost_rows(r, inputs)
        allrows[r["key"]] = rows
        P = r["path"]
        if r["opt"] == "skilladam":
            ref = [x["rho"] for x in rows]
            before = [x["rho"] for x in rows if x["at_or_before_best"]]
            after = [x["rho"] for x in rows if x["after_best"]]
            best_rho = None
        else:
            before = [x["rho"] for x in rows if x["at_or_before_best"]]
            after = [x["rho"] for x in rows if x["after_best"]]
            best_rho = rows[P["k_best"] - 1]["rho"] if P["k_best"] else None
        kept = [x["rho"] for x in rows if x["kept"]]
        rej = [x["rho"] for x in rows if not x["kept"]]
        ks = np.array([x["k"] for x in rows], dtype=float)
        lr = np.log([x["rho"] for x in rows])
        slope = float(np.polyfit(ks, lr, 1)[0]) if len(rows) >= 3 else None
        rest = [x["rho"] for x in rows[1:]]
        per_run[r["key"]] = {"rho_first": rows[0]["rho"], "gm_rest": gm(rest), "rest_over_first": (gm(rest) / rows[0]["rho"] if rest else None), "slope_log_rho_per_candidate": slope,"n_scored": len(rows), "tok_ref": (P["tok0"] if r["opt"] != "skilladam" else None), "rho_gm": gm([x["rho"] for x in rows]),
                             "rho_min": min(x["rho"] for x in rows), "rho_max": max(x["rho"] for x in rows), "rho_gm_kept": gm(kept), "rho_gm_rejected": gm(rej),
                             "n_kept": len(kept), "n_rejected": len(rej), "best_rho": best_rho, "gm_before_or_at_best": gm(before), "gm_after_best": gm(after),
                             "n_before_or_at_best": len(before), "n_after_best": len(after),
                             "after_over_best": (gm(after) / best_rho if (best_rho and after) else None),
                             "after_over_before": (gm(after) / gm(before) if (before and after) else None),
                             "rho_seed_matched_gm": gm([x.get("rho_seed_matched") for x in rows]) if r["opt"] == "skilladam" else None,
                             "rho_incumbent_seed_matched_last": next((x["rho_incumbent_seed_matched"] for x in reversed(rows) if x.get("rho_incumbent_seed_matched")), None)
                             if r["opt"] == "skilladam" else None}
    # pooled by optimizer: kept vs rejected, relation to dev change, after the best
    pooled = {}
    for opt in OPTS:
        for grp, cond in (("hand-written", lambda r: r["seed"] == "hand-written"), ("near-empty", lambda r: r["seed"] == "near-empty")):
            rs = [r for r in runs if r["opt"] == opt and cond(r)]
            if not rs:
                continue
            rows = [x for r in rs for x in allrows[r["key"]]]
            kept = np.log([x["rho"] for x in rows if x["kept"]]) if any(x["kept"] for x in rows) else np.array([])
            rej = np.log([x["rho"] for x in rows if not x["kept"]]) if any(not x["kept"] for x in rows) else np.array([])
            groups = [(np.log([x["rho"] for x in allrows[r["key"]]]), np.array([x["dsucc_pts"] for x in allrows[r["key"]]])) for r in rs]
            rc, pc, nc = within_run_perm_corr(groups, rng)
            raw_x = np.log([x["rho"] for x in rows])
            raw_y = np.array([x["dsucc_pts"] for x in rows])
            diff = (float(np.exp(kept.mean() - rej.mean())) if len(kept) and len(rej) else None)
            ci = boot_diff(kept, rej, rng) if len(kept) > 1 and len(rej) > 1 else None
            ab = [(per_run[r["key"]]["gm_after_best"], per_run[r["key"]]["best_rho"], per_run[r["key"]]["gm_before_or_at_best"]) for r in rs]
            ratios = [per_run[r["key"]]["after_over_best"] for r in rs if per_run[r["key"]]["after_over_best"] is not None]
            ratios_b = [per_run[r["key"]]["after_over_before"] for r in rs if per_run[r["key"]]["after_over_before"] is not None]
            slopes = [per_run[r["key"]]["slope_log_rho_per_candidate"] for r in rs if per_run[r["key"]]["slope_log_rho_per_candidate"] is not None]
            rof = [per_run[r["key"]]["rest_over_first"] for r in rs if per_run[r["key"]]["rest_over_first"] is not None]
            pooled[f"{opt}|{grp}"] = {
                "n_runs": len(rs), "n_scored": len(rows), "n_kept": int(len(kept)), "n_rejected": int(len(rej)),
                "gm_kept": float(np.exp(kept.mean())) if len(kept) else None, "gm_rejected": float(np.exp(rej.mean())) if len(rej) else None,
                "kept_over_rejected": diff, "kept_over_rejected_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))] if ci else None,
                "spearman_within_run": rc, "perm_p_within_run": pc, "n_in_corr": nc, "spearman_raw_pooled": spearman(raw_x, raw_y) if len(rows) > 2 else None,
                "n_runs_after_over_best": len(ratios), "n_runs_after_over_best_gt1": int(sum(1 for x in ratios if x > 1)),
                "sign_p_after_over_best": sign_test_p(int(sum(1 for x in ratios if x > 1)), len(ratios)),
                "gm_after_over_best_pooled": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
                "n_runs_slope": len(slopes), "n_runs_slope_pos": int(sum(1 for x in slopes if x > 0)), "sign_p_slope": sign_test_p(int(sum(1 for x in slopes if x > 0)), len(slopes)),
                "mean_slope_pct_per_candidate": (100 * float(np.mean(slopes)) if slopes else None),
                "n_runs_rest_over_first": len(rof), "n_runs_rest_over_first_gt1": int(sum(1 for x in rof if x > 1)), "gm_rest_over_first": (float(np.exp(np.mean(np.log(rof)))) if rof else None),
                "n_runs_after_over_before": len(ratios_b), "n_runs_after_over_before_gt1": int(sum(1 for x in ratios_b if x > 1)),
                "gm_after_over_before_pooled": float(np.exp(np.mean(np.log(ratios_b)))) if ratios_b else None}
    return per_run, pooled, allrows


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Q3: content along the path.  The keyword pass of the paper (src/analysis/skilltext.py), copied verbatim; `verify_keyword_copy` asserts that
# CATS, RX, units and classify give the same result as skilltext.py on every skill text of the paths.
# AUTOMATIC FIRST PASS ONLY: none of the paper's 129 hand corrections (OVERRIDES) applies to path steps; no unit was read by hand.
# ------------------------------------------------------------------------------------------------------------------------------------------------
CATS = [
    ("serialize", "serialize / one step or one call at a time"),
    ("parallelize", "parallelize / batch independent calls"),
    ("limit_read", "limit what is read or returned (read once, head/tail, grep, concise output)"),
    ("ask_stop", "ask / confirm with the user / stop / deny / hand off to a human or another party"),
    ("verify", "verify / re-check / re-read / read back / enumerate-and-compare / test against a reference"),
    ("domain", "domain or environment facts and procedures (policy, eligibility, tool or file conventions, build and run steps)"),
    ("format", "formatting / wording / what to put in the final answer"),
    ("other", "other (not a behaviour rule)"),
]
CAT_IDS = [c for c, _ in CATS]

RX = {
    "serialize": r"one at a time|sequential|one step at a time|\bONE candidate|single step|never (?:in )?parallel|not in parallel|without (?:batching|overlap)",
    "parallelize": r"in parallel|parallel(?:ly| calls| tool)|batch(?:ed|ing)? (?:the )?(?:calls|reads|requests)|concurrent|simultaneous|same turn|multiple (?:tool )?calls at once",
    "limit_read": r"read (?:it |each |the file )?once|don'?t re-?read|do not re-?read|\bhead\b|\btail\b|\bgrep\b|truncat|only the (?:relevant|needed)|concise|limit (?:the )?output",
    "ask_stop": r"explicit (?:\"?yes|confirmation|consent)|ask the user|obtain (?:explicit )?confirmation|consent|\bdeny\b|\bdenied\b|escalat|transfer to|hand ?off|\bstop\b|state (?:it|that) cannot be done|the user can|performed by the user|user-run|wait for the (?:user|result)|\bblocker\b|cannot (?:be done|run it yourself)",
    "verify": r"verif|re-?read|read (?:it )?back|re-?query|re-?run|re-?check|re-?measure|re-?apply|confirm|evidence|\bfresh|audit|validat|diff\b|compare|\bcheck|proof|prove|\btests?\b|oracle|reference|assert|\binspect|count",
    "domain": r"polic|eligib|refund|\border|record|account|\bfee|price|\brate\b|balance|dispute|\bSIM\b|roaming|\bAPI|tool|\bfile|path|binary|compile|\bgcc\b|python|clone|commit|push|curl|directory|volume|schema|sheet|email|recipient|window|deadline|clock|time|authenticat|identity|identif",
    "format": r"format|wording|summary|quote|paraphrase|phrase|report (?:the|exactly)|filename|tone|\bsay\b|\bstate\b",
}
RX_SRC = dict(RX)
RX = {k: re.compile(v, re.I) for k, v in RX.items()}

MARKER = re.compile(r"^\s*(?:[-*]\s|\d+\.\s|\||✅|❌)")
FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^#{1,6}\s")
TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
RULE = re.compile(r"^---\s*$")


def classify(text):
    """First-pass category of one unit: the first category in precedence order whose regex matches the unit text, else 'other'."""
    for cat in CAT_IDS:
        if cat == "other":
            continue
        # a category with no regex never fires
        if cat not in RX:
            continue
        if RX[cat].search(text):
            return cat
    return "other"


def units(text):
    """Split a skill into logical units.

    Returns a list of dicts {kind, lines: [line indices], text}. kind is 'rule' (a codable unit), or one of
    'heading', 'fence', 'separator', 'frontmatter' (structure, not coded). A wrapped paragraph or an indented continuation line inside a code fence is
    merged into the unit that precedes it."""
    lines = text.splitlines()
    out = []
    in_fence = False
    in_front = False
    prev_kind = None
    prev_blank = True
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0 and RULE.match(line):
            # opening frontmatter marker
            in_front = True
            out.append({"kind": "frontmatter", "lines": [i], "text": s})
            prev_kind = "frontmatter"
            prev_blank = False
            continue
        if in_front:
            out.append({"kind": "frontmatter", "lines": [i], "text": s})
            if RULE.match(line):
                in_front = False
            prev_kind = "frontmatter"
            prev_blank = False
            continue
        if not s:
            prev_blank = True
            prev_kind = None
            continue
        if FENCE.match(line):
            in_fence = not in_fence
            out.append({"kind": "fence", "lines": [i], "text": s})
            prev_kind = "fence"
            prev_blank = False
            continue
        if TABLE_SEP.match(line):
            out.append({"kind": "separator", "lines": [i], "text": s})
            prev_kind = "separator"
            prev_blank = False
            continue
        if not in_fence and HEADING.match(line):
            out.append({"kind": "heading", "lines": [i], "text": s})
            prev_kind = "heading"
            prev_blank = False
            continue
        # a continuation joins the previous rule unit
        continues = False
        if out and out[-1]["kind"] == "rule" and not prev_blank and prev_kind == "rule":
            if in_fence:
                continues = line[:1] in (" ", "\t") and not MARKER.match(line)
            else:
                continues = not MARKER.match(line) and not s.startswith("**") and not s.startswith("|")
        if continues:
            out[-1]["lines"].append(i)
            out[-1]["text"] += " " + s
        else:
            out.append({"kind": "rule", "lines": [i], "text": s})
        prev_kind = "rule"
        prev_blank = False
    return out


def verify_keyword_copy(runs):
    """This copy of the keyword pass gives the units and categories of src/analysis/skilltext.py on every skill text of the paths."""
    from src.analysis import skilltext
    assert [list(c) for c in CATS] == [list(c) for c in skilltext.CATS], "CATS differs"
    assert RX_SRC == {k: v.pattern for k, v in skilltext.RX.items()}, "RX differs"
    texts = {SEED_HAND.read_text(), SEED_MIN.read_text()} | {s["skill_text"] for r in runs for s in r["d"]["steps"] if s.get("skill_text")}
    for text in texts:
        mine = units(text)
        assert mine == skilltext.units(text), "units differ from skilltext.py"
        rules = [u["text"] for u in mine if u["kind"] == "rule"]
        assert [classify(x) for x in rules] == [skilltext.classify(x) for x in rules], "classify differs from skilltext.py"
    return {"identical": True, "texts_checked": len(texts)}


def added_units(parent_text, child_text):
    """Coded categories of the rule units of `child_text` that contain a line inserted or replaced relative to `parent_text` (whole-line SequenceMatcher,
    blank lines included, as skilltext.added_line_set / code_cell), and the number of added structural units (heading, fence, separator, frontmatter)."""
    a, b = parent_text.splitlines(), child_text.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    added = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            added.update(range(j1, j2))
    coded, structure = [], 0
    for u in units(child_text):
        if not any(i in added for i in u["lines"]):
            continue
        if u["kind"] != "rule":
            structure += 1
            continue
        coded.append(classify(u["text"]))
    return coded, structure


GENERIC = ("verify", "serialize", "parallelize", "limit_read", "ask_stop")


def content_run(r):
    """Per step (every candidate: kept or rejected, in the order produced) the categories of the units it added against its parent."""
    texts = text_of(r)
    out, missing = [], []
    for i, s in enumerate(r["steps"][1:], start=1):
        pid = s["parent_id"]
        if pid is None or pid not in texts:
            missing.append(s["candidate_id"])
            out.append({"i": i, "step": s["step"], "id": s["candidate_id"], "parent": pid, "kept": s["kept"], "cats": None, "n_struct": None})
            continue
        coded, structure = added_units(texts[pid], s["skill_text"])
        out.append({"i": i, "step": s["step"], "id": s["candidate_id"], "parent": pid, "kept": s["kept"], "cats": dict(Counter(coded)), "n_struct": structure})
    return out, missing


def order_test(per_run, rng, nperm=20000):
    """Does domain content come later than generic strategy?  Pooled over the runs of a group: T = mean normalised position (0 = first candidate, 1 =
    last) of the domain units minus that of the generic units (positive = domain later).  Null: the order of the candidates is permuted within each run
    (whole candidates move, so the units of one candidate stay together).  Two-sided permutation p; also first-half / second-half domain share."""
    def stat(orders):
        dom, gen = [], []
        for pos, cats in orders:
            for p, c in zip(pos, cats):
                dom += [p] * c.get("domain", 0)
                gen += [p] * sum(c.get(g, 0) for g in GENERIC)
        if not dom or not gen:
            return float("nan"), len(dom), len(gen)
        return float(np.mean(dom) - np.mean(gen)), len(dom), len(gen)
    orders = []
    first = {"domain": 0, "generic": 0, "all": 0}
    second = {"domain": 0, "generic": 0, "all": 0}
    for rows in per_run:
        rows = [x for x in rows if x["cats"] is not None]
        n = len(rows)
        if n < 2:
            continue
        pos = np.arange(n) / (n - 1)
        orders.append((pos, [x["cats"] for x in rows]))
        for p, x in zip(pos, rows):
            tgt = first if p < 0.5 else second
            c = x["cats"]
            tgt["domain"] += c.get("domain", 0)
            tgt["generic"] += sum(c.get(g, 0) for g in GENERIC)
            tgt["all"] += sum(c.values())
    obs, nd, ng = stat(orders)
    if math.isnan(obs):
        return {"T": None, "n_domain": nd, "n_generic": ng, "p": None, "first_half": first, "second_half": second}
    cnt = 0
    for _ in range(nperm):
        perm = [(pos[rng.permutation(len(pos))], cats) for pos, cats in orders]
        if abs(stat(perm)[0]) >= abs(obs) - 1e-12:
            cnt += 1
    return {"T": obs, "n_domain": nd, "n_generic": ng, "p": (cnt + 1) / (nperm + 1), "first_half": first, "second_half": second}


def q3_stats(runs):
    rng = np.random.default_rng([SEED, 33])
    verify = verify_keyword_copy(runs)
    per_run, missing = {}, {}
    for r in runs:
        per_run[r["key"]], missing[r["key"]] = content_run(r)
    # final skills of the DeepSeek runs against the paper's unit counts and the paper's hand coding
    paper = json.load(open(CACHE / "skilltext.json"))
    seed_text = SEED_HAND.read_text()
    final_check = {}
    for r in runs:
        if r["model"] != "deepseek-flash":
            continue
        texts = text_of(r)
        coded, structure = added_units(seed_text, texts[r["d"]["returned_skill_id"]])
        key = f"{r['opt']}|{r['bench']}"
        final_check[key] = {"mine_units": len(coded), "paper_units": paper["units_per_skill"][key], "mine_counts_auto": dict(Counter(coded)),
                            "paper_counts_hand": paper["counts"][key]}
    # how good is the automatic pass: paper's own units, first pass against the hand-corrected category
    cu = paper["coded_units"]
    conf = Counter()
    n = agree = 0
    coarse = lambda c: "generic" if c in GENERIC else c
    cc = Counter()
    for key, units_ in cu.items():
        for u in units_:
            conf[(u["first_pass"], u["category"])] += 1
            cc[(coarse(u["first_pass"]), coarse(u["category"]))] += 1
            n += 1
            agree += u["first_pass"] == u["category"]
    audit = {"n_units": n, "agree": agree, "agree_share": agree / n, "confusion": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
             "coarse_confusion": {f"{a}->{b}": v for (a, b), v in sorted(cc.items())}, "paper_first_pass_changed_by_hand": paper["first_pass_changed_by_hand"]}
    # groups: hand-written seeds (all six runs per optimizer) and the near-empty seed
    groups = {}
    for opt in OPTS:
        for gname, cond in (("hand-written", lambda r: r["seed"] == "hand-written"), ("near-empty", lambda r: r["seed"] == "near-empty")):
            rs = [r for r in runs if r["opt"] == opt and cond(r)]
            if not rs:
                continue
            cum = defaultdict(lambda: Counter())
            tmax = max(len(per_run[r["key"]]) for r in rs)
            by_step = {t: Counter() for t in range(1, tmax + 1)}
            for r in rs:
                for x in per_run[r["key"]]:
                    if x["cats"]:
                        by_step[x["i"]].update(x["cats"])
            run_tot = Counter()
            for t in range(1, tmax + 1):
                run_tot.update(by_step[t])
                cum[t] = Counter(run_tot)
            # also per-run tests need only runs with >= 2 candidates
            test = order_test([per_run[r["key"]] for r in rs], rng)
            groups[f"{opt}|{gname}"] = {"n_runs": len(rs), "by_step": {t: dict(c) for t, c in by_step.items()}, "cumulative": {t: dict(c) for t, c in cum.items()},
                                        "total": dict(run_tot), "order_test": test,
                                        "n_candidates": sum(len(per_run[r["key"]]) for r in rs)}
    return {"per_run": per_run, "missing_parent": missing, "keyword_copy": verify, "final_check": final_check, "audit": audit, "groups": groups}


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Q4: skill length against dev score
# ------------------------------------------------------------------------------------------------------------------------------------------------
def q4_stats(runs):
    out = {}
    for r in [r for r in runs]:
        P = r["path"]
        seed_words = r["steps"][0]["skill_words"]
        ret = r["d"]["returned_skill_id"]
        ret_words = next((s["skill_words"] for s in r["steps"] if s["candidate_id"] == ret), None)
        rows = P["rows"]
        if r["opt"] == "skilladam":
            kb = P["k_last_accept"]
            ks = [x["k"] for x in rows if x["live"]]
            words_best = rows[kb - 1]["words"] if kb else seed_words
            after = [x for x in rows if x["k"] > kb and x["live"]]
            dev_after = None
        else:
            kb = P["k_best"]
            words_best = rows[kb - 1]["words"] if kb else seed_words
            after = [x for x in rows if x["k"] > kb]
        w_after = [x["words"] for x in after]
        longer = sum(1 for w in w_after if w > words_best)
        sp = spearman([x["k"] for x in after], w_after) if len(after) >= 3 else None
        # words of every scored candidate, for the path figure
        out[r["key"]] = {"seed_words": seed_words, "returned_words": ret_words, "returned_over_seed": (ret_words / seed_words if ret_words else None),
                         "k_best": kb, "words_best": words_best, "n_after_best": len(after), "n_after_longer_than_best": longer,
                         "gm_after_over_best": gm([w / words_best for w in w_after]) if w_after else None,
                         "max_after_over_best": (max(w_after) / words_best if w_after else None), "last_words": rows[-1]["words"] if rows else None,
                         "last_over_best": (rows[-1]["words"] / words_best if rows and words_best else None),
                         "spearman_k_words_after_best": sp, "max_words_all": max([x["words"] for x in rows] + [seed_words]),
                         "words_path": [x["words"] for x in rows],
                         "kept_words_path": [x["words"] for x in rows if x["kept"]]}
    # length against dev change inside a run (rank correlation of words with the candidate's dev change over seed / over incumbent), pooled within run
    rng = np.random.default_rng([SEED, 44])
    corr = {}
    for opt in OPTS:
        for gname, cond in (("hand-written", lambda r: r["seed"] == "hand-written"), ("near-empty", lambda r: r["seed"] == "near-empty")):
            rs = [r for r in runs if r["opt"] == opt and cond(r)]
            if not rs:
                continue
            groups = []
            per = {}
            for r in rs:
                P = r["path"]
                if opt == "skilladam":
                    xs = [(x["words"], x["gain_pts"]) for x in P["rows"] if x["live"]]
                else:
                    xs = [(x["words"], 100.0 * x["gain"] / P["n"]) for x in P["rows"]]
                if len(xs) >= 3:
                    groups.append((np.array([a for a, _ in xs], float), np.array([b for _, b in xs], float)))
                    per[r["key"]] = spearman([a for a, _ in xs], [b for _, b in xs])
            rc, pc, nc = within_run_perm_corr(groups, rng)
            corr[f"{opt}|{gname}"] = {"n_runs": len(groups), "n_candidates": nc, "within_run_spearman": rc, "perm_p": pc, "per_run": per}
    out["_corr"] = corr
    # pooled: runs with at least one later candidate: sign count of 'later candidates are on average longer than the best skill'
    pooled = {}
    for opt in OPTS:
        for gname, cond in (("hand-written", lambda r: r["seed"] == "hand-written"), ("near-empty", lambda r: r["seed"] == "near-empty")):
            rs = [r for r in runs if r["opt"] == opt and cond(r) and out[r["key"]]["n_after_best"] > 0]
            if not rs:
                continue
            ratios = [out[r["key"]]["gm_after_over_best"] for r in rs]
            pooled[f"{opt}|{gname}"] = {"n_runs": len(rs), "n_runs_after_longer": int(sum(1 for x in ratios if x > 1)), "sign_p": sign_test_p(int(sum(1 for x in ratios if x > 1)), len(ratios)),
                                        "gm_after_over_best": float(np.exp(np.mean(np.log(ratios)))),
                                        "gm_returned_over_seed": gm([out[r["key"]]["returned_over_seed"] for r in rs])}
    return out, pooled


# ------------------------------------------------------------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------------------------------------------------------------
def parse_edges_md():
    """Edge counts of the paper's record mech_edges.md (unweighted table): {(scope, status): edges}."""
    txt = (layout.OUT / "data/mech/mech_edges.md").read_text()
    out = {}
    for line in txt.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 3 and cells[0] in ("all", "gepa", "skillopt", "skilladam") and cells[1] in ("accepted", "rejected") and cells[2].isdigit():
            out.setdefault((cells[0], cells[1]), int(cells[2]))   # first occurrence = the unweighted table
    return out


def run_checks(runs, nulls, q1):
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    sel = json.load(open(CACHE / "selection.json"))
    sel_l = json.load(open(CACHE / "selection_luna.json"))
    # dev-scored non-seed candidates per run vs the paper's K / iterations and observed gain
    for r in runs:
        if r["seed"] != "hand-written":
            continue
        P = r["path"]
        ref = (sel[r["bench"]] if r["model"] == "deepseek-flash" else sel_l[r["bench"]])["observed"]
        if r["opt"] == "skillopt":
            exp = ref["SkillOpt"]
            add(f"K and gain {r['key']}", P["K"] == exp["K"] and P["b"] == exp["baseline"] and P["G_final"] == exp["gain"],
                f"mine K={P['K']} baseline={P['b']} gain={P['G_final']}; paper K={exp['K']} baseline={exp['baseline']} gain={exp['gain']}")
            if r["model"] == "gpt-6-luna":
                add(f"candidate scores {r['key']}", [x["S"] for x in P["rows"]] == exp["candidates"], f"mine {[x['S'] for x in P['rows']]} paper {exp['candidates']}")
        elif r["opt"] == "gepa":
            exp = ref["GEPA"]
            add(f"K and gain {r['key']}", P["K"] == exp["K"] and P["b"] == exp["baseline"] and P["G_final"] == exp["gain"],
                f"mine K={P['K']} baseline={P['b']} gain={P['G_final']}; paper K={exp['K']} baseline={exp['baseline']} gain={exp['gain']}")
            if r["model"] == "gpt-6-luna":
                add(f"candidate scores {r['key']}", [x["S"] for x in P["rows"]] == exp["candidates"], f"mine {[x['S'] for x in P['rows']]} paper {exp['candidates']}")
        else:
            exp = ref["SkillAdam"]
            add(f"iterations, accepted, summed gain {r['key']}", P["T"] == exp["iterations"] and P["accepted"] == exp["accepted"] and abs(P["G_final_pts"] - exp["reported_gain_points"]) < 1e-6
                and [x["gain_t"] for x in P["rows"] if x["kept"]] == exp["accepted_gains"],
                f"mine T={P['T']} accepted={P['accepted']} gain={P['G_final_pts']:.1f} {[x['gain_t'] for x in P['rows'] if x['kept']]}; paper {exp['iterations']} {exp['accepted']} {exp['reported_gain_points']:.1f} {exp['accepted_gains']}")
    # returned skill: derived from the dev path equals the optimizer's returned id (SkillOpt, GEPA); SkillAdam: last accepted
    for r in runs:
        P = r["path"]
        ret = r["d"]["returned_skill_id"]
        seed_ids = (r["steps"][0]["candidate_id"], "seed")
        if r["opt"] == "skilladam":
            kept = [x["id"] for x in P["rows"] if x["kept"]]
            derived = kept[-1] if kept else r["steps"][0]["candidate_id"]
        else:
            derived = P["best_id"]
        add(f"returned skill = derived best {r['key']}", derived == ret or (derived in seed_ids and ret in seed_ids), f"derived {derived}; extractor returned {ret}")
    # SkillOpt kept flag = strict > current
    for r in runs:
        if r["opt"] == "skillopt":
            bad = [x["id"] for x in r["path"]["rows"] if x["kept_rule_ok"] is False]
            add(f"SkillOpt kept flag = strict > current {r['key']}", not bad, f"violations {bad}")
    # returned skill text = the skill whose test evaluation exists (version id of its text vs the test evaluation's skill version)
    for r in runs:
        texts = text_of(r)
        ret = r["d"]["returned_skill_id"]
        ft = r["d"]["final_test"]
        ok, detail = False, "no final_test"
        if ft:
            src = ft.get("source", "")
            m = re.search(r"test/([0-9a-f]{12})-", src)
            v = vid(texts[ret])
            ok = bool(m) and v[:12] == m.group(1) and (ft.get("version_id") is None or ft["version_id"] == v)
            detail = f"vid(text of {ret})={v[:12]}; test evaluation {m.group(1) if m else None}"
        add(f"returned text = tested skill {r['key']}", ok, detail)
    # dev edges of the DeepSeek runs (dev-scored, text differs from parent) vs mech_edges.md; SkillAdam x TB2 excluded
    paper = parse_edges_md()
    mine = Counter()
    for r in runs:
        if r["model"] != "deepseek-flash":
            continue
        if r["opt"] == "skilladam" and r["bench"] == "terminal-bench-2-1":
            continue
        texts = text_of(r)
        for s in r["steps"][1:]:
            if not s["dev"]:
                continue
            if s["skill_text"] == texts.get(s["parent_id"]):
                continue
            mine[(r["opt"], "accepted" if s["kept"] else "rejected")] += 1
    for (opt, status), n in sorted(mine.items()):
        add(f"dev edges {opt} {status}", paper.get((opt, status), 0) == n, f"mine {n}; mech_edges.md {paper.get((opt, status), 0)}")
    add("dev edges total (59)", sum(mine.values()) == 59 == paper[("all", "accepted")] + paper[("all", "rejected")], f"mine {sum(mine.values())}; mech_edges.md {paper[('all', 'accepted')]}+{paper[('all', 'rejected')]}")
    # SkillAdam x TB2 (DeepSeek): the parent of every step is unambiguous in the extractor's records
    for r in runs:
        if r["opt"] == "skilladam" and r["bench"] == "terminal-bench-2-1":
            amb = [s["step"] for s in r["steps"][1:] if s["extra"].get("paper_edge_parent_ambiguous")]
            nulls_ = [s["step"] for s in r["steps"][1:] if s["parent_id"] is None]
            add(f"SkillAdam TB2 parents {r['key']}", not nulls_, f"parent null at steps {nulls_}; paper_edge_parent_ambiguous at steps {amb}")
    # the null replay reproduces the paper's P and E (SkillOpt, GEPA, SkillAdam paper replay)
    for r in runs:
        if r["seed"] != "hand-written":
            continue
        n = nulls[r["key"]]
        if r["opt"] != "skilladam":
            ref = (sel[r["bench"]]["priors"] if r["model"] == "deepseek-flash" else None)
            rname = "SkillOpt" if r["opt"] == "skillopt" else "GEPA"
            for pn in ("fitted", "moderate"):
                if r["model"] == "deepseek-flash":
                    rr = sel[r["bench"]]["priors"][pn]["rules"][rname]
                else:
                    rr = sel_l[r["bench"]]["models"][pn]["rules"][rname]
                mp, ep = n["priors"][pn]["P_final_ge_max_obs_1"], n["priors"][pn]["E_G_final_tasks"]
                add(f"null replay reproduces paper P/E {r['key']} {pn}", abs(mp - rr["P_ge_observed_observed_baseline"]) < 0.006 and abs(ep - rr["E_gain_observed_baseline"]) < 0.03,
                    f"P mine {mp:.4f} paper {rr['P_ge_observed_observed_baseline']:.4f}; E mine {ep:.3f} paper {rr['E_gain_observed_baseline']:.3f}")
        else:
            for pn in ("fitted", "moderate"):
                if r["model"] == "deepseek-flash":
                    rr = sel[r["bench"]]["priors"][pn]["rules"]["SkillAdam"]
                else:
                    rr = sel_l[r["bench"]]["models"][pn]["rules"]["SkillAdam"]
                v = n["priors"][pn]["variants"]["paper_replay"]
                mp = v["P_final_ge_obs"] if v["P_final_ge_obs"] is not None else v["P_final_gt_0"]
                add(f"SkillAdam paper replay reproduces paper P/E {r['key']} {pn}", abs(mp - rr["P_reported_gain_ge_observed"]) < 0.008 and abs(v["E_final_pts"] - rr["E_reported_gain_points"]) < 1.0,
                    f"P mine {mp:.4f} paper {rr['P_reported_gain_ge_observed']:.4f}; E mine {v['E_final_pts']:.1f} paper {rr['E_reported_gain_points']:.1f}")
    return checks


# ------------------------------------------------------------------------------------------------------------------------------------------------
# figures (serif 7 pt, GEPA #009E73, SkillOpt #0072B2, SkillAdam #D55E00, light titles)
# ------------------------------------------------------------------------------------------------------------------------------------------------
CAT_COL = {"verify": "#332288", "serialize": "#88CCEE", "parallelize": "#44AA99", "limit_read": "#DDCC77", "ask_stop": "#CC6677", "domain": "#882255",
           "format": "#AA4499", "other": "#BBBBBB"}
STACK = ["verify", "serialize", "parallelize", "limit_read", "ask_stop", "domain", "format", "other"]
BENCH_TEX = {"tau3-bench": r"$\tau^3$-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
BENCH_MK = {"tau3-bench": "o", "automationbench": "s", "terminal-bench-2-1": "^"}
GREY = "#BBBBBB"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 7, "axes.linewidth": 0.6,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.2, length=2, pad=1)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)


def find(runs, model, opt, bench, seed="hand-written"):
    return next((r for r in runs if r["model"] == model and r["opt"] == opt and r["bench"] == bench and r["seed"] == seed), None)


def marker_kw(opt, kept, color=None):
    c = color or OC[opt]
    return dict(marker=MK[opt], s=11, facecolor=c if kept else "white", edgecolor=c, linewidth=0.8, zorder=4)


def fig_score(plt, runs, nulls, model, name):
    cols = [(b, "hand-written") for b in BENCHES] + ([("automationbench", "near-empty")] if model == "gpt-6-luna" else [])
    order = ["gepa", "skillopt", "skilladam"]
    fig, axes = plt.subplots(3, len(cols), figsize=(6.4, 4.9 if len(cols) == 3 else 4.7), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.115, top=0.93, wspace=0.34, hspace=0.42)
    for i, opt in enumerate(order):
        for j, (bench, seedk) in enumerate(cols):
            ax = axes[i][j]
            style(ax)
            r = find(runs, model, opt, bench, seedk)
            if i == 0:
                ax.set_title(BENCH_TEX[bench] + (", near-empty seed" if seedk == "near-empty" else ""), loc="left", fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(f"{OPT_NAME[opt]}\n" + ("dev gain over seed (pts)" if opt != "skilladam" else "summed accepted gain (pts)"), fontsize=6.6, labelpad=1, color=OC[opt])
            if r is None:
                ax.text(0.5, 0.5, "no run\n(cancelled)", transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color="#777777")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            P, N = r["path"], nulls[r["key"]]
            pr = N["priors"][N["primary"]]
            ax.axhline(0, color="#333333", lw=0.5, zorder=1)
            if opt != "skilladam":
                n = P["n"]
                k = np.arange(1, P["K"] + 1)
                b = pr["band_pts"]
                ax.fill_between(k, b["q05"], b["q95"], step="mid", color=GREY, alpha=0.55, lw=0, zorder=0)
                ax.plot(k, b["q50"], color="#777777", lw=0.6, ls="--", drawstyle="steps-mid", zorder=1)
                ax.plot(k, [100.0 * x["G"] / n for x in P["rows"]], color=OC[opt], lw=1.2, drawstyle="steps-mid", zorder=3)
                for x in P["rows"]:
                    ax.scatter([x["k"]], [100.0 * x["gain"] / n], **marker_kw(opt, bool(x["kept"])))
                if P["k_best"]:
                    ax.scatter([P["k_best"]], [100.0 * P["G_final"] / n], marker="*", s=34, color="#222222", zorder=5, linewidths=0)
                ptxt = f"K={P['K']}  P={pr['P_final_ge_max_obs_1']:.2f}" + ("" if P["G_final"] > 0 else "*")
                if r["seed"] == "near-empty":
                    pb = N["priors"]
                    ptxt = f"K={P['K']}  P={pb['moderate']['P_final_ge_max_obs_1']:.2f} / {pb['moderate_devm']['P_final_ge_max_obs_1']:.2f} / {pb['null_plugin']['P_final_ge_max_obs_1']:.2f}"
                ymax = max(max(b["q95"]), max(100.0 * x["G"] / n for x in P["rows"]), max(100.0 * x["gain"] / n for x in P["rows"])) + 5
                ymin = min(0, min(100.0 * x["gain"] / n for x in P["rows"])) - 4
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(0.4, P["K"] + 0.6)
                ax.set_xticks(range(1, P["K"] + 1, 1 if P["K"] <= 8 else 2 if P["K"] <= 16 else 3))
                if i == 1:
                    ax.set_xlabel("scored candidate", fontsize=6.4, labelpad=1)
            else:
                v = pr["variants"]["one_live_candidate"]
                vp = pr["variants"]["paper_replay"]
                k = np.arange(1, P["T"] + 1)
                ax.fill_between(k, v["band_pts"]["q05"], v["band_pts"]["q95"], step="mid", color=GREY, alpha=0.55, lw=0, zorder=0)
                ax.plot(k, v["band_pts"]["q50"], color="#777777", lw=0.6, ls="--", drawstyle="steps-mid", zorder=1)
                for x in P["rows"]:
                    if not x["live"]:
                        ax.scatter([x["k"]], [0], marker="x", s=7, color="#999999", linewidths=0.6, zorder=2)
                    else:
                        ax.bar([x["k"]], [x["gain_pts"]], width=0.55, color=OC[opt] if x["kept"] else "white", edgecolor=OC[opt], linewidth=0.7, zorder=2)
                ax.plot(k, [x["cum"] for x in P["rows"]], color=OC[opt], lw=1.2, drawstyle="steps-mid", zorder=3)
                obsP = pr["variants"]["one_live_candidate"]["P_final_ge_obs"] if P["G_final_pts"] > 0 else pr["variants"]["one_live_candidate"]["P_final_gt_0"]
                ptxt = f"P={obsP:.2f}" + ("" if P["G_final_pts"] > 0 else "*")
                lo = min(0, min(x["gain_pts"] for x in P["rows"])) - 6
                hi = max(max(v["band_pts"]["q95"]), max(x["cum"] for x in P["rows"]), max(x["gain_pts"] for x in P["rows"])) + 8
                ax.set_ylim(lo, hi)
                ax.set_xlim(0.2, P["T"] + 0.8)
                ax.set_xticks(range(2, P["T"] + 1, 2))
                if i == 2:
                    ax.set_xlabel("iteration", fontsize=6.4, labelpad=1)
            ax.text(0.03, 0.97, ptxt, transform=ax.transAxes, ha="left", va="top", fontsize=5.7, color="#333333")
            if r["partial"]:
                ax.text(0.97, 0.97, "partial", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="#AA0000", style="italic")
    handles = [plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="#444444", lw=0, ms=3.6, label="scored candidate, kept"),
               plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="white", lw=0, ms=3.6, label="scored candidate, rejected"),
               plt.Line2D([], [], color="#444444", lw=1.2, label="best so far"),
               plt.Line2D([], [], marker="*", color="#222222", lw=0, ms=5.5, label="best kept"),
               plt.Rectangle((0, 0), 1, 1, facecolor=GREY, alpha=0.55, label="null 5-95%: own rule on exact copies of the seed"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444444", lw=0.7, label="SkillAdam: gate difference (filled: accepted); line: summed accepted gain")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.005), ncol=3, frameon=False, fontsize=5.9, handlelength=1.3, handletextpad=0.4,
               columnspacing=1.0)
    fig.text(0.55, -0.045, "P: share of replays reaching the observed final gain (*: observed 0, P of any gain). Near-empty seed: P under three assumed nulls "
             "(moderate, m pooled / moderate, m dev / plug-in).", ha="center", va="bottom", fontsize=5.6, color="#444444")
    save(fig, name)
    plt.close(fig)


def fig_pooled(plt, pooled, name):
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.3), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.135, top=0.92, wspace=0.28, hspace=0.5)
    for i, g in enumerate(("DeepSeek", "Luna")):
        for j, opt in enumerate(("gepa", "skillopt", "skilladam")):
            ax = axes[i][j]
            style(ax)
            c = pooled["curves"][f"{opt}|{g}"]
            k = [x["k"] for x in c]
            ax.fill_between(k, [x["q05"] for x in c], [x["q95"] for x in c], step="mid", color=GREY, alpha=0.55, lw=0)
            ax.plot(k, [x["null_mean"] for x in c], color="#777777", lw=0.6, ls="--", drawstyle="steps-mid")
            ax.plot(k, [x["obs_mean"] for x in c], color=OC[opt], lw=1.3, drawstyle="steps-mid", marker=MK[opt], ms=2.6, markerfacecolor=OC[opt])
            ax.axhline(0, color="#333333", lw=0.5)
            ax.set_xlim(0.5, max(k) + 0.5)
            if i == 0:
                ax.set_title(OPT_NAME[opt], loc="left", fontsize=7, pad=3, color=OC[opt])
            if j == 0:
                ax.set_ylabel(f"{g}, 3 runs\nmean gain (pts)", fontsize=6.6, labelpad=1)
            ax.set_xlabel("scored candidate" if opt != "skilladam" else "iteration", fontsize=6.4, labelpad=1)
            nn = [x["n_runs"] for x in c]
            ax.text(0.97, 0.05, f"runs: {nn[0]}->{nn[-1]}" if nn[0] != nn[-1] else f"runs: {nn[0]}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="#555555")
    handles = [plt.Line2D([], [], color="#444444", lw=1.3, label="mean best-so-far over the hand-written-seed runs"),
               plt.Rectangle((0, 0), 1, 1, facecolor=GREY, alpha=0.55, label="null 5-95% of that mean (own rule, exact copies of the seed)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.005), ncol=2, frameon=False, fontsize=6, handlelength=1.3, handletextpad=0.4)
    save(fig, name)
    plt.close(fig)


def fig_cost(plt, runs, allrows, name):
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.35), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0], wspace=0.34))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.27, top=0.87)
    axa, axb, axc = axes
    for ax in axes:
        style(ax)
    for r in runs:
        rows = allrows[r["key"]]
        col = OC[r["opt"]]
        ls = "--" if r["seed"] == "near-empty" else "-"
        lw = 0.6
        if r["opt"] in ("gepa", "skillopt"):
            ax = axa
            ax.plot([x["k"] for x in rows], [x["rho"] for x in rows], color=col, lw=lw, ls=ls, alpha=0.75, zorder=2)
            for x in rows:
                ax.scatter([x["k"]], [x["rho"]], **marker_kw(r["opt"], x["kept"]))
            if r["path"]["k_best"]:
                kb = r["path"]["k_best"]
                ax.scatter([kb], [rows[kb - 1]["rho"]], marker="*", s=30, color="#222222", zorder=5, linewidths=0)
        else:
            ax = axb
            ax.plot([x["k"] for x in rows], [x["rho"] for x in rows], color=col, lw=lw, ls=ls, alpha=0.75, zorder=2)
            for x in rows:
                ax.scatter([x["k"]], [x["rho"]], **marker_kw(r["opt"], x["kept"]))
        for x in rows:
            axc.scatter([x["dsucc_pts"]], [x["rho"]], **marker_kw(r["opt"], x["kept"]))
    for ax in (axa, axb, axc):
        ax.axhline(1.0, color="#333333", lw=0.5, ls="--", zorder=1)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1.0, 1.4, 2.0, 2.8])
        ax.set_yticklabels(["0.5", "0.7", "1", "1.4", "2", "2.8"])
        ax.minorticks_off()
        ax.set_ylim(0.45, 3.1)
    axc.axvline(0, color="#333333", lw=0.5, ls="--", zorder=1)
    axa.set_title("(a) GEPA, SkillOpt: tokens / seed", loc="left", fontsize=7)
    axb.set_title("(b) SkillAdam: tokens / incumbent", loc="left", fontsize=7)
    axc.set_title("(c) against the change in dev success", loc="left", fontsize=7)
    axa.set_xlabel("scored candidate", fontsize=6.4, labelpad=1)
    axb.set_xlabel("iteration", fontsize=6.4, labelpad=1)
    axc.set_xlabel("dev success change (pts)", fontsize=6.4, labelpad=1)
    axa.set_ylabel("mean input tokens per dev episode,\nrelative", fontsize=6.4, labelpad=1)
    handles = [plt.Line2D([], [], marker=MK[o], color=OC[o], lw=0, ms=3.8, label=OPT_NAME[o]) for o in OPTS]
    handles += [plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="#444444", lw=0, ms=3.8, label="kept"),
                plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="white", lw=0, ms=3.8, label="rejected"),
                plt.Line2D([], [], marker="*", color="#222222", lw=0, ms=5.5, label="best kept (a)"),
                plt.Line2D([], [], color="#444444", lw=0.7, ls="--", label="near-empty seed")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.015), ncol=7, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.9)
    save(fig, name)
    plt.close(fig)


def fig_content(plt, q3, name):
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.45), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.9, wspace=0.26, hspace=0.62)
    for i, gname in enumerate(("hand-written", "near-empty")):
        for j, opt in enumerate(("gepa", "skillopt", "skilladam")):
            ax = axes[i][j]
            style(ax)
            g = q3["groups"].get(f"{opt}|{gname}")
            if i == 0:
                ax.set_title(OPT_NAME[opt], loc="left", fontsize=7, pad=3, color=OC[opt])
            if g is None:
                ax.text(0.5, 0.5, "no run (cancelled)", transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color="#777777")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            steps = sorted(int(t) for t in g["cumulative"])
            bottom = np.zeros(len(steps))
            tot = np.array([sum(g["cumulative"][t].values()) for t in steps], dtype=float)
            for c in STACK:
                v = np.array([g["cumulative"][t].get(c, 0) for t in steps], dtype=float)
                share = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
                ax.bar(steps, share, bottom=bottom, width=0.86, color=CAT_COL[c], linewidth=0)
                bottom += share
            ax.set_ylim(0, 1)
            ax.set_xlim(0.4, steps[-1] + 0.6)
            ax.set_xlabel("step (cumulative over proposals)", fontsize=6.2, labelpad=1)
            if j == 0:
                ax.set_ylabel(("hand-written seed" if gname == "hand-written" else "near-empty seed") + "\nshare of added units", fontsize=6.4, labelpad=1)
            ax.text(0.99, 1.02, f"{int(tot[-1])} units, {g['n_runs']} run{'s' if g['n_runs'] > 1 else ''}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="#444444")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COL[c], label=c.replace("_", " ")) for c in STACK]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.005), ncol=8, frameon=False, fontsize=6, handlelength=0.9, handletextpad=0.3,
               columnspacing=0.9)
    save(fig, name)
    plt.close(fig)


def fig_length(plt, runs, name):
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 5.0), squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.115, top=0.93, wspace=0.3, hspace=0.45)
    cols = [("deepseek-flash", "hand-written", "DeepSeek"), ("gpt-6-luna", "hand-written", "Luna"), ("gpt-6-luna", "near-empty", "Luna, near-empty seed")]
    for i, opt in enumerate(("gepa", "skillopt", "skilladam")):
        for j, (model, seedk, title) in enumerate(cols):
            ax = axes[i][j]
            style(ax)
            if i == 0:
                ax.set_title(title, loc="left", fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(f"{OPT_NAME[opt]}\n" + ("dev gain over seed (pts)" if opt != "skilladam" else "summed gain, est. (pts)"), fontsize=6.4, labelpad=1, color=OC[opt])
            rs = [r for r in runs if r["model"] == model and r["opt"] == opt and r["seed"] == seedk]
            if not rs:
                ax.text(0.5, 0.5, "no run (cancelled)", transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color="#777777")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.axhline(0, color="#333333", lw=0.5, zorder=1)
            for r in rs:
                P = r["path"]
                col, mk = OC[opt], BENCH_MK[r["bench"]]
                seed_w = r["steps"][0]["skill_words"]
                if opt != "skilladam":
                    n = P["n"]
                    lvl = {x["id"]: (x["words"], 100.0 * x["gain"] / n) for x in P["rows"]}
                    lvl[r["steps"][0]["candidate_id"]] = (seed_w, 0.0)
                    for x in P["rows"]:
                        w, y = lvl[x["id"]]
                        if x["kept"] and x["parent"] in lvl:
                            pw, py = lvl[x["parent"]]
                            ax.plot([pw, w], [py, y], color=col, lw=0.6, alpha=0.55, zorder=2)
                        ax.scatter([w], [y], marker=mk, s=11, facecolor=col if x["kept"] else "white", edgecolor=col, linewidth=0.8, zorder=4)
                    ret = r["d"]["returned_skill_id"]
                    if ret in lvl:
                        ax.scatter([lvl[ret][0]], [lvl[ret][1]], marker="*", s=34, color="#222222", zorder=5, linewidths=0)
                else:
                    cur_w, cur_y = seed_w, 0.0
                    for x in P["rows"]:
                        y = x["cum_before"] + x["gain_pts"]
                        if x["kept"]:
                            ax.plot([cur_w, x["words"]], [cur_y, x["cum"]], color=col, lw=0.6, alpha=0.55, zorder=2)
                            cur_w, cur_y = x["words"], x["cum"]
                        if x["live"]:
                            ax.scatter([x["words"]], [y], marker=mk, s=11, facecolor=col if x["kept"] else "white", edgecolor=col, linewidth=0.8, zorder=4)
                    ax.scatter([cur_w], [cur_y], marker="*", s=34, color="#222222", zorder=5, linewidths=0)
                ax.scatter([seed_w], [0], marker=mk, s=11, facecolor="#444444", edgecolor="#444444", linewidth=0.5, zorder=3)
            ax.set_xscale("log")
            xt = [10, 30, 100, 300, 1000, 3000] if seedk == "near-empty" else [500, 1000, 2000, 3000]
            ax.set_xticks(xt)
            ax.set_xticklabels([str(x) for x in xt])
            ax.minorticks_off()
            if seedk == "near-empty":
                ax.set_xlim(9, 6000)
            else:
                ax.set_xlim(330, 4200)
            if i == 2:
                ax.set_xlabel("skill words", fontsize=6.4, labelpad=1)
    handles = [plt.Line2D([], [], marker=BENCH_MK[b], color="#444444", lw=0, ms=3.8, label=BENCH_TEX[b]) for b in BENCHES]
    handles += [plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="#444444", lw=0, ms=3.8, label="kept (grey: seed)"),
                plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="white", lw=0, ms=3.8, label="rejected"),
                plt.Line2D([], [], marker="*", color="#222222", lw=0, ms=5.5, label="returned"),
                plt.Line2D([], [], color="#444444", lw=0.7, label="parent to kept child")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.005), ncol=7, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.9)
    save(fig, name)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------------------------------------------------------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def compute(draws):
    runs = load_runs()
    inputs = json.load(open(INPUTS))
    for r in runs:
        r["path"] = sa_path(r) if r["opt"] == "skilladam" else panel_path(r)
    nulls, arrays = run_nulls(runs, inputs, draws)
    q1 = {r["key"]: q1_run(r, nulls[r["key"]]) for r in runs}
    pooled = pooled_stats(runs, arrays, nulls)
    q1_first = q1_first_stats(runs, q1)
    per_run_cost, pooled_cost, allrows = q2_stats(runs, inputs)
    q3 = q3_stats(runs)
    q4, q4_pooled = q4_stats(runs)
    checks = run_checks(runs, nulls, q1)
    R = {"draws": draws, "seed": SEED, "q1": q1, "q1_first": q1_first, "pooled": pooled, "cost_per_run": per_run_cost, "cost_pooled": pooled_cost, "q3": q3, "q4": q4, "q4_pooled": q4_pooled,
         "checks": checks, "nulls": {k: {"model_desc": v["model_desc"], "primary": v["primary"], "priors": {pk: {kk: vv for kk, vv in pv.items()} for pk, pv in v["priors"].items()}}
                                      for k, v in nulls.items()},
         "inputs_note": {"near_empty": {k: v for k, v in inputs["gpt-6-luna-nearempty"]["automationbench"].items() if k not in ("skillopt", "gepa")},
                         "counts": {m: {b: inputs[m][b].get("counts_dev_tasks_dev_exec_test_exec") or inputs[m][b].get("counts_dev_tasks_dev_exec") for b in BENCHES} for m in MODELS},
                         "prior_checks": {m: {b: (inputs[m][b]["prior_check"] or {}).get("max_abs_diff") for b in BENCHES} for m in MODELS}}}
    return runs, nulls, arrays, R, allrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100000)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--no-md", action="store_true")
    args = ap.parse_args()
    runs, nulls, arrays, R, allrows = compute(args.draws)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(to_jsonable(R), indent=1, sort_keys=True))
    if not args.no_figs:
        plt = setup_mpl()
        fig_score(plt, runs, nulls, "deepseek-flash", "paths_score_deepseek")
        fig_score(plt, runs, nulls, "gpt-6-luna", "paths_score_luna")
        fig_pooled(plt, R["pooled"], "paths_pooled")
        fig_cost(plt, runs, allrows, "paths_cost")
        fig_content(plt, R["q3"], "paths_content")
        fig_length(plt, runs, "paths_length")
        fig_length_aligned(plt, runs, "paths_length_aligned")
    if not args.no_md:
        OUT_MD.write_text(render_md(runs, R))
    bad = [c for c in R["checks"] if not c["ok"]]
    print(f"checks: {len(R['checks'])} run, {len(bad)} failed")
    for c in bad:
        print("FAILED:", c["check"], c["detail"])


def short(r):
    return f"{MODEL_NAME[r['model']]} {BENCH_NAME[r['bench']]}" + (" near-empty" if r["seed"] == "near-empty" else "")


def render_md(runs, R):
    L = []
    A = L.append
    q1, q1f, pooled = R["q1"], R["q1_first"], R["pooled"]
    cp, cpo, q3, q4, q4p = R["cost_per_run"], R["cost_pooled"], R["q3"], R["q4"], R["q4_pooled"]
    ne = R["inputs_note"]["near_empty"]
    hand = [r for r in runs if r["seed"] == "hand-written"]
    panel_hand = [r for r in hand if r["opt"] != "skilladam"]
    sa_hand = [r for r in hand if r["opt"] == "skilladam"]
    pcs = [q1[r["key"]]["P_final_primary"] for r in panel_hand]
    p_sa = [q1[r["key"]]["P_final_one_live"] for r in sa_hand]
    P_sig = sum(1 for p in pcs if p <= 0.05)
    P_sig_sa = sum(1 for p in p_sa if p <= 0.05)
    above = sum(q1[r["key"]]["n_pointwise_above_q95"] for r in panel_hand) + sum(q1[r["key"]]["n_above_q95_one_live"] for r in sa_hand)
    A("# mech_paths\n")
    A("Audit record written by the script named below; rerunning it refreshes this file. "
      "Do not edit by hand.\n")
    A("## What it shows\n")
    A("The optimization path of every GEPA, SkillOpt and SkillAdam run behind Table 1 (and of the two near-empty-seed runs): for each step what was proposed, "
      "whether the optimizer's own gate kept it, and what it scored, cost and wrote. Four questions: (1) when do the dev gains arrive, and is the best-so-far path "
      "distinguishable from the paper's null (the optimizer's own selection rule replayed on exact copies of the seed, mech_selection); (2) how the input-token "
      "cost moves along the path, kept against rejected, and against dev success; (3) what kind of rule each step adds (the paper's eight categories, keyword pass "
      "only), cumulatively, for hand-written-seed and near-empty-seed runs; (4) how skill length moves against dev score. "
      "Descriptive; the only inference is the one-sided replay probability P below, computed per run and pooled, and it is not a test of a true gain.\n")
    A("## Runs\n")
    A(f"- 20 runs, read from `OUT/skill_paths/paths/*.json` (written by `extract_gepa.py`, `extract_skillopt.py`, `extract_skilladam.py` from the run directories; "
      "each extractor's own cross-checks against the optimizers' logs are in those files). DeepSeek-V4.1-Flash native-20261008-https x 9; GPT-6 Luna native-20261012 x 9; "
      "GPT-6 Luna AutoBench from the near-empty seed (`Solve the task.`, run id fromscratch) x 2 (SkillOpt, GEPA; the SkillAdam fromscratch directory only holds "
      "CANCELLED.txt and is skipped). Seed of the other 18 runs: skills/seed/SKILL.md (580 words).")
    A("- Partial: both fromscratch runs (their test repetitions were still being written; the optimizer loops are finished, summary.json and the final "
      "skill exist, so the steps are final, but test evaluations may still be added; their test numbers are not used here). Last writes seen by the extractors: "
      + "; ".join(f"{short(r)} {r['opt']}: " + (r["d"]["mtimes_seen"].get("results/evaluations/test/bb509d254dad-dfaebc25fa4f/evaluation.json") or r["d"]["mtimes_seen"].get("newest_evaluation_json") or "see json")
                  for r in runs if r["partial"]) + ". No process or file was started, stopped or changed.")
    A("- What the dev number is, per optimizer (verified in the optimizers' code and the extractors' notes, not guessed). **SkillOpt**: every candidate is scored once on the "
      "whole dev panel and accepted iff strictly above the stored score of the current skill (current = best, so best-so-far is the running maximum). **GEPA**: a child is first "
      "compared with its parent on an 8-task train minibatch (strict sum improvement); only accepted children are scored on the whole dev panel and all of them enter the pool "
      "(no dev gate); minibatch-rejected children are never scored on dev (dev = null), so GEPA has no kept-versus-rejected cost contrast; returned = argmax of the dev mean over "
      "the pool, ties to the seed and then the lowest index; K = number of pool entrants, as in the paper's replay. **SkillAdam**: no whole-panel dev score exists; every "
      "iteration rolls the incumbent and its ONE candidate on one fresh random draw of 10 dev tasks (TB2: 9) and accepts iff candidate - incumbent > 0 on that draw "
      "(`feedback_loop.py`: one candidate per iteration; up to 3 patch attempts are retries of the same patch); its path quantity is the paired gate difference and the "
      "summed accepted gain (points of the draw) = the paper's reported gain.")
    A("- Seed inputs of the null (prep_inputs.py, bounded reads): the seed's distinct dev executions per task, with the paper's own functions "
      "(`selection.seed_files/executions/check_inputs`, `luna_seed_files/luna_check_inputs`): counts (dev tasks, dev executions, test executions) "
      + ", ".join(f"{MODEL_NAME[m]} {BENCH_NAME[b]} {R['inputs_note']['counts'][m][b]}" for m in MODELS for b in BENCHES)
      + "; the fitted priors reproduce the paper's cache (max abs difference " + f"{max(v for m in MODELS for v in R['inputs_note']['prior_checks'][m].values() if v is not None):.1e}" + ").\n")
    A("## Aggregation\n")
    A("- Path: scored candidates in the order the optimizer produced them (index k; GEPA k counts dev-scored pool entrants, its iteration number is listed as `step`, so the "
      "first scored candidate of GEPA tau3 DeepSeek is iteration 8). Best kept = the first candidate reaching the maximum dev score (the optimizers' own tie rules); best-so-far "
      "gain G_k = max(0, max over j <= k of score_j - seed draw), in dev tasks (points = tasks / panel x 100). The seed draw is the run's own (paper: observed baseline).")
    A(f"- Null band (Q1): per run, {R['draws']} replays; each replay draws the per-task success probabilities from the paper's task model (posterior from the seed's dev "
      "executions; fitted prior for hand-written seeds, moderate prior Beta(4m, 4(1-m)) as sensitivity), then K candidates identical to the seed with fresh outcomes, the "
      "baseline fixed at the run's seed draw; G_k per k; band = 5/50/95% discrete quantiles. P = share of replays whose final G_K >= the observed final gain (for an observed gain "
      "of 0 the paper's convention: P(G_K >= 1), the chance that selection alone would keep one candidate; marked `*`). Pointwise P per k is in the JSON. "
      "The replay code is selection.py's (best_of semantics), checked below to reproduce the paper's cached P and E for all 12 SkillOpt/GEPA hand-written runs.")
    A("- SkillAdam null: `selection.skilladam_reported_gain` ported with the cumulative gain after every iteration (random 10-task subsets, current skill's outcomes cached per "
      "task, accepted patch becomes current). Two variants: **paper replay** (selection.py as written: `budget` = candidate patches per iteration, budgets from summary.json) and "
      "**one live candidate** (one candidate per iteration that has a live patch, i.e. a patch that applied and changed the text; iterations with an empty or failed patch cannot "
      "accept). `iterations[].budget` is the volatility-scaled EDIT budget (max hunks per patch, `edit_budget.py`; feedback_loop.py generates one candidate per iteration), "
      "not a number of candidates, so the paper replay gives the null more chances than the loop has; both are reported, and the paper replay reproduces the paper's cached "
      "E and P (checks).")
    A(f"- Near-empty seed (assumption): the paper has no model for it. One dev draw of the seed exists (13/30) and one test draw (32/120). Three nulls, all assumptions: "
      f"moderate prior with m = pooled dev+test success of the seed ({ne['pooled_mean_success_dev_and_test']:.3f}, the paper's definition of m) = primary; moderate prior with m = the dev "
      f"draw alone ({ne['dev_mean_success']:.3f}); and the paper's prior-free Luna model 'null plug-in' (per-task rate = share of successes over all {ne['all_dev_n_files']} dev "
      "evaluations of the two fromscratch runs, 22-23 executions per task; it treats every version as a draw of the seed, which a 13-word seed against 500-word skills makes "
      "implausible, but it is the paper's conservative convention). The verdict for these two runs depends on which null is taken.")
    A("- Pooled (hand-written-seed runs only; DeepSeek 3, Luna 3, both 6): the mean final gain in points against the replay's distribution of the same mean (independent replays "
      "per run); pooled curves show a k only while at least two runs still have a k-th scored candidate.")
    A("- Cost (Q2): rho = mean input tokens per dev episode (sum of per-request input_tokens, cached tokens included) of the candidate over that of the seed on the same dev panel "
      "(SkillOpt, GEPA); SkillAdam: over the incumbent on the same draw (paired); and, as a seed-relative figure, over the seed's mean per-task tokens on the draw's tasks "
      "(from the seed's distinct dev executions; one execution per task on Luna, so noisy). Kept vs rejected: ratio of geometric means with a bootstrap 95% interval over "
      "candidates. Relation to dev success: rank correlation of log rho with the dev change (over the seed; SkillAdam: gate difference), centred within run, with a within-run "
      "permutation p (exploratory; candidates of a run are not independent). 'After the best' = scored after the best kept candidate (SkillAdam: after the last accepted iteration).")
    A("- Content (Q3): every added unit of every candidate against its parent (whole-line SequenceMatcher as skilltext.added_line_set; units = skilltext.units: a bullet, numbered "
      "item, table row, example line or wrapped paragraph; headings, fences, separators and front matter are not coded), classified by skilltext.classify. "
      "The keyword code is copied verbatim and checked against skilltext.py on every skill text of the paths (identical). "
      "**Automatic first pass only: none of the paper's 129 hand corrections applies and no unit was read by hand**; counts therefore differ from mech_skilltext 4.2 "
      "(audit below). All proposals count, kept or rejected, in the order produced; GEPA's candidates form a tree (parents are Pareto-selected), so 'cumulative' is a sum over "
      "steps, not along a chain, and a GEPA rewrite re-adds reworded seed lines. Generic strategy = verify, serialize, parallelize, limit read, ask or stop. "
      "Order test: T = mean normalised position (0 first candidate, 1 last) of domain units minus that of generic units; null = the order of the candidates permuted within each "
      "run (20000 permutations); two-sided p.")
    A("- Length (Q4): `skill_words` of every candidate; length against dev change by within-run rank correlation as above.\n")
    A("## Numbers\n")
    # ---- run table
    rows = []
    for r in runs:
        d = r["d"]
        cand = r["steps"][1:]
        rows.append([short(r), OPT_NAME[r["opt"]], d["dev_panel_size"] if r["opt"] != "skilladam" else f"{r['path']['rows'][0]['n_draw']}-task draws of {d['dev_panel_size']}",
                     len(cand), sum(1 for s in cand if s["dev"]), sum(1 for s in cand if s["kept"]), d["returned_skill_id"],
                     f"{d['final_test']['n_success']}/{d['final_test']['n_episodes']}" + (f" ({d['final_test']['n_unknown']} unknown)" if d["final_test"]["n_unknown"] else ""),
                     "yes" if r["partial"] else "no"])
    A("### Runs, steps and the returned skill\n")
    A(md_table(["run", "optimizer", "dev panel", "steps (candidates)", "dev-scored", "kept", "returned", "test of the returned skill (this run's own evaluation)", "partial"], rows))
    A("\nSteps = candidates produced (GEPA: including minibatch-rejected proposals; SkillAdam: iterations, including those without a live patch). The returned skill is the one whose "
      "test evaluation exists (checked below). The test column is the single in-run test evaluation, not Table 1's avg@3.\n")
    # ---- Q1
    A("### 1. When do gains arrive\n")
    A("#### SkillOpt and GEPA: dev score of every scored candidate, in order\n")
    rows = []
    for r in runs:
        if r["seed"] != "hand-written" or r["opt"] == "skilladam":
            continue
        v = q1[r["key"]]
        pr = R["nulls"][r["key"]]["priors"]
        rows.append([OPT_NAME[r["opt"]], short(r), f"{v['seed_dev']}/{v['n_dev']}", " ".join(map(str, v["scores"])),
                     (f"k={v['k_best']} (iteration {v['step_best']})" if v["k_best"] else "seed"),
                     f"+{v['G_final_tasks']} ({v['G_final_pts']:.1f} pts)", f2(v["share_at_first_scored"]) if v["share_at_first_scored"] is not None else "n/a", v["n_after_best"],
                     f"{pr['fitted']['P_final_ge_max_obs_1']:.3f}" + ("" if v["G_final_tasks"] > 0 else "*"), f"{pr['moderate']['P_final_ge_max_obs_1']:.3f}" + ("" if v["G_final_tasks"] > 0 else "*")])
    A(md_table(["optimizer", "run", "seed draw", "dev scores (tasks) of the K scored candidates, in order", "best kept", "final dev gain", "share of the final gain at the first scored candidate",
                "scored after the best", "P fitted", "P moderate"], rows))
    A("\nn/a: final gain 0 (the share is undefined). `*`: observed gain 0, P(G >= 1). No dev outcome was unknown (n_unknown 0 in every dev evaluation used).\n")
    A("#### Near-empty seed (Luna AutoBench; assumed nulls)\n")
    rows = []
    for r in runs:
        if r["seed"] != "near-empty":
            continue
        v = q1[r["key"]]
        pr = R["nulls"][r["key"]]["priors"]
        rows.append([OPT_NAME[r["opt"]], f"{v['seed_dev']}/{v['n_dev']}", " ".join(map(str, v["scores"])), f"k={v['k_best']} (iteration {v['step_best']})", f"+{v['G_final_tasks']} ({v['G_final_pts']:.1f} pts)",
                     f2(v["share_at_first_scored"]), v["n_after_best"], f"{pr['moderate']['P_final_ge_max_obs_1']:.3f}", f"{pr['moderate_devm']['P_final_ge_max_obs_1']:.3f}",
                     f"{pr['null_plugin']['P_final_ge_max_obs_1']:.3f}", f"{pr['moderate']['E_G_final_tasks']:.2f} / {pr['moderate_devm']['E_G_final_tasks']:.2f} / {pr['null_plugin']['E_G_final_tasks']:.2f}"])
    A(md_table(["optimizer", "seed draw", "dev scores of the scored candidates, in order", "best kept", "final dev gain", "share at first scored", "scored after best",
                "P moderate, m pooled (primary)", "P moderate, m dev", "P plug-in", "E[G_K] under the three (tasks)"], rows))
    A("\nPartial runs (see Runs). Under the primary assumed null the observed path is above the band; under the other two it is not distinguishable at 0.05: the verdict is "
      "assumption-driven (the seed's pooled success 0.30 is dragged down by the harder test panel).\n")
    A("#### SkillAdam: gate difference of every iteration and the summed accepted gain\n")
    rows = []
    for r in sa_hand:
        v = q1[r["key"]]
        pr = R["nulls"][r["key"]]["priors"]
        rows.append([short(r), f"{v['T']} ({v['n_live']} live)", ", ".join(f"it{k + 1}: +{g}" for k, g in [(i, x["gain_t"]) for i, x in enumerate(r["path"]["rows"]) if x["kept"]]) or "none",
                     f"{v['G_final_pts']:.1f}", v["n_after_last_accept"],
                     f"{pr['fitted']['variants']['one_live_candidate']['E_final_pts']:.1f} / {pr['fitted']['variants']['paper_replay']['E_final_pts']:.1f}",
                     f"{v['P_by_prior']['fitted']['one_live']:.3f}" + ("" if v["G_final_pts"] > 0 else "*"), f"{v['P_by_prior']['fitted']['paper_replay']:.3f}" + ("" if v["G_final_pts"] > 0 else "*"),
                     f"{v['P_by_prior']['moderate']['one_live']:.3f}" + ("" if v["G_final_pts"] > 0 else "*"), f"{v['P_by_prior']['moderate']['paper_replay']:.3f}" + ("" if v["G_final_pts"] > 0 else "*")])
    A(md_table(["run", "iterations", "accepted (gate difference, tasks)", "summed gain (pts)", "iterations after the last accept", "E[null gain] one live / paper replay (pts)",
                "P fitted, one live", "P fitted, paper replay", "P moderate, one live", "P moderate, paper replay"], rows))
    A("\n`*`: observed 0, P(summed gain > 0). SkillAdam's dev numbers are on 10-task (TB2: 9) draws, a fresh draw per iteration, so no best-so-far on a fixed panel exists; the 'path' is "
      "the cumulative announced gain. SkillAdam x TB2 on DeepSeek is included here; its parents are unambiguous in the extractor's records (checks), and it is excluded only from the 59-edge check.\n")
    A("#### Pooled over the hand-written-seed runs (mean final gain in points)\n")
    rows = []
    for opt in ("skillopt", "gepa", "skilladam"):
        for g in ("DeepSeek", "Luna", "both"):
            if opt != "skilladam":
                f, m = pooled[f"{opt}|{g}|fitted"], pooled[f"{opt}|{g}|moderate"]
                rows.append([OPT_NAME[opt], g, f["n_runs"], f"{f['obs_mean_pts']:.1f}", f"{f['null_mean_pts']:.1f} [{f['q05']:.1f}, {f['q95']:.1f}]", f"{f['P_ge_obs']:.3f}",
                             f"{m['null_mean_pts']:.1f} [{m['q05']:.1f}, {m['q95']:.1f}]", f"{m['P_ge_obs']:.3f}"])
            else:
                f, m = pooled[f"{opt}|{g}|fitted|one_live_candidate"], pooled[f"{opt}|{g}|moderate|one_live_candidate"]
                fp, mp = pooled[f"{opt}|{g}|fitted|paper_replay"], pooled[f"{opt}|{g}|moderate|paper_replay"]
                rows.append([OPT_NAME[opt] + " (one live candidate)", g, f["n_runs"], f"{f['obs_mean_pts']:.1f}", f"{f['null_mean_pts']:.1f} [{f['q05']:.1f}, {f['q95']:.1f}]", f"{f['P_ge_obs']:.3f}",
                             f"{m['null_mean_pts']:.1f} [{m['q05']:.1f}, {m['q95']:.1f}]", f"{m['P_ge_obs']:.3f}"])
                rows.append([OPT_NAME[opt] + " (paper replay)", g, fp["n_runs"], f"{fp['obs_mean_pts']:.1f}", f"{fp['null_mean_pts']:.1f} [{fp['q05']:.1f}, {fp['q95']:.1f}]", f"{fp['P_ge_obs']:.3f}",
                             f"{mp['null_mean_pts']:.1f} [{mp['q05']:.1f}, {mp['q95']:.1f}]", f"{mp['P_ge_obs']:.3f}"])
    A(md_table(["optimizer", "group", "runs", "observed mean gain (pts)", "null mean [5%, 95%], fitted", "P fitted", "null mean [5%, 95%], moderate", "P moderate"], rows))
    A("\n#### When within the run (SkillOpt and GEPA runs with a positive final gain)\n")
    rows = []
    for key, v in q1f.items():
        opt, g = key.split("|")
        rows.append([OPT_NAME[opt], g, v["n_runs"], v["n_positive"], f"{v['obs_first_is_best']} (null expectation {v['exp_first_is_best']:.2f})",
                     f"{v['obs_mean_share1']:.2f} (null {v['null_mean_share1']:.2f})" if v["obs_mean_share1"] is not None else "-",
                     f"{v['median_k_best_over_K']:.2f} (uniform 0.5)" if v["median_k_best_over_K"] is not None else "-", f"{v['median_step_best']:.1f}" if v["median_step_best"] is not None else "-"])
    A(md_table(["optimizer", "seed", "runs", "runs with a positive final gain", "of these, best already at the first scored candidate", "mean share of the final gain at the first scored candidate",
                "median position of the best, k / K", "median iteration of the best"], rows))
    A(f"\nReading. Of {len(panel_hand)} SkillOpt and GEPA hand-written-seed runs, {P_sig} have P <= 0.05 under the fitted prior (smallest {min(pcs):.2f}, largest {max(pcs):.2f}); of {len(sa_hand)} SkillAdam runs "
      f"{P_sig_sa} under the one-live-candidate null (smallest {min(p_sa):.2f}); the observed best-so-far lies above the null's 95% quantile at {above} (run, k) points in all. "
      "This is not evidence of no effect: the replay's power to flag one truly better candidate is 0.02-0.24 (one candidate better by 1-3 tasks, 30-task panels; power tables of mech_selection and "
      "mech_selection_luna; the 9-task TB2 panels reach 0.82-0.96 only for +3 tasks under the fitted model, and 0 under DeepSeek's moderate prior). "
      "Where the gain arrives: see the first-scored-candidate columns; the replay's own expectation is listed next to each count.\n")
    # ---- Q2
    A("### 2. Cost along the path\n")
    rows = []
    for r in runs:
        v = cp[r["key"]]
        rows.append([short(r), OPT_NAME[r["opt"]], v["n_scored"], f"{v['n_kept']}/{v['n_rejected']}", f2(v["rho_first"]), f"{f2(v['rho_min'])}-{f2(v['rho_max'])}", f2(v["rho_gm"]),
                     f2(v["rho_gm_kept"]), f2(v["rho_gm_rejected"]), f2(v["after_over_best"]) if r["opt"] != "skilladam" else f2(v["gm_after_best"]), f2(v["rest_over_first"]),
                     (f"{100 * v['slope_log_rho_per_candidate']:+.1f}" if v["slope_log_rho_per_candidate"] is not None else "-"),
                     (f2(v["rho_seed_matched_gm"]) if r["opt"] == "skilladam" else "-")])
    A(md_table(["run", "optimizer", "scored", "kept / rejected", "rho at the first scored", "rho range", "rho (geometric mean)", "gm kept", "gm rejected",
                "later than the best: gm rho / rho of the best (SkillAdam: gm paired rho)", "gm rho of the others / rho first", "slope (% per candidate)", "SkillAdam: gm rho over the seed, task-matched"], rows))
    A("\nrho = mean input tokens per dev episode over the seed's on the same panel (SkillAdam: over the incumbent's on the same draw). GEPA has no rejected row (rejected children are not scored on dev). "
      "SkillAdam's seed-relative column uses the seed's own per-task tokens (one execution per task on Luna; noisy).\n")
    A("#### Pooled by optimizer (hand-written seeds unless stated)\n")
    rows = []
    for key, v in cpo.items():
        opt, g = key.split("|")
        rows.append([OPT_NAME[opt], g, v["n_runs"], f"{v['n_kept']}/{v['n_rejected']}", f2(v["gm_kept"]), f2(v["gm_rejected"]),
                     (f"{v['kept_over_rejected']:.2f} [{v['kept_over_rejected_ci95'][0]:.2f}, {v['kept_over_rejected_ci95'][1]:.2f}]" if v["kept_over_rejected_ci95"] else (f2(v["kept_over_rejected"]) if v["kept_over_rejected"] else "-")),
                     f"{v['spearman_within_run']:+.2f} (p {v['perm_p_within_run']:.2f}, n {v['n_in_corr']})" if v["spearman_within_run"] == v["spearman_within_run"] else "-",
                     f"{v['mean_slope_pct_per_candidate']:+.1f} ({v['n_runs_slope_pos']}/{v['n_runs_slope']} runs rising, sign p {v['sign_p_slope']:.2f})" if v["mean_slope_pct_per_candidate"] is not None else "-",
                     f"{v['gm_rest_over_first']:.2f} ({v['n_runs_rest_over_first_gt1']}/{v['n_runs_rest_over_first']} above 1)" if v["gm_rest_over_first"] else "-",
                     (f"{v['gm_after_over_best_pooled']:.2f} ({v['n_runs_after_over_best_gt1']}/{v['n_runs_after_over_best']} above 1, sign p {v['sign_p_after_over_best']:.2f})" if v["gm_after_over_best_pooled"] else "-")])
    A(md_table(["optimizer", "seed", "runs", "kept / rejected scored candidates", "gm rho kept", "gm rho rejected", "kept / rejected [95%]", "log rho vs dev change, within-run rank correlation",
                "mean slope of log rho (% per candidate)", "gm rho of later candidates / first", "gm rho after the best / rho of the best (runs)"], rows))
    A("\nSkillAdam rows use the paired rho (candidate over incumbent on the same draw). The correlation is exploratory (p from within-run permutation, candidates of a run not independent).\n")
    # Reading (Q2): every number below is a field of the tables above
    ph = {m: [cp[r["key"]]["rho_first"] for r in panel_hand if r["model"] == m] for m in MODELS}
    gmf = lambda xs: math.exp(float(np.mean(np.log(xs))))
    cg, cs, ca = cpo["gepa|hand-written"], cpo["skillopt|hand-written"], cpo["skilladam|hand-written"]
    kor = lambda v: (f"{v['kept_over_rejected']:.2f} [{v['kept_over_rejected_ci95'][0]:.2f}, {v['kept_over_rejected_ci95'][1]:.2f}]")
    sp_first = {k: sign_test_p(v["n_runs_rest_over_first_gt1"], v["n_runs_rest_over_first"]) for k, v in (("gepa", cg), ("skillopt", cs))}
    A("Reading. The level of cost is already there at the first scored candidate and drifts only slightly afterwards; it does not climb steadily once the dev score stops improving. "
      f"At the first scored candidate rho has geometric mean {gmf(ph['deepseek-flash']):.2f} over the DeepSeek SkillOpt and GEPA runs ({sum(1 for x in ph['deepseek-flash'] if x > 1)} of {len(ph['deepseek-flash'])} above 1) and "
      f"{gmf(ph['gpt-6-luna']):.2f} over the Luna ones ({sum(1 for x in ph['gpt-6-luna'] if x > 1)} of {len(ph['gpt-6-luna'])}). "
      f"Candidates scored after the best cost {cg['gm_after_over_best_pooled']:.2f}x (GEPA; {cg['n_runs_after_over_best_gt1']} of {cg['n_runs_after_over_best']} runs above 1, sign p {cg['sign_p_after_over_best']:.2f}) and "
      f"{cs['gm_after_over_best_pooled']:.2f}x (SkillOpt; {cs['n_runs_after_over_best_gt1']} of {cs['n_runs_after_over_best']}, sign p {cs['sign_p_after_over_best']:.2f}) the tokens of the best candidate, and the later candidates "
      f"of a run cost {cg['gm_rest_over_first']:.2f}x (GEPA; sign p {sp_first['gepa']:.2f}) and {cs['gm_rest_over_first']:.2f}x (SkillOpt; sign p {sp_first['skillopt']:.2f}) the first scored candidate: "
      f"a slight rise, not distinguishable from none by those sign tests over {cs['n_runs_rest_over_first']} runs (with 5 runs the smallest attainable two-sided p is 0.0625). "
      f"Kept and rejected SkillOpt candidates cost the same (kept / rejected {kor(cs)}); SkillAdam's accepted and rejected candidates likewise relative to their incumbent ({kor(ca)}); GEPA has no rejected row. "
      f"The mean slope of log rho per scored candidate is {cg['mean_slope_pct_per_candidate']:+.1f}% (GEPA; {cg['n_runs_slope_pos']} of {cg['n_runs_slope']} runs rising), {cs['mean_slope_pct_per_candidate']:+.1f}% (SkillOpt; {cs['n_runs_slope_pos']} of {cs['n_runs_slope']}) "
      f"and {ca['mean_slope_pct_per_candidate']:+.1f}% (SkillAdam; {ca['n_runs_slope_pos']} of {ca['n_runs_slope']}). "
      f"Within a run, log rho is unrelated to the dev change (within-run rank correlation {cg['spearman_within_run']:+.2f} GEPA, {cs['spearman_within_run']:+.2f} SkillOpt, {ca['spearman_within_run']:+.2f} SkillAdam; "
      f"permutation p {cg['perm_p_within_run']:.2f}, {cs['perm_p_within_run']:.2f}, {ca['perm_p_within_run']:.2f}). All of it is exploratory: a run has 4-18 scored candidates and none of these comparisons has a computed power.\n")
    # ---- Q3
    A("### 3. Content along the path (automatic keyword pass, no hand reading)\n")
    au = q3["audit"]
    cc = au["coarse_confusion"]
    coarse = ["generic", "domain", "format", "other"]
    A(f"Audit of the automatic pass on the paper's own 337 coded units (assets/cache/analysis/main/skilltext.json: first pass against the hand-corrected category): {au['agree']}/{au['n_units']} "
      f"= {100 * au['agree_share']:.0f}% unchanged ({au['paper_first_pass_changed_by_hand']} changed by hand). Coarse groups (generic = verify, serialize, parallelize, limit read, ask or stop), rows = automatic, columns = hand:\n")
    A(md_table(["automatic \\ hand"] + coarse + ["units"], [[a] + [cc.get(f"{a}->{b}", 0) for b in coarse] + [sum(cc.get(f"{a}->{b}", 0) for b in coarse)] for a in coarse]))
    A("\nThe unit counts of this pipeline on the nine DeepSeek final skills equal the paper's (checks); only the categories differ.\n")
    rows = []
    for key, g in q3["groups"].items():
        opt, gname = key.split("|")
        t = g["total"]
        ot = g["order_test"]
        fh, sh = ot["first_half"], ot["second_half"]
        share = lambda d: (f"{100 * d['domain'] / d['all']:.0f}%" if d["all"] else "-")
        rows.append([OPT_NAME[opt], gname, g["n_runs"], g["n_candidates"], sum(t.values())] + [t.get(c, 0) for c in STACK] + [share(fh), share(sh),
                     (f"{ot['T']:+.3f} (p {ot['p']:.2f})" if ot["T"] is not None else "-"), f"{ot['n_domain']}/{ot['n_generic']}"])
    A(md_table(["optimizer", "seed", "runs", "candidates", "units"] + STACK + ["domain share, first half of the steps", "second half", "T = later position of domain than generic (p)", "domain / generic units"], rows))
    A("\nUnits are added units of every proposal (kept or rejected) against its parent; GEPA rewrites re-add reworded seed lines, and the seed itself is a verification skill, so `verify` dominates "
      "every row. Near-empty rows are one run each (n = 1 per optimizer): no generalisation.\n")
    # Reading (Q3): every number below is a field of the table above
    gq = q3["groups"]

    def _s1(key, cat):
        bs = gq[key]["by_step"]      # int keys in memory, str keys after the json round trip
        s1 = bs[min(int(k) for k in bs)]
        return 100 * s1.get(cat, 0) / max(1, sum(s1.values()))

    def _tot(key, cat):
        t = gq[key]["total"]
        return 100 * t.get(cat, 0) / max(1, sum(t.values()))

    def _half(key):
        ot = gq[key]["order_test"]
        return 100 * ot["first_half"]["domain"] / max(1, ot["first_half"]["all"]), 100 * ot["second_half"]["domain"] / max(1, ot["second_half"]["all"])

    def _T(key):
        ot = gq[key]["order_test"]
        return f"T {ot['T']:+.3f}, p {ot['p']:.2f}"

    hw = [("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
    A("Reading (automatic coding only; no unit was read by hand). Neither order appears in the hand-written-seed runs. Domain share of the added units in the first and the second half of the steps: "
      + "; ".join(f"{nm} {_half(o + '|hand-written')[0]:.0f}% and {_half(o + '|hand-written')[1]:.0f}% ({_T(o + '|hand-written')})" for o, nm in hw)
      + ". The first step's proposals already have about the cumulative mix (domain share of the first step / of all steps: "
      + "; ".join(f"{nm} {_s1(o + '|hand-written', 'domain'):.0f}% / {_tot(o + '|hand-written', 'domain'):.0f}%" for o, nm in hw)
      + "). Generic units (verify, serialize, parallelize, limit read, ask or stop) outnumber domain units "
      + ", ".join(f"{gq[o + '|hand-written']['order_test']['n_generic'] / max(1, gq[o + '|hand-written']['order_test']['n_domain']):.1f}:1 ({nm})" for o, nm in hw)
      + ". Near-empty seed (one run per optimizer, no generalisation): GEPA's domain share is "
      + f"{_half('gepa|near-empty')[0]:.0f}% in the first and {_half('gepa|near-empty')[1]:.0f}% in the second half of the steps ({_T('gepa|near-empty')}), and its first candidate is {_s1('gepa|near-empty', 'verify'):.0f}% verify units against "
      + f"{_tot('gepa|near-empty', 'verify'):.0f}% over all steps; SkillOpt's four candidates add {gq['skillopt|near-empty']['order_test']['n_domain']} domain units in {sum(gq['skillopt|near-empty']['total'].values())}. "
      + f"The automatic pass agrees with the paper's hand labels on {100 * au['agree_share']:.0f}% of the paper's {au['n_units']} units and errs in both directions ({cc.get('domain->generic', 0)} units it called domain were generic by hand, "
      + f"{cc.get('generic->domain', 0)} it called generic were domain), so the shares are rough.\n")
    # ---- Q4
    A("### 4. Skill length against dev score\n")
    rows = []
    for r in runs:
        v = q4[r["key"]]
        rows.append([short(r), OPT_NAME[r["opt"]], v["seed_words"], f"{v['returned_words']} (x{v['returned_over_seed']:.2f})", f"k={v['k_best']}" + (f" (iteration {q1[r['key']].get('step_best', '')})" if r["opt"] != "skilladam" and v["k_best"] else ""),
                     v["words_best"], f"{v['n_after_longer_than_best']}/{v['n_after_best']}", f2(v["gm_after_over_best"]), f2(v["max_after_over_best"]),
                     f"{v['spearman_k_words_after_best']:+.2f}" if v["spearman_k_words_after_best"] is not None else "-"])
    A(md_table(["run", "optimizer", "seed words", "returned words", "best kept (SkillAdam: last accepted)", "words of the best", "later candidates longer than the best",
                "gm later / best", "max later / best", "rank corr. of k with words after the best"], rows))
    A("\nSkillAdam later = live candidates after the last accepted iteration. For SkillOpt and SkillAdam the later candidates are children of the best (or of it plus later accepts), so being longer is mostly "
      "mechanical (edits are appended); they are not in the returned skill. GEPA's later pool entrants are kept without a dev gate, so its pool length does grow while dev does not improve.\n")
    rows = []
    for key, v in q4p.items():
        opt, g = key.split("|")
        c = q4["_corr"].get(key, {})
        rows.append([OPT_NAME[opt], g, v["n_runs"], f"{v['n_runs_after_longer']}/{v['n_runs']} (sign p {v['sign_p']:.2f})", f"{v['gm_after_over_best']:.2f}", f"{v['gm_returned_over_seed']:.2f}",
                     (f"{c['within_run_spearman']:+.2f} (p {c['perm_p']:.2f}, n {c['n_candidates']})" if c and c.get("within_run_spearman") == c.get("within_run_spearman") else "-")])
    A(md_table(["optimizer", "seed", "runs with later candidates", "runs whose later candidates are longer than the best (geometric mean of later / best above 1)", "gm later / best", "gm returned / seed words",
                "within-run rank correlation of words with dev change"], rows))

    # Reading (Q4): every number below is a field of the two tables above
    def _first_later_longer(r):
        P = r["path"]
        kb = P["k_best"] if r["opt"] != "skilladam" else P["k_last_accept"]
        wb = r["steps"][0]["skill_words"] if kb == 0 else P["rows"][kb - 1]["words"]
        later = [x for x in P["rows"] if x.get("live", True) and x["k"] > kb]
        return None if not later else later[0]["words"] > wb

    fl = {}
    for o, _ in hw:
        v = [_first_later_longer(r) for r in hand if r["opt"] == o]
        fl[o] = (sum(1 for x in v if x), sum(1 for x in v if x is not None))
    gk = [q4[r["key"]]["spearman_k_words_after_best"] for r in hand if r["opt"] == "gepa" and q4[r["key"]]["spearman_k_words_after_best"] is not None]
    A("\nReading. Length rises along the paths: the returned skill has "
      + ", ".join(f"{q4p[o + '|hand-written']['gm_returned_over_seed']:.2f}x ({nm})" for o, nm in hw)
      + f" the seed's words (geometric means over the hand-written-seed runs; the three Luna TB2 runs returned the seed itself, x1.00) and {q4p['gepa|near-empty']['gm_returned_over_seed']:.1f}x / {q4p['skillopt|near-empty']['gm_returned_over_seed']:.1f}x the near-empty seed's (GEPA / SkillOpt). "
      + "Within a run the words are unrelated to the dev change (rank correlation "
      + ", ".join(f"{q4['_corr'][o + '|hand-written']['within_run_spearman']:+.2f} {nm}" for o, nm in hw)
      + "; permutation p " + ", ".join(f"{q4['_corr'][o + '|hand-written']['perm_p']:.2f}" for o, _ in hw)
      + "). After the best candidate the later candidates are mostly longer than the best, with no dev gain: the first later candidate is longer than the best in "
      + ", ".join(f"{fl[o][0]} of {fl[o][1]} {nm}" for o, nm in hw)
      + " runs with later candidates, and the later candidates are longer than the best, on the geometric mean, in "
      + ", ".join(f"{q4p[o + '|hand-written']['n_runs_after_longer']}/{q4p[o + '|hand-written']['n_runs']} {nm} runs (pooled {q4p[o + '|hand-written']['gm_after_over_best']:.2f}x, sign p {q4p[o + '|hand-written']['sign_p']:.2f})" for o, nm in hw)
      + " (with 5 runs the smallest attainable two-sided p is 0.0625). For SkillOpt and SkillAdam this is mechanical: later candidates are edits of the best skill, were rejected, and are not in the returned skill. "
      + "For GEPA the later pool entrants stay in the pool without a dev gate, but the returned skill is the dev argmax, and the growth after the best is not monotone "
      + f"(rank correlation of the step with the words after the best from {min(gk):+.2f} to {max(gk):+.2f} over the {len(gk)} hand-written-seed runs where it is defined).\n")
    # ---- checks
    A("\n### Checks run\n")
    fam = defaultdict(list)
    for c in R["checks"]:
        key = re.sub(r" (deepseek-flash|gpt-6-luna)__.*$", "", c["check"])
        key = re.sub(r"^dev edges (gepa|skillopt|skilladam) (accepted|rejected)$", "dev edges per optimizer and status", key)
        fam[key].append(c)
    rows = []
    for k, v in fam.items():
        rows.append([k, f"{sum(c['ok'] for c in v)}/{len(v)}", v[0]["detail"][:150]])
    A(md_table(["check", "passed", "first detail"], rows))
    edge = [c for c in R["checks"] if c["check"].startswith("dev edges")]
    A("\nDev edges of the DeepSeek runs (dev-scored, text differs from the parent; SkillAdam x TB2 excluded): " + "; ".join(f"{c['check'][10:]}: {c['detail']}" for c in edge) + ".\n")
    bad = [c for c in R["checks"] if not c["ok"]]
    A(f"{len(R['checks'])} checks, {len(bad)} failed." + ("" if not bad else " Failed: " + "; ".join(c["check"] for c in bad)) + "\n")
    # ---- anomalies
    A("### Anomalies\n")
    A("- Extractor notes carried over (aborted attempts, orphan dev evaluations, resumes): " + "; ".join(
        f"{short(r)} {OPT_NAME[r['opt']]}: " + " | ".join(a[:90] for a in (r["d"].get("anomalies") or [])) for r in runs if r["d"].get("anomalies")) + ".")
    sa_tau3 = next(r for r in runs if r["opt"] == "skilladam" and r["model"] == "deepseek-flash" and r["bench"] == "tau3-bench")
    nv = R["nulls"][sa_tau3["key"]]["priors"]["fitted"]["variants"]
    A("- The paper's SkillAdam replay (selection.py `skilladam_accepts` / `skilladam_reported_gain`) treats the per-iteration `budget` as the number of candidate patches scored on the draw; in the loop it is the "
      "edit budget and one candidate is scored per iteration, so the replay's null accepts more often than the loop could (tau3 DeepSeek, fitted prior: expected accepts "
      f"{nv['paper_replay']['E_accepts']:.2f} in the paper replay against {nv['one_live_candidate']['E_accepts']:.2f} with one live candidate; expected summed gain "
      f"{nv['paper_replay']['E_final_pts']:.1f} against {nv['one_live_candidate']['E_final_pts']:.1f} pts, observed {sa_tau3['path']['G_final_pts']:.1f}). Reported both ways; whether the observed summed "
      f"gain is distinguishable does not change ({sum(1 for r in sa_hand if q1[r['key']]['P_by_prior']['fitted']['one_live'] <= 0.05)} of {len(sa_hand)} runs have P <= 0.05 under the one-live-candidate null, "
      f"{sum(1 for r in sa_hand if q1[r['key']]['P_by_prior']['fitted']['paper_replay'] <= 0.05)} under the paper replay). mech_selection was not changed.")
    A("- SkillAdam candidates with an empty patch (text identical to the parent, a cached tie) or a failed patch (no candidate) are iterations of the path but not 'dev edges' of the paper: "
      + "; ".join(f"{short(r)}: {r['path']['T']} iterations, {r['path']['n_scored']} with a dev score, {r['path']['n_live']} live (text differs)" for r in runs if r["opt"] == "skilladam" and r["seed"] == "hand-written") + ".")
    A("- GEPA minibatch-rejected proposals have text and a train-minibatch score but no dev score; kept = entered the pool.")
    A("- SkillOpt candidates are compared with the current skill's STORED dev score (not re-run together), so equal-text re-scoring noise enters its gate; the seed's dev draw of each run is a single draw.\n")
    A("## Regenerate (from the repository root)\n")
    A("```\n"
      "# 1. per-run path files (read-only on the runs, writes OUT/skill_paths/paths/)\n"
      "python -m aux.skill_paths.extract_gepa\n"
      "python -m aux.skill_paths.extract_skillopt\n"
      "python -m aux.skill_paths.extract_skilladam\n"
      "# 2. the seed's per-task dev executions (imports the paper's selection.py; asserts the paper's counts)\n"
      "python -m aux.skill_paths.prep_inputs\n"
      "# 3. this record (OUT/skill_paths/mech_paths.md), the figures (OUT/skill_paths/fig/*.pdf) and analysis/paths_analysis.json\n"
      "python -m aux.skill_paths.analyze_paths\n"
      "```\n")
    return "\n".join(L)



def fig_length_aligned(plt, runs, name):
    """Candidates aligned at the best step: x = scored index minus the index of the best (0 = best; SkillAdam: the last accepted iteration),
    y = skill words relative to the best skill."""
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.87, wspace=0.08)
    for ax, opt in zip(axes, ("gepa", "skillopt", "skilladam")):
        style(ax)
        ax.axvspan(0.5, 30, color="#F2F2F2", zorder=0, lw=0)
        ax.axhline(1.0, color="#333333", lw=0.5, ls="--", zorder=1)
        ax.axvline(0, color="#333333", lw=0.5, zorder=1)
        xmax = 1
        for r in [r for r in runs if r["opt"] == opt]:
            P = r["path"]
            kb = P["k_best"] if opt != "skilladam" else P["k_last_accept"]
            wb = r["steps"][0]["skill_words"] if kb == 0 else P["rows"][kb - 1]["words"]
            rows = [x for x in P["rows"] if (x.get("live", True))]
            xs = [x["k"] - kb for x in rows]
            ys = [x["words"] / wb for x in rows]
            ax.plot(xs, ys, color=OC[opt], lw=0.6, ls="--" if r["seed"] == "near-empty" else "-", alpha=0.7, zorder=2)
            for x, a, b in zip(rows, xs, ys):
                ax.scatter([a], [b], **marker_kw(opt, bool(x["kept"])))
            xmax = max(xmax, max(xs))
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
        ax.set_yticklabels(["0.5", "0.7", "1", "1.4", "2", "3"])
        ax.minorticks_off()
        ax.set_ylim(0.3, 3.6)
        ax.set_xlim(-8.5, xmax + 0.8)
        ax.set_title(OPT_NAME[opt], loc="left", fontsize=7, color=OC[opt])
        ax.set_xlabel("scored candidate relative to the best" if opt != "skilladam" else "iteration relative to the last accepted", fontsize=6.2, labelpad=1)
        ax.text(0.97, 0.04, "after the best: dev does not improve", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color="#666666")
    axes[0].set_ylabel("skill words / words of the best skill", fontsize=6.4, labelpad=1)
    handles = [plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="#444444", lw=0, ms=3.8, label="kept"),
               plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="white", lw=0, ms=3.8, label="rejected"),
               plt.Line2D([], [], color="#444444", lw=0.7, ls="--", label="near-empty seed")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.015), ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.0)
    save(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    main()
