"""Winner's-curse null model and seed-variability audit (appendix 'selection noise'), driven by the frozen submission manifest (analysis/submission_protocol.py).

Inputs only from the manifest's pinned runs (--manifest FILE or env BACH_SUBMISSION_MANIFEST); nothing falls back to an older run, a pending cell is reported as pending.
Units are (task_id, repetition) of an execution; an execution = one evaluation.json, identified by the sha256 of its request ids (cached copies of one execution share it and are
dropped, independent reruns with identical outcomes stay; evaluations without request ids are excluded and logged) -> dedup_log.json. Seed test executions = the manifest's three
repetitions; seed dev executions = the seed run's dev evaluation plus the dev evaluation of the seed inside each pinned optimizer run that has a summary.json matching the frozen hash.
Per-task success probabilities get a beta prior (fitted beta-binomial / concentration 4 / homogeneous = 1e6) and are updated by the seed executions.
1. calibration (calibration.json): leave-one-execution-out on the seed, the held-out execution's total is predicted from the others (coverage of the central 90% interval, z, predicted vs observed spread).
2. per bench x variant: zero-effect SkillOpt / GEPA selection on dev (apparent gain vs observed, from the pinned summaries' own baselines), SkillAdam accepted patches, and the test comparison of a
   zero-effect candidate with the seed (single run and avg@3); observed test cells are compared with it when the cell is complete. A variant whose LOO calibration fails on a bench, or whose null spread is 0 while the
   observation lies in the tail (< 0.05), is `null_misfit` and gives no bound; the script never certifies a zero effect.
Priors fitted on test executions (and the dev prior's concentration) make these analyses retrospective: never fed back into online selection. Needs scipy. Fixed random seed, Monte Carlo SE reported.
Output (--out DIR, created): calibration.json, cells.json, out.json / out_c4.json / out_c1000000.json, dedup_log.json, summary.txt.
Run: <bach-harness venv python> -m analysis.mechanism.nullmodel --manifest FILE --out DIR   (from bipa/)"""
import argparse, glob, hashlib, json, os
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from analysis import common
from analysis.common import BENCHES, SEED_TEXT, Report, eval_file, vid
from analysis.submission_protocol import execution, fsha

NS, SEED = 200000, 20261008
VARIANTS = (("fitted", None, "out.json"), ("conc4", 4, "out_c4.json"), ("homog", 1000000, "out_c1000000.json"))
OPTS = ("skillopt", "gepa", "skilladam")


# ---- loaders (manifest only) -----------------------------------------------------------------------------------------------------------------------

def load_exec(f):
    """one evaluation.json as an execution: signature, {(task, repetition): success} of the known outcomes, panel multiplicity per task."""
    d = json.load(open(f)); e = execution(f); eps = {}
    for x in d["episodes"]: eps.setdefault((x["task_id"], x.get("repetition", 0)), x)
    return dict(file=str(f), sig=e["signature"], panel_hash=e["panel_hash"], n=len(d["panel"]), mult=Counter(t["task_id"] for t in d["panel"]),
                succ={k: bool(x["success"]) for k, x in eps.items() if x.get("success") is not None})


def dedupe(files, split, bench, log, ref_panel=None):
    """distinct executions among evaluation files: request-id identity; cached copies, no-identity and other-panel files are logged and dropped."""
    out, seen = [], {}
    for run, f in files:
        x = load_exec(f); rec = dict(bench=bench, split=split, run=Path(run).name, file=str(f), signature=x["sig"])
        if not x["sig"]: log.append(dict(rec, kind="no_identity_excluded")); continue
        if ref_panel and x["panel_hash"] != ref_panel: log.append(dict(rec, kind="other_panel_excluded")); continue
        if x["sig"] in seen: log.append(dict(rec, kind="cached_copy", of=seen[x["sig"]])); continue
        seen[x["sig"]] = str(f); out.append(x); log.append(dict(rec, kind="kept"))
    return out


def frozen_runs(m, bench, rows=OPTS):
    """pinned run dirs of the optimizer cells whose summary.json still has the frozen hash (finished runs; the others are pending)."""
    return [Path(c["run"]) for r in rows if (c := m["cells"][bench][r]).get("run") and (Path(c["run"])/"summary.json").is_file() and fsha(Path(c["run"])/"summary.json") == c.get("summary_sha256")]


def seed_execs(m, bench, split, log):
    """distinct seed executions on a split: test = manifest repetitions 0-2; dev = seed run + the seed baseline inside every frozen optimizer run (full dev panel only)."""
    if split == "test":
        reps = common.manifest_reps(bench, "native-seed")
        return dedupe([(f.parent, f) for _, f in sorted(reps.items())], split, bench, log) if reps else []
    srun = Path(m["cells"][bench]["native-seed"]["run"]); v = vid(SEED_TEXT)[:12]
    files = [(r, f) for r in [srun] + frozen_runs(m, bench) for f in sorted(glob.glob(f"{r}/results/evaluations/dev/{v}-*/evaluation.json"))]
    ref = load_exec(files[0][1])["panel_hash"] if files else None
    return dedupe(files, split, bench, log, ref)


def tally(ex):
    """per-task success counts K and observation counts n over executions (repetitions kept as separate observations), the task list and panel multiplicity."""
    K, n = Counter(), Counter()
    for e in ex:
        for (t, _), s in e["succ"].items(): K[t] += s; n[t] += 1
    tasks = sorted(n); mult = np.array([ex[0]["mult"][t] for t in tasks])
    return tasks, np.array([K[t] for t in tasks]), np.array([n[t] for t in tasks]), mult


def totals(ex): return [sum(e["succ"].values()) for e in ex]


# ---- model ----------------------------------------------------------------------------------------------------------------------------------------------

def bb_fit(k, n):
    """maximum-likelihood beta-binomial (a, b) of per-task success counts k out of n observations."""
    def nll(th):
        a, b = np.exp(th)
        return -(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)).sum()
    best = min((minimize(nll, x0, method="Nelder-Mead") for x0 in ([0, 0], [2, 1], [1, 2], [3, 3])), key=lambda r: r.fun)
    return np.exp(best.x)


def prior(K, n, conc, c0=None):
    """(a, b): fitted when conc is None, else concentration conc at the observed mean; c0 overrides the concentration (dev prior takes the test-fitted one)."""
    m = K.sum() / n.sum()
    if c0 is not None: return m * c0, (1 - m) * c0
    if conc is None: return tuple(bb_fit(K, n))
    return m * conc, (1 - m) * conc


class Sim:
    def __init__(self, rng, mult): self.rng, self.idx = rng, np.repeat(np.arange(len(mult)), mult)
    def draw_p(self, K, n, a, b, size): return self.rng.beta(K + a, n - K + b, size=(size, len(K)))  # posterior draws of p_t
    def runsum(self, p): q = p[:, self.idx]; return (self.rng.random(q.shape) < q).sum(1)  # one fresh execution of a zero-effect candidate


def pse(p): return float((p * (1 - p) / NS) ** .5)
def pct(x, q): return float(np.percentile(x, q))
def pit(draws, obs): lo, hi = float((draws < obs).mean()), float((draws <= obs).mean()); return lo, hi, bool(hi >= .05 and lo <= .95)


# ---- calibration ----------------------------------------------------------------------------------------------------------------------------------

def calibrate(data, c_test):
    """leave-one-execution-out on the seed: predict the held-out execution's total from the other executions (posterior-predictive), per bench x split x variant."""
    out = {}
    for b, d in data.items():
        for split in ("test", "dev"):
            ex = d[split]
            for vn, conc, _ in VARIANTS:
                rows = []
                for j in range(len(ex)):
                    rest = ex[:j] + ex[j + 1:]
                    if not rest: continue
                    tasks, K, n, _ = tally(rest); mult = np.array([ex[j]["mult"][t] for t in tasks])
                    a, bb = prior(K, n, conc, c_test[b][vn] if split == "dev" else None)
                    s = Sim(np.random.default_rng([SEED, 1, BENCHES.index(b), j, conc or 0]), mult); X = s.runsum(s.draw_p(K, n, a, bb, NS)); obs = totals([ex[j]])[0]
                    lo, hi, ok = pit(X, obs)
                    rows.append(dict(held_out=ex[j]["file"], obs=obs, pred_mean=float(X.mean()), pred_sd=float(X.std()), z=float((obs - X.mean()) / X.std()) if X.std() > 0 else None, pit=[lo, hi], inside_90=ok,
                                     tail=float(2 * min(hi, 1 - lo)), prior_ab=[float(a), float(bb)], n_others=len(rest)))
                out.setdefault(b, {}).setdefault(split, {})[vn] = dict(rows=rows, retrospective="prior concentration fitted on test executions" if split == "dev" else "prior fitted on seed test executions",
                    coverage_90=(sum(r["inside_90"] for r in rows) / len(rows)) if rows else None, n=len(rows), pass_all_tail_ge_005=all(r["tail"] >= .05 for r in rows),
                    obs_sd=float(np.std(totals(ex), ddof=1)) if len(ex) > 1 else None)
    cov = {vn: [r["inside_90"] for b in out.values() for sp in b.values() for r in sp[vn]["rows"]] for vn, _, _ in VARIANTS}
    out["_overall"] = {vn: dict(coverage_90=sum(v) / len(v) if v else None, n=len(v)) for vn, v in cov.items()}
    return out


# ---- observed selection and test ---------------------------------------------------------------------------------------------------------------------

def selection(m, b):
    """SkillOpt / GEPA dev histories of the pinned runs: candidates' dev scores in tasks (V = dev panel), the seed baseline inside that run, observed gain, returned seed or not."""
    sel = {}
    for meth in ("skillopt", "gepa"):
        c = m["cells"][b][meth]; run = Path(c["run"]) if c.get("run") else None
        if not run or not (run/"summary.json").is_file(): sel[meth] = dict(status="pending", reason=c.get("reason")); continue
        if fsha(run/"summary.json") != c.get("summary_sha256"): sel[meth] = dict(status="changed_since_freeze"); continue
        s = json.load(open(run/"summary.json")); nV = s["panels"]["sel"]; x = s["extra"]
        if meth == "skillopt":
            hist = json.load(open(run/"skillopt/history.json")); acts = [h["action"] for h in hist]
            base, best = round(x["baseline_selection_hard"] * nV), round(x["best_selection_hard"] * nV)
            sel[meth] = dict(status="ok", nV=nV, K=len(hist), cand=[round(h["selection_hard"] * nV) for h in hist], actions=acts, base=base, best=best, gain=best - base,
                             returned_seed=best == base and "accept_new_best" not in acts)
        else:
            sc = np.round(np.array(x["val_scores"]) * nV).astype(int).tolist()
            sel[meth] = dict(status="ok", nV=nV, K=len(sc) - 1, scores=sc, base=sc[0], best=max(sc), gain=max(sc) - sc[0], returned_seed=x["best_idx"] == 0)
    return sel


def test_cell(m, b, meth, seed_ex):
    """the observed test cell of an optimizer: pending / returned_seed / complete with the three repetitions' totals (repetitions kept)."""
    c = m["cells"][b][meth]
    if c["status"] != "complete": return dict(status="pending", reason=c.get("reason"))
    if c.get("executions_from"): return dict(status="returned_seed", note="row = Seed row's executions; excluded from changed-cell counts")
    reps = common.manifest_reps(b, meth)
    cand = [load_exec(f) for _, f in sorted(reps.items())]
    return dict(status="complete", cand_totals=totals(cand), cand_files=[e["file"] for e in cand], seed_totals=totals(seed_ex))


def simulate(m, data, sel, cal, vi):
    """one variant of the null model over all benches."""
    vn, conc, _ = VARIANTS[vi]; out = {}
    for b in BENCHES:
        d = data[b]; r = dict(variant=vn, retrospective="prior fitted on seed test executions (test), dev prior concentration from test"); out[b] = r
        if len(d["test"]) < 2 or not d["dev"]: r["status"] = "pending: fewer than 2 seed test executions or no seed dev execution"; continue
        tt, Kt, nt, mt = tally(d["test"]); tv, Kv, nv, mv = tally(d["dev"]); a, bb = prior(Kt, nt, conc)
        s = Sim(np.random.default_rng([SEED, 2, BENCHES.index(b), vi]), mt)
        r.update(test_execs=len(d["test"]), dev_execs=len(d["dev"]), test_tasks=len(tt), dev_tasks=len(tv), beta_prior_a_b=[float(a), float(bb)], concentration=float(a + bb),
                 fit_sigma0=bool(a + bb > 1e4), test_seed_totals=totals(d["test"]), dev_seed_totals=totals(d["dev"]), test_pooled_rate=float(Kt.sum() / nt.sum()), dev_pooled_rate=float(Kv.sum() / nv.sum()))
        aV, bV = prior(Kv, nv, None, a + bb); sv = Sim(np.random.default_rng([SEED, 3, BENCHES.index(b), vi]), mv); pV = sv.draw_p(Kv, nv, aV, bV, NS)
        r["dev_single_run_sd"] = float(sv.runsum(pV).std())
        def v_sel(cond):
            res = {}
            for meth, o in sel[b].items():
                if o["status"] != "ok": res[meth] = dict(status=o["status"]); continue
                s0 = np.full(NS, o["base"]) if cond else sv.runsum(pV)
                cand = np.stack([sv.runsum(pV) for _ in range(o["K"])], 1)
                if meth == "skillopt":  # strict > running best
                    best, changed = s0.copy(), np.zeros(NS, bool)
                    for i in range(o["K"]): up = cand[:, i] > best; best = np.where(up, cand[:, i], best); changed |= up
                    g = best - s0
                else:  # argmax over the seed and the candidates, first index wins ties
                    allc = np.concatenate([s0[:, None], cand], 1); changed = allc.argmax(1) != 0; g = allc.max(1) - s0
                p = float((g >= o["gain"]).mean())
                res[meth] = dict(status="ok", K=o["K"], obs_gain=o["gain"], obs_base=o["base"], obs_returned_seed=o["returned_seed"], p_returned_ne_seed=float(changed.mean()),
                                 E_gain_tasks=float(g.mean()), E_gain_se=float(g.std() / NS ** .5), E_gain_given_changed=float(g[changed].mean()) if changed.any() else 0.0,
                                 P_gain_ge_obs=p, P_gain_ge_obs_se=pse(p), gain_q05_50_95=[pct(g, q) for q in (5, 50, 95)])
            return res
        r["dev_uncond"], r["dev_cond_on_obs_baseline"] = v_sel(False), v_sel(True)
        dxy = sv.runsum(pV) - sv.runsum(pV); r["dev_two_zero_effect_skills"] = dict(sd_diff=float(dxy.std()), P_abs_diff_ge=[float((abs(dxy) >= k).mean()) for k in (1, 2, 3)])
        pT = s.draw_p(Kt, nt, a, bb, NS); Sobs = totals(d["test"]); S0, Sbar = Sobs[0], float(np.mean(Sobs))
        X1, X2, X3 = s.runsum(pT), s.runsum(pT), s.runsum(pT); d1, d3 = X1 - S0, (X1 + X2 + X3) / 3 - Sbar
        r["test"] = dict(n=len(tt), seed_totals=Sobs, null_single_sd=float(X1.std()), seed_observed_sd=float(np.std(Sobs, ddof=1)),
                         single_vs_seed_rep0=dict(mean=float(d1.mean()), sd=float(d1.std()), P_below=float((d1 < 0).mean()), P_equal=float((d1 == 0).mean()), q05_q95=[pct(d1, 5), pct(d1, 95)]),
                         avg3_vs_seed_avg3=dict(mean=float(d3.mean()), sd=float(d3.std()), P_below=float((d3 < 0).mean()), q05_q95=[pct(d3, 5), pct(d3, 95)]), pct_scale=100.0 / len(tt))
        ok = {sp: bool(cal[b][sp][vn]["rows"]) and cal[b][sp][vn]["pass_all_tail_ge_005"] for sp in ("test", "dev")}; r["calibration_pass"] = ok
        r["cells"] = {}
        for meth in ("skillopt", "gepa", "skilladam"):
            tc = test_cell(m, b, meth, d["test"]); r["cells"][meth] = tc
            if tc["status"] != "complete": continue
            c = tc["cand_totals"]; o1, o3 = c[0] - Sobs[0], float(np.mean(c) - np.mean(Sobs))
            for nm, o, dd in (("single_rep0", o1, d1), ("avg3", o3, d3)):
                lo, hi = float((dd <= o).mean()), float((dd >= o).mean()); tail = 2 * min(lo, hi)
                tc[nm] = dict(obs_diff=o, null_mean=float(dd.mean()), null_sd=float(dd.std()), z=float((o - dd.mean()) / dd.std()) if dd.std() > 0 else None, P_le=lo, P_ge=hi, tail2=min(1.0, tail),
                              verdict="null_misfit" if (dd.std() == 0 and tail < .05) or not ok["test"] else "incompatible_with_null" if tail < .05 else "compatible_with_null",
                              bound="undefined" if (dd.std() == 0 and tail < .05) or not ok["test"] else None)
            tc["cand_sd_vs_null_single_sd"] = [float(np.std(c, ddof=1)), r["test"]["null_single_sd"]]
    return out


def skilladam_accepts(m, b, out, vi):
    """SkillAdam: accepted patches and gain when every strict patch is a zero-effect candidate on a fresh train batch (pending until the pinned run has a summary.json)."""
    c = m["cells"][b]["skilladam"]; run = Path(c["run"]) if c.get("run") else None
    if not run or not (run/"summary.json").is_file(): return dict(status="pending", reason=c.get("reason"))
    its = json.load(open(run/"summary.json"))["extra"]["iterations"]; a, bb = out[b]["beta_prior_a_b"]
    f = eval_file(Path(m["cells"][b]["native-seed"]["run"]), "train", vid(SEED_TEXT)); rate = np.mean([e["success"] for e in json.load(open(f))["episodes"]]); cc = a + bb
    aT, bT = rate * cc, (1 - rate) * cc; rng = np.random.default_rng([SEED, 4, BENCHES.index(b), vi])
    bsize = lambda it: 8 if any(v in (0.75, 0.875) for v in (it["baseline"], it["candidate"])) else 10 if b != "terminal-bench-2-1" else 9
    strict = [it for it in its if it["patch_mode"] == "strict"]; bs = [bsize(it) for it in strict]; acc = [it for it in its if it["accepted"]]
    gains = [round((it["candidate"] - it["baseline"]) * bsize(it)) for it in acc]; n_acc, g_acc = np.zeros(NS, int), np.zeros(NS)
    for n in bs:
        p = rng.beta(aT, bT, size=(NS, n)); x, y = (rng.random(p.shape) < p).sum(1), (rng.random(p.shape) < p).sum(1)
        w = y > x; n_acc += w; g_acc += np.where(w, y - x, 0)
    pa, ps = float((n_acc >= len(acc)).mean()), float((g_acc >= sum(gains)).mean())
    return dict(status="ok", n_iter=len(its), n_scored=len(strict), batch_sizes=bs, obs_accepts=len(acc), obs_sum_gain_tasks=int(sum(gains)), train_seed_rate=float(rate),
                E_accepts=float(n_acc.mean()), P_ge_1_accept=float((n_acc > 0).mean()), P_accepts_ge_obs=pa, P_accepts_ge_obs_se=pse(pa), E_sum_gain_tasks=float(g_acc.mean()),
                P_sum_gain_ge_obs=ps, P_sum_gain_ge_obs_se=pse(ps))


# ---- report ---------------------------------------------------------------------------------------------------------------------------------------------

def summarize(variants, cal, say):
    say("##### calibration: leave-one-execution-out on the seed (retrospective: priors use test executions)")
    for vn, o in cal["_overall"].items(): say(f"  {vn}: central-90% coverage {o['coverage_90']} over {o['n']} held-out executions (small n: coverage is coarse)")
    for b in BENCHES:
        for sp in ("test", "dev"):
            for vn, _, _ in VARIANTS:
                c = cal[b][sp][vn]; say(f"  [{b}] {sp:4s} {vn:6s} n={c['n']} cov90={c['coverage_90']} pass(tail>=.05)={c['pass_all_tail_ge_005']} obs_sd={c['obs_sd']} " + "; ".join(f"obs {r['obs']} pred {r['pred_mean']:.1f}+-{r['pred_sd']:.1f} z={r['z']}" for r in c["rows"]))
    for vn, o in variants.items():
        say(f"##### variant {vn} (NS={NS}, seed={SEED})")
        for b in BENCHES:
            r = o[b]
            if "status" in r: say(f"[{b}] {r['status']}"); continue
            say(f"[{b}] prior a,b={r['beta_prior_a_b'][0]:.3f},{r['beta_prior_a_b'][1]:.3f} sigma0_fit={r['fit_sigma0']} dev seed totals {r['dev_seed_totals']} (single-run sd {r['dev_single_run_sd']:.2f}) test seed totals {r['test_seed_totals']}")
            for key in ("dev_uncond", "dev_cond_on_obs_baseline"):
                for meth, x in r[key].items():
                    say(f"  {key:24s}{meth:9s} " + (f"K={x['K']} obs base {x['obs_base']} gain {x['obs_gain']} E[gain]={x['E_gain_tasks']:.2f}+-{x['E_gain_se']:.3f} P(gain>=obs)={x['P_gain_ge_obs']:.3f}+-{x['P_gain_ge_obs_se']:.3f}" if x["status"] == "ok" else x["status"]))
            t = r["test"]; say(f"  test null: single-run sd {t['null_single_sd']:.2f} vs observed seed sd {t['seed_observed_sd']:.2f}; rep0 diff sd {t['single_vs_seed_rep0']['sd']:.2f}; avg@3 diff sd {t['avg3_vs_seed_avg3']['sd']:.2f}")
            sa = skilladam_s(o, b)
            say(f"  SkillAdam: " + (f"scored={sa['n_scored']} E[acc]={sa['E_accepts']:.2f} (obs {sa['obs_accepts']}) P(>=obs)={sa['P_accepts_ge_obs']:.3f}+-{sa['P_accepts_ge_obs_se']:.3f}" if sa["status"] == "ok" else sa["status"]))
            for meth, tc in r["cells"].items():
                say(f"  test cell {meth}: " + (tc["status"] + (": " + str(tc.get("reason")) if tc.get("reason") else "") if tc["status"] != "complete" else
                    f"cand {tc['cand_totals']} seed {tc['seed_totals']} avg3 {tc['avg3']['obs_diff']:+.2f} z={tc['avg3']['z']} -> {tc['avg3']['verdict']}; rep0 {tc['single_rep0']['obs_diff']:+d} -> {tc['single_rep0']['verdict']}"))


def skilladam_s(o, b): return o[b]["SkillAdam_accept_batch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.environ.get("BACH_SUBMISSION_MANIFEST"), help="protocol.json from submission_protocol.py"); ap.add_argument("--out", required=True, help="output directory (created)")
    args = ap.parse_args(); os.environ["BACH_SUBMISSION_MANIFEST"] = args.manifest; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m = common.manifest(); log = []
    data = {b: {sp: seed_execs(m, b, sp, log) for sp in ("test", "dev")} for b in BENCHES}
    c_test = {}
    for b in BENCHES:  # test-fitted concentration per variant (the dev prior takes it)
        tasks, K, n, _ = tally(data[b]["test"]); c_test[b] = {vn: (sum(prior(K, n, conc)) if conc is None else conc) for vn, conc, _ in VARIANTS}
    cal = calibrate(data, c_test); sel = {b: selection(m, b) for b in BENCHES}
    variants = {}
    for vi, (vn, conc, fn) in enumerate(VARIANTS):
        variants[vn] = simulate(m, data, sel, cal, vi)
        for b in BENCHES:
            if "status" not in variants[vn][b]: variants[vn][b]["SkillAdam_accept_batch"] = skilladam_accepts(m, b, variants[vn], vi)
        (out/fn).write_text(json.dumps(variants[vn], indent=1, default=float))
    (out/"calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    (out/"cells.json").write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), manifest_created=m["created_utc"], NS=NS, seed=SEED, selection=sel,
        seed_executions={b: {sp: [e["file"] for e in data[b][sp]] for sp in data[b]} for b in BENCHES}), indent=1, default=float))
    (out/"dedup_log.json").write_text(json.dumps(log, indent=1))
    s = Report(); summarize(variants, cal, s); (out/"summary.txt").write_text("\n".join(s.lines) + "\n")


if __name__ == "__main__":
    main()
