"""Task-level paired summary against the Seed row, from the frozen submission manifest (analysis/submission_protocol.py; --manifest FILE or env BACH_SUBMISSION_MANIFEST).

1. rows (paired.json): every complete row of a benchmark vs the seed's three test executions, same definitions as analysis/table1.py (per-task mean over the row's executions; Q on tasks with a known outcome,
   X and N on tasks with N, X > 0 on both sides): dQ (points), dlogX, rho = exp(mean dlogX), dlogN, effective tasks, bootstrap over tasks (95% percentile interval, fixed seed, tasks resampled jointly),
   wins / losses / ties on per-task success, executions, missing / unknown / invalid episodes. Pending rows are listed with the manifest's reason, returned seeds are the Seed row itself (dQ = 0 by identity).
   `seed_floor`: each seed execution against the other two. The seed's variation is the yardstick.
2. dev -> test (dev_test.json, candidates.json): the frozen candidate set = every non-seed version with a full-panel dev evaluation and a test evaluation in the manifest's finished optimizer runs
   (summary.json hash as frozen) or in runs listed with --extra-run PATH (.../<method>/<bench>/<run>); reference = the seed's executions on the same panel (dev: distinct seed dev executions, test: three repetitions).
   Per bench, success and log X side by side, with sensitivity to the test repetitions (rep 0 only vs all available) and to lineage (candidates of one run share ancestry: one candidate per run, leave-one-run-out).
Executions are identified by request-id signature (dedup_log.json). Read-only on artifacts. Run: <bach-harness venv python> -m src.analysis.mechanism.paired --manifest FILE --out DIR   (from bipa/)"""
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np
from src.analysis import common
from src.analysis.common import BENCHES, EX, SEED_TEXT, vid
from src.analysis.submission_protocol import execution, fsha, pin, scan
from src.analysis.table1 import rows_of
from src.analysis.mechanism.nullmodel import seed_execs

SEED, BOOT = 20261008, 10000
ROWS = ("native-noseed", "gepa", "skillopt", "skilladam")
OPTS = ("gepa", "skillopt", "skilladam")


def per_task(files):
    """{task: {s: mean success over executions with a known outcome, N, X: mean over those with N, X}} of a list of evaluation.json."""
    T = {}
    for f in files:
        for (t, _), r in rows_of(Path(f)).items():
            if r["success"] is None: continue
            x = T.setdefault(t, dict(s=[], N=[], X=[])); x["s"].append(r["success"] is True)
            if r["N"] is not None and r["X"] is not None: x["N"].append(r["N"]); x["X"].append(r["X"])
    return {t: dict(s=float(np.mean(v["s"])), N=float(np.mean(v["N"])) if v["N"] else None, X=float(np.mean(v["X"])) if v["X"] else None) for t, v in T.items()}


def compare(R, S, rng):
    """paired summary of a row's per-task means R against the reference S: dQ (points), dlogX, rho, dlogN, task counts, wins/losses/ties, bootstrap over tasks."""
    ts = sorted(set(R) & set(S)); dq = np.array([R[t]["s"] - S[t]["s"] for t in ts])
    lg = lambda k: np.array([math.log(R[t][k] / S[t][k]) if R[t][k] and S[t][k] else np.nan for t in ts])
    dlx, dln = lg("X"), lg("N"); idx = rng.integers(0, len(ts), size=(BOOT, len(ts)))
    def stat(v, scale=1.0, f=lambda x: x):
        ok = ~np.isnan(v); b = np.nanmean(v[idx], axis=1)
        return dict(est=float(f(np.mean(v[ok]) * scale)) if ok.any() else None, ci95=[float(f(np.percentile(b, q) * scale)) for q in (2.5, 97.5)] if ok.any() else None)
    return dict(n_tasks_Q=len(ts), n_tasks_X=int((~np.isnan(dlx)).sum()), dQ=stat(dq, 100.0), dlogX=stat(dlx), rho_X=stat(dlx, 1.0, math.exp), dlogN=stat(dln), rho_N=stat(dln, 1.0, math.exp),
                wins=int((dq > 0).sum()), losses=int((dq < 0).sum()), ties=int((dq == 0).sum()), X_lower=int((dlx < 0).sum()), X_higher=int((dlx > 0).sum()),
                Q_row=100 * float(np.mean([R[t]["s"] for t in ts])), Q_ref=100 * float(np.mean([S[t]["s"] for t in ts])))


# ---- rows ---------------------------------------------------------------------------------------------------------------------------------------------

def rows_summary(m):
    out, floor, pend = {}, {}, []
    for bi, b in enumerate(BENCHES):
        seed = common.manifest_reps(b, "native-seed")
        if not seed: pend += [f"{b}/native-seed: pending"]; continue
        sf = [str(f) for _, f in sorted(seed.items())]; S = per_task(sf); out[b] = {}
        floor[b] = {k: compare(per_task([f]), per_task(sf[:k] + sf[k + 1:]), np.random.default_rng([SEED, 1, bi, k])) for k, f in enumerate(sf)}
        for ri, row in enumerate(ROWS):
            c = m["cells"][b][row]
            if c["status"] != "complete": out[b][row] = dict(status="pending", reason=c.get("reason")); pend.append(f"{b}/{row}: {c.get('reason')}"); continue
            if c.get("executions_from"): out[b][row] = dict(status="returned_seed", note="identical to the Seed row (same executions)", dQ=dict(est=0.0), rho_X=dict(est=1.0)); continue
            reps = common.manifest_reps(b, row); ex = [c["executions"][str(k)] for k in sorted(reps)]
            r = compare(per_task([str(f) for _, f in sorted(reps.items())]), S, np.random.default_rng([SEED, 2, bi, ri]))
            out[b][row] = dict(status="complete", run=c["run_name"], executions=len(ex), seed_executions=len(sf), signatures=[e["signature"] for e in ex],
                               missing=sum(e["missing"] for e in ex), unknown=sum(e["unknown"] for e in ex), invalid=sum(e["invalid"] for e in ex), **r)
    return out, floor, pend


# ---- dev -> test ---------------------------------------------------------------------------------------------------------------------------------------

def finished(m, bench, extra):
    """[(method, bench, run, source)] of frozen finished optimizer runs plus the explicitly listed ones."""
    runs = []
    for meth in OPTS:
        c = m["cells"][bench][meth]
        if c.get("run") and (Path(c["run"])/"summary.json").is_file() and fsha(Path(c["run"])/"summary.json") == c.get("summary_sha256"): runs.append((meth, Path(c["run"]), "manifest"))
    return runs + [(p.parent.parent.name, p, "explicit") for p in extra if p.parent.name == bench]


def candidates(m, bench, extra, log):
    """the frozen candidate set of a benchmark: non-seed versions with full-panel dev executions and test executions, with their dev / test evaluation files and source runs."""
    sd = seed_execs(m, bench, "dev", log); panel = execution(sd[0]["file"])["panel_hash"]; sv = vid(SEED_TEXT)[:12]; found = {}
    for meth, run, src in finished(m, bench, extra):
        for d in sorted((run/"results/evaluations/dev").glob("*/evaluation.json")):
            v = d.parent.name.split("-")[0]
            if v != sv: x = found.setdefault(v, dict(runs=[], dev=[])); x["runs"].append((meth, run, src)); x["dev"].append((run, d))
    out = []
    for v, x in sorted(found.items()):
        dirs = [r for _, r, _ in x["runs"]] + [EX/"deepseek-flash/test-reps"/f"{bench}__{me}__{r.name}" for me, r, _ in x["runs"]]
        test = pin(scan(dirs, v), bench, x["runs"][0][0], log)
        dev = list({execution(f)["signature"]: f for _, f in x["dev"] if execution(f)["panel_hash"] == panel}.values())
        if test and dev: out.append(dict(bench=bench, version12=v, method=x["runs"][0][0], runs=sorted({f"{me}/{r.name}" for me, r, _ in x["runs"]}), source=x["runs"][0][2], dev_files=[str(f) for f in dev],
                                        test_files={k: e["file"] for k, e in sorted(test.items())}, returned=sorted({f"{me}/{r.name}" for me, r, _ in x["runs"] if vid(json.load(open(r/"summary.json"))["final_skill"])[:12] == v})))
    return out, [e["file"] for e in sd]


def dev_test(m, extra, log):
    cand, summ = [], {}
    for bi, b in enumerate(BENCHES):
        cs, sdev = candidates(m, b, extra, log); seed = common.manifest_reps(b, "native-seed")
        if not seed: summ[b] = dict(status="pending: no seed row"); continue
        sf = [str(f) for _, f in sorted(seed.items())]; Sd, St, S0 = per_task(sdev), per_task(sf), per_task([sf[0]])
        for ci, c in enumerate(cs):
            rng = np.random.default_rng([SEED, 3, bi, ci])
            c["dev"] = compare(per_task(c["dev_files"]), Sd, rng); c["test_all"] = compare(per_task(list(c["test_files"].values())), St, rng)
            c["test_rep0"] = compare(per_task([c["test_files"]["0"]]), S0, rng) if "0" in c["test_files"] else None
            c["n_dev_exec"], c["n_test_exec"], c["n_seed_dev_exec"] = len(c["dev_files"]), len(c["test_files"]), len(sdev)
        cand += cs; summ[b] = sensitivity(cs)
    return cand, summ


def sensitivity(cs):
    """per bench: candidates, mean dev and test dQ / dlogX, mean test - dev (shrinkage), slope and correlation when n >= 3, with repetition and lineage variants."""
    def agg(sel, key):
        g = lambda c, s, k: c[key][s][k]["est"] if c[key] else None
        d = [(c["dev"]["dQ"]["est"], c[key]["dQ"]["est"], c["dev"]["dlogX"]["est"], c[key]["dlogX"]["est"]) for c in sel if c[key] and c[key]["dQ"]["est"] is not None]
        if not d: return dict(n=0)
        a = np.array(d, dtype=float); r = dict(n=len(a), dev_dQ=float(a[:, 0].mean()), test_dQ=float(a[:, 1].mean()), shrink_dQ=float((a[:, 1] - a[:, 0]).mean()),
                                               dev_dlogX=float(a[:, 2].mean()), test_dlogX=float(a[:, 3].mean()), shrink_dlogX=float((a[:, 3] - a[:, 2]).mean()))
        if len(a) >= 3 and a[:, 0].std() > 0: r["slope_dQ"] = float(np.polyfit(a[:, 0], a[:, 1], 1)[0]); r["corr_dQ"] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        else: r["slope_dQ"] = r["corr_dQ"] = "undefined (n<3 or no dev spread)"
        return r
    runs = sorted({r for c in cs for r in c["runs"]}); ret = [c for c in cs if c["returned"]]
    return dict(n_candidates=len(cs), main=agg(cs, "test_all"), test_rep0_only=agg(cs, "test_rep0"), one_per_run_returned_skill=agg(ret, "test_all"),
                leave_one_run_out={r: agg([c for c in cs if r not in c["runs"]], "test_all") for r in runs})


# ---- output ---------------------------------------------------------------------------------------------------------------------------------------------

def fmt(x, p=2): return "NA" if x is None else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.environ.get("BACH_SUBMISSION_MANIFEST")); ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--extra-run", action="append", default=[], help="explicit optimizer run dir <...>/<method>/<bench>/<run> whose dev/test evaluations join the candidate set")
    args = ap.parse_args(); os.environ["BACH_SUBMISSION_MANIFEST"] = args.manifest; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m = common.manifest(); log = []; meta = dict(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), manifest_created=m["created_utc"], seed=SEED, bootstrap=BOOT, extra_runs=args.extra_run)
    rows, floor, pend = rows_summary(m); cand, summ = dev_test(m, [Path(p) for p in args.extra_run], log)
    (out/"paired.json").write_text(json.dumps(dict(meta, rows=rows, seed_floor=floor, pending=pend), indent=1))
    (out/"dev_test.json").write_text(json.dumps(dict(meta, candidates=cand, summary=summ), indent=1, default=str))
    (out/"candidates.json").write_text(json.dumps([{k: c[k] for k in ("bench", "version12", "method", "runs", "source", "dev_files", "test_files", "returned")} for c in cand], indent=1, default=str))
    (out/"dedup_log.json").write_text(json.dumps(log, indent=1))
    L = ["# rows vs the seed's 3 test executions (dQ points, rho = X ratio, 95% bootstrap over tasks)"]
    for b, r in rows.items():
        L.append(f"[{b}] seed floor (execution vs other two) dQ " + ", ".join(f"{fmt(f['dQ']['est'])} [{fmt(f['dQ']['ci95'][0])},{fmt(f['dQ']['ci95'][1])}]" for f in floor[b].values()))
        for n, x in r.items():
            L.append(f"  {n:14s} " + (f"{x['status']}: {x.get('reason') or x.get('note')}" if x["status"] != "complete" else
                f"{x['executions']} exec, dQ {fmt(x['dQ']['est'])} [{fmt(x['dQ']['ci95'][0])},{fmt(x['dQ']['ci95'][1])}] rho {fmt(x['rho_X']['est'], 3)} [{fmt(x['rho_X']['ci95'][0], 3)},{fmt(x['rho_X']['ci95'][1], 3)}] "
                f"tasks Q/X {x['n_tasks_Q']}/{x['n_tasks_X']} W/L/T {x['wins']}/{x['losses']}/{x['ties']} missing/unknown/invalid {x['missing']}/{x['unknown']}/{x['invalid']}"))
    L.append("# dev -> test, frozen candidate set (success dQ and dlogX side by side)")
    for b, s in summ.items(): L.append(f"[{b}] " + json.dumps(s)[:900])
    for c in cand:
        t = c["test_all"]; L.append(f"  {c['bench']} {c['method']} {c['version12']} runs {c['runs']} dev dQ {fmt(c['dev']['dQ']['est'])} dlogX {fmt(c['dev']['dlogX']['est'], 3)} -> test({c['n_test_exec']} exec) dQ {fmt(t['dQ']['est'])} dlogX {fmt(t['dlogX']['est'], 3)}")
    (out/"paired.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
