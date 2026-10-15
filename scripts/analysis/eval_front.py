"""Matched-workload evaluation of BACH-ZO against the baselines, offline on stored test evaluations (run with the slim venv python; calls no model).

Per cell (model x bench) with a finished zo run: points (Q, rho) on test for ours (every front node with a test evaluation) and for skillopt / gepa / skilladam
(final skill and the seed of their own run). Writes assets/artifacts/results/matched-workload.{md,json}:
  success at budgets rho <= 1.1, 1.25, 1.5 (best test point within the budget; ours picked on test, so post-hoc, marked * above 1.1)
  matched workload: ours' best front point with rho <= the baseline's rho, dQ with a paired bootstrap CI over test tasks
  hypervolume in (Q, -log rho), reference (Q = 0, rho = 1.5 | 2.0), bootstrap CI (tasks resampled jointly for all methods); no ranking when CIs overlap
  winner's curse: sel vs test for every node with both; the node ours' D_sel rule picks at each budget (LN.front_by_kappa)."""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "bach-harness/src")]
import main_table as M
from bach import lineage as LN
from src.layout import SKILLS, RESULTS, current_run

BUDGETS, REFS, NB = (1.1, 1.25, 1.5), (1.5, 2.0), 2000
OURS = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
seed_vid = M.version((SKILLS / "seed/SKILL.md").read_text())
done = lambda r: r and (r / "summary.json").is_file()


def point(label, rows, seed, **kw):
    p = M.pair(rows, seed)
    return {"label": label, "Q": p["Q"], "rho": p["rho"], "n": p["rho_n"], "rows": rows, **kw}


def hv(Q, rho, R):
    """Area of the union of [-log R, -log rho_i] x [0, Q_i] over the points with rho_i <= R."""
    pts = sorted((r, q) for q, r in zip(Q, rho) if r <= R)
    xs, m, a = [-math.log(r) for r, _ in pts] + [-math.log(R)], 0.0, 0.0
    for i, (_, q) in enumerate(pts): m = max(m, q); a += m * (xs[i] - xs[i + 1])
    return a


def best(pts, b): return max((p for p in pts if p["rho"] <= b), key=lambda p: (p["Q"], -p["rho"]), default=None)
def ci(x): return [float(v) for v in np.percentile(x, [2.5, 97.5])]
def f(v, d=3): return "-" if v is None else f"{v:.{d}f}"
def pq(p): return "-" if p is None else f"{p['Q']:.1f}/{p['rho']:.3f} ({p['label']})"


cells = {}
for model, tags in ((m, t) for m, t in OURS.items() if not m.startswith("_")):
    for bench, tag in tags.items():
        run = current_run("zo", model, bench)  # tag = ours_runs.json
        if not done(run): continue
        res = json.loads((run / "zo-result.json").read_text()); seed = M.evaluation(run, seed_vid)
        nodes = {n["id"]: n for n in res["nodes"]}
        ev = {i: M.evaluation(run, M.version(n["text"])) for i, n in nodes.items() if n["sel"]}
        M_ = {"ours": [point(i, ev[i], seed, node=i, returned=i == res["returned"]) for i in res["front"] if ev[i]]}
        for arm in ("skillopt", "gepa", "skilladam"):
            r = current_run(arm, model, bench)
            if not done(r): continue
            fin, bs = M.evaluation(r, M.version(json.loads((r / "summary.json").read_text())["final_skill"])), M.evaluation(r, seed_vid)
            if fin and bs: M_[arm] = [point("final", fin, bs), point("seed", bs, bs)]
        sel_pick = LN.front_by_kappa({i: SimpleNamespace(id=i, sel=n["sel"]) for i, n in nodes.items() if n["sel"]})
        cells[(model, bench)] = {"tag": tag, "methods": M_, "nodes": nodes, "ev": ev, "run": run, "seed": seed, "pick": sel_pick, "front": res["front"]}

rng = np.random.default_rng(0)
out, L = {}, []
for (model, bench), c in cells.items():
    keys = sorted({k for ps in c["methods"].values() for p in ps for k in p["rows"]})
    n = len(keys); idx = rng.integers(0, n, (NB, n))
    for ps in c["methods"].values():
        for p in ps:
            p["y"] = np.array([float(r["success"]) if (r := p["rows"].get(k)) and r["success"] is not None else np.nan for k in keys])
            p["r"] = np.array([r["rho"] if (r := p["rows"].get(k)) and r["rho"] is not None else np.nan for k in keys])
            p["bq"], p["br"] = np.nanmean(p["y"][idx], 1), np.exp(np.nanmean(p["r"][idx], 1))
    c["hv"] = {m: {R: (hv([p["Q"] / 100 for p in ps], [p["rho"] for p in ps], R), np.array([hv([p["bq"][b] for p in ps], [p["br"][b] for p in ps], R) for b in range(NB)])) for R in REFS} for m, ps in c["methods"].items()}
    out[f"{model}/{bench}"] = {"run": str(c["run"].relative_to(ROOT)), "n_test": n, "points": {m: [{k: p[k] for k in ("label", "Q", "rho", "n")} for p in ps] for m, ps in c["methods"].items()},
                               "budget": {}, "matched": {}, "hv": {}, "curse": [], "sel_pick": {str(k): v for k, v in c["pick"].items()}}
    o = out[f"{model}/{bench}"]
    for m, ps in c["methods"].items():
        o["budget"][m] = {str(b): (lambda p: p and {"Q": p["Q"], "rho": p["rho"], "node": p["label"], "post_hoc": m == "ours" and b > 1.1})(best(ps, b)) for b in BUDGETS}
        o["hv"][m] = {str(R): {"hv": v, "ci": ci(bs)} for R, (v, bs) in c["hv"][m].items()}
    for m, ps in c["methods"].items():
        if m == "ours": continue
        pb = ps[0]; po = best(c["methods"]["ours"], pb["rho"])
        d = None if po is None else {"base": {"Q": pb["Q"], "rho": pb["rho"]}, "ours": {"Q": po["Q"], "rho": po["rho"], "node": po["label"]}, "dQ": po["Q"] - pb["Q"]}
        if d: d["dQ_ci"] = ci(100 * (np.nanmean(po["y"][idx], 1) - np.nanmean(pb["y"][idx], 1)))
        o["matched"][m] = d
    for i, n_ in c["nodes"].items():
        if n_["sel"] and c["ev"][i] and i != "n0":
            p = M.pair(c["ev"][i], c["seed"])
            o["curse"].append({"node": i, "front": i in c["front"], "selQ": 100 * n_["sel"]["Q"], "testQ": p["Q"], "selL": n_["sel"]["L"], "se": n_["sel"]["se"], "testL": math.log(p["rho"]) if p["rho"] else None})

verdict = lambda a, b: "tie" if not (a[1] < b[0] or b[1] < a[0]) else "ours>" if a[0] > b[1] else "<ours"
L += ["# Matched-workload evaluation", "", "Generated by bipa/scripts/analysis/eval_front.py from stored test evaluations (no model call). Q = % success on test, rho = per-task geometric-mean "
      "cost ratio to the seed of the same run (main_table.pair). Cells with a finished zo run only. One run per cell; gaps under about 5 pp are noise (seed drift between runs).",
      "Ours = every front node of the archive with a test evaluation; the archive was grown under kappa = log 1.1. Choosing a node by test Q within a budget is post-hoc selection on test "
      "(optimistic for ours, which has up to 4 candidates per cell against 2 for the baselines); a * marks budgets above 1.1. Bootstrap: 2000 resamples of the test tasks, shared by all methods of a cell.", ""]
L += ["## 1. Points on test: Q / rho", "", "| cell | method | points (Q / rho, label) |", "|---|---|---|"]
for k, o in out.items():
    for m, ps in o["points"].items(): L.append(f"| {k} | {m} | " + "; ".join(f"{p['Q']:.1f} / {p['rho']:.3f} ({p['label']})" for p in ps) + " |")
L += ["", "## 2. Success at budget (best test point with rho <= b; baselines: final or seed)", "", "| cell | method | rho<=1.1 | rho<=1.25 | rho<=1.5 |", "|---|---|---|---|---|"]
for k, o in out.items():
    for m, bd in o["budget"].items():
        L.append(f"| {k} | {m} | " + " | ".join((lambda p, b: "-" if not p else f"{p['Q']:.1f} / {p['rho']:.3f} ({p['node']}){'*' if p['post_hoc'] else ''}")(bd[str(b)], b) for b in BUDGETS) + " |")
L += ["", "## 3. Matched workload: ours' best front point with rho <= baseline's rho (dQ in pp, paired bootstrap 95% CI; tie when the CI contains 0)", "",
      "| cell | baseline (Q / rho) | ours (Q / rho, node) | dQ [CI] | verdict |", "|---|---|---|---|---|"]
for k, o in out.items():
    for m, d in o["matched"].items():
        b = o["points"][m][0]
        if d is None: L.append(f"| {k} | {m} ({b['Q']:.1f} / {b['rho']:.3f}) | none within budget | - | - |"); continue
        lo, hi = d["dQ_ci"]; L.append(f"| {k} | {m} ({b['Q']:.1f} / {b['rho']:.3f}) | {d['ours']['Q']:.1f} / {d['ours']['rho']:.3f} ({d['ours']['node']}) | {d['dQ']:+.1f} [{lo:+.1f}, {hi:+.1f}] | {'tie' if lo <= 0 <= hi else 'ours' if lo > 0 else m} |")
L += ["", "## 4. Hypervolume in (Q, -log rho), Q as a fraction; reference (0, rho = R); bootstrap 95% CI; verdict vs ours only when the CIs are disjoint", "",
      "| cell | method | HV R=1.5 [CI] | vs ours | HV R=2.0 [CI] | vs ours |", "|---|---|---|---|---|---|"]
for k, o in out.items():
    for m, h in o["hv"].items():
        L.append(f"| {k} | {m} | " + " | ".join(f"{h[str(R)]['hv']:.3f} [{h[str(R)]['ci'][0]:.3f}, {h[str(R)]['ci'][1]:.3f}] | {'-' if m == 'ours' else verdict(o['hv']['ours'][str(R)]['ci'], h[str(R)]['ci'])}" for R in REFS) + " |")
L += ["", "## 5. Winner's curse: D_sel vs test for every non-seed node with both (L = log rho; se = sel SE of L; n0 is 0 by construction)", "",
      "| cell | node | front | sel Q | test Q | dQ | sel L | se | test L | dL |", "|---|---|---|---|---|---|---|---|---|---|"]
dq, dl, z = [], [], []
for k, o in out.items():
    for r in o["curse"]:
        g = r["testL"] - r["selL"] if r["testL"] is not None else None
        dq.append(r["testQ"] - r["selQ"]); dl += [g] * (g is not None); z += [g / r["se"]] * (g is not None and r["se"] > 0)
        L.append(f"| {k} | {r['node']} | {'yes' if r['front'] else ''} | {r['selQ']:.1f} | {r['testQ']:.1f} | {dq[-1]:+.1f} | {f(r['selL'])} | {f(r['se'])} | {f(r['testL'])} | {f(g)} |")
rms = lambda x: math.sqrt(np.mean(np.square(x))) if len(x) else None
out["_curse"] = {"n": len(dq), "mean_dQ": float(np.mean(dq)), "rms_dL": rms(dl), "mean_dL": float(np.mean(dl)), "rms_z": rms(z)}
L += ["", f"n = {len(dq)} nodes: mean test-sel dQ {np.mean(dq):+.1f} pp, mean dL {np.mean(dl):+.3f}, RMS dL {f(rms(dl))}, RMS dL/se {f(rms(z), 2)} (1 = pure noise from the sel SE).", "",
      "## 6. Node that D_sel picks at each budget (LN.front_by_kappa), with its test point (post-hoc above 1.1)", "", "| cell | rho<=1.1 | rho<=1.25 | rho<=1.5 |", "|---|---|---|---|"]
for (model, bench), c in cells.items():
    pt_ = lambda i: (lambda p: f"{i}: " + (f"{M.pair(c['ev'][i], c['seed'])['Q']:.1f} / {p['rho']:.3f}" if p else "no test eval"))(M.pair(c["ev"][i], c["seed"]) if c["ev"][i] else None)
    L.append(f"| {model}/{bench} | " + " | ".join(pt_(c["pick"][b]) for b in BUDGETS) + " |")
(RESULTS / "matched-workload.md").write_text("\n".join(L) + "\n")
(RESULTS / "matched-workload.json").write_text(json.dumps(out, indent=1, default=float))
print("\n".join(L))
