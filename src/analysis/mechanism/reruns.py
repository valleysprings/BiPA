"""Rerun checks of the elasticity analysis (DeepSeek-V4.1-Flash): the numbers behind the within-task slope table and the rerun paragraph of Sec. 3.4 and 5.

check: per-episode e_R = 1 + s_H R/(R-1) range, the within-task slope of log input tokens on log requests (task fixed effects on the main runs, plus skill fixed
effects, seed-only over its three runs, task x skill over all reruns), and the success rate of every run and pooled over the reruns.
seed spread: the spread of the seed's three runs in mean log input tokens (and tool calls) on the tasks all three completed; it is the criterion of the lever
test (tab:lever-test): a skill's change counts as real only beyond this spread.
Episodes: the test evaluations of the seed and of the optimizers' returned skills (main run, rep-1, rep-2); the three baseline optimizers only
(BiPA reports no BACH rows). Outputs: check.json, seed_spread.json.
"""
import argparse, collections, json, math
import numpy as np
from src.analysis.common import *
from src.graph import episode as W

MODEL = "deepseek-flash"
FIXED = {"system", "tool_schemas", "task", "skill"}  # the components every request re-sends
DS_ROOT = EX/MODEL

def fixed_part(we):
    """F: mean over requests of the input tokens in the fixed components (system, tool schemas, task, skill); first request's input if the graph has none."""
    idx = [i for i, c in enumerate(we.comps) if c in FIXED]
    return float(we.Xrg[:, idx].sum(1).mean()) if idx else float(we.req_in[0])

def rerun_files(bench, names=OPTIMIZERS):
    """{skill: [(run label, test evaluation file)]}: the main run of the seed and of each returned optimizer skill, plus its reruns
    (seed: rep-1/rep-2 beside the main run; optimizer skills: the repetitions under test-reps/<bench>__<method>__<run>, as table1.rep_files)."""
    from src.analysis.table1 import rep_files as t1_rep_files
    texts = {"seed": SEED_TEXT}; runs = {}
    main = {"seed": eval_file(current_run("native-seed", MODEL, bench), "test", vid(SEED_TEXT))}
    for disp, (run, t) in returned_skills(MODEL, bench, names).items():
        texts[disp] = t; runs[disp] = run; main[disp] = eval_file(run, "test", vid(t))
    reps = collections.defaultdict(list)
    for disp, f in main.items(): reps[disp].append(("main", f))
    for disp in texts:
        meth = "native-seed" if disp == "seed" else names[disp]
        if disp == "seed":
            for r in ("rep-1", "rep-2"):
                d = DS_ROOT/meth/bench/r
                if d.is_dir():
                    f = eval_file(d, "test", vid(texts[disp]))
                    if f: reps[disp].append((r, f))
        else:
            found = t1_rep_files([runs[disp], DS_ROOT/"test-reps"/f"{bench}__{meth}__{runs[disp].name}"], vid(texts[disp]))
            for rep, f in sorted(found.items()):
                if rep != 0: reps[disp].append((f"rep-{rep}", f))
    return reps

def metrics(ep):
    if ep.R < 2 or ep.N < 1: return None
    F = fixed_part(ep); X = ep.X
    if X - ep.R*F <= 0 or F <= 0: return None
    sF = ep.R*F/X
    return dict(R=ep.R, N=ep.N, X=X, sF=sF, eR=1+(1-sF)*ep.R/(ep.R-1), succ=ep.success)

def slope(rs, groups):
    """Within-group slope of log X on log R (groups = fixed effects) with its standard error."""
    x = np.log([m["R"] for *_, m in rs]); y = np.log([m["X"] for *_, m in rs])
    g = collections.defaultdict(list)
    for i, r in enumerate(rs): g[groups(r)].append(i)
    g = {k: i for k, i in g.items() if len(i) > 1}
    idx = [i for v in g.values() for i in v]
    xs, ys = x.copy(), y.copy()
    for ii in g.values(): xs[ii] -= x[ii].mean(); ys[ii] -= y[ii].mean()
    xs, ys = xs[idx], ys[idx]
    sl = float((xs*ys).sum()/(xs*xs).sum()); res = ys - sl*xs
    dof = len(xs) - len(g) - 1; se = float(np.sqrt((res**2).sum()/dof/(xs*xs).sum()))
    return dict(slope=round(sl, 3), se=round(se, 3), n=len(xs), groups=len(g))

def twoway(rs):
    """Slope with task and skill fixed effects (alternating projections)."""
    x = np.log([m["R"] for *_, m in rs]); y = np.log([m["X"] for *_, m in rs])
    T = [r[2] for r in rs]; S = [r[0] for r in rs]
    def demean(v):
        v = v.copy()
        for _ in range(200):
            for key in (T, S):
                g = collections.defaultdict(list)
                for i, k in enumerate(key): g[k].append(i)
                for ii in g.values(): v[ii] -= v[ii].mean()
        return v
    xs, ys = demean(x), demean(y)
    return round(float((xs*ys).sum()/(xs*xs).sum()), 3)

def check(names):
    out = {}
    for bench in BENCHES:
        reps = rerun_files(bench, names); recs = []
        for disp, lst in reps.items():
            for rn, f in lst:
                for t, e in {e.task: e for e in W.load(f)}.items():
                    m = metrics(e)
                    if m: recs.append((disp, rn, t, m))
        mainrecs = [r for r in recs if r[1] == "main"]
        eR = np.array([m["eR"] for *_, m in recs]); eRm = np.array([m["eR"] for *_, m in mainrecs])
        Rs = np.array([m["R"] for *_, m in mainrecs])
        out[bench] = dict(
            runs={k: [r for r, _ in l] for k, l in reps.items()},
            slope_main_taskFE=slope(mainrecs, lambda r: r[2]),
            slope_main_task_plus_skillFE=twoway(mainrecs),
            slope_seed_only_taskFE=slope([r for r in recs if r[0] == "seed"], lambda r: r[2]),
            slope_taskxskill_FE_reps=slope(recs, lambda r: (r[2], r[0])),
            eR_main=dict(min=round(float(eRm.min()), 3), max=round(float(eRm.max()), 3), n=len(eRm), above2=int((eRm > 2).sum()), below1=int((eRm < 1).sum())),
            eR_all=dict(min=round(float(eR.min()), 3), max=round(float(eR.max()), 3), n=len(eR), above2=int((eR > 2).sum())),
            R_main=dict(min=int(Rs.min()), median=float(np.median(Rs)), max=int(Rs.max())),
            sH_mean_main=round(float(np.mean([1-m["sF"] for *_, m in mainrecs])), 3))
        ps = {}   # success (%) of each run, then pooled over the runs
        for disp, lst in reps.items():
            q = []
            for rn, f in lst:
                d = json.loads(f.read_text()); eps = {}
                for e in d["episodes"]: eps.setdefault(e["task_id"], e)
                ys = [eps[t["task_id"]].get("success") for t in d["panel"] if t["task_id"] in eps]
                q.append(100*np.mean([y for y in ys if y is not None]))
            ps[disp] = [round(float(z), 1) for z in q] + ["pooled", round(float(np.mean(q)), 1)]
        out[bench]["success_runs"] = ps
    return out

def seed_spread():
    out = {}
    for bench in BENCHES:
        files = [eval_file(current_run("native-seed", MODEL, bench), "test", vid(SEED_TEXT))]
        for r in ("rep-1", "rep-2"): files.append(eval_file(DS_ROOT/"native-seed"/bench/r, "test", vid(SEED_TEXT)))
        runs = [{e.task: e for e in W.load(f)} for f in files]
        common_ = set.intersection(*[{t for t, e in r.items() if e.X and e.X > 0} for r in runs])
        means = [float(np.mean([math.log(r[t].X) for t in common_])) for r in runs]
        calls = [float(np.mean([math.log(max(r[t].N, 1)) for t in common_])) for r in runs]
        out[bench] = dict(n=len(common_), mean_logX=[round(m, 4) for m in means], spread_logX=round(max(means)-min(means), 4), spread_logN=round(max(calls)-min(calls), 4))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(out_dir("reruns")))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = check(OPTIMIZERS)
    (out/"check.json").write_text(json.dumps(c, indent=1)); print(json.dumps(c, indent=1))
    s = seed_spread()
    (out/"seed_spread.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s, indent=1))

if __name__ == "__main__": main()
