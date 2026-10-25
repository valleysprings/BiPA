"""Every test run of the seed and of the returned skills: success and input-token ratio against the pooled seed (the rerun figure's numbers).
  python -m aux.noise_and_domains.appb   -> CACHE/analysis/appb/reruns.json
Reruns: the runs are pinned below (the https reruns `native-20261008-https`, the same runs as Table 1;
each skill's three test executions are the run's own evaluation plus the two under `test-reps/<bench>__<method>__<run>`); per-run rho_X is the
geometric-mean ratio against the pooled seed (per-task geometric mean over the seed's three runs), a skill's pooled rho_X pools its own runs
the same way (gm_ratio)."""
import json, math
import numpy as np
from src.utils.dataloader import gmean

BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
SEED_RUN = "native-20261003-trace-final"
PINS = {"tau3-bench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "automationbench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "terminal-bench-2-1": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")}}

# frozen values of the rerun record; compute asserts against them, nothing here re-derives them
FROZEN = dict(
    seed_succ={"tau3-bench": [78.7, 76.0, 74.7], "automationbench": [40.8, 49.2, 45.0], "terminal-bench-2-1": [72.2, 77.8, 77.8]},
    rerun_skill={"tau3-bench": "SkillAdam", "automationbench": "SkillOpt", "terminal-bench-2-1": "SkillOpt"},
    rerun_succ={"tau3-bench": [74.7, 77.3, 73.3], "automationbench": [43.3, 40.8, 40.8], "terminal-bench-2-1": [66.7, 66.7, 61.1]},
    seed_span={"tau3-bench": 4.0, "automationbench": 8.4, "terminal-bench-2-1": 5.6},
    seed_spread_X={"tau3-bench": 1.05, "automationbench": 1.14, "terminal-bench-2-1": 1.04},
    flip={"tau3-bench": 14, "automationbench": 12, "terminal-bench-2-1": 4},
    anyflip={"tau3-bench": 21, "automationbench": 18, "terminal-bench-2-1": 6},
    pooled_ratio={("SkillAdam", "tau3-bench"): 1.08, ("SkillOpt", "automationbench"): 1.26, ("SkillOpt", "terminal-bench-2-1"): 1.48},
    single_ratio={("SkillAdam", "automationbench"): 1.08},
)


def out_dir():
    from src.utils.dataloader import out_dir as od
    d = od("appb"); d.mkdir(parents=True, exist_ok=True); return d


def price(reqs):
    """input tokens priced at uncached 1, cached 0.1, plus output 4 per token."""
    tot = 0.0
    for r in reqs:
        X = r["input_tokens"]; res = min((r.get("captured_usage") or {}).get("cache_tokens") or 0, X)
        tot += (X - res) + 0.1 * res + 4 * r["output_tokens"]
    return tot


def load(f):
    """{task: succ, input tokens X, output tokens O, turns R, executed calls N, price P} of an evaluation.json (first episode per task, duplicate requests dropped)."""
    eps = {}
    for e in json.load(open(f))["episodes"]: eps.setdefault(e["task_id"], e)
    out = {}
    for t, e in eps.items():
        seen = set(); reqs = []
        for r in e.get("requests", []):
            if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
        calls = [c for r in reqs for c in r.get("tool_calls", [])]
        out[t] = dict(succ=1.0 if e.get("success") else 0.0, X=sum(r["input_tokens"] for r in reqs), O=sum(r["output_tokens"] for r in reqs),
                      R=len(reqs), N=sum(c.get("executed") is True for c in calls), P=price(reqs))
    return out


def gm(a, b, tasks, key="X"):
    l = [math.log(a[t][key] / b[t][key]) for t in tasks if a[t][key] > 0 and b[t][key] > 0]
    return gmean(np.exp(l)), len(l)


def gm_ratio(A, Bs, tasks, k):
    """geometric-mean over tasks of (per-task geometric mean over runs in A) / (same over Bs)."""
    d = []
    for t in tasks:
        a = [r[t][k] for r in A if t in r and r[t][k] > 0]; b = [r[t][k] for r in Bs if t in r and r[t][k] > 0]
        if a and b: d.append(sum(map(math.log, a)) / len(a) - sum(map(math.log, b)) / len(b))
    return gmean(np.exp(d)), len(d)


# ---- compute ---------------------------------------------------------------------------------------------------------------------------------

def compute_reruns():
    import itertools
    from src.utils.dataloader import DS, SEED_TEXT, eval_file, final_skill, vid
    from src.analysis.table1 import rep_files
    out, raw = {}, []
    for bench, _ in BENCH:
        F = {"seed": [("main", eval_file(DS / "native-seed" / bench / SEED_RUN, "test", vid(SEED_TEXT)))]}
        for r in ("rep-1", "rep-2"):
            F["seed"].append((r, eval_file(DS / "native-seed" / bench / r, "test", vid(SEED_TEXT))))
        for disp, (meth, rn) in PINS[bench].items():
            run = DS / meth / bench / rn; t = final_skill(meth, run)
            assert t != SEED_TEXT, (bench, disp)
            reps = rep_files([run, DS / "test-reps" / f"{bench}__{meth}__{rn}"], vid(t))
            L = [("main" if rep == 0 else f"rep-{rep}", f) for rep, f in sorted(reps.items())]
            F[disp] = L
        D = {}
        for s, l in F.items():
            for rn, f in l: assert f, (bench, s, rn); D[(s, rn)] = load(f); raw.append(str(f))
        tasks = sorted(set.intersection(*[set(v) for v in D.values()]))
        seedruns = [k for k in D if k[0] == "seed"]
        succ = lambda k: 100 * float(np.mean([D[k][t]["succ"] for t in tasks]))
        pooled = {t: dict(X=gmean([D[k][t]["X"] for k in seedruns])) for t in tasks}
        skills = {}
        for s in sorted(set(k[0] for k in D)):
            ks = [k for k in D if k[0] == s]
            row = dict(runs=[k[1] for k in ks], succ=[round(succ(k), 1) for k in ks], rho_pool=[round(gm(D[k], pooled, tasks)[0], 4) for k in ks])
            if s != "seed":
                row["rho_pooled"] = round(gm_ratio([D[k] for k in ks], [D[k] for k in seedruns], tasks, "X")[0], 4)
            skills[s] = row
        pairs = [gm(D[a], D[b], tasks)[0] for a, b in itertools.combinations(seedruns, 2)]
        flips = [float(np.mean([D[a][t]["succ"] != D[b][t]["succ"] for t in tasks])) for a, b in itertools.combinations(seedruns, 2)]
        anyflip = float(np.mean([len(set(D[k][t]["succ"] for k in seedruns)) > 1 for t in tasks]))
        out[bench] = dict(n_tasks=len(tasks), skills=skills,
                          seed=dict(succ=skills["seed"]["succ"], rho_pool=skills["seed"]["rho_pool"],
                                    spread_X=round(math.exp(max(abs(math.log(x)) for x in pairs)), 4),
                                    flip=round(float(np.mean(flips)), 4), anyflip=round(anyflip, 4),
                                    span=round(max(skills["seed"]["succ"]) - min(skills["seed"]["succ"]), 1)))
        # frozen-value checks of the seed spreads
        sd = out[bench]["seed"]
        assert sd["succ"] == FROZEN["seed_succ"][bench], (bench, sd["succ"])
        assert sd["span"] == FROZEN["seed_span"][bench], (bench, sd["span"])
        assert round(sd["spread_X"], 2) == FROZEN["seed_spread_X"][bench], (bench, sd["spread_X"])
        assert round(100 * sd["flip"]) == FROZEN["flip"][bench] and round(100 * sd["anyflip"]) == FROZEN["anyflip"][bench], bench
        rs = FROZEN["rerun_skill"][bench]
        assert skills[rs]["succ"] == FROZEN["rerun_succ"][bench], (bench, skills[rs]["succ"])
        assert round(skills[rs]["rho_pooled"], 2) == FROZEN["pooled_ratio"][(rs, bench)], (bench, skills[rs]["rho_pooled"])
    assert round(out["automationbench"]["skills"]["SkillAdam"]["rho_pool"][0], 2) == FROZEN["single_ratio"][("SkillAdam", "automationbench")]
    f = out_dir() / "reruns.json"; f.write_text(json.dumps(dict(benches=out, raw=sorted(set(raw))), indent=1)); print("wrote", f)
    return out


def compute():
    compute_reruns()


OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}  # optimizer colors, also used by selection_fig


if __name__ == "__main__":
    compute()
