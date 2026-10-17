"""Reproducibility of cost and success across reruns (paragraph 'Success gains do not survive reruns', appendix reproducibility table): seed x3 as noise floor,
SkillAdam tau3 / SkillOpt AutoB and TB2 x3 reruns, geometric-mean input-token ratios rho_X, rho_R of every run against the seed (first run and pooled seed runs),
success differences, pairwise flip fraction and log-X / log-R correlation across a skill's own reruns. Also holds the run loaders analysis/mechanism/pooled.py shares.
Output: assets/cache/analysis/reproducibility/reproducibility.txt. Run: <bach-harness venv python> -m src.analysis.mechanism.reproducibility   (from bipa/)"""
import itertools, json, math
import numpy as np
from src.analysis.common import BENCHES, DS, SEED_TEXT, Report, current_run, gmean, eval_file, returned_skills, vid

NAMES = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}


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


def run_files(bench, noskill=False):
    """{display name: [(run name, evaluation.json)]} on the test split: the seed (main run + reruns), optionally no skill, and each optimizer's returned skill (main + reruns)."""
    res = {"seed": [("main", eval_file(current_run("native-seed", "deepseek-flash", bench), "test", vid(SEED_TEXT)))]}
    for r in ("rep-1", "rep-2"):
        d = DS/"native-seed"/bench/r
        if d.is_dir(): res["seed"].append((r, eval_file(d, "test", vid(SEED_TEXT))))
    if noskill: res["no skill"] = [("main", eval_file(current_run("native-noseed", "deepseek-flash", bench), "test", vid(None)))]
    for disp, (run, t) in returned_skills("deepseek-flash", bench, NAMES).items():
        L = [("main", eval_file(run, "test", vid(t)))]
        for r in ("rep-1", "rep-2"):
            d = DS/NAMES[disp]/bench/r
            if d.is_dir(): L.append((r, eval_file(d, "test", vid(t))))
        res[disp] = L
    return {k: [(n, f) for n, f in v if f] for k, v in res.items()}


def gm(a, b, tasks, key="X"):
    l = [math.log(a[t][key] / b[t][key]) for t in tasks if a[t][key] > 0 and b[t][key] > 0]
    return gmean(np.exp(l)), len(l)


def main():
    say = Report(); spread = {}; rawf = []
    for bench in BENCHES:
        D = {}
        for s, l in run_files(bench).items():
            for rn, f in l: D[(s, rn)] = load(f); rawf.append(f)
        tasks = sorted(set.intersection(*[set(v) for v in D.values()]))
        say("\n==", bench, "tasks", len(tasks), "runs", list(D))
        seedruns = [k for k in D if k[0] == "seed"]
        succ = lambda k: 100 * np.mean([D[k][t]["succ"] for t in tasks])
        pooled = {t: dict(X=gmean([D[k][t]["X"] for k in seedruns]), R=gmean([D[k][t]["R"] for k in seedruns])) for t in tasks}
        ss = []
        for a, b in itertools.combinations(seedruns, 2):
            rho, _ = gm(D[a], D[b], tasks); ds = succ(a) - succ(b)
            ss.append((a[1], b[1], rho, ds))
            say(f"seed {a[1]} vs {b[1]}: rho_X={rho:.3f} dsucc={ds:+.1f}")
        srho = [abs(math.log(x[2])) for x in ss]
        ks_ = [k for k in D if k[0] == "seed"]
        spread[bench] = dict(flip=float(np.mean([np.mean([D[a][t]["succ"] != D[b][t]["succ"] for t in tasks]) for a, b in itertools.combinations(ks_, 2)])),
                             anyflip=float(np.mean([len(set(D[k][t]["succ"] for k in ks_)) > 1 for t in tasks])),
                             span=round(max(round(succ(k), 1) for k in ks_) - min(round(succ(k), 1) for k in ks_), 1), xmax=math.exp(max(srho)))
        say("seed floor |log rho| max", max(srho), "rho range", min(x[2] for x in ss), max(x[2] for x in ss), "dsucc abs range", min(abs(x[3]) for x in ss), max(abs(x[3]) for x in ss))
        say("seed success runs", [round(succ(k), 1) for k in seedruns])
        say("skill | run | succ | rho_X vs seed(main) | rho_X vs pooled | rho_R vs pooled | dsucc vs seed main | dsucc vs pooled mean")
        pm = np.mean([succ(k) for k in seedruns])
        for k in D:
            rx, _ = gm(D[k], pooled, tasks); rr, _ = gm(D[k], pooled, tasks, "R")
            rm, _ = gm(D[k], D[("seed", "main")], tasks)
            say(f"{k[0]:10s} {k[1]:6s} {succ(k):5.1f} rhoX_main={rm:.3f} rhoX_pool={rx:.3f} rhoR_pool={rr:.3f} dS_main={succ(k)-succ(('seed','main')):+.1f} dS_pool={succ(k)-pm:+.1f}")
        for s in sorted(set(k[0] for k in D)):
            ks = [k for k in D if k[0] == s]
            if len(ks) < 2: continue
            flips = []; cx = []; cr = []
            for a, b in itertools.combinations(ks, 2):
                flips.append(np.mean([D[a][t]["succ"] != D[b][t]["succ"] for t in tasks]))
                cx.append(np.corrcoef([math.log(D[a][t]["X"]) for t in tasks], [math.log(D[b][t]["X"]) for t in tasks])[0, 1])
                cr.append(np.corrcoef([math.log(D[a][t]["R"]) for t in tasks], [math.log(D[b][t]["R"]) for t in tasks])[0, 1])
            anyflip = np.mean([len(set(D[k][t]["succ"] for k in ks)) > 1 for t in tasks])
            say(f"{s:10s} reruns={len(ks)} pairwise flip frac {np.mean(flips):.3f} (any-flip {anyflip:.3f}) corr logX {np.mean(cx):.3f} [{min(cx):.3f},{max(cx):.3f}] corr logR {np.mean(cr):.3f}")
    say.save("reproducibility", "reproducibility.txt")


if __name__ == "__main__":
    main()
