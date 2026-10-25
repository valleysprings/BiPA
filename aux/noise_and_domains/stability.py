"""Stability of cost and success across reruns (paragraph 'Success gains do not survive reruns' and the appendix reproducibility table). Two subcommands:

  pooled            pooled dual-currency table: per optimizer x benchmark, success pooled over the three runs (main + two reruns) and the pooled geometric-mean
                    ratio to the seed of input tokens X, price P (cached 0.1, output 4), turns R, executed calls N and output tokens O, with the seed's own
                    run-to-run floor (pairwise and leave-one-out ratios) -> assets/cache/analysis/pooled/pooled.json and pooled.txt
  reproducibility   reproducibility across reruns: seed x3 as noise floor, SkillAdam tau3 / SkillOpt AutoB and TB2 x3 reruns, geometric-mean input-token ratios
                    rho_X, rho_R of every run against the seed (first run and pooled seed runs), success differences, pairwise flip fraction and log-X / log-R
                    correlation across a skill's own reruns -> assets/cache/analysis/reproducibility/reproducibility.txt

The run loaders (load, run_files, gm) are shared by both subcommands and by appb.py.
Run: python -m aux.noise_and_domains.stability {pooled|reproducibility}"""
import argparse, itertools, json, math
import numpy as np
from src.utils.dataloader import BENCHES, BN, DS, SEED_TEXT, current_run, eval_file, gmean, out_dir, returned_skills, vid
from src.utils.logger import Report

NAMES = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}


# ---- shared loaders (also used by appb.py) ---------------------------------------------------------------------------------------------


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


# ---- pooled ------------------------------------------------------------------------------------------------------------------------------

KEYS = ["X", "P", "R", "N", "O"]


def gm_ratio(A, Bs, tasks, k):
    """geometric-mean over tasks of (per-task geometric mean over runs in A) / (same over Bs)."""
    d = []
    for t in tasks:
        a = [r[t][k] for r in A if t in r and r[t][k] > 0]; b = [r[t][k] for r in Bs if t in r and r[t][k] > 0]
        if a and b: d.append(sum(map(math.log, a)) / len(a) - sum(map(math.log, b)) / len(b))
    return gmean(np.exp(d)), len(d)


def pooled():
    out, rawf = {}, []
    for short, bench in ((v, k) for k, v in BN.items()):
        F = run_files(bench, noskill=True); rawf += [f for v in F.values() for n, f in v]; R = {k: [load(f) for n, f in v] for k, v in F.items()}
        seed = R["seed"]; st = set.intersection(*[set(r) for r in seed])
        floor = {}
        for k in KEYS:
            rs = [gm_ratio([seed[j]], [seed[i]], st, k)[0] for i in range(len(seed)) for j in range(i + 1, len(seed))]
            loo = [gm_ratio([seed[i]], [s for j, s in enumerate(seed) if j != i], st, k)[0] for i in range(len(seed))] if len(seed) > 2 else []
            floor[k] = dict(pairs=[round(x, 3) for x in rs], max_dev=round(max(abs(math.log(x)) for x in rs), 3) if rs else None, loo=[round(x, 3) for x in loo])
        rows = {}
        for name, runs in R.items():
            tasks = set.intersection(*[set(r) for r in runs])
            s = [sum(r[t]["succ"] for t in r) / len(r) for r in runs]
            row = dict(runs=len(runs), run_names=[n for n, f in F[name]], n_tasks=len(runs[0]), succ_runs=[round(100 * x, 1) for x in s], succ_pooled=round(100 * sum(s) / len(s), 1))
            if name != "seed":
                for k in KEYS:
                    v, n = gm_ratio(runs, seed, tasks & st, k); row["ratio_" + k] = round(v, 3); row["n_paired"] = n
            rows[name] = row
        out[short] = dict(rows=rows, seed_floor=floor, seed_run_means={k: round(sum(sum(r[t][k] for t in r) / len(r) for r in seed) / len(seed)) for k in KEYS})
    json.dump(out, open(out_dir("pooled")/"pooled.json", "w"), indent=1)
    say = Report()
    for b, v in out.items():
        say("##", b, "floor", {k: (f["pairs"], f["max_dev"]) for k, f in v["seed_floor"].items() if k in "XPRN"})
        for n, r in v["rows"].items(): say(n, r)
    say.save("pooled", "pooled.txt")


# ---- reproducibility ---------------------------------------------------------------------------------------------------------------------


def reproducibility():
    say = Report(); spread = {}; rawf = []
    for bench in BENCHES:
        D = {}
        for s, l in run_files(bench).items():
            for rn, f in l: D[(s, rn)] = load(f); rawf.append(f)
        tasks = sorted(set.intersection(*[set(v) for v in D.values()]))
        say("\n==", bench, "tasks", len(tasks), "runs", list(D))
        seedruns = [k for k in D if k[0] == "seed"]
        succ = lambda k: 100 * np.mean([D[k][t]["succ"] for t in tasks])
        pooled_ = {t: dict(X=gmean([D[k][t]["X"] for k in seedruns]), R=gmean([D[k][t]["R"] for k in seedruns])) for t in tasks}
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
            rx, _ = gm(D[k], pooled_, tasks); rr, _ = gm(D[k], pooled_, tasks, "R")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["pooled", "reproducibility"])
    a = ap.parse_args()
    (pooled if a.cmd == "pooled" else reproducibility)()


if __name__ == "__main__":
    main()
