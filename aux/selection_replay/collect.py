"""Per-task dev and test outcomes of the seed and of the returned skill, from each optimizer's own run (read-only).

For every optimizer run (SkillOpt, GEPA; DeepSeek native-20261008-https, Luna native-2026101x) the run itself evaluated the seed and the returned skill
on the dev panel and on the test panel.  A task 'gained' when the seed fails it and the returned skill solves it in that one run, 'lost' the other way.
The returned skill = the non-seed version whose test success equals summary.evaluated_panels.test.final.success; its dev evaluation has the same version id.
Run: python -m aux.selection_replay.collect
"""
import glob, json, os, pickle
from aux.selection_replay import selection as S

from src import layout

EXE = str(layout.EXECUTION)
OUT = str(layout.OUT / "selection_replay")
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]


def episodes(path):
    d = {}
    for e in json.load(open(path + "/evaluation.json"))["episodes"]:
        if S.verified(e):
            d[e["task_id"]] = int(bool(e["success"]))
    return d


def evals(run, split):
    out = {}
    for d in sorted(glob.glob(f"{run}/results/evaluations/{split}/*")):
        out.setdefault(os.path.basename(d).split("-")[0], []).append(d)
    return out


def pick_full(dirs, n):
    best = None
    for d in dirs:
        e = episodes(d)
        if best is None or len(e) > len(best[1]):
            best = (d, e)
    return best[1] if best and len(best[1]) >= 0.8 * n else None


def one(model, opt, bench):
    base = f"{EXE}/{model}/{opt}/{bench}"
    keep = {"deepseek-flash": ("native-20261008-https",), "gpt-6-luna": ("native-20261012",)}[model]
    runs = sorted(r for r in glob.glob(base + "/native-*") if os.path.exists(r + "/summary.json") and os.path.basename(r) in keep)
    res = []
    for run in runs:
        s = json.load(open(run + "/summary.json"))
        ep = s.get("evaluated_panels", {}).get("test", {})
        if "seed" not in ep or "final" not in ep:
            continue
        tests, devs = evals(run, "test"), evals(run, "dev")
        n_test, n_dev = s["panels"]["test"], s["panels"]["sel"]
        def vid_for(succ):
            cands = []
            for v, ds in tests.items():
                e = pick_full(ds, n_test)
                if e and abs(sum(e.values()) / max(len(e), 1) - succ) < 1e-6 and (len(e) == ep["final"]["known"] or True):
                    cands.append(v)
            return cands
        seed_c, fin_c = vid_for(ep["seed"]["success"]), vid_for(ep["final"]["success"])
        seed_c = [v for v in seed_c if v in devs]
        fin_c = [v for v in fin_c if v in devs]
        if len(seed_c) < 1 or len(fin_c) < 1:
            res.append(dict(run=os.path.basename(run), error=f"seed {seed_c} final {fin_c}", dev_versions=list(devs), test_versions=list(tests)))
            continue
        # prefer the version that is not the seed for 'final'; break ties with the best dev score the optimizer recorded
        fin = [v for v in fin_c if v not in seed_c] or fin_c
        extra = s.get("extra", {})
        best = extra.get("best_selection_hard")
        if best is None and extra.get("val_scores"):
            best = max(extra["val_scores"])
        if len(fin) > 1 and best is not None:
            match = []
            for v in fin:
                e = pick_full(devs[v], n_dev)
                if e and abs(sum(e.values()) / len(e) - best) < 0.5 / max(len(e), 1):
                    match.append(v)
            fin = match or fin
        sv, fv = seed_c[0], fin[0]
        d = dict(model=model, opt=opt, bench=bench, run=os.path.basename(run), seed=sv, final=fv, n_dev=n_dev, n_test=n_test,
                 dev_seed=pick_full(devs[sv], n_dev), dev_final=pick_full(devs[fv], n_dev),
                 test_seed=pick_full(tests[sv], n_test), test_final=pick_full(tests[fv], n_test), amb=(len(seed_c), len(fin_c)))
        res.append(d)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = []
    for model in ("deepseek-flash", "gpt-6-luna"):
        for opt in ("skillopt", "gepa"):
            for bench in BENCH:
                try:
                    r = one(model, opt, bench)
                except Exception as ex:
                    r = [dict(model=model, opt=opt, bench=bench, error=repr(ex))]
                data += r
    pickle.dump(data, open(OUT + "/devtest_data.pkl", "wb"))
    for d in data:
        if "error" in d:
            print("ERR", d.get("model"), d.get("opt"), d.get("bench"), d.get("run"), d["error"][:160]); continue
        def flips(a, b):
            t = [k for k in a if k in b]
            return sum(1 for k in t if a[k] == 0 and b[k] == 1), sum(1 for k in t if a[k] == 1 and b[k] == 0), len(t)
        dv = flips(d["dev_seed"], d["dev_final"]); ts = flips(d["test_seed"], d["test_final"])
        print("%-14s %-8s %-18s %-26s dev gained/lost %d/%d of %d | test gained/lost %d/%d of %d | amb %s" % (d["model"], d["opt"], d["bench"], d["run"], *dv, *ts, d["amb"]))
