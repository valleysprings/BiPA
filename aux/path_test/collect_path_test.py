"""Collect the trajectory test passes into OUT/path_test/path_test.json (read by aux.selection_replay.fig_chance) and a plain summary.

For every run, the path skills (seed, intermediate, returned) with a test success: DeepSeek from this script's new-model passes (out/deepseek-flash/...; the new-model seed calibration is
out/deepseek-flash/seed__<bench>), Luna from the passes here plus the run's own seed and returned test evaluations (same model).  Test success = share of the panel's tasks with a verified
success; gain = points over the seed of the same model.  GEPA/SkillOpt x-position k = candidates scored on dev when the skill became the best: SkillOpt round number; GEPA index of the first
val_scores entry equal to the skill's dev score.
Run:  python -m aux.path_test.collect_path_test
"""
import glob, json, os
from aux.path_test import run_path_test as R
from src import layout

OUT = str(R.OUT)
HERE = str(layout.OUT / "path_test")


def success(path):
    d = json.load(open(path))
    eps = [e for e in d["episodes"] if type(e.get("success")) is bool and e.get("outcome_valid", True) is True]
    return (sum(e["success"] for e in eps) / len(eps), len(eps), len(d.get("panel", d["episodes"]))) if eps else None


def find(dirpat):
    f = sorted(glob.glob(dirpat))
    return f[0] if f else None


def k_of(model, opt, bench, entry):
    if opt == "skillopt":
        return entry["round"]
    summ = json.load(open(f"{R.EXECUTION}/{model}/{opt}/{bench}/{R.RUNS[model]}/summary.json"))["extra"]
    vals = summ.get("val_scores") or []
    dev = entry["dev"][0] / entry["dev"][1] if entry.get("dev") else None
    best = -1
    for i, v in enumerate(vals):
        if i and v > best + 1e-9 and dev is not None and abs(v - dev) < 1e-6:
            return i
        best = max(best, v)
    return None


def main():
    os.makedirs(HERE, exist_ok=True)
    result, detail = {}, []
    for model in ("deepseek-flash", "gpt-6-luna"):
        for opt in R.OPTIMIZERS:
            for bench in R.BENCHES:
                if (model, bench) in R.SKIP_CELLS:
                    continue
                cell = R.path_of(model, opt, bench, False)
                pts = []
                if model == "deepseek-flash":
                    sp = find(f"{OUT}/{model}/seed__{bench}/results/evaluations/test/{cell['entries'][0]['vid'][:12]}-*/evaluation.json")
                    seed = success(sp) if sp else None
                else:
                    seed = None
                    if cell["entries"][0].get("test"):
                        seed = success(str(cell["entries"][0]["test"][0]))
                for e in cell["entries"]:
                    role = e["role"]
                    if role in ("seed", "seed=returned"):
                        continue
                    p = find(f"{OUT}/{model}/{opt}__{bench}/results/evaluations/test/{e['vid'][:12]}-*/evaluation.json")
                    s = success(p) if p else None
                    if s is None and model == "gpt-6-luna" and e.get("test"):
                        s = success(str(e["test"][0]))
                    rec = dict(model=model, opt=opt, bench=bench, role=role, round=e["round"], vid=e["vid"][:12], dev=e.get("dev"), test=s, seed_test=seed)
                    if s and seed:
                        rec["gain_points"] = round(100 * (s[0] - seed[0]), 1)
                        rec["k"] = k_of(model, opt, bench, e)
                        pts.append([rec["k"], rec["gain_points"], role])
                    detail.append(rec)
                if pts:
                    result[f"{model}|{opt}|{bench}"] = pts
    json.dump(result, open(HERE + "/path_test.json", "w"), indent=1)
    json.dump(detail, open(HERE + "/path_test_detail.json", "w"), indent=1, default=str)
    for r in detail:
        print(r["model"][:4], r["opt"], r["bench"][:6], r["role"], "r%s" % r["round"], r["vid"], "dev", r["dev"], "test", r["test"], "seed", r["seed_test"], "gain", r.get("gain_points"), "k", r.get("k"))


if __name__ == "__main__":
    main()
