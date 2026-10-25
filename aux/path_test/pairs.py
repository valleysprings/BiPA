"""Paired test comparison of each new-model tau3 path skill with the new-model seed rerun: tasks gained / lost, difference and its paired standard error.
Run: python -m aux.path_test.pairs   (reads ARTIFACTS/path-test/out/deepseek-flash/*/results/evaluations/test/<vid12>-*/evaluation.json)"""
import glob, json, math
from src import layout

OUT = str(layout.ARTIFACTS / "path-test/out/deepseek-flash")
SKILLS = {"skillopt r1": ("skillopt__tau3-bench", "55f8a4a38828"), "skillopt r2": ("skillopt__tau3-bench", "45ca0b86e2b2"), "skillopt returned": ("skillopt__tau3-bench", "a0bdead535e7"),
          "gepa r8": ("gepa__tau3-bench", "43226f3d6a15"), "gepa r11": ("gepa__tau3-bench", "9fc436ada6e6"), "gepa returned": ("gepa__tau3-bench", "1e1cc8c08336")}


def load(d, vid):
    f = glob.glob(f"{OUT}/{d}/results/evaluations/test/{vid}-*/evaluation.json")
    if not f:
        return None
    return {e["task_id"]: bool(e["success"]) for e in json.load(open(f[0]))["episodes"] if e.get("outcome_valid", True) is True and type(e.get("success")) is bool}


def main():
    seed = load("seed__tau3-bench", "4fd5ca441c55")
    print("seed (new model) test", sum(seed.values()), "of", len(seed))
    rows = {}
    for name, (d, vid) in SKILLS.items():
        s = load(d, vid)
        if s is None:
            print(name, "not finished"); continue
        t = [k for k in seed if k in s]
        g = sum(1 for k in t if not seed[k] and s[k]); l = sum(1 for k in t if seed[k] and not s[k]); n = len(t)
        se = math.sqrt(g + l - (g - l) ** 2 / n) / n * 100
        rows[name] = dict(solved=sum(s.values()), n=n, gained=g, lost=l, diff_points=round(100 * (g - l) / n, 1), paired_se_points=round(se, 1))
        print(name, rows[name])
    (layout.OUT / "path_test").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(layout.OUT / "path_test/pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
