"""Shared loader: DeepSeek Table 1 test runs (outcomes per task x run) and the seed's dev/test executions.
Run: python -m aux.noise_and_domains.load (read-only)."""
import pickle, os
import numpy as np
from aux.noise_and_domains import domains as D
from src import layout
from aux.selection_replay import selection as S
from src.analysis import table1 as T

BENCHES = ["terminal-bench-2-1", "tau3-bench", "automationbench"]
ARMS = [("Seed", "native-seed"), ("No skill", "native-noseed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam")]
OUT = str(layout.OUT / "noise_and_domains")
CACHE = os.path.join(OUT, "data.pkl")

def domain(bench, t):
    if bench == "terminal-bench-2-1":
        return "all"
    return D.domain_of(bench, t)

def load():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    out = {}
    for bench in BENCHES:
        arms = {}
        for name, m in ARMS:
            files = D.cell_runs(name, m, bench)
            runs = {}
            meta = {}
            for rep, f in sorted(files.items()):
                rows = T.rows_of(f)
                runs[rep] = {task: (row["success"] is True) for (task, r), row in rows.items() if row["success"] is not None}
                meta[rep] = {"file": str(f), "mtime": os.path.getmtime(f), "suffix": f.parent.name.split("-")[-1]}
            arms[name] = {"runs": runs, "meta": meta}
        dev, test, _ = S.seed_executions(bench)
        out[bench] = {"arms": arms, "seed_dev": dev, "seed_test_exec": test}
    pickle.dump(out, open(CACHE, "wb"))
    return out

if __name__ == "__main__":
    d = load()
    import datetime
    for bench, c in d.items():
        print(bench, "dev tasks", len(c["seed_dev"]), "test exec tasks", len(c["seed_test_exec"]))
        for name, a in c["arms"].items():
            print("  ", name, {rep: (len(r), round(100*np.mean(list(r.values())),1), a["meta"][rep]["suffix"], datetime.datetime.fromtimestamp(a["meta"][rep]["mtime"]).strftime("%m-%d %H:%M")) for rep, r in a["runs"].items()})
