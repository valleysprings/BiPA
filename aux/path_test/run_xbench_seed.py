"""Seed panels on the 1017 model on benchmarks outside the three of the paper (deepswe) and on TB2: the seed skill under opencode, a
deterministic spread of n train tasks (panels from harness_bench.baseline.panel_ids), k repetitions, fresh cache root; Q, $/episode and api
logs for analyze.py.  Out = execution/deepseek-flash/l2/<bench>/{seed|smoke}-<bench>-<tag>, cache l2-<same name>.
Run (host.env sourced; --plan prints the plan and calls nothing):
  python -u -m aux.path_test.run_xbench_seed --bench deepswe --n 30 [--smoke] [--plan]
"""
import argparse
import sys

from harness_bench.baseline import panel_ids
from harness_bench.eval import run, version_from_files

from src.layout import CACHE, EXECUTION, HARNESS_BENCH, SEED_FILE
from src.panels import CONFIG

SEED = SEED_FILE.read_text(encoding="utf-8")

MODEL, HARNESS = "deepseek-flash", "opencode"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["deepswe", "terminal-bench-2-1"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=57600)
    ap.add_argument("--tag", default="1017-20261021")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    train = sorted(panel_ids(a.bench, "train"))
    step = max(1, len(train) // a.n)
    tasks = train[a.offset::step][: a.n]
    rows = [{"task_id": t, "repetition": r} for t in tasks for r in range(a.reps)]
    name = f"{'smoke' if a.smoke else 'seed'}-{a.bench}-{a.tag}"
    OUT = (EXECUTION / MODEL / "l2" / a.bench / name).resolve()
    CACHE_ROOT = (CACHE / f"l2-{name}").resolve()
    state = lambda p: "absent" if not p.exists() else ("empty" if not any(p.iterdir()) else "NON-EMPTY")
    print(f"{a.bench}: train={len(train)} -> tasks={len(tasks)} reps={a.reps} episodes={len(rows)}; version "
          f"{version_from_files({'SKILL.md': SEED})['version_id'][:12]}\n  out={OUT} [{state(OUT)}]\n  cache={CACHE_ROOT} [{state(CACHE_ROOT)}]\n  tasks={tasks[:6]}", flush=True)
    if a.plan:
        sys.exit(0)
    if "NON-EMPTY" in (state(OUT), state(CACHE_ROOT)):
        sys.exit("refusing to start: output dir or cache root not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    b = CONFIG["backend"]
    res = run({"SKILL.md": SEED}, a.bench, "train", tasks=rows, model=MODEL, harness=HARNESS, out=OUT, slots=a.slots, cache_root=CACHE_ROOT,
              bench_root=str(HARNESS_BENCH), injection=b["injection"], infrastructure_retries=b["infrastructure_retries"], timeout_seconds=a.timeout)
    ok = [r for r in res.rows if r["success"] is not None]
    print(f"{a.bench}: {len(res.rows)} episodes, {len(ok)} known, success {sum(int(r['success']) for r in ok)}/{len(ok)}; evaluation {res.path}", flush=True)


if __name__ == "__main__":
    main()
