"""Main results table straight from the current run of every cell (no manifest, no certificates).

model x bench x method -> layout.current_run; a run with summary.json is scored on its test evaluation(s): Q = successes / known outcomes,
rho = geometric mean over tasks of c_i / c_i(seed) with c = N + X/1000 (rho_inv: N, rho_trj: X), each paired on the same task with its own n.
Seed reference: the native-seed cell's current run (native-noseed, native-seed); an optimizer run's own seed test evaluation (optimizers).
Cells not scored yet show their status (weekly_status.progress). Writes results/main-table.{md,csv} and results/per_task.csv."""
import csv, hashlib, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from src.layout import SKILLS, BASELINE_ROOT, RESULTS, current_run, relocate
from weekly_status import progress

MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = (("tau3-bench", "τ³"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2"))
METHODS = (("native-noseed", "none"), ("native-seed", "seed"), ("skillopt", "skillopt"), ("skilladam", "skilladam"), ("gepa", "gepa"))
METRICS = (("rho", "c"), ("rho_inv", "N"), ("rho_trj", "X"))


def version(text):
    """version_id of a skill text, as stored in evaluation.json."""
    files = {} if text is None else {"SKILL.md": text}
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def task_row(ep):
    """Exact captured totals over unique requests: N executed calls, X input tokens; None when the capture is incomplete."""
    cap, seen, reqs = ep.get("capture") or {}, set(), []
    for r in ep.get("requests", []):
        if r["request_id"] not in seen:
            seen.add(r["request_id"])
            reqs.append(r)
    calls = [c for r in reqs for c in r.get("tool_calls", [])]
    n = sum(c.get("executed") is True for c in calls)
    if cap.get("invocations_complete") is not True or any(c.get("executed") not in (True, False) for c in calls):
        n = None
    x = sum(r["input_tokens"] for r in reqs) if cap.get("requests_complete") is True and all(finite(r["input_tokens"]) for r in reqs) else None
    return {"success": ep.get("success"), "N": n, "X": x, "c": n + x / 1000 if n is not None and x is not None else None}


def evaluation(run, vid):
    """Rows of the newest test evaluation of `run` with this skill version, one per planned task (first episode wins)."""
    files = [f for f in (run / "results/evaluations/test").glob("*/evaluation.json") if f.parent.name.startswith(vid[:12])]
    if not files:
        return None
    path = max(files, key=lambda f: f.stat().st_mtime)
    data = json.loads(relocate(path).read_text())
    eps = {}
    for e in data["episodes"]:
        eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    return {(t["task_id"], t.get("repetition", 0)): task_row(eps.get((t["task_id"], t.get("repetition", 0)), {})) for t in data["panel"]}


def pair(rows, seed):
    out = {"known": sum(r["success"] is not None for r in rows.values()), "total": len(rows)}
    out["Q"] = 100 * sum(r["success"] is True for r in rows.values()) / out["known"] if out["known"] else None
    logs = {m: [] for m, _ in METRICS}
    for key, r in rows.items():
        b = seed.get(key) if seed else None
        r["seed"] = b
        ok = b is not None and r["success"] is not None and b["success"] is not None
        for name, k in METRICS:
            r[name] = math.log(r[k] / b[k]) if ok and r[k] and b[k] else None
            if r[name] is not None:
                logs[name].append(r[name])
    for name, _ in METRICS:
        out[name] = math.exp(sum(logs[name]) / len(logs[name])) if logs[name] else None
        out[name + "_n"] = len(logs[name])
    return out


def per_task(cell, final):
    """per_task.csv rows of a scored cell, each with its paired seed episode."""
    return [{"model": cell["model"], "bench": cell["bench"], "method": cell["method"], "task_id": k[0], "repetition": k[1],
             **{f: r[f] for f in ("success", "N", "X", "c", *(m for m, _ in METRICS))},
             **{"seed_" + f: (r["seed"] or {}).get(f) for f in ("success", "N", "X", "c")}} for k, r in final.items()]


def main():
    seed_text = (SKILLS / "seed/SKILL.md").read_text()
    seed_vid = version(seed_text)
    cells, tasks = [], []
    for model in MODELS:
        for bench, _ in BENCHES:
            seed_run = current_run("seed", model, bench)
            ref = evaluation(seed_run, seed_vid) if seed_run and (seed_run / "summary.json").is_file() else None
            for method, arm in METHODS:
                run = current_run(arm, model, bench)
                summary = json.loads((run / "summary.json").read_text()) if run and (run / "summary.json").is_file() else None
                cell = {"model": model, "bench": bench, "method": method, "run": run.name if run else "", "status": "done" if summary else progress(run, arm)}
                if summary:
                    final = evaluation(run, version(summary["final_skill"] if arm not in ("none", "seed") else seed_text if arm == "seed" else None))
                    base = {"none": ref, "seed": ref}.get(arm) if arm in ("none", "seed") else evaluation(run, seed_vid)
                    cell.update(pair(final, base))
                    tasks += per_task(cell, final)
                cells.append(cell)
    cols = ["model", "bench", "method", "run", "status", "known", "total", "Q"] + [f"{m}{s}" for m, _ in METRICS for s in ("", "_n")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (("main-table.csv", cells, cols), ("per_task.csv", tasks, list(tasks[0]))):
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fields, restval="")
            w.writeheader()
            w.writerows(rows)
    fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    lines = ["# Main results", "", "Generated by bipa/scripts/main_table.py from each cell's current run. Q = % success over known outcomes (known/total); "
             "ρ, ρ_inv, ρ_trj = per-task geometric-mean ratio of c = N + X/1000, N, X to the paired seed run, with paired n.", ""]
    for model in MODELS:
        lines += [f"## {model}", "", "| method | " + " | ".join(f"{b}: Q | ρ | ρ_inv | ρ_trj" for _, b in BENCHES) + " |", "|---|" + "---|" * 4 * len(BENCHES)]
        for method in [m for m, _ in METHODS]:
            row = []
            for bench, _ in BENCHES:
                c = next(c for c in cells if (c["model"], c["bench"], c["method"]) == (model, bench, method))
                if c["status"] != "done":
                    row += [c["status"], "", "", ""]
                else:
                    row += [f"{fmt(c['Q'], 1)} ({c['known']}/{c['total']})"] + [f"{fmt(c[m])} ({c[m + '_n']})" for m, _ in METRICS]
            lines.append(f"| {method} | " + " | ".join(row) + " |")
        lines.append("")
    (RESULTS / "main-table.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
