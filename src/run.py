"""Optimize, repeat and transfer skills. Every command prints what it would run; --execute runs it.

  python -m src.run run METHOD BENCH MODEL         optimize (seed, none: no optimizer) and test the final skill
  python -m src.run repeat METHOD BENCH MODEL      complete avg@3 test repetitions of a run's final skill
  python -m src.run transfer METHOD BENCH MODEL --to MODEL|BENCH   test a returned skill unchanged elsewhere
  python -m src.run suite MODEL                    run, then repeat, every method on every benchmark
"""
import argparse, fcntl, importlib, json, os, shlex, subprocess, sys
from collections import Counter

from harness_bench.eval import version_from_files

from src.executor import SEED, Cell, evaluate, record_round
from src.layout import CACHE, EXECUTION, METHOD_DIR, MODELS, ROOT, relocate, repetition_dir, run_dir
from src.panels import BENCHMARKS, batches, panels

METHODS = ("seed", "none", "skillopt", "gepa", "skilladam")
HARNESS = "opencode"
ENV = {"HARNESS_CAPTURE_PROTOCOL": "native-v1"}  # harness-bench's request capture; part of its episode cache key
ALIASES = {spec["alias"]: name for name, spec in BENCHMARKS.items()}
ARM_OPTIONS = {"skilladam": {"gate": "train-batch"}}  # recorded in native_protocol.json and summary.json


def episodes(out):
    """Per split: episodes the evaluations consumed and of them executed (not cache hits), retries included."""
    counts = {}
    for path in sorted((out / "results/evaluations").rglob("evaluation.json")):
        accounting = json.loads(path.read_text(encoding="utf-8"))["slot_accounting"]
        entry = counts.setdefault(path.relative_to(out / "results/evaluations").parts[0], {"episodes": 0, "executed": 0})
        entry["episodes"] += accounting["slots"]
        entry["executed"] += accounting["executed"]
    return counts


def score(rows):
    """Mean success and mean cost over the tasks whose value is known."""
    known = [row["success"] for row in rows if row["success"] is not None]
    costs = [row["cost"] for row in rows if row["cost"] is not None]
    return {"success": sum(known) / len(known) if known else None, "known": len(known), "tasks": len(rows),
            "outcomes_complete": bool(rows) and len(known) == len(rows),
            "cost": sum(costs) / len(costs) if costs else None}


def cache_root(args, out):
    """The run's recorded cache root on a resume (--cache-root must agree), else --cache-root or the model's."""
    given = (CACHE / args.cache_root).resolve() if args.cache_root else None
    if (out / "native_protocol.json").is_file():
        recorded = relocate(json.loads((out / "native_protocol.json").read_text())["cache_root"]).resolve()
        if given and given != recorded:
            raise SystemExit(f"{out} was started with cache root {recorded}, not {given}")
        return recorded
    return given or (CACHE / MODELS[args.model]["cache_root"]).resolve()


def started(out):
    """A started run's native_protocol.json with the keys older runs lack."""
    spec = json.loads((out / "native_protocol.json").read_text())
    return {"seed": SEED, **spec, "arm_options": spec.get("arm_options") or {}}


def protocol(args, out, cache):
    """native_protocol.json: written when a run starts; a resume must match it and keeps its panels and batches."""
    spec = {"protocol": "native-v1", "feedback_mode": "full", "arm": args.method, "model": args.model,
            "benchmark": args.bench, "harness": HARNESS, "scale": args.scale, "cache_root": str(cache),
            "arm_options": ARM_OPTIONS.get(args.method, {}), "seed": SEED}
    marker = out / "native_protocol.json"
    if marker.is_file():
        recorded = started(out)
        if args.method == "skilladam" and recorded["arm_options"].get("gate") != "train-batch":
            raise SystemExit(f"{out} was started with the dev-batch gate, which this code does not run; "
                             "resume it with the original code")
        recorded["cache_root"] = str(relocate(recorded["cache_root"]).resolve())
        if changed := [key for key in spec if recorded[key] != spec[key]]:
            raise SystemExit(f"{out}: the resumed run was started with other {', '.join(changed)}")
        return recorded
    spec.update(panels=panels(args.bench, args.scale), batches=batches(args.bench, args.scale, args.method),
                train_exclude=[])
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    return spec


def train(args, out):
    """Run the method's optimizer (none for seed and none), then test the final skill (seed: on every split)."""
    if (out / "native_protocol.json").is_file() and not args.resume:
        raise SystemExit(f"{out} holds a started run: pass --resume to continue it")
    cache = cache_root(args, out)
    llm, result = None, {"final_skill": None if args.method == "none" else SEED, "episodes": {}, "extra": {}}
    if args.method not in ("seed", "none"):
        from src.llm import LLM
        llm = LLM(args.model, out)  # first: a refused endpoint leaves no started run
    spec = protocol(args, out, cache)
    sample, schedule = spec["panels"], spec["batches"]
    cell = Cell(args.bench, args.model, HARNESS, out, cache_root=cache, slots=args.slots)
    if llm is not None:
        result = importlib.import_module(f"src.adapters.{args.method}").run(cell, sample, schedule, llm, out)
    final, consumed = result["final_skill"], episodes(out)
    (out / "final").mkdir(exist_ok=True)
    (out / "final/SKILL.md").write_text(final or "", encoding="utf-8")
    splits = (("train", "train"), ("sel", "dev"), ("test", "test")) if args.method == "seed" else (("test", "test"),)
    skills = {"seed": SEED, "final": final} if llm is not None else {"final": final}
    evaluated = {}
    for panel, split in splits:
        for name, text in skills.items():
            rows = evaluate(cell, text, sample[panel], split)
            record_round(out, arm=args.method, step=0 if llm is None else len(schedule), role=f"{name}_{split}",
                         skill_text=text, evaluation_path=rows[0]["evaluation_path"])
            evaluated.setdefault(split, {})[name] = score(rows)
    summary = {"arm": args.method, "benchmark": args.bench, "model": args.model, "harness": HARNESS,
               "scale": args.scale, "panels": {split: len(ids) for split, ids in sample.items()}, "T": len(schedule),
               "episodes": consumed, "arm_episodes": result["episodes"], "llm": dict(llm.usage) if llm else {},
               "extra": result["extra"], "final_skill": final, "final_words": len((final or "").split()),
               "seed_words": len(SEED.split()), "test": evaluated["test"], "protocol": "native-v1",
               "arm_options": spec["arm_options"], "feedback_mode": "full", "evaluated_panels": evaluated}
    (out / "summary.tmp").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.tmp").replace(out / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("final_skill", "extra")}, indent=1))


def slot(row):
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def known(row):
    return type(row.get("success")) is bool and row.get("outcome_valid", True) is True


def evaluations(directory, version):
    """(path, evaluation, panel slots) of a directory's test evaluations of this skill whose episodes are planned slots,
    each at most once."""
    for path in sorted((directory / "results/evaluations/test").glob("*/evaluation.json")):
        data = json.loads(path.read_text())
        panel, done = [slot(row) for row in data.get("panel", [])], [slot(row) for row in data.get("episodes", [])]
        if (data.get("version_id") == version and data.get("split") == "test" and len(set(panel)) == len(panel)
                and len(set(done)) == len(done) and set(done) <= set(panel)):
            yield path, data, panel


def collect(run, out, version, tasks, test_runs):
    """Coverage of the requested repetitions: repetition 0 is the run's own test evaluation, the others live in
    ``out`` (seed runs: also rep-1, rep-2 next to the run). Evaluations in ``out`` fill slots that have no verdict yet;
    a known verdict is never replaced. A task that ran but got no verdict in every repetition counts as a failure."""
    runs, unresolved = [], []
    for rep in range(test_runs):
        expected = {(task, rep) for task in tasks}
        directories = [run, out] if rep == 0 else [out, run.parent / f"rep-{rep}"]
        path, data = next(((p, d) for directory in directories for p, d, panel in evaluations(directory, version)
                            if set(panel) == expected), (None, None))
        evidence, paths = {slot(row): row for row in (data or {}).get("episodes", [])}, [str(path)] if path else []
        for extra_path, extra, panel in evaluations(out, version):
            if not panel or not set(panel) <= expected:
                continue
            fills = [row for row in extra.get("episodes", [])
                     if slot(row) in panel and not known(evidence.get(slot(row), {}))]
            evidence.update((slot(row), row) for row in fills)
            if fills and str(extra_path) not in paths:
                paths.append(str(extra_path))
        verdicts = {key: row["success"] for key, row in evidence.items() if known(row)}
        missing = [task for task in tasks if (task, rep) not in verdicts]
        ran = bool(data) and Counter(slot(row) for row in data.get("episodes", [])) == Counter(expected)
        runs.append({"repetition": rep, "tasks": len(tasks), "known": len(verdicts), "panel_complete": ran or not missing,
                     "outcomes_complete": len(verdicts) == len(tasks),
                     "Q_known": sum(verdicts.values()) / len(verdicts) if verdicts else None,
                     "evaluation_path": str(path) if path else None, "evaluation_paths": paths, "missing_tasks": missing,
                     "successes": sum(verdicts.values())})
        unresolved.append({task: row.get("outcome_reason") or "unresolved" for (task, _), row in evidence.items()
                           if (task, rep) not in verdicts})
    adjudicated = {task: {str(rep): reasons[task] for rep, reasons in enumerate(unresolved)} for task in tasks
                   if test_runs >= 2 and all(task in reasons for reasons in unresolved)}
    for row in runs:
        row["adjudicated_failures"] = sorted(adjudicated)
        row["missing_tasks"] = [task for task in row["missing_tasks"] if task not in adjudicated]
        successes = row.pop("successes")
        row["Q"] = successes / len(tasks) if not row["missing_tasks"] else None
    complete = all(row["Q"] is not None for row in runs)
    return {"source_run": str(run), "version_id": version, "requested_repetitions": list(range(test_runs)), "runs": runs,
            "repetitions_complete": all(row["outcomes_complete"] for row in runs), "adjudicated_failures": adjudicated,
            "avg": sum(row["Q"] for row in runs) / test_runs if complete else None}


def source(run):
    """(benchmark, model, cache root, skill text, test tasks) of a finished run, or of a transfer evaluation."""
    if (run / "native_protocol.json").is_file():
        protocol = json.loads((run / "native_protocol.json").read_text())
        return (protocol["benchmark"], protocol["model"], relocate(protocol["cache_root"]).resolve(),
                json.loads((run / "summary.json").read_text())["final_skill"],
                [slot(row)[0] for row in protocol["panels"]["test"]])
    (path,) = run.glob("results/evaluations/test/*/evaluation.json")
    text = (run / "results/skills/deliveries/evaluations/test" / path.parent.name / "native/SKILL.md").read_text()
    model, _, bench, _ = run.parts[-4:]
    return (bench, model, (CACHE / MODELS[model]["cache_root"]).resolve(), text,
            [slot(row)[0] for row in json.loads(path.read_text())["panel"]])


def repeat(args, out):
    """Evaluate the slots the repetitions still miss, then write <out>/results.json."""
    run = run_dir(args.method, args.bench, args.model, args.tag).resolve()
    bench, model, cache, text, tasks = source(run)
    version = version_from_files({} if text is None else {"SKILL.md": text})["version_id"]
    cell = Cell(bench, model, HARNESS, out.resolve(), cache_root=cache, slots=args.slots)
    for row in collect(run, out, version, tasks, args.test_runs)["runs"]:
        if not row["outcomes_complete"] and row["missing_tasks"]:
            evaluate(cell, text, [{"task_id": task, "repetition": row["repetition"], "weight": 1.0}
                                  for task in row["missing_tasks"]], "test")
    result = collect(run, out, version, tasks, args.test_runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def transfer(args):
    """(source run, skill text, cell) of a returned skill (or the seed) tested on another model or benchmark."""
    to_bench = args.to in BENCHMARKS
    model, bench = (args.model, args.to) if to_bench else (args.to, args.bench)
    run, text, name = None, SEED, "ood-bench-seed" if to_bench else "ood-seed"
    if args.method != "seed":
        from src.utils.dataloader import current_run
        run = run_dir(args.method, args.bench, args.model, args.tag) if args.tag else \
            current_run(METHOD_DIR.get(args.method, args.method), args.model, args.bench)
        if run is None or not (run / "summary.json").is_file():
            raise SystemExit(f"{args.method} on {args.bench} with {args.model} has no finished run")
        text = json.loads((run / "summary.json").read_text())["final_skill"]
        name = f"ood-from-bench-{args.bench}" if to_bench else f"ood-from-{args.model}"
    out = (EXECUTION / model / METHOD_DIR.get(args.method, args.method) / bench / name).resolve()
    return run, text, Cell(bench, model, HARNESS, out, cache_root=(CACHE / MODELS[model]["cache_root"]).resolve(),
                           slots=args.slots)


def command(args, action, method, bench):
    """The resolved command line of one run or repeat job; a run that has started is resumed."""
    run = run_dir(method, bench, args.model, args.tag)
    argv = [sys.executable, "-u", "-m", "src.run", action, method, bench, args.model, "--slots", str(args.slots),
            "--tag", run.name]
    if action == "repeat":
        return argv + ["--test-runs", str(args.test_runs), "--execute"]
    resume = ["--resume"] if (run / "native_protocol.json").is_file() else []
    return argv + ["--scale", args.scale, "--cache-root", str(cache_root(args, run))] + resume + ["--execute"]


def locked(out, job):
    """Run ``job`` holding <out>/.launch.lock, so that two launches never write one run."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{out} is in use by another launch")
        job()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bench = lambda name: ALIASES.get(name, name)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "repeat", "transfer", "suite"):
        sub = actions.add_parser(action)
        if action == "suite":
            sub.add_argument("--benches", nargs="+", type=bench, choices=list(BENCHMARKS), default=list(BENCHMARKS))
            sub.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        else:
            sub.add_argument("method", choices=METHODS)
            sub.add_argument("bench", type=bench, choices=list(BENCHMARKS))
        sub.add_argument("model", choices=list(MODELS))
        sub.add_argument("--tag", help="run directory name (default: the cell's recorded run in configs/experiments.yaml)")
        sub.add_argument("--slots", type=int, default=16, help="episodes at once")
        if action == "run":
            sub.add_argument("--resume", action="store_true", help="continue the run recorded in the run directory")
        if action in ("run", "suite"):
            sub.add_argument("--scale", default="main", choices=("main", "smoke"))
            sub.add_argument("--cache-root", help="episode cache: a directory under assets/cache or a path")
        if action in ("repeat", "suite"):
            sub.add_argument("--test-runs", type=int, default=3)
        if action == "transfer":
            sub.add_argument("--to", required=True, type=bench, choices=[*MODELS, *BENCHMARKS],
                             help="the model or benchmark to test the skill on")
        sub.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "suite":
        jobs = [command(args, action, method, bench) for bench in args.benches for method in args.methods
                for action in ("run", "repeat")]
    elif args.action == "transfer":
        source_run, text, cell = transfer(args)
        jobs = [f"{source_run or 'seed'} -> {cell.root}"]
    else:
        run = run_dir(args.method, args.bench, args.model, args.tag).resolve()
        out = run if args.action == "run" else repetition_dir(run)
        jobs = [command(args, args.action, args.method, args.bench)]
    if not args.execute:
        print("\n".join(job if isinstance(job, str) else shlex.join(job) for job in jobs))
        return
    os.environ.update(ENV)
    if args.action == "suite":
        for job in jobs:
            subprocess.run(job, cwd=ROOT, check=True)
    elif args.action == "transfer":
        print(json.dumps(score(evaluate(cell, text, panels(cell.benchmark)["test"], "test")), indent=1))
    elif args.action == "run":
        locked(out, lambda: print(f"already finished: {out}") if (out / "summary.json").is_file() else train(args, out))
    else:
        locked(out, lambda: repeat(args, out))


if __name__ == "__main__":
    main()
