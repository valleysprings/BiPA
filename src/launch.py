"""Plan or execute BiPA runs and frozen-skill repetitions. Plans never call a model."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment_config as config
from src.layout import BASELINE_ROOT, EXECUTION

ARMS = ("seed", "none", "skillopt", "gepa", "skilladam")


def job(model, bench, arm, *, repeat=False, tag=None, slots=None, test_runs=None, scale=None):
    base = config.model_config(model)
    run = config.run_dir(model, bench, arm, tag)
    slots = slots or base["slots"]
    scale = scale or config.CFG["scale"]
    if repeat:
        argv = [str(BASELINE_ROOT / "repeat.py"), "--run", str(run), "--slots", str(slots),
                "--test-runs", str(test_runs or config.CFG["test_runs"]), "--execute"]
        out = config.repetition_dir(run)
    else:
        argv = [str(BASELINE_ROOT / "run.py"), "--protocol", "native-v1", "--feedback-mode", "full",
                "--arm", arm, "--benchmark", bench, "--model", model, "--harness", config.HARNESS,
                "--scale", scale, "--slots", str(slots), "--out", str(run),
                "--cache-root", str(config.baseline_cache(model)), *base["args"]]
        if (run / "native_protocol.json").is_file():
            argv.append("--resume")
        out = run
    return {"command": [sys.executable, "-u", *argv], "out": out, "repeat": repeat}


def plan(args):
    benches = [config.bench_id(b) for b in (args.benches if args.action == "suite" else [args.bench])]
    arms = args.arms if args.action == "suite" else [args.arm]
    jobs = []
    for bench in benches:
        for arm in arms:
            options = dict(tag=args.tag, slots=args.slots, test_runs=args.test_runs,
                           scale=getattr(args, "scale", None))
            if args.action != "repeat":
                jobs.append(job(args.model, bench, arm, **options))
            if args.action in ("repeat", "suite"):
                jobs.append(job(args.model, bench, arm, repeat=True, **options))
    return jobs


def execute(jobs):
    config.configure_environment()
    for item in jobs:
        out = item["out"]
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not item["repeat"] and (out / "summary.json").is_file():
                print(f"Optimization already finished: {out}; see completion_audit.json for evidence status.", flush=True)
                continue
            subprocess.run(item["command"], cwd=BASELINE_ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "repeat", "suite"):
        command = sub.add_parser(action)
        if action != "suite":
            command.add_argument("bench")
        command.add_argument("model", choices=tuple(config.CFG["models"]))
        if action == "suite":
            command.add_argument("--benches", nargs="+", default=list(config.BENCHMARKS))
            command.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
        else:
            command.add_argument("--arm", choices=ARMS, required=True)
        command.add_argument("--tag")
        command.add_argument("--scale", default=config.CFG["scale"],
                             help="panel scale from configs/tasks.yaml (main, smoke, lodo-*)")
        command.add_argument("--slots", type=int)
        command.add_argument("--test-runs", type=int, default=config.CFG["test_runs"])
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--execute", action="store_true")
        mode.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if (args.slots is not None and args.slots < 1) or args.test_runs < 1:
        parser.error("slots and test-runs must be positive")
    try:
        jobs = plan(args)
        if args.detach:
            # Detach the entire sequence, so repetitions wait for their optimizer to finish.
            raw = list(sys.argv[1:] if argv is None else argv)
            raw[raw.index("--detach")] = "--execute"
            out = EXECUTION / args.model / "launches"
            out.mkdir(parents=True, exist_ok=True)
            with (out / "bipa.log").open("a") as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *raw],
                    cwd=BASELINE_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            print(f"pid={process.pid} log={out / 'bipa.log'}")
        elif args.execute:
            execute(jobs)
        else:
            for item in jobs:
                print(shlex.join(item["command"]))
    except (ValueError, BlockingIOError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
