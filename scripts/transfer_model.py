"""Cross-model transfer: a returned skill evaluated unchanged with another model (appendix transfer table).

The final skill of (train model A, method) is evaluated with model B on B's TEST panel of the same benchmark
(panels(bench)["test"], repetition 0, as the baselines' own test evaluation), next to B's Seed skill. Q and the cost
ratios rho are paired against B's Seed on the same tasks (main_table.pair). Default is a dry run: per cell the skill
version, whether the method returned the seed, the cache hits on the test panel and the new episodes; --execute calls
the model. Evaluations already in the cache are read back.

Run resolution: skillopt / gepa / skilladam = layout.current_run(method, A, bench) (what main_table.py scores; it must
have summary.json).

Usage:  python bipa/scripts/transfer_model.py [--methods skillopt,gepa,skilladam] [--slots 16] [--execute]
In the background (the cache is read back, so a restart resumes):
  setsid nohup python bipa/scripts/transfer_model.py --methods skillopt,gepa,skilladam --slots 16 --execute >> LOG 2>&1 &
  (LOG: assets/logs/ood-launch.log)

Outputs:
  execution/<B>/<method>/<bench>/ood-from-<A>/        evaluations, summary.json
  execution/<B>/native-seed/<bench>/ood-seed          the Seed reference
  results/<B>/<method>/<bench>/ood-from-<A>/{summary.json,final/SKILL.md}
  results/ood-cross-model.{json,md}                   Q, rho, rho_inv, rho_trj against B's Seed
"""

import argparse
import json
import sys

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # bipa root, for src.*
sys.path.insert(0, str(HERE))         # this dir, for main_table
from src import experiment_config as experiments
experiments.configure_environment()
from src.layout import CACHE, EXECUTION
from src.utils.executor import SEED, Cell, evaluate, evaluator
from src.layout import RESULTS, current_run
from src.panels import panels

from main_table import evaluation, pair, version  # noqa: E402

MODELS = tuple(experiments.CFG["models"])   # train and eval models of the transfer (config/experiments.yaml models)
BENCHES = experiments.BENCHMARKS


def run_of(method, model, bench):
    """The returned-skill run of a train cell, or None when it has no summary.json."""
    run = current_run(method, model, bench)
    if run and (run / "summary.json").is_file():
        return run
    return None


def build_cells(methods, slots, cache, benches=BENCHES, evals=None):
    """The Seed reference rows and the transfer cells, each with its version, output dir, executor cell and cache hits.

    A transfer cell is one (bench, train model -> eval model, method).  benches limits the benchmarks, evals the eval models
    (default: every model); a seed row is built for each eval model only."""
    eval_models = evals or MODELS
    seeds = []
    for bench in benches:
        for model in eval_models:
            seeds.append({"bench": bench, "eval": model, "train": model, "method": "seed", "run": "",
                          "text": SEED, "returned_seed": False})

    transfers = []
    for bench in benches:
        for evaluated in eval_models:
            for trained in (m for m in MODELS if m != evaluated):
                for method in methods:
                    run = run_of(method, trained, bench)
                    # a train cell without a finished run has nothing to transfer
                    if not run:
                        continue
                    text = json.loads((run / "summary.json").read_text())["final_skill"]
                    transfers.append({"bench": bench, "train": trained, "eval": evaluated,
                                      "method": method,
                                      "run": str(run.relative_to(experiments.MITIGATION_ROOT)), "text": text,
                                      "returned_seed": text == SEED})

    for c in seeds + transfers:
        is_seed = c["method"] == "seed"
        c["vid"] = version(c["text"])
        arm = "native-seed" if is_seed else c["method"]
        name = "ood-seed" if is_seed else f"ood-from-{c['train']}"
        c["out"] = (EXECUTION / c["eval"] / arm / c["bench"] / name).resolve()
        c["cell"] = Cell(c["bench"], c["eval"], experiments.HARNESS, c["out"], slots=slots, cache_root=cache or experiments.baseline_cache(c["eval"]))
        c["test"] = panels(c["bench"])["test"]
        rows = [{"task_id": t, "repetition": 0, "weight": 1.0} for t in c["test"]]
        version_files = {"version_id": c["vid"], "files": {"SKILL.md": c["text"]}}
        # read only: how many test episodes of this skill the cache already has
        c["cached"] = len(evaluator(c["cell"]).cached_episodes(version_files, rows, "test"))
    return seeds, transfers


def dry_run(seeds, transfers):
    """Print one row per cell with its cache hits and the number of new test episodes."""
    print(f"{'bench':19} {'train->eval':29} {'method':9} {'run':38} {'version':13} {'returned':13} cached/needed")
    new = 0
    seen = set()
    order = sorted(seeds + transfers, key=lambda c: (c["bench"], c["eval"], c["train"], c["method"] != "seed", c["method"]))
    for c in order:
        key = (c["eval"], c["bench"], c["vid"])
        n = len(c["test"])
        # the same skill on the same model and benchmark is evaluated once
        need = 0 if key in seen else n - c["cached"]
        seen.add(key)
        if c["method"] != "seed":
            new += need
        returned = "returned seed" if c["returned_seed"] else "-"
        print(f"{c['bench']:19} {c['train'] + '->' + c['eval']:29} {c['method']:9} {c['run'].split('/')[-1]:38} "
              f"{c['vid'][:12]:13} {returned:13} {c['cached']}/{n - c['cached']}")
    print(f"total new test episodes, optimized skills (deduplicated by eval model x bench x version): {new}; "
          "eval-model seed reference rows not counted")


def fmt(v, d=3):
    """A number with d decimals, or "-" when missing."""
    return "-" if v is None else f"{v:.{d}f}"


def render(out):
    """results/ood-cross-model.md: one table row per transfer result."""
    lines = ["# Cross-model transfer (OOD)", "",
             "Final skill optimized with the train model, evaluated unchanged with the eval model on its test panel. "
             "Q = % success (known/total); rho = per-task geometric-mean cost ratio against the eval model's seed on "
             "the same tasks (main_table.pair), paired n.", "",
             "| bench | train -> eval | method | returned seed | Q | rho | rho_inv | rho_trj |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out:
        ratios = " | ".join(f"{fmt(r[m])} ({r[m + '_n']})" for m in ("rho", "rho_inv", "rho_trj"))
        lines.append(f"| {r['bench']} | {r['train']} -> {r['eval']} | {r['method']} | "
                     f"{'yes' if r['returned_seed'] else ''} | {fmt(r['Q'], 1)} ({r['known']}/{r['total']}) | {ratios} |")
    return "\n".join(lines) + "\n"


def execute(seeds, transfers):
    """Evaluate every cell (seed rows first), pair each transfer with its eval model's Seed, write the results."""
    ref = {}
    out = []
    for c in seeds + transfers:
        evaluate(c["cell"], c["text"], c["test"], "test")
        final = evaluation(c["out"], c["vid"])
        # a Seed row is the reference of the transfers to the same model and benchmark
        if c["method"] == "seed":
            ref[c["eval"], c["bench"]] = final
            continue
        keys = ("bench", "train", "eval", "method", "run", "vid", "returned_seed")
        r = {k: c[k] for k in keys}
        r |= {"source_run": c["run"], "eval_run": str(c["out"].relative_to(experiments.MITIGATION_ROOT))}
        r |= pair(final, ref[c["eval"], c["bench"]])
        out.append(r)
        cell_dir = RESULTS / c["eval"] / c["method"] / c["bench"] / f"ood-from-{c['train']}"
        (cell_dir / "final").mkdir(parents=True, exist_ok=True)
        (cell_dir / "final/SKILL.md").write_text(c["text"])
        (cell_dir / "summary.json").write_text(json.dumps(r, indent=1))
        (c["out"] / "summary.json").write_text(json.dumps(r, indent=1))
    (RESULTS / "ood-cross-model.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "ood-cross-model.md").write_text(render(out))


def main(argv=None):
    """Dry run (default) or run (--execute) of the cross-model transfer evaluation."""
    a = argparse.ArgumentParser(prog="python bipa/scripts/transfer_model.py", description=__doc__.splitlines()[0])
    a.add_argument("--methods", default="skillopt,gepa,skilladam")
    a.add_argument("--benches", default=",".join(BENCHES), help="comma-separated benchmark ids (default: all)")
    a.add_argument("--eval", default=None, help="the eval model (default: every model, i.e. both directions)")
    a.add_argument("--slots", type=int, default=16)
    a.add_argument("--execute", action="store_true")
    args = a.parse_args(argv)

    evals = [args.eval] if args.eval else None
    seeds, transfers = build_cells(args.methods.split(","), args.slots, None, args.benches.split(","), evals)
    if args.execute:
        execute(seeds, transfers)
    else:
        dry_run(seeds, transfers)


if __name__ == "__main__":
    main()
