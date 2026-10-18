"""Cross-benchmark transfer: a returned skill evaluated unchanged on another benchmark (appendix transfer table).

The returned skill of (model, method) trained on benchmark A is evaluated with the SAME model on benchmark B's TEST
panel (panels(B)["test"], repetition 0, as the baselines' own test evaluation; B != A), next to the same model's Seed
skill on B. Q and the cost ratios rho are paired against that Seed on the same tasks (main_table.pair). A returned skill
identical to the Seed shares the Seed row (reported, not executed again). Default is a dry run: per cell the skill
version, whether the method returned the seed, the cache hits on B's test panel and the new episodes; --execute calls
the model. Cells whose train run has no summary.json yet are skipped (listed in the dry run).

Run resolution of the train cell (model, A): skillopt / gepa / skilladam = layout.current_run (must have summary.json).

Usage:  python -m src.dynamics.transfer_bench [--models deepseek-flash] [--methods skillopt,gepa,skilladam]
                                          [--slots 16] [--execute]   (from the bipa root)

Outputs:
  execution/<model>/<method>/<B>/ood-from-bench-<A>/   evaluations, summary.json
  execution/<model>/native-seed/<B>/ood-bench-seed     the Seed reference (cache hits of the seed's own test row)
  results/<model>/<method>/<B>/ood-from-bench-<A>/{summary.json,final/SKILL.md}
  results/ood-cross-bench.{json,md}                    Q, rho, rho_inv, rho_trj; rows of models, methods and tags
                                                       not in this call stay
"""

import argparse
import json

from src import experiment_config as experiments
experiments.configure_environment()
from src.layout import EXECUTION
from src.utils.executor import SEED, Cell, evaluate, evaluator
from src.layout import ARM_DIR, RESULTS, current_run
from src.panels import panels

from src.dynamics.main_table import evaluation, pair, version  # noqa: E402

BENCHES = experiments.BENCHMARKS


def train_run(method, model, bench):
    """The returned-skill run of a train cell, or None when it has no summary.json (yet)."""
    run = current_run(method, model, bench)
    if run and (run / "summary.json").is_file():
        return run
    return None


def build_cells(models, methods, benches=BENCHES):
    """Seed references (one per model x eval bench) and the transfer cells: skill trained on A -> eval bench B != A."""
    seeds = []
    for model in models:
        for bench in benches:
            seeds.append({"model": model, "bench": bench, "train": bench, "method": "seed", "run": "",
                          "text": SEED, "returned_seed": False})

    transfers = []
    skipped = []
    for model in models:
        for trained_on in benches:
            for method in methods:
                run = train_run(method, model, trained_on)
                # a train cell without a finished run has nothing to transfer
                if not run:
                    skipped.append((model, method, trained_on))
                    continue
                text = json.loads((run / "summary.json").read_text())["final_skill"]
                for bench in benches:
                    if bench == trained_on:
                        continue
                    transfers.append({"model": model, "bench": bench, "train": trained_on,
                                      "method": method,
                                      "run": str(run.relative_to(experiments.MITIGATION_ROOT)), "text": text,
                                      "returned_seed": text == SEED})

    for c in seeds + transfers:
        c["vid"] = version(c["text"])
        c["is_seed"] = c["method"] == "seed"
        name = "ood-bench-seed" if c["is_seed"] else f"ood-from-bench-{c['train']}"
        c["out"] = (EXECUTION / c["model"] / ARM_DIR.get(c["method"], c["method"]) / c["bench"] / name).resolve()
        c["test"] = panels(c["bench"])["test"]
    return seeds, transfers, skipped


def executor_cell(c, slots):
    """The executor cell of a transfer row (the baselines' evaluation cache, whatever the model)."""
    return Cell(c["bench"], c["model"], experiments.HARNESS, c["out"], slots=slots, cache_root=experiments.baseline_cache(c["model"]))


def cached(c, slots):
    """How many test episodes of this skill the cache already has (read only)."""
    rows = [{"task_id": t, "repetition": 0, "weight": 1.0} for t in c["test"]]
    version_files = {"version_id": c["vid"], "files": {"SKILL.md": c["text"]}}
    return len(evaluator(executor_cell(c, slots)).cached_episodes(version_files, rows, "test"))


def fmt(v, d=3):
    """A number with d decimals, or "-" when missing."""
    return "-" if v is None else f"{v:.{d}f}"


def render(out):
    """results/ood-cross-bench.md: one table row per transfer result."""
    lines = ["# Cross-benchmark transfer (OOD)", "",
             "Returned skill optimized on the train benchmark, evaluated unchanged with the same model on the eval "
             "benchmark's test panel. Q = % success (known/total); rho = per-task geometric-mean cost ratio against "
             "the same model's Seed skill on the same tasks (main_table.pair), paired n. A returned skill equal to the "
             "Seed shares the Seed row.", "",
             "| model | train -> eval | method | tag | returned seed | Q | rho | rho_inv | rho_trj |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        ratios = " | ".join(f"{fmt(r[m])} ({r[m + '_n']})" for m in ("rho", "rho_inv", "rho_trj"))
        lines.append(f"| {r['model']} | {r['train']} -> {r['bench']} | {r['method']} | {r['source_run'].split('/')[-1]} | "
                     f"{'yes' if r['returned_seed'] else ''} | {fmt(r['Q'], 1)} ({r['known']}/{r['total']}) | {ratios} |")
    return "\n".join(lines) + "\n"


def dry_run(seeds, transfers, skipped):
    """Print one row per cell with its cache hits and the number of new test episodes."""
    print(f"{'model':15} {'bench':19} {'train':19} {'method':9} {'run':34} {'version':13} {'returned':13} cached/new")
    new = 0
    seen = set()
    order = sorted(seeds + transfers, key=lambda c: (c["model"], c["bench"], not c["is_seed"], c["train"], c["method"]))
    for c in order:
        key = (c["model"], c["bench"], c["vid"])
        n = len(c["test"])
        # the same skill on the same model and benchmark is evaluated once
        need = 0 if key in seen else n - c["cached"]
        seen.add(key)
        if not c["is_seed"]:
            new += need
        returned = "returned seed" if c["returned_seed"] else "-"
        print(f"{c['model']:15} {c['bench']:19} {c['train']:19} {c['method']:9} {c['run'].split('/')[-1]:34} "
              f"{c['vid'][:12]:13} {returned:13} {c['cached']}/{n - c['cached']}")
    for model, method, bench in skipped:
        print(f"skipped (no summary.json yet): {model} {method} trained on {bench}")
    print(f"total new test episodes, optimized skills (deduplicated by model x eval bench x version): {new}; "
          "Seed reference rows not counted")


def execute(seeds, transfers, slots):
    """Evaluate every cell (seed rows first), pair each transfer with its model's Seed, merge into the results."""
    ref = {}
    out = []
    for c in seeds + transfers:
        evaluate(executor_cell(c, slots), c["text"], c["test"], "test")
        final = evaluation(c["out"], c["vid"])
        # a Seed row is the reference of the transfers to the same model and benchmark
        if c["is_seed"]:
            ref[c["model"], c["bench"]] = final
            continue
        keys = ("model", "bench", "train", "method", "run", "vid", "returned_seed")
        r = {k: c[k] for k in keys}
        r |= {"source_run": c["run"], "eval_run": str(c["out"].relative_to(experiments.MITIGATION_ROOT))}
        r |= pair(final, ref[c["model"], c["bench"]])
        out.append(r)
        cell_dir = RESULTS / c["model"] / c["method"] / c["bench"] / f"ood-from-bench-{c['train']}"
        (cell_dir / "final").mkdir(parents=True, exist_ok=True)
        (cell_dir / "final/SKILL.md").write_text(c["text"])
        (cell_dir / "summary.json").write_text(json.dumps(r, indent=1))
        (c["out"] / "summary.json").write_text(json.dumps(r, indent=1))

    path = RESULTS / "ood-cross-bench.json"

    def key(r):
        return (r["model"], r["train"], r["bench"], r["method"], r["source_run"])

    # rows of models, methods and tags not in this call stay
    earlier = json.loads(path.read_text()) if path.is_file() else []
    merged = {key(r): r for r in earlier} | {key(r): r for r in out}
    path.write_text(json.dumps(list(merged.values()), indent=1))
    (RESULTS / "ood-cross-bench.md").write_text(render(list(merged.values())))


def main(argv=None):
    """Dry run (default) or run (--execute) of the cross-benchmark transfer evaluation."""
    a = argparse.ArgumentParser(prog="python -m src.dynamics.transfer_bench", description=__doc__.splitlines()[0])
    a.add_argument("--models", default="deepseek-flash")
    a.add_argument("--methods", default="skillopt,gepa,skilladam")
    a.add_argument("--slots", type=int, default=16)
    a.add_argument("--execute", action="store_true")
    args = a.parse_args(argv)

    seeds, transfers, skipped = build_cells(args.models.split(","), args.methods.split(","))
    for c in seeds + transfers:
        c["cached"] = cached(c, args.slots)
    if args.execute:
        execute(seeds, transfers, args.slots)
    else:
        dry_run(seeds, transfers, skipped)


if __name__ == "__main__":
    main()
