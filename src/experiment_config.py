"""BiPA run selection and protocol environment; no BACH imports or model calls."""
import os

import yaml

from src.layout import BASELINE_ROOT, CACHE, EXECUTION, MITIGATION_ROOT

CFG = yaml.safe_load((BASELINE_ROOT / "configs/experiments.yaml").read_text())
HARNESS = CFG["harness"]
BENCHMARKS = tuple(CFG["benchmarks"])
ALIASES = {spec["alias"]: bench for bench, spec in CFG["benchmarks"].items()}


def bench_id(name):
    bench = ALIASES.get(name, name)
    if bench not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    return bench


def model_config(model):
    return CFG["models"][model]


def baseline_cache(model="deepseek-flash"):
    return (CACHE / model_config(model)["cache_root"]).resolve()


def run_dir(model, bench, arm, tag=None):
    base = model_config(model)
    method = {"seed": "native-seed", "none": "native-noseed"}.get(arm, arm)
    name = base["seed"] if arm == "seed" else base["noseed"][bench] if arm == "none" else base["run"]
    return EXECUTION / model / method / bench / (tag or name)


def repetition_dir(run):
    """Include the source run tag so repetitions of different returned skills cannot collide."""
    model, method, bench, tag = run.parts[-4:]
    return EXECUTION / model / "test-reps" / f"{bench}__{method}__{tag}"


def configure_environment():
    os.environ.update(CFG["env"])
