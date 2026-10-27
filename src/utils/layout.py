"""Where the code, the recorded runs and the episode caches live; absolute paths come from the environment."""
import os, re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("BIPA_ASSETS", ROOT / "assets"))
HARNESS_BENCH = Path(os.environ.get("HARNESS_BENCH_ROOT", ROOT.parent / "harness-bench"))
OUT = Path(os.environ.get("BIPA_OUT", ROOT / "out"))
SEED_FILE = Path(os.environ.get("BIPA_SEED_FILE", ROOT / "skills/seed/SKILL.md"))
LEGACY_ROOTS = [root for root in os.environ.get("BIPA_LEGACY_ROOTS", "").split(os.pathsep) if root]

ARTIFACTS, CACHE, OPTIMIZERS = ASSETS / "artifacts", ASSETS / "cache", ROOT / "optimizers"
EXECUTION, RESULTS = ARTIFACTS / "execution", ARTIFACTS / "results"
# the analyses' intermediate files; a check on another pin set points this at an empty directory
ANALYSIS = Path(os.environ.get("BIPA_ANALYSIS_CACHE", CACHE / "analysis"))

# each model's episode cache (models), the run of every cell (pins) and the other runs the analyses read (aux_runs);
# BIPA_EXPERIMENTS names another file, e.g. the pins an earlier version of the paper was computed on
EXPERIMENTS = Path(os.environ.get("BIPA_EXPERIMENTS", ROOT / "configs/experiments.yaml"))
_CONFIG = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))
BENCHMARK_NAMES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
MODELS = _CONFIG["models"]
# {model: {method: {benchmark: run}}}; a pin is one run for every benchmark or {default: run, <benchmark>: run}
PINS = {model: {method: {bench: pin if isinstance(pin, str) else pin.get(bench, pin.get("default")) for bench in BENCHMARK_NAMES}
                for method, pin in methods.items()} for model, methods in _CONFIG["pins"].items()}
AUX_RUNS = _CONFIG.get("aux_runs")  # named runs that no cell pins; read through src.analysis.settings.extra
METHOD_DIR = {"none": "native-noseed", "noskill": "native-noseed", "seed": "native-seed"}


def run_dir(method, bench, model, tag=None):
    """execution/<model>/<method dir>/<bench>/<tag>; the tag defaults to the cell's pinned run."""
    method = METHOD_DIR.get(method, method)
    return EXECUTION / model / method / bench / (tag or PINS[model][method][bench])


def repetition_dir(run):
    """Where the repetitions of a run's final skill go: execution/<model>/test-reps/<bench>__<method>__<tag>."""
    model, method, bench, tag = run.parts[-4:]
    return EXECUTION / model / "test-reps" / f"{bench}__{method}__{tag}"


def _exported(text):
    """execution/runs/<arm>/<model>/opencode/<bench>/<run>/... -> execution/<model>/<method>/<bench>/<run>/..."""
    m = re.match(re.escape(str(EXECUTION / "runs")) + r"/([a-z0-9_]+)/([a-z0-9-]+)/opencode/([a-z0-9-]+)/([^/]+)", text)
    if m:
        arm, model, bench, run = m.groups()
        new = EXECUTION / model / METHOD_DIR.get(arm, arm) / bench / run
        if new.exists():
            return str(new) + text[m.end():]
    return text


def relocate(path):
    """A path an older record holds -> where that file is now; the old roots are BIPA_LEGACY_ROOTS."""
    text = str(path)
    for root in LEGACY_ROOTS:
        for old, new in ((root + "/assets/artifacts/cache", CACHE), (root + "/assets/artifacts", ARTIFACTS),
                         (root + "/artifacts", ARTIFACTS), (root + "/assets/optimizers", OPTIMIZERS)):
            if text == old or text.startswith(old + "/"):
                return Path(_exported(str(new) + text[len(old):]))
    return Path(_exported(text))
