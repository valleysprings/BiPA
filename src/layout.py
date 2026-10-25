"""Where the code, the recorded runs and the episode caches live; absolute paths come from the environment."""
import os, re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("BIPA_ASSETS", ROOT / "assets"))
HARNESS_BENCH = Path(os.environ.get("HARNESS_BENCH_ROOT", ROOT.parent / "harness-bench"))
OUT = Path(os.environ.get("BIPA_OUT", ROOT / "out"))
SEED_FILE = Path(os.environ.get("BIPA_SEED_FILE", ROOT / "skills/seed/SKILL.md"))
LEGACY_ROOTS = [root for root in os.environ.get("BIPA_LEGACY_ROOTS", "").split(os.pathsep) if root]

ARTIFACTS, CACHE, OPTIMIZERS = ASSETS / "artifacts", ASSETS / "cache", ROOT / "optimizers"
EXECUTION, RESULTS = ARTIFACTS / "execution", ARTIFACTS / "results"

MODELS = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))["models"]
METHOD_DIR = {"none": "native-noseed", "noskill": "native-noseed", "seed": "native-seed"}


def run_dir(method, bench, model, tag=None):
    """execution/<model>/<method dir>/<bench>/<tag>; the tag defaults to the cell's recorded run."""
    spec = MODELS[model]
    default = spec["seed"] if method == "seed" else spec["noseed"][bench] if method == "none" else spec["run"]
    return EXECUTION / model / METHOD_DIR.get(method, method) / bench / (tag or default)


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
