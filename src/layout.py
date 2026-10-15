"""Shared locations for baseline execution, reports, and the seed episode cache."""
import re
from pathlib import Path

import yaml

BASELINE_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = BASELINE_ROOT.parent
MITIGATION_ROOT = BASELINES_ROOT.parent
# the harness-bench checkout: the campaign's backend (configs/tasks.yaml backend.bench_root), the one place it is set
HARNESS_BENCH = Path(yaml.safe_load((BASELINE_ROOT / "configs/tasks.yaml").read_text(encoding="utf-8"))
                     ["backend"]["bench_root"])
ASSETS = MITIGATION_ROOT / "assets"
ARTIFACTS = ASSETS / "artifacts"
OPTIMIZERS = BASELINE_ROOT / "optimizers"
SKILLS = ASSETS / "skills"
RESULTS = ARTIFACTS / "results"
EXECUTION = ARTIFACTS / "execution"
ATTRIBUTION = ARTIFACTS / "attribution"
PROFILING = ARTIFACTS / "profiling"
CACHE = ASSETS / "cache"  # mutable eval cache; artifacts/ holds static outputs only

# Records written before 2026-10-03 store absolute paths under these roots.
_MIT = "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation"
OLD_ARTIFACT_ROOTS = (
    "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-bench/assets/artifacts",
    _MIT + "/artifacts",
    _MIT + "/baselines/assets/artifacts",
    _MIT + "/bach-harness/assets/artifacts",
    _MIT + "/bach-harness/assets/artifacts",
)
_SOURCE_ROOTS = tuple(dict.fromkeys((_MIT, str(MITIGATION_ROOT),
                                   "/mnt/shared-storage-user/zhoujiawei/coharness/harness-mitigation", "")))
OLD_OPTIMIZER_ROOTS = tuple((root + "/" if root else "") + suffix
                           for root in _SOURCE_ROOTS
                           for suffix in ("assets/optimizers", "baselines/assets/optimizers"))
OLD_CACHE_ROOTS = (_MIT + "/assets/artifacts/cache",
                   "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-bench/assets/artifacts/cache")
OLD_BENCH_ROOTS = (
    "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-bench-exec",
    _MIT + "/harness-bench",
    _MIT + "/harness-bench-exec",
)


# Three files the native-20261003 certificates bind in both old harness-bench trees with different
# content; the exec (faffb13) versions are kept as evidence only (see that directory's README).
_EXEC = "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-bench-exec/"
_EXEC_EVIDENCE = MITIGATION_ROOT / ".trash/assets/campaigns-20261004/native-20261003/source-evidence/harness-bench-exec-faffb13"
EXEC_ONLY_FILES = {_EXEC + rel: _EXEC_EVIDENCE / rel
                   for rel in ("src/harness/harbor.py", "src/sandbox/base.py", "src/sandbox/dind.py")}


# every run writes directly to EXECUTION / <model> / <method> / <bench> / <run-id>; there is no separate live area
ARM_DIR = {"none": "native-noseed", "noskill": "native-noseed", "seed": "native-seed", "zo": "ours"}
OURS_RUNS = MITIGATION_ROOT / "bach-harness/config/ours_runs.json"  # the run each (model, bench) cell reports as `ours`
_OLD_LIVE = re.compile(re.escape(_MIT) + r"/runs/(zo|skilladam|ood|probe)/([a-z0-9-]+)/([a-z0-9-]+)/([^/]+)")
_LIVE_RUN = re.compile(re.escape(str(EXECUTION / "runs")) + r"/([a-z0-9_]+)/([a-z0-9-]+)/opencode/([a-z0-9-]+)/([^/]+)")


def _exported(text):
    """execution/runs/<arm>/<model>/opencode/<bench>/<run>/... -> execution/<model>/<method>/<bench>/<run>/... once that run was exported."""
    m = _LIVE_RUN.match(text)
    if m:
        arm, model, bench, run = m.groups()
        new = EXECUTION / model / ARM_DIR.get(arm, arm) / bench / run
        if new.exists():
            return str(new) + text[m.end():]
    return text


def _from_live(text):
    """<mitigation>/runs/<kind>/<model>/<bench>/<run>/... (the removed live area) -> execution/<model>/<method>/<bench>/<run-name>/..."""
    m = _OLD_LIVE.match(text)
    if not m:
        return text
    kind, model, bench, run = m.groups()
    if kind == "ood":
        method, name = ("native-seed", "ood-seed") if run == "seed" else (run.rsplit("-", 1)[1], "ood-from-" + run.rsplit("-", 1)[0])
    else:
        method, name = {"zo": "ours", "skilladam": "skilladam", "probe": "ours"}[kind], (f"probe-{run}" if kind == "probe" else run)
    return str(EXECUTION / model / method / bench / name) + text[m.end():]


def relocate(path):
    """Map a path recorded under an old artifact/optimizer root or in the removed runs/ live area to the current location."""
    text = str(path)
    if text in EXEC_ONLY_FILES:
        return EXEC_ONLY_FILES[text]
    # Preserve historical provenance strings after the 2026-10-14 source cleanup.
    for root in _SOURCE_ROOTS:
        for directory in ("baselines/candidate-skill", "baselines/bipa"):
            old = (root + "/" if root else "") + directory
            if text == old or text.startswith(old + "/"):
                relative = text[len(old):].lstrip("/")
                if relative.startswith("arms/"):
                    relative = "adapters/" + relative[len("arms/"):]
                if relative in ("executor.py", "llm.py"):
                    relative = "utils/" + relative
                if relative.startswith("skills/"):
                    return SKILLS / relative[len("skills/"):]
                target = BASELINE_ROOT / relative
                archived = MITIGATION_ROOT / ".trash/baselines/bipa-cleanup-20261014/candidate-skill" / relative
                return target if target.exists() or not archived.exists() else archived
    for old, new in [(o, CACHE) for o in OLD_CACHE_ROOTS] + [(o, ARTIFACTS) for o in OLD_ARTIFACT_ROOTS] + [(o, OPTIMIZERS) for o in OLD_OPTIMIZER_ROOTS] + [(o, HARNESS_BENCH) for o in OLD_BENCH_ROOTS]:
        if text == old or text.startswith(old + "/"):
            return Path(_exported(str(new) + text[len(old):]))
    return Path(_exported(_from_live(text)))


def result_dir(arm, model, benchmark):
    """results/<model>/<method>/<bench>: the cell's current run only; no-skill is native-noseed, seed is native-seed."""
    return RESULTS / model / ARM_DIR.get(arm, arm) / benchmark


def run_path(arm, model, bench, run_id):
    """The run dir execution/<model>/<method>/<bench>/<run-id>; runs are written here from the start (resume reads the same dir)."""
    return EXECUTION / model / ARM_DIR.get(arm, arm) / bench / run_id


def run_dirs(arm, model, bench):
    """Every run dir of a cell: execution/<model>/<method>/<bench>/*."""
    cell = EXECUTION / model / ARM_DIR.get(arm, arm) / bench
    return [d for d in cell.iterdir() if d.is_dir()] if cell.is_dir() else []


def ours_run(model, bench):
    """The BACH run the cell reports as `ours` (tag in bach-harness/config/ours_runs.json)."""
    import json
    return run_path("zo", model, bench, json.loads(OURS_RUNS.read_text())[model][bench])


def current_run(arm, model, bench):
    """Newest native-* run of a cell that is not ABORTED, preferring one with summary.json; `ours` (arm zo) is the run named in ours_runs.json."""
    if arm in ("zo", "ours"):
        return ours_run(model, bench)
    runs = [d for d in run_dirs(arm, model, bench) if d.name.startswith("native-") and not (d / "ABORTED.md").exists()]
    done = [d for d in runs if (d / "summary.json").is_file()]
    return max(done or runs, key=lambda d: ((d / "summary.json") if d in done else (d / "native_protocol.json")).stat().st_mtime, default=None)
