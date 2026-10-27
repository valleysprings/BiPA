"""Run discovery and evaluation readers shared by the analyses."""
import json, math, os
from pathlib import Path

from harness_bench.eval import version_from_files
from scipy.stats import gmean as _gmean

from src.utils import layout

MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
METHODS = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")
BN = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
EX = layout.EXECUTION
CACHE = layout.ANALYSIS / "main"
SEED_TEXT = layout.SEED_FILE.read_text()
PINS = layout.PINS


def gmean(values):
    return float(_gmean(values)) if len(values) else None


def vid(text):
    return version_from_files({} if text is None else {"SKILL.md": text})["version_id"]


def out_dir(module):
    d = layout.ANALYSIS / module
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(name, obj):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1))
    return path


def manifest():
    """The run manifest named by BIPA_MANIFEST, or None."""
    p = os.environ.get("BIPA_MANIFEST")
    return json.loads(Path(p).read_text()) if p else None


def _cell(m, bench, row):
    c = (m["cells"].get(bench) or {}).get(row)
    return c if c and c["status"] == "complete" else None


def current_run(method, model, bench):
    """The run a cell reports: the manifest's, else the one pinned in the experiments file, else the cell's only finished run."""
    m = manifest()
    if m:
        c = _cell(m, bench, method)
        return Path(c["run"]) if c and model == m["model"] else None
    if tag := PINS.get(model, {}).get(method, {}).get(bench):
        run = EX / model / method / bench / tag
        if not (run / "summary.json").is_file():
            raise FileNotFoundError(f"pinned run {run} has no summary.json")
        return run
    return only_run(method, model, bench)


def only_run(method, model, bench):
    """The one finished (summary.json) native run of an unpinned cell that is not aborted."""
    cell = EX / model / method / bench
    if not cell.is_dir():
        return None
    runs = [d for d in cell.iterdir() if d.name.startswith("native-") and (d / "summary.json").is_file()
            and not (d / "ABORTED.md").exists()]
    if len(runs) > 1:
        names = ", ".join(sorted(d.name for d in runs))
        raise LookupError(f"{model}/{method}/{bench} has several runs ({names}); pin one in configs/experiments.yaml")
    return runs[0] if runs else None


def _count(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def task_row(ep):
    """success, executed calls N, input tokens X, c = N + X / 1000 and request/turn counts nreq and R of one episode;
    N, X None when not captured."""
    cap, seen, reqs = ep.get("capture") or {}, set(), []
    for r in ep.get("requests", []):
        if r["request_id"] not in seen:
            seen.add(r["request_id"])
            reqs.append(r)
    calls = [c for r in reqs for c in r.get("tool_calls", [])]
    n = sum(c.get("executed") is True for c in calls)
    if cap.get("invocations_complete") is not True or any(c.get("executed") not in (True, False) for c in calls):
        n = None
    ok = cap.get("requests_complete") is True and all(_count(r["input_tokens"]) for r in reqs)
    x = sum(r["input_tokens"] for r in reqs) if ok else None
    return {"success": ep.get("success"), "N": n, "X": x,
            "c": n + x / 1000 if n is not None and x is not None else None, "nreq": len(reqs), "R": len(reqs)}


def final_skill(method, run):
    if method == "native-noseed":
        return None
    if method == "native-seed":
        return SEED_TEXT
    return json.loads((run / "summary.json").read_text())["final_skill"]
