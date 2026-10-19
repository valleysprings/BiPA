"""Shared data loading for the analyses: run paths, evaluation files, per-task rows, the optimizer registry, and the analysis cache (CACHE, write_json)."""
import json, math, hashlib, re, os
from pathlib import Path
import numpy as np
from scipy.stats import gmean as _scipy_gmean
H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
EX = H/"assets/artifacts/execution"
MODELS = ("deepseek-flash","gpt-6-luna")
BENCHES = ("tau3-bench","automationbench","terminal-bench-2-1")
METHODS = ("native-noseed","native-seed","gepa","skillopt","skilladam")
SEED_TEXT = (H/"assets/skills/seed/SKILL.md").read_text()
def gmean(values):
    """Geometric mean of positive values (a list or array), None for empty input; the one geometric mean used by every analysis module.
    Callers that hold log-ratios pass np.exp(logs)."""
    if len(values) == 0:
        return None
    return float(_scipy_gmean(values))
def vid(text):
    files = {} if text is None else {"SKILL.md": text}
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode()).hexdigest()
def sha(t): return hashlib.sha256(t.encode()).hexdigest()
def manifest():
    """The frozen submission manifest (assets/cache/analysis/submission-20261008/protocol.json) named by env BACH_SUBMISSION_MANIFEST, else None."""
    p = os.environ.get("BACH_SUBMISSION_MANIFEST")
    return json.loads(Path(p).read_text()) if p else None
def manifest_cell(bench, row, m=None):
    """The manifest cell of (bench, row) when it is complete, else None; a returned-seed row resolves to the seed row's cell for its executions."""
    c = ((m or manifest())["cells"].get(bench) or {}).get(row)
    return c if c and c["status"] == "complete" else None
def manifest_reps(bench, row, model="deepseek-flash"):
    """{repetition: evaluation.json path} pinned by the manifest for a row (the seed row's for a returned seed); None while the cell is pending."""
    m = manifest(); c = manifest_cell(bench, row, m)
    if not c or model != m["model"]: return None
    c = manifest_cell(bench, c["executions_from"], m) if c.get("executions_from") else c
    return {int(r): Path(e["file"]) for r, e in c["executions"].items()} if c else None
def current_run(method, model, bench):
    m = manifest()
    if m:
        c = manifest_cell(bench, method, m)
        return Path(c["run"]) if c and model == m["model"] else None
    return latest_run(method, model, bench)
def latest_run(method, model, bench):
    """The run a cell reports without a manifest (the newest finished native run)."""
    cell = EX/model/method/bench
    if not cell.is_dir(): return None
    runs = [d for d in cell.iterdir() if d.is_dir() and d.name.startswith("native-") and not (d/"ABORTED.md").exists()]
    done = [d for d in runs if (d/"summary.json").is_file()]
    key = lambda d: ((d/"summary.json") if d in done else (d/"native_protocol.json")).stat().st_mtime if ((d/"summary.json") if d in done else (d/"native_protocol.json")).exists() else 0
    return max(done or runs, key=key, default=None)
def reloc(p):
    s = str(p)
    m = re.match(re.escape(str(H))+r"/assets/artifacts/execution/runs/([a-z0-9_]+)/([a-z0-9-]+)/opencode/([a-z0-9-]+)/([^/]+)(.*)", s)
    if m:
        arm, model, bench, run, rest = m.groups()
        arm = {"none":"native-noseed","noskill":"native-noseed","seed":"native-seed"}.get(arm, arm)
        return Path(f"{EX}/{model}/{arm}/{bench}/{run}{rest}")
    m = re.match(re.escape(str(H))+r"/runs/(skilladam)/([a-z0-9-]+)/([a-z0-9-]+)/([^/]+)(.*)", s)
    if m:
        k, model, bench, run, rest = m.groups()
        return Path(f"{EX}/{model}/{k}/{bench}/{run}{rest}")
    return Path(s)
def eval_file(run, split, version):
    fs = [f for f in (run/"results/evaluations"/split).glob("*/evaluation.json") if f.parent.name.startswith(version[:12])]
    return max(fs, key=lambda f: f.stat().st_mtime) if fs else None
def task_row(ep):
    cap, seen, reqs = ep.get("capture") or {}, set(), []
    for r in ep.get("requests", []):
        if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
    calls = [c for r in reqs for c in r.get("tool_calls", [])]
    n = sum(c.get("executed") is True for c in calls)
    if cap.get("invocations_complete") is not True or any(c.get("executed") not in (True, False) for c in calls): n = None
    fin = lambda v: isinstance(v,(int,float)) and math.isfinite(v) and v>=0
    x = sum(r["input_tokens"] for r in reqs) if cap.get("requests_complete") is True and all(fin(r["input_tokens"]) for r in reqs) else None
    return {"success": ep.get("success"), "N": n, "X": x, "c": n + x/1000 if n is not None and x is not None else None, "nreq": len(reqs)}
def evaluation(run, split, version):
    f = eval_file(run, split, version)
    if not f: return None
    d = json.loads(f.read_text()); eps = {}
    for e in d["episodes"]: eps.setdefault((e["task_id"], e.get("repetition",0)), e)
    return {(t["task_id"], t.get("repetition",0)): task_row(eps.get((t["task_id"], t.get("repetition",0)), {})) for t in d["panel"]}, f, d
def pair(rows, seed):
    out = {}; logs = {k: [] for k in ("c","N","X")}; per = {}
    for key, r in rows.items():
        b = seed.get(key) if seed else None
        ok = b is not None and r["success"] is not None and b["success"] is not None
        for k in logs:
            if ok and r[k] and b[k]:
                logs[k].append(math.log(r[k]/b[k])); per.setdefault(key, {})[k] = math.log(r[k]/b[k])
    for k, v in logs.items():
        out["rho_"+k] = gmean(np.exp(v)); out["n_"+k] = len(v)
    return out, per
def final_skill(method, run):
    if method == "native-noseed": return None
    if method == "native-seed": return SEED_TEXT
    return json.loads((run/"summary.json").read_text())["final_skill"]
def q(p, xs):
    xs = sorted(xs);
    if not xs: return None
    i = p*(len(xs)-1); lo=int(math.floor(i)); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(i-lo)

# ---- added by group finding-figs: output root, optimizer names, returned skills, analysis imports ----
import sys
sys.path.insert(0, str(H/"bipa"))  # qualified src imports; do not expose src/utils as top-level utils
ART = H/"assets/artifacts"
def out_dir(module):
    """assets/cache/analysis/<module>/ (created): derived intermediates and caches, rebuildable by the scripts; the paper's numbers go through paperout.record()."""
    d = H/"assets/cache/analysis"/module; d.mkdir(parents=True, exist_ok=True); return d
OPTIMIZERS = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}  # display name -> method directory
def returned_skills(model, bench, names=OPTIMIZERS):
    """{display name: (run, final skill text)} of the optimizers that finished in this cell with a skill other than the seed (returned seeds have no row)."""
    out = {}
    for disp, meth in names.items():
        run = current_run(meth, model, bench)
        if not run or not (run/"summary.json").is_file(): continue
        t = final_skill(meth, run)
        if t != SEED_TEXT: out[disp] = (run, t)
    return out

# ---- added by group new-analyses: DeepSeek cell root, short bench names ----
DS = EX/"deepseek-flash"
BN = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}

# ---- analysis cache ----
CACHE = H / "assets/cache/analysis/main"
def write_json(name, obj):
    """Write <name>.json under the analysis cache dir (CACHE)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1))
    return path
