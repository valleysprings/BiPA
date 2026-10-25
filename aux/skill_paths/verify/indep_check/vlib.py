"""Independent verification helpers (written without reading the extractors)."""
import json, hashlib, os, difflib, glob, re, time
from pathlib import Path

from harness_bench.eval import version_from_files

from src.layout import EXECUTION, OUT

EX = EXECUTION
PATHS = OUT / "skill_paths/paths"
CACHE = OUT / "skill_paths/evalcache"


def vid(text):
    return version_from_files({"SKILL.md": text})["version_id"]


def rd(p):
    return Path(p).read_text(encoding="utf-8")


def eval_dirs(run, split, v12):
    """[(panel12, dir)] of results/evaluations/<split>/<v12>-*"""
    base = Path(run) / "results/evaluations" / split
    out = []
    if not base.is_dir():
        return out
    for d in sorted(os.listdir(base)):
        if d.startswith(v12 + "-"):
            out.append((d.split("-", 1)[1], base / d))
    return out


def summarize_eval(path, tag=None):
    """Summary of one evaluation.json (cached on disk)."""
    path = Path(path)
    key = hashlib.md5(str(path).encode()).hexdigest() + ".json"
    cf = CACHE / key
    mt = path.stat().st_mtime
    if cf.exists():
        c = json.loads(cf.read_text())
        if c.get("mtime") == mt:
            return c
    d = json.load(open(path))
    eps = d["episodes"]
    n = len(eps)
    ns = sum(1 for e in eps if e["success"] is True)
    nf = sum(1 for e in eps if e["success"] is False)
    nu = sum(1 for e in eps if e["success"] is None)
    # input tokens per episode: sum over requests
    toks = []
    for e in eps:
        toks.append(sum((r.get("input_tokens") or 0) for r in e.get("requests", [])))
    mean_tok = sum(toks) / n if n else None
    # alternative: mean over known only
    kn = [t for t, e in zip(toks, eps) if e["success"] is not None]
    mean_tok_known = sum(kn) / len(kn) if kn else None
    nreq0 = sum(1 for e in eps if not e.get("requests"))
    out = {
        "path": str(path), "mtime": mt, "version_id": d.get("version_id"), "split": d.get("split"),
        "panel_len": len(d.get("panel", [])), "panel_tasks": sorted(set(p["task_id"] for p in d.get("panel", []))),
        "n": n, "n_success": ns, "n_false": nf, "n_unknown": nu,
        "mean_tok": mean_tok, "mean_tok_known": mean_tok_known, "eps_without_requests": nreq0,
        "slot": d.get("slot_accounting"), "ep_tasks": [e["task_id"] for e in eps],
        "ep_success": [e["success"] for e in eps], "ep_tok": toks,
        "reps": sorted(set(e.get("repetition", 0) for e in eps)),
        "outcome_valid_false": sum(1 for e in eps if e.get("outcome_valid") is False),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def line_diff(parent, child):
    a = [l.strip() for l in parent.splitlines() if l.strip()]
    b = [l.strip() for l in child.splitlines() if l.strip()]
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += b[j1:j2]
        if op in ("delete", "replace"):
            removed += a[i1:i2]
    return added, removed


def words(t):
    return len(t.split())
