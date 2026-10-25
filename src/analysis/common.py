"""Readers of a skill's test repetitions and the per-task cost ratios shared by the analyses."""
import json, math

import numpy as np

from src.utils.dataloader import gmean, task_row


def _known(e):
    return type(e.get("success")) is bool and e.get("outcome_valid", True) is True


def _by_rep(run, version):
    """{repetition: [test evaluation.json of `version` in one run dir]}, oldest first."""
    got = {}
    for f in (run / "results/evaluations/test").glob("*/evaluation.json"):
        if f.parent.name.startswith(version[:12]):
            for rep in {t.get("repetition", 0) for t in json.loads(f.read_text())["panel"]}:
                got.setdefault(rep, []).append(f)
    return got


def rep_files(dirs, version):
    """{repetition: newest evaluation.json}; the first dir that has a repetition wins."""
    out = {}
    for run in dirs:
        for rep, fs in _by_rep(run, version).items():
            newest = fs[0]
            for f in fs[1:]:
                if f.stat().st_mtime > newest.stat().st_mtime:
                    newest = f
            out.setdefault(rep, newest)
    return out


def rep_groups(dirs, version):
    """{repetition: [evaluation.json, ...]}: every file of a repetition, the later ones fill unresolved slots."""
    out = {}
    for run in dirs:
        for rep, fs in _by_rep(run, version).items():
            out.setdefault(rep, []).extend(sorted(fs, key=lambda f: f.stat().st_mtime))
    return out


def merged_episodes(files):
    """{(task, rep): episode} over one repetition's files; a known verdict is never overridden."""
    eps = {}
    for f in files:
        d = json.loads(f.read_text())
        tasks = {t["task_id"] for t in d["panel"]}
        for e in d["episodes"]:
            k = (e["task_id"], e.get("repetition", 0))
            if e["task_id"] in tasks and not (k in eps and _known(eps[k])):
                eps[k] = e
    return eps


def rows_of(f):
    d = json.loads(f.read_text())
    eps = {}
    for e in d["episodes"]:
        eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    keys = [(t["task_id"], t.get("repetition", 0)) for t in d["panel"]]
    return {k: task_row(eps.get(k, {})) for k in keys}


def merged_rows_of(files, row=None):
    """Per-slot rows of one repetition over all its files (first file's panel order)."""
    eps, rows, row = merged_episodes(files), {}, row or task_row
    for f in files:
        for t in json.loads(f.read_text())["panel"]:
            k = (t["task_id"], t.get("repetition", 0))
            if k not in rows:
                rows[k] = row(eps.get(k, {}))
    return rows


def unresolved_of(files):
    """{slot: outcome_reason} of the merged episodes that ran without a valid verdict."""
    return {k: e.get("outcome_reason") or "unresolved" for k, e in merged_episodes(files).items() if not _known(e)}


def luna_row(ep):
    """Per-episode row for GPT-6 Luna: provider-failed requests (no usage) are dropped; R = turns."""
    reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r.get("input_tokens") is not None]
    s = ep["success"] if _known(ep) else None
    if not reqs:
        return {"success": s, "N": None, "X": None, "c": None, "nreq": None, "R": None}
    n = sum(c.get("executed") is not False for r in reqs for c in r.get("tool_calls", []))
    x = sum(r["input_tokens"] for r in reqs)
    return {"success": s, "N": n, "X": x, "c": n + x / 1000, "nreq": len(reqs), "R": len(reqs)}


def accumulate(allrows):
    """{(task, 0): row} with N, X, c (and R when every run has it) averaged over the runs with a verdict and N, X."""
    per = {}
    for rows in allrows:
        for (t, _), r in rows.items():
            if r["success"] is not None and r["N"] is not None and r["X"] is not None:
                per.setdefault(t, []).append(r)
    out = {}
    for t, v in per.items():
        m = {"success": True, "N": sum(r["N"] for r in v) / len(v), "X": sum(r["X"] for r in v) / len(v),
             "c": sum(r["N"] + r["X"] / 1000 for r in v) / len(v)}
        if all(r.get("R") is not None for r in v):
            m["R"] = sum(r["R"] for r in v) / len(v)
        out[(t, 0)] = m
    return out


def mean_rows(reps):
    """(per-run rows, task means) over {rep: file}."""
    allrows = [rows_of(f) for _, f in sorted(reps.items())]
    return allrows, accumulate(allrows)


def merged_mean_rows(reps, row=None):
    """(per-run rows, task means) over {rep: [files]}."""
    allrows = [merged_rows_of(fs, row) for _, fs in sorted(reps.items())]
    return allrows, accumulate(allrows)


def geo(rows, ref, turns=False):
    """(n, geometric mean over common tasks of row / ref for c, N, X and optionally R)."""
    keys = ["c", "N", "X"] + (["R"] if turns else [])
    logs = {f: [] for f in keys}
    for k, r in rows.items():
        b = ref.get(k)
        if b is None or r["success"] is None or b["success"] is None:
            continue
        if all(x and x > 0 for x in (r["N"], r["X"], b["N"], b["X"])):
            for f in keys:
                logs[f].append(math.log(r[f] / b[f]))
    names = {"c": "rho", "N": "rho_inv", "X": "rho_trj", "R": "rho_turns"}
    return len(logs["c"]), {names[f]: gmean(np.exp(logs[f])) for f in keys}
