"""Accepted vs rejected optimizer edges: does the editor itself add the input-token tax, or does selection ratchet it?

Analysis role: each of the 59 dev parent->child edges of the current runs (native-20261008-https; table built by
analysis/mech/edges_build.py from the logs) is a scored candidate against its parent on the same dev tasks.
Accepted and rejected candidates are compared on the change in log input tokens (dlogX = actions + turn-splitting + content + text + resid)
and on the dev success change (dQ, points).
 - If rejected edges also have dlogX > 0, the editor adds the tax and success-only selection lets it through.
 - If accepted edges have a larger dlogX than rejected ones, selection itself ratchets cost.
Per group (all / each optimizer x accepted / rejected): edge count, unweighted mean over edges and task-pair-weighted mean (weights n_pair) of dlogX and its terms,
mean dQ, and a bootstrap 95% interval over edges (10,000 resamples, seed 20261012, resampling within accepted and within rejected) for accepted minus rejected dlogX.
Outputs: assets/cache/analysis/mech/edges.json and paper/assets/data/mech/mech_edges.md. All values are x100 log points in the record.
"""
import json

import numpy as np

from analysis.mech import common as C

TABLE = C.CACHE / "edges_table.json"
# Excluded cell: SkillAdam on terminal-bench-2-1. Two versions of that run match the same patch (one is the other plus a later line), so the parent of its
# edits is ambiguous in the logs; the table labels 11 of its 14 dev edges with the wrong parent. All its edges are dropped from every analysis here.
EXCLUDED_CELL = ("skilladam", "terminal-bench-2-1")
EXCLUDED_REASON = "two versions of that run match the same patch, so the parent of its edits is ambiguous in the logs"


def is_excluded(row):
    """True for the edges of the excluded cell (SkillAdam x terminal-bench-2-1)."""
    return (row["opt"], row["bench"]) == EXCLUDED_CELL
TERMS = ["dlogX", "actions", "turnsplit", "content", "text", "resid", "dQ"]
OPTS = ["gepa", "skillopt", "skilladam"]
NAMES = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
RESAMPLES = 10000
SEED = 20261012


def load_dev_edges():
    """The dev edges (split == 'dev') of the edge table, without the excluded SkillAdam x terminal-bench-2-1 cell."""
    rows = json.loads(TABLE.read_text())
    return [r for r in rows if r["split"] == "dev" and not is_excluded(r)]


def group_stats(edges):
    """Count and mean of every term, unweighted and weighted by n_pair (task pairs of the edge)."""
    out = {"n_edges": len(edges), "n_pairs": int(sum(e["n_pair"] for e in edges))}
    if not edges:
        return out
    weights = np.array([e["n_pair"] for e in edges], float)
    for term in TERMS:
        values = np.array([e[term] for e in edges], float)
        out[term] = float(values.mean())
        out[term + "_w"] = float(np.average(values, weights=weights))
    return out


def boot_diff(accepted, rejected, term, weighted, rng):
    """Bootstrap 95% interval over edges of mean(accepted) - mean(rejected) for one term; the two groups are resampled separately."""
    def mean_of(group):
        values = np.array([e[term] for e in group], float)
        if not weighted:
            return values.mean()
        return np.average(values, weights=[e["n_pair"] for e in group])

    point = mean_of(accepted) - mean_of(rejected)
    if len(accepted) < 2 or len(rejected) < 2:
        return point, None, None
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        a = [accepted[j] for j in rng.integers(0, len(accepted), len(accepted))]
        r = [rejected[j] for j in rng.integers(0, len(rejected), len(rejected))]
        draws[i] = mean_of(a) - mean_of(r)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def boot_mean(group, term, rng):
    """Bootstrap 95% interval over edges of the unweighted mean of one term within one group."""
    values = np.array([e[term] for e in group], float)
    if len(values) < 2:
        return None, None
    draws = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def compute():
    """All groups, the accepted-minus-rejected differences and the intervals."""
    edges = load_dev_edges()
    rng = np.random.default_rng(SEED)
    result = {"n_dev_edges": len(edges), "groups": {}, "diff": {}}
    scopes = {"all": edges}
    for opt in OPTS:
        scopes[opt] = [e for e in edges if e["opt"] == opt]
    for scope, members in scopes.items():
        accepted = [e for e in members if e["accepted"]]
        rejected = [e for e in members if not e["accepted"]]
        result["groups"][scope] = {"accepted": group_stats(accepted), "rejected": group_stats(rejected)}
        # interval of the mean dlogX within each group (is the rejected mean itself above zero?)
        for name, group in (("accepted", accepted), ("rejected", rejected)):
            low, high = boot_mean(group, "dlogX", rng)
            result["groups"][scope][name]["dlogX_ci"] = [low, high]
        diff = {}
        if accepted and rejected:
            for term in TERMS:
                for weighted in (False, True):
                    key = term + ("_w" if weighted else "")
                    point, low, high = boot_diff(accepted, rejected, term, weighted, rng)
                    diff[key] = [point, low, high]
        result["diff"][scope] = diff
    return result


def pct(x):
    """x100 with sign, one decimal."""
    return "n/a" if x is None else f"{100 * x:+.1f}"


def ci(lo_hi):
    """[lo, hi] x100."""
    return "n/a" if lo_hi[0] is None else f"[{100 * lo_hi[0]:.1f}, {100 * lo_hi[1]:.1f}]"


def tables(res):
    """Markdown tables: group means (unweighted and weighted), then accepted minus rejected."""
    head = ["scope", "status", "edges", "pairs", "dlogX", "[95%]", "actions", "turn-split", "content", "text", "resid", "dQ (pts)"]
    rows_u, rows_w = [], []
    for scope, groups in res["groups"].items():
        for status in ("accepted", "rejected"):
            g = groups[status]
            if g["n_edges"] == 0:
                rows_u.append([scope, status, 0, 0] + ["-"] * 8)
                continue
            rows_u.append([scope, status, g["n_edges"], g["n_pairs"], pct(g["dlogX"]), ci(g["dlogX_ci"]), pct(g["actions"]), pct(g["turnsplit"]),
                           pct(g["content"]), pct(g["text"]), pct(g["resid"]), f"{g['dQ']:+.1f}"])
            rows_w.append([scope, status, g["n_edges"], g["n_pairs"], pct(g["dlogX_w"]), "", pct(g["actions_w"]), pct(g["turnsplit_w"]),
                           pct(g["content_w"]), pct(g["text_w"]), pct(g["resid_w"]), f"{g['dQ_w']:+.1f}"])
    dhead = ["scope", "weighting", "d dlogX", "d actions", "d turn-split", "d content", "d text", "d dQ (pts)"]
    drows = []
    for scope, diff in res["diff"].items():
        if not diff:
            continue
        for label, suffix in (("edges", ""), ("pairs", "_w")):
            cells = []
            for term in ("dlogX", "actions", "turnsplit", "content", "text"):
                p, lo, hi = diff[term + suffix]
                cells.append(f"{pct(p)} {ci((lo, hi))}")
            p, lo, hi = diff["dQ" + suffix]
            cells.append(f"{p:+.1f} [{lo:.1f}, {hi:.1f}]" if lo is not None else f"{p:+.1f}")
            drows.append([scope, label] + cells)
    return (C.md_table(head, rows_u), C.md_table(head, rows_w), C.md_table(dhead, drows))


def run():
    """Compute, write json and record."""
    res = compute()
    C.write_json("edges", res)
    unweighted, weighted, difference = tables(res)
    table = ("Mean change per edge against its parent, dev panel, log points x100 (dQ in success points). Unweighted over edges:\n\n" + unweighted +
             "\n\nTask-pair-weighted (weights n_pair):\n\n" + weighted +
             "\n\nAccepted minus rejected (bootstrap 95% over edges, 10,000 resamples, accepted and rejected resampled separately):\n\n" + difference)
    C.write_record(
        "edges",
        "Whether rejected candidates also read more than their parents (the editor adds the input-token tax and success-only selection lets it through) "
        "or whether accepted ones read more than rejected ones (selection ratchets cost). Each edge is a scored candidate against its parent on the same dev tasks.",
        "Current runs native-20261008-https (GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1); 59 dev edges from "
        "`assets/cache/analysis/mech/edges_table.json` (built by `analysis/mech/edges_build.py`), accepted and rejected candidates. "
        "Excluded: all 14 SkillAdam terminal-bench-2-1 dev edges, because " + EXCLUDED_REASON + ".",
        "dlogX = change in mean log input tokens per task of child vs parent on the shared dev tasks, split into actions, turn-splitting, content, text (+ resid). "
        "Group means are unweighted over edges and weighted by n_pair. Differences accepted minus rejected carry a bootstrap 95% interval over edges.",
        table,
        ".venv/bin/python -m analysis.mech.edges",
        "Source: `analysis/mech/edges.py`; numbers in `assets/cache/analysis/mech/edges.json`. GEPA has no rejected dev edge in the table (all 22 are accepted), so its rows have no comparison.")
    return res


if __name__ == "__main__":
    run()
