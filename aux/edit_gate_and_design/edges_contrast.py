"""Accepted vs rejected optimizer edges and the Holm-adjusted rule-category contrasts, on the edge table of
aux.rule_coding.edges (59 dev parent->child edges of native-20261008-https; SkillAdam x terminal-bench-2-1 excluded).

Accepted and rejected candidates are compared on the change in log input tokens (dlogX = actions + turn-splitting + content + text + resid)
and on the dev success change (dQ, points), with a bootstrap 95% interval over edges (10,000 resamples, seed 20261012).

  python -m aux.edit_gate_and_design.edges_contrast        -> CACHE/analysis/main/edges.json, data/mech/mech_edges.md
  python -m aux.edit_gate_and_design.edges_contrast holm   -> CACHE/analysis/main/edges_holm.json, data/mech/mech_edges_holm.md
"""
import sys

import numpy as np

from aux.rule_coding.edges import EXCLUDED_REASON, OPTS, load_dev_edges
from src.utils import dataloader, logger

TERMS = ["dlogX", "actions", "turnsplit", "content", "text", "resid", "dQ"]
RESAMPLES = 10000
SEED = 20261012


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
    return (logger.md_table(head, rows_u), logger.md_table(head, rows_w), logger.md_table(dhead, drows))


def run():
    """Compute, write json and record."""
    res = compute()
    dataloader.write_json("edges", res)
    unweighted, weighted, difference = tables(res)
    table = ("Mean change per edge against its parent, dev panel, log points x100 (dQ in success points). Unweighted over edges:\n\n" + unweighted +
             "\n\nTask-pair-weighted (weights n_pair):\n\n" + weighted +
             "\n\nAccepted minus rejected (bootstrap 95% over edges, 10,000 resamples, accepted and rejected resampled separately):\n\n" + difference)
    logger.write_record(
        "edges",
        "Whether rejected candidates also read more than their parents (the editor adds the input-token tax and success-only selection lets it through) "
        "or whether accepted ones read more than rejected ones (selection ratchets cost). Each edge is a scored candidate against its parent on the same dev tasks.",
        "Current runs native-20261008-https (GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1); 59 dev edges from "
        "`CACHE/analysis/main/edges_table.json` (built by `aux.rule_coding.edges build`), accepted and rejected candidates. "
        "Excluded: all 14 SkillAdam terminal-bench-2-1 dev edges, because " + EXCLUDED_REASON + ".",
        "dlogX = change in mean log input tokens per task of child vs parent on the shared dev tasks, split into actions, turn-splitting, content, text (+ resid). "
        "Group means are unweighted over edges and weighted by n_pair. Differences accepted minus rejected carry a bootstrap 95% interval over edges.",
        table,
        "python -m aux.edit_gate_and_design.edges_contrast",
        "Source: `aux/edit_gate_and_design/edges_contrast.py`; numbers in `CACHE/analysis/main/edges.json`. GEPA has no rejected dev edge in the table (all 22 are accepted), so its rows have no comparison.")
    return res


CATS = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
HOLM_TERMS = [("actions", "more actions"), ("turnsplit", "turn-splitting")]
MIN_CHARS = 150          # an edge contains a category when its added text of the category has at least this many characters
HOLM_RESAMPLES = 20_000
HOLM_SEED = 20261012
ALPHA = 0.05


def bootstrap(with_values, without_values, rng) -> tuple[float, float, float, float]:
    """(difference of means, 2.5th and 97.5th percentiles, two-sided p-value) of with-minus-without, each side resampled within itself."""
    a = np.asarray(with_values)
    b = np.asarray(without_values)
    reps = (a[rng.integers(0, len(a), (HOLM_RESAMPLES, len(a)))].mean(axis=1)
            - b[rng.integers(0, len(b), (HOLM_RESAMPLES, len(b)))].mean(axis=1))
    low = (np.sum(reps <= 0) + 1) / (HOLM_RESAMPLES + 1)
    high = (np.sum(reps >= 0) + 1) / (HOLM_RESAMPLES + 1)
    p = min(1.0, 2 * min(low, high))
    return float(a.mean() - b.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), float(p)


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values (monotone, capped at 1), in the input order."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_holm():
    """Compute the 20 contrasts, adjust them and write the cache and the record."""
    edges = load_dev_edges()
    rng = np.random.default_rng(HOLM_SEED)
    tests = []
    for category in CATS:
        has = [e for e in edges if e["mass_add"][category] >= MIN_CHARS]
        without = [e for e in edges if e["mass_add"][category] < MIN_CHARS]
        if not 1 < len(has) < len(edges) - 1:
            continue
        for term, label in HOLM_TERMS:
            diff, lo, hi, p = bootstrap([e[term] for e in has], [e[term] for e in without], rng)
            tests.append({"category": category, "term": label, "with": len(has), "without": len(without), "diff": diff,
                          "ci": [lo, hi], "p": p})
    for test, adjusted in zip(tests, holm([t["p"] for t in tests])):
        test["p_holm"] = adjusted
        test["survives"] = adjusted < ALPHA
        test["unadjusted_ci_excludes_zero"] = test["ci"][0] > 0 or test["ci"][1] < 0
    dataloader.write_json("edges_holm", {"edges": len(edges), "tests": tests})
    rows = [[t["category"], t["term"], f"{t['with']}/{t['without']}", f"{100 * t['diff']:+.1f}",
             f"[{100 * t['ci'][0]:.1f}, {100 * t['ci'][1]:.1f}]", f"{t['p']:.4f}", f"{t['p_holm']:.4f}",
             "yes" if t["unadjusted_ci_excludes_zero"] else "no", "yes" if t["survives"] else "no"]
            for t in sorted(tests, key=lambda t: t["p"])]
    logger.write_record(
        "edges_holm",
        "Whether the rule-category contrasts on the per-edge four-term split (with-minus-without the category, on the actions and turn-splitting "
        "terms) survive Holm's adjustment for the 20 tests.",
        f"The {len(edges)} dev parent->child edges of CACHE/analysis/main/edges_table.json (Table 1 baseline runs) without SkillAdam x "
        f"terminal-bench-2-1 ({EXCLUDED_REASON}); an edge has a category when it adds at least {MIN_CHARS} characters of it.",
        f"Unweighted means over edges. Bootstrap: {HOLM_RESAMPLES:,} resamples, each side resampled within itself, seed {HOLM_SEED}; two-sided p = 2 x the "
        f"smaller tail mass at zero with the +1 correction; Holm step-down over the {len(tests)} tests, alpha {ALPHA}. Differences in log points x 100.",
        logger.md_table(["category", "term", "edges with/without", "difference", "95% bootstrap interval", "p", "p (Holm, 20 tests)",
                    "unadjusted interval excludes 0", f"survives Holm at {ALPHA}"], rows),
        "python -m aux.edit_gate_and_design.edges_contrast holm")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "holm":
        run_holm()
    else:
        run()
