"""mech_edges_holm: Holm-adjusted bootstrap p-values for the rule-category contrasts of the 59 dev edges (paper 1, appendix F).

The contrasts are those of skilltext.emp_check: over the dev parent->child edges of edges_table.json without the excluded SkillAdam x
terminal-bench-2-1 cell (edges.is_excluded), an edge "contains" a rule category when its added text of that category has at least 150 characters;
for each of the ten categories with more than one edge on each side, the contrast is the unweighted mean over edges with the category minus the
mean over edges without it, on the actions and the turn-splitting terms (20 tests).  The bootstrap resamples edges within each side, as the
record does (here 20,000 resamples, seed 20261012, for p-value resolution); the two-sided p-value is 2 x the smaller tail mass of the bootstrap
differences at zero, with the +1 correction.  Holm's step-down adjustment runs over all 20 tests.

  .venv/bin/python -m analysis.mech.edges_holm      -> assets/cache/analysis/mech/edges_holm.json, paper/assets/data/mech/mech_edges_holm.md
"""
import json

import numpy as np

from analysis.mech import common as M
from analysis.mech.edges import EXCLUDED_REASON, TABLE, is_excluded

CATS = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
TERMS = [("actions", "more actions"), ("turnsplit", "turn-splitting")]
MIN_CHARS = 150          # an edge contains a category when its added text of the category has at least this many characters
RESAMPLES = 20_000
SEED = 20261012
ALPHA = 0.05


def dev_edges() -> list[dict]:
    """The dev edges of the edge table without the excluded cell."""
    return [e for e in json.loads(TABLE.read_text()) if e["split"] == "dev" and not is_excluded(e)]


def bootstrap(with_values, without_values, rng) -> tuple[float, float, float, float]:
    """(difference of means, 2.5th and 97.5th percentiles, two-sided p-value) of with-minus-without, each side resampled within itself."""
    a = np.asarray(with_values)
    b = np.asarray(without_values)
    reps = (a[rng.integers(0, len(a), (RESAMPLES, len(a)))].mean(axis=1)
            - b[rng.integers(0, len(b), (RESAMPLES, len(b)))].mean(axis=1))
    low = (np.sum(reps <= 0) + 1) / (RESAMPLES + 1)
    high = (np.sum(reps >= 0) + 1) / (RESAMPLES + 1)
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


def main():
    """Compute the 20 contrasts, adjust them and write the cache and the record."""
    edges = dev_edges()
    rng = np.random.default_rng(SEED)
    tests = []
    for category in CATS:
        has = [e for e in edges if e["mass_add"][category] >= MIN_CHARS]
        without = [e for e in edges if e["mass_add"][category] < MIN_CHARS]
        if not 1 < len(has) < len(edges) - 1:
            continue
        for term, label in TERMS:
            diff, lo, hi, p = bootstrap([e[term] for e in has], [e[term] for e in without], rng)
            tests.append({"category": category, "term": label, "with": len(has), "without": len(without), "diff": diff,
                          "ci": [lo, hi], "p": p})
    for test, adjusted in zip(tests, holm([t["p"] for t in tests])):
        test["p_holm"] = adjusted
        test["survives"] = adjusted < ALPHA
        test["unadjusted_ci_excludes_zero"] = test["ci"][0] > 0 or test["ci"][1] < 0
    M.write_json("edges_holm", {"edges": len(edges), "tests": tests})
    rows = [[t["category"], t["term"], f"{t['with']}/{t['without']}", f"{100 * t['diff']:+.1f}",
             f"[{100 * t['ci'][0]:.1f}, {100 * t['ci'][1]:.1f}]", f"{t['p']:.4f}", f"{t['p_holm']:.4f}",
             "yes" if t["unadjusted_ci_excludes_zero"] else "no", "yes" if t["survives"] else "no"]
            for t in sorted(tests, key=lambda t: t["p"])]
    M.write_record(
        "edges_holm",
        "Whether the rule-category contrasts on the per-edge four-term split (with-minus-without the category, on the actions and turn-splitting "
        "terms) survive Holm's adjustment for the 20 tests.",
        f"The {len(edges)} dev parent->child edges of assets/cache/analysis/mech/edges_table.json (Table 1 baseline runs) without SkillAdam x "
        f"terminal-bench-2-1 ({EXCLUDED_REASON}); categories and the {MIN_CHARS}-character rule as in analysis/mech/skilltext.py emp_check.",
        f"Unweighted means over edges. Bootstrap: {RESAMPLES:,} resamples, each side resampled within itself, seed {SEED}; two-sided p = 2 x the "
        f"smaller tail mass at zero with the +1 correction; Holm step-down over the {len(tests)} tests, alpha {ALPHA}. Differences in log points x 100.",
        M.md_table(["category", "term", "edges with/without", "difference", "95% bootstrap interval", "p", "p (Holm, 20 tests)",
                    "unadjusted interval excludes 0", f"survives Holm at {ALPHA}"], rows),
        ".venv/bin/python -m analysis.mech.edges_holm")


if __name__ == "__main__":
    main()
