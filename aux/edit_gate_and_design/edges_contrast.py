"""Accepted vs rejected optimizer edges, the Holm-adjusted rule-category contrasts and the mech_edges figure, on the edge table of
src.analysis.edges (59 dev parent->child edges of native-20261008-https; SkillAdam x terminal-bench-2-1 excluded).

Accepted and rejected candidates are compared on the change in log input tokens (dlogX = actions + turn-splitting + content + text + resid)
and on the dev success change (dQ, points), with a bootstrap 95% interval over edges (10,000 resamples, seed 20261012).

  python -m aux.edit_gate_and_design.edges_contrast        -> CACHE/analysis/main/edges.json, data/mech/mech_edges.md
  python -m aux.edit_gate_and_design.edges_contrast holm   -> CACHE/analysis/main/edges_holm.json, data/mech/mech_edges_holm.md
  python -m aux.edit_gate_and_design.edges_contrast draw   -> fig/mech/mech_edges.{pdf,svg}, data/mech/mech_edges_fig.md
"""
import json
import sys

import numpy as np

from src.analysis.edges import EXCLUDED_REASON, OPTS, load_dev_edges
from src.utils import dataloader, fig, logger

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
        "`CACHE/analysis/main/edges_table.json` (built by `src.analysis.edges build`), accepted and rejected candidates. "
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
        f"terminal-bench-2-1 ({EXCLUDED_REASON}); categories and the {MIN_CHARS}-character rule as in category_presence.",
        f"Unweighted means over edges. Bootstrap: {HOLM_RESAMPLES:,} resamples, each side resampled within itself, seed {HOLM_SEED}; two-sided p = 2 x the "
        f"smaller tail mass at zero with the +1 correction; Holm step-down over the {len(tests)} tests, alpha {ALPHA}. Differences in log points x 100.",
        logger.md_table(["category", "term", "edges with/without", "difference", "95% bootstrap interval", "p", "p (Holm, 20 tests)",
                    "unadjusted interval excludes 0", f"survives Holm at {ALPHA}"], rows),
        "python -m aux.edit_gate_and_design.edges_contrast holm")


GROUPS = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
# plain names of the keyword groups (patterns in src.analysis.edges.CATEGORY_PATTERNS)
NAME = {"verify": "verify", "reread": "re-read state", "enumerate": "check every item", "test": "run tests", "format": "output format",
        "business": "domain rules", "scope_stop": "stop or budget", "economize": "save calls or reads", "serialize": "one at a time", "other": "other"}
FIG_TERMS = [("actions", "more actions", "#0072B2", "o"), ("turnsplit", "turn-splitting", "#E69F00", "s")]


def category_presence(dev):
    """Per keyword group: edges with / without >= 150 added characters of it, and the with-minus-without contrast of actions, turn-splitting
    and dlogX (unweighted means over edges) with a 3,000-resample bootstrap over edges, each side resampled within itself, seed 0."""
    rng = np.random.default_rng(0)
    present = {}
    for c in GROUPS:
        has = [e for e in dev if e["mass_add"][c] >= 150]
        present[c] = {"edges_with": len(has), "edges_without": len(dev) - len(has)}
        if 1 < len(has) < len(dev) - 1:
            no = [e for e in dev if e["mass_add"][c] < 150]
            present[c]["with"] = {t: float(np.mean([e[t] for e in has])) for t in ("actions", "turnsplit", "dlogX")}
            present[c]["without"] = {t: float(np.mean([e[t] for e in no])) for t in ("actions", "turnsplit", "dlogX")}
            present[c]["contrast"] = {}
            for t in ("actions", "turnsplit", "dlogX"):
                diff = present[c]["with"][t] - present[c]["without"][t]
                reps = []
                for _ in range(3000):
                    a = [has[i][t] for i in rng.integers(0, len(has), len(has))]
                    b = [no[i][t] for i in rng.integers(0, len(no), len(no))]
                    reps.append(float(np.mean(a) - np.mean(b)))
                present[c]["contrast"][t] = [diff, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    return present


def draw_edges():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pres = category_presence(load_dev_edges())
    holm = {(t["category"], t["term"]): t for t in json.load(open(dataloader.CACHE / "edges_holm.json"))["tests"]}
    term_name = {"actions": "more actions", "turnsplit": "turn-splitting"}
    # rows sorted by their largest difference, biggest effect on top
    rows = sorted(GROUPS, key=lambda g: -max(pres[g]["contrast"]["actions"][0], pres[g]["contrast"]["turnsplit"][0]))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"], "font.size": 7,
                         "mathtext.fontset": "stix"})
    f, ax = plt.subplots(figsize=(4.5, 2.8))
    off = {"actions": -0.17, "turnsplit": 0.17}
    ax.axvline(0, color="black", lw=1.2, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.92", lw=0.5)
    table = []
    for i, g in enumerate(rows):
        line = [NAME.get(g, g), pres[g]["edges_with"], pres[g]["edges_without"]]
        for key, _, col, mk in FIG_TERMS:
            d, lo, hi = pres[g]["contrast"][key]
            y = i + off[key]
            h = holm[(g, term_name[key])]
            thin = h["without"] < 5 and h["survives"]  # survives, but the comparison group is tiny
            if thin:
                ax.scatter([d], [y], s=20, marker=mk, facecolor="white", edgecolor="0.55", lw=0.9, zorder=3)
                ax.annotate("only 2 edits without", (d, y), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                            fontsize=6.5, color="0.4")
            elif h["survives"]:
                ax.scatter([d], [y], s=30, marker=mk, color=col, edgecolor="black", lw=0.6, zorder=4)
                ax.annotate(f"{d:+.2f}", (d, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=7, weight="bold")
            else:
                ax.scatter([d], [y], s=16, marker=mk, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
            line += [d, lo, hi]
        table.append(line)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NAME.get(g, g)}  {pres[g]['edges_with']}/{pres[g]['edges_without']}" for g in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.2, 0.26)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xticklabels(["$-$0.2", "$-$0.1", "0", "0.1", "0.2"])
    ax.set_xlabel("edits with the group minus edits without, change in log input tokens")
    ax.spines[["top", "right", "left"]].set_visible(False)
    h = [plt.Line2D([], [], marker=mk, ls="none", color=col, ms=4.5, mec="black", mew=0.5) for _, _, col, mk in FIG_TERMS]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", ms=4.5, mew=0.9))
    f.legend(h, ["more actions", "turn-splitting", "hollow: not reliable after correcting for 20 comparisons"],
             loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.55, 1.0))
    f.subplots_adjust(left=0.24, right=0.985, top=0.9, bottom=0.17)
    fig.save_figure(f, "edges")
    body = logger.md_table(["group", "with", "without", "d actions", "[95%]", "d turn-splitting", "[95%]"],
                      [[r[0], r[1], r[2], f"{r[3]*100:+.1f}", f"[{r[4]*100:.1f}, {r[5]*100:.1f}]", f"{r[6]*100:+.1f}", f"[{r[7]*100:.1f}, {r[8]*100:.1f}]"] for r in table])
    logger.write_record("edges_fig", "Per keyword group of the added text, the with-minus-without difference of the actions and turn-splitting terms of the change in log input tokens per edit (59 dev edits), with 95% bootstrap intervals over edits. Replaces the appendix table of edit categories.",
                   "59 canonical dev edges of native-20261008-https (SkillAdam x TB2 excluded); an edit is 'with' a group when its added text holds >= 150 characters of the group's keywords.",
                   "Unweighted mean over edits, with minus without; 95% bootstrap over edits (3000 resamples, each side within itself). Values x100 log points in this table; the figure axis is in log units.",
                   body, "python -m aux.edit_gate_and_design.edges_contrast draw")
    return table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "holm":
        run_holm()
    elif cmd == "draw":
        for r in draw_edges():
            print(r)
    else:
        run()
