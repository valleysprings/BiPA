"""Accepted vs rejected optimizer edges: does the editor itself add the input-token tax, or does selection ratchet it?

Analysis role: each of the 59 dev parent->child edges of the current runs (native-20261008-https; table built by
the `build` subcommand from the logs) is a scored candidate against its parent on the same dev tasks.
Accepted and rejected candidates are compared on the change in log input tokens (dlogX = actions + turn-splitting + content + text + resid)
and on the dev success change (dQ, points).
 - If rejected edges also have dlogX > 0, the editor adds the tax and success-only selection lets it through.
 - If accepted edges have a larger dlogX than rejected ones, selection itself ratchets cost.
Per group (all / each optimizer x accepted / rejected): edge count, unweighted mean over edges and task-pair-weighted mean (weights n_pair) of dlogX and its terms,
mean dQ, and a bootstrap 95% interval over edges (10,000 resamples, seed 20261012, resampling within accepted and within rejected) for accepted minus rejected dlogX.

Subcommands (from bipa):
  (bare)  the accepted-vs-rejected comparison -> assets/cache/analysis/rq/edges.json, paper/assets/data/mech/mech_edges.md (x100 log points in the record)
  build   (merged from edges_build.py) rebuild the parent->child edge table of the baseline optimizer runs from the run artifacts
          -> assets/cache/analysis/rq/edges_table.json
  holm    (merged from edges_holm.py) Holm-adjusted bootstrap p-values for the rule-category contrasts of the dev edges (appendix F)
          -> assets/cache/analysis/rq/edges_holm.json, paper/assets/data/mech/mech_edges_holm.md
  draw    (merged from edges_fig.py) the mech_edges figure: per keyword group, the with-minus-without difference of the actions and
          turn-splitting terms -> paper/assets/fig/mech/mech_edges.{pdf,svg}, paper/assets/data/mech/mech_edges_fig.md
"""
import argparse
import difflib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

from src.utils import fig, logger, dataloader
from src.utils.dataloader import BENCHES, DS, SEED_TEXT, vid
from src.graph import episode as GE
TABLE = dataloader.CACHE / "edges_table.json"
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
        "`assets/cache/analysis/rq/edges_table.json` (built by `analysis/rq3/edges.py`), accepted and rejected candidates. "
        "Excluded: all 14 SkillAdam terminal-bench-2-1 dev edges, because " + EXCLUDED_REASON + ".",
        "dlogX = change in mean log input tokens per task of child vs parent on the shared dev tasks, split into actions, turn-splitting, content, text (+ resid). "
        "Group means are unweighted over edges and weighted by n_pair. Differences accepted minus rejected carry a bootstrap 95% interval over edges.",
        table,
        ".venv/bin/python -m src.analysis.rq3.edges",
        "Source: `analysis/rq3/edges.py`; numbers in `assets/cache/analysis/rq/edges.json`. GEPA has no rejected dev edge in the table (all 22 are accepted), so its rows have no comparison.")
    return res


def main():
    """Bare run = the accepted-vs-rejected record; build / holm / draw are the merged subcommands."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["build", "holm", "draw"])
    cmd = ap.parse_args().cmd
    if cmd == "build":
        run_build()
    elif cmd == "holm":
        run_holm()
    elif cmd == "draw":
        for r in draw_edges():
            print(r)
    else:
        run()


# ---------------------------------------------------------------------------------------------------------------------------------
# Edge-table builder (merged from edges_build.py; the `build` subcommand)
#
# Rebuild the parent->child edge table of the baseline optimizer runs from the run artifacts (paper 1, Sec. 4.4 / 4.5): every version an
# optimizer proposes is a child of an earlier version (its parent). For each such edge we pair the evaluation episodes of parent and child
# on the tasks both ran (same split and same panel), decompose the per-task change of input tokens
#     dlogX = actions + turnsplit + content + text + resid
# and attach what the text diff added, in rule categories. The bare run of this module (accepted vs rejected) and skilltext.py (Sec. 4.5
# category contrasts) read the table written here.
#
# This is a port of .scratch/evidence-20261011/emp_baseline_1_edges.py, emp_baseline_2_pairs.py and emp_baseline_3_analyze.py (the edge-table
# part) with one rule changed: for SkillAdam the patch of an iteration is the LAST attempt (patch_attempt_NN.json) that holds a parseable
# patch; the scratch builder read only attempt 01, so an iteration whose attempt 01 is "(unparseable)" lost its edge. The patch only matters
# for matching a SkillAdam iteration to its candidate version (the candidate whose new lines occur in the patch); every number of an edge
# comes from the evaluation episodes.
#
# Edge terms (per task, then mean over tasks with a valid lever model; T_p, T_c = parent / child episode):
#   e_R = s_F + (1 - s_F) (2R - 1) / (R - 1), s_F and R the means of parent and child
#   actions = e_R dlogN, turnsplit = e_R (dlogR - dlogN), content = (1 - s_F) dlogh, text = s_F dlogF, resid = dlogX - e_R dlogR - content - text.
# ---------------------------------------------------------------------------------------------------------------------------------
TAG = "native-20261008-https"
# Per-task rows of the scratch builder (emp_baseline_2_pairs.py). The seed evaluations reused by the current runs recorded their request files
# under the run native-20261004, which the 2026-10-12 disk cleanup deleted (.trash/cleanup-20261012.md); for exactly those panels the rows of this
# frozen file stand in (read-only). Every other panel, and every edge the file does not hold, is rebuilt from the run artifacts.
FROZEN_ROWS = dataloader.H / ".scratch/evidence-20261011/emp_baseline_pairs.json"
OPTIMIZERS = ("skillopt", "gepa", "skilladam")
OUT = dataloader.CACHE / "edges_table.json"
UNPARSEABLE = "(unparseable)"
MIN_TERM_TASKS = 3  # an edge-panel needs at least this many tasks with a valid lever model
EDGE_TERMS = ["dlogX", "actions", "turnsplit", "content", "text", "resid"]

# Rule categories of an added or removed skill line (line level, multi-label, case-insensitive).
CATEGORY_PATTERNS = {
    "verify": r"verif|confirm|double-check|validat|sanity|evidence|prove|check (that|whether|the|each|every)|audit|acceptance|oracle|grader|assert|cross-check",
    "reread": r"re-?read|read (it |them |back )?again|read back|re-?fetch|re-?query|re-?check|re-?run|re-?inspect|re-?open|fresh (read|get|state|fetch|query)|after (any |every )?(write|mutat)|persisted state",
    "enumerate": r"\bevery\b|\beach\b|enumerat|checklist|exhaust|list (all|every)|line[- ]by[- ]line|for each|silently (drop|omit)|\ball (required|items|records|deliverables|entities)",
    "test": r"\btests?\b|pytest|unit test|reproduc|verifier|smoke|tiny (sample|input)|edge case|malformed|stress|test_outputs",
    "format": r"format|exact(ly)? (name|file|path|column|header|value|spelling)|schema|\bjson\b|\bcsv\b|column names?|header|output (path|file)|naming|template|spelling",
    "business": r"polic|eligib|refund|cancel|\bfee|membership|compensat|threshold|boundary|deadline|exempt|exclusion|conditional|\belse\b|\bSOP\b|authorit|superseding|recipient|customer|booking|reservation|order|invoice|ticket|tracker|\bCRM\b",
    "scope_stop": r"\bstop\b|pivot|budget|time-?box|\bclock\b|timeout|do not (continue|keep|burn|spend|retry)|give up|abandon|deliverable early|final slice|reserve|never end|yak-?shav|when to stop",
    "economize": r"parallel|batch|single call|in one call|at most once|only once|read once|do not re-?read|avoid (re|repeat|redundant|unnecessary)|redundant|minimal|minimi[sz]e|concise|brief|fewer|token|efficien|\bcache|do not repeat|don't repeat|fast feedback|background",
    "serialize": r"one at a time|sequential|serializ|never (in )?parallel|one by one",
}
CATEGORY_REGEX = {name: re.compile(pattern, re.I) for name, pattern in CATEGORY_PATTERNS.items()}


# ---------------------------------------------------------------------------------------------------------------------------------
# Request records of reused seed evaluations
# ---------------------------------------------------------------------------------------------------------------------------------
RENAMED_RUNS = {"/native-20261004/": "/native-20261004-upstream-batch/"}
_resolve = GE.resolve


def resolve_renamed(path, anchor):
    """graph.episode.resolve plus the run directory rename: the seed evaluations reused by the current runs record request files under
    native-20261004, which now lives at native-20261004-upstream-batch."""
    found = _resolve(path, anchor)
    if Path(found).exists():
        return found
    for old, new in RENAMED_RUNS.items():
        if old in str(path) and Path(str(path).replace(old, new)).exists():
            return Path(str(path).replace(old, new))
    return found


# ---------------------------------------------------------------------------------------------------------------------------------
# Run artifacts: delivered skill texts and the evaluation index
# ---------------------------------------------------------------------------------------------------------------------------------
def delivered_texts(run):
    """{vid12: skill text} read from results/skills/deliveries (the guidance.md the agent really received, framed in <bach-skill>)."""
    texts = {}
    for guidance in (run / "results/skills/deliveries/evaluations").glob("*/*/guidance.md"):
        raw = guidance.read_text()
        framed = re.match(r"<bach-skill [^>]*>\n(.*)\n</bach-skill>\s*$", raw, re.S)
        texts[guidance.parent.name.split("-")[0]] = framed.group(1) if framed else raw
    return texts


def eval_index(run):
    """{vid12: {(split, panel12): evaluation.json path}} of every evaluation of the run."""
    index = {}
    for path in (run / "results/evaluations").glob("*/*/evaluation.json"):
        split, name = path.parent.parent.name, path.parent.name
        version, panel = name.split("-")
        index.setdefault(version, {})[(split, panel)] = str(path)
    return index


# ---------------------------------------------------------------------------------------------------------------------------------
# Lineage of each optimizer: nodes {name: text} and edges {parent, child, accepted, meta}
# ---------------------------------------------------------------------------------------------------------------------------------
def skillopt_edges(run):
    """SkillOpt: each step proposes candidate_skill.md against the current accepted skill (seed at first); a rejected step leaves the parent unchanged."""
    directory = run / "skillopt"
    history = json.loads((directory / "history.json").read_text())
    nodes = {"seed": SEED_TEXT}
    edges = []
    current = "seed"
    for step in history:
        number = step["step"]
        name = f"s{number}"
        nodes[name] = (directory / f"steps/step_{number:04d}/candidate_skill.md").read_text()
        accepted = step["action"] != "reject"
        edges.append({"parent": current, "child": name, "accepted": accepted, "meta": {"n_edits": step.get("n_edits_ranked")}})
        if accepted:
            current = name
    return nodes, edges


def gepa_edges(run):
    """GEPA: candidates.json holds every candidate; gepa_state.bin records the parent program of each (accepted) candidate."""
    directory = run / "gepa"
    candidates = json.loads((directory / "candidates.json").read_text())
    with open(directory / "gepa_state.bin", "rb") as handle:
        parents = pickle.load(handle)["parent_program_for_candidate"]
    nodes = {f"c{i}": entry["current_candidate"] for i, entry in enumerate(candidates)}
    edges = [{"parent": f"c{p[0]}", "child": f"c{i}", "accepted": True, "meta": {}} for i, p in enumerate(parents) if p and p[0] is not None]
    return nodes, edges


def last_parseable_patch(iteration_dir):
    """The patch of the last attempt (patch_attempt_NN.json) whose patch is parseable; '' when no attempt is."""
    patch = ""
    for attempt in sorted(iteration_dir.glob("patch_attempt_??.json")):
        text = json.loads(attempt.read_text()).get("patch", "")
        if text != UNPARSEABLE:
            patch = text
    return patch


def plus_lines(patch):
    """Stripped non-empty added lines of a unified diff."""
    return [line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++") and line[1:].strip()]


def success_vector(panel_key, index, version):
    """{task id: success as float} of one version on one panel (the evaluation.json episodes)."""
    episodes = json.loads(Path(index[version][panel_key]).read_text())["episodes"]
    return {e["task_id"]: float(bool(e["success"])) for e in episodes}


def recorded_pair(iteration, added, on_panel, nodes, name_of, index):
    """(baseline vid, candidate vid) of an iteration, found from the per-task success the checkpoint recorded for its baseline and candidate.

    Both versions must have exactly the recorded per-task success on the panel, the candidate must differ from the baseline and must contain every
    line the patch adds that the baseline lacks; None unless exactly one such pair exists."""
    def recorded(side):
        return {t: float(m["hard"]) for t, m in iteration[side]["case_metrics"].items()}

    baselines = [v for v in on_panel if success_vector(on_panel[v], index, v) == recorded("baseline")]
    candidates = [v for v in on_panel if success_vector(on_panel[v], index, v) == recorded("candidate")]
    pairs = []
    for b in baselines:
        base_lines = set(line.strip() for line in nodes[name_of[b]].splitlines())
        new_lines = [line for line in added if line not in base_lines]
        for c in candidates:
            cand_lines = set(line.strip() for line in nodes[name_of[c]].splitlines())
            if c != b and new_lines and all(line in cand_lines for line in new_lines):
                pairs.append((b, c))
    return pairs[0] if len(pairs) == 1 else None


def skilladam_edges(run, texts, index):
    """SkillAdam: iteration k evaluates a candidate against the current accepted skill on one validation panel (its task-id set).

    The candidate of an iteration is the not yet used version evaluated on that panel; when several qualify, the one that contains most of the
    patch's new lines (match score >= 0.5). The baseline is the current accepted version; an accepted iteration moves it to the candidate."""
    directory = run / "skilladam"
    iterations = json.loads((directory / "checkpoint.json").read_text())["iterations"]
    # dev panels: task-id set -> {vid12: (split, panel)}
    panels = {}
    for version, evaluations in index.items():
        for (split, panel), path in evaluations.items():
            if split != "dev":
                continue
            task_ids = frozenset(t["task_id"] for t in json.loads(Path(path).read_text())["panel"])
            panels.setdefault(task_ids, {})[version] = (split, panel)
    nodes = dict(texts)
    nodes["seed"] = SEED_TEXT
    name_of = {vid(text)[:12]: name for name, text in nodes.items()}
    current = vid(SEED_TEXT)[:12]
    used = set()
    edges = []
    for iteration in iterations:
        on_panel = panels.get(frozenset(iteration["validation_case_ids"]), {})
        patch = last_parseable_patch(directory / "iterations" / f"iter_{iteration['iteration']:03d}")
        added = plus_lines(patch)
        parent_lines = set(line.strip() for line in nodes[name_of[current]].splitlines())
        options = [v for v in on_panel if v != current and v not in used]
        best, best_score = None, 0.0
        if len(options) == 1 and (patch or iteration["accepted"]):
            best, best_score = options[0], 1.0
        for option in (options if len(options) > 1 else []):
            option_lines = set(line.strip() for line in nodes[name_of[option]].splitlines())
            new_lines = [line for line in added if line not in parent_lines]
            score = (sum(line in option_lines for line in new_lines) / len(new_lines)) if new_lines else 0.0
            if score > best_score:
                best, best_score = option, score
        if best and best_score >= 0.5:
            used.add(best)
            meta = {"iter": iteration["iteration"], "mode": iteration["patch_mode"], "budget": iteration["edit_budget"], "match": best_score, "patch": patch}
            edges.append({"parent": name_of[current], "child": name_of[best], "accepted": bool(iteration["accepted"]), "meta": meta})
            if iteration["accepted"]:
                current = best
        elif iteration["accepted"]:
            edges.append({"parent": "?", "child": "?", "accepted": True, "meta": {"lost": iteration["iteration"]}})
        elif added:
            # a rejected iteration without a free candidate: its versions were already taken by earlier iterations (several versions share the
            # new lines of a patch), so identify the pair by what the checkpoint recorded for the iteration
            pair = recorded_pair(iteration, added, on_panel, nodes, name_of, index)
            if pair:
                meta = {"iter": iteration["iteration"], "mode": iteration["patch_mode"], "budget": iteration["edit_budget"], "match": 1.0, "patch": patch, "recovered": True}
                edges.append({"parent": name_of[pair[0]], "child": name_of[pair[1]], "accepted": False, "meta": meta})
    return nodes, edges


def lineage(optimizer, bench):
    """(nodes, edges, evaluation index) of one optimizer x benchmark run."""
    run = DS / optimizer / bench / TAG
    index = eval_index(run)
    if optimizer == "skillopt":
        nodes, edges = skillopt_edges(run)
    elif optimizer == "gepa":
        nodes, edges = gepa_edges(run)
    else:
        nodes, edges = skilladam_edges(run, delivered_texts(run), index)
    return nodes, edges, index


# ---------------------------------------------------------------------------------------------------------------------------------
# Per-task decomposition of one parent/child episode pair
# ---------------------------------------------------------------------------------------------------------------------------------
def task_terms(parent, child):
    """Per-task decomposition for one parent/child episode pair; None when the lever model does not apply."""
    mp, mc = GE.moments(parent), GE.moments(child)
    if not mp or not mc:
        return None
    # moments = (R, N, X, F, h, s_F)
    share_fixed = (mp[5] + mc[5]) / 2
    turns = (mp[0] + mc[0]) / 2
    if turns <= 1:
        return None
    elasticity = share_fixed + (1 - share_fixed) * (2 * turns - 1) / (turns - 1)

    def dlog(i):
        return math.log(mc[i] / mp[i])

    d_turns, d_calls, d_cost, d_content, d_prefix = dlog(0), dlog(1), dlog(2), dlog(4), dlog(3)
    skill_p = parent.Xrg[:, parent.comps.index("skill")].mean() if "skill" in parent.comps else 0.0
    skill_c = child.Xrg[:, child.comps.index("skill")].mean() if "skill" in child.comps else 0.0
    return dict(dlogX=d_cost, dlogR=d_turns, dlogN=d_calls, actions=elasticity * d_calls, turnsplit=elasticity * (d_turns - d_calls),
                content=(1 - share_fixed) * d_content, text=share_fixed * d_prefix,
                resid=d_cost - elasticity * d_turns - (1 - share_fixed) * d_content - share_fixed * d_prefix,
                eR=elasticity, sF=share_fixed, Rp=mp[0], Rc=mc[0], Np=mp[1], Nc=mc[1], Xp=mp[2], Xc=mc[2],
                skill_tok_p=float(skill_p), skill_tok_c=float(skill_c), Fp=mp[3], Fc=mc[3])


def exposure_by_component(episode):
    """{component: summed exposure} over the calls of an episode (call = (_, component, _, exposure, ...))."""
    out = {}
    for call in episode.calls:
        out[call[1]] = out.get(call[1], 0.0) + call[3]
    return out


def pair_rows(parent_file, child_file):
    """Per-task rows for the tasks both evaluations ran: successes, terms (or None) and the per-component calls and exposure."""
    parents = {e.task: e for e in GE.load(parent_file)}
    children = {e.task: e for e in GE.load(child_file)}
    rows = []
    for task, p in parents.items():
        if task not in children:
            continue
        c = children[task]
        row = {"task": task, "yp": p.success, "yc": c.success}
        row["terms"] = task_terms(p, c) if min(p.X, c.X) > 0 else None
        row["Ng_p"], row["Ng_c"] = p.N_g(), c.N_g()
        row["Eg_p"], row["Eg_c"] = exposure_by_component(p), exposure_by_component(c)
        row["Xp_all"] = p.X
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
# Text diff in rule categories
# ---------------------------------------------------------------------------------------------------------------------------------
def line_diff(parent_text, child_text):
    """(added lines, removed lines) between parent and child text, non-blank and stripped."""
    a, b = parent_text.splitlines(), child_text.splitlines()
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += [line.strip() for line in b[j1:j2] if line.strip()]
        if op in ("delete", "replace"):
            removed += [line.strip() for line in a[i1:i2] if line.strip()]
    return added, removed


def category_mass(lines):
    """{category: characters of lines matching it} (a line counts for every category it matches; 'other' when none)."""
    mass = {name: 0 for name in CATEGORY_PATTERNS}
    mass["other"] = 0
    for line in lines:
        hits = [name for name, regex in CATEGORY_REGEX.items() if regex.search(line)]
        for name in hits:
            mass[name] += len(line)
        if not hits:
            mass["other"] += len(line)
    return mass


# ---------------------------------------------------------------------------------------------------------------------------------
# Edge x panel records
# ---------------------------------------------------------------------------------------------------------------------------------
def panel_record(edge_info, panel_rows, split, panel):
    """The record of one edge on one common panel (None when fewer than MIN_TERM_TASKS tasks have a valid lever model)."""
    valid = [r for r in panel_rows if r["terms"]]
    if len(valid) < MIN_TERM_TASKS:
        return None
    rec = dict(edge_info["head"], split=split, panel=panel, n_pair=len(panel_rows), n=len(valid))
    rec.update(edge_info["text"])
    for term in EDGE_TERMS:
        rec[term] = float(np.mean([r["terms"][term] for r in valid]))
    for term in EDGE_TERMS:
        rec["sd_" + term] = float(np.std([r["terms"][term] for r in valid], ddof=1))
    rec["dlogN"] = float(np.mean([r["terms"]["dlogN"] for r in valid]))
    rec["dlogR"] = float(np.mean([r["terms"]["dlogR"] for r in valid]))
    # success change on all common tasks with a known outcome on both sides
    known = [r for r in panel_rows if r["yp"] is not None and r["yc"] is not None]
    rec["dQ"] = float(np.mean([int(r["yc"]) - int(r["yp"]) for r in known])) * 100 if known else None
    rec["n_q"] = len(known)
    rec["Qp"] = float(np.mean([int(r["yp"]) for r in known])) * 100 if known else None
    # text price prediction: chars added x skill tokens per char x mean R / mean X on the parent's episodes
    skill_tokens_p = np.mean([r["terms"]["skill_tok_p"] for r in valid])
    per_char = skill_tokens_p / max(edge_info["head"]["len_p"], 1)
    rec["pi_pred"] = edge_info["head"]["chars_added"] * per_char * np.mean([r["terms"]["Rp"] for r in valid]) / np.mean([r["terms"]["Xp"] for r in valid])
    rec["skill_tok_dp"] = float(np.mean([r["terms"]["skill_tok_c"] - r["terms"]["skill_tok_p"] for r in valid]))
    rec["dlogF"] = float(np.mean([math.log(r["terms"]["Fc"] / r["terms"]["Fp"]) for r in valid]))
    # per-component call-count and exposure change per episode
    components = set()
    for r in valid:
        components.update(r["Ng_p"])
        components.update(r["Ng_c"])
    rec["dN_g"] = {c: float(np.mean([r["Ng_c"].get(c, 0) - r["Ng_p"].get(c, 0) for r in valid])) for c in components}
    rec["N_g_p"] = {c: float(np.mean([r["Ng_p"].get(c, 0) for r in valid])) for c in components}
    rec["dE_g"] = {c: float(np.mean([r["Eg_c"].get(c, 0) - r["Eg_p"].get(c, 0) for r in valid])) / 1000 for c in components}
    rec["Xp_k"] = float(np.mean([r["Xp_all"] for r in valid])) / 1000
    rec["dN_abs"] = float(np.mean([r["terms"]["Nc"] - r["terms"]["Np"] for r in valid]))
    rec["dR_abs"] = float(np.mean([r["terms"]["Rc"] - r["terms"]["Rp"] for r in valid]))
    return rec


def frozen_rows():
    """{(opt, bench, parent, child, split, panel): per-task rows} of the scratch builder; {} when the file is absent."""
    if not FROZEN_ROWS.is_file():
        return {}
    out = {}
    for edge in json.loads(FROZEN_ROWS.read_text()):
        for panel in edge["panels"]:
            out[(edge["opt"], edge["bench"], edge["parent"], edge["child"], panel["split"], panel["panel"])] = panel["rows"]
    return out


def build():
    """All edge x panel records of the three optimizers on the three benchmarks."""
    frozen = frozen_rows()
    fallbacks = []
    records = []
    for optimizer in OPTIMIZERS:
        for bench in BENCHES:
            nodes, edges, index = lineage(optimizer, bench)
            for edge in edges:
                if edge["parent"] == "?":
                    print("lost edge", optimizer, bench, edge)
                    continue
                text_p, text_c = nodes[edge["parent"]], nodes[edge["child"]]
                vp, vc = vid(text_p)[:12], vid(text_c)[:12]
                if vp == vc:
                    continue
                added, removed = line_diff(text_p, text_c)
                info = {
                    "head": {"opt": optimizer, "bench": bench, "parent": edge["parent"], "child": edge["child"], "accepted": edge["accepted"],
                             "chars_added": len(text_c) - len(text_p), "len_p": len(text_p), "len_c": len(text_c)},
                    "text": {"add_chars": sum(map(len, added)), "rem_chars": sum(map(len, removed)),
                             "mass_add": category_mass(added), "mass_rem": category_mass(removed)},
                }
                # the panels both versions were evaluated on (same split and same panel)
                for split, panel in sorted(set(index.get(vp, {})) & set(index.get(vc, {}))):
                    rows = pair_rows(index[vp][(split, panel)], index[vc][(split, panel)])
                    # request records gone: fewer episodes rebuilt than the scratch builder paired -> its rows
                    key = (optimizer, bench, edge["parent"], edge["child"], split, panel)
                    if key in frozen and len(frozen[key]) > len(rows):
                        rows = frozen[key]
                        fallbacks.append(key)
                    record = panel_record(info, rows, split, panel)
                    if record:
                        records.append(record)
                print(optimizer, bench, edge["parent"], "->", edge["child"], flush=True)
    print("panels taken from the frozen scratch rows:", len(fallbacks))
    for key in fallbacks:
        print("  frozen", key)
    return records


def run_build():
    """Build the table and write assets/cache/analysis/rq/edges_table.json."""
    # the rename-aware resolver is only needed to rebuild the table; patch it here, not at import, so importing this module has no side effect
    GE.resolve = resolve_renamed
    records = build()
    dataloader.CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, default=float))
    dev = [r for r in records if r["split"] == "dev"]
    print("records", len(records), "dev edges", len(dev), {o: sum(r["opt"] == o for r in dev) for o in OPTIMIZERS})
    return records


# ---------------------------------------------------------------------------------------------------------------------------------
# Holm-adjusted rule-category contrasts (merged from edges_holm.py; the `holm` subcommand)
#
# Holm-adjusted bootstrap p-values for the rule-category contrasts of the 59 dev edges (paper 1, appendix F). The contrasts are those of
# skilltext.emp_check: over the dev parent->child edges of edges_table.json without the excluded SkillAdam x terminal-bench-2-1 cell
# (is_excluded), an edge "contains" a rule category when its added text of that category has at least 150 characters; for each of the ten
# categories with more than one edge on each side, the contrast is the unweighted mean over edges with the category minus the mean over
# edges without it, on the actions and the turn-splitting terms (20 tests). The bootstrap resamples edges within each side, as the record
# does (here 20,000 resamples, seed 20261012, for p-value resolution); the two-sided p-value is 2 x the smaller tail mass of the bootstrap
# differences at zero, with the +1 correction. Holm's step-down adjustment runs over all 20 tests.
# ---------------------------------------------------------------------------------------------------------------------------------
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
        f"The {len(edges)} dev parent->child edges of assets/cache/analysis/rq/edges_table.json (Table 1 baseline runs) without SkillAdam x "
        f"terminal-bench-2-1 ({EXCLUDED_REASON}); categories and the {MIN_CHARS}-character rule as in analysis/rq3/skilltext.py emp_check.",
        f"Unweighted means over edges. Bootstrap: {HOLM_RESAMPLES:,} resamples, each side resampled within itself, seed {HOLM_SEED}; two-sided p = 2 x the "
        f"smaller tail mass at zero with the +1 correction; Holm step-down over the {len(tests)} tests, alpha {ALPHA}. Differences in log points x 100.",
        logger.md_table(["category", "term", "edges with/without", "difference", "95% bootstrap interval", "p", "p (Holm, 20 tests)",
                    "unadjusted interval excludes 0", f"survives Holm at {ALPHA}"], rows),
        ".venv/bin/python -m src.analysis.rq3.edges holm")


# ---------------------------------------------------------------------------------------------------------------------------------
# mech_edges figure (merged from edges_fig.py; the `draw` subcommand)
#
# What an edit says and what it changes: for each of the ten keyword groups of the 59 canonical dev edits, the difference (edits whose added
# text contains the group minus edits without it, unweighted mean over edits) in the actions term and in the turn-splitting term of the
# change in log input tokens (the 95% bootstrap intervals over edits are kept in the record, not drawn). Groups sorted by the actions
# difference; labels carry the with/without edit counts. Reads skilltext.json (emp_baseline_check.category_presence) and edges_holm.json.
# ---------------------------------------------------------------------------------------------------------------------------------
GROUPS = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
# plain names of the keyword groups (patterns in CATEGORY_PATTERNS above); "one at a time" as in the rules figure
NAME = {"verify": "verify", "reread": "re-read state", "enumerate": "check every item", "test": "run tests", "format": "output format",
        "business": "domain rules", "scope_stop": "stop or budget", "economize": "save calls or reads", "serialize": "one at a time", "other": "other"}
FIG_TERMS = [("actions", "more actions", "#0072B2", "o"), ("turnsplit", "turn-splitting", "#E69F00", "s")]


def draw_edges():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pres = json.load(open(dataloader.CACHE / "skilltext.json"))["emp_baseline_check"]["category_presence"]
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
                   body, ".venv/bin/python -m src.analysis.rq3.edges draw")
    return table


if __name__ == "__main__":
    main()
