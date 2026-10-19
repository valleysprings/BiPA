"""Rebuild the parent->child edge table of the baseline optimizer runs from the run artifacts (paper 1, Sec. 4.4 / 4.5).

Analysis role: every version an optimizer proposes is a child of an earlier version (its parent). For each such edge we pair the evaluation
episodes of parent and child on the tasks both ran (same split and same panel), decompose the per-task change of input tokens
    dlogX = actions + turnsplit + content + text + resid
and attach what the text diff added, in rule categories. mech/edges.py (accepted vs rejected) and mech/skilltext.py (Sec. 4.5 category
contrasts) read the table written here.

This is a port of .scratch/evidence-20261011/emp_baseline_1_edges.py, emp_baseline_2_pairs.py and emp_baseline_3_analyze.py (the edge-table part)
with one rule changed: for SkillAdam the patch of an iteration is the LAST attempt (patch_attempt_NN.json) that holds a parseable patch; the
scratch builder read only attempt 01, so an iteration whose attempt 01 is "(unparseable)" lost its edge. The patch only matters for matching a
SkillAdam iteration to its candidate version (the candidate whose new lines occur in the patch); every number of an edge comes from the
evaluation episodes.

Edge terms (per task, then mean over tasks with a valid lever model; T_p, T_c = parent / child episode):
  e_R = s_F + (1 - s_F) (2R - 1) / (R - 1), s_F and R the means of parent and child
  actions = e_R dlogN, turnsplit = e_R (dlogR - dlogN), content = (1 - s_F) dlogh, text = s_F dlogF, resid = dlogX - e_R dlogR - content - text.

Run (from bipa): .venv/bin/python -m src.analysis.mech.edges_build  -> assets/cache/analysis/mech/edges_table.json
"""
import difflib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

from src.utils.dataloader import BENCHES, DS, SEED_TEXT, vid
from src.utils import dataloader
from src.graph import episode as GE

TAG = "native-20261008-https"
# Per-task rows of the scratch builder (emp_baseline_2_pairs.py). The seed evaluations reused by the current runs recorded their request files
# under the run native-20261004, which the 2026-10-12 disk cleanup deleted (.trash/cleanup-20261012.md); for exactly those panels the rows of this
# frozen file stand in (read-only). Every other panel, and every edge the file does not hold, is rebuilt from the run artifacts.
FROZEN_ROWS = dataloader.H / ".scratch/evidence-20261011/emp_baseline_pairs.json"
OPTIMIZERS = ("skillopt", "gepa", "skilladam")
OUT = dataloader.CACHE / "edges_table.json"
UNPARSEABLE = "(unparseable)"
MIN_TERM_TASKS = 3  # an edge-panel needs at least this many tasks with a valid lever model
TERMS = ["dlogX", "actions", "turnsplit", "content", "text", "resid"]

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


GE.resolve = resolve_renamed


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
    for term in TERMS:
        rec[term] = float(np.mean([r["terms"][term] for r in valid]))
    for term in TERMS:
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


def run():
    """Build the table and write assets/cache/analysis/mech/edges_table.json."""
    records = build()
    dataloader.CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, default=float))
    dev = [r for r in records if r["split"] == "dev"]
    print("records", len(records), "dev edges", len(dev), {o: sum(r["opt"] == o for r in dev) for o in OPTIMIZERS})
    return records


if __name__ == "__main__":
    run()
