"""Parent->child skill edges of the optimizer runs: the history reader, the edge table and the edges that lengthen the skill.

  python -m aux.rule_coding.edges build  -> CACHE/edges_table.json (the lengthening count is in skilltext luna)
"""
import argparse
import difflib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

from src import layout
from src.graph import episode as GE
from src.utils import dataloader
from src.utils.dataloader import BENCHES, EX, SEED_TEXT, vid

TABLE = dataloader.CACHE / "edges_table.json"
# per-task rows of the seed panels whose request records were deleted before the table was built
FROZEN_ROWS = layout.ROOT / "aux/data/edges_frozen_rows.json"
RUNS = {"deepseek-flash": "native-20261008-https", "gpt-6-luna": "native-20261012"}
OPTS = ["gepa", "skillopt", "skilladam"]
OPTIMIZERS = ("skillopt", "gepa", "skilladam")
NAMES = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
EXCLUDED_CELL = ("skilladam", "terminal-bench-2-1")
EXCLUDED_REASON = "two versions of that run match the same patch, so the parent of its edits is ambiguous in the logs"
EDGE_TERMS = ["dlogX", "actions", "turnsplit", "content", "text", "resid"]
MIN_TERM_TASKS = 3
UNPARSEABLE = "(unparseable)"
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


def is_excluded(row):
    return (row["opt"], row["bench"]) == EXCLUDED_CELL


def load_dev_edges():
    """Dev edges of the edge table without the excluded cell."""
    return [r for r in json.loads(TABLE.read_text()) if r["split"] == "dev" and not is_excluded(r)]


def delivered_texts(run):
    """{version: skill text the agent received} from the run's delivered guidance files."""
    texts = {}
    for guidance in (run / "results/skills/deliveries/evaluations").glob("*/*/guidance.md"):
        raw = guidance.read_text()
        framed = re.match(r"<bach-skill [^>]*>\n(.*)\n</bach-skill>\s*$", raw, re.S)
        texts[guidance.parent.name.split("-")[0]] = framed.group(1) if framed else raw
    return texts


def eval_index(run):
    """{version: {(split, panel): evaluation.json}} of a run."""
    index = {}
    for path in (run / "results/evaluations").glob("*/*/evaluation.json"):
        version, panel = path.parent.name.split("-")
        index.setdefault(version, {})[(path.parent.parent.name, panel)] = str(path)
    return index


def skillopt_edges(run):
    """Each step proposes a candidate against the current accepted skill; a rejected step keeps the parent."""
    nodes, edges, current = {"seed": SEED_TEXT}, [], "seed"
    for step in json.loads((run / "skillopt/history.json").read_text()):
        name = f"s{step['step']}"
        nodes[name] = (run / f"skillopt/steps/step_{step['step']:04d}/candidate_skill.md").read_text()
        accepted = step["action"] != "reject"
        edges.append({"parent": current, "child": name, "accepted": accepted, "meta": {"n_edits": step.get("n_edits_ranked")}})
        current = name if accepted else current
    return nodes, edges


def gepa_edges(run):
    """Every GEPA candidate with the parent program its state records."""
    candidates = json.loads((run / "gepa/candidates.json").read_text())
    with open(run / "gepa/gepa_state.bin", "rb") as handle:
        parents = pickle.load(handle)["parent_program_for_candidate"]
    nodes = {f"c{i}": c["current_candidate"] for i, c in enumerate(candidates)}
    return nodes, [{"parent": f"c{p[0]}", "child": f"c{i}", "accepted": True, "meta": {}}
                   for i, p in enumerate(parents) if p and p[0] is not None]


def last_parseable_patch(iteration_dir):
    patch = ""
    for attempt in sorted(iteration_dir.glob("patch_attempt_??.json")):
        text = json.loads(attempt.read_text()).get("patch", "")
        patch = text if text != UNPARSEABLE else patch
    return patch


def plus_lines(patch):
    return [line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++") and line[1:].strip()]


def _lines(text):
    return {line.strip() for line in text.splitlines()}


def success_vector(panel_key, index, version):
    episodes = json.loads(Path(index[version][panel_key]).read_text())["episodes"]
    return {e["task_id"]: float(bool(e["success"])) for e in episodes}


def recorded_pair(iteration, added, on_panel, nodes, name_of, index):
    """The unique (baseline, candidate) whose per-task success equals what the checkpoint recorded and whose texts fit the patch."""
    def matching(side):
        recorded = {t: float(m["hard"]) for t, m in iteration[side]["case_metrics"].items()}
        return [v for v in on_panel if success_vector(on_panel[v], index, v) == recorded]

    candidates, pairs = matching("candidate"), []
    for b in matching("baseline"):
        new = [line for line in added if line not in _lines(nodes[name_of[b]])]
        pairs += [(b, c) for c in candidates if c != b and new and all(line in _lines(nodes[name_of[c]]) for line in new)]
    return pairs[0] if len(pairs) == 1 else None


def skilladam_edges(run, texts, index):
    """Each iteration scores a candidate against the current accepted skill on one dev panel.

    The candidate is the unused version on that panel holding most of the patch's new lines (at least half)."""
    panels = {}
    for version, evaluations in index.items():
        for (split, panel), path in evaluations.items():
            if split == "dev":
                tasks = frozenset(t["task_id"] for t in json.loads(Path(path).read_text())["panel"])
                panels.setdefault(tasks, {})[version] = (split, panel)
    nodes = dict(texts, seed=SEED_TEXT)
    name_of = {vid(text)[:12]: name for name, text in nodes.items()}
    current, used, edges = vid(SEED_TEXT)[:12], set(), []
    for it in json.loads((run / "skilladam/checkpoint.json").read_text())["iterations"]:
        on_panel = panels.get(frozenset(it["validation_case_ids"]), {})
        patch = last_parseable_patch(run / f"skilladam/iterations/iter_{it['iteration']:03d}")
        added = plus_lines(patch)
        new = [line for line in added if line not in _lines(nodes[name_of[current]])]
        options = [v for v in on_panel if v != current and v not in used]
        best, score = (options[0], 1.0) if len(options) == 1 and (patch or it["accepted"]) else (None, 0.0)
        for option in options if len(options) > 1 else []:
            s = sum(line in _lines(nodes[name_of[option]]) for line in new) / len(new) if new else 0.0
            best, score = (option, s) if s > score else (best, score)
        meta = {"iter": it["iteration"], "mode": it["patch_mode"], "budget": it["edit_budget"], "match": score, "patch": patch}
        if best and score >= 0.5:
            used.add(best)
            edges.append({"parent": name_of[current], "child": name_of[best], "accepted": bool(it["accepted"]), "meta": meta})
            current = best if it["accepted"] else current
        elif it["accepted"]:
            edges.append({"parent": "?", "child": "?", "accepted": True, "meta": {"lost": it["iteration"]}})
        elif added and (pair := recorded_pair(it, added, on_panel, nodes, name_of, index)):
            meta.update(match=1.0, recovered=True)
            edges.append({"parent": name_of[pair[0]], "child": name_of[pair[1]], "accepted": False, "meta": meta})
    return nodes, edges


def lineage(optimizer, bench, model="deepseek-flash"):
    """(nodes, edges, evaluation index) of one optimizer run."""
    run = EX / model / optimizer / bench / RUNS[model]
    index = eval_index(run)
    if optimizer == "skilladam":
        return (*skilladam_edges(run, delivered_texts(run), index), index)
    return (*(skillopt_edges if optimizer == "skillopt" else gepa_edges)(run), index)


def scored_edges(model="deepseek-flash", optimizers=OPTIMIZERS):
    """(optimizer, bench, edge, parent text, child text, index) of every edge that changes the text."""
    for optimizer in optimizers:
        for bench in BENCHES:
            nodes, edges, index = lineage(optimizer, bench, model)
            for edge in edges:
                if edge["parent"] == "?":
                    print("lost edge", optimizer, bench, edge)
                elif vid(nodes[edge["parent"]])[:12] != vid(nodes[edge["child"]])[:12]:
                    yield optimizer, bench, edge, nodes[edge["parent"]], nodes[edge["child"]], index


def task_terms(parent, child):
    """Per-task split of the change in log input tokens; None when the lever model does not apply."""
    mp, mc = GE.moments(parent), GE.moments(child)
    if not mp or not mc or (mp[0] + mc[0]) / 2 <= 1:
        return None
    share, turns = (mp[5] + mc[5]) / 2, (mp[0] + mc[0]) / 2
    elasticity = share + (1 - share) * (2 * turns - 1) / (turns - 1)
    d_turns, d_calls, d_cost, d_prefix, d_content = (math.log(mc[i] / mp[i]) for i in range(5))
    skill_p = parent.Xrg[:, parent.comps.index("skill")].mean() if "skill" in parent.comps else 0.0
    skill_c = child.Xrg[:, child.comps.index("skill")].mean() if "skill" in child.comps else 0.0
    return dict(dlogX=d_cost, dlogR=d_turns, dlogN=d_calls, actions=elasticity * d_calls, turnsplit=elasticity * (d_turns - d_calls),
                content=(1 - share) * d_content, text=share * d_prefix,
                resid=d_cost - elasticity * d_turns - (1 - share) * d_content - share * d_prefix,
                eR=elasticity, sF=share, Rp=mp[0], Rc=mc[0], Np=mp[1], Nc=mc[1], Xp=mp[2], Xc=mc[2],
                skill_tok_p=float(skill_p), skill_tok_c=float(skill_c), Fp=mp[3], Fc=mc[3])


def exposure_by_component(episode):
    out = {}
    for call in episode.calls:
        out[call[1]] = out.get(call[1], 0.0) + call[3]
    return out


def pair_rows(parent_file, child_file):
    """Per-task rows of the tasks both evaluations ran."""
    children = {e.task: e for e in GE.load(child_file)}
    rows = []
    for p in GE.load(parent_file):
        if (c := children.get(p.task)) is not None:
            rows.append({"task": p.task, "yp": p.success, "yc": c.success, "terms": task_terms(p, c) if min(p.X, c.X) > 0 else None,
                         "Ng_p": p.N_g(), "Ng_c": c.N_g(), "Eg_p": exposure_by_component(p), "Eg_c": exposure_by_component(c), "Xp_all": p.X})
    return rows


def line_diff(parent_text, child_text):
    """(added, removed) non-blank stripped lines."""
    a, b = parent_text.splitlines(), child_text.splitlines()
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        added += [x.strip() for x in b[j1:j2] if x.strip()] if op in ("insert", "replace") else []
        removed += [x.strip() for x in a[i1:i2] if x.strip()] if op in ("delete", "replace") else []
    return added, removed


def category_mass(lines):
    """{category: characters of the lines matching it}; a line counts for every category it matches, else for 'other'."""
    mass = dict.fromkeys([*CATEGORY_PATTERNS, "other"], 0)
    for line in lines:
        hits = [name for name, regex in CATEGORY_REGEX.items() if regex.search(line)] or ["other"]
        for name in hits:
            mass[name] += len(line)
    return mass


def panel_record(head, text, rows, split, panel):
    """One edge on one shared panel; None with fewer than MIN_TERM_TASKS valid tasks."""
    valid = [r for r in rows if r["terms"]]
    if len(valid) < MIN_TERM_TASKS:
        return None

    def mean(key, f=lambda r, k: r["terms"][k]):
        return float(np.mean([f(r, key) for r in valid]))

    rec = dict(head, split=split, panel=panel, n_pair=len(rows), n=len(valid), **text)
    rec.update({term: mean(term) for term in EDGE_TERMS})
    rec.update({"sd_" + term: float(np.std([r["terms"][term] for r in valid], ddof=1)) for term in EDGE_TERMS})
    rec["dlogN"], rec["dlogR"] = mean("dlogN"), mean("dlogR")
    known = [r for r in rows if r["yp"] is not None and r["yc"] is not None]
    rec["dQ"] = float(np.mean([int(r["yc"]) - int(r["yp"]) for r in known])) * 100 if known else None
    rec["n_q"] = len(known)
    rec["Qp"] = float(np.mean([int(r["yp"]) for r in known])) * 100 if known else None
    per_char = np.mean([r["terms"]["skill_tok_p"] for r in valid]) / max(head["len_p"], 1)
    rec["pi_pred"] = head["chars_added"] * per_char * np.mean([r["terms"]["Rp"] for r in valid]) / np.mean([r["terms"]["Xp"] for r in valid])
    rec["skill_tok_dp"] = mean(None, lambda r, _: r["terms"]["skill_tok_c"] - r["terms"]["skill_tok_p"])
    rec["dlogF"] = mean(None, lambda r, _: math.log(r["terms"]["Fc"] / r["terms"]["Fp"]))
    comps = set().union(*(set(r["Ng_p"]) | set(r["Ng_c"]) for r in valid))
    rec["dN_g"] = {c: mean(c, lambda r, k: r["Ng_c"].get(k, 0) - r["Ng_p"].get(k, 0)) for c in comps}
    rec["N_g_p"] = {c: mean(c, lambda r, k: r["Ng_p"].get(k, 0)) for c in comps}
    rec["dE_g"] = {c: mean(c, lambda r, k: r["Eg_c"].get(k, 0) - r["Eg_p"].get(k, 0)) / 1000 for c in comps}
    rec["Xp_k"] = mean(None, lambda r, _: r["Xp_all"]) / 1000
    rec["dN_abs"] = mean(None, lambda r, _: r["terms"]["Nc"] - r["terms"]["Np"])
    rec["dR_abs"] = mean(None, lambda r, _: r["terms"]["Rc"] - r["terms"]["Rp"])
    return rec


def frozen_rows():
    """{(opt, bench, parent, child, split, panel): per-task rows} of the frozen file; raises when it is missing."""
    return {(e["opt"], e["bench"], e["parent"], e["child"], p["split"], p["panel"]): p["rows"]
            for e in json.loads(FROZEN_ROWS.read_text()) for p in e["panels"]}


def build():
    """Edge x shared-panel records of the DeepSeek optimizer runs."""
    frozen, used, records = frozen_rows(), [], []
    for optimizer, bench, edge, text_p, text_c, index in scored_edges():
        vp, vc = vid(text_p)[:12], vid(text_c)[:12]
        added, removed = line_diff(text_p, text_c)
        head = {"opt": optimizer, "bench": bench, "parent": edge["parent"], "child": edge["child"], "accepted": edge["accepted"],
                "chars_added": len(text_c) - len(text_p), "len_p": len(text_p), "len_c": len(text_c)}
        text = {"add_chars": sum(map(len, added)), "rem_chars": sum(map(len, removed)),
                "mass_add": category_mass(added), "mass_rem": category_mass(removed)}
        for split, panel in sorted(set(index.get(vp, {})) & set(index.get(vc, {}))):
            rows = pair_rows(index[vp][(split, panel)], index[vc][(split, panel)])
            key = (optimizer, bench, edge["parent"], edge["child"], split, panel)
            if len(frozen.get(key, ())) > len(rows):
                rows = frozen[key]
                used.append(key)
            if record := panel_record(head, text, rows, split, panel):
                records.append(record)
    print("panels taken from the frozen rows:", len(used))
    return records


def lengths(model):
    """Edges of a model's runs that share a dev panel, with the skill length before and after."""
    return [{"opt": opt, "bench": bench, "parent": e["parent"], "child": e["child"], "accepted": e["accepted"], "dev_panels": dev,
             "chars_p": len(tp), "chars_c": len(tc), "words_p": len(tp.split()), "words_c": len(tc.split())}
            for opt, bench, e, tp, tc, index in scored_edges(model, OPTS)
            if (dev := sum(k[0] == "dev" for k in set(index.get(vid(tp)[:12], {})) & set(index.get(vid(tc)[:12], {}))))]


def lengthen(model):
    """How many dev-scored edges lengthen the skill, outside and inside the excluded cell."""
    rows = lengths(model)
    kept, excluded = [r for r in rows if not is_excluded(r)], [r for r in rows if is_excluded(r)]
    longer = lambda group: sum(r["chars_c"] > r["chars_p"] for r in group)  # noqa: E731
    return {"scored": len(kept), "longer": longer(kept), "sa_tb2_scored": len(excluded), "sa_tb2_longer": longer(excluded),
            "shorter": [r for r in kept if r["chars_c"] <= r["chars_p"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["build"])
    parser.parse_args()
    records = build()
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(records, default=float))
    dev = [r for r in records if r["split"] == "dev"]
    print("records", len(records), "dev edges", len(dev), {o: sum(r["opt"] == o for r in dev) for o in OPTIMIZERS})
