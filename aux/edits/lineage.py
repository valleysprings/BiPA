"""The edits of every dev-gated optimizer run: which parent each candidate skill came from, and the table of what each
DeepSeek edit changed on the panels its candidate and parent share.

An edit is a candidate against its parent. SkillOpt proposes each step's candidate against the skill the step started
from, skills/skill_v<step - 1>.md (history.json's current_origin names it: the seed, the last accepted step, or the slow
update that closed an epoch). GEPA records each candidate's parent program (summary.json, extra.parents). SkillAdam scores
a patch against its current skill on one dev panel; the candidate is the unused version on that panel that holds at least
half of the patch's new lines, and a rejected patch no version holds is matched by the unique pair whose per-task success
equals what the checkpoint recorded. A version is the skill text delivered to the agent.

The edge table holds one record per DeepSeek edit and shared panel with at least MIN_TERM_TASKS tasks: the mean per-task
change in log input tokens split into more actions, turn-splitting, content per turn and skill text (with the residual
that closes the identity), the change in success, and the characters the edit adds and removes with the keyword tags of
the added text. Panels whose request records were deleted take their per-task rows from aux/edits/data/edges_frozen_rows.json.
SkillAdam on Terminal-Bench 2.1 is left out of every count of edits: two versions of that run match the same patch, so
the parent of its edits is ambiguous in the logs.

  python -m aux.edits.lineage   -> edges_table.json in the analysis cache (folder edits)"""
import difflib, json, math, re
from pathlib import Path

import numpy as np

from src.analysis import settings as S
from src.graph import episode as GE
from src.utils.layout import ROOT
from src.utils.skill_frame import SKILL_FRAME
from src.utils.dataloader import BENCHES, SEED_TEXT

TABLE = "edges_table.json"
FROZEN_ROWS = ROOT / "aux/edits/data/edges_frozen_rows.json"
EXCLUDED = ("skilladam", "terminal-bench-2-1")
METHODS = ("skillopt", "gepa", "skilladam")
MIN_TERM_TASKS = 3
TERMS = ("dlogX", "actions", "turnsplit", "content", "text", "resid", "dlogR", "dlogN")
UNPARSEABLE = "(unparseable)"
# keyword tags of the text an edit adds (overlapping; a line counts for every tag it matches, else for "other")
TAGS = {name: re.compile(pattern, re.I) for name, pattern in {
    "verify": r"verif|confirm|double-check|validat|sanity|evidence|prove|check (that|whether|the|each|every)|audit|acceptance|oracle|grader|assert|cross-check",
    "reread": r"re-?read|read (it |them |back )?again|read back|re-?fetch|re-?query|re-?check|re-?run|re-?inspect|re-?open|fresh (read|get|state|fetch|query)|after (any |every )?(write|mutat)|persisted state",
    "enumerate": r"\bevery\b|\beach\b|enumerat|checklist|exhaust|list (all|every)|line[- ]by[- ]line|for each|silently (drop|omit)|\ball (required|items|records|deliverables|entities)",
    "test": r"\btests?\b|pytest|unit test|reproduc|verifier|smoke|tiny (sample|input)|edge case|malformed|stress|test_outputs",
    "format": r"format|exact(ly)? (name|file|path|column|header|value|spelling)|schema|\bjson\b|\bcsv\b|column names?|header|output (path|file)|naming|template|spelling",
    "business": r"polic|eligib|refund|cancel|\bfee|membership|compensat|threshold|boundary|deadline|exempt|exclusion|conditional|\belse\b|\bSOP\b|authorit|superseding|recipient|customer|booking|reservation|order|invoice|ticket|tracker|\bCRM\b",
    "scope_stop": r"\bstop\b|pivot|budget|time-?box|\bclock\b|timeout|do not (continue|keep|burn|spend|retry)|give up|abandon|deliverable early|final slice|reserve|never end|yak-?shav|when to stop",
    "economize": r"parallel|batch|single call|in one call|at most once|only once|read once|do not re-?read|avoid (re|repeat|redundant|unnecessary)|redundant|minimal|minimi[sz]e|concise|brief|fewer|token|efficien|\bcache|do not repeat|don't repeat|fast feedback|background",
    "serialize": r"one at a time|sequential|serializ|never (in )?parallel|one by one",
}.items()}


def version(text):
    return S.version(text)[:12]


def excluded(row):
    return (row["opt"], row["bench"]) == EXCLUDED


# ---- lineage of one run

def eval_index(run):
    """{version: {(split, panel): evaluation.json}} of a run."""
    index = {}
    for path in (run / "results/evaluations").glob("*/*/evaluation.json"):
        v, panel = path.parent.name.split("-")
        index.setdefault(v, {})[(path.parent.parent.name, panel)] = str(path)
    return index


def skillopt_edges(run):
    base = run / "skillopt"
    history = json.loads((base / "history.json").read_text())
    nodes = {"seed": SEED_TEXT, **{f"s{s['step']}": (base / f"steps/step_{s['step']:04d}/candidate_skill.md").read_text() for s in history}}
    for f in sorted(base.glob("skills/skill_v*.md")):  # the slow update that closed an epoch is a version of its own
        if (text := f.read_text()) not in nodes.values():
            nodes[f"u{int(f.stem.split('_v')[1])} (slow update)"] = text
    name = {t: n for n, t in reversed(nodes.items())}
    return nodes, [{"parent": name[(base / f"skills/skill_v{s['step'] - 1:04d}.md").read_text()], "child": f"s{s['step']}",
                    "accepted": s["action"] != "reject"} for s in history]


def gepa_edges(run):
    candidates = json.loads((run / "gepa/candidates.json").read_text())
    parents = json.loads((run / "summary.json").read_text())["extra"]["parents"]
    nodes = {f"c{i}": c["current_candidate"] for i, c in enumerate(candidates)}
    return nodes, [{"parent": f"c{p[0]}", "child": f"c{i}", "accepted": True} for i, p in enumerate(parents) if p and p[0] is not None]


def delivered_texts(run):
    """{version: skill text the agent received} from the run's delivered guidance files."""
    texts = {}
    for guidance in (run / "results/skills/deliveries/evaluations").glob("*/*/guidance.md"):
        raw = guidance.read_text()
        framed = SKILL_FRAME.fullmatch(raw.rstrip())
        body = framed.group("body") if framed else ""
        texts[guidance.parent.name.split("-")[0]] = body[1:-1] \
            if body.startswith("\n") and body.endswith("\n") else raw
    return texts


def plus_lines(patch):
    return [x[1:].strip() for x in patch.splitlines() if x.startswith("+") and not x.startswith("+++") and x[1:].strip()]


def last_parseable_patch(folder):
    patch = ""
    for attempt in sorted(folder.glob("patch_attempt_??.json")):
        text = json.loads(attempt.read_text()).get("patch", "")
        patch = text if text != UNPARSEABLE else patch
    return patch


def skilladam_edges(run, index):
    lines = lambda text: {x.strip() for x in text.splitlines()}  # noqa: E731
    success = lambda v, key: {e["task_id"]: float(bool(e["success"])) for e in json.loads(Path(index[v][key]).read_text())["episodes"]}  # noqa: E731
    panels = {}
    for v, evaluations in index.items():
        for (split, panel), path in evaluations.items():
            if split == "dev":
                panels.setdefault(frozenset(t["task_id"] for t in json.loads(Path(path).read_text())["panel"]), {})[v] = (split, panel)
    nodes = {**delivered_texts(run), "seed": SEED_TEXT}
    name = {version(text): n for n, text in nodes.items()}
    current, used, edges = version(SEED_TEXT), set(), []
    for it in json.loads((run / "skilladam/checkpoint.json").read_text())["iterations"]:
        on_panel = panels.get(frozenset(it["validation_case_ids"]), {})
        patch = last_parseable_patch(run / f"skilladam/iterations/iter_{it['iteration']:03d}")
        added = plus_lines(patch)
        new = [x for x in added if x not in lines(nodes[name[current]])]
        options = [v for v in on_panel if v != current and v not in used]
        best, score = (options[0], 1.0) if len(options) == 1 and (patch or it["accepted"]) else (None, 0.0)
        for option in options if len(options) > 1 else []:
            s = sum(x in lines(nodes[name[option]]) for x in new) / len(new) if new else 0.0
            best, score = (option, s) if s > score else (best, score)
        if best and score >= 0.5:
            used.add(best)
            edges.append({"parent": name[current], "child": name[best], "accepted": bool(it["accepted"])})
            current = best if it["accepted"] else current
        elif it["accepted"]:
            print("lost edge", run, "iteration", it["iteration"])
        elif added:  # the unique (baseline, candidate) whose per-task success the checkpoint recorded and whose texts fit the patch
            match = lambda side: [v for v in on_panel if success(v, on_panel[v]) == {t: float(m["hard"]) for t, m in it[side]["case_metrics"].items()}]  # noqa: E731
            pairs = [(b, c) for b in match("baseline") for c in match("candidate")
                     if c != b and (fresh := [x for x in added if x not in lines(nodes[name[b]])])
                     and all(x in lines(nodes[name[c]]) for x in fresh)]
            if len(pairs) == 1:
                edges.append({"parent": name[pairs[0][0]], "child": name[pairs[0][1]], "accepted": False})
    return nodes, edges


def lineage(model, bench, method):
    """(nodes {name: text}, edges [{parent, child, accepted}], evaluation index) of the run the pins give."""
    run = S.run(model, bench, method)
    index = eval_index(run)
    nodes, edges = skilladam_edges(run, index) if method == "skilladam" else (skillopt_edges if method == "skillopt" else gepa_edges)(run)
    return nodes, edges, index


def scored_edges(model):
    """(method, bench, edge, parent text, child text, index) of every edit of a dev-gated run that changes the skill."""
    for method in METHODS:
        for bench in BENCHES:
            if S.dev_gated(model, bench, method):
                nodes, edges, index = lineage(model, bench, method)
                for e in edges:
                    if version(nodes[e["parent"]]) != version(nodes[e["child"]]):
                        yield method, bench, e, nodes[e["parent"]], nodes[e["child"]], index


def shared_panels(index, parent, child, split=None):
    return sorted(k for k in set(index.get(version(parent), {})) & set(index.get(version(child), {})) if split in (None, k[0]))


def lengths(model):
    """The dev-scored edits of a model's runs (parent and child share a dev panel), with the skill length before and after."""
    return [{"opt": method, "bench": bench, "parent": e["parent"], "child": e["child"], "accepted": e["accepted"],
             "chars_p": len(tp), "chars_c": len(tc)}
            for method, bench, e, tp, tc, index in scored_edges(model) if shared_panels(index, tp, tc, "dev")]


# ---- the edge table

def task_terms(parent, child):
    """Per-task split of the change in log input tokens; None when the turn model does not apply."""
    mp, mc = GE.moments(parent), GE.moments(child)
    if not mp or not mc or (mp[0] + mc[0]) / 2 <= 1:
        return None
    share, turns = (mp[5] + mc[5]) / 2, (mp[0] + mc[0]) / 2
    elasticity = share + (1 - share) * (2 * turns - 1) / (turns - 1)
    d_turns, d_calls, d_tokens, d_prefix, d_content = (math.log(mc[i] / mp[i]) for i in range(5))
    return dict(dlogX=d_tokens, dlogR=d_turns, dlogN=d_calls, actions=elasticity * d_calls, turnsplit=elasticity * (d_turns - d_calls),
                content=(1 - share) * d_content, text=share * d_prefix,
                resid=d_tokens - elasticity * d_turns - (1 - share) * d_content - share * d_prefix)


def pair_rows(parent_file, child_file):
    """Per-task rows of the tasks both evaluations ran."""
    children = {e.task: e for e in GE.load(child_file)}
    return [{"yp": p.success, "yc": c.success, "terms": task_terms(p, c) if min(p.X, c.X) > 0 else None}
            for p in GE.load(parent_file) if (c := children.get(p.task)) is not None]


def line_diff(a, b):
    """(added, removed) non-blank stripped lines of a whole-line diff."""
    a, b, added, removed = a.splitlines(), b.splitlines(), [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        added += [x.strip() for x in b[j1:j2] if x.strip()] if op in ("insert", "replace") else []
        removed += [x.strip() for x in a[i1:i2] if x.strip()] if op in ("delete", "replace") else []
    return added, removed


def tag_mass(lines):
    """{tag: characters of the lines carrying it}."""
    mass = dict.fromkeys([*TAGS, "other"], 0)
    for line in lines:
        for tag in [t for t, rx in TAGS.items() if rx.search(line)] or ["other"]:
            mass[tag] += len(line)
    return mass


def panel_record(head, rows):
    valid = [r for r in rows if r["terms"]]
    if len(valid) < MIN_TERM_TASKS:
        return None
    known = [r for r in rows if r["yp"] is not None and r["yc"] is not None]
    return dict(head, n_pair=len(rows), n=len(valid), **{t: float(np.mean([r["terms"][t] for r in valid])) for t in TERMS},
                dQ=float(np.mean([int(r["yc"]) - int(r["yp"]) for r in known])) * 100 if known else None)


def build():
    """Edit x shared-panel records of DeepSeek's dev-gated runs."""
    frozen = {(e["opt"], e["bench"], e["parent"], e["child"], p["split"], p["panel"]): p["rows"]
              for e in json.loads(FROZEN_ROWS.read_text()) for p in e["panels"]}
    records, used = [], 0
    for method, bench, edge, tp, tc, index in scored_edges(S.DEEPSEEK):
        added, removed = line_diff(tp, tc)
        head = {"opt": method, "bench": bench, "parent": edge["parent"], "child": edge["child"], "accepted": edge["accepted"],
                "len_p": len(tp), "len_c": len(tc), "add_chars": sum(map(len, added)), "rem_chars": sum(map(len, removed)),
                "mass_add": tag_mass(added)}
        for split, panel in shared_panels(index, tp, tc):
            rows = pair_rows(index[version(tp)][(split, panel)], index[version(tc)][(split, panel)])
            if len(kept := frozen.get((method, bench, edge["parent"], edge["child"], split, panel), ())) > len(rows):
                rows, used = kept, used + 1
            if record := panel_record(dict(head, split=split, panel=panel), rows):
                records.append(record)
    print("panels taken from the frozen rows:", used)
    return records


def dev_edges():
    """The dev records of the edge table, without the excluded cell: one per DeepSeek edit scored on dev."""
    return [r for r in json.loads((S.cache("edits") / TABLE).read_text()) if r["split"] == "dev" and not excluded(r)]


if __name__ == "__main__":
    table = build()
    (S.cache("edits") / TABLE).write_text(json.dumps(table, default=float))
    dev = [r for r in table if r["split"] == "dev"]
    print("records", len(table), "dev edges", len(dev), {m: sum(r["opt"] == m for r in dev) for m in METHODS})
