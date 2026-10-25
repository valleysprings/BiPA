"""What the optimizers write into the skill: every added rule of the learned skills, coded into eight categories.

  python -m src.analysis.skilltext compute  -> CACHE/skilltext.json (DeepSeek, nine skills)
  python -m src.analysis.skilltext record   -> OUT/data/mech/mech_skilltext.md
  python -m src.analysis.skilltext luna     -> CACHE/rules_luna.json, OUT/data/mech/mech_rules_luna.md (Luna, and the edges that lengthen)
"""
import argparse
import collections
import difflib
import json
import re

from src.analysis import edges as E
from src.utils import dataloader as C
from src.utils.logger import md_table, write_record

RUN_TAG, MODEL = E.RUNS["deepseek-flash"], "deepseek-flash"
LUNA = "gpt-6-luna"
OPTS = [("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
BENCHES = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
LUNA_CELLS = [(o, b) for o, _ in OPTS for b in ("tau3-bench", "automationbench")]
OUT_JSON = C.CACHE / "skilltext.json"
ATTR_JSON = C.CACHE.parent / "appc/attr.json"

# mutually exclusive; the first pass takes the first category whose pattern matches, in this order
CATS = [
    ("serialize", "serialize / one step or one call at a time"),
    ("parallelize", "parallelize / batch independent calls"),
    ("limit_read", "limit what is read or returned (read once, head/tail, grep, concise output)"),
    ("ask_stop", "ask / confirm with the user / stop / deny / hand off to a human or another party"),
    ("verify", "verify / re-check / re-read / read back / enumerate-and-compare / test against a reference"),
    ("domain", "domain or environment facts and procedures (policy, eligibility, tool or file conventions, build and run steps)"),
    ("format", "formatting / wording / what to put in the final answer"),
    ("other", "other (not a behaviour rule)"),
]
CAT_IDS = [c for c, _ in CATS]
RX = {k: re.compile(v, re.I) for k, v in {
    "serialize": r"one at a time|sequential|one step at a time|\bONE candidate|single step|never (?:in )?parallel|not in parallel|without (?:batching|overlap)",
    "parallelize": r"in parallel|parallel(?:ly| calls| tool)|batch(?:ed|ing)? (?:the )?(?:calls|reads|requests)|concurrent|simultaneous|same turn|multiple (?:tool )?calls at once",
    "limit_read": r"read (?:it |each |the file )?once|don'?t re-?read|do not re-?read|\bhead\b|\btail\b|\bgrep\b|truncat|only the (?:relevant|needed)|concise|limit (?:the )?output",
    "ask_stop": r"explicit (?:\"?yes|confirmation|consent)|ask the user|obtain (?:explicit )?confirmation|consent|\bdeny\b|\bdenied\b|escalat|transfer to|hand ?off|\bstop\b|state (?:it|that) cannot be done|the user can|performed by the user|user-run|wait for the (?:user|result)|\bblocker\b|cannot (?:be done|run it yourself)",
    "verify": r"verif|re-?read|read (?:it )?back|re-?query|re-?run|re-?check|re-?measure|re-?apply|confirm|evidence|\bfresh|audit|validat|diff\b|compare|\bcheck|proof|prove|\btests?\b|oracle|reference|assert|\binspect|count",
    "domain": r"polic|eligib|refund|\border|record|account|\bfee|price|\brate\b|balance|dispute|\bSIM\b|roaming|\bAPI|tool|\bfile|path|binary|compile|\bgcc\b|python|clone|commit|push|curl|directory|volume|schema|sheet|email|recipient|window|deadline|clock|time|authenticat|identity|identif",
    "format": r"format|wording|summary|quote|paraphrase|phrase|report (?:the|exactly)|filename|tone|\bsay\b|\bstate\b",
}.items()}
PAR_RX = re.compile(r"in parallel|parallel(?:ly)?\b|\bbatch|concurrent|simultaneous|same turn|at once|multiple (?:tool )?calls", re.I)
LIM_RX = re.compile(r"read (?:it |them |each file |the file )?once|do not re-?read|don'?t re-?read|\bhead\b|\btail\b|\bgrep\b|\bcache|avoid (?:re|repeat|redundant)|redundant|minimi[sz]e|only once|concise|do not repeat", re.I)
MARKER = re.compile(r"^\s*(?:[-*]\s|\d+\.\s|\||✅|❌)")
FENCE, HEADING, RULE = re.compile(r"^\s*```"), re.compile(r"^#{1,6}\s"), re.compile(r"^---\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
TERM_NAMES = [("more_actions", "more actions"), ("more_requests", "turn-splitting"), ("longer_content", "more content per turn"),
              ("longer_skill", "longer text")]

# Hand corrections of the first pass, {unit index: category}; every unit was read. Red-flag rows take the category of the behaviour
# they forbid, section labels and triggers are 'other', authentication and policy steps are 'domain' even when they say verify.
OVERRIDES = {
    ("gepa", "tau3-bench"): {2: "domain", 3: "domain", 4: "domain", 5: "domain", 10: "serialize", 12: "domain", 15: "domain", 18: "domain",
                             19: "domain", 20: "domain", 23: "domain", 24: "domain", 25: "format", 35: "serialize", 37: "ask_stop", 38: "domain",
                             39: "domain", 40: "domain", 45: "serialize", 47: "format", 48: "format", 49: "domain", 51: "domain", 53: "ask_stop",
                             57: "verify"},
    ("gepa", "automationbench"): {3: "verify", 7: "verify", 9: "domain", 11: "parallelize", 14: "verify", 18: "verify", 19: "verify", 20: "verify",
                                  21: "verify", 22: "verify", 23: "verify", 26: "serialize", 27: "format", 28: "format", 31: "verify",
                                  33: "serialize", 34: "verify", 36: "format", 37: "format", 38: "serialize", 39: "serialize", 40: "serialize",
                                  41: "other", 45: "other", 48: "verify", 50: "other", 53: "other"},
    ("gepa", "terminal-bench-2-1"): {1: "domain", 4: "domain", 5: "domain", 7: "domain", 9: "ask_stop", 12: "domain", 15: "domain", 29: "domain",
                                     32: "format", 35: "domain", 38: "ask_stop", 39: "verify", 42: "domain", 44: "ask_stop", 48: "domain",
                                     49: "domain", 51: "domain", 54: "domain", 57: "other"},
    ("skillopt", "tau3-bench"): {1: "verify", 4: "verify", 6: "ask_stop", 7: "verify", 10: "verify", 11: "ask_stop", 14: "domain", 18: "serialize",
                                 27: "verify", 31: "ask_stop", 32: "domain", 34: "domain", 36: "verify", 38: "verify", 40: "verify", 46: "ask_stop",
                                 50: "verify", 51: "ask_stop", 54: "domain", 55: "verify", 58: "verify", 60: "ask_stop", 61: "domain",
                                 62: "ask_stop", 63: "ask_stop", 65: "verify", 66: "verify", 67: "verify", 72: "domain", 74: "ask_stop",
                                 80: "verify", 83: "verify", 84: "verify", 85: "verify", 86: "verify", 95: "verify", 96: "verify", 97: "domain",
                                 99: "verify", 100: "verify", 103: "verify"},
    ("skillopt", "automationbench"): {0: "verify", 1: "verify", 7: "other", 8: "domain", 9: "domain", 10: "domain", 14: "serialize", 19: "serialize"},
    ("skillopt", "terminal-bench-2-1"): {2: "verify", 3: "verify", 6: "verify", 8: "verify", 9: "verify"},
    ("skilladam", "tau3-bench"): {0: "verify", 7: "verify"},
    ("skilladam", "automationbench"): {4: "verify", 6: "verify"},
    ("skilladam", "terminal-bench-2-1"): {},
}
LUNA_OVERRIDES = {
    ("gepa", "tau3-bench"): {1: "serialize", 3: "verify", 4: "domain", 7: "ask_stop", 8: "ask_stop", 9: "domain", 15: "domain", 16: "domain",
                             19: "domain", 20: "domain", 21: "domain", 22: "domain", 23: "domain", 24: "format", 25: "domain", 26: "domain",
                             27: "domain", 28: "domain", 29: "serialize", 30: "domain", 31: "domain", 32: "domain", 33: "ask_stop", 36: "format"},
    ("gepa", "automationbench"): {0: "verify", 1: "domain", 3: "verify", 5: "format", 8: "verify", 9: "verify", 11: "domain", 14: "domain",
                                  16: "domain", 17: "domain", 18: "parallelize", 21: "format"},
    ("skillopt", "tau3-bench"): {1: "other", 3: "verify", 4: "verify", 5: "verify"},
    ("skillopt", "automationbench"): {1: "verify", 3: "verify"},
    ("skilladam", "tau3-bench"): {},
    ("skilladam", "automationbench"): {1: "domain", 3: "serialize", 5: "serialize"},
}


def classify(text):
    return next((cat for cat, rx in RX.items() if rx.search(text)), "other")


def units(text):
    """Logical units of a skill: {kind, lines, text}; wrapped lines and indented fence continuations join the rule before them."""
    out, in_fence, in_front, prev, blank = [], False, False, None, True
    for i, line in enumerate(text.splitlines()):
        s, kind = line.strip(), None
        if (i == 0 and RULE.match(line)) or in_front:
            in_front, kind = i == 0 or not RULE.match(line), "frontmatter"
        elif not s:
            prev, blank = None, True
            continue
        elif FENCE.match(line):
            in_fence, kind = not in_fence, "fence"
        elif TABLE_SEP.match(line):
            kind = "separator"
        elif not in_fence and HEADING.match(line):
            kind = "heading"
        if kind:
            out.append({"kind": kind, "lines": [i], "text": s})
        elif out and out[-1]["kind"] == "rule" and prev == "rule" and not blank and (
                line[:1] in (" ", "\t") and not MARKER.match(line) if in_fence
                else not MARKER.match(line) and not s.startswith(("**", "|"))):
            out[-1]["lines"].append(i)
            out[-1]["text"] += " " + s
        else:
            out.append({"kind": "rule", "lines": [i], "text": s})
        prev, blank = kind or "rule", False
    return out


def _ops(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def _added_removed(ops):
    return (sum(j2 - j1 for t, _, _, j1, j2 in ops if t in ("insert", "replace")),
            sum(i2 - i1 for t, i1, i2, _, _ in ops if t in ("delete", "replace")))


def added_line_set(seed_lines, final_lines):
    """Indices of final lines inserted or replaced relative to the seed."""
    return {j for t, _, _, j1, j2 in _ops(seed_lines, final_lines) if t in ("insert", "replace") for j in range(j1, j2)}


def added_lines(old, new):
    b = new.splitlines()
    return [line for t, _, _, j1, j2 in _ops(old.splitlines(), b) if t in ("insert", "replace") for line in b[j1:j2] if line.strip()]


def final_texts(model=MODEL, cells=None):
    """{(opt, bench): (run, final skill)} of the recorded runs."""
    tag = E.RUNS[model]
    cells = cells or [(o, b) for o, _ in OPTS for b, _ in BENCHES]
    return {(o, b): (C.EX / model / o / b / tag, C.final_skill(o, C.EX / model / o / b / tag)) for o, b in cells}


def code_cell(opt, bench, text, overrides=None):
    """Coded added units of one final skill and the counts of added structure lines (not coded)."""
    hand = OVERRIDES[(opt, bench)] if overrides is None else overrides
    added = added_line_set(C.SEED_TEXT.splitlines(), text.splitlines())
    coded, structure = [], collections.Counter()
    for u in units(text):
        if not added.intersection(u["lines"]):
            continue
        if u["kind"] != "rule":
            structure[u["kind"]] += 1
            continue
        first = classify(u["text"])
        coded.append({"idx": len(coded), "text": u["text"], "first_pass": first, "category": hand.get(len(coded), first)})
    return coded, dict(structure)


def size_rows(texts):
    """Words and lines of each skill against the seed, and the seed lines it replaced or deleted."""
    seed, seed_words, rows = C.SEED_TEXT.splitlines(), len(C.SEED_TEXT.split()), {}
    for (opt, bench), (run, text) in texts.items():
        lines = text.splitlines()
        ops = _ops(seed, lines)
        added, removed = _added_removed(ops)
        lost = {tag: [seed[k] for t, i1, i2, _, _ in ops if t == tag for k in range(i1, i2)] for tag in ("replace", "delete")}
        rows[f"{opt}|{bench}"] = {
            "opt": opt, "bench": bench, "run": str(run), "words": len(text.split()), "seed_words": seed_words,
            "words_ratio": len(text.split()) / seed_words, "lines": len(lines), "seed_lines": len(seed), "lines_added": added,
            "lines_added_nonblank": sum(1 for i in added_line_set(seed, lines) if lines[i].strip()), "lines_removed": removed,
            "seed_lines_deleted_outright": len(lost["delete"]), "seed_lines_replaced_in_place": len(lost["replace"]),
            "seed_text_ever_removed": removed > 0,
            "replaced_seed_lines": [x.strip() for x in lost["replace"] if x.strip()],
            "deleted_seed_lines": [x.strip() for x in lost["delete"] if x.strip()]}
    return rows


def skillopt_history(run):
    """Per step the action and the ranked edit ops; op totals over all, accepted, merged and per-minibatch patches."""
    base, steps = run / "skillopt", []
    total = {k: collections.Counter() for k in ("all", "accepted", "merged", "minibatch")}
    for s in json.loads((base / "history.json").read_text()):
        step = base / f"steps/step_{s['step']:04d}"
        report = json.loads((step / "edit_apply_report.json").read_text())
        ops = collections.Counter(e["op"] for e in report)
        total["merged"].update(e["op"] for e in json.loads((step / "merged_patch.json").read_text())["edits"])
        for f in sorted((step / "patches").glob("*.json")):
            body = json.loads(f.read_text())["patch"]
            edits = body["edits"] if isinstance(body, dict) and "edits" in body else body if isinstance(body, list) else []
            total["minibatch"].update(e.get("op", "?") for e in edits)
        total["all"].update(ops)
        total["accepted"].update(ops if s["action"].startswith("accept") else {})
        steps.append({"step": s["step"], "action": s["action"], "ops": dict(ops), "apply_status": dict(collections.Counter(e["status"] for e in report)),
                      "skill_len_chars": s["skill_len"], "selection_hard": s["selection_hard"]})
    accepted = sum(s["action"].startswith("accept") for s in steps)
    return {"steps": steps, "ops_all_steps": dict(total["all"]), "ops_accepted_steps": dict(total["accepted"]),
            "ops_merged_before_ranking": dict(total["merged"]), "ops_minibatch_patches": dict(total["minibatch"]),
            "accepted_steps": accepted, "rejected_steps": len(steps) - accepted}


def hunk_kinds(patch):
    """Change blocks (runs of +/- lines inside hunks) of a unified diff as insert/delete/replace counts, and the +/- line totals."""
    blocks, block, in_hunk = [], [0, 0], False
    for line in patch.splitlines():
        plus, minus = line.startswith("+") and not line.startswith("+++"), line.startswith("-") and not line.startswith("---")
        if line.startswith("@@") or (in_hunk and not plus and not minus):
            blocks.append(block)
            block, in_hunk = [0, 0], True
        elif in_hunk:
            block[0 if plus else 1] += 1
    blocks = [b for b in blocks + [block] if any(b)]
    kinds = collections.Counter("insert" if not m else "delete" if not p else "replace" for p, m in blocks)
    return dict(kinds), sum(p for p, _ in blocks), sum(m for _, m in blocks)


def skilladam_history(run):
    """Per iteration the change blocks of the last patch attempt; a failed application changed nothing."""
    base, iters = run / "skilladam", []
    total, lines = {"all": collections.Counter(), "accepted": collections.Counter()}, {"all": [0, 0], "accepted": [0, 0]}
    for it in json.loads((base / "checkpoint.json").read_text())["iterations"]:
        files = sorted((base / f"iterations/iter_{it['iteration']:03d}").glob("patch_attempt_??.json"))
        failed = it["reason"].startswith("patch application failed")
        kinds, plus, minus = hunk_kinds(json.loads(files[-1].read_text()).get("patch") or "") if files and not failed else ({}, 0, 0)
        for scope in ("all", "accepted") if it["accepted"] else ("all",):
            total[scope].update(kinds)
            lines[scope][0] += plus
            lines[scope][1] += minus
        iters.append({"iteration": it["iteration"], "accepted": it["accepted"], "reason": it["reason"], "change_blocks": kinds,
                      "plus_lines": plus, "minus_lines": minus})
    failed = [i["reason"].startswith("patch application failed") for i in iters]
    return {"iterations": iters, "n_iterations": len(iters), "accepted": sum(1 for i in iters if i["accepted"]), "patch_failed": sum(failed),
            "empty_or_no_gain": sum(1 for i, f in zip(iters, failed) if not i["accepted"] and not f),
            "blocks_all_iterations": dict(total["all"]), "blocks_accepted_iterations": dict(total["accepted"]),
            "plus_minus_all": lines["all"], "plus_minus_accepted": lines["accepted"]}


def gepa_candidates(run):
    """[(candidate text, parent index or None)] of a GEPA run, parents as its summary records them."""
    parents = json.loads((run / "summary.json").read_text())["extra"]["parents"]
    return [(c["current_candidate"], (parents[i] or [None])[0]) for i, c in enumerate(json.loads((run / "gepa/candidates.json").read_text()))]


def gepa_history(run):
    """Every candidate with its parent, dev score and line diff against the parent (the seed for the first)."""
    extra, cands = json.loads((run / "summary.json").read_text())["extra"], gepa_candidates(run)
    rows = []
    for i, (text, parent) in enumerate(cands):
        add, rem = _added_removed(_ops((cands[parent][0] if parent is not None else C.SEED_TEXT).splitlines(), text.splitlines()))
        rows.append({"candidate": i, "parent": parent, "val_score": extra["val_scores"][i], "words": len(text.split()),
                     "lines_added_vs_parent": add, "lines_removed_vs_parent": rem})
    return {"candidates": rows, "n_candidates": len(rows), "best_idx": extra["best_idx"], "metric_calls": extra["metric_calls"],
            "rewrite_candidates_that_remove_lines": sum(1 for r in rows if r["parent"] is not None and r["lines_removed_vs_parent"] > 0)}


def proposed_texts(opt, run):
    """(label, added lines) of every proposal, kept or not."""
    out = []
    if opt == "skillopt":
        current = C.SEED_TEXT
        for s in json.loads((run / "skillopt/history.json").read_text()):
            cand = (run / f"skillopt/steps/step_{s['step']:04d}/candidate_skill.md").read_text()
            out.append((f"step{s['step']}:{s['action']}", added_lines(current, cand)))
            current = cand if s["action"].startswith("accept") else current
    elif opt == "skilladam":
        for it in json.loads((run / "skilladam/checkpoint.json").read_text())["iterations"]:
            for f in sorted((run / f"skilladam/iterations/iter_{it['iteration']:03d}").glob("patch_attempt_??.json")):
                patch = json.loads(f.read_text()).get("patch") or ""
                out.append((f"iter{it['iteration']}:{f.stem[-2:]}:{'accepted' if it['accepted'] else 'rejected'}",
                            [x[1:] for x in patch.splitlines() if x.startswith("+") and not x.startswith("+++")]))
    else:
        cands = gepa_candidates(run)
        out = [(f"cand{i}", added_lines(cands[p][0] if p is not None else C.SEED_TEXT, text)) for i, (text, p) in enumerate(cands)]
    return out


def proposal_scan(texts):
    """Distinct added lines over all proposals, and those with parallel/batch or read-once/limit vocabulary (kept verbatim)."""
    scan = {}
    for (opt, bench), (run, _) in texts.items():
        seen, hits = set(), {"parallel_vocab": [], "limit_vocab": []}
        for label, lines in proposed_texts(opt, run):
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    for key, rx in (("parallel_vocab", PAR_RX), ("limit_vocab", LIM_RX)):
                        hits[key] += [{"at": label, "line": line.strip()[:240]}] if rx.search(line) else []
        scan[f"{opt}|{bench}"] = {"distinct_added_lines_over_all_proposals": len(seen), **hits}
    return scan


def four_terms():
    """Four-term split of each skill against the seed (attr.json of the attribution appendix)."""
    ids = {name: opt for opt, name in OPTS}
    return {f"{ids[r['method']]}|{r['bench']}": {**{n: r[n] for n, _ in TERM_NAMES}, "dlogX": r["dlogX"], "n": r["n"]}
            for r in json.loads(ATTR_JSON.read_text())["skills"]}


def relate(counts, terms):
    """Dominant rule category of each skill next to its largest four-term component."""
    rel = {}
    for key, t in terms.items():
        c, total = counts[key], sum(counts[key].values())
        dominant = max(CAT_IDS, key=lambda k: c.get(k, 0))
        rel[key] = {"units": total, "dominant_category": dominant, "dominant_share": c.get(dominant, 0) / total,
                    "verify_share": c.get("verify", 0) / total, "serialize_units": c.get("serialize", 0),
                    "parallelize_units": c.get("parallelize", 0), "limit_read_units": c.get("limit_read", 0),
                    "largest_term": max(TERM_NAMES, key=lambda x: t[x[0]])[1], "largest_term_abs": max(TERM_NAMES, key=lambda x: abs(t[x[0]]))[1],
                    "terms": {label: t[name] for name, label in TERM_NAMES}, "dlogX": t["dlogX"]}
    return rel


def code_all(texts, overrides):
    """{opt|bench: coded units}, {opt|bench: structure counts}, {opt|bench: category counts}."""
    coded, structure = {}, {}
    for (opt, bench), (_, text) in texts.items():
        coded[f"{opt}|{bench}"], structure[f"{opt}|{bench}"] = code_cell(opt, bench, text, overrides[(opt, bench)])
    return coded, structure, {k: dict(collections.Counter(u["category"] for u in us)) for k, us in coded.items()}


def compute():
    texts = final_texts()
    coded, structure, counts = code_all(texts, OVERRIDES)
    per_opt = {opt: dict(sum((collections.Counter(counts[f"{opt}|{b}"]) for b, _ in BENCHES), collections.Counter())) for opt, _ in OPTS}
    total = dict(sum(map(collections.Counter, per_opt.values()), collections.Counter()))
    history = {f"{o}|{b}": {"skillopt": skillopt_history, "skilladam": skilladam_history, "gepa": gepa_history}[o](run)
               for (o, b), (run, _) in texts.items()}
    terms = four_terms()
    result = {
        "model": MODEL, "run_tag": RUN_TAG, "seed_words": len(C.SEED_TEXT.split()), "seed_lines": len(C.SEED_TEXT.splitlines()),
        "categories": dict(CATS), "sizes": size_rows(texts), "units_per_skill": {k: len(v) for k, v in coded.items()},
        "structure_lines_added_not_coded": structure, "counts": counts, "counts_per_optimizer": per_opt, "counts_total": total,
        "units_total": sum(map(len, coded.values())),
        "first_pass_changed_by_hand": sum(u["first_pass"] != u["category"] for us in coded.values() for u in us),
        "examples": {cat: [{"skill": k, "text": u["text"]} for k, us in coded.items() for u in us if u["category"] == cat][:3] for cat in CAT_IDS},
        "coded_units": coded, "history": history, "proposal_scan": proposal_scan(texts), "four_terms": terms, "relation": relate(counts, terms)}
    C.CACHE.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("units", result["units_total"], "changed by hand", result["first_pass_changed_by_hand"])
    print("totals", total)


def record():
    r = json.loads(OUT_JSON.read_text())
    name, bn = dict(OPTS), dict(BENCHES)
    cells = [(o, b, f"{o}|{b}") for o, _ in OPTS for b, _ in BENCHES]
    sizes = md_table(["Optimizer", "Bench", "words", "x seed", "lines", "lines added", "lines removed", "seed lines deleted outright",
                      "seed lines replaced in place"],
                     [[name[v["opt"]], bn[v["bench"]], v["words"], f"{v['words_ratio']:.2f}", v["lines"], v["lines_added"], v["lines_removed"],
                       v["seed_lines_deleted_outright"], v["seed_lines_replaced_in_place"]] for v in r["sizes"].values()])
    rows = [[f"{name[o]} {bn[b]}"] + [r["counts"][k].get(c, 0) for c in CAT_IDS] + [r["units_per_skill"][k]] for o, b, k in cells]
    rows += [[f"**{name[o]} total**"] + [r["counts_per_optimizer"][o].get(c, 0) for c in CAT_IDS] + [sum(r["counts_per_optimizer"][o].values())]
             for o, _ in OPTS]
    rows.append(["**All nine**"] + [r["counts_total"].get(c, 0) for c in CAT_IDS] + [r["units_total"]])
    structure = "; ".join(f"{k.replace('|', ' ')}: {sum(v.values())}" for k, v in r["structure_lines_added_not_coded"].items() if v)
    body = "\n\n".join([
        f"Size against the seed ({r['seed_words']} words, {r['seed_lines']} lines; whole-line diff):", sizes,
        f"Added rules by category: {r['units_total']} units, {r['first_pass_changed_by_hand']} first-pass categories changed by hand. "
        f"Added structure lines, not coded: {structure}.", md_table(["Skill"] + CAT_IDS + ["units"], rows),
        "Edit history, proposal scan and the four-term relation of each skill: skilltext.json (history, proposal_scan, relation)."])
    write_record("skilltext", "What GEPA, SkillOpt and SkillAdam add to the seed skill, coded into eight rule categories.",
                 f"{MODEL}, {RUN_TAG}: GEPA, SkillOpt, SkillAdam on tau3-bench, AutomationBench, Terminal-Bench 2.1; final skill = summary.json final_skill.",
                 "Unit = one added bullet, numbered item, table row or example line, wrapped lines merged; first pass by keyword precedence "
                 "(" + " > ".join(CAT_IDS) + "), then every unit read by hand (OVERRIDES).", body,
                 "python -m src.analysis.skilltext compute\npython -m src.analysis.skilltext record")


def luna():
    """Luna's six learned skills coded as the DeepSeek ones, and the dev edges that lengthen the skill under both models."""
    texts = final_texts(LUNA, LUNA_CELLS)
    coded, structure, counts = code_all(texts, LUNA_OVERRIDES)
    res = {"luna": {k: {"run": str(texts[tuple(k.split("|"))][0]), "units": us, "structure": structure[k],
                        "changed_by_hand": sum(u["first_pass"] != u["category"] for u in us), "counts": counts[k]} for k, us in coded.items()},
           "sizes": size_rows(texts), "edges": {model: E.lengthen(model) for model in E.RUNS}}
    C.CACHE.mkdir(parents=True, exist_ok=True)
    (C.CACHE / "rules_luna.json").write_text(json.dumps(res, indent=1, default=str))
    ds = sum((collections.Counter(c) for c in code_all(final_texts(), OVERRIDES)[2].values()), collections.Counter())
    tot, n = sum((collections.Counter(c["counts"]) for c in res["luna"].values()), collections.Counter()), sum(len(c["units"]) for c in res["luna"].values())
    names = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam", "tau3-bench": "$\\tau^3$-bench", "automationbench": "AutoBench"}
    rows = [[names[k.split("|")[0]], names[k.split("|")[1]]] + [c["counts"].get(x, 0) for x in CAT_IDS]
            + [len(c["units"]), c["changed_by_hand"], res["sizes"][k]["words"], f"{res['sizes'][k]['words_ratio']:.2f}"] for k, c in res["luna"].items()]
    rows.append(["**all six**", ""] + [f"**{tot.get(x, 0)}**" for x in CAT_IDS] + [f"**{n}**", "", "", ""])
    ds_n = sum(ds.values())
    lines = [md_table(["optimizer", "benchmark"] + CAT_IDS + ["units", "changed by hand", "words", "words / seed"], rows), "",
             f"verify share: {tot['verify']}/{n} = {tot['verify'] / n:.0%} (DeepSeek: {ds['verify']}/{ds_n} = {ds['verify'] / ds_n:.0%}). "
             f"serialize: {tot['serialize']} (DeepSeek {ds['serialize']}). parallelize: {tot['parallelize']} (DeepSeek {ds['parallelize']}). "
             f"limit_read: {tot['limit_read']} (DeepSeek {ds['limit_read']}).", ""]
    lines += [f"- parallelize unit: `{u['text']}`" for c in res["luna"].values() for u in c["units"] if u["category"] == "parallelize"]
    lines += ["", "Seed text removed (lines replaced in place / deleted outright): "
              + "; ".join(f"{k}: {v['seed_lines_replaced_in_place']}/{v['seed_lines_deleted_outright']}" for k, v in res["sizes"].items()), "",
              "Dev edges that lengthen the skill (parent and child share a dev panel):", "",
              md_table(["model", "dev-scored edges", "lengthen", "SkillAdam TB2 (excluded): lengthen / edges"],
                       [[m, e["scored"], e["longer"], f"{e['sa_tb2_longer']} / {e['sa_tb2_scored']}"] for m, e in res["edges"].items()]), ""]
    lines += [f"- {m} shorter edge: {r['opt']} {r['bench']} {r['parent']} -> {r['child']}, {r['chars_p']} -> {r['chars_c']} chars, accepted={r['accepted']}"
              for m, e in res["edges"].items() for r in e["shorter"]]
    write_record("rules_luna", "The rules GPT-6-Luna's optimizers add, coded as the DeepSeek ones, and how many dev edges lengthen the skill under each model.",
                 f"{LUNA}, {E.RUNS[LUNA]}: GEPA, SkillOpt, SkillAdam on tau3-bench and AutomationBench (on Terminal-Bench 2.1 all three return the seed); "
                 f"edges of both models' runs, SkillAdam on Terminal-Bench 2.1 reported apart ({E.EXCLUDED_REASON}).",
                 "Same coder as mech_skilltext; hand corrections in LUNA_OVERRIDES; every unit with its category in rules_luna.json.", "\n".join(lines), "python -m src.analysis.skilltext luna")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["compute", "record", "luna"])
    {"compute": compute, "record": record, "luna": luna}[parser.parse_args().cmd]()
