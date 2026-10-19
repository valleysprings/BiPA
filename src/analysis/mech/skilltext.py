"""Paper 1 (mechanism paper): what the three success-only optimizers write into the skill, on the nine Table-1 skills (DeepSeek-V4.1-Flash, OpenCode, native-20261008-https).

Role in the argument: the text side of the mechanism. The four-term profile (appC_attribution) says which part of the input-token bill grew; this module
says what kind of rule each optimizer added to the seed skill, so the two can be set side by side. Descriptive only, no BACH, no new rollouts.

Four steps:
  1. size/diff: words and lines against the 580-word seed, lines added/removed, and whether seed text was ever deleted (line-level SequenceMatcher).
  2. coding: every added logical unit (a bullet, numbered item, table row, example line, or wrapped paragraph) of the nine final skills is assigned to one of
     eight mutually exclusive behaviour categories by a deterministic keyword/regex pass (`classify`), then every assignment was read by hand; the hand
     corrections are the explicit table OVERRIDES (key = optimizer, benchmark, first 50 characters of the unit).
  3. history: the edit operations over each run (SkillOpt ranked edits and apply report, SkillAdam unified-diff hunks, GEPA candidate lineage).
  4. relation: dominant rule category of each skill next to its largest four-term component (assets/cache/analysis/appc/attr.json), and the check of the
     59 dev parent->child edges (assets/cache/analysis/mech/edges_table.json, SkillAdam x terminal-bench-2-1 excluded, see mech/edges.py) against the run tag.

Command (from bipa): .venv/bin/python -m src.analysis.mech.skilltext compute   -> assets/cache/analysis/mech/skilltext.json
                              .venv/bin/python -m src.analysis.mech.skilltext record    -> paper/assets/data/mech/mech_skilltext.md (new file only)
"""
import argparse
import collections
import difflib
import json
import pickle
import re
from pathlib import Path

from src.utils import dataloader as C
from src.analysis.mech.edges import EXCLUDED_REASON, is_excluded

H = C.H
RUN_TAG = "native-20261008-https"
MODEL = "deepseek-flash"
OPTS = [("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
BENCHES = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
OUT_JSON = H / "assets/cache/analysis/mech/skilltext.json"
OUT_MD = H / "paper/assets/data/mech/mech_skilltext.md"
ATTR_JSON = H / "assets/cache/analysis/appc/attr.json"
EMP_MD = H / ".scratch/evidence-20261011/emp_baseline.md"
EDGE_TABLE = H / "assets/cache/analysis/mech/edges_table.json"

# ---------------------------------------------------------------------------------------------------------------------------------
# Behaviour categories (mutually exclusive; one per unit). Order = precedence of the first pass: the rare, mechanism-relevant categories come first,
# so a line that says "one at a time" is serialize even if it also says "re-read".
# ---------------------------------------------------------------------------------------------------------------------------------
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

RX = {
    "serialize": r"one at a time|sequential|one step at a time|\bONE candidate|single step|never (?:in )?parallel|not in parallel|without (?:batching|overlap)",
    "parallelize": r"in parallel|parallel(?:ly| calls| tool)|batch(?:ed|ing)? (?:the )?(?:calls|reads|requests)|concurrent|simultaneous|same turn|multiple (?:tool )?calls at once",
    "limit_read": r"read (?:it |each |the file )?once|don'?t re-?read|do not re-?read|\bhead\b|\btail\b|\bgrep\b|truncat|only the (?:relevant|needed)|concise|limit (?:the )?output",
    "ask_stop": r"explicit (?:\"?yes|confirmation|consent)|ask the user|obtain (?:explicit )?confirmation|consent|\bdeny\b|\bdenied\b|escalat|transfer to|hand ?off|\bstop\b|state (?:it|that) cannot be done|the user can|performed by the user|user-run|wait for the (?:user|result)|\bblocker\b|cannot (?:be done|run it yourself)",
    "verify": r"verif|re-?read|read (?:it )?back|re-?query|re-?run|re-?check|re-?measure|re-?apply|confirm|evidence|\bfresh|audit|validat|diff\b|compare|\bcheck|proof|prove|\btests?\b|oracle|reference|assert|\binspect|count",
    "domain": r"polic|eligib|refund|\border|record|account|\bfee|price|\brate\b|balance|dispute|\bSIM\b|roaming|\bAPI|tool|\bfile|path|binary|compile|\bgcc\b|python|clone|commit|push|curl|directory|volume|schema|sheet|email|recipient|window|deadline|clock|time|authenticat|identity|identif",
    "format": r"format|wording|summary|quote|paraphrase|phrase|report (?:the|exactly)|filename|tone|\bsay\b|\bstate\b",
}
RX = {k: re.compile(v, re.I) for k, v in RX.items()}

MARKER = re.compile(r"^\s*(?:[-*]\s|\d+\.\s|\||✅|❌)")
FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^#{1,6}\s")
TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
RULE = re.compile(r"^---\s*$")


def classify(text):
    """First-pass category of one unit: the first category in precedence order whose regex matches the unit text, else 'other'."""
    for cat in CAT_IDS:
        if cat == "other":
            continue
        # a category with no regex never fires
        if cat not in RX:
            continue
        if RX[cat].search(text):
            return cat
    return "other"


# ---------------------------------------------------------------------------------------------------------------------------------
# Units of a skill text
# ---------------------------------------------------------------------------------------------------------------------------------
def units(text):
    """Split a skill into logical units.

    Returns a list of dicts {kind, lines: [line indices], text}. kind is 'rule' (a codable unit), or one of
    'heading', 'fence', 'separator', 'frontmatter' (structure, not coded). A wrapped paragraph or an indented continuation line inside a code fence is
    merged into the unit that precedes it."""
    lines = text.splitlines()
    out = []
    in_fence = False
    in_front = False
    prev_kind = None
    prev_blank = True
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0 and RULE.match(line):
            # opening frontmatter marker
            in_front = True
            out.append({"kind": "frontmatter", "lines": [i], "text": s})
            prev_kind = "frontmatter"
            prev_blank = False
            continue
        if in_front:
            out.append({"kind": "frontmatter", "lines": [i], "text": s})
            if RULE.match(line):
                in_front = False
            prev_kind = "frontmatter"
            prev_blank = False
            continue
        if not s:
            prev_blank = True
            prev_kind = None
            continue
        if FENCE.match(line):
            in_fence = not in_fence
            out.append({"kind": "fence", "lines": [i], "text": s})
            prev_kind = "fence"
            prev_blank = False
            continue
        if TABLE_SEP.match(line):
            out.append({"kind": "separator", "lines": [i], "text": s})
            prev_kind = "separator"
            prev_blank = False
            continue
        if not in_fence and HEADING.match(line):
            out.append({"kind": "heading", "lines": [i], "text": s})
            prev_kind = "heading"
            prev_blank = False
            continue
        # a continuation joins the previous rule unit
        continues = False
        if out and out[-1]["kind"] == "rule" and not prev_blank and prev_kind == "rule":
            if in_fence:
                continues = line[:1] in (" ", "\t") and not MARKER.match(line)
            else:
                continues = not MARKER.match(line) and not s.startswith("**") and not s.startswith("|")
        if continues:
            out[-1]["lines"].append(i)
            out[-1]["text"] += " " + s
        else:
            out.append({"kind": "rule", "lines": [i], "text": s})
        prev_kind = "rule"
        prev_blank = False
    return out


def added_line_set(seed_lines, final_lines):
    """Indices of final-skill lines that are inserted or replaced relative to the seed (SequenceMatcher on whole lines)."""
    sm = difflib.SequenceMatcher(None, seed_lines, final_lines, autojunk=False)
    added = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            added.update(range(j1, j2))
    return added


def seed_line_fate(seed_lines, final_lines):
    """Seed lines that disappear: those replaced in place (opcode replace) versus removed with nothing in their place (opcode delete)."""
    sm = difflib.SequenceMatcher(None, seed_lines, final_lines, autojunk=False)
    replaced = []
    deleted = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            replaced.extend(seed_lines[i1:i2])
        if tag == "delete":
            deleted.extend(seed_lines[i1:i2])
    return replaced, deleted


def final_texts():
    """{(opt, bench): final skill text} of the pinned runs, plus the run directory."""
    texts = {}
    for opt, _ in OPTS:
        for bench, _ in BENCHES:
            run = C.EX / MODEL / opt / bench / RUN_TAG
            texts[(opt, bench)] = (run, C.final_skill(opt, run))
    return texts


# ---------------------------------------------------------------------------------------------------------------------------------
# Hand corrections of the first pass. Every one of the 277 coded units was read; OVERRIDES lists the units whose category was changed by hand.
# Key = (optimizer, benchmark); value = {unit index in the order printed by `review`: category}.
# Reasons, in short: red-flag and rationalization-table rows take the category of the behaviour they forbid; section labels such as "Email:" or
# "Rules:" and "when to apply" triggers are 'other'; "truncate/head/reference/state" regex hits that were not about limiting reads were reassigned;
# authentication and policy-procedure lines are 'domain' even when they contain the word verify.
# ---------------------------------------------------------------------------------------------------------------------------------
OVERRIDES = {
    ("gepa", "tau3-bench"): {
        2: "domain", 3: "domain", 4: "domain", 5: "domain", 10: "serialize", 12: "domain", 15: "domain", 18: "domain", 19: "domain", 20: "domain",
        23: "domain", 24: "domain", 25: "format", 35: "serialize", 37: "ask_stop", 38: "domain", 39: "domain", 40: "domain", 45: "serialize",
        47: "format", 48: "format", 49: "domain", 51: "domain", 53: "ask_stop", 57: "verify",
    },
    ("gepa", "automationbench"): {
        3: "verify", 7: "verify", 9: "domain", 11: "parallelize", 14: "verify", 18: "verify", 19: "verify", 20: "verify", 21: "verify", 22: "verify",
        23: "verify", 26: "serialize", 27: "format", 28: "format", 31: "verify", 33: "serialize", 34: "verify", 36: "format", 37: "format",
        38: "serialize", 39: "serialize", 40: "serialize", 41: "other", 45: "other", 48: "verify", 50: "other", 53: "other",
    },
    ("gepa", "terminal-bench-2-1"): {
        1: "domain", 4: "domain", 5: "domain", 7: "domain", 9: "ask_stop", 12: "domain", 15: "domain", 29: "domain", 32: "format", 35: "domain",
        38: "ask_stop", 39: "verify", 42: "domain", 44: "ask_stop", 48: "domain", 49: "domain", 51: "domain", 54: "domain", 57: "other",
    },
    ("skillopt", "tau3-bench"): {
        1: "verify", 4: "verify", 6: "ask_stop", 7: "verify", 10: "verify", 11: "ask_stop", 14: "domain", 18: "serialize", 27: "verify", 31: "ask_stop",
        32: "domain", 34: "domain", 36: "verify", 38: "verify", 40: "verify", 46: "ask_stop", 50: "verify", 51: "ask_stop", 54: "domain",
        55: "verify", 58: "verify", 60: "ask_stop", 61: "domain", 62: "ask_stop", 63: "ask_stop", 65: "verify", 66: "verify", 67: "verify",
        72: "domain", 74: "ask_stop", 80: "verify", 83: "verify", 84: "verify", 85: "verify", 86: "verify", 95: "verify", 96: "verify",
        97: "domain", 99: "verify", 100: "verify", 103: "verify",
    },
    ("skillopt", "automationbench"): {
        0: "verify", 1: "verify", 7: "other", 8: "domain", 9: "domain", 10: "domain", 14: "serialize", 19: "serialize",
    },
    ("skillopt", "terminal-bench-2-1"): {
        2: "verify", 3: "verify", 6: "verify", 8: "verify", 9: "verify",
    },
    ("skilladam", "tau3-bench"): {0: "verify", 7: "verify"},
    ("skilladam", "automationbench"): {4: "verify", 6: "verify"},
    ("skilladam", "terminal-bench-2-1"): {},
}


def code_cell(opt, bench, text):
    """Code every added unit of one final skill. Returns the list of coded units and the counts of added structural lines (not coded)."""
    seed_lines = C.SEED_TEXT.splitlines()
    final_lines = text.splitlines()
    added = added_line_set(seed_lines, final_lines)
    hand = OVERRIDES[(opt, bench)]
    coded = []
    structure = collections.Counter()
    for u in units(text):
        # a unit counts as added when any of its lines is new
        if not any(i in added for i in u["lines"]):
            continue
        if u["kind"] != "rule":
            structure[u["kind"]] += 1
            continue
        n = len(coded)
        first = classify(u["text"])
        final = hand.get(n, first)
        coded.append({"idx": n, "text": u["text"], "first_pass": first, "category": final})
    return coded, dict(structure)


def review():
    """Print every coded unit with its first-pass and final category (the by-hand check)."""
    texts = final_texts()
    for opt, _ in OPTS:
        for bench, _ in BENCHES:
            run, text = texts[(opt, bench)]
            coded, structure = code_cell(opt, bench, text)
            print(f"##### {opt} {bench} units={len(coded)} structure={structure}")
            for c in coded:
                mark = "*" if c["first_pass"] != c["category"] else " "
                print(f"[{c['idx']}]{mark}{c['category']:11s} | {c['text'][:200]}")


# ---------------------------------------------------------------------------------------------------------------------------------
# Step 1: size and diff
# ---------------------------------------------------------------------------------------------------------------------------------
def size_rows(texts):
    """Per skill: words, lines, lines added/removed (SequenceMatcher on whole lines), seed lines replaced in place versus deleted outright."""
    seed_lines = C.SEED_TEXT.splitlines()
    seed_words = len(C.SEED_TEXT.split())
    rows = {}
    for (opt, bench), (run, text) in texts.items():
        final_lines = text.splitlines()
        sm = difflib.SequenceMatcher(None, seed_lines, final_lines, autojunk=False)
        added = 0
        removed = 0
        pure_deleted = 0
        replaced = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("insert", "replace"):
                added += j2 - j1
            if tag in ("delete", "replace"):
                removed += i2 - i1
            if tag == "delete":
                pure_deleted += i2 - i1
            if tag == "replace":
                replaced += i2 - i1
        # non-blank counts, so blank-line churn does not inflate the numbers
        nb_added = sum(1 for i in added_line_set(seed_lines, final_lines) if final_lines[i].strip())
        replaced_lines, deleted_lines = seed_line_fate(seed_lines, final_lines)
        rows[f"{opt}|{bench}"] = {
            "opt": opt,
            "bench": bench,
            "run": str(run),
            "words": len(text.split()),
            "seed_words": seed_words,
            "words_ratio": len(text.split()) / seed_words,
            "lines": len(final_lines),
            "seed_lines": len(seed_lines),
            "lines_added": added,
            "lines_added_nonblank": nb_added,
            "lines_removed": removed,
            "seed_lines_deleted_outright": pure_deleted,
            "seed_lines_replaced_in_place": replaced,
            "seed_text_ever_removed": removed > 0,
            "replaced_seed_lines": [l.strip() for l in replaced_lines if l.strip()],
            "deleted_seed_lines": [l.strip() for l in deleted_lines if l.strip()],
        }
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
# Step 3 (history): edit operations over each run
# ---------------------------------------------------------------------------------------------------------------------------------
def skillopt_history(run):
    """SkillOpt: per step the action (accept/reject), the ranked edits (op and apply status), and the pre-ranking merged and per-minibatch patch ops.

    The candidate of a rejected step is dropped, so 'accepted' ops are those of steps whose action is accept_new_best."""
    base = run / "skillopt"
    history = json.loads((base / "history.json").read_text())
    steps = []
    ops_all = collections.Counter()
    ops_accepted = collections.Counter()
    merged_ops = collections.Counter()
    minibatch_ops = collections.Counter()
    for s in history:
        sdir = base / "steps" / f"step_{s['step']:04d}"
        report = json.loads((sdir / "edit_apply_report.json").read_text())
        ops = collections.Counter(e["op"] for e in report)
        status = collections.Counter(e["status"] for e in report)
        merged = json.loads((sdir / "merged_patch.json").read_text())
        for e in merged["edits"]:
            merged_ops[e["op"]] += 1
        for f in sorted((sdir / "patches").glob("*.json")):
            patch = json.loads(f.read_text())
            body = patch["patch"]
            edits = body["edits"] if isinstance(body, dict) and "edits" in body else (body if isinstance(body, list) else [])
            for e in edits:
                minibatch_ops[e.get("op", "?")] += 1
        accepted = s["action"].startswith("accept")
        ops_all.update(ops)
        if accepted:
            ops_accepted.update(ops)
        steps.append({
            "step": s["step"],
            "action": s["action"],
            "ops": dict(ops),
            "apply_status": dict(status),
            "skill_len_chars": s["skill_len"],
            "selection_hard": s["selection_hard"],
        })
    return {
        "steps": steps,
        "ops_all_steps": dict(ops_all),
        "ops_accepted_steps": dict(ops_accepted),
        "ops_merged_before_ranking": dict(merged_ops),
        "ops_minibatch_patches": dict(minibatch_ops),
        "accepted_steps": sum(1 for s in steps if s["action"].startswith("accept")),
        "rejected_steps": sum(1 for s in steps if not s["action"].startswith("accept")),
    }


def hunk_kinds(patch_text):
    """Classify each hunk of a unified diff: insert (only + lines), delete (only - lines), replace (both). Returns the counts and the +/- line totals."""
    kinds = collections.Counter()
    plus = 0
    minus = 0
    in_hunk = False
    # the hunk bodies are split into change blocks (runs of +/- lines between context lines), one kind per block
    blocks = []
    run_p = 0
    run_m = 0
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            if run_p or run_m:
                blocks.append((run_p, run_m))
            run_p = 0
            run_m = 0
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            run_p += 1
            plus += 1
        elif line.startswith("-") and not line.startswith("---"):
            run_m += 1
            minus += 1
        else:
            # a context line ends a change block
            if run_p or run_m:
                blocks.append((run_p, run_m))
            run_p = 0
            run_m = 0
    if run_p or run_m:
        blocks.append((run_p, run_m))
    for p, m in blocks:
        if p and not m:
            kinds["insert"] += 1
        elif m and not p:
            kinds["delete"] += 1
        else:
            kinds["replace"] += 1
    return dict(kinds), plus, minus


def skilladam_history(run):
    """SkillAdam: per iteration the accepted flag and the change blocks of the applied (last) patch attempt, counted as insert / delete / replace."""
    base = run / "skilladam"
    ckpt = json.loads((base / "checkpoint.json").read_text())
    iters = []
    totals = {"all": collections.Counter(), "accepted": collections.Counter()}
    plus_minus = {"all": [0, 0], "accepted": [0, 0]}
    for it in ckpt["iterations"]:
        idir = base / "iterations" / f"iter_{it['iteration']:03d}"
        files = sorted(idir.glob("patch_attempt_??.json"))
        kinds = {}
        plus = 0
        minus = 0
        if files:
            last = json.loads(files[-1].read_text())
            patch_text = last.get("patch") or ""
            kinds, plus, minus = hunk_kinds(patch_text)
        failed = it["reason"].startswith("patch application failed")
        # a failed patch application changed nothing, so its blocks are not edits to the skill
        if failed:
            kinds = {}
            plus = 0
            minus = 0
        totals["all"].update(kinds)
        plus_minus["all"][0] += plus
        plus_minus["all"][1] += minus
        if it["accepted"]:
            totals["accepted"].update(kinds)
            plus_minus["accepted"][0] += plus
            plus_minus["accepted"][1] += minus
        iters.append({
            "iteration": it["iteration"],
            "accepted": it["accepted"],
            "reason": it["reason"],
            "change_blocks": kinds,
            "plus_lines": plus,
            "minus_lines": minus,
        })
    return {
        "iterations": iters,
        "n_iterations": len(iters),
        "accepted": sum(1 for i in iters if i["accepted"]),
        "patch_failed": sum(1 for i in iters if i["reason"].startswith("patch application failed")),
        "empty_or_no_gain": sum(1 for i in iters if not i["accepted"] and not i["reason"].startswith("patch application failed")),
        "blocks_all_iterations": dict(totals["all"]),
        "blocks_accepted_iterations": dict(totals["accepted"]),
        "plus_minus_all": plus_minus["all"],
        "plus_minus_accepted": plus_minus["accepted"],
    }


def gepa_history(run):
    """GEPA: candidate pool with parents and validation scores; per candidate the line diff against its parent (GEPA rewrites the whole skill each time)."""
    base = run / "gepa"
    cands = json.loads((base / "candidates.json").read_text())
    summary = json.loads((run / "summary.json").read_text())["extra"]
    parents = summary["parents"]
    seed_lines = C.SEED_TEXT.splitlines()
    rows = []
    for i, c in enumerate(cands):
        text = c["current_candidate"]
        parent = parents[i][0] if parents[i] and parents[i][0] is not None else None
        ref = cands[parent]["current_candidate"].splitlines() if parent is not None else seed_lines
        new_lines = text.splitlines()
        sm = difflib.SequenceMatcher(None, ref, new_lines, autojunk=False)
        add = 0
        rem = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("insert", "replace"):
                add += j2 - j1
            if tag in ("delete", "replace"):
                rem += i2 - i1
        rows.append({
            "candidate": i,
            "parent": parent,
            "val_score": summary["val_scores"][i],
            "words": len(text.split()),
            "lines_added_vs_parent": add,
            "lines_removed_vs_parent": rem,
        })
    return {
        "candidates": rows,
        "n_candidates": len(rows),
        "best_idx": summary["best_idx"],
        "metric_calls": summary["metric_calls"],
        "rewrite_candidates_that_remove_lines": sum(1 for r in rows if r["parent"] is not None and r["lines_removed_vs_parent"] > 0),
    }


PAR_RX = re.compile(r"in parallel|parallel(?:ly)?\b|\bbatch|concurrent|simultaneous|same turn|at once|multiple (?:tool )?calls", re.I)
LIM_RX = re.compile(r"read (?:it |them |each file |the file )?once|do not re-?read|don'?t re-?read|\bhead\b|\btail\b|\bgrep\b|\bcache|avoid (?:re|repeat|redundant)|redundant|minimi[sz]e|only once|concise|do not repeat", re.I)


def proposed_texts(opt, run):
    """Every text the optimizer proposed as added material: (label, added lines) per proposal, including proposals that were rejected or never applied."""
    out = []
    if opt == "skillopt":
        history = json.loads((run / "skillopt" / "history.json").read_text())
        current = C.SEED_TEXT
        for s in history:
            sdir = run / "skillopt" / "steps" / f"step_{s['step']:04d}"
            cand = (sdir / "candidate_skill.md").read_text()
            added = [l for l in added_lines(current, cand)]
            out.append((f"step{s['step']}:{s['action']}", added))
            if s["action"].startswith("accept"):
                current = cand
    if opt == "skilladam":
        base = run / "skilladam"
        ckpt = json.loads((base / "checkpoint.json").read_text())
        for it in ckpt["iterations"]:
            idir = base / "iterations" / f"iter_{it['iteration']:03d}"
            for f in sorted(idir.glob("patch_attempt_??.json")):
                patch_text = json.loads(f.read_text()).get("patch") or ""
                plus = [l[1:] for l in patch_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
                tag = "accepted" if it["accepted"] else "rejected"
                out.append((f"iter{it['iteration']}:{f.stem[-2:]}:{tag}", plus))
    if opt == "gepa":
        cands = json.loads((run / "gepa" / "candidates.json").read_text())
        parents = json.loads((run / "summary.json").read_text())["extra"]["parents"]
        for i, c in enumerate(cands):
            parent = parents[i][0] if parents[i] and parents[i][0] is not None else None
            ref = cands[parent]["current_candidate"] if parent is not None else C.SEED_TEXT
            out.append((f"cand{i}", added_lines(ref, c["current_candidate"])))
    return out


def added_lines(old, new):
    """Non-blank lines of `new` that are inserted or replaced relative to `old`."""
    a = old.splitlines()
    b = new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    got = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            got.extend(l for l in b[j1:j2] if l.strip())
    return got


def proposal_scan(texts):
    """Over every proposed text (accepted or not), the added lines that mention parallel/batch (PAR_RX) or read-once/limit (LIM_RX) vocabulary.

    The hits are kept verbatim so their polarity (permit vs forbid) can be read by hand."""
    scan = {}
    for (opt, bench), (run, _) in texts.items():
        par_hits = []
        lim_hits = []
        n_lines = 0
        seen = set()
        for label, lines in proposed_texts(opt, run):
            for l in lines:
                if l in seen:
                    continue
                seen.add(l)
                n_lines += 1
                if PAR_RX.search(l):
                    par_hits.append({"at": label, "line": l.strip()[:240]})
                if LIM_RX.search(l):
                    lim_hits.append({"at": label, "line": l.strip()[:240]})
        scan[f"{opt}|{bench}"] = {"distinct_added_lines_over_all_proposals": n_lines, "parallel_vocab": par_hits, "limit_vocab": lim_hits}
    return scan


# ---------------------------------------------------------------------------------------------------------------------------------
# Step 4 (relation): rule categories next to the four-term profile
# ---------------------------------------------------------------------------------------------------------------------------------
TERM_NAMES = [
    ("more_actions", "more actions"),
    ("more_requests", "turn-splitting"),
    ("longer_content", "more content per turn"),
    ("longer_skill", "longer text"),
]
OPT_DISP = {"skillopt": "SkillOpt", "skilladam": "SkillAdam", "gepa": "GEPA"}


def four_terms():
    """The per-skill four-term numbers of appC_attribution (assets/cache/analysis/appc/attr.json, written by analysis.mechanism.appc compute)."""
    attr = json.loads(ATTR_JSON.read_text())
    out = {}
    for row in attr["skills"]:
        inv = {v: k for k, v in OPT_DISP.items()}
        key = f"{inv[row['method']]}|{row['bench']}"
        out[key] = {name: row[name] for name, _ in TERM_NAMES}
        out[key]["dlogX"] = row["dlogX"]
        out[key]["n"] = row["n"]
    return out


def relate(counts, terms):
    """For each skill, the dominant rule category (by units) and the largest four-term component (largest positive, and largest in absolute value)."""
    rel = {}
    for key, t in terms.items():
        c = counts[key]
        total = sum(c.values())
        dominant = max(CAT_IDS, key=lambda k: c.get(k, 0))
        by_value = max(TERM_NAMES, key=lambda x: t[x[0]])
        by_abs = max(TERM_NAMES, key=lambda x: abs(t[x[0]]))
        rel[key] = {
            "units": total,
            "dominant_category": dominant,
            "dominant_share": c.get(dominant, 0) / total,
            "verify_share": c.get("verify", 0) / total,
            "serialize_units": c.get("serialize", 0),
            "parallelize_units": c.get("parallelize", 0),
            "limit_read_units": c.get("limit_read", 0),
            "largest_term": by_value[1],
            "largest_term_abs": by_abs[1],
            "terms": {label: t[name] for name, label in TERM_NAMES},
            "dlogX": t["dlogX"],
        }
    return rel


# ---------------------------------------------------------------------------------------------------------------------------------
# Step 5: the 59-edge evidence file
# ---------------------------------------------------------------------------------------------------------------------------------
def emp_check():
    """Which runs the 59 dev parent->child edges of edges_table.json come from (SkillAdam x terminal-bench-2-1 excluded: two versions of that run match the same patch, so the parent of its edits is ambiguous in the logs), and the baseline-only numbers recomputed from its edge table.

    The run tag is read from emp_baseline_1_edges.py. The category counts use the rule of that analysis: an edge 'contains' a category when the added
    mass of that category is at least 150 characters (MIN in emp_baseline_3_analyze.py). Means are weighted by the number of paired tasks n, as in the md."""
    import numpy as np
    ev = EMP_MD.parent
    src = (ev / "emp_baseline_1_edges.py").read_text()
    tags = sorted(set(re.findall(r"native-\d{8}-[a-z]+", src)))
    edges = json.loads(EDGE_TABLE.read_text())
    dev = [e for e in edges if e["split"] == "dev" and not is_excluded(e)]
    cats = ["verify", "reread", "enumerate", "test", "format", "business", "scope_stop", "economize", "serialize", "other"]
    terms = ["dlogX", "actions", "turnsplit", "content", "text", "dQ"]
    rng = np.random.default_rng(0)

    def wmean(rows, term):
        w = np.array([r["n"] for r in rows], dtype=float)
        v = np.array([r[term] for r in rows], dtype=float)
        return float((w * v).sum() / w.sum())

    def boot(rows, term, reps=4000):
        n = len(rows)
        vals = []
        for _ in range(reps):
            idx = rng.integers(0, n, n)
            vals.append(wmean([rows[i] for i in idx], term))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    def summary(rows):
        out = {"edges": len(rows), "pairs": int(sum(r["n"] for r in rows))}
        for t in terms:
            out[t] = wmean(rows, t)
            out[t + "_ci"] = boot(rows, t)
        return out

    per_run = collections.Counter(f"{e['opt']}|{e['bench']}" for e in dev)
    present = {}
    for c in cats:
        has = [e for e in dev if e["mass_add"][c] >= 150]
        present[c] = {"edges_with": len(has), "edges_without": len(dev) - len(has)}
        if 1 < len(has) < len(dev) - 1:
            no = [e for e in dev if e["mass_add"][c] < 150]
            # with/without means and their difference are unweighted means over edges, as in emp_baseline_3_analyze.py (the md's contrasts)
            present[c]["with"] = {t: float(np.mean([e[t] for e in has])) for t in ("actions", "turnsplit", "dlogX")}
            present[c]["without"] = {t: float(np.mean([e[t] for e in no])) for t in ("actions", "turnsplit", "dlogX")}
            # contrast with-minus-without and its bootstrap over edges (each group resampled within itself)
            present[c]["contrast"] = {}
            for t in ("actions", "turnsplit", "dlogX"):
                diff = present[c]["with"][t] - present[c]["without"][t]
                reps = []
                for _ in range(3000):
                    a = [has[i][t] for i in rng.integers(0, len(has), len(has))]
                    b = [no[i][t] for i in rng.integers(0, len(no), len(no))]
                    reps.append(float(np.mean(a) - np.mean(b)))
                present[c]["contrast"][t] = [diff, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    # where the serialize edges come from
    serialize_where = collections.Counter(f"{e['opt']}|{e['bench']}" for e in dev if e["mass_add"]["serialize"] >= 150)
    md_text = EMP_MD.read_text()
    quotes = {
        "verify_68_of_72": "verify 在 68/72" in md_text,
        "parallel_0_of_72": "0 条" in md_text,
        "serialize_6_edges": "serialize | 6/66" in md_text,
    }
    return {
        "run_tags_in_edge_builder": tags,
        "edge_builder": str(ev / "emp_baseline_1_edges.py"),
        "from_current_runs": tags == ["native-20261008-https"],
        "edge_table_rows": len(edges),
        "dev_edges": len(dev),
        "dev_edges_per_run": dict(per_run),
        "all_dev": summary(dev),
        "by_optimizer": {o: summary([e for e in dev if e["opt"] == o]) for o in ("skillopt", "gepa", "skilladam")},
        "by_benchmark": {b: summary([e for e in dev if e["bench"] == b]) for b, _ in BENCHES},
        "category_presence": present,
        "serialize_edges_per_run": dict(serialize_where),
        "md_statements_found": quotes,
    }


# ---------------------------------------------------------------------------------------------------------------------------------
# Compute and record
# ---------------------------------------------------------------------------------------------------------------------------------
def compute():
    """Run all steps and write assets/cache/analysis/mech/skilltext.json."""
    texts = final_texts()
    sizes = size_rows(texts)
    coded = {}
    structure = {}
    counts = {}
    for (opt, bench), (run, text) in texts.items():
        units_coded, struct = code_cell(opt, bench, text)
        key = f"{opt}|{bench}"
        coded[key] = units_coded
        structure[key] = struct
        counts[key] = dict(collections.Counter(u["category"] for u in units_coded))
    # totals per optimizer and per category
    per_opt = {}
    for opt, _ in OPTS:
        c = collections.Counter()
        for bench, _ in BENCHES:
            c.update(counts[f"{opt}|{bench}"])
        per_opt[opt] = dict(c)
    total = collections.Counter()
    for c in per_opt.values():
        total.update(c)
    # examples: the first three units per category by length order of appearance across the nine skills (verbatim)
    examples = {}
    for cat in CAT_IDS:
        pool = [(k, u) for k, us in coded.items() for u in us if u["category"] == cat]
        examples[cat] = [{"skill": k, "text": u["text"]} for k, u in pool[:3]]
    history = {}
    for (opt, bench), (run, _) in texts.items():
        key = f"{opt}|{bench}"
        if opt == "skillopt":
            history[key] = skillopt_history(run)
        if opt == "skilladam":
            history[key] = skilladam_history(run)
        if opt == "gepa":
            history[key] = gepa_history(run)
    terms = four_terms()
    rel = relate(counts, terms)
    changed = sum(1 for us in coded.values() for u in us if u["first_pass"] != u["category"])
    n_units = sum(len(us) for us in coded.values())
    result = {
        "model": MODEL,
        "run_tag": RUN_TAG,
        "seed_words": len(C.SEED_TEXT.split()),
        "seed_lines": len(C.SEED_TEXT.splitlines()),
        "categories": dict(CATS),
        "sizes": sizes,
        "units_per_skill": {k: len(v) for k, v in coded.items()},
        "structure_lines_added_not_coded": structure,
        "counts": counts,
        "counts_per_optimizer": per_opt,
        "counts_total": dict(total),
        "units_total": n_units,
        "first_pass_changed_by_hand": changed,
        "examples": examples,
        "coded_units": coded,
        "history": history,
        "proposal_scan": proposal_scan(texts),
        "four_terms": terms,
        "relation": rel,
        "emp_baseline_check": emp_check(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("wrote", OUT_JSON)
    print("units", n_units, "changed by hand", changed)
    print("totals", dict(total))
    return result


def fmt_row(cells):
    """One markdown table row."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def record():
    """Write the audit record paper/assets/data/mech/mech_skilltext.md (a new file; no paperout call) from the cached JSON."""
    import datetime
    r = json.loads(OUT_JSON.read_text())
    L = []
    L.append("# mech_skilltext")
    L.append("")
    L.append("Audit record, written by the producing script below; rerunning `record` refreshes this file. Do not edit by hand.")
    L.append("")
    L.append("## 1. Paper")
    L.append("")
    L.append("- Paper 1 (mechanism paper, paper/draft2mech): the text side of the mechanism. What GEPA, SkillOpt and SkillAdam write into the skill on the nine Table-1 cells "
             "(DeepSeek-V4.1-Flash, OpenCode, native-20261008-https), set against the four-term profile of `appC_attribution`. Baselines only; no BACH, no new rollouts.")
    L.append("- Not typeset yet; no figure. Numbers feed prose and an appendix table.")
    L.append("")
    L.append("## 2. Source")
    L.append("")
    L.append("- `bipa/src/analysis/mech/skilltext.py`; numbers in `assets/cache/analysis/mech/skilltext.json`.")
    L.append("")
    L.append("Regenerate (from `/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bipa`):")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python -m src.analysis.mech.skilltext compute")
    L.append(".venv/bin/python -m src.analysis.mech.skilltext record")
    L.append("```")
    L.append("")
    L.append("## 3. When")
    L.append("")
    L.append("| run | summary.json written |")
    L.append("|---|---|")
    for key, v in r["sizes"].items():
        f = Path(v["run"]) / "summary.json"
        stamp = datetime.datetime.fromtimestamp(f.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        L.append(fmt_row([f"`{f}`", stamp]))
    L.append("")
    L.append("Record generated (UTC): " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M") + ".")
    L.append("")
    L.append("## 4. Result")
    L.append("")
    names = {o: d for o, d in OPTS}
    bn = {b: d for b, d in BENCHES}
    L.append("### 4.1 Size against the seed (580 words, 120 lines)")
    L.append("")
    L.append(fmt_row(["Optimizer", "Bench", "words", "x seed", "lines", "lines added", "lines removed", "seed lines deleted outright", "seed lines replaced in place"]))
    L.append(fmt_row(["---"] * 9))
    for key, v in r["sizes"].items():
        L.append(fmt_row([names[v["opt"]], bn[v["bench"]], v["words"], f"{v['words_ratio']:.2f}", v["lines"], v["lines_added"], v["lines_removed"],
                          v["seed_lines_deleted_outright"], v["seed_lines_replaced_in_place"]]))
    L.append("")
    L.append("Lines by `difflib.SequenceMatcher` on whole lines (blank lines included, as in `casestudy.skills_diff`). Seed text removed: " +
             ", ".join(f"{names[v['opt']]} {bn[v['bench']]} ({v['lines_removed']})" for v in r["sizes"].values() if v["seed_text_ever_removed"]) +
             ". GEPA rewrites the whole skill each round (tau3: 92 of the 120 seed lines gone from the final skill, AutoB: 62, TB2: 5); SkillOpt never removed a seed line; "
             "SkillAdam changed 3 and 2 seed lines in place on AutoB and TB2. Every removal is a line replaced by new text (opcode replace); no line was deleted with nothing in its place.")
    L.append("")
    L.append("Seed lines replaced in the cells with at most five (GEPA TB2, SkillAdam AutoB and TB2); the GEPA tau3 and AutoB lists are in the JSON (`sizes`):")
    L.append("")
    for key, v in r["sizes"].items():
        if 0 < len(v["replaced_seed_lines"]) <= 5:
            for l in v["replaced_seed_lines"]:
                L.append(f"- {names[v['opt']]} {bn[v['bench']]}: `{l[:200]}`")
    L.append("")
    L.append("### 4.2 Added rules by behaviour category")
    L.append("")
    L.append("Unit = one added logical line (bullet, numbered item, table row, example line; wrapped paragraphs and indented example continuations merged). "
             "Headings, code-fence markers and table separators added by the optimizers are not coded: " +
             "; ".join(f"{k.replace('|', ' ')}: {sum(v.values())}" for k, v in r["structure_lines_added_not_coded"].items() if v) + ". "
             f"{r['units_total']} units in all; {r['first_pass_changed_by_hand']} first-pass assignments were changed by hand after reading every unit.")
    L.append("")
    L.append("Categories (mutually exclusive, first-pass precedence in this order):")
    L.append("")
    for cid, desc in CATS:
        L.append(f"- `{cid}`: {desc}")
    L.append("")
    L.append(fmt_row(["Skill"] + CAT_IDS + ["units"]))
    L.append(fmt_row(["---"] * (len(CAT_IDS) + 2)))
    for opt, _ in OPTS:
        for bench, _ in BENCHES:
            key = f"{opt}|{bench}"
            c = r["counts"][key]
            L.append(fmt_row([f"{names[opt]} {bn[bench]}"] + [c.get(k, 0) for k in CAT_IDS] + [r["units_per_skill"][key]]))
    for opt, _ in OPTS:
        c = r["counts_per_optimizer"][opt]
        L.append(fmt_row([f"**{names[opt]} total**"] + [c.get(k, 0) for k in CAT_IDS] + [sum(c.values())]))
    c = r["counts_total"]
    L.append(fmt_row(["**All nine**"] + [c.get(k, 0) for k in CAT_IDS] + [r["units_total"]]))
    L.append("")
    L.append("Verbatim examples (first three units of each category, in the order optimizer, benchmark):")
    L.append("")
    for cid, _ in CATS:
        L.append(f"**{cid}**")
        L.append("")
        for ex in r["examples"][cid]:
            L.append(f"- ({ex['skill'].replace('|', ' ')}) `{ex['text'][:260]}`")
        if not r["examples"][cid]:
            L.append("- (none)")
        L.append("")
    L.append("### 4.3 Rule categories next to the four-term profile (log units, `appC_attribution`)")
    L.append("")
    L.append(fmt_row(["Skill", "units", "dominant category (share)", "verify share", "serialize", "parallelize", "limit_read", "more actions", "turn-splitting",
                      "more content", "longer text", "largest term"]))
    L.append(fmt_row(["---"] * 12))
    for opt, _ in OPTS:
        for bench, _ in BENCHES:
            v = r["relation"][f"{opt}|{bench}"]
            t = v["terms"]
            L.append(fmt_row([f"{names[opt]} {bn[bench]}", v["units"], f"{v['dominant_category']} ({v['dominant_share']:.2f})", f"{v['verify_share']:.2f}",
                              v["serialize_units"], v["parallelize_units"], v["limit_read_units"], f"{t['more actions']:+.3f}", f"{t['turn-splitting']:+.3f}",
                              f"{t['more content per turn']:+.3f}", f"{t['longer text']:+.3f}", v["largest_term"]]))
    L.append("")
    L.append("### 4.4 Edit operations over each run")
    L.append("")
    L.append("SkillOpt (ranked edits that reach the candidate; a rejected step's candidate is dropped):")
    L.append("")
    L.append(fmt_row(["Bench", "steps accepted/rejected", "ops, all steps", "ops, accepted steps", "ops merged before ranking", "ops in minibatch patches"]))
    L.append(fmt_row(["---"] * 6))
    for bench, _ in BENCHES:
        h = r["history"][f"skillopt|{bench}"]
        L.append(fmt_row([bn[bench], f"{h['accepted_steps']}/{h['rejected_steps']}", h["ops_all_steps"], h["ops_accepted_steps"],
                          h["ops_merged_before_ranking"], h["ops_minibatch_patches"]]))
    L.append("")
    L.append("SkillAdam (change blocks of the applied patch of each of the iterations; failed applications changed nothing):")
    L.append("")
    L.append(fmt_row(["Bench", "iterations", "accepted", "patch failed", "empty / no gain", "blocks, all", "blocks, accepted", "+/- lines, all", "+/- lines, accepted"]))
    L.append(fmt_row(["---"] * 9))
    for bench, _ in BENCHES:
        h = r["history"][f"skilladam|{bench}"]
        L.append(fmt_row([bn[bench], h["n_iterations"], h["accepted"], h["patch_failed"], h["empty_or_no_gain"], h["blocks_all_iterations"],
                          h["blocks_accepted_iterations"], h["plus_minus_all"], h["plus_minus_accepted"]]))
    L.append("")
    L.append("GEPA (whole-skill rewrites; line diff of each pool candidate against its parent, seed = candidate 0):")
    L.append("")
    L.append(fmt_row(["Bench", "candidates", "best idx", "metric calls", "candidates that remove lines of their parent", "words of candidates"]))
    L.append(fmt_row(["---"] * 6))
    for bench, _ in BENCHES:
        h = r["history"][f"gepa|{bench}"]
        L.append(fmt_row([bn[bench], h["n_candidates"], h["best_idx"], h["metric_calls"], h["rewrite_candidates_that_remove_lines"],
                          [c["words"] for c in h["candidates"]]]))
    L.append("")
    L.append("Proposal scan (every proposed text, accepted or not): added lines that mention parallel/batch or read-once/limit vocabulary; hits are in the JSON "
             "(`proposal_scan`) and were read by hand: the parallel hits are 'serialize writes / reads may be parallel' lines, 'do not batch several fixes' warnings, ` -j` build jobs, and a few false positives; the limit hits are "
             "'one-shot tool called once', 'be concise' (tone), 'avoid redundant reboots' (one GEPA tau3 candidate outside the final lineage), and `grep -c` as a checking tool. "
             "No proposal tells the agent to batch independent calls to save turns or to read a file once.")
    L.append("")
    L.append("| Skill | distinct added lines over all proposals | parallel-vocab lines | limit-vocab lines |")
    L.append("|---|---|---|---|")
    for key, v in r["proposal_scan"].items():
        L.append(fmt_row([key.replace("|", " "), v["distinct_added_lines_over_all_proposals"], len(v["parallel_vocab"]), len(v["limit_vocab"])]))
    L.append("")
    L.append("### 4.5 The 59 dev parent->child edges of `assets/cache/analysis/mech/edges_table.json`")
    L.append("")
    e = r["emp_baseline_check"]
    L.append(f"- Run tag in the edge builder (`emp_baseline_1_edges.py`): {e['run_tags_in_edge_builder']}; from the current runs: **{e['from_current_runs']}**.")
    L.append(f"- Edge table `edges_table.json`: {e['edge_table_rows']} rows, {e['dev_edges']} dev edges; per run: {e['dev_edges_per_run']}. "
             "Edges include rejected candidates (SkillOpt 13, GEPA 22, SkillAdam 24 dev edges); they are dev-panel measurements, not test. "
             f"Excluded: the 14 SkillAdam terminal-bench-2-1 dev edges, because {EXCLUDED_REASON}.")
    a = e["all_dev"]
    L.append(f"- Recomputed (n-weighted over {a['pairs']} task pairs, 95% bootstrap over edges): dlogX {a['dlogX']*100:+.1f} [{a['dlogX_ci'][0]*100:.1f}, {a['dlogX_ci'][1]*100:.1f}], "
             f"actions {a['actions']*100:+.1f} [{a['actions_ci'][0]*100:.1f}, {a['actions_ci'][1]*100:.1f}], turn-splitting {a['turnsplit']*100:+.1f} "
             f"[{a['turnsplit_ci'][0]*100:.1f}, {a['turnsplit_ci'][1]*100:.1f}], content {a['content']*100:+.1f}, text {a['text']*100:+.1f}. (old 72-edge md, with the excluded cell: +9.2 / +5.4 / +1.6 / +0.3 / +1.8.)")
    L.append("")
    L.append("Edges whose added text contains each category (at least 150 characters of the category, the md's rule), and the difference in mean term, containing minus not containing "
             "(unweighted mean over edges, as in the md; log x 100; 95% bootstrap over edges):")
    L.append("")
    L.append(fmt_row(["category (md regex)", "edges with / without", "d actions", "d turn-splitting", "d dlogX"]))
    L.append(fmt_row(["---"] * 5))
    for cat, v in e["category_presence"].items():
        if "contrast" in v:
            cc = v["contrast"]
            cells = [f"{cc[t][0]*100:+.1f} [{cc[t][1]*100:.1f}, {cc[t][2]*100:.1f}]" for t in ("actions", "turnsplit", "dlogX")]
        else:
            cells = ["-", "-", "-"]
        L.append(fmt_row([cat, f"{v['edges_with']} / {v['edges_without']}"] + cells))
    L.append("")
    L.append(f"Serialize edges (6) come from: {e['serialize_edges_per_run']}.")
    L.append("")
    L.append("## 5. Raw records")
    L.append("")
    for key, v in r["sizes"].items():
        L.append(f"- `{Path(v['run']) / 'summary.json'}`")
    L.append(f"- `{ATTR_JSON}`")
    L.append(f"- `{EMP_MD}`, `{EDGE_TABLE}`, `{e['edge_builder']}`")
    L.append("")
    L.append("## 6. Notes")
    L.append("")
    L.append("- Final skill = `summary.json` -> `final_skill` of the pinned run (same as `casestudy.skills_diff`); for SkillOpt it equals `best_skill.md`, for SkillAdam `final_skill.md`.")
    L.append("- Coding is single-label per unit. Boundary cases: GEPA AutoB 'Parallelize only read-only discovery (api_search, GETs) -- never mutations' is the only unit coded `parallelize` "
             "(a permission for reads inside a serialize-writes rule, not an instruction to batch); SkillAdam AutoB 'Mutations ... one at a time ... Parallel reads and searches remain fine' "
             "is coded `serialize`; SkillOpt tau3 'Read the ENTIRE returned snapshot' lines are coded `verify` (they ask for more reading, the opposite of `limit_read`); "
             "red-flag and rationalization-table rows take the category of the behaviour they forbid.")
    L.append("- Keyword categories are a first pass only; the final category of every unit is in the JSON (`coded_units`: `first_pass` and `category`).")
    L.append("- Four-term numbers are those of `appC_attribution` (change in log input tokens against the seed on the Table-1 common test tasks).")
    L.append("")
    OUT_MD.write_text("\n".join(L))
    print("wrote", OUT_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["review", "compute", "record"])
    a = ap.parse_args()
    if a.cmd == "review":
        review()
    if a.cmd == "compute":
        compute()
    if a.cmd == "record":
        record()
