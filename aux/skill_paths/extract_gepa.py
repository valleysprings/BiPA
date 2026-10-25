#!/usr/bin/env python
"""Optimization path of every GEPA run (read-only on the run artifacts; writes only under OUT/skill_paths/).

Regenerate:  python -m aux.skill_paths.extract_gepa [--only <substring>] [--verify]
Output:      OUT/skill_paths/paths/<model>__gepa__<benchmark>__<runid>.json   (idempotent: no extraction timestamp is written; the
             only time stamps are the mtimes of the run's own files, which change only if the run writes again)

What GEPA's gate is (read from src/adapters/gepa.py and optimizers/gepa/src/gepa/{core/engine.py, proposer/reflective_mutation,
strategies/acceptance.py, gepa_launcher.py}, then checked against run_log.txt / run_log.json):
  * iteration t (1-based; log label 'Iteration t', run_log.json entry i = t-1) reflects on the shared train batch B_t (8 tasks);
  * the parent is drawn from the dev Pareto front ('pareto' selector, log 'Selected program X'); the parent is re-run on B_t, the child
    (one reflective rewrite of the whole skill) is run on B_t;
  * StrictImprovementAcceptance: accept iff sum(child scores on B_t) > sum(parent scores on B_t) (binary scores, unknown -> 0.0 but strict
    outcome policy aborts on unknown); 'All subsample scores perfect' or a failed reflection call skips the iteration (no candidate);
  * an accepted child is scored on ALL of D_sel (dev) and ALWAYS appended to the pool (no dev gate); a rejected child is never scored on dev;
  * GEPA returns argmax over the pool of the dev mean (full_eval policy; ties -> larger coverage then the LOWER index, i.e. the seed first).
  * no merge proposer (GEPAConfig.merge is None in the adapter).
'kept' here = entered the pool = minibatch gate accepted. Decided from three agreeing sources: run_log.json entry has new_program_idx,
run_log.txt has 'Accepted candidate (subsample score a -> b)' with the same sums, and candidates.json / summary extra.parents hold the child.
"""
import argparse
import collections
import datetime
import difflib
import glob
import json
import os
import re
import sys
from pathlib import Path

from harness_bench.eval import version_from_files

from src.layout import ASSETS, CACHE, EXECUTION, OUT as OUT_ROOT, SEED_FILE

EXEC = EXECUTION
OUT = OUT_ROOT / "skill_paths/paths"
SEED_FILES = {"hand-written": SEED_FILE, "near-empty": ASSETS / "skills/seed-minimal/SKILL.md"}
EDGES_TABLE = CACHE / "analysis/main/edges_table.json"
BENCHES = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
RUNS = ([("deepseek-flash", b, "native-20261008-https", "hand-written") for b in BENCHES]
        + [("gpt-6-luna", b, "native-20261012", "hand-written") for b in BENCHES]
        + [("gpt-6-luna", "automationbench", "fromscratch", "near-empty")])
OPTIMIZER = "gepa"
MIN_COMPLETE = ("summary.json", "completion_audit.json", "final/SKILL.md")
# Test repetitions of the fromscratch run were still being written when it was extracted; its optimizer loop was complete.
PARTIAL_OVERRIDE = {"fromscratch": "test repetitions of this run were still being written at extraction. The optimizer loop itself is complete (summary.json, completion_audit.json, 20/20 "
                                   "iterations, no write to gepa/ since the mtimes listed), so steps are final but test evaluations may still be added."}


# ----------------------------------------------------------------------------------------------------------------------------------
def vid(text):
    """harness-bench version id of a one-file skill."""
    return version_from_files({"SKILL.md": text})["version_id"]


def jload(path):
    with open(path) as handle:
        return json.load(handle)


def iso(path):
    try:
        return datetime.datetime.fromtimestamp(os.stat(path).st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def words(text):
    return len(text.split())


def line_diff(parent_text, child_text):
    """(added, removed) lines of child vs parent: difflib on the non-empty stripped lines (the lead's rule)."""
    a = [x.strip() for x in parent_text.splitlines() if x.strip()]
    b = [x.strip() for x in child_text.splitlines() if x.strip()]
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += b[j1:j2]
        if op in ("delete", "replace"):
            removed += a[i1:i2]
    return added, removed


# ----------------------------------------------------------------------------------------------------------------------------------
class Evals:
    """Evaluation.json files of one run, found by (split, version id, task panel); loaded lazily and summarised."""

    def __init__(self, run):
        self.root = run / "results/evaluations"
        self.cache = {}
        self.dirs = {}
        for split in ("train", "dev", "test"):
            for d in sorted(glob.glob(str(self.root / split / "*"))):
                name = os.path.basename(d)
                if re.fullmatch(r"[0-9a-f]{12}-[0-9a-f]{12}", name) and os.path.isfile(d + "/evaluation.json"):
                    self.dirs.setdefault((split, name[:12]), []).append(d + "/evaluation.json")

    def load(self, path):
        if path not in self.cache:
            try:
                data = jload(path)
            except (OSError, ValueError) as err:  # a file being written by a live run
                raise RuntimeError(f"cannot read {path}: {err}") from err
            eps = data["episodes"]
            toks = [sum((q.get("input_tokens") or 0) for q in e["requests"]) for e in eps if e.get("requests")]
            self.cache[path] = {
                "path": path, "version_id": data["version_id"], "panel": sorted(p["task_id"] for p in data["panel"]),
                "n": len(eps), "n_success": sum(1 for e in eps if e["success"] is True),
                "n_unknown": sum(1 for e in eps if e["success"] is None),
                "mean_input_tokens_per_episode": (sum(toks) / len(toks)) if toks else None, "n_token_episodes": len(toks),
                "per_task": {e["task_id"]: e["success"] for e in eps},
            }
        return self.cache[path]

    def find(self, split, text, panel_tasks=None):
        """Evaluation summary of skill `text` in `split` (on the panel with these task ids when given); None when absent."""
        full = vid(text)
        hits = []
        for path in self.dirs.get((split, full[:12]), []):
            ev = self.load(path)
            if ev["version_id"] != full:
                continue
            if panel_tasks is not None and ev["panel"] != sorted(panel_tasks):
                continue
            hits.append(ev)
        if len(hits) > 1:
            raise RuntimeError(f"several evaluations of {full[:12]} in {split}: {[h['path'] for h in hits]}")
        return hits[0] if hits else None

    def all_dev(self, panel_tasks):
        """Every dev evaluation on the dev panel: {version_id: summary}."""
        out = {}
        for (split, _), paths in self.dirs.items():
            if split != "dev":
                continue
            for path in paths:
                ev = self.load(path)
                if ev["panel"] == sorted(panel_tasks):
                    out[ev["version_id"]] = ev
        return out


def stats(ev):
    """{n, n_success, n_unknown, mean_input_tokens_per_episode (+ n_token_episodes)} of an evaluation summary, or None."""
    if ev is None:
        return None
    return {"n": ev["n"], "n_success": ev["n_success"], "n_unknown": ev["n_unknown"],
            "mean_input_tokens_per_episode": ev["mean_input_tokens_per_episode"], "n_token_episodes": ev["n_token_episodes"]}


# ----------------------------------------------------------------------------------------------------------------------------------
STRUCT = re.compile(
    r"^Iteration (?P<t>\d+): (?:(?P<base>Base program full valset score)"
    r"|Selected program (?P<sel>\d+) score: (?P<selval>[0-9.]+)"
    r"|(?P<perfect>All subsample scores perfect for parent \d+)"
    r"|(?P<noprop>Reflective mutation did not propose a new candidate)"
    r"|(?P<prop>Proposed new text for current_candidate: )"
    r"|New subsample score (?P<rej_new>[0-9.]+) is not better than old score (?P<rej_old>[0-9.]+), skipping"
    r"|Accepted candidate \(subsample score (?P<acc_old>[0-9.]+) -> (?P<acc_new>[0-9.]+)\); running full eval\."
    r"|(?P<better>Found a better program)"
    r"|Valset score for new program: (?P<dv>[0-9.]+) \(coverage (?P<dcov>\d+) / (?P<dof>\d+)\)"
    r"|(?P<fixed>Val aggregate|Individual valset scores|Objective aggregate|New valset pareto|Objective pareto|Valset pareto|Updated valset"
    r"|Updated objective|Best valset|Best program|Best score|Linear pareto)"
    r"|New program candidate index: (?P<newidx>\d+)"
    r"|(?P<exc>Exception during optimization))")


def parse_log(path):
    """The attempts of run_log.txt. An attempt starts at 'Iteration t: Selected program X' and ends at the next such line or 'Base program'
    line (a resume). Returns (attempts, resumes) with attempt = {t, parent, proposal(list of lines)|None, gate: None|dict, perfect, noprop,
    new_idx, dev_logged, misc, after_resume}."""
    attempts, resumes, cur, in_text, after_resume = [], [], None, False, False
    with open(path) as handle:
        lines = handle.read().split("\n")
    for line in lines:
        if line.startswith("Loading gepa state from run dir"):  # printed by the process that resumes; ends the text of a cut-off proposal
            in_text = False
            continue
        m = STRUCT.match(line)
        if not m:
            if in_text and cur is not None:
                cur["proposal"].append(line)
            elif cur is not None and line.strip():
                cur["misc"].append(line[:300])
            continue
        in_text = False
        t = int(m.group("t"))
        if m.group("base"):
            first = not resumes and not attempts  # the first 'Base program' line is the fresh start, later ones are resumes
            resumes.append({"label": t, "before_attempt": len(attempts), "fresh_start": first})
            after_resume, cur = (not first), None
        elif m.group("sel") is not None:
            cur = {"t": t, "parent": int(m.group("sel")), "parent_val": float(m.group("selval")), "proposal": None, "gate": None,
                   "perfect": False, "noprop": False, "new_idx": None, "dev_logged": None, "misc": [], "after_resume": after_resume}
            after_resume = False
            attempts.append(cur)
        elif cur is None or cur["t"] != t:
            raise RuntimeError(f"log line outside an attempt: {line[:120]}")
        elif m.group("perfect"):
            cur["perfect"] = True
        elif m.group("noprop"):
            cur["noprop"] = True
        elif m.group("prop"):
            cur["proposal"] = [line.split("current_candidate: ", 1)[1]]
            in_text = True
        elif m.group("rej_new") is not None:
            cur["gate"] = {"decision": "reject", "child_sum": float(m.group("rej_new")), "parent_sum": float(m.group("rej_old"))}
        elif m.group("acc_new") is not None:
            cur["gate"] = {"decision": "accept", "parent_sum": float(m.group("acc_old")), "child_sum": float(m.group("acc_new"))}
        elif m.group("dv") is not None:
            cur["dev_logged"] = {"score": float(m.group("dv")), "coverage": int(m.group("dcov")), "of": int(m.group("dof"))}
        elif m.group("newidx") is not None:
            cur["new_idx"] = int(m.group("newidx"))
        elif m.group("exc"):
            cur["misc"].append(line[:300])
    return attempts, resumes


def text_variants(block_lines):
    """Candidate texts of a logged proposal block: as logged, trailing-whitespace variants, then with up to 3 trailing lines dropped (log noise
    after a cut-off proposal). The caller keeps the first variant whose version id exists on disk."""
    seen, out = set(), []
    for drop in range(0, 4):
        part = block_lines[:len(block_lines) - drop] if drop else block_lines
        raw = "\n".join(part)
        for label, text in (("raw", raw), ("rstrip-newlines", raw.rstrip("\n")), ("strip", raw.strip()), ("rstrip", raw.rstrip())):
            if text not in seen:
                seen.add(text)
                out.append((label if not drop else f"{label}-drop{drop}", text))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
def extract(model, bench, run_id, seed_kind):
    run = EXEC / model / OPTIMIZER / bench / run_id
    g = run / "gepa"
    anomalies, checks = [], {}
    trace = jload(g / "run_log.json")
    cands = [c["current_candidate"] for c in jload(g / "candidates.json")]
    proto = jload(run / "native_protocol.json")
    batches, sel_tasks, test_tasks = proto["batches"], proto["panels"]["sel"], proto["panels"]["test"]
    summary = jload(run / "summary.json") if (run / "summary.json").exists() else None
    extra = (summary or {}).get("extra", {})
    policy = jload(run / "gepa-outcome-policy.json") if (run / "gepa-outcome-policy.json").exists() else None
    attempts, resumes = parse_log(g / "run_log.txt")
    ev = Evals(run)

    # mtimes and partial flag -------------------------------------------------------------------------------------------------
    files = ["gepa/run_log.json", "gepa/run_log.txt", "gepa/candidates.json", "gepa/gepa_state.bin", "gepa/gepa.stop", "gepa/candidate_tree.html",
             "summary.json", "completion_audit.json", "final/SKILL.md", "native-resumes.jsonl", "gepa-outcome-policy.json", "gepa-incomplete.json"]
    mtimes = {f: iso(run / f) for f in files if (run / f).exists()}
    eval_files = glob.glob(str(run / "results/evaluations/*/*/evaluation.json"))
    if eval_files:
        newest = max(eval_files, key=os.path.getmtime)
        mtimes["newest_evaluation_json"] = iso(newest)
        mtimes["newest_evaluation_json_path"] = os.path.relpath(newest, run)
    missing = [f for f in MIN_COMPLETE if not (run / f).exists()]
    partial = bool(missing) or (run / "gepa-incomplete.json").exists() or not (summary and (summary.get("test") or {}).get("final"))
    if partial:
        anomalies.append(f"PARTIAL: completion markers missing {missing}")
    if run_id in PARTIAL_OVERRIDE:
        partial = True
        anomalies.append("PARTIAL (by the task's rule): " + PARTIAL_OVERRIDE[run_id])

    # seed ----------------------------------------------------------------------------------------------------------------------
    seed_text = SEED_FILES[seed_kind].read_text()
    checks["pool_0_equals_seed_file"] = (cands[0] == seed_text) or (vid(cands[0]) == vid(seed_text)) or (cands[0].strip() == seed_text.strip())
    if cands[0] != seed_text:
        anomalies.append(f"candidates[0] != {SEED_FILES[seed_kind].name} byte for byte (strip-equal: {cands[0].strip() == seed_text.strip()}); "
                         f"version ids {vid(cands[0])[:12]} / {vid(seed_text)[:12]}")
    dev_panel_size = len(sel_tasks)

    # pool structure from the trace -----------------------------------------------------------------------------------------------
    parent_of, accepted_at = {0: None}, {}
    for entry in trace:
        if "new_program_idx" in entry:
            idx = entry["new_program_idx"]
            parent_of[idx] = entry["selected_program_candidate"]
            accepted_at[idx] = entry["i"] + 1
    checks["pool_size_matches_candidates"] = sorted(parent_of) == list(range(len(cands)))
    if extra.get("parents"):
        sp = {i: (p[0] if p and p[0] is not None else None) for i, p in enumerate(extra["parents"])}
        checks["parents_trace_equals_summary"] = sp == parent_of
    checks["batches_are_consecutive_dataset_ids"] = all(
        entry.get("subsample_ids") == list(range(sum(len(b) for b in batches[:entry["i"]]), sum(len(b) for b in batches[:entry["i"] + 1])))
        for entry in trace)

    # match attempts to trace iterations -------------------------------------------------------------------------------------------
    by_t = collections.defaultdict(list)
    for k, a in enumerate(attempts):
        by_t[a["t"]].append(k)
    committed = {}
    for entry in trace:
        t = entry["i"] + 1
        want_acc = "new_program_idx" in entry
        scored = "new_subsample_scores" in entry
        match = []
        for k in by_t.get(t, []):
            a = attempts[k]
            if a["parent"] != entry["selected_program_candidate"]:
                continue
            if scored:
                gate = a["gate"]
                if not gate or gate["decision"] != ("accept" if want_acc else "reject"):
                    continue
                if gate["parent_sum"] != sum(entry["subsample_scores"]) or gate["child_sum"] != sum(entry["new_subsample_scores"]):
                    continue
                if want_acc and a["new_idx"] != entry["new_program_idx"]:
                    continue
            else:
                if a["gate"] or a["proposal"] is not None or not (a["perfect"] or a["noprop"]):
                    continue
            match.append(k)
        if not match:
            anomalies.append(f"iteration {t}: no log attempt matches the run_log.json entry")
            continue
        committed[t] = match[-1]  # the last matching attempt is the one the saved state holds
        if len(match) > 1:
            anomalies.append(f"iteration {t}: {len(match)} log attempts match the trace; took the last")
    checks["every_trace_iteration_has_a_logged_attempt"] = all((e["i"] + 1) in committed for e in trace)
    aborted_idx = [k for k in range(len(attempts)) if k not in set(committed.values())]

    # steps ------------------------------------------------------------------------------------------------------------------------
    seed_ev = ev.find("dev", cands[0], sel_tasks)
    if seed_ev is None:
        anomalies.append("seed dev evaluation not found")
    steps = [{"step": 0, "event": "seed", "candidate_id": "c0", "parent_id": None, "kept": None, "dev": stats(seed_ev), "gate": None, "train": None,
              "skill_chars": len(cands[0]), "skill_words": words(cands[0]), "skill_text": cands[0], "added_lines": [], "removed_lines": [],
              "pool_index": 0, "version_id": vid(cands[0]), "val_score_recorded": (extra.get("val_scores") or [None])[0]}]
    skipped, text_source = [], collections.Counter()
    gate_mismatch = []
    dev_by_idx = {0: seed_ev}
    for entry in trace:
        t = entry["i"] + 1
        k = committed.get(t)
        a = attempts[k] if k is not None else None
        parent_idx = entry["selected_program_candidate"]
        ids = entry["subsample_ids"]
        task_ids = batches[t - 1]
        if "new_subsample_scores" not in entry:
            reason = "all_scores_perfect" if (a and a["perfect"]) else ("reflection_failed" if a and a["noprop"] else "unknown")
            misc = [m for m in (a["misc"] if a else []) if m.startswith(("Batched reflection failed", "Per-task reflection failed"))]
            pev = ev.find("train", cands[parent_idx], task_ids)
            skipped.append({"step": t, "reason": reason, "parent_id": f"c{parent_idx}", "minibatch_task_ids": task_ids, "log": misc,
                            "parent_minibatch": stats(pev),
                            "note": "no candidate was produced; the iteration still consumed batch B_t"})
            continue
        accepted = "new_program_idx" in entry
        pidx = entry.get("new_program_idx")
        old, new = entry["subsample_scores"], entry["new_subsample_scores"]
        # candidate text
        variants = text_variants(a["proposal"]) if a and a["proposal"] is not None else []
        if accepted:
            text = cands[pidx]
            hit = next((lab for lab, tx in variants if tx == text), None)
            if hit is None:
                anomalies.append(f"iteration {t}: log proposal text differs from candidates.json[{pidx}] (candidates.json used)")
            text_source["candidates.json (log text equal: " + str(hit) + ")"] += 1
        else:
            text, hit = None, None
            for lab, tx in variants:
                if ev.find("train", tx, task_ids) is not None:
                    text, hit = tx, lab
                    break
            if text is None:
                text = variants[0][1]
                anomalies.append(f"iteration {t}: rejected proposal text does not match any train evaluation of batch B_{t} (log text used as is)")
                hit = "unmatched"
            text_source[f"run_log.txt ({hit})"] += 1
        parent_text = cands[parent_idx]
        cev = ev.find("train", text, task_ids)
        pev = ev.find("train", parent_text, task_ids)
        # verify the logged minibatch scores against the episode files
        for who, e_, vec in (("parent", pev, old), ("child", cev, new)):
            if e_ is None:
                gate_mismatch.append(f"iteration {t}: {who} minibatch evaluation not found")
                continue
            from_eval = [float(e_["per_task"][x] or 0) for x in task_ids]
            if from_eval != [float(x) for x in vec]:
                gate_mismatch.append(f"iteration {t}: {who} minibatch scores differ from episodes ({vec} vs {from_eval})")
        added, removed = line_diff(parent_text, text)
        dev = None
        if accepted:
            dev_by_idx[pidx] = ev.find("dev", text, sel_tasks)
            dev = stats(dev_by_idx[pidx])
            if dev is None:
                anomalies.append(f"iteration {t}: accepted candidate c{pidx} has no dev evaluation")
        gate = {"kind": "minibatch_sum_strict_improvement",
                "detail": {"rule": "accept iff sum(child scores on B_t) > sum(parent scores on B_t); parent re-run on B_t; binary success, unknown scored 0",
                           "iteration": t, "parent_pool_index": parent_idx, "minibatch_dataset_ids": ids, "minibatch_task_ids": task_ids,
                           "parent_scores": old, "child_scores": new, "parent_sum": sum(old), "child_sum": sum(new),
                           "accept": sum(new) > sum(old), "parent_minibatch": stats(pev), "parent_dev_score_logged_at_selection": a["parent_val"] if a else None}}
        if gate["detail"]["accept"] != accepted:
            anomalies.append(f"iteration {t}: recomputed gate says {gate['detail']['accept']} but the trace says accepted={accepted}")
        train = None if cev is None else {"n": cev["n"], "n_success": cev["n_success"], "n_unknown": cev["n_unknown"],
                                          "mean_input_tokens_per_episode": cev["mean_input_tokens_per_episode"]}
        cid = f"c{pidx}" if accepted else f"r{t}"
        steps.append({"step": t, "event": "candidate", "candidate_id": cid, "parent_id": f"c{parent_idx}", "kept": accepted, "dev": dev, "gate": gate,
                      "train": train, "skill_chars": len(text), "skill_words": words(text), "skill_text": text, "added_lines": added,
                      "removed_lines": removed, "pool_index": pidx, "version_id": vid(text),
                      "val_score_recorded": (extra["val_scores"][pidx] if (accepted and extra.get("val_scores")) else None)})
    checks["logged_minibatch_scores_equal_episode_files"] = not gate_mismatch
    kept_vids = [s["version_id"] for s in steps if s["event"] == "seed" or s["kept"]]
    checks["pool_texts_are_distinct"] = len(set(kept_vids)) == len(kept_vids)
    noop = [s["candidate_id"] for s in steps if s["event"] == "candidate" and s["version_id"] == vid(cands[int(s["parent_id"][1:])])]
    checks["no_proposal_identical_to_its_parent"] = not noop
    if noop:
        anomalies.append(f"proposals whose text equals their parent's (memoized, same episodes): {noop}")
    anomalies += gate_mismatch

    # dev scores against GEPA's own val_scores ----------------------------------------------------------------------------------
    if extra.get("val_scores"):
        vs = extra["val_scores"]
        checks["pool_size_equals_len_val_scores"] = len(vs) == len(cands)
        ok = True
        for i, e_ in dev_by_idx.items():
            if e_ is None or abs(e_["n_success"] / dev_panel_size - vs[i]) > 1e-9:
                ok = False
                anomalies.append(f"c{i}: dev n_success {None if e_ is None else e_['n_success']}/{dev_panel_size} != logged val score {vs[i]}")
        checks["dev_n_success_equals_val_scores"] = ok
    dev_unknown = {f"c{i}": e_["n_unknown"] for i, e_ in dev_by_idx.items() if e_ and e_["n_unknown"]}
    if dev_unknown:
        anomalies.append(f"pool candidates with unknown dev outcomes (scored 0 by GEPA): {dev_unknown}")

    # aborted attempts and dev evaluations outside the pool ------------------------------------------------------------------------
    pool_vids = {vid(c) for c in cands}
    aborted = []
    for k in aborted_idx:
        a = attempts[k]
        task_ids = batches[a["t"] - 1]
        rec = {"iteration": a["t"], "parent_id": f"c{a['parent']}", "after_resume": a["after_resume"],
               "committed_attempt_exists_for_iteration": a["t"] in committed, "stage": "selected_only" if a["proposal"] is None else "proposal_not_gated"}
        if a["proposal"] is not None:
            tx = None
            for lab, v in text_variants(a["proposal"]):
                if ev.find("train", v, task_ids) is not None or vid(v) in pool_vids or ev.find("dev", v, sel_tasks) is not None:
                    tx = v
                    break
            tx = tx if tx is not None else text_variants(a["proposal"])[0][1]
            c_ev, p_ev = ev.find("train", tx, task_ids), ev.find("train", cands[a["parent"]], task_ids)
            would = None
            if c_ev is not None and p_ev is not None:
                would = {"note": "the optimizer's own gate never ran or was not logged for this attempt; this is the minibatch comparison recomputed from the "
                                 "episode files, shown for information only", "parent_sum": p_ev["n_success"], "child_sum": c_ev["n_success"],
                         "would_accept": c_ev["n_success"] > p_ev["n_success"]}
            rec.update({"version_id": vid(tx), "skill_chars": len(tx), "skill_words": words(tx), "would_be_gate": would,
                        "train_minibatch": stats(c_ev), "dev": stats(ev.find("dev", tx, sel_tasks)),
                        "skill_text": tx, "added_lines": line_diff(cands[a["parent"]], tx)[0], "removed_lines": line_diff(cands[a["parent"]], tx)[1]})
        rec["log_misc"] = a["misc"][:3]
        aborted.append(rec)
    outside = []
    aborted_vids = {r.get("version_id") for r in aborted}
    for vv, e_ in sorted(ev.all_dev(sel_tasks).items()):
        if vv not in pool_vids:
            outside.append({"version_id": vv, "dev": stats(e_), "matches_aborted_attempt": vv in aborted_vids})
    if outside:
        anomalies.append(f"{len(outside)} dev evaluation(s) of versions that are not in the pool: {[(o['version_id'][:12], o['matches_aborted_attempt']) for o in outside]}")
    if aborted:
        anomalies.append(f"{len(aborted)} aborted attempt(s) (iterations {[r['iteration'] for r in aborted]}); not steps, see aborted_attempts")

    # returned skill and test ------------------------------------------------------------------------------------------------------
    vs = extra.get("val_scores")
    if vs is None:  # run without summary: recompute from dev evaluations
        vs = [None if dev_by_idx.get(i) is None else dev_by_idx[i]["n_success"] / dev_panel_size for i in range(len(cands))]
    best = max(range(len(vs)), key=lambda i: (vs[i] if vs[i] is not None else -1, -i))
    if "best_idx" in extra:
        checks["best_idx_is_argmax_ties_lowest_index"] = extra["best_idx"] == best
        best_idx = extra["best_idx"]
    else:
        best_idx = best
    returned_text = cands[best_idx]
    final_ok = {"summary_final_skill": None if not summary else summary.get("final_skill") == returned_text,
                "final_SKILL_md": ((run / "final/SKILL.md").read_text() == returned_text) if (run / "final/SKILL.md").exists() else None}
    checks["returned_text_equals_summary_final_skill_and_final_SKILL_md"] = final_ok
    tev = ev.find("test", returned_text, test_tasks)
    final_test = None
    if tev is not None:
        final_test = {"n_episodes": tev["n"], "n_success": tev["n_success"], "n_unknown": tev["n_unknown"],
                      "source": os.path.relpath(tev["path"], EXEC), "version_id": tev["version_id"],
                      "returned_skill_is_seed": best_idx == 0}
        checks["test_evaluation_version_equals_returned_skill"] = tev["version_id"] == vid(returned_text)
        if summary and (summary.get("test") or {}).get("final"):
            sf = summary["test"]["final"]
            known = tev["n"] - tev["n_unknown"]
            checks["test_n_success_equals_summary_test_final"] = (sf.get("known") == known and known > 0
                                                                  and abs(tev["n_success"] / known - sf["success"]) < 1e-9)
    else:
        anomalies.append("no test evaluation found for the returned skill")

    # notes -------------------------------------------------------------------------------------------------------------------------
    n_dev_scored = sum(1 for s in steps if s["event"] == "candidate" and s["dev"] is not None)
    notes = [
        f"GEPA iterations completed (trace entries) {len(trace)} of T={(summary or {}).get('T')}; steps = {n_dev_scored} accepted (pool entrants, dev-scored) + "
        f"{sum(1 for s in steps if s['event'] == 'candidate' and not s['kept'])} rejected-on-minibatch (dev=null) + the seed; "
        f"{len(skipped)} iteration(s) produced no candidate (skipped_iterations); {len(aborted)} aborted attempt(s) (aborted_attempts).",
        "step = GEPA iteration number t (the log's 'Iteration t', batch B_t); steps are therefore not contiguous when iterations were skipped. "
        "candidate_id: c<k> = pool index k (c0 = seed, as edges.py), r<t> = proposal of iteration t rejected on the minibatch.",
        "gate.kind minibatch_sum_strict_improvement = StrictImprovementAcceptance: accept iff sum(child) > sum(parent) on the 8-task train batch B_t; "
        "the parent is re-run on B_t. There is no dev gate: an accepted child is scored on all of D_sel and always enters the pool; the returned "
        "skill is the pool's argmax of the dev mean (ties to the lowest index, so the seed). kept=True means pool entrant. Verified in engine.py "
        "(_run_reflective_batch), strategies/acceptance.py and against run_log.txt ('Accepted candidate' / 'New subsample score ... not better').",
        "parent = the log's 'Selected program X' = the entry of run_log.json selected_program_candidate (drawn from the dev Pareto front, 'pareto' "
        "selector, not the best program); for pool entrants it equals extra.parents in summary.json (checked).",
        "train = the CHILD's outcome on B_t read from results/evaluations/train (n tasks, successes, unknown); the gate sums are in gate.detail. "
        "dev mean_input_tokens_per_episode = mean over episodes that have request records of the sum of requests[].input_tokens (cached tokens included); "
        "n_token_episodes counts those episodes.",
        "candidate texts: accepted = candidates.json; rejected = parsed from run_log.txt ('Proposed new text for current_candidate:' until the gate line) "
        "and accepted only if its version id has a train evaluation on B_t. text source counts: " + json.dumps(dict(text_source)) + ".",
        "outcome policy (gepa-outcome-policy.json): " + json.dumps(policy) + ". Under strict outcomes an unknown verdict aborts the checkpoint and the run resumes, "
        "which is why some iterations appear twice in run_log.txt (aborted attempt, then a retry that may select a different parent).",
        "final_test = the single test evaluation of the returned skill recorded inside this run (results/evaluations/test); it is not the avg@3 over the "
        "extra test reruns that the paper's Table 1 reports. If the returned skill is the seed (returned_skill_is_seed) GEPA kept the seed. "
        "gepa_state.bin is not unpickled: summary.json extra.parents/val_scores/best_idx are that state's own result fields and run_log.json is its trace.",
    ]
    n_resumes = sum(1 for r in resumes if not r["fresh_start"])
    if n_resumes:
        notes.append(f"run_log.txt records {n_resumes} resume(s) (a 'Base program' line after the first one; a resume reloads the last saved state, so "
                     "an iteration cut off by an abort is redone, possibly with another parent).")
    if (run / "gepa/gepa.stop").exists():
        notes.append("gepa/gepa.stop exists (FileStopper: graceful stop requested at " + mtimes.get("gepa/gepa.stop", "?") + "), so the run ended before T iterations.")
    if partial:
        notes.append("Marked partial: " + "; ".join(anomalies[:1]))
    else:
        notes.append("partial=false because the completion markers exist (summary.json with test.final, completion_audit.json, final/SKILL.md); "
                     "mtimes_seen lists the last write of each file.")

    result = {
        "model": model, "optimizer": OPTIMIZER, "benchmark": bench, "run_id": run_id, "seed": seed_kind, "partial": partial, "mtimes_seen": mtimes,
        "dev_panel_size": dev_panel_size, "seed_dev": stats(seed_ev),
        "steps": steps, "returned_skill_id": f"c{best_idx}", "final_test": final_test, "notes": " ".join(notes),
        # additions beyond the schema (harmless extra keys)
        "T": (summary or {}).get("T"), "iterations_completed": len(trace), "skipped_iterations": skipped, "aborted_attempts": aborted,
        "dev_evaluations_outside_pool": outside, "verification": checks, "anomalies": anomalies,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--verify", action="store_true", help="cross-check the dev-scored candidate counts against the paper's edge table")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for model, bench, run_id, seed_kind in RUNS:
        name = f"{model}__{OPTIMIZER}__{bench}__{run_id}"
        if args.only and args.only not in name:
            continue
        res = extract(model, bench, run_id, seed_kind)
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        n_dev = sum(1 for s in res["steps"] if s["event"] == "candidate" and s["dev"] is not None)
        print(f"{name}: steps={len(res['steps'])} dev_scored_nonseed={n_dev} kept={sum(1 for s in res['steps'] if s['kept'])} partial={res['partial']} "
              f"skipped={len(res['skipped_iterations'])} aborted={len(res['aborted_attempts'])} anomalies={len(res['anomalies'])}")
        written.append(res)
    if args.verify:
        verify(written)


def verify(results):
    """DeepSeek: dev-scored non-seed GEPA candidates per benchmark against edges_table.json (the table the paper's mech_edges.md is computed from)."""
    table = [r for r in jload(EDGES_TABLE) if r["opt"] == "gepa" and r["split"] == "dev"]
    total = 0
    for res in results:
        if res["model"] != "deepseek-flash":
            continue
        mine = {(f"c{s['parent_id'][1:]}", s["candidate_id"]) for s in res["steps"] if s["event"] == "candidate" and s["kept"] and s["dev"]}
        theirs = {(r["parent"], r["child"]) for r in table if r["bench"] == res["benchmark"]}
        total += len(mine)
        print("verify", res["benchmark"], "mine", len(mine), "edges_table", len(theirs), "equal edge sets:", mine == theirs)
    print("verify DeepSeek GEPA dev-scored non-seed total", total, "(mech_edges.md: 22 gepa edges, all accepted)")


if __name__ == "__main__":
    main()
