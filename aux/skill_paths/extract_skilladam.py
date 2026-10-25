#!/usr/bin/env python
"""Optimization-path extractor for SkillAdam (idempotent; read-only on the run artifacts).

Run:  python -m aux.skill_paths.extract_skilladam
Writes ONLY OUT/skill_paths/paths/<model>__skilladam__<benchmark>__<runid>.json (one object per run) and prints a summary.

How SkillAdam's loop works (verified in optimizers/skilladam/skilladam/core/feedback_loop.py and src/adapters/skilladam.py)
  * One iteration k (0-based; "step" below = k+1) = ONE training batch (10 tasks, split 'train') and ONE validation draw (min(10,|D_sel|) task
    ids drawn afresh from the dev panel D_sel with random.Random(42); on TB2 |D_sel| = 9 so the whole pool every time).
  * The incumbent ("current skill") is rolled out on the training batch; its trajectories feed the editor, which writes ONE unified-diff patch
    (up to patch_attempts = 3 tries; a try that is unparseable or applies no hunk is retried with the diagnostics).
  * candidate = apply_patch_with_fallback(incumbent, patch)  (upstream applier; a patch that never applies gives NO candidate: patch_mode
    'failed', candidate metric := baseline metric, rejected; an empty patch gives candidate == incumbent, patch_mode 'empty').
  * GATE (AcceptanceGate(improvements=(primary, min_gain 0.0))): rollout of the incumbent (baseline) and of the candidate on the SAME validation
    draw; accept iff mean(hard)[candidate] - mean(hard)[baseline] > 0 (strict; hard = success, unknown -> 0, but strict outcome mode aborts and
    re-runs an iteration with an unknown verdict, so none enters the gate). Accepted -> candidate becomes the incumbent. No dev score of the
    candidate on the whole dev panel is ever computed: the dev numbers of a step are on its 10-task (TB2: 9-task) validation draw.
  * The executor caches one rollout per (skill, task, split), so an incumbent's baseline on a task it already ran is the earlier episode.
  * Returned skill = the last accepted candidate (the seed if none was accepted).

Evidence used for every step (all read from the run directory, nothing guessed)
  checkpoint.json         upstream's own iteration records (accepted, patch_mode, patch_attempts, validation_case_ids, case_metrics)
  iterations/iter_NNN/patch_attempt_MM.json   the patch (MM = patch_attempts, the one applied) and the editor's reasoning
  seed.md                 initial skill (the hand-written seed or seed-minimal)
  round-evidence/*.json   the adapter's own records (step, role, skill text + version_id, evaluation_path)
  results/evaluations/{train,dev,test}/<vid12>-<panel12>/evaluation.json   episodes (success true/false/null, per-request input_tokens)
Candidate text = upstream applier(incumbent, patch); it must equal the skill text witnessed by the round-evidence 'validation_candidate' record
of that step, the baseline must equal the 'validation_baseline' record, and the rebuilt final incumbent must equal checkpoint current_skill,
final_skill.md, final/SKILL.md and the skill of the 'final_test' evaluation. Any disagreement is recorded in 'verification' (ok false).

Schema notes (extensions of the shared path schema, all additive)
  * candidate_id = 'step%02d' % step ('seed' for step 0); parent_id = the incumbent when the patch was proposed.
  * dev (candidate) = the candidate's rollout on the step's validation draw (n = 10, TB2 9); gate.detail.baseline = the incumbent on the same draw.
  * train = the INCUMBENT's (= the parent's) rollout on the step's training batch, the one the editor saw (not a candidate score).
  * steps without a candidate (patch_mode 'failed'): dev = null, skill_text = the parent's text (the optimizer itself treats the candidate as the
    baseline), added/removed = [], extra.candidate_exists = false.
  * extra (per step) and verification / extra (top level) are additions; seed_dev = the seed's rollout on the step-1 validation draw (SkillAdam
    never scores the seed on the whole dev panel).
"""
import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; never write .pyc next to read-only sources

import difflib
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from harness_bench.eval import version_from_files

from src.layout import ASSETS, CACHE, EXECUTION, OPTIMIZERS, OUT as OUT_ROOT, ROOT, SEED_FILE

EX = EXECUTION
OUT = OUT_ROOT / "skill_paths"
PATHS = OUT / "paths"
OPT = "skilladam"
SEED_PATH = SEED_FILE
SEED_MIN_PATH = ASSETS / "skills/seed-minimal/SKILL.md"
DIFF_APPLY = OPTIMIZERS / "skilladam/skilladam/core/diff_apply.py"  # the vendored upstream applier
EDGES_TABLE = CACHE / "analysis/main/edges_table.json"
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
RUNS = [("deepseek-flash", b, "native-20261008-https") for b in BENCHES] + [("gpt-6-luna", b, "native-20261012") for b in BENCHES]
# The edge builder (src/analysis/edges.py) drops this cell because it cannot tell the parent of an edit from the logs. Here the parent is
# known by construction and checked against the round-evidence, so parent_id is filled; set to {("deepseek-flash","terminal-bench-2-1")} to force null.
FORCE_NULL_PARENT = set()
PAPER_AMBIGUOUS = {("deepseek-flash", "terminal-bench-2-1")}


# ------------------------------------------------------------------------------------------------------------------------------------
def vid(text):
    """harness-bench version id of a one-file skill."""
    return version_from_files({"SKILL.md": text})["version_id"]


def iso(path):
    return datetime.fromtimestamp(os.stat(path).st_mtime, tz=timezone.utc).isoformat()


def load_applier():
    spec = importlib.util.spec_from_file_location("skilladam_diff_apply", DIFF_APPLY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skilladam_diff_apply"] = module  # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module.apply_patch_with_fallback


def patch_mode(report):  # skilladam.core.feedback_loop._patch_mode
    if report.hunks_total == 0:
        return "empty"
    if report.hunks_failed:
        return "partial"
    if report.hunks_intent:
        return "intent"
    return "strict"


def line_diff(parent, child):
    """(added, removed): difflib on the non-empty stripped lines of parent and child (insert/replace -> added, delete/replace -> removed)."""
    a = [x.strip() for x in parent.splitlines() if x.strip()]
    b = [x.strip() for x in child.splitlines() if x.strip()]
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += b[j1:j2]
        if op in ("delete", "replace"):
            removed += a[i1:i2]
    return added, removed


_EVAL_CACHE = {}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def summarize(evaluation, counters):
    """{n, n_success, n_unknown, mean_input_tokens_per_episode}: n = episodes, success True/False/None (None = unknown), tokens = sum of the
    input_tokens of the requests of an episode, averaged over the episodes that have requests."""
    episodes = evaluation["episodes"]
    tokens = []
    for episode in episodes:
        requests = episode.get("requests") or []
        if requests:
            tokens.append(sum(int(r.get("input_tokens") or 0) for r in requests))
        else:
            counters["episodes_without_requests"] += 1
        counters["episodes"] += 1
        if episode["success"] is None and episode.get("outcome_valid"):
            counters["unknown_but_outcome_valid"] += 1
        if episode["success"] is not None and episode.get("outcome_valid") is False:
            counters["known_but_outcome_invalid"] += 1
    return {"n": len(episodes), "n_success": sum(e["success"] is True for e in episodes), "n_unknown": sum(e["success"] is None for e in episodes),
            "mean_input_tokens_per_episode": (sum(tokens) / len(tokens)) if tokens else None}


class Run:
    def __init__(self, model, bench, run_id, apply_patch):
        self.model, self.bench, self.run_id, self.apply_patch = model, bench, run_id, apply_patch
        self.run = EX / model / OPT / bench / run_id
        self.sdir = self.run / OPT
        self.checks, self.anomalies, self.counters = [], [], Counter()

    # -- bookkeeping --------------------------------------------------------------------------------------------------------------
    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.anomalies.append(f"{self.model}/{self.bench}: CHECK FAILED {name}: {detail}")

    def anomaly(self, text):
        self.anomalies.append(f"{self.model}/{self.bench}: {text}")

    # -- evidence loaders ---------------------------------------------------------------------------------------------------------
    def load_round_evidence(self):
        records = defaultdict(list)
        states = {}
        for path in sorted((self.run / "round-evidence").glob("*.json")):
            record = read_json(path)
            record["_file"] = path.name
            if record["kind"] == "round_evaluation" and record.get("arm") == OPT:
                records[(int(record["step"]), record["role"])].append(record)
            elif record["kind"] == "round_state" and record.get("arm") == OPT:
                states[int(record["step"])] = record
        for key, group in records.items():
            group.sort(key=lambda r: r["recorded_utc"])
        self.evidence, self.states = records, states

    def evidence_for(self, step, role):
        """The last round-evidence record of (step, role) or None; duplicates (resumed iterations) must agree on skill and evaluation."""
        group = self.evidence.get((step, role))
        if not group:
            return None
        if len({(r["skill"]["text"], r["evaluation_path"]) for r in group}) > 1:
            self.anomaly(f"step {step} {role}: {len(group)} round-evidence records disagree (skill/evaluation); the last one is used")
        elif len(group) > 1:
            self.counters["duplicate_round_evidence_agreeing"] += 1
        return group[-1]

    def evaluation_of(self, record):
        """(evaluation dict, relative path); the path is rebuilt inside the run directory from the last three components of the recorded
        evaluation_path."""
        parts = Path(record["evaluation_path"]).parts[-4:]  # <split>/<vid12>-<panel12>/evaluation.json preceded by 'evaluations'
        assert parts[0] == "evaluations" and parts[-1] == "evaluation.json", parts
        rel = Path("results/evaluations") / parts[1] / parts[2] / parts[3]
        path = self.run / rel
        if path not in _EVAL_CACHE:
            _EVAL_CACHE[path] = read_json(path)
        evaluation = _EVAL_CACHE[path]
        if evaluation["version_id"] != record["skill"]["version_id"]:
            self.anomaly(f"{rel}: version_id differs from the round-evidence skill version")
        return evaluation, str(rel)

    # -- main ---------------------------------------------------------------------------------------------------------------------
    def extract(self):
        run, sdir = self.run, self.sdir
        ckpt = read_json(sdir / "checkpoint.json")
        summary = read_json(run / "summary.json")
        seed = (sdir / "seed.md").read_text()
        seed_kind = "hand-written" if seed == SEED_PATH.read_text() else "near-empty" if seed == SEED_MIN_PATH.read_text() else "unknown"
        self.check("seed.md equals the paper seed (skills/seed/SKILL.md) or seed-minimal", seed_kind != "unknown", f"seed kind {seed_kind}")
        self.load_round_evidence()
        T = summary["T"]
        iterations = ckpt["iterations"]
        self.check("checkpoint holds T completed iterations", len(iterations) == T == ckpt["next_iteration"] == len(summary["extra"]["iterations"]),
                   f"T={T} checkpoint={len(iterations)} next_iteration={ckpt['next_iteration']} summary={len(summary['extra']['iterations'])}")
        self.check("terminal_stop_reason empty and no pending post-decision", ckpt["terminal_stop_reason"] == "" and ckpt["pending_post_decision"] is None)
        incomplete = (run / "skilladam-incomplete.json").exists()
        self.check("no skilladam-incomplete.json (run not aborted mid-iteration)", not incomplete)
        partial = bool(incomplete or len(iterations) != T or not (run / "summary.json").exists() or ckpt["next_iteration"] != T)
        sel = summary["panels"]["sel"]
        union = set()
        for r in iterations:
            union |= set(r["validation_case_ids"])
        self.check("every validation draw is a subset of the dev panel (union of draws <= |D_sel|)", len(union) <= sel, f"|union|={len(union)} |D_sel|={sel}")

        steps = []
        incumbent_text, incumbent_id = seed, "seed"
        # seed step: the seed's rollout on the first validation draw (its first gate baseline)
        first_baseline = self.evidence_for(1, "validation_baseline")
        seed_dev = None
        if first_baseline:
            self.check("step-1 validation baseline is the seed", first_baseline["skill"]["text"] == seed)
            seed_dev = summarize(self.evaluation_of(first_baseline)[0], self.counters)
        else:
            self.anomaly("no validation_baseline round-evidence at step 1: seed_dev null")
        steps.append({"step": 0, "event": "seed", "candidate_id": "seed", "parent_id": None, "kept": None, "dev": seed_dev, "gate": None, "train": None,
                      "skill_chars": len(seed), "skill_words": len(seed.split()), "skill_text": seed, "added_lines": [], "removed_lines": [],
                      "extra": {"version_id": vid(seed),
                                "dev_note": "seed's rollout on the step-1 validation draw (n = draw size); SkillAdam never scores the seed on the whole dev panel"}})

        n_checked_candidates = n_checked_baselines = n_checked_states = 0
        for r in iterations:
            k = r["iteration"]
            step = k + 1
            cid = f"step{step:02d}"
            att = r["patch_attempts"]
            attempt_path = sdir / "iterations" / f"iter_{k:03d}" / f"patch_attempt_{att:02d}.json"
            attempt = read_json(attempt_path)
            patch = attempt["patch"]
            earlier = []
            for a in range(1, att):
                earlier_path = sdir / "iterations" / f"iter_{k:03d}" / f"patch_attempt_{a:02d}.json"
                earlier.append({"attempt": a, "patch_chars": len(read_json(earlier_path)["patch"]), "unparseable": read_json(earlier_path)["patch"] == "(unparseable)"})
            report = self.apply_patch(incumbent_text, patch)
            malformed = bool(patch.strip()) and report.hunks_total == 0
            would_fail = malformed or report.all_failed
            self.check(f"step {step}: patch_mode failed <=> upstream applier fails the last-attempt patch", (r["patch_mode"] == "failed") == would_fail,
                       f"checkpoint {r['patch_mode']}, applier hunks {report.summary()}")
            if r["patch_mode"] != "failed":
                self.check(f"step {step}: patch_mode reproduced by the applier", patch_mode(report) == r["patch_mode"], f"{patch_mode(report)} vs {r['patch_mode']}")
            candidate_text = None if r["patch_mode"] == "failed" else report.text
            kept = bool(r["accepted"])
            b_rec, c_rec, t_rec = (self.evidence_for(step, "validation_baseline"), self.evidence_for(step, "validation_candidate"), self.evidence_for(step, "training_rollout"))

            # incumbent / candidate identity against the adapter's witnessed texts
            if b_rec:
                n_checked_baselines += 1
                self.check(f"step {step}: validation_baseline witnessed skill == rebuilt incumbent", b_rec["skill"]["text"] == incumbent_text)
            else:
                self.anomaly(f"step {step}: no validation_baseline round-evidence")
            if t_rec:
                self.check(f"step {step}: training_rollout witnessed skill == rebuilt incumbent", t_rec["skill"]["text"] == incumbent_text)
            if candidate_text is not None:
                if c_rec:
                    n_checked_candidates += 1
                    self.check(f"step {step}: validation_candidate witnessed skill == incumbent + patch (upstream applier)", c_rec["skill"]["text"] == candidate_text)
                else:
                    self.anomaly(f"step {step}: candidate exists (patch_mode {r['patch_mode']}) but no validation_candidate round-evidence: dev null")
            else:
                self.check(f"step {step}: no validation_candidate round-evidence when the patch failed", c_rec is None)
            state = self.states.get(step)
            if state:
                n_checked_states += 1
                got_candidate = (state["candidate"] or {}).get("text")
                after_text = candidate_text if kept else incumbent_text
                self.check(f"step {step}: round_state input/candidate/returned texts match the rebuild",
                           state["input"]["text"] == incumbent_text and got_candidate == candidate_text and state["returned"]["text"] == after_text
                           and state["decision"] == ("accept" if kept else "no_proposal" if candidate_text is None else "reject"),
                           f"decision {state['decision']}")

            # scores
            base_eval = cand_eval = train_eval = None
            if b_rec:
                base_eval, base_path = self.evaluation_of(b_rec)
            if c_rec:
                cand_eval, cand_path = self.evaluation_of(c_rec)
            if t_rec:
                train_eval, train_path = self.evaluation_of(t_rec)
            base_sum = summarize(base_eval, self.counters) if base_eval else None
            cand_sum = summarize(cand_eval, self.counters) if cand_eval else None
            train_sum = summarize(train_eval, self.counters) if train_eval else None
            draw = list(r["validation_case_ids"])
            if base_eval:
                self.check(f"step {step}: baseline evaluation panel == checkpoint validation_case_ids", {e["task_id"] for e in base_eval["episodes"]} == set(draw) and len(base_eval["episodes"]) == len(draw))
            if cand_eval:
                self.check(f"step {step}: candidate evaluation panel == checkpoint validation_case_ids", {e["task_id"] for e in cand_eval["episodes"]} == set(draw) and len(cand_eval["episodes"]) == len(draw))
            if t_rec and train_eval:
                self.check(f"step {step}: training evaluation panel == checkpoint training_case_ids", {e["task_id"] for e in train_eval["episodes"]} == set(r["training_case_ids"]))

            # the gate as upstream computed it (checkpoint) against the episodes
            gb, gc = r["baseline"], r["candidate"]
            n_val = len(gb["case_metrics"])
            gain_tasks = round((gc["primary"] - gb["primary"]) * n_val)
            rule_accepts = (gc["primary"] - gb["primary"]) > 0.0
            self.check(f"step {step}: accepted == (candidate primary - baseline primary > 0)", rule_accepts == kept, f"{gb['primary']:.3f} -> {gc['primary']:.3f}")
            if base_eval:
                from_ep = {e["task_id"]: float(e["success"] is True) for e in base_eval["episodes"]}
                self.check(f"step {step}: checkpoint baseline case_metrics == baseline episodes", all(abs(gb["case_metrics"][t]["hard"] - from_ep[t]) < 1e-9 for t in gb["case_metrics"]))
            if cand_eval:
                from_ep = {e["task_id"]: float(e["success"] is True) for e in cand_eval["episodes"]}
                self.check(f"step {step}: checkpoint candidate case_metrics == candidate episodes", all(abs(gc["case_metrics"][t]["hard"] - from_ep[t]) < 1e-9 for t in gc["case_metrics"]))
            for label, s in (("baseline", base_sum), ("candidate", cand_sum)):
                if s and s["n_unknown"]:
                    self.anomaly(f"step {step}: {label} evaluation holds {s['n_unknown']} unknown outcome(s) although strict mode should have aborted")
            if candidate_text is None:
                self.check(f"step {step}: failed patch -> candidate metric equals baseline metric", gb["case_metrics"] == gc["case_metrics"])

            parent_id = None if (self.model, self.bench) in FORCE_NULL_PARENT else incumbent_id
            added, removed = line_diff(incumbent_text, candidate_text) if candidate_text is not None else ([], [])
            text = candidate_text if candidate_text is not None else incumbent_text
            detail = {
                "comparison": "fresh validation draw: incumbent (baseline) and candidate rolled out on the same draw",
                "accept_rule": "accept iff mean hard(candidate) - mean hard(baseline) > 0 (AcceptanceGate improvements=(primary, min_gain 0.0), strict)",
                "reason": r["reason"], "accepted": kept,
                "validation_case_ids": draw,
                "baseline": {"skill": "parent (incumbent)", "primary": gb["primary"], "n_success": round(gb["primary"] * n_val), "n": n_val,
                             "dev": base_sum, "evaluation": base_path if base_eval else None},
                "candidate": {"skill": "this step's candidate" if candidate_text is not None else "none: patch failed, candidate metric := baseline metric",
                              "primary": gc["primary"], "n_success": round(gc["primary"] * n_val), "n": n_val, "dev": cand_sum,
                              "evaluation": cand_path if cand_eval else None},
                "gain_tasks": gain_tasks, "gain_points": 100.0 * (gc["primary"] - gb["primary"]),
                "patch_mode": r["patch_mode"], "patch_attempts": att, "edit_budget": r["edit_budget"], "sigma_sq": r["sigma_sq"], "v_ema_after": r["v_ema_after"],
            }
            extra = {
                "iteration": k, "version_id": vid(text),
                "candidate_exists": candidate_text is not None,
                "identical_to_parent": text == incumbent_text,
                "incumbent_after": cid if kept else incumbent_id,
                "train_rollout_of": "parent (incumbent) skill on this step's training batch (what the editor saw), not a candidate score",
                "train_evaluation": train_path if t_rec else None,
                "patch": patch, "patch_reasoning": attempt.get("reasoning"),
                "applier_hunks": {"total": report.hunks_total, "strict": report.hunks_strict, "intent": report.hunks_intent, "failed": report.hunks_failed},
                "earlier_attempts": earlier,
                "parent_resolution": "incumbent by construction (FeedbackLoop applies each patch to state.current_skill); text equal to the round-evidence validation_baseline/candidate witnessed skills",
                "paper_edge_parent_ambiguous": (self.model, self.bench) in PAPER_AMBIGUOUS,
            }
            if (self.model, self.bench) in FORCE_NULL_PARENT:
                extra["parent_resolution"] = "forced null (FORCE_NULL_PARENT)"
            steps.append({"step": step, "event": "candidate", "candidate_id": cid, "parent_id": parent_id, "kept": kept,
                          "dev": cand_sum, "gate": {"kind": "skilladam-fresh-validation-draw", "detail": detail},
                          "train": ({k_: train_sum[k_] for k_ in ("n", "n_success", "n_unknown")} if train_sum else None),
                          "skill_chars": len(text), "skill_words": len(text.split()), "skill_text": text,
                          "added_lines": added, "removed_lines": removed, "extra": extra})
            if kept:
                incumbent_text, incumbent_id = candidate_text, cid

        # final / returned skill
        final_files = {"checkpoint.current_skill": ckpt["current_skill"], "skilladam/current_skill.md": (sdir / "current_skill.md").read_text(),
                       "skilladam/final_skill.md": (sdir / "final_skill.md").read_text(), "final/SKILL.md": (run / "final/SKILL.md").read_text()}
        for name, content in final_files.items():
            self.check(f"rebuilt final incumbent == {name}", content == incumbent_text)
        returned_id = incumbent_id
        final_test = seed_test = None
        rec = self.evidence_for_global("final_test")
        if rec:
            ev, rel = self.evaluation_of(rec)
            self.check("final_test evaluation version_id == version id of the returned skill", ev["version_id"] == vid(incumbent_text) and rec["skill"]["text"] == incumbent_text)
            s = summarize(ev, self.counters)
            final_test = {"n_episodes": s["n"], "n_success": s["n_success"], "n_unknown": s["n_unknown"], "mean_input_tokens_per_episode": s["mean_input_tokens_per_episode"],
                          "source": f"{rel} (round-evidence role final_test; its version_id equals the returned skill's)"}
            tf = summary.get("test", {}).get("final", {})
            self.check("final_test successes/known == summary.json test.final", abs(tf.get("success", -1) - (s["n_success"] / max(s["n"] - s["n_unknown"], 1))) < 1e-9 and tf.get("known") == s["n"] - s["n_unknown"],
                       f"episodes {s['n_success']}/{s['n']} unknown {s['n_unknown']} vs summary {tf}")
        else:
            self.anomaly("no final_test round-evidence")
        rec = self.evidence_for_global("seed_test")
        if rec:
            ev, rel = self.evaluation_of(rec)
            s = summarize(ev, self.counters)
            self.check("seed_test evaluation is of the seed", ev["version_id"] == vid(seed))
            seed_test = {"n_episodes": s["n"], "n_success": s["n_success"], "n_unknown": s["n_unknown"], "source": rel}
        self.check("summary.json final_words == words of the returned skill; seed_words == words of the seed", summary["final_words"] == len(incumbent_text.split()) and summary["seed_words"] == len(seed.split()),
                   f"summary {summary['final_words']}/{summary['seed_words']} computed {len(incumbent_text.split())}/{len(seed.split())}")
        self.check("summary.json extra.accepted == accepted steps", summary["extra"]["accepted"] == sum(s["kept"] is True for s in steps))

        # mtimes (read-only evidence that the run was finished when read)
        newest = max((p.stat().st_mtime for p in (sdir / "iterations").glob("iter_*/*")), default=0)
        mtimes = {name: iso(path) for name, path in [("summary.json", run / "summary.json"), ("skilladam/checkpoint.json", sdir / "checkpoint.json"),
                                                     ("skilladam/final_skill.md", sdir / "final_skill.md"), ("skilladam/episodes.json", sdir / "episodes.json"),
                                                     ("skilladam/usage.jsonl", sdir / "usage.jsonl"), ("final/SKILL.md", run / "final/SKILL.md"),
                                                     ("launch.log", run / "launch.log"), ("completion_audit.json", run / "completion_audit.json"),
                                                     ("round-evidence/", run / "round-evidence")] if path.exists()}
        mtimes["skilladam/iterations/*/* (newest)"] = datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
        for name in ("skilladam-incomplete.json", ".launch.lock"):
            mtimes[name] = iso(run / name) if (run / name).exists() else None

        # context for the notes
        stale = sorted(p.name for p in (sdir / "iterations-stale").glob("*")) if (sdir / "iterations-stale").is_dir() else []
        stale_info = {}
        for name in stale:  # an aborted attempt of an iteration (unknown verdict in strict mode) moved aside by the adapter on resume
            done = sdir / "iterations" / name.split("-")[0]
            files = sorted(p.name for p in (sdir / "iterations-stale" / name).glob("*"))
            first = sdir / "iterations-stale" / name / "patch_attempt_01.json"
            stale_info[name] = {"files": files, "has_candidate_evaluation": "validation_candidate_evaluation.json" in files,
                                "patch_attempt_01_equals_completed_iteration": (first.read_bytes() == (done / "patch_attempt_01.json").read_bytes()) if first.exists() and (done / "patch_attempt_01.json").exists() else None}
        dup_steps = sorted([step, role, len(g)] for (step, role), g in self.evidence.items() if len(g) > 1)
        resumes = sum(1 for _ in open(run / "native-resumes.jsonl")) if (run / "native-resumes.jsonl").exists() else 0
        retry_dev = sum(1 for _ in open(run / "luna-retry-deviation.jsonl")) if (run / "luna-retry-deviation.jsonl").exists() else 0
        reply_retries = sum(1 for _ in open(run / "skilladam-reply-retries.jsonl")) if (run / "skilladam-reply-retries.jsonl").exists() else 0
        usage = defaultdict(lambda: Counter())
        for line in open(sdir / "usage.jsonl"):
            u = json.loads(line)
            usage[u["stage"]]["calls"] += 1
            usage[u["stage"]]["input_tokens"] += u.get("input_tokens") or 0
            usage[u["stage"]]["output_tokens"] += u.get("output_tokens") or 0
        cands = [s for s in steps if s["event"] == "candidate"]
        dev_scored = [s for s in cands if s["dev"] is not None]
        modes = Counter(s["gate"]["detail"]["patch_mode"] for s in cands)
        notes = (
            f"SkillAdam {self.model} / {self.bench}, run {self.run_id}. Complete run (summary.json present, {len(iterations)} of T={T} iterations in checkpoint.json, no skilladam-incomplete.json); partial = {partial}. "
            f"One step = one iteration = one patch -> one candidate (patch_attempts is the retry count for an unparseable or non-applying patch, not a number of candidates; the patch of the last attempt is applied). "
            f"The gate compares the incumbent and the candidate on one FRESH validation draw of {len(draws_of(iterations)[0])} dev tasks per step (|D_sel| = {sel}; TB2: the whole 9-task pool every time), strict gain > 0; "
            f"dev numbers of a step are on that draw only, never on the whole dev panel. Kept steps: {sum(s['kept'] is True for s in cands)} of {len(cands)}; patch modes {dict(modes)}; "
            f"dev-scored candidates {len(dev_scored)} ({sum(1 for s in dev_scored if s['extra']['identical_to_parent'])} with text identical to the parent: empty patch, served by the executor's cache); "
            f"steps without a candidate (patch failed, no dev score): {[s['step'] for s in cands if not s['extra']['candidate_exists']]}. "
            f"train = the incumbent's rollout on the step's training batch (the editor's input), not a candidate score. seed_dev = seed on the step-1 draw. "
            f"Candidate text = upstream applier(incumbent, patch), equal to the text the adapter witnessed in round-evidence (checked: {n_checked_candidates} candidates, {n_checked_baselines} baselines, {n_checked_states} round_states); parents are the incumbent by construction. "
            f"{'The paper edge builder cannot resolve the parents of this cell (all draws share one 9-task panel); here they are resolved by the round-evidence step records and the incumbent chain. ' if (self.model, self.bench) in PAPER_AMBIGUOUS else ''}"
            f"Aux: reply retries (tracker) {reply_retries}; resumes {resumes}; luna unknown-verdict extra-retry grants {retry_dev}; stale aborted iteration dirs {stale_info} (an aborted attempt is not a step of the path: its proposal was never gate-scored; the completed attempt of the iteration is what this file records); "
            f"steps with repeated (agreeing) round-evidence after resumes {dup_steps}; counters {dict(self.counters)}; "
            f"optimizer LLM usage {json.dumps({k_: dict(v) for k_, v in usage.items()})}."
        )
        result = {"model": self.model, "optimizer": OPT, "benchmark": self.bench, "run_id": self.run_id, "seed": seed_kind, "partial": partial,
                  "mtimes_seen": mtimes, "dev_panel_size": sel, "seed_dev": seed_dev, "steps": steps, "returned_skill_id": returned_id,
                  "final_test": final_test, "notes": notes,
                  "extra": {"seed_test": seed_test, "draw_size": len(draws_of(iterations)[0]), "iterations": T, "accepted_steps": [s["step"] for s in cands if s["kept"]],
                            "optimizer_usage": {k_: dict(v) for k_, v in usage.items()}, "stale_iteration_dirs": stale, "stale_iteration_info": stale_info,
                            "round_evidence_duplicates_[step,role,count]": dup_steps, "native_resumes": resumes, "reply_retries": reply_retries,
                            "luna_retry_deviation_grants": retry_dev, "counters": dict(self.counters)},
                  "verification": {"all_ok": all(c["ok"] for c in self.checks), "n_checks": len(self.checks), "n_failed": sum(not c["ok"] for c in self.checks),
                                   "failed": [c for c in self.checks if not c["ok"]], "applier": str(DIFF_APPLY.relative_to(ROOT))}}
        return result

    def evidence_for_global(self, role):
        """Records of role final_test / seed_test (their step is the iteration count; take the one record of that role)."""
        found = [g[-1] for (step, r), g in self.evidence.items() if r == role]
        if len(found) != 1:
            self.anomaly(f"{len(found)} round-evidence records with role {role}")
            return found[-1] if found else None
        return found[0]


def draws_of(iterations):
    return [r["validation_case_ids"] for r in iterations]


# ------------------------------------------------------------------------------------------------------------------------------------
def edges_crosscheck(result, table):
    """Compare the dev-scored candidates (text != parent) with the paper's dev edge table (DeepSeek only): multiset of (len_parent, len_child, accepted)."""
    steps = {s["candidate_id"]: s for s in result["steps"]}
    mine = Counter()
    for s in result["steps"]:
        if s["event"] != "candidate" or s["dev"] is None or s["extra"]["identical_to_parent"]:
            continue
        parent_text = steps[s["parent_id"]]["skill_text"]
        mine[(len(parent_text), len(s["skill_text"]), bool(s["kept"]))] += 1
    rows = [r for r in table if r["opt"] == OPT and r["bench"] == result["benchmark"] and r["split"] == "dev"]
    seen = set()
    theirs = Counter()
    for r in rows:
        key = (r["parent"], r["child"])
        if key in seen:
            continue
        seen.add(key)
        theirs[(r["len_p"], r["len_c"], bool(r["accepted"]))] += 1
    return {"paper_table_dev_edges": sum(theirs.values()), "paper_table_dev_records": len(rows), "mine_dev_scored_text_differs": sum(mine.values()),
            "matched": sum((mine & theirs).values()), "only_mine": [list(k) for k in (mine - theirs)], "only_paper_table": [list(k) for k in (theirs - mine)]}


def main():
    PATHS.mkdir(parents=True, exist_ok=True)
    apply_patch = load_applier()
    table = json.loads(EDGES_TABLE.read_text()) if EDGES_TABLE.exists() else []
    all_anomalies, summary_rows = [], []
    for model, bench, run_id in RUNS:
        runner = Run(model, bench, run_id, apply_patch)
        result = runner.extract()
        if model == "deepseek-flash":
            cross = edges_crosscheck(result, table)
            result["verification"]["edges_table_crosscheck"] = cross
        name = f"{model}__{OPT}__{bench}__{run_id}.json"
        tmp = PATHS / (name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        os.replace(tmp, PATHS / name)
        cands = [s for s in result["steps"] if s["event"] == "candidate"]
        summary_rows.append((name, len(result["steps"]), sum(s["dev"] is not None for s in cands), result["partial"], sum(s["kept"] is True for s in cands),
                             sum(1 for s in cands if s["dev"] is not None and not s["extra"]["identical_to_parent"]), result["verification"]))
        all_anomalies += runner.anomalies
    for name, n_steps, n_dev, partial, kept, n_diff, ver in summary_rows:
        extra = ver.get("edges_table_crosscheck")
        print(f"{name}: steps={n_steps} (incl. seed) dev_scored_candidates={n_dev} (text differs from parent: {n_diff}) kept={kept} partial={partial} "
              f"checks={ver['n_checks']} failed={ver['n_failed']}" + (f" | paper edge table: {extra['paper_table_dev_edges']} edges, matched {extra['matched']}, only_mine {extra['only_mine']}, only_paper {extra['only_paper_table']}" if extra else ""))
    print("ANOMALIES:")
    for a in all_anomalies:
        print("  -", a)


if __name__ == "__main__":
    main()
