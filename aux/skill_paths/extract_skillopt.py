#!/usr/bin/env python
"""Optimization path of every SkillOpt run -> paths/<model>__skillopt__<benchmark>__<run-id>.json.

Idempotent: reads the run artifacts (read-only), writes only into OUT/skill_paths/paths/. Run with
    python -m aux.skill_paths.extract_skillopt [--only <substring>] [--print-checks]

What a SkillOpt run is (verified in src/adapters/skillopt.py and optimizers/skillopt/skillopt/engine/trainer.py, evaluation/gate.py):
  * one epoch; step t takes the CURRENT accepted skill, rolls it out on training batch B_t (the shared batches of native_protocol.json),
    reflects on the trajectories, merges the patches, ranks <= edit_budget edits, applies them to the current skill -> candidate_skill.md;
  * the candidate is scored once on the selection panel ("sel" = the dev panel; adapter.build_eval_batch maps valid_seen -> panels["sel"]);
  * GATE (evaluation/gate.py evaluate_gate, metric "hard" = verifier success, unknown outcome counts as failure in the adapter, but strict
    runs abort the whole panel on an unknown dev/train outcome before scoring): accept iff cand_score > current_score (STRICT >), where
    current_score is the STORED dev score of the current skill from its own earlier evaluation (it is not re-scored on a fresh draw);
    action accept_new_best iff additionally cand_score > best_score, "accept" if only > current, else "reject" (current unchanged);
  * a candidate whose text hash was scored before is served from the trainer's sel_cache (no new rollout); the executor also caches
    episodes by (skill text, task, split);
  * at the end of the epoch the slow-update step (epoch 1) injects an EMPTY placeholder block into the current skill (no dev gate, force-accept),
    then one dev evaluation of that final skill ("final_selection_eval") may PROMOTE it over best (promotion only if its score > best_score);
  * the returned skill is best_skill.md (adapter.run: final_skill = root/best_skill.md), test-evaluated once ("test_eval").
The extractor re-derives every gate decision and cross-checks history.json against evaluation.json, candidate files, skills/ and launch.log;
mismatches are returned in "checks"/"anomalies" of each file (never silently fixed).
"""
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from harness_bench.eval import version_from_files

from src.layout import ASSETS, CACHE, EXECUTION, OUT as OUT_ROOT, SEED_FILE

EX = EXECUTION
OUT = OUT_ROOT / "skill_paths/paths"
SEED_HAND = SEED_FILE
SEED_MIN = ASSETS / "skills/seed-minimal/SKILL.md"
OPT = "skillopt"
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
RUNS = ([("deepseek-flash", b, "native-20261008-https") for b in BENCHES]
        + [("gpt-6-luna", b, "native-20261012") for b in BENCHES]
        + [("gpt-6-luna", "automationbench", "fromscratch")])
ACCEPT = {"accept", "accept_new_best", "force_accept"}
RECENT_S = 30 * 60  # a file younger than this at extraction time marks a run as possibly still running


def vid(text):
    """harness-bench version id of a one-file skill."""
    return version_from_files({"SKILL.md": text})["version_id"]


def thash(text):
    """SkillOpt's own skill_hash (utils/scoring.py): sha256(content)[:16]."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def read(path):
    return Path(path).read_text(encoding="utf-8")


def words(text):
    return len(text.split())


def line_diff(parent, child):
    """(added, removed) lines of child against parent: difflib over non-empty stripped lines."""
    a = [x.strip() for x in parent.splitlines() if x.strip()]
    b = [x.strip() for x in child.splitlines() if x.strip()]
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += b[j1:j2]
        if op in ("delete", "replace"):
            removed += a[i1:i2]
    return added, removed


class Run:
    def __init__(self, model, bench, run_id):
        self.model, self.bench, self.run_id = model, bench, run_id
        self.root = EX / model / OPT / bench / run_id
        self.so = self.root / OPT
        self.anomalies, self.checks, self.mtimes = [], [], {}
        self._eval_cache = {}

    # ---- bookkeeping -------------------------------------------------------------------------------------------------------------
    def anomaly(self, msg):
        self.anomalies.append(msg)

    def check(self, ok, msg):
        self.checks.append(("OK  " if ok else "FAIL") + " " + msg)
        if not ok:
            self.anomaly("CHECK FAILED: " + msg)
        return ok

    def seen(self, path, key=None):
        """Record the mtime of a file or directory read (relative to the run root)."""
        path = Path(path)
        try:
            self.mtimes[key or str(path.relative_to(self.root))] = iso(path.stat().st_mtime)
        except OSError:
            pass

    # ---- evaluations -------------------------------------------------------------------------------------------------------------
    def eval_dirs(self, split):
        """{vid12: [(panel12, evaluation.json path)]} of results/evaluations/<split>/."""
        out = {}
        base = self.root / "results/evaluations" / split
        self.seen(base)
        with os.scandir(base) as it:
            for entry in sorted(it, key=lambda e: e.name):
                v, _, panel = entry.name.partition("-")
                path = Path(entry.path) / "evaluation.json"
                if path.exists():
                    out.setdefault(v, []).append((panel, path))
        return out

    def load_eval(self, path):
        """Compact view of one evaluation.json: counts, per-episode input tokens, task-id set."""
        path = str(path)
        if path in self._eval_cache:
            return self._eval_cache[path]
        d = json.loads(read(path))
        eps = d["episodes"]
        toks = [sum(r.get("input_tokens", 0) or 0 for r in e.get("requests", [])) for e in eps]
        view = {
            "path": path, "version_id": d["version_id"], "split": d["split"],
            "n": len(eps), "n_success": sum(1 for e in eps if e["success"] is True),
            "n_failure": sum(1 for e in eps if e["success"] is False),
            "n_unknown": sum(1 for e in eps if e["success"] is None),
            "mean_input_tokens_per_episode": (sum(toks) / len(toks)) if toks else None,
            "tasks": frozenset(e["task_id"] for e in eps),
            "repetitions": sorted({e.get("repetition", 0) for e in eps}),
            "panel_size": len(d["panel"]),
            "slots": {k: d.get("slot_accounting", {}).get(k) for k in ("slots", "executed", "cache_hits", "unknown_outcomes", "backend_failures")},
        }
        self._eval_cache[path] = view
        self.seen(path)
        return view

    def pick(self, index, version, task_set=None, what=""):
        """The evaluation of `version` (vid12) in an eval_dirs index; with task_set, the one whose tasks equal it."""
        cands = index.get(version, [])
        if task_set is not None:
            cands = [(p, f) for p, f in cands if self.load_eval(f)["tasks"] == task_set]
        if not cands:
            return None
        if len(cands) > 1:
            self.anomaly(f"{what}: {len(cands)} evaluations of {version} match; the first is used")
        return self.load_eval(cands[0][1])


def counts(view):
    return None if view is None else {"n": view["n"], "n_success": view["n_success"], "n_unknown": view["n_unknown"],
                                     "mean_input_tokens_per_episode": view["mean_input_tokens_per_episode"]}


def parse_launch_log(run):
    """launch.log is the optimizer's stdout over all (re)starts ('---- skillopt <bench> <time>' header per start).
    Per step: how often the log started it and its last decision line. Events: every traceback (step, stage 'k/6 ...', exception line) with
    fatal=True when the session ended right after it (strict policy: an unknown dev/train outcome raises NativeIncomplete and the trainer exits)
    and fatal=False for handled ones (an analyst/LLM call that failed after its retries: reflect.py drops that minibatch, the step goes on);
    plus sessions that simply ended mid-step (killed) without a traceback."""
    path = run.root / "launch.log"
    info = {"baseline": None, "steps": {}, "events": [], "sessions": 0, "final": [], "exists": path.exists(), "header": None}
    if not path.exists():
        return info
    run.seen(path)
    lines = read(path).splitlines()
    cur, stage, in_tb, session = None, None, False, 0
    last_step_state = None  # (step, stage) while a step is in flight in this session

    def end_session():
        nonlocal last_step_state
        if last_step_state is not None and not (info["events"] and info["events"][-1].get("session") == session and info["events"][-1]["fatal"]):
            info["events"].append({"session": session, "step": last_step_state[0], "stage": last_step_state[1], "exception": None, "fatal": True,
                                   "kind": "session ended mid-step without a traceback (process killed or interrupted)"})
        last_step_state = None

    for i, line in enumerate(lines):
        if line.startswith("---- "):
            info.setdefault("header", line.strip()[:240])
            end_session()
            session += 1
            info["sessions"] = session
            cur, stage, in_tb = None, None, False
            continue
        if in_tb:
            if line.startswith(" ") or not line.strip():
                continue
            in_tb = False
            nxt = next((x for x in lines[i + 1:] if x.strip()), None)
            fatal = nxt is None or nxt.startswith("---- ")
            ev = {"session": session, "step": tb["step"], "stage": tb["stage"], "exception": line.strip()[:200], "fatal": fatal,
                  "kind": "aborted: unknown outcome on a strict panel" if "NativeIncomplete" in line or "UnknownOutcome" in line else
                          ("fatal" if fatal else "handled: LLM call failed after retries, analyst minibatch dropped")}
            info["events"].append(ev)
            if fatal:
                last_step_state = None
            continue
        if line.startswith("Traceback"):
            in_tb = True
            tb = {"step": cur, "stage": stage}
            continue
        m = re.search(r"\[STEP (\d+)/(\d+)\]", line)
        if m and "done" not in line:
            cur, stage = int(m[1]), "start"
            last_step_state = (cur, stage)
            st = info["steps"].setdefault(cur, {"starts": 0, "decision": None, "cache_hit": None, "handled_errors": 0})
            st["starts"] += 1
            st["decision"] = None
            st["cache_hit"] = None
            continue
        if re.search(r"\[STEP \d+ done\]", line):
            cur, stage, last_step_state = None, "between steps / end of epoch", None
            continue
        m = re.search(r"\[baseline result\] selection hard=([0-9.]+)", line)
        if m:
            info["baseline"] = float(m[1])
        m = re.search(r"\[(\d)/6 ([A-Z]+)", line)
        if m:
            stage = f"{m[1]}/6 {m[2]}"
            if cur is not None:
                last_step_state = (cur, stage)
        if cur is not None:
            st = info["steps"][cur]
            m = re.search(r"\[6/6 EVALUATE\] (ACCEPT \(new best\)|ACCEPT|REJECT) hard=([0-9.]+)", line)
            if m:
                st["decision"] = {"ACCEPT (new best)": "accept_new_best", "ACCEPT": "accept", "REJECT": "reject"}[m[1]]
                st["decision_hard"] = float(m[2])
            m = re.search(r"cache hit (\w+): hard=([0-9.]+)", line)
            if m:
                st["cache_hit"] = m[1]
        if re.search(r"\[final skill|\[promote\]", line):
            info["final"].append(line.strip())
    end_session()
    for e in info["events"]:
        if not e["fatal"] and e["step"] in info["steps"]:
            info["steps"][e["step"]]["handled_errors"] += 1
    return info


def active_processes(run):
    """True if a process args line mentions this run (read-only ps)."""
    try:
        txt = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    keys = [f"/{run.model}/{OPT}/{run.bench}/{run.run_id}", f"{run.bench}__{OPT}__{run.run_id}"]
    return any(k in line for line in txt.splitlines() for k in keys)


def extract(model, bench, run_id):
    run = Run(model, bench, run_id)
    root, so = run.root, run.so
    if not so.is_dir():
        raise SystemExit(f"missing run {root}")
    run.seen(root, "<run root>")
    run.seen(so, "skillopt/")
    for name in ("history.json", "runtime_state.json", "summary.json", "best_skill.md", "seed.md", "config.json"):
        run.seen(so / name)
    for name in ("summary.json", "launch.log", "native_protocol.json", "native-resumes.jsonl", "skillopt/unknown_outcome.json", "final/SKILL.md"):
        if (root / name).exists():
            run.seen(root / name)
    reps_dir = EX / model / "test-reps" / f"{bench}__{OPT}__{run_id}"  # replicate test evaluations of the returned skill (separate job, not optimizer state)
    if reps_dir.exists():
        run.seen(reps_dir, f"<test-reps dir {reps_dir.name}>")
    # newest direct child of the run root and of skillopt/ (bounded scandir, no recursion)
    newest = 0.0
    for d in (root, so):
        with os.scandir(d) as it:
            for e in it:
                newest = max(newest, e.stat().st_mtime)
    run.mtimes["<newest direct child of run root or skillopt/>"] = iso(newest)

    history = json.loads(read(so / "history.json"))
    summary = json.loads(read(so / "summary.json"))
    runsum = json.loads(read(root / "summary.json"))
    config = summary["config"]
    state = json.loads(read(so / "runtime_state.json"))
    proto = json.loads(read(root / "native_protocol.json"))
    seed_text = read(so / "seed.md")
    best_text = read(so / "best_skill.md")
    log = parse_launch_log(run)

    # ---- seed kind ---------------------------------------------------------------------------------------------------------------
    if seed_text == read(SEED_HAND):
        seed_kind = "hand-written"
    elif seed_text == read(SEED_MIN):
        seed_kind = "near-empty"
    else:
        seed_kind = "unknown"
        run.anomaly("skillopt/seed.md equals neither skills/seed/SKILL.md nor seed-minimal/SKILL.md")
    seed_v = vid(seed_text)[:12]

    # ---- panels and evaluations --------------------------------------------------------------------------------------------------
    dev_tasks = frozenset(proto["panels"]["sel"])
    dev_size = len(dev_tasks)
    batches = proto["batches"]
    dev_idx, train_idx, test_idx = run.eval_dirs("dev"), run.eval_dirs("train"), run.eval_dirs("test")
    run.check(config["num_epochs"] == 1 and config["steps_per_epoch"] * config["num_epochs"] == len(batches),
              f"config: num_epochs={config['num_epochs']}, steps_per_epoch={config['steps_per_epoch']}, native_protocol batches={len(batches)}")
    run.check(config["use_gate"] is True and config.get("gate_metric", "hard") == "hard",
              "config: use_gate true, gate_metric hard (default; not overridden)")
    if not config.get("strict_outcomes"):
        run.anomaly("config.strict_outcomes is not true: unknown outcomes would count as failures in the gate")

    seed_dev = run.pick(dev_idx, seed_v, dev_tasks, "seed dev")
    if seed_dev is None:
        run.anomaly("no dev evaluation of the seed found")

    # ---- baseline score the optimizer used (current_score before step 1) -------------------------------------------------------
    base_n = seed_dev["n_success"] if seed_dev else None
    baseline_sources = {}
    if seed_dev:
        baseline_sources["seed_dev_evaluation"] = seed_dev["n_success"] / seed_dev["n"]
    if summary.get("baseline_selection_hard") is not None:
        baseline_sources["summary.baseline_selection_hard"] = summary["baseline_selection_hard"]
    if log["baseline"] is not None:
        baseline_sources["launch.log [baseline result]"] = log["baseline"]
    if history and history[0]["action"] == "reject":
        baseline_sources["history step 1 current_score (rejected)"] = history[0]["current_score"]
    vals = {round(v * dev_size) for v in baseline_sources.values()}
    run.check(len(vals) <= 1, f"baseline dev score agrees across sources {baseline_sources} (tasks of {dev_size})")
    if len(baseline_sources) < 2:
        run.anomaly(f"the baseline dev score the optimizer started from has a single source ({list(baseline_sources)}); no log or summary value to confirm it")
    baseline_score = (base_n / seed_dev["n"]) if seed_dev else None
    if baseline_score is None and baseline_sources:
        baseline_score = next(iter(baseline_sources.values()))

    # ---- steps -------------------------------------------------------------------------------------------------------------------
    run_dev_size = dev_size
    steps = []
    seed_entry = {"step": 0, "event": "seed", "candidate_id": "seed", "parent_id": None, "kept": None, "dev": counts(seed_dev),
                  "dev_slots": seed_dev["slots"] if seed_dev else None, "gate": None, "train": None, "skill_chars": len(seed_text), "skill_words": words(seed_text), "skill_text": seed_text,
                  "added_lines": [], "removed_lines": [], "version_id": vid(seed_text)}
    steps.append(seed_entry)
    texts = {"seed": seed_text}
    current_id, best_id = "seed", "seed"
    current_score, best_score = baseline_score, baseline_score
    seen_hash = {thash(seed_text): "seed"}
    prev_best_for_log = None
    n_dev_scored = 0
    for h in history:
        n = h["step"]
        cid = f"s{n}"
        cand_path = so / f"steps/step_{n:04d}/candidate_skill.md"
        if not cand_path.exists():
            run.anomaly(f"step {n}: no candidate_skill.md (action {h['action']})")
            text = None
        else:
            text = read(cand_path)
            run.seen(cand_path)
        texts[cid] = text
        parent_text = texts[current_id]
        # lineage cross-check: skills/skill_v{n-1}.md is the current skill the trainer saved after step n-1
        if n >= 2:
            sk = so / f"skills/skill_v{n - 1:04d}.md"
            if sk.exists():
                run.check(read(sk) == parent_text, f"step {n}: skills/skill_v{n - 1:04d}.md equals the tracked parent text ({current_id})")
        else:
            run.check(parent_text == seed_text, "step 1: parent is the seed")
        origin_ok = h.get("current_origin") == (f"step_{n:04d}" if h["action"] in ACCEPT else (f"step_{int(current_id[1:]):04d}" if current_id != "seed" else "initial_skill"))
        run.check(origin_ok, f"step {n}: history current_origin '{h.get('current_origin')}' consistent with tracked lineage")

        dev_view = None
        if text is not None:
            dev_view = run.pick(dev_idx, vid(text)[:12], dev_tasks, f"step {n} dev")
        cache_hit = h["candidate_hash"] in seen_hash if h.get("candidate_hash") else None
        if dev_view is None and h.get("selection_hard") is not None:
            run.anomaly(f"step {n}: history has selection_hard but no dev evaluation.json found for the candidate text; dev left null")
        if dev_view is not None:
            run.check(abs(dev_view["n_success"] / dev_view["n"] - h["selection_hard"]) < 1e-9 and dev_view["n_unknown"] == 0,
                      f"step {n}: dev evaluation.json success {dev_view['n_success']}/{dev_view['n']} equals history selection_hard {h['selection_hard']:.4f}, no unknown")
            n_dev_scored += 1
        # train rollout of the CURRENT skill (parent) on batch B_n
        train = None
        batch_tasks = frozenset(batches[n - 1])
        tv = run.pick(train_idx, vid(parent_text)[:12], batch_tasks, f"step {n} train (parent {current_id})")
        if tv is not None:
            run.check(abs(tv["n_success"] / tv["n"] - h["rollout_hard"]) < 1e-5 and tv["n"] == h["rollout_n"],
                      f"step {n}: train evaluation of the parent {tv['n_success']}/{tv['n']} equals history rollout_hard {h['rollout_hard']}")
            train = {"n": tv["n"], "n_success": tv["n_success"], "n_unknown": tv["n_unknown"],
                     "evaluated_skill": "parent", "evaluated_skill_id": current_id, "batch": f"B_{n}",
                     "mean_input_tokens_per_episode": tv["mean_input_tokens_per_episode"]}
        else:
            run.anomaly(f"step {n}: no train evaluation.json of the parent on batch B_{n}; train taken from history (n_unknown assumed 0, strict policy)")
            train = {"n": h["rollout_n"], "n_success": round(h["rollout_hard"] * h["rollout_n"]), "n_unknown": 0,
                     "evaluated_skill": "parent", "evaluated_skill_id": current_id, "batch": f"B_{n}", "mean_input_tokens_per_episode": None}

        # gate
        cand_score = h.get("candidate_gate_score")
        cand_k = round(cand_score * dev_size) if cand_score is not None else None
        cur_k = round(current_score * dev_size) if current_score is not None else None
        best_k = round(best_score * dev_size) if best_score is not None else None
        if cand_k is not None and cur_k is not None:
            derived = ("accept_new_best" if cand_k > best_k else "accept") if cand_k > cur_k else "reject"
            run.check(derived == h["action"], f"step {n}: gate re-derived '{derived}' (cand {cand_k} vs current {cur_k}, best {best_k} of {dev_size}) equals history action '{h['action']}'")
        ls = log["steps"].get(n)
        if ls and ls["decision"]:
            run.check(ls["decision"] == h["action"] and abs(ls["decision_hard"] - round(cand_score, 4)) < 1e-4,
                      f"step {n}: last launch.log attempt says {ls['decision']} hard={ls['decision_hard']} (history {h['action']} {cand_score:.4f}); log attempts={ls['starts']}")
        elif log["exists"]:
            run.anomaly(f"step {n}: launch.log has no decision line for the last attempt")
        kept = True if h["action"] in ACCEPT else (False if h["action"] == "reject" else None)
        gate = {"kind": "dev_selection_gate_vs_current", "detail": {
            "metric": h.get("gate_metric", "hard"),
            "rule": "accept iff candidate dev score > score of the current skill stored at its own evaluation (strict >); accept_new_best iff also > best score",
            "candidate_score": cand_score, "candidate_n_success": cand_k, "dev_panel_size": dev_size,
            "current_id": current_id, "current_score_before": current_score, "current_n_success_before": cur_k,
            "best_id_before": best_id, "best_score_before": best_score, "best_n_success_before": best_k,
            "margin_tasks_over_current": (cand_k - cur_k) if cand_k is not None and cur_k is not None else None,
            "action": h["action"], "cache_hit_in_trainer_sel_cache": bool(cache_hit),
            "note": "the current skill's score is the one drawn when it was evaluated; candidate and current are NOT re-run together"}}
        if cache_hit:
            gate["detail"]["cache_hit_of"] = seen_hash[h["candidate_hash"]]
        # text
        if text is not None:
            added, removed = line_diff(parent_text, text)
        else:
            added, removed = [], []
        entry = {
            "step": n, "event": "candidate", "candidate_id": cid, "parent_id": current_id, "kept": kept,
            "dev": counts(dev_view), "dev_slots": dev_view["slots"] if dev_view else None, "gate": gate, "train": train,
            "skill_chars": len(text) if text is not None else len(parent_text),
            "skill_words": words(text) if text is not None else words(parent_text),
            "skill_text": text if text is not None else parent_text,
            "added_lines": added, "removed_lines": removed,
            "version_id": vid(text) if text is not None else None,
            "log_attempts": ls["starts"] if ls else None,
            "log_handled_errors": ls["handled_errors"] if ls else None,
            "optimizer_cost": {"tokens_by_stage": h.get("tokens"), "timing_s": h.get("timing"), "wall_time_s": h.get("wall_time_s")},
            "proposal": {k: h.get(k) for k in ("n_patches", "n_failure_patches", "n_success_patches", "n_edits_merged", "n_edits_ranked",
                                               "edit_budget", "support_counts", "edit_apply_summary", "action")},
        }
        rk = so / f"steps/step_{n:04d}/ranked_edits.json"
        if rk.exists():
            r = json.loads(read(rk))
            entry["proposal"]["ranked_edits"] = r.get("edits")
            entry["proposal"]["ranker_reasoning"] = r.get("reasoning")
        steps.append(entry)
        if h.get("candidate_hash"):
            seen_hash.setdefault(h["candidate_hash"], cid)
        # trainer state after the step
        if h["action"] in ACCEPT:
            current_id, current_score = cid, cand_score
        if h["action"] == "accept_new_best":
            best_id, best_score = cid, cand_score
        run.check(abs(h["current_score"] - current_score) < 1e-9 and abs(h["best_score"] - best_score) < 1e-9,
                  f"step {n}: history current/best score after step match tracked ({current_score:.4f}/{best_score:.4f})")

    # ---- the end of the run: final (placeholder) evaluation, returned skill, test ----------------------------------------------
    final_text = read(so / f"skills/skill_v{len(history):04d}.md")
    run.seen(so / f"skills/skill_v{len(history):04d}.md")
    final_view = run.pick(dev_idx, vid(final_text)[:12], dev_tasks, "final skill dev")
    placeholder_added, _ = line_diff(texts[current_id], final_text)
    final_eval = {
        "what": "skill_v%04d.md = last accepted skill + the empty slow-update placeholder; SkillOpt scores it once on dev and promotes it iff its score > best score" % len(history),
        "parent_id": current_id, "version_id": vid(final_text), "dev": counts(final_view),
        "summary_final_selection_hard": summary.get("final_selection_hard"),
        "best_score_at_that_time": best_score, "added_lines_vs_parent": placeholder_added, "skill_chars": len(final_text)}
    if final_text == texts[current_id]:
        final_eval["same_text_as_current"] = True
        final_eval["dev"] = None
        final_eval["note"] = "final skill equals the current skill: SkillOpt reuses the best score, no extra evaluation"
    elif final_view is not None and summary.get("final_selection_hard") is not None:
        run.check(abs(final_view["n_success"] / final_view["n"] - summary["final_selection_hard"]) < 1e-9,
                  f"final_selection_eval dev {final_view['n_success']}/{final_view['n']} equals summary final_selection_hard {summary['final_selection_hard']:.4f}")
    promoted = (best_text == final_text) and final_text != texts[best_id]
    run.check(not promoted and best_text == texts[best_id],
              f"best_skill.md equals the candidate text of the tracked best ({best_id}); final placeholder skill not promoted")
    run.check(summary["best_origin"] == ("initial_skill" if best_id == "seed" else f"step_{int(best_id[1:]):04d}"),
              f"summary best_origin '{summary['best_origin']}' equals tracked best {best_id}")
    run.check(abs(summary["best_selection_hard"] - best_score) < 1e-9, f"summary best_selection_hard {summary['best_selection_hard']:.4f} equals tracked best score")
    run.check(summary["total_accepts"] == sum(1 for h in history if h["action"] in ACCEPT) and
              summary["total_rejects"] == sum(1 for h in history if h["action"] == "reject") and summary["total_skips"] == 0,
              f"summary accepts/rejects/skips {summary['total_accepts']}/{summary['total_rejects']}/{summary['total_skips']} equal history counts")
    run.check(len(history) == len(batches), f"history has {len(history)} steps = {len(batches)} planned batches (no step skipped or missing)")
    # returned skill
    final_md = root / "final/SKILL.md"
    run.check(final_md.exists() and read(final_md) == best_text, "run final/SKILL.md equals skillopt/best_skill.md")
    run.check(runsum.get("final_skill") == best_text, "run summary.json final_skill equals best_skill.md (adapter returns best_skill.md)")
    returned_id = best_id
    returned_v = vid(best_text)[:12]
    test_view = run.pick(test_idx, returned_v, None, "returned skill test")
    final_test = None
    if test_view is not None:
        final_test = {"n_episodes": test_view["n"], "n_success": test_view["n_success"], "n_unknown": test_view["n_unknown"],
                      "repetitions": test_view["repetitions"], "version_id": test_view["version_id"],
                      "mean_input_tokens_per_episode": test_view["mean_input_tokens_per_episode"],
                      "source": str(Path(test_view["path"]).relative_to(root)) + " (episodes; success True counted, None = unknown counted separately)"}
        th = summary.get("test_hard")
        run.check(th is not None and abs(test_view["n_success"] / test_view["n"] - th) < 1e-9,
                  f"returned skill test evaluation {test_view['n_success']}/{test_view['n']} (unknown {test_view['n_unknown']}) equals summary test_hard {th} (unknown counted as failure by the adapter)")
        run.check(test_view["version_id"] == vid(best_text), "test evaluation version_id equals vid(best_skill.md)")
    else:
        run.anomaly("no test evaluation found for the returned skill")
    run.check(texts[best_id] is not None and best_text == texts[best_id], "returned skill text equals the text of its candidate id")

    # ---- orphans: dev evaluations of versions that never got a gate decision -------------------------------------------------------
    known = {vid(t)[:12] for t in texts.values() if t is not None} | {vid(final_text)[:12]}
    orphans = []
    for v, lst in sorted(dev_idx.items()):
        if v in known:
            continue
        for panel, f in lst:
            ev = run.load_eval(f)
            plan = Path(f).parent / "plan.json"
            created = plan.stat().st_mtime if plan.exists() else Path(f).stat().st_mtime
            o = {"version_id12": v, "dev": counts(ev), "tasks_equal_dev_panel": ev["tasks"] == dev_tasks,
                 "evaluation": str(Path(f).relative_to(root)), "_created": created, "plan_json_mtime": iso(created)}
            exhausted = Path(f).parent / "native-infrastructure-exhausted.json"
            if exhausted.exists():
                try:
                    o["native_infrastructure_exhausted"] = json.loads(read(exhausted))
                except Exception:
                    o["native_infrastructure_exhausted"] = "unreadable"
            deliv = root / "results/skills/deliveries/evaluations/dev" / f"{Path(f).parent.name}/guidance.md"
            if deliv.exists():
                raw = read(deliv)
                m = re.match(r"<bach-skill [^>]*>\n(.*)\n</bach-skill>\s*$", raw, re.S)
                text = m.group(1) if m else raw
                o["skill_text_source"] = str(deliv.relative_to(root))
                o["version_id_matches_text"] = vid(text)[:12] == v
                if o["version_id_matches_text"]:
                    o.update(skill_chars=len(text), skill_words=words(text), skill_text=text)
                else:
                    o["skill_text"] = None
            orphans.append(o)
    orphans.sort(key=lambda o: o["_created"])
    # Which step attempt produced each orphan: launch.log lists the aborts (Traceback) with step and stage. An abort at stage "6/6 EVALUATE" of step k is a
    # candidate of step k that was proposed and (re)scored but never gated. Assign them in time order only when the counts agree; otherwise leave unassigned.
    eval_aborts = [a for a in log["events"] if a["fatal"] and a["step"] is not None and a["stage"] == "6/6 EVALUATE"]
    assign = len(eval_aborts) == len(orphans) and bool(orphans)
    track_cur = {}  # step -> (current id, current score) the trainer had when it started that step
    cid_, sc_ = "seed", baseline_score
    for h in history:
        track_cur[h["step"]] = (cid_, sc_)
        if h["action"] in ACCEPT:
            cid_, sc_ = f"s{h['step']}", h["candidate_gate_score"]
    for i, o in enumerate(orphans):
        o.pop("_created")
        o["inferred_step"] = eval_aborts[i]["step"] if assign else None
        if assign:
            par, sc = track_cur[o["inferred_step"]]
            o["inferred_parent_id"], o["inferred_current_n_success"] = par, round(sc * dev_size)
            o["hypothetical_gate"] = ("would have been ACCEPTED" if o["dev"]["n_success"] > round(sc * dev_size) else "would have been REJECTED") + " by the strict > rule (hypothetical: the trainer aborted before gating)"
            if o.get("skill_text") is not None:
                plines = {x.strip() for x in texts[par].splitlines() if x.strip()}
                olines = {x.strip() for x in o["skill_text"].splitlines() if x.strip()}
                o["parent_line_containment"] = round(len(plines & olines) / max(len(plines), 1), 3)
                o["added_lines_vs_inferred_parent"], o["removed_lines_vs_inferred_parent"] = line_diff(texts[par], o["skill_text"])
        else:
            o["inference_note"] = ("launch.log absent" if not log["exists"] else
                                   f"{len(eval_aborts)} aborts at the dev-evaluation stage in launch.log vs {len(orphans)} orphan evaluations: step not inferred")
    if orphans:
        run.anomaly(f"{len(orphans)} dev evaluation(s) of versions that are not in history.json (orphans): proposals of a step attempt that was aborted "
                    f"(strict policy: an unknown dev outcome aborts the whole panel and the trainer exits), re-scored later by the retry rounds, and replaced "
                    f"by a FRESH candidate when the trainer was resumed from the last completed step (native-resumes.jsonl: new LLM requests fresh). "
                    f"They never met the gate; steps in launch.log started more than once: {[k for k, v in sorted(log['steps'].items()) if v['starts'] > 1]}. "
                    f"Orphans (version, dev success, inferred step): {[(o['version_id12'], o['dev']['n_success'], o['inferred_step']) for o in orphans]}")
    for e in log["events"]:
        if not e["fatal"]:
            run.anomaly(f"launch.log session {e['session']}, step {e['step']}, stage {e['stage']}: {e['kind']} ({e['exception']}); the step went on with fewer analyst patches")
        elif not (e["step"] is not None and e["stage"] == "6/6 EVALUATE"):
            run.anomaly(f"launch.log session {e['session']}, step {e['step']}, stage {e['stage']}: {e['kind']} {('(' + e['exception'] + ')') if e['exception'] else ''}; "
                        "the trainer was resumed (the step, if any, was redone from its start; the final-stage abort re-ran the end-of-epoch dev evaluation)")
    resumes = root / "native-resumes.jsonl"
    n_resumes = len(read(resumes).splitlines()) if resumes.exists() else 0
    unknown_file = so / "unknown_outcome.json"
    unknown_last = None
    if unknown_file.exists():
        u = json.loads(read(unknown_file))
        unknown_last = {"split": u.get("split"), "out_dir": u.get("out_dir"), "requested": u.get("requested"), "unknown_ids": [x["id"] for x in u.get("unknown", [])]}
    if not log["exists"] and n_resumes:
        run.anomaly(f"no launch.log in the run directory although native-resumes.jsonl has {n_resumes} resume(s): aborted attempts cannot be located; "
                    "all history/evaluation cross-checks still pass")

    # ---- the selection record (DeepSeek runs only): K = candidates scored, baseline and announced gain in dev tasks ----------------------------
    sel_md = OUT_ROOT / "data/mech/mech_selection.md"
    if model == "deepseek-flash" and run_id == "native-20261008-https" and sel_md.exists():
        run.seen(sel_md, "<OUT/data/mech/mech_selection.md>")
        m = re.search(rf"\| {re.escape(bench)} \| fitted \| SkillOpt \| (\d+) \| (\d+)/(\d+) \| \+(\d+) \|", read(sel_md))
        if m:
            k_paper, base_paper, n_paper, gain_paper = map(int, m.groups())
            scored = [x for x in steps if x["event"] == "candidate" and x["dev"]]
            gain = round(best_score * dev_size) - round(baseline_score * dev_size)
            run.check((len(scored), round(baseline_score * dev_size), dev_size, gain) == (k_paper, base_paper, n_paper, gain_paper),
                      f"mech_selection.md SkillOpt row (K={k_paper}, baseline {base_paper}/{n_paper}, gain +{gain_paper}) equals extracted (K={len(scored)}, baseline {round(baseline_score * dev_size)}/{dev_size}, gain +{gain})")
        else:
            run.anomaly("mech_selection.md has no SkillOpt row for this benchmark")
    # ---- the paper's edge table (DeepSeek runs only): same parent->child edges, accepted flags and dev success changes ------------------------------
    table = CACHE / "analysis/main/edges_table.json"
    if model == "deepseek-flash" and run_id == "native-20261008-https" and table.exists():
        run.seen(table, "<assets/cache/analysis/main/edges_table.json>")
        rows = [r for r in json.loads(read(table)) if r["opt"] == OPT and r["bench"] == bench and r["split"] == "dev"]
        mine = {(s["parent_id"], s["candidate_id"]): s for s in steps if s["event"] == "candidate" and s["dev"]}
        theirs = {(r["parent"], r["child"]): r for r in rows}
        run.check(set(mine) == set(theirs), f"edges_table.json dev edges {sorted(theirs)} equal the dev-scored (parent, child) pairs of the trace")
        for key, r in theirs.items():
            if key in mine:
                s = mine[key]
                gd = s["gate"]["detail"]
                run.check(bool(r["accepted"]) == bool(s["kept"]) and abs(r["dQ"] - 100 * gd["margin_tasks_over_current"] / run_dev_size) < 0.06,
                          f"edges_table {key}: accepted={r['accepted']} dQ={r['dQ']:.1f} equals kept={s['kept']} margin {gd['margin_tasks_over_current']}/{run_dev_size} tasks")
    # ---- partial? ----------------------------------------------------------------------------------------------------------------
    reasons = []
    if not (root / "summary.json").exists() or summary.get("test_hard") is None or summary.get("final_test_hard") is None:
        reasons.append("summary/test evaluation incomplete")
    if len(history) < len(batches):
        reasons.append("history shorter than planned steps")
    if time.time() - newest < RECENT_S:
        reasons.append(f"a direct child of the run root or skillopt/ was written in the last {RECENT_S // 60} minutes")
    running = active_processes(run)
    if running:
        reasons.append("a process whose arguments name this run (e.g. src.run repeat writing test repetitions under test-reps/) was running at extraction time")
    partial = bool(reasons)

    notes = [
        "SkillOpt path: step t proposes ONE candidate = current accepted skill + <= edit_budget ranked edits (patch mode); history.json is the complete, ordered log (no skip actions in this run: "
        f"{sum(1 for h in history if h['action'].startswith('skip'))} skips).",
        "kept = SkillOpt's own gate accepted the candidate (action accept/accept_new_best); gate = dev success (verifier 'hard') of the candidate vs the STORED dev score of the current skill, strict >, "
        "so an equal score is rejected; the baseline/current score is a single earlier draw, not re-run with the candidate. parent_id of a rejected candidate is the current accepted skill at that step (seed until the first accept).",
        "train = rollout of the PARENT (current) skill on the step's training batch B_t, the trajectories the analyst reflected on (not a score of the candidate); verified against train evaluation.json (task set = native_protocol batch).",
        "dev counts come from results/evaluations/dev/<vid12>-<panel>/evaluation.json (success True/False/None); mean_input_tokens_per_episode = mean over all episodes of the sum of per-request input_tokens.",
        "returned skill = skillopt/best_skill.md (the val-argmax; strict > means the earliest of equal scores wins); the end-of-epoch slow-update step only adds an EMPTY placeholder to the current skill and its dev score is in final_selection_eval; it did not replace best.",
        f"seed_dev: the seed's dev evaluation reused in this run (episodes cached by skill text; results of an earlier run may be reused: see its evaluation slot_accounting).",
        f"native-resumes.jsonl lines: {n_resumes}; skillopt/unknown_outcome.json (last abort only): {json.dumps(unknown_last)}.",
        "baseline of history: " + json.dumps(baseline_sources) + ".",
        (f"launch.log header: {log['header']}" if log.get("header") else "launch.log has no header line."),
        "fromscratch run: no launch.log in the run directory; decisions verified against history.json/evaluation.json only." if not log["exists"] else "launch.log used to verify baseline and per-step decisions.",
    ]
    if partial:
        notes.append("PARTIAL reasons: " + "; ".join(reasons) + ".")
    result = {
        "model": model, "optimizer": OPT, "benchmark": bench, "run_id": run_id, "seed": seed_kind, "partial": partial,
        "partial_reasons": reasons, "mtimes_seen": run.mtimes, "dev_panel_size": dev_size,
        "seed_dev": {k: counts(seed_dev)[k] for k in ("n", "n_success", "n_unknown", "mean_input_tokens_per_episode")} if seed_dev else None,
        "steps": steps, "returned_skill_id": returned_id, "final_test": final_test,
        "final_selection_eval": final_eval, "orphan_dev_evaluations": orphans, "launch_log_events": log["events"], "launch_log_sessions": log["sessions"],
        "optimizer_state": {"best_step": summary["best_step"], "best_origin": summary["best_origin"], "current_origin_at_end": summary["current_origin"],
                            "total_accepts": summary["total_accepts"], "total_rejects": summary["total_rejects"],
                            "baseline_test_hard": summary.get("baseline_test_hard"), "test_hard": summary.get("test_hard"),
                            "final_test_hard": summary.get("final_test_hard"), "n_resumes": n_resumes},
        "notes": " ".join(notes), "anomalies": run.anomalies, "checks": run.checks,
    }
    return result, n_dev_scored


def cross_checks(results):
    """Totals against the paper's record mech_edges.md (SkillOpt accepted/rejected dev edges over the three DeepSeek runs; 59 dev edges in all)."""
    out = []
    md = OUT_ROOT / "data/mech/mech_edges.md"
    ds = [r for r in results if r["model"] == "deepseek-flash" and r["run_id"] == "native-20261008-https"]
    if md.exists() and len(ds) == 3:
        text = read(md)
        acc = int(re.search(r"\| skillopt \| accepted \| (\d+) \|", text)[1])
        rej = int(re.search(r"\| skillopt \| rejected \| (\d+) \|", text)[1])
        mine_acc = sum(1 for r in ds for s in r["steps"] if s["event"] == "candidate" and s["dev"] and s["kept"] is True)
        mine_rej = sum(1 for r in ds for s in r["steps"] if s["event"] == "candidate" and s["dev"] and s["kept"] is False)
        out.append(("OK  " if (acc, rej) == (mine_acc, mine_rej) else "FAIL") +
                   f" mech_edges.md SkillOpt dev edges accepted/rejected {acc}/{rej} equal extracted DeepSeek {mine_acc}/{mine_rej} ({mine_acc + mine_rej} of the 59 dev edges)")
    return out


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    OUT.mkdir(parents=True, exist_ok=True)
    report, results = [], []
    for model, bench, run_id in RUNS:
        name = f"{model}__{OPT}__{bench}__{run_id}"
        if only and only not in name:
            continue
        t0 = time.time()
        result, n_dev = extract(model, bench, run_id)
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        results.append(result)
        failed = [c for c in result["checks"] if c.startswith("FAIL")]
        print(f"{name}: steps={len(result['steps'])} dev_scored_non_seed={n_dev} partial={result['partial']} "
              f"checks={len(result['checks'])} failed={len(failed)} anomalies={len(result['anomalies'])} ({time.time() - t0:.0f}s)", flush=True)
        for c in failed:
            print("   ", c)
        report.append((name, len(result["steps"]), n_dev, result["partial"], len(result["anomalies"])))
    for line in cross_checks(results):
        print(line)
    return report


if __name__ == "__main__":
    main()
