"""SkillAdam's upstream FeedbackLoop through our Runner, offline: strict unknown abort, resume reusing the persisted
proposal, truncated tracker tool calls asked again, and per-round states derived from upstream's own logs."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE.parent), str(BASE.parent / "scripts/analysis")]
from src.adapters import skilladam as arm
from src import native_trace
from audit_skill_history import Store, skilladam as derive

SEED = "# Skill\n\nStep one.\nStep two.\n"
BATCHES = [["a", "b", "c"], ["d", "e", "f"]]
SEL = {"sel": ["v1", "v2", "v3"]}  # D_sel: fewer than 10, so every iteration validates on all of it
GOOD = {"label": "rule-missing", "description": "d", "evidence": "e", "failure_type": "rule_missing",
        "inherited_from_iteration": False}


def version(text):
    return hashlib.sha256(json.dumps({"SKILL.md": text}, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class Executor:
    """Success on the first (number of 'Rule' lines + 1) tasks of a batch; ``unknown`` names (skill marker, task)."""
    def __init__(self, root, unknown=()):
        self.root, self.unknown, self.calls = root, set(unknown), []

    def __call__(self, cell, text, ids, split):
        self.calls.append((text, tuple(ids), split))
        path = self.root / "evals" / version(text)[:12] / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version_id": version(text), "split": split, "panel": list(ids)}))
        good = text.count("Rule") + 1
        return [{"id": task, "evaluation_path": str(path), "verifier": None,
                 "success": None if any(m in text and t == task for m, t in self.unknown) else int(i < good),
                 "messages": [{"role": "system", "content": "sys"},
                              {"role": "user", "content": f"task {task} <bach-skill>{text}</bach-skill>"},
                              {"role": "assistant", "content": "done"}]} for i, task in enumerate(ids)]


class LLM:
    """Iteration t's patch adds 'Rule t.' (t=1) or 'Noise.' (t=2); tracker replies come from ``tracker``."""
    model = "offline"

    def __init__(self, tracker=()):
        self.usage, self.tracker, self.stages = Counter(), list(tracker), []

    def complete(self, messages, tools=None, temperature=1.0, max_tokens=None):
        self.usage.update(calls=1)
        usage = {"prompt_tokens": 1, "completion_tokens": 1}
        if tools:
            self.stages.append("momentum")
            arguments = self.tracker.pop(0) if self.tracker else json.dumps(GOOD)
            return "", [{"id": "1", "type": "function",
                         "function": {"name": "report_problem", "arguments": arguments}}], usage
        self.stages.append("patch")
        line = "Noise." if "Rule 1." in messages[-1]["content"] else "Rule 1."
        return json.dumps({"reasoning": "add a rule", "patch": f"@@ -3,1 +3,2 @@\n Step one.\n+{line}\n"}), [], usage


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.setattr(arm.executor, "SEED", SEED)
    monkeypatch.setenv("HARNESS_STRICT_OUTCOMES", "1")
    monkeypatch.setenv("HARNESS_PROTOCOL", "native-v1")
    return SimpleNamespace(benchmark="tau3-bench", model="offline", root=tmp_path)


def test_unknown_aborts_resume_reuses_the_persisted_proposal(cell, tmp_path, monkeypatch):
    first = Executor(tmp_path, unknown={("Rule 1.", "v1")})  # the candidate of iteration 1 has an unknown verdict
    monkeypatch.setattr(arm.executor, "evaluate", first)
    llm = LLM()
    with pytest.raises(arm.IncompleteSkillAdamEvaluation):
        arm.run(cell, SEL, BATCHES, llm, tmp_path)
    assert not issubclass(arm.IncompleteSkillAdamEvaluation, Exception)
    assert json.loads((tmp_path / "skilladam-incomplete.json").read_text())["unknown"] == ["v1"]
    assert not (tmp_path / "skilladam/checkpoint.json").exists() and llm.stages == ["patch"]

    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path))  # the executor's retry recovered the verdict
    llm = LLM()
    result = arm.run(cell, SEL, BATCHES, llm, tmp_path)
    assert llm.stages == ["momentum", "patch", "momentum"]  # iteration 1's proposal reused, not asked again
    assert result["final_skill"] == "# Skill\n\nStep one.\nRule 1.\nStep two.\n"
    assert [i["accepted"] for i in result["extra"]["iterations"]] == [True, False]
    assert result["extra"]["outcome_policy"]["unknown_outcome"] == "abort_and_rerun_iteration"
    assert not (tmp_path / "skilladam-incomplete.json").exists()
    assert any(d.startswith("Unknown outcomes: a train or dev verdict") for d in arm.DEVIATIONS)


def test_truncated_or_failing_tracker_calls_are_asked_again(cell, tmp_path, monkeypatch):
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path))
    llm = LLM(tracker=['{"label": "rule-miss', json.dumps({**GOOD, "label": " "})])  # a max_tokens cut, then a failing apply
    arm.run(cell, SEL, BATCHES, llm, tmp_path)
    retries = [json.loads(line) for line in (tmp_path / "skilladam-reply-retries.jsonl").read_text().splitlines()]
    assert [(r["stage"], r["iteration"], r["attempt"]) for r in retries] == [("momentum_update", 0, 1), ("momentum_update", 0, 2)]
    tracker = json.loads((tmp_path / "skilladam/checkpoint.json").read_text())["momentum"]
    assert "rule-missing" in json.dumps(tracker)

    llm = LLM(tracker=['{"label'] * 3)
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path / "other"))
    with pytest.raises(ValueError):  # three unusable replies: abort, upstream's resume asks again
        arm.run(SimpleNamespace(benchmark="tau3-bench", model="offline", root=tmp_path / "other"), SEL, BATCHES, llm, tmp_path / "other")


def test_round_states_derived_from_upstream_logs(cell, tmp_path, monkeypatch):
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path))
    result = arm.run(cell, SEL, BATCHES, LLM(), tmp_path)
    events, final, proposed = derive(tmp_path, Store(tmp_path / "derived"))
    assert [(e["step"], e["decision"]) for e in events] == [(1, "accept"), (2, "reject")] and proposed == 2
    assert Path(events[0]["proposed"]["snapshot_path"]).read_text() == result["final_skill"]
    assert events[1]["after"]["sha256"] == events[1]["before"]["sha256"] != events[1]["proposed"]["sha256"]
    native_trace.materialize_round_states(tmp_path, "skilladam")
    records = [json.loads(p.read_text()) for p in (tmp_path / "round-evidence").glob("*.json")]
    states = {r["step"]: r for r in records if r["kind"] == "round_state"}
    roles = Counter((r["step"], r["role"]) for r in records if r["kind"] == "round_evaluation")
    assert set(states) == {1, 2} and roles == {(1, "training_rollout"): 1, (1, "validation_baseline"): 1,
                                               (1, "validation_candidate"): 1, (2, "training_rollout"): 1,
                                               (2, "validation_baseline"): 1, (2, "validation_candidate"): 1}
    assert states[1]["returned"]["text"] == result["final_skill"] and states[2]["decision"] == "reject"
    assert all(s["evaluation_records"] and s["source_files"] for s in states.values())
    import render_native_skill_history as render  # per-round skills exported to results/<model>/skilladam/<bench>/skills
    initial, rounds = render.skilladam(render.Reader(), tmp_path)
    assert initial["text"] == SEED and [r["decision"] for r in rounds] == ["accepted", "rejected"]
    assert rounds[0]["selected"]["text"] == result["final_skill"] == rounds[1]["parent"]["text"]
    assert "Noise." in rounds[1]["candidate"]["text"] and rounds[1]["scores"]["batch_n"] == 3


class Scripted(LLM):
    """Proposal replies in order; tracker replies as ``LLM``."""
    def __init__(self, patches):
        super().__init__()
        self.patches = list(patches)

    def complete(self, messages, tools=None, temperature=1.0, max_tokens=None):
        if tools:
            return super().complete(messages, tools, temperature, max_tokens)
        self.usage.update(calls=1)
        self.stages.append("patch")
        return self.patches.pop(0), [], {"prompt_tokens": 1, "completion_tokens": 1}


def test_failed_patches_replayed_and_derived(cell, tmp_path, monkeypatch):
    """Iteration 1: attempt 1 does not apply, attempt 2 does, its candidate is unknown -> abort; the resume replays both
    attempts. Iteration 2: all three attempts fail -> no candidate (no_proposal, rendered patch_failed)."""
    bad = json.dumps({"reasoning": "r", "patch": "@@ -3,1 +3,1 @@\n-Absent line.\n+X\n"})
    rule = json.dumps({"reasoning": "r", "patch": "@@ -3,1 +3,2 @@\n Step one.\n+Rule 1.\n"})
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path, unknown={("Rule 1.", "v1")}))
    with pytest.raises(arm.IncompleteSkillAdamEvaluation):
        arm.run(cell, SEL, BATCHES, Scripted([bad, rule]), tmp_path)
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path))
    llm = Scripted([bad] * 3)
    arm.run(cell, SEL, BATCHES, llm, tmp_path)
    assert llm.stages == ["momentum", "patch", "patch", "patch", "momentum"]
    events, _, proposed = derive(tmp_path, Store(tmp_path / "derived"))
    assert [(e["decision"], e["record"]["patch_attempts"]) for e in events] == [("accept", 2), ("no_proposal", 3)]
    assert proposed == 1 and events[1]["proposed"]["status"] == "unknown" and events[1]["after"] == events[1]["before"]
    import render_native_skill_history as render
    _, rounds = render.skilladam(render.Reader(), tmp_path)
    assert [r["decision"] for r in rounds] == ["accepted", "patch_failed"] and rounds[1]["candidate"]["status"] == "unwitnessed"


def test_legacy_scores_unknown_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(arm.executor, "SEED", SEED)
    monkeypatch.delenv("HARNESS_STRICT_OUTCOMES", raising=False)
    monkeypatch.delenv("HARNESS_PROTOCOL", raising=False)
    monkeypatch.setattr(arm.executor, "evaluate", Executor(tmp_path, unknown={("Step", "v1")}))
    result = arm.run(SimpleNamespace(benchmark="tau3-bench", model="offline", root=tmp_path), {"sel": ["v1"]}, BATCHES, LLM(), tmp_path)
    assert result["extra"]["outcome_policy"]["unknown_outcome"] == "score_zero"
    assert result["extra"]["iterations"][0]["baseline"] == 0.0


def test_upstream_batch_size_and_budget():
    from src.panels import batches
    for benchmark, (T, episodes) in {"tau3-bench": (16, 155), "automationbench": (15, 148),
                                     "terminal-bench-2-1": (18, 160)}.items():
        planned = batches(benchmark, "main", "skilladam")
        assert (len(planned), sum(map(len, planned)), max(map(len, planned))) == (T, episodes, 10 if T < 18 else 9)


def test_disjoint_validation_on_dsel(cell, tmp_path, monkeypatch):
    """Upstream's disjoint mode: a fresh min(10, |D_sel|) sample per iteration from D_sel, gated on dev rollouts."""
    import random
    sel = [f"s{i}" for i in range(25)]
    spy = Executor(tmp_path)
    monkeypatch.setattr(arm.executor, "evaluate", spy)
    result = arm.run(cell, {"sel": sel}, BATCHES, LLM(), tmp_path)
    rng = random.Random(42)
    drawn = [rng.sample(sel, 10) for _ in BATCHES]
    assert drawn[0] != drawn[1]
    dev = [ids for _, ids, split in spy.calls if split == "dev"]
    assert dev == [tuple(d) for d in drawn for _ in range(2)]  # incumbent, then candidate, on each iteration's batch
    assert all(set(ids) <= set(sum(BATCHES, [])) for _, ids, split in spy.calls if split == "train")
    assert result["episodes"]["dev"] == len({(t, i) for t, ids, s in spy.calls if s == "dev" for i in ids})
    assert result["episodes"]["train"] == len({(t, i) for t, ids, s in spy.calls if s == "train" for i in ids})
    assert any(d.startswith("Validation: upstream's default disjoint mode") for d in arm.DEVIATIONS)
