"""SkillAdam's upstream FeedbackLoop through our adapter, offline."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from src.adapters import skilladam as arm

SEED = "# Skill\n\nStep one.\nStep two.\n"
RULE = "# Skill\n\nStep one.\nRule 1.\nStep two.\n"
NOISE = "# Skill\n\nStep one.\nNoise.\nRule 1.\nStep two.\n"
BATCHES = [["a", "b", "c"], ["d", "e", "f"]]
GOOD = {"label": "rule-missing", "description": "d", "evidence": "e", "failure_type": "rule_missing",
        "inherited_from_iteration": False}


class Executor:
    """Success on the first (number of 'Rule' lines + 1) tasks of a batch; ``unknown`` names (skill marker, task)."""
    def __init__(self, unknown=()):
        self.unknown, self.calls, self.rounds = set(unknown), [], []

    def __call__(self, cell, text, ids, split):
        self.calls.append((text, tuple(ids), split))
        good = text.count("Rule") + 1
        return [{"id": task, "evaluation_path": f"evals/{len(text)}/evaluation.json", "verifier": None,
                 "success": None if any(m in text and t == task for m, t in self.unknown) else i < good,
                 "messages": [{"role": "system", "content": "sys"},
                              {"role": "user", "content": f"task {task} <bach-skill>{text}</bach-skill>"},
                              {"role": "assistant", "content": "done"}]} for i, task in enumerate(ids)]

    def record(self, root, *, arm, step, role, skill_text, evaluation_path):
        self.rounds.append((step, role, skill_text))


class LLM:
    """Iteration t's patch adds 'Rule 1.' (t=1) or 'Noise.' (t=2); tracker replies come from ``tracker``."""
    model = "offline"

    def __init__(self, tracker=()):
        self.usage, self.tracker, self.stages = Counter(), list(tracker), []

    def complete(self, messages, tools=None, temperature=1.0, max_tokens=None):
        self.usage.update(calls=1, prompt_tokens=1, completion_tokens=1)
        usage = {"prompt_tokens": 1, "completion_tokens": 1}
        if tools:
            self.stages.append("momentum")
            arguments = self.tracker.pop(0) if self.tracker else json.dumps(GOOD)
            return "", [{"id": "1", "type": "function", "function": {"name": "report_problem", "arguments": arguments}}], usage
        self.stages.append("patch")
        line = "Noise." if "Rule 1." in messages[-1]["content"] else "Rule 1."
        return json.dumps({"reasoning": "add a rule", "patch": f"@@ -3,1 +3,2 @@\n Step one.\n+{line}\n"}), [], usage


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(arm.executor, "SEED", SEED)

    def go(executor, llm):
        monkeypatch.setattr(arm.executor, "evaluate", executor)
        monkeypatch.setattr(arm.executor, "record_round", executor.record)
        cell = SimpleNamespace(benchmark="tau3-bench", model="offline", root=tmp_path)
        return arm.run(cell, {"sel": ["v1", "v2"]}, BATCHES, llm, tmp_path)
    return go


def test_training_batch_gate(run, tmp_path):
    """Per iteration: training rollout, gate baseline on the same (skill, ids, 'train'), candidate on the same ids;
    D_sel is never read."""
    executor = Executor()
    result = run(executor, LLM())
    abc, d_f = tuple(BATCHES[0]), tuple(BATCHES[1])
    assert executor.calls == [(SEED, abc, "train"), (SEED, abc, "train"), (RULE, abc, "train"),
                              (RULE, d_f, "train"), (RULE, d_f, "train"), (NOISE, d_f, "train")]
    roles = ["training_rollout", "validation_baseline", "validation_candidate"]
    assert executor.rounds == [(1, *x) for x in zip(roles, (SEED, SEED, RULE))] + [(2, *x) for x in zip(roles, (RULE, RULE, NOISE))]
    assert [(i["baseline"], i["candidate"], i["accepted"]) for i in result["extra"]["iterations"]] == [
        (pytest.approx(1 / 3), pytest.approx(2 / 3), True), (pytest.approx(2 / 3), pytest.approx(2 / 3), False)]
    assert result["final_skill"] == RULE and result["episodes"] == {"train": 6, "gate_train": 6}
    assert result["llm"] == {"calls": 4, "tokens": 8} and result["extra"]["accepted"] == 1
    assert result["extra"]["gate"] == {"mode": "train-batch", "validation_size": None}
    assert set(result["extra"]["iterations"][0]) == {"t", "accepted", "reason", "patch_mode", "attempts", "budget",
                                                      "v_ema", "baseline", "candidate"}
    folder = tmp_path / "skilladam"
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    assert [tuple(r["validation_case_ids"]) for r in checkpoint["iterations"]] == [abc, d_f]
    assert (folder / "seed.md").read_text() == SEED and (folder / "current_skill.md").read_text() == RULE
    for step in (0, 1):
        iteration = folder / f"iterations/iter_{step:03d}"
        assert json.loads((iteration / "patch_attempt_01.json").read_text())["patch"].startswith("@@")
        training = json.loads((iteration / "training_evaluation.json").read_text())
        baseline = json.loads((iteration / "validation_baseline_evaluation.json").read_text())
        assert (baseline["results"], baseline["metrics"]) == (training["results"], training["metrics"])


def test_unknown_verdict_aborts_and_resume_reuses_the_proposal(run, tmp_path):
    llm = LLM()
    with pytest.raises(arm.UnknownVerdict):
        run(Executor(unknown={("Rule 1.", "a")}), llm)
    assert not issubclass(arm.UnknownVerdict, Exception)
    assert not (tmp_path / "skilladam/checkpoint.json").exists() and llm.stages == ["patch"]
    llm = LLM()
    result = run(Executor(), llm)
    assert llm.stages == ["momentum", "patch", "momentum"]
    assert result["final_skill"] == RULE and [i["accepted"] for i in result["extra"]["iterations"]] == [True, False]


def test_unknown_verdict_after_a_checkpoint_asks_again(run, tmp_path):
    with pytest.raises(arm.UnknownVerdict):
        run(Executor(unknown={("Noise.", "d")}), LLM())
    assert json.loads((tmp_path / "skilladam/checkpoint.json").read_text())["next_iteration"] == 1
    llm = LLM()
    result = run(Executor(), llm)
    assert (tmp_path / "skilladam/iterations-stale/iter_001-0/patch_attempt_01.json").is_file()
    assert llm.stages == ["patch", "momentum"] and result["final_skill"] == RULE


def test_unusable_tracker_replies_are_asked_again(run, tmp_path):
    llm = LLM(tracker=['{"label": "rule-miss', json.dumps({**GOOD, "label": " "})])  # a max_tokens cut, a failing apply
    run(Executor(), llm)
    assert llm.stages == ["patch", "momentum", "momentum", "momentum", "patch", "momentum"]
    assert "rule-missing" in json.dumps(json.loads((tmp_path / "skilladam/checkpoint.json").read_text())["momentum"])


def test_three_unusable_tracker_replies_abort(run):
    with pytest.raises(ValueError):
        run(Executor(), LLM(tracker=['{"label'] * 3))


def test_upstream_batch_size_and_budget():
    """Upstream --train-size 10 over the 160 train episodes: T = 16 batches of 10 distinct tasks everywhere."""
    from src.panels import batches
    for benchmark in ("tau3-bench", "automationbench", "terminal-bench-2-1"):
        planned = batches(benchmark, "main", "skilladam")
        assert len(planned) == 16 and all(len(set(b)) == 10 for b in planned)
