"""SkillOpt adapter on the vendored upstream trainer with a fake executor; no model is called."""
from collections import Counter
from types import SimpleNamespace

import pytest

from src.adapters import skillopt

CELL = SimpleNamespace(benchmark="offline", model="offline", root="run")
PANELS = {"train": ["a", "b"], "sel": ["d"], "test": ["t"]}


def fake_executor(monkeypatch, success):
    calls, rounds = [], []

    def evaluate(cell, text, ids, split):
        calls.append(split)
        return [{"id": i, "success": success(i, split), "verifier": None, "evaluation_path": f"{split}.json",
                 "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"do {i}"}]}
                for i in ids]

    monkeypatch.setattr(skillopt.executor, "evaluate", evaluate)
    monkeypatch.setattr(skillopt.executor, "record_round", lambda root, **kw: rounds.append((kw["step"], kw["role"])))
    monkeypatch.setattr(skillopt.executor, "SEED", "seed skill")
    return calls, rounds


def test_conversation_keeps_the_whole_trajectory():
    long = "evidence " * 20000
    call = {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
    row = {"verifier": "failed", "messages": [
        {"role": "system", "content": "system"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": long, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c", "content": long}]}
    system, user, turns = skillopt.conversation(row)
    assert (system, user) == ("system", "task")
    assert turns == [{"type": "message", "content": long}, {"type": "tool_call", "cmd": "bash({})", "obs": long},
                     {"role": "system", "content": "failed"}]


@pytest.mark.parametrize("fault", ["train", "dev"])
def test_unknown_train_or_dev_outcome_aborts_before_any_step(monkeypatch, tmp_path, fault):
    calls, _ = fake_executor(monkeypatch, lambda task, split: None if split == fault else 1)
    llm = SimpleNamespace(model="offline", usage=Counter(), complete=None)
    with pytest.raises(skillopt.Incomplete):
        skillopt.run(CELL, PANELS, [["a", "b"]], llm, tmp_path)
    assert calls[-1] == fault and not issubclass(skillopt.Incomplete, Exception)
    assert not (tmp_path / "skillopt/history.json").exists()
    assert not (tmp_path / "skillopt/steps/step_0001/step_record.json").exists()


@pytest.mark.parametrize("ok", [1, True])
def test_unknown_test_outcome_counts_as_failure(monkeypatch, tmp_path, ok):
    _, rounds = fake_executor(monkeypatch, lambda task, split: None if split == "test" else ok)
    harness = harness_at(tmp_path)
    [result] = harness.rollout(skillopt.items(["t"], "valid_unseen"), "skill", tmp_path / "test_eval")
    assert (result["hard"], result["fail_reason"], result["task_description"]) == (0.0, "unknown: no outcome", "do t")
    harness.rollout(skillopt.items(["a"], "train"), "skill", tmp_path / "slow_update/epoch_02/rollout_prev")
    harness.rollout(skillopt.items(["a", "b"], "train"), "skill", tmp_path / "steps/step_0003/rollout")
    assert harness.episodes == {"test": 1, "train_slow_update": 1, "train": 2}
    assert rounds == [(None, "test_eval"), (None, "slow_update/epoch_02/rollout_prev"), (3, "steps/step_0003/rollout")]
    assert harness_at(tmp_path).episodes == harness.episodes  # counts survive a resume


def harness_at(root, panels=PANELS, batches=()):
    return skillopt.Harness(CELL, panels, list(batches), {"out_root": root, "analyst_workers": 1, "failure_only": False,
                                                          "minibatch_size": 8, "edit_budget": 4})


def test_a_resumed_rollout_replaces_its_aborted_count(monkeypatch, tmp_path):
    verdict = [None]
    fake_executor(monkeypatch, lambda task, split: verdict[0])
    batch, out_dir = skillopt.items(list("abcd"), "train"), tmp_path / "steps/step_0001/rollout"
    with pytest.raises(skillopt.Incomplete):
        harness_at(tmp_path).rollout(batch, "skill", out_dir)
    verdict[0] = 1
    resumed = harness_at(tmp_path)
    resumed.rollout(batch, "skill", out_dir)
    assert resumed.episodes == {"train": 4} and harness_at(tmp_path).episodes == {"train": 4}


def test_a_missing_outcome_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(skillopt.executor, "evaluate", lambda cell, text, ids, split: [])
    with pytest.raises(skillopt.Incomplete):
        harness_at(tmp_path).rollout(skillopt.items(["a"], "train"), "skill", tmp_path / "steps/step_0001/rollout")


def test_upstream_splits_map_to_our_panels(tmp_path):
    harness = harness_at(tmp_path, {"train": list("abcdefgh"), "sel": ["d1", "d2"], "test": ["t1"]},
                         [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]])
    seen, unseen = harness.build_eval_batch(9, "valid_seen", 3), harness.build_eval_batch(9, "valid_unseen", 3)
    assert seen.payload == [{"id": "d1", "split": "dev"}, {"id": "d2", "split": "dev"}]
    assert unseen.payload == [{"id": "t1", "split": "test"}]
    slow = harness.build_train_batch(3, 7)
    assert [item["id"] for item in slow.payload] == ["g", "h", "c"] and {item["split"] for item in slow.payload} == {"train"}
    plan = harness.plan_train_epoch(epoch=2, steps_per_epoch=2, seed=5)
    assert [(spec.seed, [item["id"] for item in spec.payload]) for spec in plan] == [(2006, ["e", "f"]), (2007, ["g", "h"])]


def test_epochs_follow_upstream_steps_per_epoch():
    tb2 = [list(range(40)), list(range(40, 62))] * 2 + [list(range(18)), list(range(18, 36))]
    assert skillopt.epochs(tb2) == {"batch_size": 40, "train_size": 62, "num_epochs": 3}
    assert skillopt.epochs([list(range(i, i + 40)) for i in range(0, 160, 40)])["num_epochs"] == 1
    with pytest.raises(ValueError):
        skillopt.epochs(tb2[:5])


def test_upstream_trainer_runs_three_epochs(tmp_path):
    """Upstream's ReflACTTrainer over a 3-pass schedule: epoch boundaries at the passes, the slow update and the
    meta-skill act from epoch 2, and their train rollouts are counted apart from the scheduled ones."""
    import json
    from pathlib import Path
    from unittest.mock import patch
    from skillopt.engine import trainer
    from src.panels import chunk_passes, plan_passes
    tasks = [f"t{i}" for i in range(5)]
    planned = chunk_passes(plan_passes(tasks, 3, 7, 13, 1), 4)
    root = tmp_path / "skillopt"
    root.mkdir()
    (root / "seed.md").write_text("seed")
    cfg = skillopt.flatten_config(skillopt.load_config(
        str(Path(skillopt.skillopt.__file__).parents[1] / "configs/_base_/default.yaml")))
    cfg.update(out_root=str(root), skill_init=str(root / "seed.md"), env="harness", optimizer_backend="openai_compatible",
               optimizer_model="offline", target_model="offline", **skillopt.epochs(planned))
    harness = skillopt.Harness(SimpleNamespace(benchmark="offline", root=tmp_path),
                               {"train": tasks, "sel": ["v"], "test": ["x"]}, planned, cfg)
    rows = lambda cell, text, ids, split: [{"id": i, "success": 1, "messages": [], "verifier": None,
                                            "evaluation_path": "e"} for i in ids]
    with patch.object(skillopt.executor, "evaluate", rows), \
            patch.object(skillopt.executor, "record_round", lambda root, **kw: None), \
            patch.object(harness, "reflect", return_value=[]), \
            patch.object(trainer, "run_slow_update", return_value=None), \
            patch.object(trainer, "run_meta_skill", return_value=None):
        trainer.ReflACTTrainer(cfg, harness).train()
    history = json.loads((root / "history.json").read_text())
    assert [(h["epoch"], h["rollout_n"]) for h in history] == [(e, len(b)) for e, b in zip((1, 1, 2, 2, 3, 3), planned)]
    slow = [json.loads((root / f"slow_update/epoch_{e:02d}/slow_result.json").read_text())["action"] for e in (1, 2, 3)]
    meta = [json.loads((root / f"meta_skill/epoch_{e:02d}/meta_skill_result.json").read_text())["action"] for e in (1, 2, 3)]
    assert slow == ["inject_placeholder", "no_content", "no_content"]  # acted (here without an LLM) from epoch 2
    assert meta == ["skip_first_epoch", "no_content", "no_content"]
    assert harness.episodes["train"] == 13 and harness.episodes["train_slow_update"] > 0
