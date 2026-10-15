import json

import pytest

from audit_native_run import certify, request_payload_matches, skill_digest
from audit_trace_integrity import Evidence, audit_episode
from bipa.native_trace import record_round_state, skill_identity


def test_empty_response_normalization_does_not_certify_incomplete_capture():
    source, record = {"request": {"input": []}, "response": None}, {"request": {"input": []}, "response": {}}
    assert request_payload_matches(source, record)
    assert not request_payload_matches(source, {**record, "response": {"output": ["invented"]}})
    assert not request_payload_matches(source, {**record, "response": []})
    episode = {"task_id": "task", "success": True, "outcome_valid": True,
               "capture": {"requests_complete": False, "invocations_complete": False},
               "skill_delivery": {"verified": True}, "requests": []}
    result = audit_episode(episode, Evidence(), {})
    assert not result["trace_verified"]
    assert {issue["kind"] for issue in result["issues"]} == {"requests_complete", "invocations_complete"}


@pytest.mark.parametrize("rounds", [4, 5, 20])
def test_audit_checks_every_frozen_round_without_requiring_ten(tmp_path, monkeypatch, rounds):
    import audit_skill_history
    protocol = dict(protocol="native-v1", feedback_mode="full", arm="skillopt", model="offline",
                    benchmark="offline", harness="fixture", scale="main", batches=[[f"t{i}"] for i in range(rounds)])
    protocol["config_hash"] = skill_digest(protocol)
    (tmp_path / "native_protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "summary.json").write_text(json.dumps({**protocol, "final_skill": "seed"}))
    original = tmp_path / "original-log.json"; original.write_text("{}")
    events = []
    for step in range(1, rounds + 1):
        events.append(dict(kind="optimizer_step", step=step, decision="skip_perfect",
                           before=skill_identity("seed"), proposed={}, after=skill_identity("seed")))
        record_round_state(tmp_path, arm="skillopt", step=step, input_skill="seed", candidate_skill=None,
                           returned_skill="seed", decision="skip_perfect", source_files=[original])
        (tmp_path / "round-evidence" / f"eval-{step}.json").write_text(json.dumps({
            "kind": "round_evaluation", "arm": "skillopt", "step": step, "skill": skill_identity("seed")}))
    monkeypatch.setattr(audit_skill_history, "skillopt", lambda *args: (events, {}, 0))
    audit = certify(tmp_path)
    assert audit["step_objects_verified"], audit["issues"]
    assert len(audit["steps"]) == rounds
    assert not audit["complete"]  # Round objects alone cannot stand in for evaluation traces.


def test_historical_source_paths_resolve_after_rename():
    from bipa.layout import BASELINE_ROOT, MITIGATION_ROOT, OPTIMIZERS, SKILLS, relocate
    old = "/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/baselines/candidate-skill"
    assert relocate(old + "/arms/gepa.py") == BASELINE_ROOT / "adapters/gepa.py"
    assert relocate(old + "/skills/seed/SKILL.md") == SKILLS / "seed/SKILL.md"
    roots = (str(MITIGATION_ROOT), "/mnt/shared-storage-user/zhoujiawei/coharness/harness-mitigation", "")
    for root in roots:
        prefix = root + "/" if root else ""
        for directory in ("baselines/candidate-skill", "baselines/bipa"):
            for name in ("executor.py", "llm.py"):
                assert relocate(prefix + directory + "/" + name) == BASELINE_ROOT / "utils" / name
        for directory in ("assets/optimizers", "baselines/assets/optimizers"):
            assert relocate(prefix + directory + "/gepa/src/gepa/core/result.py") == OPTIMIZERS / "gepa/src/gepa/core/result.py"
    # Current package paths and sibling names must not be redirected by a prefix match.
    assert relocate(BASELINE_ROOT / "utils/executor.py") == BASELINE_ROOT / "utils/executor.py"
    unchanged = MITIGATION_ROOT / "assets/optimizers-backup/gepa.py"
    assert relocate(unchanged) == unchanged


def test_gepa_resumed_log_requires_unique_committed_block():
    from audit_skill_history import gepa_block_starts
    record = {"selected_program_candidate": 2, "new_program_idx": 6}
    aborted = ["Iteration 12: Selected program 3 score: 0.6\n", "aborted\n"]
    committed = ["Iteration 12: Selected program 2 score: 0.5\n",
                 "Iteration 12: New program candidate index: 6\n"]
    assert gepa_block_starts(aborted + committed, 12, record) == [2]
    assert len(gepa_block_starts(committed + committed, 12, record)) == 2
    assert gepa_block_starts(aborted + [committed[0]], 12, record) == []


def test_round_witness_survives_effective_evaluation_retry_update(tmp_path):
    from bipa.native_trace import record_round_evaluation
    identity = skill_identity("seed")
    path = tmp_path / "evaluation.json"
    initial = {"version_id": identity["version_id"], "split": "dev", "panel": ["task"],
               "episodes": [{"task_id": "task", "success": None}]}
    path.write_text(json.dumps(initial))
    event_path = record_round_evaluation(tmp_path, arm="gepa", step=1, role="full_selection",
                                         skill_text="seed", evaluation_path=path)
    event = json.loads(event_path.read_text())
    path.write_text(json.dumps({**initial, "episodes": [{"task_id": "task", "success": True}]}))
    from pathlib import Path
    assert json.loads(Path(event["evaluation_path"]).read_text()) == initial
    assert event["effective_evaluation_path"] == str(path)
