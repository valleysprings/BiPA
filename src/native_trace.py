"""Immutable local witnesses; callers provide round/role, never inferred from hashes."""
import datetime
import hashlib
import json
import uuid
from pathlib import Path


def skill_identity(text):
    if text is None:
        return None
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "version_id": hashlib.sha256(json.dumps({"SKILL.md": text}, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "text": text,
    }


def _record(run_root, event):
    directory = Path(run_root) / "round-evidence"
    directory.mkdir(parents=True, exist_ok=True)
    event["recorded_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = directory / (uuid.uuid4().hex + ".json")
    with path.open("x") as stream:
        json.dump(event, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def record_round_evaluation(run_root, *, arm, step, role, skill_text,
                            evaluation_path, panel=None):
    path = Path(evaluation_path).resolve()
    payload = path.read_bytes()
    evaluation = json.loads(payload)
    identity = skill_identity(skill_text)
    expected_version = identity["version_id"] if identity else hashlib.sha256(b"{}").hexdigest()
    if evaluation.get("version_id") != expected_version:
        raise ValueError("round evaluation skill version mismatch")
    if panel is not None and evaluation.get("panel") != panel:
        raise ValueError("round evaluation panel mismatch")
    # harness-bench may update the effective evaluation after unknown-only retries.
    # Freeze what this callback actually saw; retain the live location separately.
    digest = hashlib.sha256(payload).hexdigest()
    snapshot = Path(run_root).resolve() / "round-evidence/evaluations" / digest / "evaluation.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        with snapshot.open("xb") as stream:
            stream.write(payload)
    return _record(run_root, {
        "kind": "round_evaluation", "arm": arm, "step": step, "role": role,
        "skill": identity, "evaluation_path": str(snapshot), "effective_evaluation_path": str(path),
        "evaluation_sha256": digest,
        "panel": evaluation.get("panel"), "split": evaluation.get("split"),
        "semantics": "Caller supplied round and role; trace completeness requires physical audit.",
    })


def record_round_state(run_root, *, arm, step, input_skill, candidate_skill,
                       returned_skill, decision, parent_candidate_idx=None,
                       failure=None, evaluation_records=(), source_files=()):
    return _record(run_root, {
        "kind": "round_state", "arm": arm, "step": step,
        "input": skill_identity(input_skill),
        "candidate": skill_identity(candidate_skill),
        "returned": skill_identity(returned_skill), "decision": decision,
        "parent_candidate_idx": parent_candidate_idx, "failure": failure,
        "evaluation_records": [str(Path(p).resolve()) for p in evaluation_records],
        "source_files": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in source_files},
        "provenance": "derived_from_upstream_logs" if source_files else "caller_round_hook",
    })


def materialize_round_states(run_root, arm):
    """Derive states from original persisted optimizer logs; no decisions invented."""
    import sys
    analysis = Path(__file__).resolve().parents[1] / "scripts/analysis"
    sys.path.insert(0, str(analysis))
    from audit_skill_history import Store, skillopt, gepa, skilladam
    root = Path(run_root).resolve()
    derive = {"skillopt": skillopt, "gepa": gepa, "skilladam": skilladam}[arm]
    events, _, _ = derive(root, Store(root / "round-evidence/derived"))
    existing = [json.loads(p.read_text()) for p in (root / "round-evidence").glob("*.json")]
    for event in events:
        if event.get("kind") != "optimizer_step":
            continue
        step = event["step"]
        if any(e.get("kind") == "round_state" and e.get("step") == step for e in existing):
            continue
        decision = event["decision"]
        def text(name):
            value = event.get(name, {})
            return Path(value["snapshot_path"]).read_text() if value.get("status") == "known" else None
        sources = set()
        for ref in event.get("sources", []):
            path = ref.split("#",1)[0].split(":lines:",1)[0]
            if Path(path).is_file(): sources.add(path)
        for name in ("before", "proposed", "after"):
            for ref in event.get(name,{}).get("sources",[]):
                path = ref.split("#",1)[0].split(":lines:",1)[0]
                if Path(path).is_file(): sources.add(path)
        links = [p for p in (root / "round-evidence").glob("*.json")
                 if (lambda e: e.get("kind") == "round_evaluation" and e.get("step") == step)(json.loads(p.read_text()))]
        record_round_state(root, arm=arm, step=step, input_skill=text("before"),
            candidate_skill=text("proposed"), returned_skill=text("after"), decision=decision,
            parent_candidate_idx=event.get("parent_candidate_idx"),
            failure=event.get("failure_evidence"), evaluation_records=links, source_files=sources)
