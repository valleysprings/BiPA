"""Exercise official retry/archive machinery with deterministic infrastructure outcomes."""
from pathlib import Path
import sys
import json
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent))
from harness_bench.eval import Evaluator, NATIVE_V1_POLICY


class Backend:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def run(self, version, panel, split, destination, attempt):
        self.calls.append((version, panel, split, attempt))
        values = self.answers[attempt]
        return {'version_id': version['version_id'], 'panel': panel, 'split': split,
                'episodes': [{'task_id': r['task_id'], 'repetition': r['repetition'],
                              'success': values[r['task_id']], 'skill_delivery': {'verified': True},
                              'outcome_reason': 'terminal_provider_failure' if values[r['task_id']] is None else 'task_verdict'}
                             for r in panel]}


@pytest.mark.parametrize('split', ['train', 'dev', 'test'])
def test_only_unknown_retried_and_false_stops(tmp_path, split):
    backend = Backend({0: {'a': False, 'b': None}, 1: {'b': False}})
    adapter = Evaluator(None, run_dir=tmp_path, executor=backend, backoff=False, policy=NATIVE_V1_POLICY)
    version = {'version_id': 'fixed-skill', 'files': {'SKILL.md': 'skill'}}
    result = adapter.evaluate(version, ['a', 'b'], split=split)
    assert [call[3] for call in backend.calls] == [0, 1]
    assert backend.calls[1][1] == [{'task_id': 'b', 'repetition': 0, 'weight': 1.0}]
    assert all(call[0] == version and call[2] == split for call in backend.calls)
    assert all(row['success'] is False for row in result['episodes'])
    lineage = result['unknown_outcome_retries']
    assert len(lineage['attempts']) == 1 and len(lineage['replaced_episodes']) == 1
    adapter.evaluate(version, ['a', 'b'], split=split)
    assert len(backend.calls) == 2


def test_exhaustion_survives_resume_and_cannot_grant(tmp_path):
    backend = Backend({0: {'a': None}, 1: {'a': None}, 2: {'a': None}})
    version = {'version_id': 'fixed', 'files': {}}
    adapter = Evaluator(None, run_dir=tmp_path, executor=backend, backoff=False, policy=NATIVE_V1_POLICY)
    result = adapter.evaluate(version, ['a'], split='train')
    assert [call[3] for call in backend.calls] == [0, 1, 2]
    lineage = result['unknown_outcome_retries']
    assert len(lineage['attempts']) == 2 and lineage['remaining_unknown']
    resumed = Evaluator(None, run_dir=tmp_path, executor=backend, backoff=False, policy=NATIVE_V1_POLICY)
    resumed.evaluate(version, ['a'], split='train')
    assert len(backend.calls) == 3
    path = resumed.directory(version, result['panel'], 'train') / 'evaluation.json'
    before = path.read_bytes()
    assert resumed._grant_retry_round(path, 'manual') is False
    assert path.read_bytes() == before
