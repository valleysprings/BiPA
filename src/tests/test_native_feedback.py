"""Long feedback remains intact in native migration and unknown never reaches optimizer."""
from pathlib import Path
import sys
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent))
from src.layout import OPTIMIZERS
sys.path.insert(0, str(OPTIMIZERS / 'skillopt'))
from src.adapters import skillopt, gepa


def test_full_feedback_preserves_system_skill_and_long_tool_output():
    long = 'evidence' * 20000
    row = {'id': 'task', 'success': 0, 'verifier': 'failed',
           'messages': [{'role': 'system', 'content': 'system'},
                        {'role': 'user', 'content': 'task <bach-skill>skill</bach-skill>'},
                        {'role': 'assistant', 'content': long}], 'trajectory': 'legacy clipped'}
    system, user, turns = skillopt.conversation(row, full=True)
    assert system == 'system' and 'skill' in user
    assert turns[0]['content'] == long
    score, feedback = gepa.side_info(row, strict=True, full=True)
    assert score == 0
    trace = feedback['Generated Outputs']['Agent Trace']
    assert long in trace and 'system' in trace and '<bach-skill>skill</bach-skill>' in trace


def test_native_abort_is_not_swallowed_by_upstream_exception_handlers(tmp_path):
    with pytest.raises(skillopt.NativeIncomplete):
        skillopt.require_known([{'id': 'task', 'success': None}], tmp_path, 'test', 'final', native=True)
    assert not issubclass(skillopt.NativeIncomplete, Exception)
    assert not issubclass(gepa.IncompleteGEPAEvaluation, Exception)
