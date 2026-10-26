"""aux.calls.classify is the call classifier of the old calls modules, moved unchanged."""
import json

import pytest

from src import layout

classify = pytest.importorskip("aux.calls.classify")
old_rw = pytest.importorskip("aux.calls_and_writes.calls_rw", reason="the old modules are gone after integration")
old_cp = pytest.importorskip("aux.calls_and_writes.callsperturn")
RUN = layout.EXECUTION / "deepseek-flash/skillopt/automationbench/native-20261008-https/results/evaluations/test"


@pytest.mark.skipif(not RUN.is_dir(), reason="needs the recorded runs under BIPA_ASSETS")
def test_every_recorded_call_is_classified_as_before():
    calls = [(c["tool"], c.get("arguments") or "") for f in sorted(RUN.glob("*/evaluation.json"))[:1]
             for e in json.loads(f.read_text())["episodes"] for r in e.get("requests", []) for c in r.get("tool_calls", [])]
    assert len(calls) > 500
    assert [classify.classify(*c) for c in calls] == [old_rw.classify(*c) for c in calls]
    assert [classify.read_only(*c) for c in calls] == [old_cp.read_only(*c) for c in calls]
    assert dict(classify.AMBIG) == dict(old_rw.AMBIG)
