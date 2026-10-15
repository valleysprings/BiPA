"""Offline tests using the installed upstream engine; no harness or model calls."""
import json
from collections import Counter
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bipa.adapters import gepa


class StrictOutcomeTests(unittest.TestCase):
    def test_legacy_unknown(self):
        row = dict(id="task", success=None, verifier=None, trajectory="")
        self.assertEqual(gepa.side_info(row)[0], 0)
        with self.assertRaises(gepa.IncompleteGEPAEvaluation):
            gepa.side_info(row, strict=True)

    def test_abort_and_resume_same_slot(self):
        for fault in ("train", "dev"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                calls = []
                def evaluate(cell, text, ids, split):
                    calls.append(split)
                    return [dict(id=task, success=None if split == fault else 0,
                                 verifier=None, trajectory="trace") for task in ids]
                def llm(prompt):
                    raise RuntimeError("offline simulated exhausted retries")
                with patch.dict("os.environ", HARNESS_STRICT_OUTCOMES="1"), patch.object(gepa, "evaluate", evaluate):
                    for attempt in range(2):
                        with self.assertRaises(gepa.IncompleteGEPAEvaluation):
                            gepa.run(SimpleNamespace(benchmark="offline"), {"sel": ["dev-task"]},
                                     [["train-task"]], llm, out)
                        self.assertEqual(json.loads((out / "gepa-incomplete.json").read_text())["status"], "incomplete")
                        if fault != "dev":
                            with (out / "gepa" / "gepa_state.bin").open("rb") as stream:
                                state = pickle.load(stream)
                            self.assertEqual(state["i"], -1)
                    if fault == "train":
                        self.assertEqual(calls.count("train"), 2)

    def test_reflection_failure_keeps_upstream_skip_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def evaluate(cell, text, ids, split):
                return [dict(id=task, success=0, verifier=None, trajectory="trace") for task in ids]
            def llm(prompt):
                raise RuntimeError("offline reflection failure")
            llm.usage = Counter()
            with patch.dict("os.environ", HARNESS_STRICT_OUTCOMES="1", HARNESS_PROTOCOL="", HARNESS_FEEDBACK_MODE="condensed"), patch.object(gepa, "evaluate", evaluate):
                result = gepa.run(SimpleNamespace(benchmark="offline"), {"sel": ["dev-task"]},
                                  [["train-task"]], llm, out)
            self.assertEqual(result["final_skill"], gepa.SEED)
            self.assertFalse((out / "gepa-incomplete.json").exists())
            self.assertIn("offline reflection failure", (out / "gepa/run_log.txt").read_text())


if __name__ == "__main__":
    unittest.main()
