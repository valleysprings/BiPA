"""Offline transport instrumentation and durable publication checks; no model calls."""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import llm
def publish(directory, index):
    ledger = llm.AttemptLedger(directory)
    recorder = llm.CallRecorder(directory, attempt_ledger=ledger)
    request = {"messages": [{"role": "user", "content": f"request {index}"}]}
    recorder.record(llm.digest(request), request, response={"content": f"reply {index}"})


class AttemptLedgerTests(unittest.TestCase):
    def test_process_append_preserves_existing_bytes_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"llm"
            folder.mkdir()
            original = b'{"key":"old", "request":{}, "response":{"text":"old"}}\n'
            (folder/"call-0007.json").write_bytes(original)
            context = multiprocessing.get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
                list(pool.map(publish, [directory]*12, range(12)))
            self.assertEqual((folder/"call-0007.json").read_bytes(), original)
            calls = list(folder.glob("call-*.json"))
            self.assertEqual(len(calls), 13)
            requests = [json.loads(p.read_text())["request"] for p in calls]
            self.assertEqual(len({llm.digest(r) for r in requests}), 13)
            events = [json.loads(line) for line in (folder/"attempts.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event"]=="record_published" for e in events), 12)

    def test_append_failure_is_visible_without_overwriting_or_partial_call(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = llm.CallRecorder(directory, attempt_ledger=llm.AttemptLedger(directory))
            recorder.record("original", {"text":"unchanged"}, response={"text":"unchanged"})
            original = (recorder.folder/"call-0001.json").read_bytes()
            with patch.object(llm.os, "link", side_effect=OSError("SECRET credential body")):
                with self.assertRaises(OSError):
                    recorder.record("failed", {"text":"new"}, response={"text":"new"})
            self.assertEqual((recorder.folder/"call-0001.json").read_bytes(), original)
            self.assertEqual(len(list(recorder.folder.glob("call-*.json"))), 1)
            self.assertEqual(list(recorder.folder.glob(".call-*.tmp")), [])
            ledger = (recorder.folder/"attempts.jsonl").read_text()
            self.assertNotIn("SECRET", ledger)
            self.assertEqual(json.loads(ledger.splitlines()[-1])["event"], "record_error")

    def test_retry_decisions_match_legacy_and_every_attempt_has_result(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(enabled):
                client = llm.LLM.__new__(llm.LLM)
                client.timeout = 600
                client.attempt_ledger = llm.AttemptLedger(directory) if enabled else None
                failures = [llm.IncompleteStream("SECRET body"), {"result":"ok"}]
                timeouts, sleeps = [], []
                def send(request, *, timeout):
                    timeouts.append(timeout)
                    value = failures.pop(0)
                    if isinstance(value, Exception): raise value
                    return value
                client._send = send
                with patch.object(llm.time, "monotonic", return_value=100), patch.object(llm.time, "sleep", side_effect=sleeps.append):
                    result = client._call({"model":"mock", "messages":[]})
                return result, timeouts, sleeps
            self.assertEqual(run(False), run(True))
            events = [json.loads(line) for line in (Path(directory)/"llm/attempts.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["started","result","started","result"])
            self.assertEqual(events[1]["next_delay_seconds"], 4)
            self.assertEqual(events[1]["error_class"], "IncompleteStream")
            self.assertEqual(events[-1]["terminal_reason"], "success")
            self.assertNotIn("SECRET", json.dumps(events))

    def test_terminal_errors_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"HARNESS_LLM_ATTEMPT_LEDGER":"0"}), \
                 patch.object(llm, "dotenv_values", return_value={"OPENAI_API_KEY_2":"offline", "OPENAI_BASE_URL_2":"https://offline.invalid"}), \
                 patch.object(llm.openai, "OpenAI"):
                client = llm.LLM("deepseek-flash", directory)
            self.assertIsNone(client.attempt_ledger)
            self.assertFalse((Path(directory)/"llm/attempts.jsonl").exists())
            client.attempt_ledger = llm.AttemptLedger(directory)
            client._send = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("SECRET"))
            with self.assertRaises(ValueError): client._call({"messages":[]})
            events = [json.loads(line) for line in client.attempt_ledger.path.read_text().splitlines()]
            self.assertEqual(events[-1]["terminal_reason"], "nonretryable")
            self.assertNotIn("SECRET", json.dumps(events))

    def test_answer_replay_optional_fresh_calls_preserve_old_record(self):
        with tempfile.TemporaryDirectory() as directory:
            request = {"model":"deepseek-flash", "messages":[{"role":"user","content":"same"}],
                       "temperature":1.0,"max_tokens":32768}
            old_response = {"choices":[{"message":{"content":"old"}}],"usage":{}}
            new_response = {"choices":[{"message":{"content":"fresh"}}],"usage":{}}
            llm.CallRecorder(directory).record(llm.digest(request),request,response=old_response)
            old_path = Path(directory)/"llm/call-0001.json"
            old_bytes = old_path.read_bytes()
            def client(**kwargs):
                with patch.dict(os.environ, {"HARNESS_LLM_ANSWER_REPLAY":"1"}), \
                     patch.object(llm,"dotenv_values",return_value={"OPENAI_API_KEY_2":"offline","OPENAI_BASE_URL_2":"https://offline.invalid"}), \
                     patch.object(llm.openai,"OpenAI"):
                    result = llm.LLM("deepseek-flash",directory,**kwargs)
                return result
            legacy = client()
            with patch.object(legacy,"_call",side_effect=AssertionError("legacy should replay")):
                self.assertEqual(legacy.complete(request["messages"])[0],"old")
            native = client(answer_replay=False)
            with patch.object(native,"_call",return_value=new_response) as fresh:
                self.assertEqual(native.complete(request["messages"])[0],"fresh")
                fresh.assert_called_once_with(request)
            self.assertEqual(old_path.read_bytes(),old_bytes)
            self.assertEqual(json.loads((old_path.parent/"call-0002.json").read_text())["response"],new_response)
            self.assertEqual(native.usage["replayed"],0)
            with patch.dict(os.environ, {"HARNESS_LLM_ANSWER_REPLAY":"0"}), \
                 patch.object(llm,"dotenv_values",return_value={"OPENAI_API_KEY_2":"offline","OPENAI_BASE_URL_2":"https://offline.invalid"}), \
                 patch.object(llm.openai,"OpenAI"):
                self.assertFalse(llm.LLM("deepseek-flash",directory).answer_replay)


if __name__ == "__main__":
    unittest.main()
