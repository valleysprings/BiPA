"""The runtime offline: optimizer LLM, relocation, round records, panels, stubbed runs, the suite plan."""
import hashlib
import json

import httpx
import openai
import pytest

from src import run
from src.utils import executor, layout, llm

PANELS_AND_BATCHES = "8c4b5f8cdbd4ea88c308641a1df15645e9fab8d2b3544ea8ac4239e08bccd087"  # the recorded runs' ids
LLM = llm.LLM


def status_error(code):
    return openai.APIStatusError("gateway", response=httpx.Response(code, request=httpx.Request("POST", "https://g")), body=None)


def client(monkeypatch, tmp_path, outcomes):
    monkeypatch.setattr(llm, "RETRY", {"initial_delay": 0.0, "max_delay": 0.0, "max_total_wait": 60.0})
    monkeypatch.setattr(llm, "GATEWAY_MIN_ELAPSED", 0.0)
    monkeypatch.setattr(llm, "REJECTED", set())
    monkeypatch.setenv("OPENAI_API_KEY_2", "key")
    monkeypatch.setenv("OPENAI_BASE_URL_2", "https://g")
    monkeypatch.delenv("OPENAI_BASE_URL_2_OPTIMIZER", raising=False)
    model, answers = llm.LLM("deepseek-flash", tmp_path), iter(outcomes)

    def send(request, timeout):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    model._send = send
    return model


def test_gateway_rejections_end_the_call_and_other_errors_reset(monkeypatch, tmp_path):
    cut = openai.APIConnectionError(request=httpx.Request("POST", "https://g"))
    for index, error in enumerate((status_error(502), cut)):
        model = client(monkeypatch, tmp_path / str(index), [error] * 4)
        with pytest.raises(llm.GatewayRejected):
            model.complete([{"role": "user", "content": "x"}])
        with pytest.raises(llm.GatewayRejected):  # the same request fails at once
            model.complete([{"role": "user", "content": "x"}])
        assert model.usage["gateway_rejected"] == 2
    done = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    model = client(monkeypatch, tmp_path, [status_error(502)] * 3 + [status_error(503), status_error(502), done])
    assert model.complete([{"role": "user", "content": "x"}])[0] == "ok"


def test_optimizer_url_must_be_https(stub_run, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LLM", LLM)
    monkeypatch.setenv("OPENAI_API_KEY_2", "key")
    monkeypatch.setenv("OPENAI_BASE_URL_2", "http://g")
    monkeypatch.delenv("OPENAI_BASE_URL_2_OPTIMIZER", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_BASE_URL_2_OPTIMIZER"):
        run.main(["run", "gepa", "tau3", "gpt-6-luna", "--execute"])
    assert not (tmp_path / "gepa/tau3-bench/t/native_protocol.json").exists()  # no started run
    monkeypatch.setenv("OPENAI_BASE_URL_2_OPTIMIZER", "https://g/v1")
    assert str(llm.LLM("gpt-6-luna", tmp_path).client.base_url).startswith("https://g/v1")


def test_calls_and_usage_survive_a_resume(monkeypatch, tmp_path):
    done = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    client(monkeypatch, tmp_path, [done]).complete([{"role": "user", "content": "x"}])
    resumed = client(monkeypatch, tmp_path, [done])
    resumed.complete([{"role": "user", "content": "y" * 1_300_000}])
    assert resumed.usage["calls"] == 2 and resumed.usage["prompt_tokens"] == 6 and resumed.usage["truncated_calls"] == 1
    records = sorted((tmp_path / "llm").glob("call-*.json"))
    assert [p.name for p in records] == ["call-0001.json", "call-0002.json"]
    assert json.loads(records[0].read_text())["request"]["model"] == "deepseek-flash"


def test_prompt_cap_keeps_the_head_and_system_messages():
    assert llm.count_tokens("x" * 1000) == 400
    head, body = "H" * llm.KEEP_HEAD, "".join(f"[{i:07d}]" for i in range(200_000))
    messages = [{"role": "system", "content": "S" * 10}, {"role": "user", "content": head + body}]
    fitted, dropped = llm.fit_prompt(messages)
    assert fitted[0] == messages[0] and fitted[1]["content"].startswith(head + "\n[... earlier content truncated")
    assert fitted[1]["content"].endswith(body[-1000:]) and dropped > 0
    assert sum(llm.count_tokens(m["content"]) for m in fitted) <= llm.MAX_PROMPT_TOKENS
    assert llm.fit_prompt(messages[:1]) == (messages[:1], 0)


def test_relocate_maps_legacy_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(layout, "LEGACY_ROOTS", ["/old/mit", "/old/bench"])
    monkeypatch.setattr(layout, "EXECUTION", tmp_path / "execution")
    assert layout.relocate("/old/bench/assets/artifacts/cache/c/x") == layout.CACHE / "c/x"
    assert layout.relocate("/old/mit/assets/artifacts/results/r") == layout.ARTIFACTS / "results/r"
    assert layout.relocate("/old/mit/assets/optimizers/gepa/x") == layout.OPTIMIZERS / "gepa/x"
    assert layout.relocate("/old/mit/artifacts/a") == layout.ARTIFACTS / "a"
    (tmp_path / "execution/m/native-seed/b/r").mkdir(parents=True)
    assert layout.relocate(tmp_path / "execution/runs/seed/m/opencode/b/r/f") == tmp_path / "execution/m/native-seed/b/r/f"
    assert layout.relocate("/elsewhere/x") == layout.Path("/elsewhere/x")


def test_panels_and_batches_are_the_recorded_ones():
    ids = {f"{bench}/{scale}": {"panels": run.panels(bench, scale),
                                "batches": {method: run.batches(bench, scale, method) for method in run.METHODS}}
           for bench in sorted(run.BENCHMARKS) for scale in ("main", "smoke")}
    assert hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest() == PANELS_AND_BATCHES


def test_record_round(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"panel": [{"task_id": "a"}], "split": "train", "episodes": []}))
    executor.record_round(tmp_path, arm="gepa", step=3, role="minibatch", skill_text="s", evaluation_path=evaluation)
    (record,) = [json.loads(p.read_text()) for p in (tmp_path / "round-evidence").glob("*.json")]
    assert record == {"kind": "round_evaluation", "arm": "gepa", "step": 3, "role": "minibatch", "skill": {"text": "s"},
                      "evaluation_path": str(evaluation), "panel": [{"task_id": "a"}], "split": "train"}


def test_run_writes_protocol_summary_and_final_skill(stub_run, tmp_path):
    run.main(["run", "skilladam", "tau3", "deepseek-flash", "--cache-root", str(tmp_path / "cache"), "--execute"])
    out = tmp_path / "skilladam/tau3-bench/t"
    protocol, summary = json.loads((out / "native_protocol.json").read_text()), json.loads((out / "summary.json").read_text())
    assert protocol["arm_options"] == {"gate": "train-batch"} and protocol["seed"] == executor.SEED
    assert protocol["batches"] == [["a"]] and protocol["cache_root"] == str((tmp_path / "cache").resolve())
    assert summary["test"]["final"]["success"] == 1 and summary["arm_episodes"] == {"train": 1} and summary["llm"] == {"calls": 2}
    assert (out / "final/SKILL.md").read_text() == "learned"
    assert [e[:2] for e in stub_run.evaluated] == [(executor.SEED, "test"), ("learned", "test")]
    assert [(r["role"], r["step"]) for r in stub_run.rounds] == [("seed_test", 1), ("final_test", 1)]
    run.main(["run", "skilladam", "tau3", "deepseek-flash", "--execute"])  # finished: skipped
    assert len(stub_run.evaluated) == 2


def test_no_skill_run_evaluates_no_text(stub_run, tmp_path):
    run.main(["run", "none", "tau3", "deepseek-flash", "--execute"])
    out = tmp_path / "none/tau3-bench/t"
    summary = json.loads((out / "summary.json").read_text())
    assert [e[:2] for e in stub_run.evaluated] == [(None, "test")] and not stub_run.adapted
    assert summary["final_skill"] is None and summary["llm"] == {} and (out / "final/SKILL.md").read_text() == ""
    assert [(r["role"], r["step"], r["skill_text"]) for r in stub_run.rounds] == [("final_test", 0, None)]


def test_resume_keeps_the_recorded_protocol(stub_run, tmp_path):
    out = tmp_path / "seed/tau3-bench/t"
    out.mkdir(parents=True)
    recorded = {"protocol": "native-v1", "feedback_mode": "full", "arm": "seed", "model": "deepseek-flash",
                "benchmark": "tau3-bench", "harness": "opencode", "scale": "main", "cache_root": str(tmp_path / "c"),
                "panels": {"train": ["x"], "sel": ["y"], "test": ["z"]}, "batches": []}
    (out / "native_protocol.json").write_text(json.dumps(recorded))
    with pytest.raises(SystemExit):
        run.main(["run", "seed", "tau3", "deepseek-flash", "--cache-root", str(tmp_path / "other"), "--resume", "--execute"])
    with pytest.raises(SystemExit):  # a started run continues only with --resume
        run.main(["run", "seed", "tau3", "deepseek-flash", "--execute"])
    run.main(["run", "seed", "tau3", "deepseek-flash", "--resume", "--execute"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["panels"] == {"train": 1, "sel": 1, "test": 1} and set(summary["evaluated_panels"]) == {"train", "dev", "test"}
    (out / "summary.json").unlink()
    (out / "native_protocol.json").write_text(json.dumps({**recorded, "scale": "smoke"}))
    with pytest.raises(SystemExit):
        run.main(["run", "seed", "tau3", "deepseek-flash", "--resume", "--execute"])


def test_suite_plans_a_run_and_its_repetitions_per_cell(stub_run, capsys):
    run.main(["suite", "gpt-6-luna", "--benches", "tau3", "--slots", "3"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10
    for method, (train, repeat) in zip(run.METHODS, zip(lines[::2], lines[1::2])):
        assert f" run {method} tau3-bench gpt-6-luna --slots 3 " in train and train.endswith("--execute")
        assert f" repeat {method} tau3-bench gpt-6-luna --slots 3 " in repeat and "--test-runs 3" in repeat


def test_transfer_plans_where_a_skill_is_tested(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "EXECUTION", tmp_path)
    monkeypatch.setattr(layout, "EXECUTION", tmp_path)
    source = tmp_path / "deepseek-flash/gepa/tau3-bench/t"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(json.dumps({"final_skill": "learned"}))
    run.main(["transfer", "seed", "tau3", "deepseek-flash", "--to", "gpt-6-luna"])
    run.main(["transfer", "gepa", "tau3", "deepseek-flash", "--to", "tb2", "--tag", "t"])
    assert capsys.readouterr().out.splitlines() == [
        f"seed -> {tmp_path.resolve()}/gpt-6-luna/native-seed/tau3-bench/ood-seed",
        f"{source} -> {tmp_path.resolve()}/deepseek-flash/gepa/terminal-bench-2-1/ood-from-bench-tau3-bench"]
