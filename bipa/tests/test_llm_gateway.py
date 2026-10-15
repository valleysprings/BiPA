import httpx, openai, pytest
from bipa.utils import llm
def status_error(code):
    response = httpx.Response(code, request=httpx.Request("POST", "http://gateway/v1/chat/completions"))
    return openai.APIStatusError("gateway", response=response, body=None)


def client(monkeypatch, outcomes):
    monkeypatch.setattr(llm, "RETRY", {"initial_delay": 0.0, "max_delay": 0.0, "max_total_wait": 60.0})
    monkeypatch.setattr(llm, "GATEWAY_MIN_ELAPSED", 0.0)  # every immediate 502 counts as a slow gateway answer
    monkeypatch.setattr(llm, "REJECTED", set())  # each test is its own process as far as the rejected-request memo goes
    model = llm.LLM.__new__(llm.LLM)
    model.timeout, model.attempt_ledger, model.usage, model.lock = 600, None, llm.Counter(), llm.threading.Lock()
    calls = iter(outcomes)
    def send(request, *, timeout=None):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    model._send = send
    return model


def test_repeated_gateway_502_ends_the_call(monkeypatch):
    model = client(monkeypatch, [status_error(502)] * llm.GATEWAY_REJECTIONS)
    with pytest.raises(llm.GatewayRejected):
        model._call({"messages": []})
    assert model.usage["gateway_rejected"] == 1


def test_other_errors_reset_and_retry(monkeypatch):
    done = {"choices": [{"message": {"content": "ok"}}]}
    model = client(monkeypatch, [status_error(502)] * 3 + [status_error(503), status_error(502), done])
    assert model._call({"messages": []}) is done


def test_slow_connection_cuts_end_the_call(monkeypatch):
    cut = openai.APIConnectionError(request=httpx.Request("POST", "http://gateway/v1/chat/completions"))
    model = client(monkeypatch, [cut] * llm.GATEWAY_REJECTIONS)
    with pytest.raises(llm.GatewayRejected):
        model._call({"messages": []})
