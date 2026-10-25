"""The optimizer LLM: streamed chat completions through the OpenAI SDK, retried, recorded in <run>/llm/."""
import json, os, threading, time
from collections import Counter
from pathlib import Path

import httpx
import openai
from dotenv import dotenv_values

from src.layout import HARNESS_BENCH

SUFFIX = {"deepseek-flash": "_2", "gpt-6-luna": "_2"}  # OPENAI_API_KEY<suffix>, OPENAI_BASE_URL<suffix>[_OPTIMIZER]
RETRY = {"initial_delay": 4.0, "max_delay": 600.0, "max_total_wait": 6 * 3600.0}
# Prompts are cut to 512k estimated tokens (characters / 2.5): the text right after the first KEEP_HEAD characters of
# the first user message goes first; system messages are never cut.
MAX_PROMPT_TOKENS, KEEP_HEAD, CHARS_PER_TOKEN = 512_000, 16_000, 2.5
# The gateway answers some request bodies, every time, with a slow empty 502, a slow cut or no answer at all.
GATEWAY_REJECTIONS, GATEWAY_MIN_ELAPSED = 4, 29.0
REJECTED = set()  # requests the gateway rejected in this process: a repeat fails at once


class IncompleteStream(RuntimeError):
    """The stream ended without a finish event."""


class GatewayRejected(RuntimeError):
    """GATEWAY_REJECTIONS rejections in a row: the call fails, so the optimizer's own failure handling runs."""


def count_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


def fit_prompt(messages):
    """Messages within MAX_PROMPT_TOKENS, earliest content dropped first -> (messages, tokens dropped)."""
    def text(content):
        if isinstance(content, str):
            return content
        return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))
    fitted = [dict(m, content=text(m.get("content"))) if isinstance(m.get("content"), list) else dict(m) for m in messages]
    sizes = [count_tokens(m["content"]) if isinstance(m.get("content"), str) else 0 for m in fitted]
    excess = sum(sizes) - MAX_PROMPT_TOKENS
    if excess <= 0:
        return messages, 0
    dropped, head_kept = 0, False
    for message, size in zip(fitted, sizes):
        if excess <= 0:
            break
        if message.get("role") == "system" or not size:
            continue
        content = message["content"]
        head = "" if head_kept else content[:KEEP_HEAD]
        head_kept = head_kept or message.get("role") == "user"
        body = content[len(head):]
        cut = min(len(body), int(len(content) * min(1.0, (excess * 1.02) / size)))
        marker = "\n[... earlier content truncated to fit the context ...]\n" if cut else ""
        message["content"] = head + marker + body[cut:]
        removed = size - count_tokens(message["content"])
        excess, dropped = excess - removed, dropped + removed
    return fitted, dropped


def retryable(error):
    """HTTP 408/409/429/5xx, a timeout, a connection error, a cut stream or an error event in a 200 stream."""
    if isinstance(error, openai.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    transient = (IncompleteStream, openai.APITimeoutError, openai.APIConnectionError, httpx.TransportError)
    return isinstance(error, transient) or type(error) is openai.APIError


def assemble(chunks):
    """A streamed chat completion -> the non-streamed response shape."""
    content, calls, finish, usage = [], {}, None, None
    for chunk in chunks:
        usage = chunk.usage.model_dump() if chunk.usage else usage
        for choice in chunk.choices:
            finish = choice.finish_reason or finish
            content.append(choice.delta.content or "")
            for call in choice.delta.tool_calls or []:
                entry = calls.setdefault(call.index, {"id": "", "type": "function",
                                                      "function": {"name": "", "arguments": ""}})
                entry["id"] += call.id or ""
                entry["function"]["name"] += (call.function and call.function.name) or ""
                entry["function"]["arguments"] += (call.function and call.function.arguments) or ""
    if finish is None:
        raise IncompleteStream("chat stream ended without a finish event")
    message = {"role": "assistant", "content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)] or None}
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


class LLM:
    """``usage`` counts calls and tokens over every process of the run (kept in llm/usage.json)."""

    def __init__(self, model, directory, max_tokens=32768, timeout=600):
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.folder, self.lock = Path(directory) / "llm", threading.Lock()
        usage = self.folder / "usage.json"
        self.usage = Counter(json.loads(usage.read_text()) if usage.is_file() else {})
        self.calls = max((int(path.stem[len("call-"):]) for path in self.folder.glob("call-*.json")), default=0)
        env, suffix = {**dotenv_values(HARNESS_BENCH / ".env"), **os.environ}, SUFFIX[model]
        url = env.get(f"OPENAI_BASE_URL{suffix}_OPTIMIZER") or env[f"OPENAI_BASE_URL{suffix}"]
        if not url.startswith("https://"):  # the host's proxy cuts some plaintext request bodies
            raise SystemExit(f"the optimizer URL {url} is not https: set OPENAI_BASE_URL{suffix}_OPTIMIZER "
                             "to the gateway's https endpoint")
        self.client = openai.OpenAI(api_key=env[f"OPENAI_API_KEY{suffix}"], base_url=url, timeout=timeout, max_retries=0)

    def _count(self, **counts):
        with self.lock:
            self.usage.update(counts)
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "usage.tmp").write_text(json.dumps(self.usage))
            (self.folder / "usage.tmp").replace(self.folder / "usage.json")

    def _record(self, request, response, seconds):
        with self.lock:
            self.calls += 1
            path = self.folder / f"call-{self.calls:04d}.json"
        self.folder.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_text(json.dumps({"request": request, "response": response, "seconds": seconds},
                                                       ensure_ascii=False, indent=1))
        path.with_suffix(".tmp").replace(path)

    def _send(self, request, timeout):
        if self.model == "gpt-6-luna" and request.get("tools"):  # the gateway rejects tools at Luna's default reasoning
            request = {**request, "reasoning_effort": "none"}
        completions = self.client.with_options(timeout=timeout).chat.completions
        return assemble(completions.create(**request, stream=True, stream_options={"include_usage": True}))

    def _call(self, request):
        """The response, retried with exponential backoff within RETRY's time budget."""
        key, retries, rejections = json.dumps(request, sort_keys=True, default=str), 0, 0
        if key in REJECTED:
            self._count(gateway_rejected=1)
            raise GatewayRejected("the gateway rejected this request earlier in this process")
        deadline = time.monotonic() + RETRY["max_total_wait"]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("optimizer LLM retry time budget exhausted")
            started = time.monotonic()
            try:
                return self._send(request, min(self.timeout, remaining))
            except Exception as error:
                delay = min(RETRY["max_delay"], RETRY["initial_delay"] * 2 ** min(retries, 20))
                elapsed, status = time.monotonic() - started, getattr(error, "status_code", None)
                slow = elapsed >= GATEWAY_MIN_ELAPSED
                rejected = (status == 502 and slow or type(error).__name__ == "APITimeoutError"
                            or type(error).__name__ == "APIConnectionError" and slow)
                rejections = rejections + 1 if rejected else rejections if status is None else 0
                if retryable(error) and rejections >= GATEWAY_REJECTIONS:
                    REJECTED.add(key)
                    self._count(gateway_rejected=1)
                    raise GatewayRejected(f"the gateway rejected the request {rejections} times in a row") from error
                if not retryable(error) or delay >= deadline - time.monotonic():
                    raise
                time.sleep(delay)
                retries += 1

    def complete(self, messages, tools=None, temperature=1.0, max_tokens=None):
        """One chat completion -> (text, tool_calls [{id, type, function: {name, arguments}}], usage)."""
        messages, dropped = fit_prompt(messages)
        if dropped:
            self._count(truncated_calls=1, truncated_tokens=dropped)
        request = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens or self.max_tokens, **({"tools": tools} if tools else {})}
        started = time.monotonic()
        response = self._call(request)
        self._record(request, response, round(time.monotonic() - started, 3))
        message, usage = response["choices"][0]["message"], response.get("usage") or {}
        self._count(calls=1, replayed=0, prompt_tokens=usage.get("prompt_tokens") or 0,
                    completion_tokens=usage.get("completion_tokens") or 0)
        return message.get("content") or "", message.get("tool_calls") or [], usage
