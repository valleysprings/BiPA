"""Optimizer LLM transport through the OpenAI SDK, with durable call records in ``<dir>/llm/``.

Native-v1 disables answer replay: resumed optimizer calls are fresh. Legacy mode can replay the k-th stored
answer for the k-th occurrence of a request hash. Proposal and acceptance decisions belong to upstream optimizers.
"""

import fcntl, hashlib, json, os, re, tempfile, threading, time, uuid, warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from dotenv import dotenv_values

from bipa.layout import HARNESS_BENCH

ENV_FILE = HARNESS_BENCH / ".env"
ENDPOINTS = {"deepseek-flash": "_2", "glm-5.3-flash": "_2", "gpt-6-luna": "_2"}  # boyue gateway env suffix (Luna on boyue from 2026-10-07, lead)
# Optimizer calls leave this pod only through the local Clash proxy, which cuts plaintext HTTP carrying BitTorrent words (announce, torrent,
# info_hash, ...): boyue's http URL in .env lost those calls as slow empty 502s. Its https name is the same gateway (35.220.164.252) and hides
# the body. Agents in the sandboxes keep .env's URL (direct, no proxy). 2026-10-08.
BASE_URLS = {"_2": "https://api.boyuerichdata.opensphereai.com/v1"}
RETRY = {"initial_delay": 4.0, "max_delay": 600.0, "max_total_wait": 6 * 3600.0}  # exponential, 10 min cap, 6 h budget
# The model takes at most 1M tokens and the gateway cuts a request whose first byte takes over 30 s (~1M-token
# prompts), so every optimizer prompt is fitted to 512k tokens by dropping its earliest content. The opening
# instructions (the first KEEP_HEAD chars of the first user message: task and current skill) always stay.
MAX_PROMPT_TOKENS, KEEP_HEAD = 512_000, 16_000

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("o200k_base")
    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text, disallowed_special=()))
except ImportError:
    def count_tokens(text: str) -> int:
        return int(len(text) / 2.5)


def fit_prompt(messages: list) -> tuple[list, int]:
    """Messages within MAX_PROMPT_TOKENS, earliest content dropped first -> (messages, tokens dropped)."""
    def text(content):  # a list content (text parts) counts by its text, as the provider does
        return content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))
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
        text = message["content"]
        head = "" if head_kept else text[:KEEP_HEAD]
        head_kept = head_kept or message.get("role") == "user"
        body = text[len(head):]
        cut = min(len(body), int(len(text) * min(1.0, (excess * 1.02) / size)))
        message["content"] = head + ("\n[... earlier content truncated to fit the context ...]\n" if cut else "") + body[cut:]
        removed = size - count_tokens(message["content"])
        excess, dropped = excess - removed, dropped + removed
    return fitted, dropped


class IncompleteStream(RuntimeError): ...  # the stream ended without a finish event: a transport cut, retried


class GatewayRejected(RuntimeError):
    """The boyue gateway answers some request contents (e.g. a τ³ banking security-tips passage) with an empty 502
    after ~30 s (52-101 s through the machine proxy, or no answer until the client timeout), every time. After GATEWAY_REJECTIONS such answers in a row (a
    connection error in between does not reset the count) the call fails, so the optimizer's own
    failure handling runs (GEPA skips the iteration, SkillOpt drops that analyst minibatch) instead of a 6 h retry."""


GATEWAY_REJECTIONS, GATEWAY_MIN_ELAPSED = 4, 29.0  # consecutive 502s that took at least this long
REJECTED = set()  # digests rejected in this process: the rejection is deterministic, so a repeat (an optimizer's own retry) fails at once


def error_metadata(error) -> dict:
    """Only bounded identifiers; never exception text, request headers or error bodies."""
    result = {"error_class": type(error).__name__}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        result["status_code"] = status
    code = getattr(error, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", code):
        result["error_code"] = code
    return result


class AttemptLedger:
    """Append-only, process-safe metadata. An unmatched start witnesses an interrupted attempt."""
    def __init__(self, directory):
        self.path = Path(directory) / "llm/attempts.jsonl"

    def append(self, **event):
        event = {"schema": "optimizer-llm-attempt-v1", "pid": os.getpid(),
                 "recorded_at": datetime.now(timezone.utc).isoformat(), **event}
        # Ledger failures must not change the existing transport/retry decision.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            warnings.warn("optimizer attempt ledger write failed: " + type(error).__name__, RuntimeWarning)


def digest(value) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def retryable(error) -> bool:
    """A transient failure: HTTP 408/409/429/5xx, timeout, connection, stream cut, error event in a 200 stream."""
    if isinstance(error, openai.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    transient = (IncompleteStream, openai.APITimeoutError, openai.APIConnectionError, httpx.TransportError)
    return isinstance(error, transient) or type(error) is openai.APIError


def assemble(chunks) -> dict:
    """A streamed chat completion -> the non-streamed response shape. The gateway cuts a non-streamed request after
    about 30 s, so every call streams."""
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
    message = {"role": "assistant", "content": "".join(content),
               "tool_calls": [calls[i] for i in sorted(calls)] or None}
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


class CallRecorder:
    def __init__(self, directory, *, attempt_ledger=None):
        self.folder, self.lock, self.answers, self.served = Path(directory) / "llm", threading.Lock(), {}, Counter()
        self.attempt_ledger = attempt_ledger
        self.count = 0
        for path in sorted(self.folder.glob("call-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            self.count = max(self.count, int(path.stem.split("-")[-1]))
            if "response" in record:
                self.answers.setdefault(record["key"], []).append(record)

    def replay(self, key: str) -> dict | None:
        with self.lock:
            if self.served[key] < len(self.answers.get(key, [])):
                self.served[key] += 1
                return self.answers[key][self.served[key] - 1]

    def record(self, key: str, request: dict, **fields) -> None:
        publication_id = uuid.uuid4().hex
        if self.attempt_ledger:
            self.attempt_ledger.append(event="record_started", publication_id=publication_id, request_digest=key)
        try:
            self._record(key, request, **fields)
        except BaseException as error:
            if self.attempt_ledger:
                self.attempt_ledger.append(event="record_error", publication_id=publication_id,
                                           request_digest=key, **error_metadata(error))
            raise
        if self.attempt_ledger:
            self.attempt_ledger.append(event="record_published", publication_id=publication_id, request_digest=key)

    def _record(self, key: str, request: dict, **fields) -> None:
        with self.lock:
            record = {"key": key, "request": request, **fields, "recorded_at": datetime.now(timezone.utc).isoformat()}
            self.folder.mkdir(parents=True, exist_ok=True)
            # Separate recorder instances/processes share this lock. Publish a complete
            # temp file via exclusive hard link: readers never see partial JSON, and
            # an existing record can never be replaced, even with numbering gaps.
            with (self.folder / ".recorder.lock").open("a") as allocation:
                fcntl.flock(allocation, fcntl.LOCK_EX)
                numbers = [int(p.stem.split("-")[-1]) for p in self.folder.glob("call-*.json")]
                self.count = max([self.count, *numbers], default=0) + 1
                descriptor, temporary = tempfile.mkstemp(prefix=".call-", suffix=".tmp", dir=self.folder)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, indent=1))
                        stream.flush()
                        os.fsync(stream.fileno())
                    while True:
                        try:
                            os.link(temporary, self.folder / f"call-{self.count:04d}.json")
                            break
                        except FileExistsError:
                            self.count += 1
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if "response" in fields:  # a later identical request is a new occurrence
                self.answers.setdefault(key, []).append(record)
                self.served[key] += 1


class LLM:
    """``model``: deepseek-flash | gpt-6-luna | glm-5.3-flash; ``directory``: where the call records go (None: none).
    ``usage`` sums calls and tokens, replayed calls included."""

    def __init__(self, model: str, directory=None, *, max_tokens: int = 32768, stream: bool = True, timeout: int = 600,
                 attempt_ledger: bool | None = None, answer_replay: bool | None = None):
        self.model, self.max_tokens, self.stream = model, max_tokens, stream
        self.timeout = timeout
        self.answer_replay = os.environ.get("HARNESS_LLM_ANSWER_REPLAY") != "0" if answer_replay is None else answer_replay
        enabled = os.environ.get("HARNESS_LLM_ATTEMPT_LEDGER") == "1" if attempt_ledger is None else attempt_ledger
        self.attempt_ledger = AttemptLedger(directory) if enabled and directory is not None else None
        self.recorder = CallRecorder(directory, attempt_ledger=self.attempt_ledger) if directory is not None else None
        self.usage, self.lock = Counter(), threading.Lock()
        env, suffix = {**dotenv_values(ENV_FILE), **os.environ}, ENDPOINTS[model]
        self.client = openai.OpenAI(api_key=env[f"OPENAI_API_KEY{suffix}"], base_url=BASE_URLS.get(suffix, env[f"OPENAI_BASE_URL{suffix}"]),
                                    timeout=timeout, max_retries=0)

    def _send(self, request: dict, *, timeout: float | None = None) -> dict:
        client = self.client if timeout is None else self.client.with_options(timeout=timeout)
        completions = client.chat.completions
        if self.model == "gpt-6-luna" and ENDPOINTS[self.model] == "_2" and request.get("tools"):  # boyue rejects tools with its default reasoning on Chat Completions; sent only, not part of the replay key
            request = {**request, "reasoning_effort": "none"}
        if not self.stream:
            return completions.create(**request).model_dump(mode="json")
        # The raw stream, assembled here: the SDK's stream helper takes only strict tools and raises on a length cut,
        # where a non-streamed call returns the truncated text (what the optimizers expect)
        return assemble(completions.create(**request, stream=True, stream_options={"include_usage": True}))

    def _call(self, request: dict) -> dict:
        retries, rejections = 0, 0
        deadline = time.monotonic() + RETRY["max_total_wait"]
        ledger = getattr(self, "attempt_ledger", None)
        call_id, key = uuid.uuid4().hex if ledger else None, digest(request)
        if key in REJECTED:
            with self.lock:
                self.usage.update(gateway_rejected=1)
            raise GatewayRejected("the gateway rejected this request earlier in this process")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if ledger:
                    ledger.append(call_id=call_id, request_digest=key, event="terminated", reason="retry_time_budget_exhausted")
                raise TimeoutError("optimizer LLM retry time budget exhausted")
            attempt_id, started = uuid.uuid4().hex, time.monotonic()
            metadata = {"call_id": call_id, "attempt_id": attempt_id, "request_digest": key, "attempt": retries + 1}
            if ledger:
                ledger.append(**metadata, event="started", timeout_seconds=min(self.timeout, remaining))
            try:
                response = self._send(request, timeout=min(self.timeout, remaining))
                if ledger:
                    ledger.append(**metadata, event="result", outcome="success",
                                  elapsed_seconds=round(time.monotonic() - started, 6), terminal_reason="success")
                return response
            except Exception as error:
                delay = min(RETRY["max_delay"], RETRY["initial_delay"] * 2 ** min(retries, 20))
                remaining = deadline - time.monotonic()
                transient = retryable(error)
                elapsed = time.monotonic() - started
                status = getattr(error, "status_code", None)
                gateway = (status == 502 and elapsed >= GATEWAY_MIN_ELAPSED) or type(error).__name__ == "APITimeoutError"  # or no answer
                # until the client timeout (the same rejected request hangs for 600 s through the machine proxy)
                # or, since 2026-10-12, the machine proxy cutting the connection after ~54 s (a terminal-bench analyst prompt, 39 attempts in 6 h)
                gateway = gateway or (type(error).__name__ == "APIConnectionError" and elapsed >= GATEWAY_MIN_ELAPSED)
                rejections = rejections + 1 if gateway else rejections if status is None else 0  # no status: connection error
                terminal = ("nonretryable" if not transient
                            else "gateway_rejected" if rejections >= GATEWAY_REJECTIONS
                            else "retry_time_budget_exhausted" if delay >= remaining else None)
                if ledger:
                    ledger.append(**metadata, event="result", outcome="error", **error_metadata(error),
                                  elapsed_seconds=round(time.monotonic() - started, 6),
                                  next_delay_seconds=None if terminal else delay, terminal_reason=terminal)
                if terminal == "gateway_rejected":
                    REJECTED.add(key)
                    with self.lock:
                        self.usage.update(gateway_rejected=1)
                    raise GatewayRejected(f"gateway returned a slow 502 {rejections} times in a row") from error
                if terminal:
                    raise
                time.sleep(delay)
                retries += 1
            except BaseException as error:
                if ledger:
                    ledger.append(**metadata, event="result", outcome="interrupted", **error_metadata(error),
                                  elapsed_seconds=round(time.monotonic() - started, 6), terminal_reason="interrupted")
                raise

    def complete(self, messages: list, tools: list | None = None, temperature: float = 1.0,
                 max_tokens: int | None = None) -> tuple[str, list, dict]:
        """One chat completion -> (text, tool_calls [{id, type, function: {name, arguments}}], usage)."""
        messages, dropped = fit_prompt(messages)
        if dropped:
            with self.lock:
                self.usage.update(truncated_calls=1, truncated_tokens=dropped)
        request = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens or self.max_tokens, **({"tools": tools} if tools else {})}
        key, started = digest(request), time.monotonic()
        recorded = self.recorder.replay(key) if self.recorder is not None and getattr(self, "answer_replay", True) else None
        response = recorded["response"] if recorded is not None else self._call(request)
        if recorded is None and self.recorder is not None:
            self.recorder.record(key, request, response=response, seconds=round(time.monotonic() - started, 3))
        message, usage = response["choices"][0]["message"], response.get("usage") or {}
        with self.lock:
            self.usage.update(calls=1, replayed=int(recorded is not None),
                              prompt_tokens=usage.get("prompt_tokens") or 0,
                              completion_tokens=usage.get("completion_tokens") or 0)
        return message.get("content") or "", message.get("tool_calls") or [], usage

    def __call__(self, prompt: str) -> str:
        """A plain-prompt callable (GEPA's reflection_lm)."""
        return self.complete([{"role": "user", "content": prompt}])[0]
