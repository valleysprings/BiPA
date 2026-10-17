"""Step 1 of the bipartite graph: split each model request's input tokens over the parts of its prompt.

A provider reports one exact input-token count X per request. This module attributes that count to the parts of the
prompt that produced it: the system prompt, the tool schemas, the task, the skill frame (a <bach-skill path=... sha256=...>
block, its own component skill:<path>), the running history, and every earlier tool output, keyed by its call id and
credited to the component of the call that produced it. Each part is weighted by its estimated content tokens
(characters / CHARS_PER_TOKEN) and the shares are rounded by largest remainder, so the parts add up to X exactly.

Connections: graph.episode calls allocate() once per request of an episode and turns the returned rows into the
request x component matrix (R x G) that the rest of BACH reads. The request record is the native JSON written next to the rollout (schema RECORD_SCHEMA); without a usable
record allocate() returns (None, reason) and the caller keeps the request under context:unknown.
Self-contained: stdlib + graph._config (the frozen graph section of the method config)."""

from __future__ import annotations

import functools
import json
import math
import os
import re

from src.graph import _config as config

# Characters per estimated token (graph/graph.yaml, graph.allocate.chars_per_token). Only the relative weights matter,
# because the shares are rescaled to the exact provider count.
CHARS_PER_TOKEN = float(config.load("bach")["graph"]["allocate"]["chars_per_token"])
RECORD_SCHEMA = "bach-native-request-record-v1"
# The skill frame the harness wraps around the skill text inside a message: group 2 is the skill path.
_SKILL_FRAME = re.compile(r'<bach-skill path=(\\*")(.+?)\1 sha256=[^>\s]*>.*?</bach-skill>', re.S)
# Keys of a message that carry bookkeeping, not prompt content; their strings are not counted.
_STRUCTURAL = frozenset({"role", "type", "id", "tool_call_id", "call_id", "tool_use_id", "status", "index"})


def estimate_tokens(text: str) -> float:
    """Estimated token count of a text: characters / CHARS_PER_TOKEN. Used only as a relative weight (and as tau(s))."""
    return len(text) / CHARS_PER_TOKEN


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Integers proportional to ``weights`` summing exactly to ``total`` (ties to the earlier position).

    Each part gets floor(quota); the leftover units go to the parts with the largest fractional remainders.
    """
    quotas = [total * weight / sum(weights) for weight in weights]
    result = [math.floor(quota) for quota in quotas]
    for index in sorted(range(len(weights)), key=lambda i: (-(quotas[i] - result[i]), i))[:total - sum(result)]:
        result[index] += 1
    return result


def _texts(value):
    """Yield every content string inside a message value (nested dicts and lists), skipping the _STRUCTURAL keys."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in _STRUCTURAL:
                yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def _parse_request(request: dict) -> dict:
    """Estimated content tokens per part of one provider request.

    Input: the request dict of a native record (config.system / config.tools / input items).
    Output: {"weights": [(part, estimated tokens), ...], "tool_names": {call id: tool name}}. A part is
    ("component", name, entrypoint) for system / tool schemas / task / history / skill frames, or ("call", call id,
    "tool_output") for the output of an earlier tool call (allocate() maps the call id to a component later).
    """
    weights, names = {}, {}

    def add(key, value):
        # Add the text of `value` to part `key`; a skill frame inside the text is cut out and counted as its own part.
        for text in _texts(value):
            position = 0
            for match in [*_SKILL_FRAME.finditer(text), None]:
                end = match.start() if match else len(text)
                if text[position:end]:
                    weights[key] = weights.get(key, 0.0) + estimate_tokens(text[position:end])
                if match:
                    skill = ("component", "skill:" + match.group(2).replace('\\"', '"'), "skill_input")
                    weights[skill] = weights.get(skill, 0.0) + estimate_tokens(match.group(0))
                    position = match.end()

    def context(name):
        # Part key of a context component (context:system, context:task, context:history, context:tool_schemas).
        return ("component", "context:" + name, "input")

    config = request.get("config") if isinstance(request.get("config"), dict) else {}
    for key in ("system", "instructions"):
        if config.get(key):
            add(context("system"), config[key])
    if config.get("tools"):
        tools = context("tool_schemas")
        weights[tools] = weights.get(tools, 0.0) + estimate_tokens(
            json.dumps(config["tools"], ensure_ascii=False, sort_keys=True))
    # Input items: before the first assistant/tool-call item the user content is the task, after it it is history.
    items = request.get("input")
    items = [{"role": "user", "content": items}] if isinstance(items, str) else items
    answered = False
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            add(context("history" if answered else "task"), item)
            continue
        role, kind = item.get("role"), item.get("type")
        if role in ("system", "developer"):
            add(context("system"), item.get("content"))
        elif role == "tool" and item.get("tool_call_id") is not None:
            add(("call", str(item["tool_call_id"]), "tool_output"), item.get("content"))
        elif kind in ("function_call_output", "custom_tool_call_output") and item.get("call_id") is not None:
            add(("call", str(item["call_id"]), "tool_output"), item.get("output"))
        elif kind in ("function_call", "custom_tool_call") or role == "assistant" or kind == "reasoning":
            # Remember which tool each call id belongs to, for outputs whose call is not in tool_components.
            if kind in ("function_call", "custom_tool_call"):
                if item.get("call_id") is not None and item.get("name"):
                    names[str(item["call_id"])] = str(item["name"])
            else:  # chat tool_calls, Anthropic tool_use blocks
                for call in item.get("tool_calls") or []:
                    function = (call or {}).get("function") or {}
                    if call.get("id") and function.get("name"):
                        names[str(call["id"])] = str(function["name"])
                content = item.get("content")
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") \
                            and block.get("name"):
                        names[str(block["id"])] = str(block["name"])
            add(context("history"), item)
            answered = True
        else:
            content = item.get("content")
            for block in content if isinstance(content, list) else [content]:
                if isinstance(content, list) and isinstance(block, dict) and block.get("type") == "tool_result" \
                        and block.get("tool_use_id") is not None:
                    add(("call", str(block["tool_use_id"]), "tool_output"), block.get("content"))
                else:
                    add(context("history" if answered else "task"), block)
    return {"weights": list(weights.items()), "tool_names": names}


@functools.lru_cache(maxsize=8192)
def _parse_record(path: str, stamp: tuple) -> dict | str:
    """Parse the request record at `path` into _parse_request's result, or a reason string if it is unusable.

    Cached per (path, mtime, size): the same record is read for many graphs, a rewritten one is re-read.
    """
    del stamp  # cache key only: a rewritten record is re-read
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return "record_unreadable"
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        return "record_schema_unsupported"
    request = record.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("input"), (list, str)):
        return "record_has_no_request_input"
    return _parse_request(request)


def allocate(request: dict, input_tokens, tool_components: dict[str, str]) -> tuple[list[dict] | None, str | None]:
    """Split one request's exact `input_tokens` over its prompt parts.

    Inputs: `request` (needs "record_path"), the provider's `input_tokens`, and `tool_components`
    (native call id -> component that produced the call). Output: (rows, None) with one row
    {component, entrypoint, tokens, estimated_content_tokens} per part, tokens summing exactly to input_tokens;
    or (None, reason) when the request cannot be split (the caller then books it under context:unknown).
    """
    if input_tokens is None:
        return None, "input_tokens_unknown"
    if isinstance(input_tokens, float) and not input_tokens.is_integer():
        return None, "input_tokens_not_integral"
    path = request.get("record_path")
    if not isinstance(path, str) or not path:
        return None, "no_record"
    try:
        status = os.stat(path)
    except OSError:
        return None, "record_missing"
    parsed = _parse_record(path, (status.st_mtime_ns, status.st_size))
    if isinstance(parsed, str):
        return None, parsed
    # Merge parts that end up in the same component: a tool output goes to the component of its call.
    merged: dict[tuple, float] = {}
    for (kind, name, entrypoint), weight in parsed["weights"]:
        if kind == "call":
            name = tool_components.get(name) or ("tool:" + parsed["tool_names"][name] if name in parsed["tool_names"]
                                                 else "tool:unknown")
        merged[(name, entrypoint)] = merged.get((name, entrypoint), 0.0) + weight
    # Parts without content get no share; the others are rounded by largest remainder to the exact total.
    keys = [key for key, weight in merged.items() if weight > 0]
    if not keys:
        return None, "record_input_empty"
    weights = [merged[key] for key in keys]
    shares = largest_remainder(int(input_tokens), weights)
    return [{"component": component, "entrypoint": entrypoint, "tokens": tokens,
             "estimated_content_tokens": round(weight, 3)}
            for (component, entrypoint), weight, tokens in zip(keys, weights, shares)], None
