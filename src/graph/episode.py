"""Per-episode bipartite graph: each request's provider input tokens split over the prompt components.

Xrg[r, g] is the number of input tokens of component g (system, tool_schemas, task, skill, history, tool:<class>) in
request r; every row sums exactly to the provider count. A call's exposure is its output's tokens summed over the later
requests, which re-send it.
"""
import hashlib, json, math, os, pickle, re, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.utils import layout
from src.utils.skill_frame import SKILL_FRAME

# What a tool call does, from its tool name and arguments (classify, bash_class): state-changing actions, lookups,
# tests, and the shell programs that read or write.
ACTION = re.compile(r"^(update|cancel|book|exchange|return|modify|transfer|refuel|enable|disable|send_(?!message)|create"
                    r"|delete|reset|pay|add|remove|set)")
LOOKUP = re.compile(r"^(get|find|search|list|check|lookup)")
TEST = re.compile(r"pytest|unittest|\btest(s)?\b|cargo test|go test|npm test|make check")
PY_STATE = re.compile(r"\.(post|put|patch|delete)\(|\b(send|create|update|delete|insert)_\w+")
BASH_READ = set("cat head tail grep rg ls find wc less file stat which sed awk tree du df env pwd echo printf diff".split())
BASH_WRITE = set("tee mv cp rm mkdir touch chmod ln apt-get apt pip pip3 npm".split())
EXT = {"code": "py js ts c h cpp rs go java sh rb", "doc": "md txt rst html pdf",
       "data": "json csv yaml yml xml tsv jsonl toml", "log": "log out err"}
SINK = {"act", "msg", "write", "edit", "bash.write"}
CONV = ("start_conversation", "end_conversation", "log_verification", "get_current_time", "get_assistant_tool_schemas")
# The components of a request's input: the fixed prefix and the context items, split by length at CHARS_PER_TOKEN.
FIXED = ("system", "tool_schemas", "task", "skill")
CTX = {f"context:{n}": n for n in ("history", "task", "system", "tool_schemas")}
CHARS_PER_TOKEN = 4.0
# Native request records and skill frames use namespaced wire formats.
RECORD_SCHEMA = re.compile(r"[A-Za-z0-9_.-]+-native-request-record-v1")
# keys of a request item that carry no content
STRUCTURAL = frozenset({"role", "type", "id", "tool_call_id", "call_id", "tool_use_id", "status", "index"})
# parsed episodes, cached by evaluation file
CACHE = layout.CACHE / "workload-parse"
EXTRA_RUN_ROOTS = []  # optional roots holding runs as <arm>/<model>/opencode/<bench>/<run>


def bash_class(cmd):
    """bash.test, bash.write, bash.read or bash.other, judged by the first program that is not setup."""
    segs = [s.strip() for s in re.split(r"&&|;|\|\|", cmd) if s.strip() and not re.match(r"(cd|set|export)\b|\w+=", s.strip())]
    words = (segs[0] if segs else cmd.strip()).split()
    first = words[0] if words else ""
    if TEST.search(cmd):
        return "bash.test"
    if re.search(r"(?<![0-9&>])>{1,2}\s*(?!/dev/null|&)\S", cmd) or re.search(r"\bsed\s+-i", cmd) or first in BASH_WRITE:
        return "bash.write"
    if first in BASH_READ and not re.search(r"\|\s*python", cmd):
        return "bash.read"
    return "bash.other"


def classify(tool, args):
    """(component, changes state) of a tool call."""
    name = tool.split("_", 1)[1] if tool.startswith("tau3-runtime_") else tool
    try:
        a = json.loads(args)
    except Exception:
        a = {}
    if name in ("call_discoverable_agent_tool", "unlock_discoverable_agent_tool"):
        inner = str(a.get("agent_tool_name", "")) if isinstance(a, dict) else ""
        return "discover", name.startswith("call") and bool(ACTION.match(inner))
    if name == "bash":
        cmd = str(a.get("command", ""))
        comp = bash_class(cmd)
        return comp, comp in SINK or bool(comp == "bash.other" and PY_STATE.search(cmd))
    if name == "read":
        ext = Path(str(a.get("filePath", ""))).suffix.lstrip(".").lower()
        comp = "read." + next((c for c, e in EXT.items() if ext in e.split()), "other")
    elif name in ("KB_search", "grep", "glob", "edit", "write"):
        comp = name
    else:
        comp = ("msg" if name == "send_message_to_user" else "conv" if name in CONV else "act" if ACTION.match(name)
                else "lookup" if LOOKUP.match(name) else "todo" if name == "todowrite" else "other")
    return comp, comp in SINK


def largest_remainder(total, weights):
    """Integers proportional to weights that sum to total; ties go to the earlier position."""
    quotas = [total * w / sum(weights) for w in weights]
    out = [math.floor(q) for q in quotas]
    for i in sorted(range(len(weights)), key=lambda i: (-(quotas[i] - out[i]), i))[:total - sum(out)]:
        out[i] += 1
    return out


def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in STRUCTURAL:
                yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def _parse_request(request):
    """({(kind, name): estimated tokens}, {call id: tool name}); kind is "component" or "call"."""
    weights, names = {}, {}

    def add(key, value):
        for text in _texts(value):
            pos = 0
            frames = [m for m in SKILL_FRAME.finditer(text) if m.group("path") is not None]
            for m in [*frames, None]:
                end = m.start() if m else len(text)
                if text[pos:end]:
                    weights[key] = weights.get(key, 0.0) + len(text[pos:end]) / CHARS_PER_TOKEN
                if m:
                    skill = ("component", "skill:" + m.group("path").replace('\\"', '"'))
                    weights[skill] = weights.get(skill, 0.0) + len(m.group(0)) / CHARS_PER_TOKEN
                    pos = m.end()

    def ctx(name):
        return ("component", "context:" + name)

    cfg = request.get("config") if isinstance(request.get("config"), dict) else {}
    for key in ("system", "instructions"):
        if cfg.get(key):
            add(ctx("system"), cfg[key])
    if cfg.get("tools"):
        text = json.dumps(cfg["tools"], ensure_ascii=False, sort_keys=True)
        weights[ctx("tool_schemas")] = weights.get(ctx("tool_schemas"), 0.0) + len(text) / CHARS_PER_TOKEN
    items = request.get("input")
    items = [{"role": "user", "content": items}] if isinstance(items, str) else items
    answered = False
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            add(ctx("history" if answered else "task"), item)
            continue
        role, kind = item.get("role"), item.get("type")
        if role in ("system", "developer"):
            add(ctx("system"), item.get("content"))
        elif role == "tool" and item.get("tool_call_id") is not None:
            add(("call", str(item["tool_call_id"])), item.get("content"))
        elif kind in ("function_call_output", "custom_tool_call_output") and item.get("call_id") is not None:
            add(("call", str(item["call_id"])), item.get("output"))
        elif kind in ("function_call", "custom_tool_call") or role == "assistant" or kind == "reasoning":
            if kind in ("function_call", "custom_tool_call"):
                if item.get("call_id") is not None and item.get("name"):
                    names[str(item["call_id"])] = str(item["name"])
            else:
                for call in item.get("tool_calls") or []:
                    function = (call or {}).get("function") or {}
                    if call.get("id") and function.get("name"):
                        names[str(call["id"])] = str(function["name"])
                content = item.get("content")
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") and block.get("name"):
                        names[str(block["id"])] = str(block["name"])
            add(ctx("history"), item)
            answered = True
        else:
            content = item.get("content")
            for block in content if isinstance(content, list) else [content]:
                if isinstance(content, list) and isinstance(block, dict) and block.get("type") == "tool_result" \
                        and block.get("tool_use_id") is not None:
                    add(("call", str(block["tool_use_id"])), block.get("content"))
                else:
                    add(ctx("history" if answered else "task"), block)
    return weights, names


@lru_cache(maxsize=8192)
def _parse_record(path, stamp):
    """_parse_request of the record at path, or None when it is unusable; stamp only keys the cache."""
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    schema = record.get("schema") if isinstance(record, dict) else None
    request = record.get("request") if isinstance(schema, str) and RECORD_SCHEMA.fullmatch(schema) else None
    return _parse_request(request) if isinstance(request, dict) and isinstance(request.get("input"), (list, str)) else None


def allocate(path, input_tokens, tool_components):
    """[(component, tokens)] of one request summing to input_tokens, or None when it cannot be split."""
    if input_tokens is None or isinstance(input_tokens, float) and not input_tokens.is_integer():
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    parsed = _parse_record(path, (st.st_mtime_ns, st.st_size))
    if parsed is None:
        return None
    weights, names = parsed
    merged = {}
    for (kind, name), w in weights.items():
        if kind == "call":
            name = tool_components.get(name) or ("tool:" + names[name] if name in names else "tool:unknown")
        merged[name] = merged.get(name, 0.0) + w
    keys = [k for k, w in merged.items() if w > 0]
    if not keys:
        return None
    return list(zip(keys, largest_remainder(int(input_tokens), [merged[k] for k in keys])))


# request directories in the exported layout <model>/<method>/<bench>[/<run>]/execution/evaluations/..., and the same
# under the old <arm>/<model>/opencode/<bench>/... layout for EXTRA_RUN_ROOTS
_RUN_GLOBS = (("*/*/*/execution/evaluations/*/*/*", "*/*/opencode/*/execution/evaluations/*/*/*"),
              ("*/*/*/*/execution/evaluations/*/*/*", "*/*/opencode/*/*/execution/evaluations/*/*/*"),
              ("*/*/*/*/execution/evaluations/*/*/*/*", "*/*/opencode/*/*/execution/evaluations/*/*/*/*"))
_index = {}


def index():
    """{request directory name: its directories under the recorded runs}, built once."""
    if not _index:
        for new, old in _RUN_GLOBS:
            found = set(layout.EXECUTION.glob(new)).union(*(root.glob(old) for root in EXTRA_RUN_ROOTS))
            for d in map(str, sorted(found)):
                if "_superseded" not in d and re.search(r"-r\d+-[0-9a-f]{12}$", d):
                    _index.setdefault(os.path.basename(d), []).append(d)
    return _index


def resolve(path, anchor=None):
    """The request record at a recorded path, else under the current artifacts root, else found by its directory name."""
    p = Path(path)
    if p.exists():
        return p
    if "assets/artifacts/" in path:
        q = layout.ARTIFACTS / path.split("assets/artifacts/", 1)[1]
        if q.exists():
            return q
    m = re.search(r"/([^/]+-r\d+-[0-9a-f]{12})/requests/([^/]+)$", path)
    for d in index().get(m[1], []) if m else []:
        if (q := Path(d) / "requests" / m[2]).exists():
            return q
    return p


@dataclass
class Episode:
    task: str
    success: bool | None
    R: int
    req_in: np.ndarray  # provider input tokens per request
    comps: list  # column names of Xrg
    Xrg: np.ndarray  # [R, G] tokens of each component in each request
    calls: list  # (request index, component, executed calls in that request, exposure) per executed call

    @property
    def X(self): return float(self.req_in.sum())

    @property
    def N(self): return len(self.calls)

    def N_g(self):
        out = {}
        for c in self.calls:
            out[c[1]] = out.get(c[1], 0) + 1
        return out


def _episode(args):
    """Episode of one recorded episode dict, or None when a request cannot be split."""
    ep, path = args
    reqs, seen = [], set()
    for r in ep["requests"]:
        if r["request_id"] not in seen:
            seen.add(r["request_id"])
            reqs.append(r)
    tc, executed = {}, []
    for k, r in enumerate(reqs):
        for c in r.get("tool_calls", []):
            native = c["call_id"].removeprefix(r["request_id"] + ":")
            comp = classify(c["tool"], c.get("arguments") or "")[0]
            tc[native] = f"tool:{comp}|{native}"
            if c.get("executed") is not False:
                executed.append((k, f"tool:{comp}", native))
    rows = []
    for r in reqs:
        shares = allocate(str(resolve(r["record_path"], Path(path))), r["input_tokens"], tc)
        if shares is None:
            return None
        row = defaultdict(float)
        for comp, tokens in shares:
            row[CTX.get(comp) or ("skill" if comp.startswith("skill:") else comp)] += tokens
        rows.append(row)
    calls = [(k, g, sum(e[0] == k for e in executed), sum(row.get(f"{g}|{native}", 0.0) for row in rows[k + 1:]))
             for k, g, native in executed]
    names = sorted({n.split("|")[0] for row in rows for n in row})
    X = np.zeros((len(reqs), len(names)))
    for i, row in enumerate(rows):
        for n, v in row.items():
            X[i, names.index(n.split("|")[0])] += v
    valid = ep.get("outcome_valid", True) and "log_error" not in str(ep.get("outcome_reason"))
    return Episode(ep["task_id"], ep["success"] if valid else None, len(reqs),
                   np.array([r["input_tokens"] for r in reqs], float), names, X, calls)


def load(path, ids=None, workers=24):
    """Episodes of an evaluation.json that have requests (optionally only the task ids), cached per file and mtime."""
    path = Path(path).resolve()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (hashlib.sha1(str(path).encode()).hexdigest()[:16] + "-v5.pkl")
    key = path.stat().st_mtime_ns
    if cache.exists() and (got := pickle.loads(cache.read_bytes()))[0] == key:
        eps = got[1]
    else:
        raw = [e for e in json.loads(path.read_text("utf-8"))["episodes"] if e["requests"]]
        with ProcessPoolExecutor(workers) as pool:
            eps = [e for e in pool.map(_episode, [(e, path) for e in raw], chunksize=2) if e is not None]
        with tempfile.NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as f:
            f.write(pickle.dumps((key, eps)))
        os.replace(f.name, cache)
    return [e for e in eps if ids is None or e.task in ids]


def moments(e):
    """(R, N, X, F, h, s_F): F the fixed prefix per request, h the history growth per request, s_F = R F / X; None if degenerate."""
    F = float(e.Xrg[:, [i for i, c in enumerate(e.comps) if c in FIXED]].sum(1).mean())
    hist = e.X - e.R * F
    ok = e.R >= 2 and e.N >= 1 and F > 0 and hist > 0
    return (e.R, e.N, e.X, F, hist / (e.R * (e.R - 1) / 2), e.R * F / e.X) if ok else None
