"""The bipartite graph of an episode: which part of the prompt each model request paid for (paper Sec. 3.1 and 4.1).

What an episode is.  The agent talks to the model R times (R requests, "turns").  The provider bills every request for ALL the
input tokens it receives, because the harness never compacts the conversation: each request re-sends the system prompt, the tool
schemas, the task, the skill, and the whole history so far.  So the cost of an episode is X = the sum over its requests of the
provider's input-token count, and it is mostly re-sent text, not new text.

The graph.  Rows are the R requests.  Columns are the components of the prompt: system, tool_schemas, task, skill, history, and the
tool outputs grouped by tool class (tool:lookup, tool:KB_search, tool:act, ... named by classify below).  Xrg[r, g] is the number of
tokens of component g inside request r.  src/graph/allocate.py splits each request's provider count by estimated content and rounds by
largest remainder, so every row sums exactly to the provider's input-token count of that request (Episode.req_in).  The second
stream is the executed tool calls (Episode.calls), one entry per call that the harness really ran.

Worked example, a real tau3 telecom episode (15 requests, 289,073 input tokens, solved).
    request 0:  17,460 = system 2,120 + tool_schemas 8,828 + task 5,641 + skill 871 + history 0
    request 3:  17,912 = system 2,140 + tool_schemas 8,915 + task 5,696 + skill 879 + history 111 + tool outputs 171
Summed over the 15 requests, the fixed prefix (system + tool_schemas + task + skill) is 268,216 of the 289,073 tokens (93%) and the
tool outputs are 1.8%.  So on this task every extra turn costs about 18k tokens whatever the turn does.

Exposure.  A tool output enters the history once and is then re-sent in every later request.  The input tokens a call costs are
therefore the sum of its output's tokens over the later requests; _episode computes it per call and stores it as Episode.calls[i][3].
Example: get_customer_by_phone, issued at request 2, returns its output once; that output then sits in requests 3 to 14 (12 requests) at 134 to 137 tokens each (each request's
split is rounded against its own exact total), so the call's exposure is 1,640 tokens = Episode.calls[i][3].

What a skill can change, read off the columns of the graph:
    - which calls happen            -> the tool-output columns (the exposure of each call)
    - how many turns there are      -> R, which multiplies the whole prefix
    - its own text                  -> the skill column, which is part of the prefix and is paid at every request

What the analyses read.  load / _episode build the graphs of a run's episodes; classify names the columns; moments gives the
four-term turn-elasticity readout of one episode (X = R F + R (R - 1) / 2 h).  The loop-facing readouts that BACH's optimizer
consumed (observe, paired, split, levers and the soft-score / verifier-detail helpers) stayed with the BACH loop and are not
part of this package.

Two vocabularies for the same call.  episode.classify names the matrix columns (for example "tool:lookup", a coarse class used
for the token accounting); graph/costmodel.behavior names the rows of the cost table (for example "get_customer_by_phone", a fine
class used to price edits).  Same call, two labels.
"""

import hashlib, json, os, pickle, re, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from src.graph import allocate as H
from src.graph import _config as config
from src.core.layout import ARTIFACTS, CACHE as ASSET_CACHE  # historical records keep their old roots

# ---- component of a tool call -------------------------------------------------------------------------------------------------------------------------

# the vocabulary of tool-call components lives in configs/graph.yaml (graph.classify)
GRAPH = config.load("bach")["graph"]
ACTION = re.compile(GRAPH["classify"]["action"])          # tool names that change state
LOOKUP = re.compile(GRAPH["classify"]["lookup"])          # tool names that read state
BASH_READ = set(GRAPH["classify"]["bash_read"])           # shell programs that only read
BASH_WRITE = set(GRAPH["classify"]["bash_write"])         # shell programs that write
TEST = re.compile(GRAPH["classify"]["test"])              # shell commands that run tests
PY_STATE = re.compile(GRAPH["classify"]["py_state"])      # python one-liners that change state
EXT = GRAPH["classify"]["ext"]                            # file extension -> read.<kind>
VOCAB = GRAPH["classify"]["vocab"]                        # every component name, in report order
SINK = set(GRAPH["classify"]["sink"])                     # components that change state


def bash_class(cmd: str) -> str:
    """Component of a shell command: bash.test, bash.write, bash.read or bash.other (looked at by its first real program)."""
    # split the command on && ; || and drop the setup segments (cd, set, export, VAR=value) that say nothing about what it does
    segs = [s.strip() for s in re.split(r"&&|;|\|\|", cmd) if s.strip() and not re.match(r"(cd|set|export)\b|\w+=", s.strip())]
    # the first remaining segment decides; its first word is the program
    seg = segs[0] if segs else cmd.strip()
    first = seg.split()[0] if seg.split() else ""
    # a test runner anywhere in the command makes it a test
    if TEST.search(cmd):
        return "bash.test"
    # a write: a redirect to a real file (not /dev/null, not an fd like 2>&1), sed -i, or a writing program
    # (git, curl, wget and make are excluded from the writing programs: they are too often used just to read)
    if re.search(r"(?<![0-9&>])>{1,2}\s*(?!/dev/null|&)\S", cmd) or re.search(r"\bsed\s+-i", cmd) or first in BASH_WRITE - {"git", "curl", "wget", "make"}:
        return "bash.write"
    # a read-only program, unless its output is piped into python (which can do anything)
    if first in BASH_READ and not re.search(r"\|\s*python", cmd):
        return "bash.read"
    return "bash.other"


def classify(tool: str, args: str) -> tuple[str, bool]:
    """(component, is a state-changing call).  The component names the tool-output column of the graph (prefixed "tool:" in _episode)."""
    # a runtime prefix ("tau3-runtime_<tool>") is not part of the tool's own name
    name = tool.split("_", 1)[1] if tool.startswith("tau3-runtime_") else tool
    # the arguments are a JSON string; anything unparsable counts as no arguments
    try:
        a = json.loads(args)
    except Exception:
        a = {}
    # discoverable tools are wrappers: the name of the wrapped tool is inside the arguments
    inner = str(a.get("agent_tool_name", "")) if isinstance(a, dict) else ""
    if name == "KB_search":
        comp = "KB_search"
    elif name in ("call_discoverable_agent_tool", "unlock_discoverable_agent_tool"):
        # only a call (not an unlock) of a wrapped state-changing tool changes state
        return "discover", name.startswith("call") and bool(ACTION.match(inner))
    elif name == "send_message_to_user":
        comp = "msg"
    elif name in ("start_conversation", "end_conversation", "log_verification", "get_current_time", "get_assistant_tool_schemas"):
        comp = "conv"
    elif ACTION.match(name):
        comp = "act"
    elif LOOKUP.match(name):
        comp = "lookup"
    elif name == "bash":
        cmd = str(a.get("command", ""))
        comp = bash_class(cmd)
        # a shell command changes state when it writes or tests; an unclassified one only if it is a python one-liner that changes state
        return comp, comp in SINK or bool(comp == "bash.other" and PY_STATE.search(cmd))
    elif name == "read":
        # a file read is named by the kind of file (read.<kind> from the extension table), read.other if unknown
        ext = Path(str(a.get("filePath", ""))).suffix.lstrip(".").lower()
        comp = "read." + next((c for c, e in EXT.items() if ext in e.split()), "other")
    elif name in ("grep", "glob", "edit", "write"):
        comp = name
    elif name == "todowrite":
        comp = "todo"
    else:
        comp = "other"
    return comp, comp in SINK


# ---- stored request records (the recorded raw/ paths are gone for some runs) -------------------------------------------------------------------------

ART = ARTIFACTS
EXEC = ART / "execution"
METHOD = {"none": "native-noseed", "seed": "native-seed"}


class Runs:
    """View of runs addressed as <arm>/<model>/opencode/<bench>/<run>/...: live ones under execution/runs, finished ones exported to
    execution/<model>/<method>/<bench>/<run>/..., pre-native old ones in .trash (both glob and `/` look in all three)."""
    trash = ART.parents[1] / ".trash/assets/execution-runs-old-20261004"

    def _alt(self, rel):
        arm, model, _, bench, *rest = rel.split("/")
        return "/".join([model, "*" if arm == "*" else METHOD.get(arm, arm), bench, *rest])

    def glob(self, pat):
        return sorted(set(EXEC.glob(self._alt(pat))) | set((EXEC / "runs").glob(pat)) | set(self.trash.glob(pat)))

    def __truediv__(self, rel):
        for cand in (EXEC / "runs" / rel, EXEC / self._alt(str(rel)), self.trash / rel):
            if cand.exists(): return cand
        return EXEC / "runs" / rel


RUNS = Runs()
_index = {}


def index():
    if not _index:
        pats = ["*/*/opencode/*/execution/evaluations/*/*/*", "*/*/opencode/*/*/execution/evaluations/*/*/*",
                "*/*/opencode/*/*/execution/evaluations/*/*/*/*"]
        for pat in pats:
            for d in map(str, RUNS.glob(pat)):
                if "_superseded" in d: continue
                name = os.path.basename(d)
                if re.search(r"-r\d+-[0-9a-f]{12}$", name): _index.setdefault(name, []).append(d)
    return _index


def resolve(path: str, anchor: Path) -> Path:
    p = Path(path)
    if p.exists(): return p
    if "assets/artifacts/" in path:
        q = ART / path.split("assets/artifacts/", 1)[1]
        if q.exists(): return q
    m = re.search(r"/([^/]+-r\d+-[0-9a-f]{12})/requests/([^/]+)$", path)
    if m:
        for d in index().get(m[1], []):
            q = Path(d) / "requests" / m[2]
            if q.exists(): return q
    return p


# ---- Eq. (workload): exact request x component matrices ---------------------------------------------------------------------------------------------

CACHE = ASSET_CACHE / "workload-parse"  # derived parse cache (rebuildable)
CTX = GRAPH["context"]
SEP = "|"


@dataclass
class Episode:
    task: str
    success: bool | None
    R: int
    req_in: np.ndarray  # exact provider input tokens per request
    comps: list  # component names, columns of Xrg
    Xrg: np.ndarray  # [R, G] input tokens of each component in each request; rows sum to req_in exactly
    calls: list  # (k, component, n_k, exposure tokens summed over later requests, call signature) for each executed call, in issue order

    @property
    def X(self): return float(self.req_in.sum())
    @property
    def N(self): return len(self.calls)
    @property
    def c(self): return self.X / 1000  # workload: the trajectory stream, input tokens in thousands

    def N_g(self) -> dict:
        out = {}
        for c in self.calls: out[c[1]] = out.get(c[1], 0) + 1
        return out

    def X_g(self) -> dict: return dict(zip(self.comps, self.Xrg.sum(0)))


def _episode(args):
    """Build the graph of one recorded episode: (episode dict, path of its evaluation.json) -> Episode, or None if a request cannot be split."""
    ep, path = args
    # 1. the requests of the episode, each request id kept once (the record may repeat a request)
    reqs = []
    for r in ep["requests"]:
        if r["request_id"] not in {q["request_id"] for q in reqs}:
            reqs.append(r)
    # 2. the second stream: for every tool call, its column label tool:<class>|<id> (tc, handed to the allocator so it can
    #    attribute the output text to the call) and, if the harness really ran it, an entry of `executed`
    tc, executed = {}, []
    for k, r in enumerate(reqs):
        for c in r.get("tool_calls", []):
            # the call id is stored as "<request id>:<native id>" or bare; keep the native id
            native = c["call_id"][len(r["request_id"]) + 1:] if c["call_id"].startswith(r["request_id"] + ":") else c["call_id"]
            comp, _ = classify(c["tool"], c.get("arguments") or "")
            tc[native] = f"tool:{comp}{SEP}{native}"
            # executed = (request index, column label, native id, 12-hex signature of tool name + arguments)
            if c.get("executed") is not False:
                executed.append((k, f"tool:{comp}", native, hashlib.sha1(f'{c["tool"]}\0{c.get("arguments")}'.encode()).hexdigest()[:12]))
    # 3. the context stream: one row per request, {component: tokens}, from the allocator (rows sum to the provider count)
    rows, comps = [], {}
    for k, r in enumerate(reqs):
        rec = dict(r, record_path=str(resolve(r["record_path"], Path(path))))
        shares, why = H.allocate(rec, r["input_tokens"], tc)
        # an episode whose request cannot be split is left out
        if shares is None:
            return None
        row = defaultdict(float)
        # a share is named by its component; CTX maps it to a fixed column (system, tool_schemas, ...), skill:<name> pieces
        # all go to the "skill" column, and tool outputs keep their label tool:<class>|<id>
        for s in shares:
            row[CTX.get(s["component"]) or ("skill" if s["component"].startswith("skill:") else s["component"])] += s["tokens"]
        rows.append(row)
    # 4. the exposure of every executed call: its output's tokens summed over the requests AFTER the one that issued it
    calls = []
    for k, g, native, sig in executed:
        key = f"{g}{SEP}{native}"
        exp = sum(row.get(key, 0.0) for row in rows[k + 1:])
        # [request index, class, number of executed calls in that request, exposure, signature]
        calls.append([k, g, sum(1 for e in executed if e[0] == k), exp, sig])
    # 5. the matrix Xrg: columns are the component names with the call ids dropped (all outputs of one class share a column)
    names = sorted({n.split(SEP)[0] for row in rows for n in row})
    X = np.zeros((len(reqs), len(names)))
    for i, row in enumerate(rows):
        for n, v in row.items():
            X[i, names.index(n.split(SEP)[0])] += v
    # 6. success is unknown (None) when the outcome was marked invalid or the run logged an error
    success = ep["success"] if ep.get("outcome_valid", True) and "log_error" not in str(ep.get("outcome_reason")) else None
    return Episode(ep["task_id"], success, len(reqs), np.array([r["input_tokens"] for r in reqs], float), names, X, [tuple(c) for c in calls])


def load(path, ids=None, workers=24) -> list[Episode]:
    """Every episode of an evaluation.json (or the task ids): allocate per request, cached per file."""
    path = Path(path).resolve()
    CACHE.mkdir(parents=True, exist_ok=True)
    # the cache file is named by the path; it is valid while the evaluation.json has the same modification time
    cache = CACHE / (hashlib.sha1(str(path).encode()).hexdigest()[:16] + "-v4.pkl")
    key = path.stat().st_mtime_ns
    if cache.exists() and (got := pickle.loads(cache.read_bytes()))[0] == key:
        eps = got[1]
    else:
        # parse the episodes that have requests, in parallel, and drop those the allocator could not split
        raw = [e for e in json.loads(path.read_text("utf-8"))["episodes"] if e["requests"]]
        with ProcessPoolExecutor(workers) as pool:
            eps = [e for e in pool.map(_episode, [(e, path) for e in raw], chunksize=2) if e is not None]
        # atomic write: several processes may parse the same file, and a reader must never see a half-written pickle
        with tempfile.NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as handle:
            handle.write(pickle.dumps((key, eps)))
        os.replace(handle.name, cache)
    return [e for e in eps if ids is None or e.task in ids]


FIXED = tuple(GRAPH["fixed"])  # the components every request re-sends: the graph's fixed context


def moments(e):
    """(R, N, X, F, h, s_F) of an episode for the lever model, F the fixed prefix per request and h the new content per request; None when R < 2, N < 1 or F, H <= 0."""
    # F = tokens of the fixed columns in a request, averaged over the requests
    F = float(e.Xrg[:, [i for i, c in enumerate(e.comps) if c in FIXED]].sum(1).mean())
    # H = everything that is not fixed prefix = the history part, which is h per request growing linearly: sum over r of r h = R (R - 1) / 2 h
    H = e.X - e.R * F
    # returns (R, N, X, F, h, s_F) with h = H / (R (R - 1) / 2) and s_F = R F / X the share of the prefix in X
    return (e.R, e.N, e.X, F, H / (e.R * (e.R - 1) / 2), e.R * F / e.X) if e.R >= 2 and e.N >= 1 and F > 0 and H > 0 else None
