"""The bipartite graph of an episode: which part of the prompt each model request paid for (paper Sec. 3.1 and 4.1).

What an episode is.  The agent talks to the model R times (R requests, "turns").  The provider bills every request for ALL the
input tokens it receives, because the harness never compacts the conversation: each request re-sends the system prompt, the tool
schemas, the task, the skill, and the whole history so far.  So the cost of an episode is X = the sum over its requests of the
provider's input-token count, and it is mostly re-sent text, not new text.

The graph.  Rows are the R requests.  Columns are the components of the prompt: system, tool_schemas, task, skill, history, and the
tool outputs grouped by tool class (tool:lookup, tool:KB_search, tool:act, ... named by classify below).  Xrg[r, g] is the number of
tokens of component g inside request r.  graph/allocate.py splits each request's provider count by estimated content and rounds by
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

Which functions feed the optimizer and which only the analyses.
    - load, _episode, observe, paired feed the loop: they build the graphs of a run and give L(S|P) = mean over the tasks both
      versions ran of log(X(S) / X(P)), the cost of the child S against the parent P (L < 0 means cheaper).  observe also attaches
      the success y and the soft score s, and Obs(task, y, s, c, ep, row) is what the loop reads.
    - split (the per-component split of L) and levers / moments (the four-term turn-elasticity readout, X = R F + R (R - 1) / 2 h)
      are recorded per version for the analyses.  They do not drive any decision.

Two vocabularies for the same call.  episode.classify names the matrix columns (for example "tool:lookup", a coarse class used
for the token accounting); graph/costmodel.behavior names the rows of the cost table (for example "get_customer_by_phone", a fine
class used to price edits).  Same call, two labels.
"""

import hashlib, json, math, os, pickle, re, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from graph import allocate as H
from harness_bench.eval import verifier as _hv  # the executor's verifier-detail parsers (assertions / tau / tests / stdout)
from graph import _config as config
from src.layout import ARTIFACTS, CACHE as ASSET_CACHE, HARNESS_BENCH, MITIGATION_ROOT, relocate  # historical records keep their old roots

# ---- component of a tool call -------------------------------------------------------------------------------------------------------------------------

# the vocabulary of tool-call components lives in graph/graph.yaml (graph.classify)
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


# ---- per-task observations Obs: success y, soft score s, workload c, the executor row -----------------------------------------------------------------

SHARE = str(MITIGATION_ROOT.parents[1]) + "/"  # the share root that recorded "proj_harness/..." paths are relative to


@dataclass
class Obs:  # one task under one skill: success, soft score, workload, its exact allocation, the executor row
    task: str; y: int | None; s: float | None; c: float | None
    ep: Episode | None = field(default=None, repr=False); row: dict = field(default_factory=dict, repr=False)
    why: str | None = None  # why the outcome is not the skill's doing (the infra function given to observe): None = a real failure/success


TAU3 = {"ACTION": ("action_checks", "action_match"), "ENV_ASSERTION": ("env_assertions", "met"), "NL_ASSERTION": ("nl_assertions", "met"), "COMMUNICATE": ("communicate_checks", "met")}
_STDOUT = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _abs(p): return relocate(p if str(p).startswith("/") else SHARE + p if str(p).startswith("proj_harness/") else str(HARNESS_BENCH / p))


def _json(p):
    try: return json.loads(Path(p).read_text("utf-8"))
    except (OSError, ValueError): return None


def files(e):
    """Verifier files of the scored case: {name: path} from cases/<task>/ next to the results, else the harbor trial's verifier/."""
    res = _abs(e["localization_sources"]["results"]) if (e.get("localization_sources") or {}).get("results") else None
    out, cand = {}, []
    if res and res.is_file():
        rows = [json.loads(l) for l in res.read_text("utf-8").splitlines() if l.strip()]
        row = next((r for r in reversed(rows) if str(r.get("item_id")) == e["task_id"]), rows[-1] if rows else {})
        cand.append(res.parent / "cases" / e["task_id"])
        if row.get("native_result_path"): cand.append(_abs(row["native_result_path"]).parent / "verifier")
    if trial := next((r.split("/agent/")[0] for r in e.get("evidence_refs") or [] if "/native/" in r and "/agent/" in r), None): cand.append(_abs(trial) / "verifier")
    for d in cand:
        for n in ("result.json", "ctrf.json", "test-stdout.txt"):
            if n not in out and (d / n).is_file(): out[n] = d / n
        if "result.json" in out or "ctrf.json" in out: break
    return out


def _tau(ri):
    """(act, partial) of a tau3 reward_info: act = share of gold actions matched; partial = mean over the reward basis of each component's check
    fraction, a component with an empty / null check list left out (tau2 scores it 1.0 with nothing checked: it would hand every DB failure a free 0.5)."""
    parts, vac, basis, bd = {}, 0, ri.get("reward_basis") or [], ri.get("reward_breakdown") or {}
    for b in basis:
        if b == "DB" and isinstance(ri.get("db_check"), dict): parts[b] = float(ri["db_check"]["db_reward"])
        elif b in TAU3 and ri.get(TAU3[b][0]): xs = ri[TAU3[b][0]]; parts[b] = sum(bool(x.get(TAU3[b][1])) for x in xs) / len(xs)
        elif b in TAU3 and TAU3[b][0] in ri: vac += 1
        elif b in bd: parts[b] = float(bd[b])
        elif basis == ["DB"]: parts[b] = float(ri["reward"])
    acts = ri.get("action_checks") or []
    return (sum(bool(a.get("action_match")) for a in acts) / len(acts) if acts else None), (sum(parts.values()) / len(parts) if parts and len(parts) + vac == len(basis) else None)


def soft(e, y, bench):
    """s_i (design 2.1): AutoB passed / non-excluded assertions (= verifier_rewards.partial_credit); TB2 passed / tests of the pytest CTRF (test-stdout PASSED
    lines if none); tau3 `act`, else the partial, else y; 0 where the protocol overrode a pass (benchmark source contaminated); y where no partial is on record."""
    vr, f, p = (e.get("backend_outcome") or {}).get("verifier_rewards") or {}, files(e), None
    res = _json(f["result.json"]) if "result.json" in f else None
    if bench == "automationbench" and isinstance(res, dict) and isinstance(res.get("assertion_results"), list):
        a = [x for x in res["assertion_results"] if not x.get("excluded")]; p = sum(bool(x.get("passed")) for x in a) / len(a) if a else None
    elif bench == "automationbench" and "partial_credit" in vr: p = float(vr["partial_credit"])
    elif bench == "tau3-bench" and isinstance(res, dict) and isinstance(res.get("reward_info"), dict): act, part = _tau(res["reward_info"]); p = part if act is None else act
    elif bench == "terminal-bench-2-1":
        tests = ((_json(f["ctrf.json"]) if "ctrf.json" in f else None) or {}).get("results", {}).get("tests")
        if isinstance(tests, list) and tests: p = sum(t.get("status") == "passed" for t in tests) / len(tests)
        elif "test-stdout.txt" in f and (hits := _STDOUT.findall(f["test-stdout.txt"].read_text("utf-8", "replace"))):
            names = dict((n, s) for s, n in hits); p = sum(s == "PASSED" for s in names.values()) / len(names)
    return 0.0 if e.get("outcome_reason") == "benchmark_source_contaminated" else float(y) if p is None else p


def detail(e):
    """The failure detail the executor's verifier string misses on these runs: harness_bench's verifier_detail looks the
    verifier files up through _files(), which resolves the recorded relative artifact paths against the results file's
    directory (they are share-root relative), so it finds nothing and only backend_outcome's rewards / score / category
    reach the row; files() here locates them. -> the verifier_detail fields built from the verifier files (checks, failed,
    output, ...), {} when none."""
    f, d = files(e), {}
    res = _json(f["result.json"]) if "result.json" in f else None
    if isinstance(res, dict) and isinstance(res.get("assertion_results"), list):
        _hv._assertions(d, res)
    elif isinstance(res, dict) and isinstance(res.get("reward_info"), dict):
        _hv._tau(d, res)
    ctrf = _json(f["ctrf.json"]) if "ctrf.json" in f else None
    tests = (ctrf.get("results") or {}).get("tests") if isinstance(ctrf, dict) else None
    if isinstance(tests, list):
        _hv._tests(d, tests)
    elif "test-stdout.txt" in f and "failed" not in d:
        _hv._stdout(d, f["test-stdout.txt"])
    return d


def _verifier_text(d):
    """The executor's formatting of a verifier_detail dict (bipa executor.evaluate)."""
    return "\n".join("\n- ".join([f"{k}:", *map(str, v)]) if isinstance(v, list) else f"{k}: {v}"
                     for k, v in d.items() if v not in (None, "", [], {}))


def observe(rows, infra=None) -> dict:
    """executor.evaluate rows (one evaluation.json) -> {task: Obs}; an episode the allocation cannot split has c = None.

    infra: function raw episode -> reason or None (bach/common/episodes.py infra_reason); its answer is Obs.why.  None = no episode is infra."""
    path = rows[0]["evaluation_path"]
    doc = json.loads(Path(path).read_text("utf-8")); raw, bench = {e["task_id"]: e for e in doc["episodes"]}, (doc.get("provenance") or {}).get("benchmark")
    eps = {e.task: e for e in load(path, {r["id"] for r in rows})}
    for r in rows:
        # a failed episode whose verifier string carries no failure detail gets it rebuilt from the verifier files
        if r["success"] == 0 and "failed:" not in (r.get("verifier") or ""):
            extra = _verifier_text(detail(raw[r["id"]]))
            if extra:
                r["verifier"] = "\n".join(x for x in (r.get("verifier"), extra) if x)
    return {r["id"]: Obs(r["id"], r["success"], None if r["success"] is None else soft(raw[r["id"]], r["success"], bench),
                         eps[r["id"]].c if r["id"] in eps else None, eps.get(r["id"]), r,
                         why=infra(raw[r["id"]]) if infra else None) for r in rows}


def paired(parent_rows, child_rows):
    """(tasks, L): the tasks both versions ran (Episodes) and L(S|P) = mean over them of log(X_i(S) / X_i(P)), the geometric-mean ratio of input tokens."""
    P, C = {e.task: e for e in parent_rows}, {e.task: e for e in child_rows}
    # the tasks both versions ran, with input tokens on both sides (the log needs positive X)
    ts = [t for t in P if t in C and min(P[t].X, C[t].X) > 0]
    # L = mean over those tasks of log(child X / parent X); 0 when there is no common task
    return ts, float(np.mean([math.log(C[t].X / P[t].X) for t in ts])) if ts else 0.0


def split(parent_rows, child_rows) -> dict:
    """The two streams of S (child) against P (parent) on the tasks both ran.
    trajectory: L(S|P) on the input tokens split exactly over components; with mu_i = (X_i(S) - X_i(P)) / log(X_i(S) / X_i(P)) (X_i(P) if equal), component g
    contributes mean_i (X_ig(S) - X_ig(P)) / mu_i, and the parts add up to L.
    harness: L_N = mean_i log(N_i(S) / N_i(P)) on the tasks with calls on both sides, split over components the same way with the call counts; it is not part of L
    and tells which actions the token change came from.
    -> {n, L, trajectory: (L, {g: part}), harness: (L_N, {g: part}), n_harness}."""
    P, C = {e.task: e for e in parent_rows}, {e.task: e for e in child_rows}
    ts, L = paired(parent_rows, child_rows)
    # the harness stream needs at least one executed call on both sides (the log of a call count)
    tn = [t for t in ts if min(P[t].N, C[t].N) > 0]

    def lmdi(tasks, tot, comp):
        """Log-mean-Divisia split: tot(episode) is the total (X or N), comp(episode) its per-component dict; returns (mean log ratio, {g: part})."""
        parts, total = defaultdict(float), 0.0
        for t in tasks:
            # a = parent total, b = child total; mu is the logarithmic mean of a and b (a itself when they are equal)
            a, b = tot(P[t]), tot(C[t])
            mu = (b - a) / math.log(b / a) if b != a else a
            # the task's log ratio, averaged over the tasks
            total += math.log(b / a) / len(tasks)
            ga, gb = comp(P[t]), comp(C[t])
            # each component contributes its change divided by mu; dividing by mu makes the parts add up exactly to log(b / a)
            for g in ga.keys() | gb.keys():
                parts[g] += (gb.get(g, 0) - ga.get(g, 0)) / mu / len(tasks)
        return total, dict(parts)

    # the trajectory stream splits X (input tokens) over the graph's columns; the harness stream splits N (call counts) over tool classes
    traj = lmdi(ts, lambda e: e.X, lambda e: e.X_g()) if ts else (0.0, {})
    harn = lmdi(tn, lambda e: e.N, lambda e: e.N_g()) if tn else (0.0, {})
    return {"n": len(ts), "L": L, "trajectory": traj, "harness": harn, "n_harness": len(tn)}


FIXED = tuple(GRAPH["fixed"])  # the components every request re-sends: the graph's fixed context


def moments(e):
    """(R, N, X, F, h, s_F) of an episode for the lever model, F the fixed prefix per request and h the new content per request; None when R < 2, N < 1 or F, H <= 0."""
    # F = tokens of the fixed columns in a request, averaged over the requests
    F = float(e.Xrg[:, [i for i, c in enumerate(e.comps) if c in FIXED]].sum(1).mean())
    # H = everything that is not fixed prefix = the history part, which is h per request growing linearly: sum over r of r h = R (R - 1) / 2 h
    H = e.X - e.R * F
    # returns (R, N, X, F, h, s_F) with h = H / (R (R - 1) / 2) and s_F = R F / X the share of the prefix in X
    return (e.R, e.N, e.X, F, H / (e.R * (e.R - 1) / 2), e.R * F / e.X) if e.R >= 2 and e.N >= 1 and F > 0 and H > 0 else None


def levers(parent_rows, child_rows) -> dict:
    """The mechanism of a trajectory-cost change, on the tasks both versions ran with R >= 2 requests and N >= 1 calls. An episode of R requests re-sends the fixed prefix F
    (FIXED components, mean per request) at every request and the history so far, which grows by h per request: X = R F + R (R - 1) / 2 h, so h is read off X, and N = R m
    with m the calls per request. Per task, with s_F = R F / X, s_H = 1 - s_F (means of the two versions) and the exact turn elasticity e_R = s_F + s_H (2R - 1) / (R - 1), to first order
    d log X = e_R d log R + s_H d log h + s_F d log F, and d log R = d log N - d log m: extra turns from extra actions (N) are capability, only fewer calls per turn (m falling) can be merged.
    Three levers can be removed: merge = -e_R d log m (same actions in fewer turns), content = s_H d log h (less per turn), text = s_F d log F (shorter skill). Merging turns back
    returns the per-turn content that splitting them removed, so the merge lever counts net of a content fall: merge_net = merge + content when the mean content change is negative. removable = the positive
    parts of merge_net, content and text (taken on the means), capped at d log X; se (over tasks) is a diagnostic field only, nothing decides on it (economize recomputes removable from these fields).
    -> {n, dlogX, dlogR, dlogN, dlogm, dlogh, dlogF, sF, eR, turns: e_R d log R, merge, merge_net, content, text, resid (dlogX minus turns, content, text), removable, se,
        dR, dN (mean change in requests and calls per task), harness: {component: part of d log N} (split()'s harness stream)}."""
    # moments of every episode (None when the lever model does not apply); the tasks both versions have a valid moment tuple for
    P, C = {e.task: moments(e) for e in parent_rows}, {e.task: moments(e) for e in child_rows}
    ts = [t for t in P if P[t] and C.get(t)]
    pe, ce = {e.task: e for e in parent_rows}, {e.task: e for e in child_rows}
    # the output fields, all zero until filled
    ks = ("dlogX", "dlogR", "dlogN", "dlogm", "dlogh", "dlogF", "sF", "eR", "turns", "merge", "merge_net", "content", "text", "resid", "removable", "se", "dR", "dN")
    out = dict.fromkeys(ks, 0.0) | {"n": len(ts), "harness": {}}
    if not ts:
        return out
    # lg(i) = per-task log ratio child / parent of the i-th moment (0 R, 1 N, 2 X, 3 F, 4 h)
    lg = lambda i: np.array([math.log(C[t][i] / P[t][i]) for t in ts])
    # s_F and R at the midpoint of the two versions
    sF = np.array([(P[t][5] + C[t][5]) / 2 for t in ts])
    R = np.array([(P[t][0] + C[t][0]) / 2 for t in ts])
    # turn elasticity of X: d log X / d log R = s_F + s_H (2R - 1) / (R - 1), with s_H = 1 - s_F, from X = R F + R (R - 1) / 2 h
    eR = sF + (1 - sF) * (2 * R - 1) / (R - 1)
    dR, dN, dX, dh, dF = lg(0), lg(1), lg(2), lg(4), lg(3)
    # the four terms: turns (all of the change of R), merge (calls per turn m = N / R fell: d log m = dN - dR),
    # content (history per turn h), text (fixed prefix F, i.e. mostly the skill)
    turns, merge, content, text = eR * dR, -eR * (dN - dR), (1 - sF) * dh, sF * dF
    merge_net = merge + content * (content.mean() < 0)  # net of the content merged turns carry back, when splitting lowered it on average
    # removable per task = sum of those of merge_net, content, text whose mean is positive (a cost the skill added)
    pos = [x for x in (merge_net, content, text) if x.mean() > 0]
    rem = sum(pos) if pos else np.zeros(len(ts))
    # means over the tasks; removable is capped by the total change dlogX and floored at 0; se is the standard error of the removable part
    out.update(dlogX=dX.mean(), dlogR=dR.mean(), dlogN=dN.mean(), dlogm=(dN - dR).mean(), dlogh=dh.mean(), dlogF=dF.mean(), sF=sF.mean(), eR=eR.mean(), turns=turns.mean(), merge=merge.mean(), merge_net=merge_net.mean(),
               content=content.mean(), text=text.mean(), resid=(dX - turns - content - text).mean(), removable=max(0.0, min(rem.mean(), dX.mean())),
               se=rem.std(ddof=1) / math.sqrt(len(ts)) if len(ts) > 1 else 0.0, dR=float(np.mean([C[t][0] - P[t][0] for t in ts])), dN=float(np.mean([C[t][1] - P[t][1] for t in ts])),
               harness=split([pe[t] for t in ts], [ce[t] for t in ts])["harness"][1])
    # plain Python floats for storage; dicts and ints (n) pass through
    return {k: float(v) if not isinstance(v, (dict, int)) else v for k, v in out.items()}


def episodes(obs): return [o.ep for o in obs.values() if o.ep]
