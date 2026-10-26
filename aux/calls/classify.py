"""What one executed tool call does, and the per-task call counts every calls analysis is computed from (a fixed API:
the calls folder, the three-word seed fingerprints and the edits figure use it).

Read-only (callsperturn's rule, decided by the recorded arguments): read, grep, glob, list / ls, webfetch, websearch; tau3
tools whose name starts get / find / search / list / check / lookup, KB_search, and a call_discoverable_agent_tool whose
wrapped tool name does; bash only when the command is a pure read: it parses (shlex), has no command substitution,
heredoc or redirect (except to /dev/null or 2>&1), and every segment (cd / set / export / VAR=value dropped) starts with
cat, head, tail, ls, grep, find (no -delete, -exec, -ok, -fprint, -fls) or wc, or sed -n without -i.

Write / read / other on AutoBench (classify): a bash call that reaches the world through automationbench_cli.py is a write
when any api_fetch chunk uses POST, PUT, PATCH or DELETE (except a POST to a query endpoint or a GraphQL body without a
mutation) or a named tool's verb writes; a read when it only reads, discovers tools or fetches; base64_encode is other.
Bash without the CLI is a read when it is a pure read, else other; write, edit and apply_patch are other (local files),
'O<tool>' so a sensitivity variant can count them as writes. AMBIG counts the ambiguous cases."""
import collections
import json
import math
import re
import shlex

import numpy as np

from src.graph.episode import LOOKUP

READ_PROGRAMS = {"cat", "head", "tail", "ls", "grep", "find", "wc"}
# sensitivity variant "lenient": also accepts the harmless separators and filters agents put between reads (echo "---", sort, ...); the strict rule is the headline
LENIENT_EXTRA = frozenset({"echo", "printf", "sort", "uniq", "cut", "tr", "pwd", "which", "stat", "file", "tree", "du", "diff"})
FIND_WRITE = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
READ_TOOLS = {"read", "grep", "glob", "list", "ls", "webfetch", "websearch"}
SETUP = re.compile(r"^(cd|set|export)$|^\w+=")
SEPARATORS = {";", "&&", "||", "|", "&", ";;"}


def bash_pure_read(command, extra=frozenset()):
    """True if the shell command is a pure read by the rule in the module docstring; False for anything else, including unparsable commands."""
    # command substitution and heredocs hide what is really run
    if "`" in command or "$(" in command or "<<" in command:
        return False
    segments = []
    # each physical line is parsed on its own; a quote spanning lines fails the parse and the command counts as not read-only
    for line in command.splitlines():
        if not line.strip():
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            return False
        current = []
        for position, token in enumerate(tokens):
            if token in SEPARATORS:
                segments.append(current)
                current = []
            elif set(token) <= set("<>&|;()") and token not in SEPARATORS:
                # a redirect operator: only to /dev/null or a file descriptor (2>&1) is harmless, input redirects and everything else are rejected
                target = tokens[position + 1] if position + 1 < len(tokens) else ""
                if token.startswith("<") or not (token.endswith("&") and target.isdigit() or target == "/dev/null"):
                    return False
                current.append(token)
            else:
                current.append(token)
        segments.append(current)
    # drop the setup segments (cd dir, set -e, export X, VAR=value) and the redirect targets that were kept as plain words
    program_segments = [s for s in segments if s and not SETUP.match(s[0])]
    if not program_segments:
        return False
    for seg in program_segments:
        program, rest = seg[0], seg[1:]
        if program in READ_PROGRAMS or program in extra:
            if program == "find" and FIND_WRITE & set(rest):
                return False
        elif program == "sed":
            flags = [t for t in rest if t.startswith("-") and not t.startswith("--")]
            if not any("n" in f[1:] for f in flags) or any("i" in f[1:] for f in flags) or "--in-place" in rest:
                return False
        else:
            return False
    return True


def bash_command(arguments):
    """The command string of a bash call's recorded arguments; "" if the arguments are truncated or not JSON (such a call is never a pure read)."""
    try:
        return str(json.loads(arguments).get("command", ""))
    except (ValueError, AttributeError):
        return ""


def read_only(tool, arguments, extra=frozenset()):
    """True if one executed call is read-only by the module-docstring rule; tool and arguments as recorded in the request."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    if tool.startswith("tau3-runtime_"):
        name = tool.split("_", 1)[1]
        if name == "KB_search":
            return True
        if name == "call_discoverable_agent_tool":
            return bool(LOOKUP.match(str(args.get("agent_tool_name", ""))))
        return bool(LOOKUP.match(name))
    if tool == "bash":
        return bash_pure_read(bash_command(arguments), extra)
    return tool in READ_TOOLS


def turn_calls(raw_episode):
    """Per turn of one recorded episode (requests deduplicated by id as graph.episode does): the list of executed calls as (tool, arguments)."""
    seen, turns = set(), []
    for request in raw_episode["requests"]:
        if request["request_id"] in seen:
            continue
        seen.add(request["request_id"])
        turns.append([(c["tool"], c.get("arguments") or "") for c in request.get("tool_calls", []) if c.get("executed") is not False])
    return turns


METHOD = re.compile(r"(?i)method\W{0,8}(GET|POST|PUT|PATCH|DELETE)\b")
URL = re.compile(r"""(?i)(?:--url[=\s]+|["']url["']\s*:\s*)['"]?([^\s'"]+)""")
POST_READ = re.compile(r"(?i)(:find\b|/search\b|/query\b|search\.|\.list\b|/lookup\b|/find\b|:search\b)")
NAMED = re.compile(r"automationbench_cli\.py\W+(?:call|invoke|execute_tool)\b[^\n]*?--(?:name|tool|tool-name)[=\s]+['\"]?([A-Za-z0-9_.\-]+)")
READ_VERB = re.compile(r"(?i)(^|_)(find|get|list|search|lookup|read|retrieve|fetch|count|query|show|describe|check)(_|$)")
WRITE_VERB = re.compile(r"(?i)(^|_)(create|update|add|send|delete|remove|post|set|move|archive|invite|reply|assign|upload|close|complete|mark|change|"
                        r"cancel|insert|append|write|edit|copy|approve|reject|submit|schedule|register|log|attach|transfer|convert|merge|enroll|unsubscribe|tag)(_|$)")
DISCOVERY = re.compile(r"automationbench_cli\.py\W+(api_search|search_tools|show_tools|describe\w*|init|help|--help|list_tools)\b")
LOCAL_OTHER_TOOLS = {"write", "edit", "apply_patch", "todowrite", "patch", "multiedit"}
AMBIG = collections.Counter()


def classify_fetch(cmd):
    """'W' / 'R' for a command containing api_fetch, chunk by chunk; also records ambiguity counters."""
    parts = cmd.split("api_fetch")[1:]
    kinds = []
    script = ("<<" in cmd or "python -c" in cmd or "python3 -c" in cmd or "subprocess" in cmd)
    for chunk in parts:
        m = METHOD.search(chunk)
        method = m.group(1).upper() if m else "GET"
        if chunk.lstrip().startswith("--help") or chunk.lstrip().startswith("-h"):
            kinds.append("R"); continue
        if method == "GET":
            kinds.append("R"); continue
        u = URL.search(chunk)
        url = u.group(1) if u else ""
        if method == "POST" and "graphql" in url.lower():
            if "mutation" in chunk.lower():
                AMBIG["graphql POST with mutation -> W"] += 1; kinds.append("W")
            else:
                AMBIG["graphql POST without mutation -> R"] += 1; kinds.append("R")
            continue
        if method == "POST" and POST_READ.search(url):
            AMBIG["POST to query endpoint (:find, /search, /query, .list) -> R"] += 1; kinds.append("R"); continue
        kinds.append("W")
    if script:
        AMBIG["api_fetch inside an agent-written python script (method read from text)"] += 1
        # a helper such as fetch(url, "POST", body) carries the method as a quoted literal, not after 'method'
        if "W" not in kinds and re.search(r"""['"](POST|PUT|PATCH|DELETE)['"]""", cmd):
            AMBIG["script with a quoted mutating method literal -> W"] += 1
            kinds.append("W")
    if len(parts) > 1:
        AMBIG["bash call with >= 2 api_fetch invocations (sequential, one call)"] += 1
        if "W" in kinds and "R" in kinds:
            AMBIG["bash call mixing GET and a mutating method -> W"] += 1
    return "W" if "W" in kinds else "R"


def classify(tool, arguments):
    if tool == "bash":
        cmd = bash_command(arguments)
        kinds = []
        if "api_fetch" in cmd:
            kinds.append(classify_fetch(cmd))
        for name in NAMED.findall(cmd):
            if name == "api_fetch":
                continue
            if READ_VERB.search(name) and not WRITE_VERB.search(name):
                kinds.append("R")
            elif WRITE_VERB.search(name):
                kinds.append("W")
            else:
                AMBIG["named tool with no read/write verb -> W: " + name] += 1; kinds.append("W")
            AMBIG["named (Zapier) tool call classified by verb"] += 1
        if DISCOVERY.search(cmd):
            kinds.append("R")
        if kinds:
            return "W" if "W" in kinds else "R"
        if "automationbench_cli.py" in cmd and "base64_encode" in cmd:
            AMBIG["base64_encode -> O"] += 1
            return "O"
        return "R" if bash_pure_read(cmd) else "O"
    if tool in READ_TOOLS:
        return "R"
    if tool in LOCAL_OTHER_TOOLS:
        return "O" + tool  # local file change; 'Owrite'/'Oedit' count as W in the sensitivity variant
    AMBIG["other tool -> O: " + tool] += 1
    return "O"


def task_arrays(reps, tasks, local_w=False, only_complete=True):
    """Per task (pooled over the row's runs): counts used by every statistic, as arrays aligned with tasks."""
    keys = ["turns", "calls", "W", "R", "O", "W2", "R2", "Wpar", "Rpar", "c2", "eps", "Nsum", "Xsum", "Rsum", "WT"]
    out = {k: np.zeros(len(tasks)) for k in keys}
    for i, t in enumerate(tasks):
        for rep in reps.values():
            ep = rep.get(t)
            if ep is None or (only_complete and not ep["complete"]):
                continue
            out["eps"][i] += 1
            out["Nsum"][i] += sum(len(x) for x in ep["turns"]); out["Rsum"][i] += len(ep["turns"])
            out["Xsum"][i] += ep["X"] if ep["X"] is not None else np.nan
            for turn in ep["turns"]:
                w = sum(1 for c in turn if c == "W" or (local_w and c in ("Owrite", "Oedit", "Oapply_patch")))
                r = sum(1 for c in turn if c == "R")
                out["turns"][i] += 1
                out["calls"][i] += len(turn)
                out["W"][i] += w; out["R"][i] += r; out["O"][i] += len(turn) - w - r
                out["W2"][i] += w >= 2; out["R2"][i] += r >= 2; out["WT"][i] += w >= 1
                out["Wpar"][i] += w if w >= 2 else 0
                out["Rpar"][i] += r if r >= 2 else 0
                out["c2"][i] += len(turn) >= 2
    return out


def common_tasks(seed, own, only_complete=True):
    """Table 1's common set: tasks with a mean over the row's usable runs on both sides (N, X > 0); loose: any recorded episode on both sides."""
    def means(reps):
        per = collections.defaultdict(list)
        for rep in reps.values():
            for t, ep in rep.items():
                if ep["complete"]:
                    per[t].append(ep)
                elif not only_complete:
                    per[t].append(dict(ep, N=sum(len(x) for x in ep["turns"]), X=1.0, R=len(ep["turns"])))
        return {t: (np.mean([e["N"] for e in v]), np.mean([e["X"] for e in v]), np.mean([e["R"] for e in v])) for t, v in per.items()}
    a, b = means(seed), means(own)
    tasks = sorted(t for t in b if t in a and all(x > 0 for x in (*a[t][:2], *b[t][:2])))
    return tasks, a, b


def stats_of(s, o, idx):
    """Statistics of one (seed, skill) pair on a task index vector (a bootstrap draw or the identity)."""
    def S(d, k):
        return d[k][idx].sum()
    res = {}
    for side, d in (("seed", s), ("skill", o)):
        turns = S(d, "turns")
        res[side] = dict(m=S(d, "calls") / turns, w_turn=S(d, "W") / turns, r_turn=S(d, "R") / turns, o_turn=S(d, "O") / turns,
                         W2=S(d, "W2") / turns, R2=S(d, "R2") / turns, c2=S(d, "c2") / turns,
                         Wpar=S(d, "Wpar") / max(S(d, "W"), 1), Rpar=S(d, "Rpar") / max(S(d, "R"), 1),
                         W_ep=S(d, "W") / S(d, "eps"), R_ep=S(d, "R") / S(d, "eps"), turns_ep=turns / S(d, "eps"), calls_ep=S(d, "calls") / S(d, "eps"))
    # Table 1 style ratios: geometric mean over tasks of per-task means
    def lg(k):
        return float(np.mean(np.log((o[k][idx] / o["eps"][idx]) / (s[k][idx] / s["eps"][idx]))))
    res["log_rho_calls"], res["log_rho_turns"], res["log_rho_X"] = lg("Nsum"), lg("Rsum"), lg("Xsum")
    res["log_W_ratio"] = math.log(res["skill"]["W_ep"] / res["seed"]["W_ep"])
    res["log_R_ratio"] = math.log(res["skill"]["R_ep"] / res["seed"]["R_ep"])
    for k in ("m", "w_turn", "r_turn", "W2", "R2", "c2", "Wpar", "Rpar"):
        res["d_" + k] = res["skill"][k] - res["seed"][k]
    res["did_par"] = res["d_Wpar"] - res["d_Rpar"]
    # packing counterfactual: the skill's writes packed at the seed's writes per write-carrying turn; turns saved per episode and the pooled turn ratio
    k_seed = S(s, "W") / S(s, "WT")
    saved = (S(o, "WT") - S(o, "W") / k_seed) / S(o, "eps")
    res["writes_per_write_turn_seed"], res["writes_per_write_turn_skill"] = k_seed, S(o, "W") / S(o, "WT")
    res["pack_saved_turns_ep"] = saved
    res["log_turns_pooled"] = math.log(res["skill"]["turns_ep"] / res["seed"]["turns_ep"])
    res["log_turns_pooled_packed"] = math.log((res["skill"]["turns_ep"] - saved) / res["seed"]["turns_ep"])
    res["pack_log_turns_removed"] = res["log_turns_pooled"] - res["log_turns_pooled_packed"]
    return res
