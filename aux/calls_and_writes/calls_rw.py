"""Rule -> behavior specificity on AutoBench (offline, recorded test episodes only).

The learned AutoBench skills add serialize rules ("issue MUTATING calls ONE AT A TIME; parallel is safe only for READS").  If those rules act, the
share of turns that carry two or more state-changing calls should fall under a skill with serialize rules, while turns with two or more reads should not.

Episodes: Table 1's avg@3 test runs (assets/cache/analysis/table1/table1.json 'files'; supplements merged with table1.merged_episodes), AutoBench,
DeepSeek-V4.1-Flash and GPT-6-Luna; rows Seed, GEPA, SkillOpt, SkillAdam (No skill as a reference).  Task set per (row vs Seed): Table 1's common set
(per-task 3-run means with verified outcome and N, X > 0 on both sides); only episodes that enter those means (task_row N, X, success known) are used.
For DeepSeek the script checks that n, rho^calls and rho (input tokens) reproduce table1.json; Luna's Table 1 rows (luna_row) drop provider-failed
requests and keep the episode, while the Luna task sets here use complete logs only, so there the check is printed only.

Call classification (one executed tool call -> W write / R read / O other), from the AutoBench tool set:
  bash commands that reach the world through automationbench_cli.py:
    api_fetch (raw HTTP; also `invoke --tool api_fetch`): split the command at each api_fetch occurrence; per chunk method = first
      `method ... GET|POST|PUT|PATCH|DELETE` token (default GET, the CLI default); GET -> read; POST/PUT/PATCH/DELETE -> write, except
      POST to a query endpoint (path ends :find, /search, /query, contains search. / .list / /lookup) and GraphQL bodies without 'mutation' -> read.
    named tools (`call --name`, `invoke --tool`, `execute_tool --tool-name`, Zapier toolset): verb in the name; find/get/list/search/lookup/read/
      retrieve/fetch/count -> read; create/update/add/send/delete/remove/post/set/move/archive/... -> write; else ambiguous -> write.
    discovery subcommands api_search, search_tools, show_tools, describe, init, help -> read.  base64_encode -> other (local encoding).
    a bash call with several invocations is one call: write if any chunk writes, else read if any chunk reads.
  bash without the CLI: read if it is a pure read by callsperturn.bash_pure_read (cat, head, ls, grep, ...), else other (python, echo, ...).
  read, grep, glob, list -> read; write, edit, apply_patch, todowrite -> other (local files, not world state; sensitivity counts write/edit as W).
Writes the per-episode compact data to calls_rw_data.pkl and the numbers to calls_rw.json (OUT/calls_and_writes).
Run:  python -m aux.calls_and_writes.calls_rw extract|stats
"""
import collections
import json
import math
import pickle
import re
import sys

import numpy as np

from src.analysis import table1 as T
from aux.calls_and_writes.callsperturn import bash_pure_read, bash_command, READ_TOOLS
from src import layout
from src.utils.dataloader import task_row

OUT = layout.OUT / "calls_and_writes"
TABLE1 = layout.CACHE / "analysis/table1/table1.json"
MODELS = ["deepseek-flash", "gpt-6-luna"]
ROWS = ["No skill", "Seed skill", "GEPA", "SkillOpt", "SkillAdam"]
BENCH = "automationbench"

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


def extract():
    """Per model, row, rep: {task: [per-turn list of classes]} for episodes with known success, N, X (Table 1's rows)."""
    t1 = json.loads(TABLE1.read_text())
    data = {}
    for model in MODELS:
        cell = t1[f"{model}|{BENCH}"]
        for row in ROWS:
            reps = {}
            for rep, files in cell[row]["files"].items():
                eps = T.merged_episodes([layout.relocate(f) for f in files])
                per = {}
                for (task, _), e in eps.items():
                    tr = task_row(e)
                    complete = not (tr["success"] is None or tr["N"] is None or tr["X"] is None)
                    seen, turns = set(), []
                    for r in e.get("requests", []):
                        if r["request_id"] in seen:
                            continue
                        seen.add(r["request_id"])
                        if not complete and r.get("input_tokens") is None:
                            continue  # loose set: completed requests only, calls not marked unexecuted
                        turns.append([classify(c["tool"], c.get("arguments") or "") for c in r.get("tool_calls", [])
                                      if (c.get("executed") is True if complete else c.get("executed") is not False)])
                    if complete:
                        assert sum(len(t) for t in turns) == tr["N"] and len(turns) == tr["nreq"], (model, row, task)
                    if not turns:
                        continue
                    per[task] = dict(turns=turns, N=tr["N"], X=tr["X"], R=tr["nreq"] if complete else len(turns), success=tr["success"], complete=complete)
                reps[int(rep)] = per
                print(model, row, rep, len(per), sum(v["complete"] for v in per.values()), flush=True)
            data[(model, row)] = reps
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(data=data, ambig=dict(AMBIG)), open(OUT / "calls_rw_data.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------
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


def flat(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def stats(B=4000, seed=20261020):
    blob = pickle.load(open(OUT / "calls_rw_data.pkl", "rb"))
    data = blob["data"]
    t1 = json.loads(TABLE1.read_text())
    rng = np.random.default_rng(seed)
    out = dict(ambig=blob["ambig"], cells=[])
    for model in MODELS:
        for row in ["No skill", "GEPA", "SkillOpt", "SkillAdam"]:
            seed_reps, own_reps = data[(model, "Seed skill")], data[(model, row)]
            tasks, a, b = common_tasks(seed_reps, own_reps)
            loose_tasks, _, _ = common_tasks(seed_reps, own_reps, only_complete=False)
            ref = t1[f"{model}|{BENCH}"][row]
            rho_calls = math.exp(np.mean([math.log(b[t][0] / a[t][0]) for t in tasks]))
            rho_x = math.exp(np.mean([math.log(b[t][1] / a[t][1]) for t in tasks]))
            check = dict(n=len(tasks), n_table1=ref["n"], rho_calls=rho_calls, rho_calls_table1=ref["rho_inv"], rho_x=rho_x, rho_x_table1=ref["rho_trj"])
            same = len(tasks) == ref["n"] and abs(rho_calls - ref["rho_inv"]) < 1e-6 and abs(rho_x - ref["rho_trj"]) < 1e-6
            assert same or model == T.LUNA, check
            for variant, local_w, complete, tset in (("main", False, True, tasks), ("local_writes_as_W", True, True, tasks), ("all_recorded_turns", False, False, loose_tasks)):
                s, o = task_arrays(seed_reps, tset, local_w, complete), task_arrays(own_reps, tset, local_w, complete)
                point = flat(stats_of(s, o, np.arange(len(tset))))
                draws = collections.defaultdict(list)
                for _ in range(B if variant == "main" else 1000):
                    idx = rng.integers(0, len(tset), len(tset))
                    for k, v in flat(stats_of(s, o, idx)).items():
                        draws[k].append(v)
                ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}
                # one-sided bootstrap share of draws at or above zero for the change in write-parallel turns (prediction: < 0)
                p_up = {k: float(np.mean(np.array(draws[k]) >= 0)) for k in ("d_W2", "d_Wpar", "d_R2", "d_Rpar", "did_par")}
                out["cells"].append(dict(model=model, row=row, variant=variant, n=len(tset), check=check, point=point, ci=ci, share_draws_ge0=p_up,
                                         turns_seed=int(s["turns"].sum()), turns_skill=int(o["turns"].sum()), eps_seed=int(s["eps"].sum()), eps_skill=int(o["eps"].sum())))
            print(model, row, "n", len(tasks), "matches Table 1" if same else check, flush=True)
    (OUT / "calls_rw.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] not in (["extract"], ["stats"]):
        sys.exit("usage: python -m aux.calls_and_writes.calls_rw extract|stats")
    {"extract": extract, "stats": stats}[sys.argv[1]]()
