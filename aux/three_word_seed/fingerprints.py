"""(e) Behaviour fingerprints on AutoBench test episodes (GPT-6-Luna): 3-word seed, hand-written seed, SkillOpt-from-scratch, GEPA-from-scratch.

Reuses the paper's own call classifier and statistics (aux/calls_and_writes/calls_rw.py: classify, task_arrays, common_tasks, stats_of; Wpar = writes that sit in a
turn with >= 2 writes / all writes) and its episode extraction (table1.merged_episodes per repetition, distinct request ids, executed calls).
Episodes: every test evaluation file of each agent's three repetitions (test-reps/<...>/results.json runs[].evaluation_paths), 120 tasks x 3 reps.

Variants of the 'writes sharing a turn' share:
  all_recorded : every episode that has at least one recorded request (executed is not False; requests with input_tokens) -- the 'all_recorded_turns' variant of mech_calls_rw.md
  main         : only capture-complete episodes (task_row: verified outcome, N and X known) on the tasks common to the agent and the reference seed -- the paper's 'main' variant
Read-back: object-level identity is not recoverable from the recorded calls (the id of a created object exists only in the tool response), so
the service-level proxy is computed: a read-class call in a strictly later turn that targets the same service (URL host; named Zapier tool: app prefix) as the write.
Run: python -m aux.three_word_seed.fingerprints
"""
import collections, json, re
from pathlib import Path
import numpy as np

from aux.calls_and_writes import calls_rw as CR
from aux.calls_and_writes.callsperturn import bash_command
from src import layout
from src.analysis import table1 as T
from src.utils.dataloader import task_row

OUT = layout.OUT / "three_word_seed"
EXE = layout.EXECUTION / "gpt-6-luna"

AGENTS = {
    "3-word seed (cf7b932190ed)": "automationbench__native-seed__fromscratch",
    "hand-written seed (4fd5ca441c55)": "automationbench__native-seed__native-20261011",
    "SkillOpt from scratch (bb509d254dad)": "automationbench__skillopt__fromscratch",
    "GEPA from scratch (52d485532050)": "automationbench__gepa__fromscratch",
}
HOST = re.compile(r"""(?i)(?:--url[=\s]+|["']url["']\s*:\s*)['"]?https?://([^/\s'"?]+)""")
HOST_ANY = re.compile(r"https?://([A-Za-z0-9.\-]+)")
NAMED_APP = re.compile(r"automationbench_cli\.py\W+(?:call|invoke|execute_tool)\b[^\n]*?--(?:name|tool|tool-name)[=\s]+['\"]?([A-Za-z0-9_.\-]+)")


def services(tool, arguments):
    """Service keys a call targets: URL hosts of api_fetch commands, or the app prefix of a named tool."""
    if tool != "bash":
        return set()
    cmd = bash_command(arguments)
    keys = set(h.lower() for h in HOST.findall(cmd))
    if not keys and "api_fetch" in cmd:
        keys |= set(h.lower() for h in HOST_ANY.findall(cmd))
    for name in NAMED_APP.findall(cmd):
        if name != "api_fetch":
            keys.add("tool:" + re.split(r"[_.]", name)[0].lower())
    return keys


def load(tag):
    rr = json.load(open(EXE / "test-reps" / tag / "results.json"))
    reps = {}
    for r in rr["runs"]:
        eps = T.merged_episodes([Path(p) for p in r["evaluation_paths"]])
        per = {}
        for (task, _), e in eps.items():
            tr = task_row(e)
            complete = not (tr["success"] is None or tr["N"] is None or tr["X"] is None)
            seen, turns, detail = set(), [], []
            for q in e.get("requests", []):
                if q["request_id"] in seen:
                    continue
                seen.add(q["request_id"])
                if not complete and q.get("input_tokens") is None:
                    continue
                cs = [c for c in q.get("tool_calls", []) if (c.get("executed") is True if complete else c.get("executed") is not False)]
                turns.append([CR.classify(c["tool"], c.get("arguments") or "") for c in cs])
                detail.append([(c["tool"], c.get("arguments") or "") for c in cs])
            if not turns:
                continue
            per[task] = dict(turns=turns, N=tr["N"], X=tr["X"], R=tr["nreq"] if complete else len(turns), success=tr["success"], complete=complete, detail=detail)
        reps[int(r["repetition"])] = per
    return reps


def wpar(reps, tasks=None, only_complete=False):
    """(writes, writes in turns with >= 2 writes, turns, turns with >= 2 writes, episodes, episodes with a write)."""
    W = Wp = turns = W2 = eps = epsw = 0
    for rep in reps.values():
        for t, ep in rep.items():
            if (tasks is not None and t not in tasks) or (only_complete and not ep["complete"]):
                continue
            eps += 1
            has = False
            for turn in ep["turns"]:
                w = sum(c == "W" for c in turn)
                turns += 1; W += w
                if w >= 2:
                    Wp += w; W2 += 1
                has |= w > 0
            epsw += has
    return dict(writes=W, writes_in_multi_write_turns=Wp, share=Wp / W if W else float("nan"), turns=turns, multi_write_turns=W2, episodes=eps, episodes_with_write=epsw)


def readback(reps):
    """Service-level read-after-write.  Per episode, calls ordered by (turn, position).  A write at turn i is 'read back (same service)' if some read-class call at turn j > i
    targets a service key shared with the write; 'read back (any)' if any read-class call exists at turn j > i.  Write-level: share of writes; episode-level: share of
    episodes with >= 1 write whose LAST write is read back (any / same service), and share where every write is read back (same service)."""
    nW = rb_any = rb_svc = 0
    eps = e_last_any = e_last_svc = e_all_svc = 0
    unk = 0   # writes whose call text yields no service key (cannot be tested at service level)
    for rep in reps.values():
        for t, ep in rep.items():
            turns, detail = ep["turns"], ep["detail"]
            writes = []   # (turn, keys)
            for i, (cls, det) in enumerate(zip(turns, detail)):
                for c, (tool, args) in zip(cls, det):
                    if c == "W":
                        writes.append((i, services(tool, args)))
            if not writes:
                continue
            reads = [(j, services(tool, args)) for j, (cls, det) in enumerate(zip(turns, detail)) for c, (tool, args) in zip(cls, det) if c == "R"]
            res = []
            for i, keys in writes:
                later = [(j, k) for j, k in reads if j > i]
                a = bool(later)
                s = any(k & keys for j, k in later) if keys else False
                unk += not keys
                res.append((a, s))
            nW += len(res); rb_any += sum(a for a, _ in res); rb_svc += sum(s for _, s in res)
            eps += 1
            last_turn = max(i for i, _ in writes)
            last = [r for (i, _), r in zip(writes, res) if i == last_turn]
            e_last_any += any(a for a, _ in last); e_last_svc += any(s for _, s in last)
            e_all_svc += all(s for _, s in res)
    return dict(writes=nW, writes_with_later_read_any=rb_any, writes_read_back_same_service=rb_svc, writes_without_service_key=unk,
                share_writes_later_read_any=rb_any / nW, share_writes_read_back_same_service=rb_svc / nW,
                episodes_with_write=eps, share_eps_last_write_followed_by_read_any=e_last_any / eps, share_eps_last_write_read_back_same_service=e_last_svc / eps,
                share_eps_every_write_read_back_same_service=e_all_svc / eps)


def boot_share(reps, tasks, B=2000, seed=20261024):
    """Task-level bootstrap interval of the Wpar share (episodes pooled per task)."""
    rng = np.random.default_rng(seed)
    per = {t: (sum(sum(c == "W" for c in turn) for rep in reps.values() if t in rep for turn in rep[t]["turns"]),
               sum(sum(c == "W" for c in turn) for rep in reps.values() if t in rep for turn in rep[t]["turns"] if sum(x == "W" for x in turn) >= 2)) for t in tasks}
    arr = np.array([per[t] for t in tasks], float)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(tasks), len(tasks))
        s = arr[idx].sum(0)
        vals.append(s[1] / s[0] if s[0] else np.nan)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load(tag) for name, tag in AGENTS.items()}
    out = {"agents": {}, "paired_main": {}}
    for name, reps in data.items():
        tasks_all = sorted({t for rep in reps.values() for t in rep})
        out["agents"][name] = dict(tag=AGENTS[name], episodes_loaded=sum(len(r) for r in reps.values()),
                                   episodes_complete=sum(ep["complete"] for r in reps.values() for ep in r.values()),
                                   wpar_all_recorded=wpar(reps), wpar_complete_only=wpar(reps, only_complete=True), wpar_all_ci95=boot_share(reps, tasks_all),
                                   readback=readback(reps), readback_complete_only=readback({k: {t: e for t, e in v.items() if e["complete"]} for k, v in reps.items()}))
        a = out["agents"][name]
        print(name, "eps", a["episodes_loaded"], "complete", a["episodes_complete"], "| Wpar all %.4f (%d/%d) ci %s | complete-only %.4f (%d/%d)" % (
            a["wpar_all_recorded"]["share"], a["wpar_all_recorded"]["writes_in_multi_write_turns"], a["wpar_all_recorded"]["writes"], [round(x, 4) for x in a["wpar_all_ci95"]],
            a["wpar_complete_only"]["share"], a["wpar_complete_only"]["writes_in_multi_write_turns"], a["wpar_complete_only"]["writes"]))
        rb = a["readback"]
        print("   read-back: writes later-read(any) %.3f, same-service %.3f | eps last write followed by read(any) %.3f, same-service %.3f, every write same-service %.3f (n eps %d, writes %d, no-key writes %d)" % (
            rb["share_writes_later_read_any"], rb["share_writes_read_back_same_service"], rb["share_eps_last_write_followed_by_read_any"], rb["share_eps_last_write_read_back_same_service"],
            rb["share_eps_every_write_read_back_same_service"], rb["episodes_with_write"], rb["writes"], rb["writes_without_service_key"]))
    # paired 'main' variant (paper's definition: common task set with the reference, capture-complete episodes, stats_of)
    names = list(AGENTS)
    for ref in (names[0], names[1]):
        for name in names:
            if name == ref:
                continue
            tasks, a, b = CR.common_tasks(data[ref], data[name])
            s, o = CR.task_arrays(data[ref], tasks, False, True), CR.task_arrays(data[name], tasks, False, True)
            pt = CR.stats_of(s, o, np.arange(len(tasks)))
            rng = np.random.default_rng(20261024)
            bs = collections.defaultdict(list)
            for _ in range(2000):
                idx = rng.integers(0, len(tasks), len(tasks))
                r = CR.stats_of(s, o, idx)
                for side in ("seed", "skill"):
                    for k in ("Wpar", "Rpar"):
                        bs[side + "." + k].append(r[side][k])
            out["paired_main"][f"{name} vs ref {ref}"] = dict(n_tasks=len(tasks), ref_Wpar=pt["seed"]["Wpar"], agent_Wpar=pt["skill"]["Wpar"], ref_Rpar=pt["seed"]["Rpar"], agent_Rpar=pt["skill"]["Rpar"],
                                                              ci_ref_Wpar=[float(np.percentile(bs["seed.Wpar"], q)) for q in (2.5, 97.5)], ci_agent_Wpar=[float(np.percentile(bs["skill.Wpar"], q)) for q in (2.5, 97.5)],
                                                              calls_per_turn_ref=pt["seed"]["m"], calls_per_turn_agent=pt["skill"]["m"])
            v = out["paired_main"][f"{name} vs ref {ref}"]
            print("paired main:", name, "vs", ref, "n", v["n_tasks"], "Wpar ref %.4f agent %.4f ci_agent %s | Rpar ref %.3f agent %.3f" % (v["ref_Wpar"], v["agent_Wpar"], [round(x, 4) for x in v["ci_agent_Wpar"]], v["ref_Rpar"], v["agent_Rpar"]))
    out["ambiguity_counters"] = dict(CR.AMBIG)
    json.dump(out, open(OUT / "fingerprints.json", "w"), indent=1)


if __name__ == "__main__":
    main()
