"""Writes OUT/data/mech/mech_calls_rw.md from calls_rw.json and boot_cf.json (OUT/calls_and_writes)."""
import json
import math

from src import layout
from src.utils.logger import DATA

D = layout.OUT / "calls_and_writes"
rw = None
MODEL = {"deepseek-flash": "DeepSeek", "gpt-6-luna": "Luna"}
RULES = {("deepseek-flash", "GEPA"): "6 serialize + 1 parallelize (reads only, never writes)", ("deepseek-flash", "SkillOpt"): "4 serialize",
         ("deepseek-flash", "SkillAdam"): "1 serialize", ("gpt-6-luna", "GEPA"): "0 serialize + 1 parallelize (parallel reads; writes sequential when they share an application)",
         ("gpt-6-luna", "SkillOpt"): "1 serialize", ("gpt-6-luna", "SkillAdam"): "2 serialize", ("deepseek-flash", "No skill"): "none (no skill)", ("gpt-6-luna", "No skill"): "none (no skill)"}


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def pct(x):
    return f"{100 * x:.1f}"


def ci(c, k, scale=100, nd=1, sign=True):
    p, (lo, hi) = c["point"][k], c["ci"][k]
    f = "+" if sign else ""
    return f"{scale * p:{f}.{nd}f} [{scale * lo:{f}.{nd}f}, {scale * hi:{f}.{nd}f}]"


def cells(variant):
    return [c for c in rw["cells"] if c["variant"] == variant]


def levels(variant):
    rows = []
    for c in cells(variant):
        p = c["point"]
        for side, eps, turns in (("seed", c["eps_seed"], c["turns_seed"]), ("skill", c["eps_skill"], c["turns_skill"])):
            label = "Seed" if side == "seed" else c["row"]
            rows.append([MODEL[c["model"]], c["row"] + " vs Seed", c["n"], label, eps, turns, f"{p[side + '.m']:.2f}", f"{p[side + '.w_turn']:.2f}", f"{p[side + '.r_turn']:.2f}",
                         f"{p[side + '.o_turn']:.2f}", pct(p[side + ".W2"]), pct(p[side + ".R2"]), pct(p[side + ".Wpar"]), pct(p[side + ".Rpar"]),
                         f"{p[side + '.W_ep']:.2f}", f"{p[side + '.R_ep']:.2f}"])
    return table(["model", "pair", "n tasks", "side", "episodes", "turns", "calls/turn", "writes/turn", "reads/turn", "other/turn", ">=2 writes % of turns",
                  ">=2 reads % of turns", "writes in a multi-write turn %", "reads in a multi-read turn %", "writes/episode", "reads/episode"], rows)


def changes(variant):
    rows = []
    for c in cells(variant):
        rows.append([MODEL[c["model"]], c["row"], RULES[(c["model"], c["row"])], c["n"], ci(c, "d_W2"), ci(c, "d_R2"), ci(c, "d_Wpar"), ci(c, "d_Rpar"), ci(c, "did_par"),
                     ci(c, "d_m", 1, 3)])
    return table(["model", "skill", "added rules on concurrency (mech_rules / mech_rules_luna)", "n", "d >=2-write turns pp", "d >=2-read turns pp",
                  "d writes in multi-write turn pp", "d reads in multi-read turn pp", "difference (writes - reads) pp", "d calls/turn"], rows)


def dose(variant):
    rows = []
    for c in cells(variant):
        e = lambda k: f"{__import__('math').exp(c['point'][k]):.3f} [{__import__('math').exp(c['ci'][k][0]):.3f}, {__import__('math').exp(c['ci'][k][1]):.3f}]"
        rows.append([MODEL[c["model"]], c["row"], RULES[(c["model"], c["row"])], c["n"], e("log_rho_calls"), e("log_rho_turns"), ci(c, "log_rho_calls", 1, 3) + " / " + ci(c, "log_rho_turns", 1, 3),
                     e("log_W_ratio"), e("log_R_ratio")])
    return table(["model", "skill", "concurrency rules", "n", "rho^calls", "rho^turns", "log rho^calls / log rho^turns", "writes/episode ratio", "reads/episode ratio"], rows)


def packing(variant):
    rows = []
    for c in cells(variant):
        p = c["point"]
        rows.append([MODEL[c["model"]], c["row"], c["n"], f"{p['writes_per_write_turn_seed']:.2f}", f"{p['writes_per_write_turn_skill']:.2f}", ci(c, "pack_saved_turns_ep", 1, 2),
                     ci(c, "log_turns_pooled", 1, 3), ci(c, "pack_log_turns_removed", 1, 3)])
    return table(["model", "skill", "n", "writes per write-carrying turn, seed", "same, skill", "turns saved / episode if packed like the seed",
                  "log turn ratio (pooled episodes)", "log turns removed by packing"], rows)


def wr(model):
    v = [math.exp(c["point"]["log_W_ratio"]) for c in cells("main") if c["model"] == model and c["row"] != "No skill"]
    return f"{min(v):.2f} to {max(v):.2f}"


def main():
    global rw
    rw = json.loads((D / "calls_rw.json").read_text())
    cf = json.loads((D / "boot_cf.json").read_text())
    amb = sorted(rw["ambig"].items(), key=lambda kv: -kv[1])
    amb_lines = "\n".join(f"- {k}: {v}" for k, v in amb)

    b = cf["batching"]
    bp, bc = b["point"], b["ci"]
    batch_rows = [["three skills (mean)", f"{bp['log_rho']:.3f} [{bc['log_rho'][0]:.3f}, {bc['log_rho'][1]:.3f}]", f"{bp['removed']:.3f} [{bc['removed'][0]:.3f}, {bc['removed'][1]:.3f}]",
                   f"{bp['after']:.3f} [{bc['after'][0]:.3f}, {bc['after'][1]:.3f}]", f"{bp['share_removed']:.2f} [{bc['share_removed'][0]:.2f}, {bc['share_removed'][1]:.2f}]"]]
    for j, k in enumerate(["SkillOpt", "SkillAdam", "GEPA"]):
        lr, rm = bp["per_cell_log_rho"][j], bp["per_cell_removed"][j]
        batch_rows.append([k, f"{lr:.3f} [{bc['per_cell_log_rho.' + k][0]:.3f}, {bc['per_cell_log_rho.' + k][1]:.3f}]", f"{rm:.3f} [{bc['per_cell_removed.' + k][0]:.3f}, {bc['per_cell_removed.' + k][1]:.3f}]",
                           f"{lr - rm:.3f} [{bc['after.' + k][0]:.3f}, {bc['after.' + k][1]:.3f}]", ""])
    batch_tab = table(["AutoBench", "log rho as recorded", "removed at the seed's calls per turn", "log rho after", "share removed"], batch_rows)

    comp_rows = [[c["method"], c["bench"], c["n"], f"{c['rho_inf']:.3f} [{c['rho_inf_ci'][0]:.3f}, {c['rho_inf_ci'][1]:.3f}]", f"{c['rho_w1']:.3f} [{c['rho_w1_ci'][0]:.3f}, {c['rho_w1_ci'][1]:.3f}]",
                  f"{c['rho_turns']:.3f}", f"{c['log_gap_w1_turns']:+.3f} [{c['log_gap_ci'][0]:+.3f}, {c['log_gap_ci'][1]:+.3f}]"] for c in cf["compaction"]]
    comp_tab = table(["skill", "bench", "n", "rho as recorded", "rho, last turn of history kept (window 1)", "rho^turns", "log(rho window 1 / rho^turns)"], comp_rows)
    w1 = sorted(cf["compaction"], key=lambda c: c["rho_w1"])
    lo_c, hi_c = w1[0], w1[-1]

    ds = {c["row"]: c for c in cells("main") if c["model"] == "deepseek-flash"}
    lu = {c["row"]: c for c in cells("main") if c["model"] == "gpt-6-luna"}

    text = f"""# mech_calls_rw

Rule -> behavior specificity, offline: do the serialize rules the optimizers add on AutoBench ("Issue MUTATING calls ONE AT A TIME (sequentially); parallel is safe only for READS") change exactly the behavior they name? Recorded episodes only, no model calls.
Written by `aux/calls_and_writes/calls_rw.py` (extract, stats), `boot_cf.py` (counterfactual intervals) and `write_record.py` (this file); numbers in `OUT/calls_and_writes/calls_rw.json` and `boot_cf.json`.

## What it shows

Each executed tool call of the AutoBench test episodes is classed as a write (changes the simulated world), a read, or other. Prediction if the rules act: the share of turns with two or more writes falls under every skill with a serialize rule, and the share with two or more reads does not. Then the dose contrast across skills (number and wording of the concurrency rules) against rho^calls, rho^turns and the write/read split; a packing counterfactual that says how many of the skills' extra turns the serialized writes account for; and bootstrap intervals for two counterfactuals already in the paper (re-billing at the seed's calls per turn; compaction to the last turn).

## Runs

Table 1's avg@3 test runs on AutoBench, exactly the evaluation files table1.json freezes (supplements merged with table1.merged_episodes): DeepSeek-V4.1-Flash (skills native-20261008-https rep 0 + test-reps rep 1/2; seed native-20261003-trace-final + rep-1/rep-2; no skill native-20261003-trace-fix + test-reps) and GPT-6-Luna (optimizers native-20261012, seed / no skill native-20261011, each + test-reps rep 1/2; SkillOpt AutoBench is the native-20261012 run, as in table1.json). Rows Seed, GEPA, SkillOpt, SkillAdam; No skill as a placebo (it has no concurrency rule).

## Classification rule (from the AutoBench tool set)

AutoBench reaches the world only through `automationbench_cli.py`, called from the shell tool: raw HTTP (`api_fetch --url ... --method ...`, also `invoke --tool api_fetch`), discovery (`api_search`, `search_tools`, `show_tools`, `describe`, `init`, `help`), named Zapier-style tools (`call --name`, `invoke --tool`, `execute_tool --tool-name`, e.g. `xero_create_sales_invoice`, `trello_find_card`), and a local helper `base64_encode`. Every mutating endpoint loads the world, applies the change and saves it, which is the stated reason of the rules.

- **write (W)**: an `api_fetch` with method POST, PUT, PATCH or DELETE (method = the first `method ... VERB` token of each api_fetch chunk; default GET, the CLI default), except the query-style POSTs below; a named tool whose name carries a state-changing verb (create, update, add, send, delete, post, set, move, archive, invite, ...) or no read verb.
- **read (R)**: an `api_fetch` GET; a POST to a query endpoint (path ending `:find`, `/search`, `/query`, `search.`, `.list`) or a GraphQL body without `mutation`; a named tool with a read verb (find, get, list, search, lookup, read, retrieve, fetch, count, ...); the discovery subcommands; the read, grep, glob, list tools; a shell command that is a pure read by the callsperturn rule (cat, head, tail, ls, grep, find, wc, sed -n; mech_callsperturn.md).
- **other (O)**: `base64_encode`, python or other shell commands that do not call the CLI, and local file tools (write, edit, apply_patch, todowrite; the sensitivity below counts write/edit as writes).
- A shell call holding several CLI invocations (chained with && or in a loop or script) is one call: a write if any invocation writes, else a read if any reads. Its invocations run in sequence, so it is not parallel writing.
- Turn = one model request (deduplicated by request id); calls = executed calls. Parallel = two or more calls of the class in the same turn (OpenCode runs the calls of one turn concurrently).

Ambiguous cases and how they were resolved (call counts over all extracted episodes of both models, all five rows):

{amb_lines}

## Aggregation

Per (model, row vs Seed): Table 1's common task set (per-task means over the row's usable runs with verified outcome and N, X > 0 on both sides); only the episodes that enter those means. For DeepSeek the script reproduces table1.json's n, rho^calls and rho (input tokens) for every pair; Luna's Table 1 rows drop provider-failed requests and keep the episode, while the Luna task sets here use complete logs only. Shares are pooled over the episodes of the task set (turns of all three runs). "Writes in a multi-write turn" = writes that sit in a turn with at least one other write, as a share of all writes (the same for reads); it does not move when a skill simply writes less. rho^calls, rho^turns: geometric mean over tasks of per-task 3-run means, as Table 1. Intervals: paired bootstrap over tasks (both sides resampled with the same tasks), 4,000 draws, 2.5 and 97.5 percentiles. Luna's capture is incomplete for many episodes (complete logs cover 61 GEPA and 78 SkillAdam tasks, against Table 1's 120); the sensitivity "all recorded turns" uses every episode with its completed requests (completed requests, calls not marked unexecuted) on all 120 tasks.

## Numbers

### A. Levels (Table 1 task sets)

{levels("main")}

### B. Change, skill minus seed, with paired bootstrap 95% intervals (percentage points; calls per turn in calls)

{changes("main")}

### C. Dose contrast: rule wording against rho^calls, rho^turns and the write/read split (Table 1 task sets)

{dose("main")}

### D. Packing counterfactual: the skill's writes packed at the seed's writes per write-carrying turn

Accounting only. Turns saved per episode = (write-carrying turns of the skill - the skill's writes / the seed's writes per write-carrying turn) / episodes; the log turn ratio here is of pooled turns per episode (it differs from the geometric-mean rho^turns of C). Times DeepSeek AutoBench's b_R = 1.18 (mech_batching), the removed log turns give the input-token effect.

{packing("main")}

### E. Sensitivity: local file writes (write, edit, apply_patch) counted as writes

{changes("local_writes_as_W")}

### F. Sensitivity: all recorded turns (every episode, completed requests; 120 tasks on both models)

{changes("all_recorded_turns")}

{dose("all_recorded_turns")}

### G. Bootstrap intervals for two counterfactuals already in the paper (DeepSeek, recorded episodes)

Re-billing at the seed's calls per turn (mech_batching; "halves its log rho, from 0.17 to 0.08"). Task-cluster bootstrap over AutoBench's {b['tasks']} tasks, {b['draws']} draws; each draw re-fits b_R (within-task regression of log input on log calls and log turns with skill dummies, design.fe_fit) on the drawn tasks and recomputes each skill's log rho and removed = mean(-b_R dlog m) on its drawn tasks. b_R = {bp['bR']:.3f} [{bc['bR'][0]:.3f}, {bc['bR'][1]:.3f}].

{batch_tab}

Compaction to the last turn of history (mech_compaction window 1; "a tax of 1.05 to 1.36"). Per skill, bootstrap over tasks ({2000} draws) of the window-1 rho and of its log gap to rho^turns (paired by task).

{comp_tab}

Range of the window-1 tax: {lo_c['rho_w1']:.2f} [{lo_c['rho_w1_ci'][0]:.2f}, {lo_c['rho_w1_ci'][1]:.2f}] ({lo_c['method']} {lo_c['bench']}) to {hi_c['rho_w1']:.2f} [{hi_c['rho_w1_ci'][0]:.2f}, {hi_c['rho_w1_ci'][1]:.2f}] ({hi_c['method']} {hi_c['bench']}); every lower bound is above 1. The gap to rho^turns excludes 0 for eight skills; for SkillAdam TB2 it is +0.011 [-0.034, +0.056], so "above each skill's turn ratio" holds at the point estimate only there.

## Reading

1. The serialize rules act, and act only on writes. On DeepSeek, under the seed 3.4% of turns carry two or more writes and 55% of writes share a turn with another write; under each learned skill (1 to 6 serialize rules) those fall to 0.0% and 0.0 to 0.3% (each change {ci(ds['GEPA'], 'd_Wpar')}, {ci(ds['SkillOpt'], 'd_Wpar')}, {ci(ds['SkillAdam'], 'd_Wpar')} points for GEPA, SkillOpt, SkillAdam). Parallel reads move by 3.1 points or less (reads in a multi-read turn {ci(ds['GEPA'], 'd_Rpar')}, {ci(ds['SkillOpt'], 'd_Rpar')}, {ci(ds['SkillAdam'], 'd_Rpar')}). The no-skill placebo writes in parallel as often as the seed ({ci(ds['No skill'], 'd_Wpar')}).
2. No dose gradient in the count of rules on DeepSeek: one unconditional rule (SkillAdam) removes parallel writes as completely as six (GEPA). The effect is saturated, so the count of rules is not the dose; the wording is.
3. The wording sets the dose on Luna. SkillOpt and SkillAdam (unconditional serialize rules) cut writes in a multi-write turn from about 50% to {pct(lu['SkillOpt']['point']['skill.Wpar'])}% and {pct(lu['SkillAdam']['point']['skill.Wpar'])}%. GEPA's only concurrency rule asks for parallel reads and sequential writes only "when they share an application or workbook": parallel writes fall less, to {pct(lu['GEPA']['point']['skill.Wpar'])}%, and parallel reads rise ({ci(lu['GEPA'], 'd_Rpar')} points; turns with two or more reads {ci(lu['GEPA'], 'd_R2')}). Calls per turn follow the sign of the rule: they rise under GEPA ({ci(lu['GEPA'], 'd_m', 1, 3)}; rho^calls 1.09 against rho^turns 0.98) and fall under SkillOpt ({ci(lu['SkillOpt'], 'd_m', 1, 3)}; 1.10 against 1.19) and SkillAdam ({ci(lu['SkillAdam'], 'd_m', 1, 3)}; 0.87 against 0.93). On DeepSeek GEPA's parallelize rule is phrased as a restriction ("Parallelize only read-only discovery ... never mutations") and parallel reads do not rise.
4. Serialized writes account for most of the AutoBench turn-splitting. Packing each DeepSeek skill's writes as densely as the seed does removes 0.07 log units of turns per skill ({ci(ds['GEPA'], 'pack_log_turns_removed', 1, 3)}, {ci(ds['SkillOpt'], 'pack_log_turns_removed', 1, 3)}, {ci(ds['SkillAdam'], 'pack_log_turns_removed', 1, 3)}), against pooled turn increases of 0.097, 0.093 and 0.018; times b_R = 1.18 that is 0.08 to 0.09 of log input, about the 0.083 the re-billing counterfactual removes (G). The re-billing counterfactual therefore mostly un-serializes writes, which the rules forbid for a stated correctness reason (concurrent read-modify-write calls drop earlier writes); the batchable part of the turn-splitting is the reads, and parallel reads did not fall. This qualifies the design principle "issue independent calls in one turn": the independent calls that can still be merged are reads.
5. Sensitivities agree. Counting local file writes as writes (E) changes no conclusion. On all recorded turns (F; Luna's 120 tasks with incomplete capture included) every skill still removes most parallel writes (-34 to -56 points) and GEPA on Luna still raises parallel reads (+6.8 [+5.4, +8.2]); there SkillOpt's parallel reads also fall (-4.1 [-6.1, -2.3]), a small effect next to its -47 points on writes.
6. The skills also write less: writes per episode fall to {wr('deepseek-flash')} of the seed's on DeepSeek and {wr('gpt-6-luna')} on Luna (C). The record does not establish why (fewer duplicated or retried writes are one possibility).

## Regenerate (from the repository root)

```
python -m aux.calls_and_writes.calls_rw extract
python -m aux.calls_and_writes.calls_rw stats
python -m aux.calls_and_writes.boot_cf
python -m aux.calls_and_writes.write_record
```
"""
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "mech_calls_rw.md").write_text(text)
    print(len(text))


if __name__ == "__main__":
    main()
