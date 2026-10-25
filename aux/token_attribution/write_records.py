"""Write the three figure records (bip_view.md, bip_split.md, bip_writes.md) from the numbers the figure scripts saved, into OUT/token_attribution."""
import json

from aux.token_attribution.bipcommon import OUT as O

CMD = ("python -m aux.token_attribution.bipdata     # once: collects the episode graphs (read-only)\n"
       "python -m aux.{script}")
SHORT = {"tau3": "tau3-bench", "AutoB": "AutoBench", "TB2": "TB2"}
MODEL = {"deepseek-flash": "DeepSeek-V4.1-Flash", "gpt-6-luna": "GPT-6-Luna"}


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    # bip_view
    v = json.loads((O / "bip_view_numbers.json").read_text())
    rows = []
    for k, d in v.items():
        model, short = k.split("|")
        rows.append([MODEL[model], SHORT[short], f"{d['n_seed']} ({d['n_tasks_seed']} tasks)", d.get("n_learned", "-"), d["rmax"], d["n_at_rmax_seed"],
                     f"{100 * d['median_prefix_share_seed']:.0f}%", f"{d['median_turn_elasticity_seed']:.2f}", f"{100 * d['prefix_share_seed']:.0f}%",
                     f"{d['mean_R_seed']:.1f}" + (f" / {d['mean_R_learned']:.1f}" if "mean_R_learned" in d else ""),
                     (f"{d['skill_tokens_seed']:.0f} / {d['skill_tokens_learned']:.0f}" if "skill_tokens_learned" in d else f"{d.get('skill_tokens_seed', 0):.0f} / -") if "skill_tokens_seed" in d else "-",
                     (f"{d['top_seed_last'] / 1e3:.1f}k / {d['top_learned_last'] / 1e3:.1f}k" if "top_learned_last" in d else "-")])
    view = f"""# bip_view

Proposed figure; not in the paper. Recorded baseline runs only, no model calls.

## What it shows

The episode graph (turns x context components) in aggregate. Each recorded test episode is a graph between its turns and its context components; here the graph is averaged over episodes. At turn r the figure shows the mean number of input tokens the turn reads from each component, over the episodes still running at turn r. Components: rest of the prompt (system prompt, tool schemas, task), skill text, tool outputs (all tools), agent and user messages (everything else in the conversation: the agent's own text, calls and reasoning, user turns). Stacked areas are the seed; the dashed line is the total at that turn for the learned skills of the three optimizers pooled. Strip under each panel: share of episodes still running. Rows are the two models, columns the three benchmarks.

## Claims it supports (and does not)

- The re-reading mechanism: every turn re-reads a fixed prefix plus a conversation that has grown, shown on real episodes for all benchmarks and both models. Measured: the prefix layers are flat from turn to turn, the tool-output and message layers grow with the turn.
- Why the turn elasticity differs by benchmark: the median episode's prompt-and-skill share of its input is 91% on tau3 (DeepSeek), 27% on AutoBench, 44% on TB2; the median turn elasticity of the seed episodes recomputed from the same graphs is 1.10, 1.77, 1.59 (the skill-against-seed pairs of the slope analysis give medians 1.11, 1.76, 1.64; the small gap is the pairing, inferred).
- Learned skills read more at (almost) every turn on DeepSeek: skill-text layer 0.9 to 1.05k tokens per turn for the seed against 1.5 to 2.4k for the learned skills (mean over the turns shown), plus a tail that goes on longer.
- Cross-model caution: the allocator books the model's own earlier output items, including the encrypted reasoning items that GPT-6-Luna's request records carry (checked on one Luna AutoBench request record: 18 reasoning items among 71 input items), under the history component, so Luna's message layer is larger than DeepSeek's for that reason (inferred for the size, measured for the items). Compare totals across models, not the tool-output / message split.
- Does NOT test that the history grows evenly within a task (the within-task turn slope of mech_slopes_input does that). The curves average over different tasks, so at late turns they are dominated by the few long, output-heavy tasks (tau3's late tool-output spike is that, inferred).

## Runs

DeepSeek-V4.1-Flash: the paper's test files exactly (skills native-20261008-https, rep 0 plus test-reps rep 1 and 2; seed native-20261003-trace-final plus rep-1 and rep-2), episode graphs from src.graph.episode.load. GPT-6-Luna: files chosen as Table 1 does (attribution_luna.luna_side: src.run repeat merge, verified outcomes only), graphs from attribution_luna's graph_cache.pkl (provider-failed requests already removed). Luna TB2: all three optimizers returned the seed, so only the seed is drawn. Learned = GEPA, SkillOpt, SkillAdam pooled (each three runs per task).

## Aggregation

All recorded test episodes with a graph (not only the paper's four-term task set), three runs per task. Components by the allocator of src/graph/allocate.py: each request's provider token count is split over components by estimated content length (4 characters per token) and rounded so the parts add up exactly; so per-turn totals are exact and the split inside a turn is an estimate. Turns shown: 1 to the last turn at which at least 30% of the seed's episodes are still running. y range shared by the two models of a benchmark.

## Numbers

{md_table(["model", "benchmark", "seed episodes", "learned episodes", "last turn shown", "seed episodes still running there", "median episode: prompt+skill share of input", "median turn elasticity (seed)", "token-weighted prompt+skill share (seed)", "mean turns seed / learned", "skill tokens per turn seed / learned", "total at last turn seed / learned"], rows)}

Note (inferred): on tau3 the median episode is prefix-heavy (91%), but the token-weighted share over all seed tokens is only 35%, because the long episodes with large tool outputs carry most tokens. The statement "short dialogues make the fixed prefix most of the input" holds for the median episode, not for the token total.

## Regenerate

```
{CMD.format(script="token_attribution.fig_view")}
```
Files: bip_view.pdf / .svg / .png; numbers in bip_view_numbers.json.
"""
    (O / "bip_view.md").write_text(view)

    # bip_split
    s = json.loads((O / "bip_split_numbers.json").read_text())
    rows_b, rows_c = [], []
    for sc in s["scopes"]:
        b, c, bt = sc["behavior"], sc["content"], sc["boot"]
        name = f"{MODEL[sc['model']]}, " + ("all skills" if sc["scope"] == "all" else SHORT[sc["scope"]])
        rows_b.append([name, sc["n_skills"], f"{b['more_actions']:+.3f}", f"{b['turn_splitting']:+.3f}", f"{b['content']:+.3f}", f"{b['skill_text']:+.3f}", f"{b['total']:+.3f}",
                       f"[{bt['total'][0]:+.3f}, {bt['total'][1]:+.3f}]", f"{b['more_actions'] + b['turn_splitting']:+.3f} [{bt['turns'][0]:+.3f}, {bt['turns'][1]:+.3f}]",
                       f"{bt['text'][0]:+.3f} to {bt['text'][1]:+.3f}", f"{100 * bt['p_turns_gt_text']:.1f}%"])
        rows_c.append([name, f"{c['skill']:+.3f}", f"{c['prompt']:+.3f}", f"{c['tools']:+.3f}", f"{c['messages']:+.3f}", f"{c['total']:+.3f}",
                       f"{100 * c['skill'] / c['total']:.0f}% / {100 * c['prompt'] / c['total']:.0f}% / {100 * c['tools'] / c['total']:.0f}% / {100 * c['messages'] / c['total']:.0f}%"])
    chk = max(x["max_abs_content_vs_measured"] for x in s["checks"])
    chk2 = max(x["max_abs_measured_vs_json"] for x in s["checks"])
    split = f"""# bip_split

Proposed figure; not in the paper. Recorded baseline runs only, no model calls.

## What it shows

The four-term decomposition beyond the DeepSeek per-skill heatmap (mech_attribution3), cut two ways on the same totals.
(a) The Luna twin of the DeepSeek heatmap: per Luna skill, the four-term split of the change in log input tokens against the Luna seed (same colour scale as the DeepSeek heatmap: largest absolute cell there, {0.27:.2f}).
(b) The four terms summed over skills (the paper's pooled sum), for DeepSeek (all nine skills; per benchmark) and Luna (all six; per benchmark). Black diamond = measured total with its 95% task-bootstrap interval. The column of percentages is the share of bootstrap draws in which the two turn terms (more actions plus turn-splitting) exceed the skill-text term.
(c) The same totals cut by content (the exact split by context component): which context component carries the change in log input tokens (skill text, rest of the prompt, tool outputs, agent and user messages).

## Claims it supports

- Pooled split and the "73% vs 21%" (DeepSeek) with an interval: no figure shows it today (the DeepSeek heatmap shows the nine cells, not their sum). Measured: the nine skills sum to {s['scopes'][0]['behavior']['total']:+.3f} in log input tokens; the two turn terms {s['scopes'][0]['behavior']['more_actions'] + s['scopes'][0]['behavior']['turn_splitting']:+.3f}, skill text {s['scopes'][0]['behavior']['skill_text']:+.3f}; turn terms exceed skill text in {100 * s['scopes'][0]['boot']['p_turns_gt_text']:.0f}% of draws (paper: 97%).
- "Under Luna the turn terms again outweigh the text": no figure today. Measured: Luna sums to {s['scopes'][4]['behavior']['total']:+.3f}; turn terms {s['scopes'][4]['behavior']['more_actions'] + s['scopes'][4]['behavior']['turn_splitting']:+.3f}, skill text {s['scopes'][4]['behavior']['skill_text']:+.3f}, content per turn {s['scopes'][4]['behavior']['content']:+.3f}. Because content per turn falls, the turn terms exceed the net total; plotted in absolute log units, not shares (the share is 167% and meaningless, see mech_attribution_luna.md). Luna tau3 alone is not distinguishable from zero.
- The honest nuance the pooled number hides (measured): on DeepSeek tau3 the turn terms and the skill text are even (0.18 vs 0.18; turns exceed text in 56% of draws); the 73% is carried by AutoBench and TB2.
- "Which content carries log rho": the exact content split is defined but no figure or table reports it on real data. Measured: the groups sum to the measured change to within {chk:.0e} on every one of the 15 cells, and the measured change equals the paper's per-task values to within {chk2:.0e}.
- Inferred: the skill-text bucket of the content split (DeepSeek {s['scopes'][0]['content']['skill']:+.3f}) is larger than the skill-text term of the behavior split ({s['scopes'][0]['behavior']['skill_text']:+.3f}) because the content bucket also counts the extra turns re-reading the skill, while the behavior term holds the number of turns fixed. The two are not the same quantity.
- Inferred: the rest-of-the-prompt bucket ({s['scopes'][0]['content']['prompt']:+.3f} on DeepSeek) is text that no skill changes; it enters only through extra turns re-reading it.

## Runs

attribution_luna.json (per-task terms of the paper's split for the nine DeepSeek and six Luna skills, DeepSeek skills native-20261008-https plus test-reps, seed native-20261003-trace-final; Luna skills native-20261012, seed native-20261011, src.run repeat merge) for (a), (b); the 3-run mean tokens per component group of the same tasks, from the episode graphs (bipdata.py), for (c). Luna TB2: every optimizer returned the seed, no row.

## Aggregation

Per cell (skill, benchmark) and task: terms as in mech_attribution3 (3-run means of R, N, X, F; midpoint shares; e_R). Pooled value = sum over the skills of a scope of the per-skill mean over tasks (the paper's definition of the 73% and 21%). Content split per task: (Xbar'_g - Xbar_g) / mu for each group g, mu the logarithmic mean of the two sides' 3-run mean input tokens, then the same task mean and sum. Interval: 10,000 bootstrap draws that resample tasks within each benchmark with the same draw for every skill of that benchmark (the procedure of attribution_luna.bootstrap, own random seed); it reproduces the record's intervals (DeepSeek total [{s['scopes'][0]['boot']['total'][0]:+.3f}, {s['scopes'][0]['boot']['total'][1]:+.3f}] against [+0.807, +2.225]; Luna total [{s['scopes'][4]['boot']['total'][0]:+.3f}, {s['scopes'][4]['boot']['total'][1]:+.3f}] against [+0.045, +0.446]). Component groups come from the allocator (split inside a turn by estimated content length, totals exact).

## Numbers

### (b) Behavior split, summed over skills (log units); tasks = skills x tasks used

{md_table(["scope", "skills", "more actions", "turn-splitting", "content per turn", "skill text", "total (measured)", "95% interval of total", "turn terms [95%]", "skill text 95%", "draws with turn terms > skill text"], rows_b)}

### (c) Content split of the same totals

{md_table(["scope", "skill text", "rest of the prompt", "tool outputs", "agent and user messages", "total", "shares of the total: skill / prompt / tools / messages"], rows_c)}

## Regenerate

```
{CMD.format(script="token_attribution.fig_split")}
```
Files: bip_split.pdf / .svg / .png; numbers in bip_split_numbers.json.
"""
    (O / "bip_split.md").write_text(split)

    # bip_writes
    w = json.loads((O / "bip_writes_numbers.json").read_text())
    rows = []
    for k, d in w["agg"].items():
        model, row = k.split("|")
        rows.append([MODEL[model], row, d["n"], f"{100 * d['seed.Wpar']:.1f}%", f"{100 * d['skill.Wpar']:.1f}% [{100 * d['ci_Wpar_skill'][0]:.1f}, {100 * d['ci_Wpar_skill'][1]:.1f}]",
                     f"{100 * d['seed.Rpar']:.1f}%", f"{100 * d['skill.Rpar']:.1f}% [{100 * d['ci_Rpar_skill'][0]:.1f}, {100 * d['ci_Rpar_skill'][1]:.1f}]", f"{d['seed.m']:.2f} -> {d['skill.m']:.2f}"])
    cs = w["case"]
    writes = f"""# bip_writes

Proposed figure; not in the paper. Recorded baseline runs only, no model calls.

## What it shows

The call edges of the graph (what each turn executes) for the learned skills' rules that serialize writes.
(a) One AutoBench task, DeepSeek-V4.1-Flash, run 0: the teaser pair (finance-vendor-payment-approval, seed against SkillOpt's skill; 14 turns and 27 calls against 21 turns and 33 calls, the same counts as the teaser). Every turn is a column and every executed tool call of that turn a square: red write (changes the simulated world), blue read, gray other. The task was pinned to tie it to the teaser, not picked for this figure. The seed issues eight writes in four turns (two turns hold three writes each); SkillOpt's skill issues six writes in six turns.
(b) All AutoBench test episodes, both models: the share of an agent's writes that share a turn with another write, and of its reads that share a turn with another read. Hollow dot: the seed on the same tasks; filled dot: the row's agent; whiskers: 95% paired-bootstrap interval. First row of each model: the no-skill agent as a placebo (it has no serialize rule).

## Claims it supports

- "On AutoBench 55% of the seed's writes share a turn with another write, against at most 0.3% under each learned skill, which issues writes one per turn while reads stay grouped": no figure today (mech_calls.pdf, the 0/1/2+ calls per turn bars, is on disk but not included; it does not separate writes from reads). measured: (DeepSeek): seed {100 * w['agg']['deepseek-flash|SkillOpt']['seed.Wpar']:.1f}%, GEPA {100 * w['agg']['deepseek-flash|GEPA']['skill.Wpar']:.1f}%, SkillOpt {100 * w['agg']['deepseek-flash|SkillOpt']['skill.Wpar']:.1f}%, SkillAdam {100 * w['agg']['deepseek-flash|SkillAdam']['skill.Wpar']:.1f}%; reads grouped {100 * w['agg']['deepseek-flash|SkillOpt']['seed.Rpar']:.0f}% in the seed and 73 to 77% under the skills; no skill: {100 * w['agg']['deepseek-flash|No skill']['skill.Wpar']:.1f}% (same as the seed).
- Luna ("the wording of the rule sets the size"): measured: seed about 50%, SkillOpt {100 * w['agg']['gpt-6-luna|SkillOpt']['skill.Wpar']:.1f}%, SkillAdam {100 * w['agg']['gpt-6-luna|SkillAdam']['skill.Wpar']:.1f}%, GEPA {100 * w['agg']['gpt-6-luna|GEPA']['skill.Wpar']:.1f}% (GEPA's rule only serializes writes within one app).
- The rule-to-behavior link: the rules named writes and the writes changed, reads did not. This is observational (no run removes a rule), as the paper says.
- Turn-splitting: panel (a) is the four-term split's second term on one episode: the same writes spread over more turns.

## Runs

AutoBench test episodes of Table 1's avg@3 runs, DeepSeek-V4.1-Flash and GPT-6-Luna, rows no skill, seed, GEPA, SkillOpt, SkillAdam: the data and numbers of aux/calls_and_writes (calls_rw_data.pkl, calls_rw.json, variant 'main'; record mech_calls_rw.md). Call classes as in that record (write = POST, PUT, PATCH, DELETE or a state-changing named tool; read = GET, search, read-only shell or file tools; other = local helpers). Panel (a): run 0 of the seed and of SkillOpt, DeepSeek.

## Aggregation

Per (model, row against seed): the tasks of Table 1's common set for that pair, the three runs pooled, shares over turns; the seed dot is the seed on the same tasks, so it differs a little between rows. n tasks in the table; for Luna GEPA (61) and SkillAdam (78) the 'main' variant has fewer than 120 tasks. measured: rule (calls_rw.py, common_tasks and the extract step): 'main' keeps only episodes whose record passes the capture-complete gate of dataloader.task_row (verified outcome, N and X all known) on both sides, while the variant 'all_recorded_turns' drops that gate and keeps 120 tasks (GEPA 13.6%, SkillAdam 6.1%, same conclusion). Unknown: why so many Luna GEPA and SkillAdam AutoBench episodes fail the gate; the four-term split uses 120 tasks for the same skills because it takes N and X from the Luna extraction, which does not use that gate.

## Numbers

{md_table(["model", "row", "tasks", "writes sharing a turn, seed", "same, row's agent [95%]", "reads sharing a turn, seed", "same, row's agent [95%]", "calls per turn seed -> row"], rows)}

Panel (a) turns (R read, W write, O other, - no call):
- seed: {" ".join(cs['seed']['turns'])}  (R={cs['seed']['R']}, N={cs['seed']['N']}, X={cs['seed']['X']:.0f})
- SkillOpt: {" ".join(cs['skillopt']['turns'])}  (R={cs['skillopt']['R']}, N={cs['skillopt']['N']}, X={cs['skillopt']['X']:.0f})
Other runs of the same task (record only): seed run 1 has six writes in one turn, SkillOpt runs 1 and 2 have six writes in six turns, SkillAdam run 0 six in six, GEPA run 2 alternates write and read for twelve turns.

## Regenerate

```
{CMD.format(script="calls_and_writes.fig_writes")}
```
Files: bip_writes.pdf / .svg / .png; numbers in bip_writes_numbers.json.
"""
    (O / "bip_writes.md").write_text(writes)
    print("ok")


if __name__ == "__main__":
    main()
