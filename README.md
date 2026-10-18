# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

Code of the paper: it trains and evaluates the three skill optimizers the paper studies (GEPA, SkillOpt, SkillAdam) plus Seed and No-skill controls — five arms — on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna), and reproduces every table and figure from the recorded runs.

## Motivation

Skill optimizers report success alone, and a success score on a small dev set cannot settle whether a learned skill fits its task family better:

- **Run-to-run noise.** One test run can rank two skills either way; a paired comparison of three runs detects only differences of 7.1 / 5.2 / 7.3 points on τ3-bench, AutoBench and TB2.
- **Selection inflates gains.** Keeping the best of several noisy dev comparisons is a winner's curse: replaying each optimizer's own acceptance rule on candidates identical to the seed reports 1.2 to 5.1 dev tasks of expected gain against the observed 1 to 5.
- **Success hides behavior and cost.** A skill acts through the agent's turns and what they read; whether a gain survives reruns, and what the skill makes the agent do and spend, go unchecked.

## Findings

- **The success gains do not hold (RQ1).** Dev gains of 3 to 40 points end at test differences of −11.1 to +2.7 points over three runs, eight of the nine negative (Fig. 5).
- **Every learned skill levies a token tax (RQ2).** All nine DeepSeek learned skills read 1.08–1.43× the seed's input tokens per task and take 1.03–1.22× its turns, on 59–87% of tasks per skill (Table 1, Fig. 4). Extra turns carry 73% of the summed log-token increase, the skill text 21%; within a task, input tokens rise with turns at slopes 1.35–1.58 while outputs and tool calls rise at ≤1.03 (Fig. 3, 6a).
- **The optimizers accumulate procedure (RQ3).** 58 of 59 traceable edits lengthen the skill (654–2,447 words against the seed's 580); of 337 added rules, 52% verify and none asks the agent to read less; kept and rejected edits cost alike — the cost arrives with the proposals (Fig. 6b, 7d).
- **The findings generalize (RQ4).** On GPT-6-Luna the pattern repeats (on TB2 all three optimizers return the seed unchanged); frozen skills carry their tax to an unseen benchmark (five of six transfers read 1.23–1.47× the seed's tokens) and to the second model (Table 1 lower panel, Tables 2–3, Fig. 16).

## How BiPA works

A harness without compaction re-sends the prefix and the whole conversation at every turn, so an episode of R turns reads `X = R·F + R(R−1)/2·h` input tokens — each extra turn re-reads the whole history, and input tokens grow faster than turns.

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components (skill text, rest of the prompt, each tool's returned content), with two streams: what the model reads (input tokens) and what the harness executes (tool calls). The change in log input tokens between a skill and its seed splits into four behaviors: **more actions**, **turn-splitting**, **more content per turn**, **longer skill text**. Success is judged separately: avg@k test runs against the seed, plus a seed-replay control that applies the optimizer's own acceptance rule to candidates identical to the seed, isolating the dev gain selection alone reports.

Task execution, caching, retry and raw traces are owned by the external `harness-bench`; this repo keeps scheduling, upstream-interface adaptation (per-optimizer adapters with documented `DEVIATIONS`; upstream search, edit and acceptance rules unchanged), repeated evaluation, offline analysis and audit.

## Layout

```
bipa/
├── src/
│   ├── run.py        # one optimization cell (the external entry)
│   ├── core/         # launch, repeat, panels, protocol/trace, layout, config
│   ├── adapters/     # per-optimizer interfaces (skillopt, skilladam, gepa)
│   ├── analysis/     # table1.py (Table 1), mech/ and mechanism/ (RQ analyses)
│   ├── dynamics/     # transfer drivers, run-evidence audits, status/refresh
│   └── graph/        # per-episode bipartite-graph metrics
├── scripts/          # entry points: baselines.sh, rq4_transfer.sh, audit.sh, refresh.sh
├── optimizers/       # vendored upstream sources, pinned commits
├── configs/          # tasks / experiments / graph yaml
├── tests/            # 66 tests, fake model and tasks, no API calls
└── docs/             # investigation notes
```

The seed skill (`verification-before-completion`, 580 words) is read from the shared artifacts tree at `../assets/skills/seed/SKILL.md`. Raw runs live outside the repo at `../assets/artifacts/execution/<model>/<method>/<bench>/<run>/`; the offline steps only read them.

## Reproduction

Run from the repo root; `python -m` resolves `src.*`. Everything previews by default — `--execute` runs, `--detach` backgrounds. Benchmark aliases: `tau3`, `autob`, `tb2`. Only training, repetitions and transfer call the model; all analysis is offline.

**Main experiments** (the paper's study):

```bash
# 1. train the five arms x three benchmarks x two models (frozen native-v1 protocol)
bash scripts/baselines.sh suite deepseek-flash --execute
# 2. avg@3 repeated test evaluation of each arm's frozen final skill
bash scripts/baselines.sh repeat tau3 deepseek-flash --arm skillopt --execute
# 3. Table 1 (both models) -> ../paper/assets/data/mech/mech_table1.md
"$BASELINE_PYTHON" -m src.analysis.table1
```

A test slot that ran in all three repetitions but never got a verifier verdict scores 0 (`adjudicated_failures`, never retried); a slot with no episode evidence blocks the average (`src/core/repeat.py`).

**Secondary experiments**:

```bash
# RQ1-RQ4 analyses (offline; one module per figure/table, docstring has its subcommands)
"$BASELINE_PYTHON" -m src.analysis.mech.<module>          # or src.analysis.mechanism.<module>
# transfer: a returned skill evaluated unchanged under another model / benchmark
bash scripts/rq4_transfer.sh model|bench|tb2auto [--execute]
# run-evidence audits: completion certificate, trace integrity, skill history
bash scripts/audit.sh native-run|trace|skill-history ...
# refresh the derived tables under ../assets/artifacts/results/
bash scripts/refresh.sh
```

RQ mapping: RQ1 ← `mech/{stats,selection,domains}.py` + `mechanism/{nullmodel,paired,power}.py`; RQ2 ← `mech/{slopes,attribution,failcost,price,compaction,optcost}.py` + `mechanism/`; RQ3 ← `mech/{skilltext,rules_fig,coder2,callsperturn,edges*}.py`; RQ4 ← `mech/{transfer,dynamics}.py` + the transfer drivers. Each writes its audit record to `../paper/assets/data/mech/`.

## Requirements

Python 3.12 with `harness-bench`; the upstream optimizers' dependencies (`pip install -e` the pinned checkouts, see [optimizers/README.md](optimizers/README.md)). Interpreter via `BASELINE_PYTHON` (defaults to the harness-mitigation `bach-harness/.venv`; only the Python environment is reused, no BACH code is imported). `baselines.sh` sources an optional `host.env` for host settings. Tests: `"$BASELINE_PYTHON" -m pytest tests -q`.
