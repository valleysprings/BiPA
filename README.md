# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

Code of the paper: it trains and evaluates the three skill optimizers the paper studies (GEPA, SkillOpt, SkillAdam) plus Seed and No-skill controls on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna), and reproduces every table and figure from the recorded experiments.

## Motivation

Skill optimizers report success alone, and a success score on a small dev set cannot settle whether a learned skill fits its task family better:

- **Evaluation noise.** A single test evaluation can rank two skills either way; a paired comparison of three repetitions detects only differences of 7.1 / 5.2 / 7.3 points on τ3-bench, AutoBench and TB2.
- **Selection inflates gains.** Keeping the best of several noisy dev comparisons is a winner's curse: replaying each optimizer's own acceptance rule on candidates identical to the seed reports 1.2 to 5.1 dev tasks of expected gain against the observed 1 to 5.
- **Success hides behavior and cost.** A skill acts through the agent's turns and what they read; whether a gain survives repeated evaluation, and what the skill makes the agent do and spend, go unchecked.

## Findings

- **The success gains do not hold (RQ1).** Dev gains of 3 to 40 points end at test differences of −11.1 to +2.7 points over three repetitions, eight of the nine negative (Fig. 5).
- **Every learned skill levies a token tax (RQ2).** All nine DeepSeek learned skills read 1.08–1.43× the seed's input tokens per task and take 1.03–1.22× its turns, on 59–87% of tasks per skill (Table 1, Fig. 4). Extra turns carry 73% of the summed log-token increase, the skill text 21%; within a task, input tokens rise with turns at slopes 1.35–1.58 while outputs and tool calls rise at ≤1.03 (Fig. 3, 6a).
- **The optimizers accumulate procedure (RQ3).** 58 of 59 traceable edits lengthen the skill (654–2,447 words against the seed's 580); of 337 added rules, 52% verify and none asks the agent to read less; kept and rejected edits cost alike — the cost arrives with the proposals (Fig. 6b, 7d).
- **The findings generalize (RQ4).** On GPT-6-Luna the pattern repeats (on TB2 all three optimizers return the seed unchanged); frozen skills carry their tax to an unseen benchmark (five of six transfers read 1.23–1.47× the seed's tokens) and to the second model (Table 1 lower panel, Tables 2–3, Fig. 16).

## How BiPA works

A harness without compaction re-sends the prefix and the whole conversation at every turn, so an episode of R turns reads `X = R·F + R(R−1)/2·h` input tokens — each extra turn re-reads the whole history, and input tokens grow faster than turns.

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components (skill text, rest of the prompt, each tool's returned content), with two streams: what the model reads (input tokens) and what the harness executes (tool calls). The change in log input tokens between a skill and its seed splits into four behaviors: **more actions**, **turn-splitting**, **more content per turn**, **longer skill text**. Success is judged separately: avg@k test repetitions against the seed, plus a seed-replay control that applies the optimizer's own acceptance rule to candidates identical to the seed, isolating the dev gain selection alone reports.

The rolling agentic sandbox environments that training and evaluation live in are managed by our internal infrastructure. This repository holds the scheduler, the per-optimizer adapters to the upstream optimizer interfaces (documented `DEVIATIONS`; upstream search, edit and acceptance rules unchanged), the repeated evaluation, and the offline analysis and audit.

## Layout

```
bipa/
├── src/
│   ├── run.py        # one optimization cell (the external entry)
│   ├── core/         # launch, repeat, panels, traces, layout, config
│   ├── adapters/     # per-optimizer interfaces (skillopt, skilladam, gepa)
│   ├── analysis/     # table1.py (Table 1), mech/ and mechanism/ (RQ analyses)
│   ├── dynamics/     # transfer drivers, execution-evidence audits, status/refresh
│   └── graph/        # per-episode bipartite-graph metrics
├── scripts/          # entry points: baselines.sh, rq4_transfer.sh, audit.sh, refresh.sh
├── optimizers/       # vendored upstream sources, pinned commits
├── configs/          # tasks / experiments / graph yaml
├── tests/            # 66 tests, fake model and tasks, no API calls
└── docs/             # investigation notes
```

The seed skill (`verification-before-completion`, 580 words) is read from the shared artifacts tree at `../assets/skills/seed/SKILL.md`. Recorded episodes live outside the repo at `../assets/artifacts/execution/<model>/<method>/<bench>/<run-id>/`; the offline steps only read them.

## Setup

Python 3.12 with `harness-bench`; the upstream optimizers' dependencies (`pip install -e` the pinned checkouts, see [optimizers/README.md](optimizers/README.md)). Interpreter via `BASELINE_PYTHON` (defaults to the harness-mitigation `bach-harness/.venv`; only the Python environment is reused, no BACH code is imported). `baselines.sh` sources an optional `host.env` for host settings. Tests: `"$BASELINE_PYTHON" -m pytest tests -q`.

## Reproduction

Run from the repo root; `python -m` resolves `src.*`. Everything previews by default — `--execute` executes, `--detach` backgrounds. Benchmark aliases: `tau3`, `autob`, `tb2`. Only training, repetitions and transfer call the model; all analysis is offline.

**Main experiments** (the paper's study):

```bash
# 1. train the three optimizers and the two controls, three benchmarks x two models
bash scripts/baselines.sh suite deepseek-flash --execute
# 2. avg@3 repeated test evaluation of each condition's frozen final skill
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
# execution-evidence audits: completion certificate, trace integrity, skill history
bash scripts/audit.sh native-run|trace|skill-history ...
# refresh the derived tables under ../assets/artifacts/results/
bash scripts/refresh.sh
```

RQ mapping (modules under `src/analysis/` unless noted):

**RQ1 — the success gains do not hold:**
- `mech/stats.py` — paired seed-vs-skill test statistics from the raw `evaluation.json` files
- `mech/selection.py` — winner's-curse replay of each optimizer's acceptance rule on seed-identical candidates, and the dev-size replay (`mech/devsize_fig.py`, `mech/devtest_fig.py` draw the figures)
- `mech/domains.py` — per-domain split of Table 1 success (`mech/domains_fig.py`)
- `mechanism/nullmodel.py` — winner's-curse null model and seed variability
- `mechanism/paired.py` — task-level paired summary against Seed (`mech/paired_fig.py` draws the appendix "other aggregations" figure)
- `mechanism/power.py` — power / minimum detectable effect of the comparison
- `mechanism/pooled.py`, `mechanism/reproducibility.py`, `mechanism/appb.py` — pooled and per-repetition reproducibility of success and cost

**RQ2 — every learned skill levies a token tax:**
- `mech/attribution.py` — the four-term split of the log input-token change (`mech/runs_fig.py` draws the per-repetition version)
- `mech/slopes.py`, `mech/slopes_input_fig.py` — within-task slopes of input tokens on turns
- `mech/failcost.py`, `mech/failcost_fig.py` — the tax on solved vs failed tasks
- `mech/price.py`, `mech/price_fig.py` — the tax under cached-input pricing
- `mech/compaction.py` — counterfactual: the tax under a compacting harness
- `mech/optcost.py`, `mech/optcost_fig.py` — the optimizer's own training bill
- `mechanism/episodes.py`, `mechanism/slopes.py`, `mechanism/slopes_checks.py`, `mechanism/pricing.py`, `mechanism/reruns.py`, `mechanism/optcost.py`, `mechanism/facts.py`, `mechanism/appc.py`, `mechanism/fig1.py` — the Section-3 elasticity numbers and the Appendix-C records behind them

**RQ3 — the optimizers accumulate procedure:**
- `mech/skilltext.py` — coded rule categories of the 337 added units (`mech/rules_fig.py` draws the figure)
- `mech/coder2.py` — second-coder reliability of the rule coding
- `mech/callsperturn.py` — tool calls per turn, seed against each learned skill (`mech/calls_fig.py`)
- `mech/edges_build.py` — rebuild the parent→child edit table of the optimizer runs
- `mech/edges.py` — accepted vs rejected edits: the cost arrives with the proposals
- `mech/edges_fig.py`, `mech/edges_holm.py` — keyword-group contrasts of the edits, Holm-adjusted
- `mech/cheatsheet.py` — answer-leakage audit of the learned skills

**RQ4 — the findings generalize:**
- `mech/transfer.py`, `mech/transfer_fig.py` — cross-model and cross-benchmark transfer results
- `mech/dynamics.py`, `mech/dynamics_fig.py` — the trajectory-dynamics hypotheses H1 / H2 / H4
- `src/dynamics/transfer_model.py`, `src/dynamics/transfer_bench.py`, `src/dynamics/transfer_bench_tb2auto.py` — the transfer drivers (the only RQ4 pieces that call the model)

**Paper-wide pieces:** `mech/teaser.py` (Figure 1), `mech/skills_fig.py` (Section 4.3: where each skill's tax goes, next to what its optimizer wrote), `mech/batching.py`, `mech/designev.py`, `mech/implications_fig.py`, `mech/selrules.py` (the Section-5 design-principle evidence), `mech/common.py` (shared paths, figure and audit writers), `mech/copies.py` (byte copies of the baseline-only figures under `mech_` names).

Each module writes its audit record to `../paper/assets/data/mech/`.
