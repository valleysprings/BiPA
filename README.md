# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

This repository is the code of the paper: it trains and evaluates the three skill optimizers the paper studies (GEPA, SkillOpt, SkillAdam) plus Seed and No-skill controls — five reported arms — on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna), and reproduces every table, figure and number from the finished runs.

BiPA (Bipartite Attribution) verifies a learned skill by what it changes in the agent: it compares the skill with its seed over repeated runs, replays the optimizer's own selection with every candidate identical to the seed (a control that isolates the dev gain selection alone reports), and records each episode as a bipartite graph between turns and the context they read, splitting extra input tokens into four behaviors: more actions, the same actions spread over more turns, more content per turn, and longer skill text.

## Why

Skill optimizers report success alone, and a success score on a small dev set cannot settle whether a learned skill fits its task family better (Section 1):

- **Run-to-run noise.** Agent success varies from run to run, so one test run can rank two skills either way; a paired comparison of three runs detects only differences of 7.1, 5.2 and 7.3 points on τ3-bench, AutoBench and TB2 (Table 1 caption, Appendix C.1).
- **Selection inflates gains.** Optimizers keep the best of several noisy dev comparisons, a winner's curse: replaying each optimizer's own acceptance rule on candidates identical to the seed reports 1.2 to 5.1 dev tasks of expected gain against the observed 1 to 5 (Appendix C.2).
- **Success hides behavior and cost.** A skill acts through the agent's turns and what they read; whether a gain survives reruns, and what the skill makes the agent do and spend, go unchecked without an instrument that records both.

## What the paper finds

All numbers below are produced by this repository from the recorded runs (see Reproduction).

- **The success gains do not hold (RQ1).** The optimizers' dev gains of 3 to 40 points end at test differences of −11.1 to +2.7 points over three runs, eight of the nine negative; the same selection reports gains of that size when every candidate is identical to the seed, and even 120 dev tasks would leave it reporting 2 to 4 points (Figure 5, Appendix C.2).
- **Every learned skill levies a token tax (RQ2).** All nine DeepSeek learned skills read 1.08 to 1.43 times the seed's input tokens per task and take 1.03 to 1.22 times its turns; the tax falls on 59 to 87% of tasks per skill and holds on tasks both skills solve in every run (Table 1, Figure 4, Appendix D.1).
- **Extra turns carry the tax.** More actions and turn-splitting together carry 73% of the summed log-token increase, the skill text 21%; within a task, input tokens rise with turns at slopes of 1.35 to 1.58 while output tokens and tool calls rise at slopes of at most 1.03 (Figures 3 and 6a, Appendix D.2). The tax survives cached-input pricing (1.02 to 1.33 times the seed) and a compaction counterfactual (1.05 to 1.36); one optimization run costs the input tokens of 491 to 2,070 seed test episodes, and the tax alone exceeds that bill after 700 to 8,900 episodes of use (Appendix D.3).
- **The optimizers accumulate procedure (RQ3).** 58 of the 59 traceable edits lengthen the skill (654 to 2,447 words against the seed's 580); of the 337 rules added, 52% verify and none asks the agent to read less; kept and rejected edits raise input tokens by similar amounts — the cost arrives with the proposals (Figures 6b and 7d, Appendix E).
- **The findings generalize (RQ4).** On GPT-6-Luna the pattern repeats (on TB2 all three optimizers return the seed unchanged); frozen DeepSeek skills carry their tax to a benchmark they never saw (five of six transfers read 1.23 to 1.47 times the seed's input tokens) and to the second model; of the tasks a skill pushed to more turns, 77 to 100% gained no success (Table 1 lower panel, Table 2, Figure 16, Table 3, Appendix F).

## How BiPA works

A harness without history compaction re-sends the prefix and the whole conversation at every turn, so an episode of R turns reads `X = R·F + R(R−1)/2·h` input tokens: each extra turn re-reads the whole history, and input tokens grow faster than turns (turn elasticity eR > 1; Section 3.4, Figure 2a).

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components — the skill text, the rest of the prompt, and each tool with the content it returns — with two streams: what the model reads (input tokens) and what the harness executes (tool calls). The change in log input tokens between a skill and its seed then splits into four terms, one per behavior (Eq. 7, Section 3.5): **more actions** (more tool calls), **turn-splitting** (the same calls spread over more turns), **content per turn** (more text entering the conversation each turn), and **skill text** (the longer prefix). Success is judged separately: avg@k test runs against the seed, and a seed-replay control that applies the optimizer's own acceptance rule to candidates identical to the seed to measure the dev gain selection alone reports (Section 3.2).

Task execution, caching, retry and raw traces are owned by the external `harness-bench`; this repo keeps scheduling, upstream-interface adaptation (per-optimizer adapters with documented `DEVIATIONS`; upstream search, edit and acceptance rules unchanged), repeated evaluation, offline analysis and audit.

## Layout

| Path | Contents |
|---|---|
| `src/run.py` | External entry: one optimization cell (freeze protocol, call the adapter, evaluate the final skill, save the summary) |
| `src/core/` | Internal machinery: `launch.py` (scheduling), `repeat.py` / `repeat_transfer.py` (avg@3 repetition protocol), `panels.py` (fixed panels), `native_protocol.py` / `native_trace.py` (frozen protocol + round witnessing), `layout.py` (paths), `experiment_config.py` (configs) |
| `src/adapters/` | Per-optimizer interface adapters (skillopt, skilladam, gepa); each documents its `DEVIATIONS` |
| `src/analysis/` | Offline analysis: `table1.py` (Table 1, both models), `mech/` (RQ figures/tables), `mechanism/` (RQ2 engine), `paperout.py` (audit records) |
| `src/dynamics/` | Runtime drivers: RQ4 transfer evaluations, run-evidence audits, live status/main table, results refresh |
| `src/graph/` | Per-episode bipartite-graph metrics (read by `analysis/`) |
| `src/prompts/`, `src/utils/` | Adapter prompt files; external evaluation interface (`executor.py`) and LLM transport (`llm.py`) |
| `tests/` | Pytest suite (66 tests, fake model/tasks, no real model calls) |
| `scripts/` | Shell entry points ONLY: `baselines.sh`, `rq4_transfer.sh`, `audit.sh`, `refresh.sh` (see [scripts/README.md](scripts/README.md)) |
| `optimizers/` | Vendored upstream sources with pinned commits (see [optimizers/README.md](optimizers/README.md)) |
| `configs/` | `tasks.yaml` (panels, budgets, backends), `experiments.yaml` (run selection, protocol env), `sensitive-train.yaml`, `graph.yaml` |
| `docs/` | Investigation notes (`luna-20261014.md`, `ood-20261015.md`) |

The seed skill (`verification-before-completion`, 580 words; Appendix G.2) is read from the shared workspace artifacts tree at `../assets/skills/seed/SKILL.md`, not vendored.

## Reproduction

All Python entry points run from the repository root (`bipa/`); `python -m` puts the root on `sys.path`. Everything previews by default (no model calls): `--execute` runs, `--detach` backgrounds the schedule. Benchmarks accept short aliases (`tau3`, `autob`, `tb2`).

**Calls the model:** baseline training and repetitions (`baselines.sh ... --execute`) and transfer evaluations (`rq4_transfer.sh ... --execute`).
**Offline / read-only on the execution artifacts:** everything under `src/analysis/`, the audits (`audit.sh`), and the results refresh (`refresh.sh`).

Raw runs and evaluation evidence live OUTSIDE the repo at `../assets/artifacts/execution/<model>/<method>/<bench>/<run>/` and are **not** regenerated by these flows — the offline steps read the recorded evidence.

### Main experiments (the paper's study)

1. **Baseline training**, five arms × three benchmarks × two models under the frozen native-v1 protocol, one cell at a time or as a suite (plans never call a model):

   ```bash
   bash scripts/baselines.sh run tau3 deepseek-flash --arm skillopt      # one optimization cell (preview)
   bash scripts/baselines.sh suite deepseek-flash                        # all five arms x all benches (preview)
   ```

   Each run freezes its protocol and witnesses every round into `native_protocol.json`; an existing protocol file resumes the run. Arms: `seed`, `none`, `skillopt`, `gepa`, `skilladam`.

2. **avg@3 repeated test evaluation** of each arm's frozen final skill (`--test-runs 3`), reusing the source run's frozen panel, cache and skill — never re-optimizing:

   ```bash
   bash scripts/baselines.sh repeat tau3 deepseek-flash --arm skillopt   # repeated test evaluation (preview)
   ```

   Adjudication rule (`src/core/repeat.py`, `src/analysis/table1.py`): an episode slot that ran in all three repetitions but never produced a verifier verdict is scored 0 and listed under `adjudicated_failures` (never retried); slots with no episode evidence block the average.

3. **The main table** (Table 1, both models; offline):

   ```bash
   "$BASELINE_PYTHON" -m src.analysis.table1
   ```

   Writes `../assets/cache/analysis/table1/{table1.json,table1.tex}` and the audit record `../paper/assets/data/mech/mech_table1.md`. Q = avg@3 test success; the ρ ratios (input tokens, tool calls) are per-task geometric means against the Seed row on a common task set, with the same adjudication rule.

### Secondary experiments (offline analyses and audits)

- **RQ1 — fit** (Appendix C; record names `mech_stats`, `mech_selection`, `mech_devsize`, `mech_domains`): `src/analysis/mech/stats.py`, `selection.py`, `devsize_fig.py`, `devtest_fig.py`, `domains.py`, `domains_fig.py`; the selection-noise checks also in `src/analysis/mechanism/` (`reproducibility.py`, `pooled.py`, `nullmodel.py`, `paired.py`, `power.py`, `reruns.py`, `appb.py`).
- **RQ2 — behavior and cost** (Appendix D; `mech_slopes`, `mech_attribution3`, `mech_price3`, `mech_compaction`, `mech_failcost`, `mech_optcost`, `mech_paired`): `src/analysis/mech/slopes.py`, `attribution.py`, `runs_fig.py`, `paired_fig.py`, `failcost.py`, `price.py`, `compaction.py`, `optcost.py`; engine in `src/analysis/mechanism/` (`episodes.py`, `slopes.py`, `slopes_checks.py`, `pricing.py`, `fig1.py`, `appc.py`, `facts.py`, `optcost.py`).
- **RQ3 — source of the tax** (Appendix E; `mech_skilltext`, `mech_rules`, `mech_coder2`, `mech_callsperturn`, `mech_edges`): `src/analysis/mech/skilltext.py`, `rules_fig.py`, `coder2.py`, `callsperturn.py`, `calls_fig.py`, `edges_build.py`, `edges.py`, `edges_fig.py`, `edges_holm.py`.
- **RQ4 — generality** (Appendix F; `mech_transfer`, `mech_dynamics`, `mech_cheatsheet`): `src/analysis/mech/transfer.py`, `transfer_fig.py`, `dynamics.py`, `dynamics_fig.py`, `cheatsheet.py`. Figure 1: `teaser.py`; Section 5 design-principle evidence (Figure 7): `batching.py`, `selrules.py`, `designev.py`, `implications_fig.py`.

  Each analysis module is an entry point — data steps `"$BASELINE_PYTHON" -m src.analysis.mech.<module>`, figure draws with the system `python3` (matplotlib), e.g. `python3 -m src.analysis.mechanism.appc draw`; the module docstring states its exact subcommands.

- **Transfer experiments** (RQ4 drivers; cached and resumable, preview by default, `--execute` calls the model): a returned skill evaluated unchanged under another model or on another benchmark's test panel, against that row's Seed:

  ```bash
  bash scripts/rq4_transfer.sh model   [--methods skillopt,gepa,skilladam] [--execute]   # cross-model
  bash scripts/rq4_transfer.sh bench   [--models deepseek-flash] [--execute]             # cross-benchmark
  bash scripts/rq4_transfer.sh tb2auto [--execute]                                       # TB2 <-> AutoB scan
  ```

- **Run-evidence audits** (offline, read-only on the original runs):

  ```bash
  bash scripts/audit.sh native-run --run RUN_DIR [--output FILE]   # completion certificate
  bash scripts/audit.sh trace [--audit-dir DIR]                    # trace integrity of selected test episodes
  bash scripts/audit.sh skill-history [--output-dir DIR]           # evidence-linked per-round skill history
  ```

- **Results refresh** (offline, no API calls): `bash scripts/refresh.sh` — per-cell skill export, five-arm main table, status block, into `../assets/artifacts/results/`.

- **Submission freeze**: `python -m src.analysis.submission_protocol` freezes which run of every row/benchmark is scored (manifest + hashes); `python -m src.analysis.submission_audit --manifest FILE` read-only re-hashes everything the manifest names.

## Requirements

- Python 3.12 with `harness-bench` (task execution, caching, retry, raw traces) installed; task lists come from harness-bench's registry.
- The dependencies of each upstream optimizer; install per [optimizers/README.md](optimizers/README.md) (`pip install -e` the pinned checkouts).
- Interpreter: `BASELINE_PYTHON`, by default reusing the harness-mitigation `bach-harness/.venv` (only the Python environment is reused; no BACH code is imported; `baselines.sh` itself falls back to `python` on `PATH`). `baselines.sh` additionally sources an optional `host.env` in the repository root for host-specific settings (e.g. proxy).
- Figure-draw steps need the system `python3` with matplotlib (the venv has the scientific stack but no matplotlib).

## Evidence & reproducibility

- Raw runs and evaluation evidence: `../assets/artifacts/execution/<model>/<method>/<bench>/<run>/` (owned by the runs, outside the repo)
- Derived tables and status: `../assets/artifacts/results/` (written by `scripts/refresh.sh`)
- Analysis intermediates and caches: `../assets/cache/analysis/<module>/` (rebuilt by the modules)
- Audit records: `../paper/assets/data/mech/<name>.md` — one record per paper table/figure, written by `src/analysis/paperout.py`'s `record()`: paper location, producing commands, timestamps of the raw records, result, raw record paths. No `.json`/`.tex`/`.txt` data files under `paper/assets/data`.

Training runs freeze their protocol (`native-v1`) and witness every round into `native_protocol.json`; repetitions reuse the source run's frozen panel, cache and skill; the audits re-check completion certificates, trace integrity and skill history from the raw evidence. The tests exercise all of this with a fake model and fake tasks — no real model calls:

```bash
"$BASELINE_PYTHON" -m pytest tests -q      # 66 tests, from the repo root
```
