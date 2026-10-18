# BiPA — Bipartite Attribution for Learned Skills

Code for the BiPA paper. BiPA verifies a learned skill by what it changes in the agent: it compares the skill with its seed over repeated runs, replays the optimizer's own selection with every candidate identical to the seed (a control that isolates the dev gain selection alone reports), and records each episode as a bipartite graph between turns and the context they read, splitting extra input tokens into four behaviors: more actions, the same actions spread over more turns, more content per turn, and longer skill text.

The repo trains and evaluates three skill optimizers (SkillOpt, SkillAdam, GEPA) plus Seed and No-skill controls — five reported arms — on three benchmarks (tau3-bench, automationbench, terminal-bench-2-1) with two models (deepseek-flash, gpt-6-luna), then runs the paper's offline analyses. Task execution, caching, retry, and raw traces are owned by the external `harness-bench`; this repo keeps scheduling, upstream-interface adaptation, repeated evaluation, offline analysis, and audit.

## Layout

| Path | Contents |
|---|---|
| `src/run.py` | External entry: one optimization cell (freeze protocol, call the adapter, evaluate the final skill, save the summary) |
| `src/core/` | Internal machinery: `launch.py` (scheduling), `repeat.py` / `repeat_transfer.py` (avg@3 repetition protocol), `panels.py` (fixed panels), `native_protocol.py` / `native_trace.py` (frozen protocol + round witnessing), `layout.py` (paths), `experiment_config.py` (configs) |
| `src/adapters/` | Per-optimizer interface adapters (skillopt, skilladam, gepa); each documents its `DEVIATIONS` |
| `src/analysis/` | Offline analysis package: `table1.py` (avg@3 main table, both models; audit record `mech_table1`), `mech/` (RQ figures/tables), `mechanism/` (RQ2 engine) |
| `src/dynamics/` | Runtime drivers: RQ4 transfer evaluations, run-evidence audits, live status/main table, results refresh |
| `src/graph/` | Per-episode bipartite-graph metrics (read by `analysis/`) |
| `src/prompts/`, `src/utils/` | Adapter prompt files; external evaluation interface (`executor.py`) and LLM transport (`llm.py`) |
| `tests/` | Pytest suite (66 tests, fake model/tasks, no real model calls) |
| `scripts/` | Shell entry points ONLY: `baselines.sh`, `rq4_transfer.sh`, `audit.sh`, `refresh.sh` (see [scripts/README.md](scripts/README.md)) |
| `optimizers/` | Vendored upstream sources with pinned commits (see [optimizers/README.md](optimizers/README.md)) |
| `configs/` | `tasks.yaml` (panels, budgets, backends), `experiments.yaml` (run selection, protocol env), `sensitive-train.yaml`, `graph.yaml` (allocation vocabulary) |
| `docs/` | Investigation notes (`luna-20261014.md`, `ood-20261015.md`) |
| `host.env` | Host settings (not committed; sourced by `baselines.sh`) |

The seed skill is read from the shared workspace artifacts tree (`../assets/skills/seed/SKILL.md` in the harness-mitigation workspace), not vendored in this repo.

The adapters convert tasks, trajectories, and verifier scores into each upstream interface; the search, edit, acceptance, and return rules remain the upstream implementations. Adaptations relative to the original papers (benchmarks, budgets, starting skill, execution interface, caching, transport) are recorded in each adapter's `DEVIATIONS`.

## Experiments

All Python entry points run from the repository root (`bipa/`); `python -m` puts the root on `sys.path`, so `src.*` resolves. Everything previews by default (no model calls): `--execute` runs, `--detach` backgrounds the schedule. Benchmarks accept short aliases (`tau3`, `autob`, `tb2`; `configs/experiments.yaml`).

**Calls the model:** `baselines.sh ... --execute`, `rq4_transfer.sh ... --execute`.
**Offline / read-only on the execution artifacts:** everything under `src/analysis/`, `audit.sh`, `refresh.sh`.

### 1. Baseline training (frozen native-v1 protocol)

One optimization cell = one (arm, model, bench) training run under the frozen `native-v1` protocol (`--feedback-mode full`), with round-by-round witnessing into `native_protocol.json`; an existing protocol file resumes the run instead of restarting it.

```bash
bash scripts/baselines.sh run tau3 gpt-6-luna --arm skillopt      # one optimization cell (preview)
bash scripts/baselines.sh suite gpt-6-luna                        # all five arms x all benches for one model
```

`baselines.sh` forwards to `src/core/launch.py`, which builds `src/run.py` commands. Arms: `seed`, `none`, `skillopt`, `gepa`, `skilladam` (reported as Seed skill, No skill, SkillOpt, GEPA, SkillAdam). Plans never call a model; add `--execute` to run, `--detach` to background.

### 2. avg@3 repeated test evaluation

Each arm's frozen final skill is re-evaluated on the test panel in three repetitions (`--test-runs 3`), recovering only unknown slots — never re-optimizing, and using the source run's frozen panel, cache, and skill rather than today's sampling defaults.

```bash
bash scripts/baselines.sh repeat tau3 gpt-6-luna --arm skillopt   # repeated test evaluation (preview)
```

Adjudication rule (`src/core/repeat.py`, `src/analysis/table1.py`): an episode slot that ran in all three repetitions but never produced a verifier verdict is scored 0 and listed under `adjudicated_failures` (never retried); slots with no episode evidence block the average.

### 3. The main table (Table 1)

```bash
"$BASELINE_PYTHON" -m src.analysis.table1        # offline
```

Writes `../assets/cache/analysis/table1/{table1.json,table1.tex}` and the audit record `../paper/assets/data/mech/mech_table1.md`. Q = avg@3 test success; the rho ratios (input tokens, tool calls) are geometric means over tasks of each method against the Seed row on a common task set, for both models; the dagger marks rho > 1.10. `table1.py` applies the same adjudication rule when it closes an avg@3 cell. `table1.json` is the cache read by several downstream analyses.

### 4. RQ1–RQ4 analysis groups

Offline analyses of finished runs, one or two entry modules per paper element; each module's docstring states its data, command, and outputs. Data steps use the venv interpreter (`python -m src.analysis.mech.<module>`); figure-draw steps use the system `python3` (which has matplotlib), e.g. `python3 -m src.analysis.mechanism.appc draw`. Every table/figure writes one audit record to `../paper/assets/data/mech/<name>.md` (see Evidence below).

- **RQ1 — Do learned skills fit their task family better?** (appendix `c-rq1.tex`) Measurement: dev-selection power, the replay-with-seed control, per-domain success. Modules: `src/analysis/mech/stats.py` (paired statistics, power), `selection.py` + `devsize_fig.py` + `devtest_fig.py` (null selection replay, dev-size replay, dev-vs-test dumbbell), `domains.py` + `domains_fig.py` (per-domain success). The selection-noise / reproducibility checks (reruns, pooled, null model, paired, power) also live in `src/analysis/mechanism/` (thread b: `reproducibility.py`, `pooled.py`, `nullmodel.py`, `paired.py`, `power.py`, `reruns.py`, `appb.py`).
- **RQ2 — What do learned skills change, and what does it cost?** (appendix `d-rq2.tex`) The mechanism: elasticity slopes, the four-term gap decomposition, cached pricing, failure cost, compaction counterfactual, optimizer cost. Modules: `src/analysis/mech/slopes.py`, `attribution.py`, `runs_fig.py`, `paired_fig.py`, `failcost.py`, `price.py`, `compaction.py`, `optcost.py` (each with its `*_fig.py` drawer where a figure exists). The engine they build on is `src/analysis/mechanism/` thread (a): `episodes.py`, `slopes.py`, `slopes_checks.py`, `pricing.py`, `fig1.py`, `appc.py`, `facts.py`, `optcost.py`.
- **RQ3 — What in the optimizers' output produces the tax?** (appendix `e-rq3.tex`) The skill text, the added rules, second-coder reliability, calls per turn, accepted-vs-rejected edit edges. Modules: `src/analysis/mech/skilltext.py`, `rules_fig.py`, `coder2.py`, `callsperturn.py`, `calls_fig.py`, `edges_build.py`, `edges.py`, `edges_fig.py`, `edges_holm.py`.
- **RQ4 — Do the findings generalize?** (appendix `f-rq4.tex`) The second model (GPT-6 Luna — Table 1 covers both models; notes in `docs/`), cross-benchmark and cross-model transfer (`src/analysis/mech/transfer.py`, `transfer_fig.py`; drivers in §5), trajectory dynamics (`dynamics.py`, `dynamics_fig.py`), and the answer-leakage audit (`cheatsheet.py`). Design-principle evidence (Sec. 6): `batching.py`, `selrules.py`, `designev.py`, `implications_fig.py`. Figure 1: `teaser.py`; byte-copy provenance figures: `copies.py`.

The submission freeze lives in the same package: `python -m src.analysis.submission_protocol` freezes which run of every row/benchmark is scored (manifest + hashes, in `../assets/cache/analysis/submission-*/`), and `python -m src.analysis.submission_audit --manifest FILE` read-only re-hashes everything the manifest names.

### 5. Transfer experiments (RQ4 drivers)

Evaluate a returned skill unchanged under another model or on another benchmark's test panel (rep 0), next to that model's own Seed row. Cached and resumable; preview by default, `--execute` calls the model.

```bash
bash scripts/rq4_transfer.sh model   [--methods skillopt,gepa,skilladam] [--execute]   # cross-model
bash scripts/rq4_transfer.sh bench   [--models deepseek-flash] [--execute]             # cross-benchmark
bash scripts/rq4_transfer.sh tb2auto [--execute]                                       # TB2 <-> AutoB scan
```

Drivers: `src/dynamics/transfer_model.py`, `transfer_bench.py`, `transfer_bench_tb2auto.py`; offline scoring of the results in `src/analysis/mech/transfer.py`.

### 6. Run-evidence audits

Offline, read-only on the original runs:

```bash
bash scripts/audit.sh native-run --run RUN_DIR [--output FILE]   # completion certificate (a summary existing != audit passing)
bash scripts/audit.sh trace [--audit-dir DIR]                    # trace integrity of selected test episodes
bash scripts/audit.sh skill-history [--output-dir DIR]           # evidence-linked per-round skill history from optimizer logs
```

(`src/dynamics/audit_native_run.py`, `audit_trace_integrity.py`, `audit_skill_history.py`.)

### 7. Results refresh

```bash
bash scripts/refresh.sh     # offline, no API calls, no loop
```

One serial refresh of `../assets/artifacts/results/`: per-cell current-run skill export (`src/dynamics/render_native_skill_history.py`) → five-arm main table (`main_table.py`) → status block (`weekly_status.py`).

## Requirements

- Python 3.12 with `harness-bench` (task execution, caching, retry, and raw traces) installed; task lists come from harness-bench's registry.
- The dependencies of each upstream optimizer; install per [optimizers/README.md](optimizers/README.md) (`pip install -e` the pinned checkouts).
- Interpreter: `BASELINE_PYTHON`, by default reusing the harness-mitigation `bach-harness/.venv` (only the Python environment is reused; no BACH code is imported; `baselines.sh` itself falls back to `python` on `PATH`). `baselines.sh` additionally sources an optional `host.env` in the repository root for host-specific settings (e.g. proxy).
- Figure-draw steps need the system `python3` with matplotlib (the venv has the scientific stack but no matplotlib).

## Evidence & reproducibility

Raw runs and derived results live OUTSIDE the repo, in the shared workspace:

- Raw runs and evaluation evidence: `../assets/artifacts/execution/<model>/<method>/<bench>/<run>/`
- Derived tables and status: `../assets/artifacts/results/` (written by `scripts/refresh.sh`)
- Analysis intermediates and caches: `../assets/cache/analysis/<module>/` (rebuilt by the modules)
- Audit records: `../paper/assets/data/mech/<name>.md` — one record per paper table/figure, written by `src/analysis/paperout.py`'s `record()`: paper location, producing commands, timestamps of the raw records, result, raw record paths. Several scripts can fill one record. No `.json`/`.tex`/`.txt` data files under `paper/assets/data`.

Training runs freeze their protocol (`native-v1`) and witness every round into `native_protocol.json`; repetitions reuse the source run's frozen panel, cache, and skill; audits (`scripts/audit.sh`) re-check completion certificates, trace integrity, and skill history from the raw evidence. The tests exercise all of this with a fake model and fake tasks — no real model calls:

```bash
"$BASELINE_PYTHON" -m pytest tests -q      # 66 tests, from the repo root
```
