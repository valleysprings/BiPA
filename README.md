# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

This is the code for the paper. It trains three skill optimizers (GEPA, SkillOpt, SkillAdam) and evaluates them against Seed and No-skill controls on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna). It regenerates every table and figure from the recorded runs.

## Motivation

Skill optimizers report only success, and a success score on a small dev set cannot tell whether a learned skill fits its task family better:

- **Evaluation noise.** A single test evaluation can rank two skills either way. A paired comparison over three repetitions detects only differences of several points.
- **Selection inflates gains.** Keeping the best of several noisy dev comparisons is a winner's curse. Replayed on candidates identical to the seed, an optimizer's acceptance rule still reports dev gains.
- **Success hides behavior and cost.** A skill acts through the agent's turns and what they read. Nobody checks whether a gain survives repeated evaluation, or what the skill makes the agent do and spend.

## Research questions

- **RQ1, success.** Do learned skills succeed more often than the seed? Do the dev gains the optimizers report hold on test?
- **RQ2, behavior and cost.** What do learned skills change in the agent's turns, tool calls and input tokens, and what does that cost?
- **RQ3, the edits.** What in the edits produces these changes?

Frozen skills are also run on another model and another benchmark, to see whether the findings hold there.

## How BiPA works

A harness without compaction re-sends the prefix and the whole conversation at every turn. With prefix $F$ and a history growing $h$ tokens per turn, an episode of $R$ turns reads

$$X = R\,F + \frac{R(R-1)}{2}\,h$$

input tokens, so input tokens grow faster than turns: the turn elasticity $e_R = 1 + s_H\,R/(R-1)$, with $s_H$ the history share of $X$, is always above 1 and approaches 2 in long episodes.

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components (the skill text, the rest of the prompt, each tool's returned content), with two streams: what the model reads (input tokens) and what the harness executes (tool calls). Cost is compared with the seed task by task, as the mean paired log-ratio of input tokens and its ratio $\rho$:

$$L(S)=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log\frac{\bar X_i(S)}{\bar X_i(S_0)},\qquad \rho=e^{L(S)}\quad(\rho>1:\ \text{a token tax})$$

With $m=N/R$ calls per turn, the change splits into four behaviors, one term each:

$$\Delta\log X \approx \underbrace{e_R\,\Delta\log N}_{\text{more actions}}\;\underbrace{-\,e_R\,\Delta\log m}_{\text{turn-splitting}}+\underbrace{s_H\,\Delta\log h}_{\text{content per turn}}+\underbrace{s_F\,\Delta\log F}_{\text{skill text}}$$

Success is judged separately: avg@k test repetitions against the seed, and a seed-replay control. In the control, each dev task's success probability is drawn from its Beta posterior $p_v\,|\,\text{seed}\sim\mathrm{Beta}(a+s_v,\;b+n_v-s_v)$ fitted to the seed's executions, and candidates identical to the seed go through the optimizer's own acceptance rule. The gain $G_0$ this keeps is what selection alone reports.

## Layout

```
bipa/
├── src/          # experiments and main-text analyses (code map: src/README.md)
│   ├── run.py        # entry point: run, repeat, transfer, suite
│   ├── panels.py     # task panels and training batches
│   ├── executor.py   # a skill on a list of tasks -> per-task results, through harness-bench
│   ├── llm.py        # the optimizer's model client
│   ├── layout.py     # paths, all taken from environment variables
│   ├── adapters/     # GEPA, SkillOpt, SkillAdam: upstream loops on our batches, executor and model client
│   ├── analysis/     # Table 1, the main-text figures and numbers
│   └── graph/        # the per-episode bipartite graph
├── aux/          # appendix analyses, one folder per topic (aux/README.md)
├── optimizers/   # vendored upstream sources at pinned commits (optimizers/README.md)
├── skills/       # the seed and every returned skill (skills/README.md)
├── configs/      # tasks.yaml: panels, batches, executor; experiments.yaml: pinned runs and caches
├── scripts/      # run.sh: sources host.env, then runs python -m src.run
└── tests/        # offline tests with a fake executor and model
```

The seed is `skills/seed/SKILL.md` (`verification-before-completion`, 580 words). Recorded runs live outside the repository, at `$BIPA_ASSETS/artifacts/execution/<model>/<method>/<benchmark>/<run>/`, with method folders `native-seed`, `native-noseed`, `gepa`, `skillopt` and `skilladam`. A run folder holds `native_protocol.json` (panels, batches, cache root, gate), `summary.json`, `final/SKILL.md`, harness-bench's `results/evaluations/<split>/<version>-<panel>/evaluation.json`, and `round-evidence/` (the skill each optimizer round evaluated). Test repetitions of a returned skill go to `execution/<model>/test-reps/<benchmark>__<method>__<run>/`; the seed's are `rep-1` and `rep-2` next to its run.

## Training protocol

All runs use the OpenCode harness at scale `main`, harness-bench's `native-v1` request capture, full trajectories as optimizer feedback, and the cell's own model as both agent and optimizer, behind an OpenAI-compatible gateway. Each optimizer runs its upstream loop on our sample, executor and model client; the settings are in `configs/tasks.yaml` and `src/panels.py`.

| Benchmark | Train | D_sel (dev) | Test |
|---|---|---|---|
| τ3-bench | 160 of 263, stratified by domain | 30 of 37, stratified by domain | all 75 |
| AutomationBench | 160 of 420, stratified by domain | 30 of 60, stratified by domain | all 120 |
| Terminal-Bench 2.1 (TB2) | all 62 | all 9 | all 18 |

| Optimizer | Batch size | T | Note |
|---|---|---|---|
| GEPA | 8 | 20 | Upstream uses 3; 8 makes its 20 rounds read the 160 slots once. |
| SkillOpt | 40 | 4 (TB2: 6, batches 40, 22, 40, 22, 18, 18) | As upstream. |
| SkillAdam | 10 | 16 | As upstream. |

- **Test set.** Never used for selection or editing; SkillOpt's upstream loop only logs test scores of its baseline, best and final skills.
- **160 training slots.** On τ3-bench and AutomationBench every optimizer reads each of the 160 training tasks once, cut into its own seeded stratified folds. On TB2 every optimizer reads one shared stream: three seeded reshuffles of the 62 tasks cut at 160 slots (62 + 62 + 36), where the first 16 tasks of a pass are not among the last 16 of the previous one. GEPA and SkillAdam take consecutive batches, which may span two passes; SkillOpt cuts each pass as upstream cuts an epoch. A task that recurs under an unchanged skill comes from the episode cache, so TB2 has fewer than 160 distinct training rollouts.
- **Earlier TB2 cut.** The TB2 runs pinned in `configs/experiments.yaml` split each pass into near-equal batches: GEPA 20 batches of 7–8 (156 slots), SkillOpt 5 of 31 (155 slots, one upstream epoch, so its slow update and meta-skill never acted), SkillAdam 18 of 8–9 (160 slots). These batches are stored in `native_protocol.json`, which repetitions, analyses and GEPA or SkillAdam resumes read; the SkillOpt adapter refuses them on resume, because 5 batches do not fill whole epochs.
- **SkillOpt epochs.** Upstream's slow update and meta-skill act from the second pass over the training tasks. On τ3-bench and AutomationBench the 160 tasks are one epoch, so neither acts. On TB2 each pass is an epoch (3; upstream uses 4), so both act at the ends of epochs 2 and 3; the meta-skill of epoch 2 guides epoch 3, and that of epoch 3 is never read.
- **Rollouts outside the 160 slots.** SkillAdam's gate rollouts (`arm_episodes` `gate_train`; runs recorded with the dev-batch gate have `gate_dev`, older ones `dev`) and, on TB2, SkillOpt's slow-update and meta-skill rollouts: from epoch 2, 20 tasks per epoch under the previous and the current skill, a seeded draw from all 62 training tasks. `summary.json` `episodes` counts every evaluated episode by split; read training-data counts from `arm_episodes`, which separates these.
- **Unknown verdicts.** A verdict can be unknown after a provider or infrastructure failure. The executor re-runs unknown verdicts at most twice per evaluation. One still unknown on a training or dev batch aborts the run before scoring, and `--resume` continues from the optimizer's last checkpoint; test scores average over the tasks with a known verdict.
- **Episode cache.** An episode is stored by (skill text, task, split) under the run's cache root and reused by every run with the same root. Each model has its own root in `configs/experiments.yaml`; a new model release should take a new `--cache-root`. The root is recorded in `native_protocol.json`, and resumes and repetitions read it from there.
- **Optimizer prompts.** The client fits each prompt to 512k estimated tokens (characters / 2.5). It never cuts system messages and keeps the first 16k characters of the first user message. Cuts are counted in `summary.json` `llm.truncated_calls`. SkillAdam's own context limit resets its editor history but does not bound the current prompt.

**Gates.** All gates require a strict gain. GEPA and SkillOpt select with D_sel:

- **GEPA** keeps a child that beats its parent on the training batch; D_sel scores each kept child, and those scores choose the next parent and the returned skill.
- **SkillOpt** keeps a candidate that beats the current skill on D_sel. On TB2 the slow update enters the working skill without a dev check (upstream's `slow_update_gate_with_selection: false`).
- **SkillAdam** keeps a candidate that beats the current skill on the training batch it was written from, the published profiles' same-batch validation. It never reads D_sel. `native_protocol.json` and `summary.json` record this as `arm_options` `{"gate": "train-batch"}`. Runs recorded with the dev-batch gate (other `arm_options`, or none) are not resumed by this code.

## Setup

Python 3.12, with this repository next to a `harness-bench` checkout. From the repository root:

```bash
pip install -e ../harness-bench -e optimizers/skillopt -e 'optimizers/skilladam[backend]' \
    numpy scipy scikit-learn matplotlib openai httpx python-dotenv pyyaml pytest
pip install --no-deps -e optimizers/gepa
cp host.env.example host.env    # then edit it
python -m pytest tests -q       # offline, a few seconds
```

Paths come only from environment variables (`src/layout.py`):

- `BIPA_ASSETS` (default `./assets`): `artifacts/` with the recorded runs and `cache/` with the episode and analysis caches.
- `HARNESS_BENCH_ROOT` (default `../harness-bench`): the harness-bench checkout, with its task registry, benchmark data and `.env`.
- `BIPA_OUT` (default `./out`): analysis records in `data/mech/` and figures in `fig/mech/`.
- `BIPA_SEED_FILE` (default `skills/seed/SKILL.md`): the seed; it must be byte-identical to the one the runs used.
- `BIPA_LEGACY_ROOTS` (default none): `:`-separated roots that paths in older records start with; `layout.relocate` maps their `assets/artifacts` (or `artifacts`) to `$BIPA_ASSETS/artifacts`, `assets/artifacts/cache` to `$BIPA_ASSETS/cache` and `assets/optimizers` to `optimizers/`.
- `OPENAI_API_KEY_2`: the optimizer's API key, read from harness-bench's `.env` or the environment.
- `OPENAI_BASE_URL_2`: the optimizer's model endpoint, read the same way.
- `OPENAI_BASE_URL_2_OPTIMIZER`: when set, replaces the URL for optimizer calls only. The optimizer URL must be `https`; plain `http` is an error.
- `PYTHON` (default `python`): the interpreter `scripts/run.sh` uses.

The agent's own model calls are configured in harness-bench. Host settings such as proxies go in `host.env`, which `scripts/run.sh` sources before it runs `python -m src.run`.

## Execution backend

BiPA does not run agents itself. Every episode (one task, one skill text, one repetition) goes through `src/executor.py`, which calls `harness_bench.eval.run` and reads back one record per task. An executor takes a skill text (delivered inside the agent's first prompt), a benchmark (`terminal-bench-2-1`, `tau3-bench`, `automationbench`), a split, task ids, a model and a harness (`opencode`). Per task it returns a verdict (true, false, or unknown for provider and infrastructure failures, which never count as failures), every model request with its exact provider input-token count, its tool calls and whether each ran, the raw request and response of each call as JSON, and a record that the exact skill text reached the model input. Without provider-level request capture the cost columns are empty and every verdict is demoted to unknown. Unknown outcomes are re-run at most twice per evaluation; known outcomes never are.

Our runs used harness-bench, a wrapper around Harbor that runs each episode in a container on a sandbox service. It is not released with this code, so the commands that call the model (`--execute` of `run`, `repeat`, `transfer`, `suite`) do not work from a fresh clone. The task ids of each split are in every run's `native_protocol.json`. The analyses read only recorded episodes; they import `harness_bench` for its skill version ids and need no executor, model or network. A few `aux/` scripts also read the benchmark task folders or call the model; their READMEs say which.

To run elsewhere, pass an `executor` keyword to `harness_bench.eval.run` from `settings()` in `src/executor.py`: an object with `run(version, panel, split, output_dir, attempt)`, which returns the evaluation record, and `cached_episodes(version, rows, split)`. Point `HARNESS_BENCH_ROOT` at your checkout; we have run no other environment. Harbor's `docker` switch is rejected for these benchmarks: AutomationBench needs one container plus a network-less grading container, Terminal-Bench 2.1 one container per task from 89 prebuilt images, and τ3-bench Docker Compose with a tool-server sidecar, a separate grading service and model routes for a user simulator and a judge. Limit the agent's network to the model endpoint and package registries, since benchmark sources and hidden tests are public. Episodes from another executor are a different measurement, so give them their own `--cache-root`.

## Reproduction

Run everything from the repository root, either as `python -m src.run ...` in a shell that has sourced `host.env`, or as `scripts/run.sh ...`. `src.run` previews by default: it prints the commands it would run, and `--execute` runs them. Benchmarks are `tau3`, `autob`, `tb2` or their full names; methods are `seed`, `none` (No skill), `gepa`, `skillopt`, `skilladam`. `run` and `repeat` default to the cell's run in `configs/experiments.yaml`, and `--tag NAME` picks another; `transfer` takes the run the analyses read, pinned under `pins` in the same file, unless `--tag` names the source. A started run continues only with `--resume`; `run` on a finished run does nothing.

```bash
# train every method and control on every benchmark, then the three test repetitions of each final skill
python -m src.run suite deepseek-flash --execute
python -m src.run run skillopt tau3 gpt-6-luna --tag my-run --execute          # one cell
python -m src.run repeat skillopt tau3 deepseek-flash --execute                # avg@3 test repetitions of one run
python -m src.analysis                                                         # Table 1, main-text figures and numbers
# transfer: a returned skill evaluated unchanged under another model or on another benchmark, then its repetitions
python -m src.run transfer skillopt tau3 deepseek-flash --to gpt-6-luna --execute    # -> gpt-6-luna/skillopt/tau3-bench/ood-from-deepseek-flash
python -m src.run transfer gepa tb2 deepseek-flash --to autob --execute              # -> deepseek-flash/gepa/automationbench/ood-from-bench-terminal-bench-2-1
python -m src.run transfer seed tau3 deepseek-flash --to gpt-6-luna --execute        # the seed's reference: ood-seed (ood-bench-seed across benchmarks)
python -m src.run repeat skillopt tau3 gpt-6-luna --tag ood-from-deepseek-flash --execute
```

A test slot that ran in all three repetitions but never got a verdict scores 0 (`adjudicated_failures`, never retried); a slot with no episode blocks the average (`collect` in `src/run.py`).

**Analyses.** They resolve every run through the tags pinned in `configs/experiments.yaml`; a cell with no pin and several finished runs is an error. `python -m src.analysis` runs the main-text analyses in dependency order, `--list` prints them, and `python -m src.analysis selection` runs one module's steps. Figures are `draw` steps of the module whose compute step wrote their cache:

- **Table 1:** `table1` (both models), `stats` (the paired statistics behind it).
- **RQ1:** `selection` (seed-replay control, Fig. 4), `pooled` (pooled success and token ratio over all cells, task coverage).
- **RQ2:** `slopes` (Fig. 3), `attribution` (the four-term split), `cumulative` (Fig. 5), `teaser` (Fig. 1).
- **RQ3:** `edges` (parent-to-child edits), `skilltext` (rule coding of the added text; `luna` for GPT-6-Luna).
- **Transfer:** `transfer` (Luna avg@3 cells, cross-benchmark runs).

Caches go to `$BIPA_ASSETS/cache/analysis/`, records to `$BIPA_OUT/data/mech/`, figures to `$BIPA_OUT/fig/mech/`. The appendix analyses are in `aux/`, one folder per topic; see [aux/README.md](aux/README.md).
