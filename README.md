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
│   ├── adapters/     # GEPA, SkillOpt, SkillAdam: upstream loops on these batches, executor and model client
│   ├── analysis/     # Table 1, its statistics, the turn slopes and the four-term attribution
│   └── graph/        # the per-episode bipartite graph
├── aux/          # every other figure, number and appendix analysis, one folder per topic (aux/README.md)
├── optimizers/   # vendored upstream sources at pinned commits (optimizers/README.md)
├── skills/       # the seed and every returned skill (skills/README.md)
├── configs/      # tasks.yaml: panels, batches, executor; experiments.yaml: pinned runs and caches
├── scripts/      # main.sh: the main experiment; aux.sh: every other analysis; run.sh: sources host.env, then runs python
└── tests/        # offline tests with a fake executor and model
```

The seed is `skills/seed/SKILL.md` (`verification-before-completion`, 580 words). Recorded runs live outside the repository, at `$BIPA_ASSETS/artifacts/execution/<model>/<method>/<benchmark>/<run>/`, with method folders `native-seed`, `native-noseed`, `gepa`, `skillopt` and `skilladam`. A run folder holds the run's settings file (panels, batches, cache root, gate), `summary.json`, `final/SKILL.md`, harness-bench's `results/evaluations/<split>/<version>-<panel>/evaluation.json`, and `round-evidence/` (the skill each optimizer round evaluated). Test repetitions of a returned skill go to `execution/<model>/test-reps/<benchmark>__<method>__<run>/`; the seed's are `rep-1` and `rep-2` next to its run.

## How the optimizers are run

Every run uses the OpenCode harness at scale `main`, harness-bench's request capture (native-v1) and full trajectories as optimizer feedback. The cell's model is both agent and optimizer, behind an OpenAI-compatible gateway. Each optimizer runs its upstream loop on the panels and batches of `src/panels.py`, with the settings in `configs/tasks.yaml`.

| Benchmark | Train | D_sel (dev) | Test |
|---|---|---|---|
| τ3-bench | 160 of 263, stratified by domain | 30 of 37, stratified by domain | all 75 |
| AutomationBench | 160 of 420, stratified by domain | 30 of 60, stratified by domain | all 120 |
| Terminal-Bench 2.1 (TB2) | all 62 | all 9 | all 18 |

No optimizer selects or edits on test.

| Optimizer | Batch size | T | Note |
|---|---|---|---|
| GEPA | 8 | 20 | Upstream uses 3; at 8 its 20 rounds read the 160 slots once. |
| SkillOpt | 40 | 4 (TB2: 6, batches 40, 22, 40, 22, 18, 18) | Upstream's batch size. |
| SkillAdam | 10 | 16 | |

**Training slots.** Every optimizer reads 160 training slots. On τ3-bench and AutomationBench these are the 160 training tasks, each once, in seeded stratified batches. On TB2 all optimizers read one shared stream: three seeded reshuffles of the 62 tasks, cut at 160 slots (62 + 62 + 36). GEPA and SkillAdam take consecutive batches from it, so a batch may span two passes. SkillOpt cuts each pass as upstream cuts an epoch. A task that recurs under an unchanged skill is served from the episode cache. TB2 runs recorded before the shared stream used near-equal batches (GEPA 156 slots, SkillOpt 155 in one epoch, SkillAdam 160), and their batch lists are stored with each run.

**Gates.** Every gate needs a strict gain. GEPA keeps a child that beats its parent on the training batch, and scores each kept child on D_sel to choose the next parent and the returned skill. SkillOpt keeps a candidate that beats the current skill on D_sel. SkillAdam keeps a candidate that beats the current skill on the training batch it was written from, and never reads D_sel. The code in this repository runs only this training-batch gate. The pinned SkillAdam runs (DeepSeek `native-20261008-https`, GPT-6-Luna `native-20261012`) used a dev-batch gate: each step drew min(10, |D_sel|) dev tasks and rolled out the candidate and the current skill on them, so these runs cannot be resumed with this code.

**SkillOpt epochs.** Upstream's slow update and meta-skill act from the second epoch. τ3-bench and AutomationBench get one epoch, so neither acts. TB2 gets three epochs (upstream uses 4), and both act at the ends of epochs 2 and 3. The slow update enters the skill without a D_sel check, as in upstream's default.

**Rollouts outside the 160 slots.** SkillAdam's gate re-runs each candidate on its training batch (in the pinned runs, the candidate and the current skill on the dev batch, counted as `dev` in `arm_episodes`), and on TB2 SkillOpt's slow update and meta-skill run 20 training tasks per epoch from epoch 2, under the previous and the current skill. `summary.json` `episodes` counts these too, so read training-data counts from `arm_episodes`.

**Unknown verdicts.** A provider or infrastructure failure leaves a verdict unknown, and the executor re-runs it at most twice. A verdict still unknown on a training or dev batch aborts the run, `--resume` continues it, and test scores average the known verdicts.

**Episode cache.** An episode is stored by skill text, task and split under a cache root, and every run with the same root reuses it. Each model has its own root in `configs/experiments.yaml`, and a new model release needs a new `--cache-root`.

**Optimizer prompts.** The client fits each prompt to 512k estimated tokens (characters / 2.5), keeps system messages and the first 16k characters of the first user message, and counts cuts in `summary.json` `llm.truncated_calls`.

## Setup

Python 3.12, with this repository next to a `harness-bench` checkout. From the repository root:

```bash
pip install -e ../harness-bench -e optimizers/skillopt -e 'optimizers/skilladam[backend]' \
    numpy scipy scikit-learn matplotlib openai httpx python-dotenv pyyaml pytest
pip install --no-deps -e optimizers/gepa
cp host.env.example host.env    # then edit it
python -m pytest tests -q       # offline
```

Paths come from environment variables (`src/layout.py`), the optimizer's endpoint from `src/llm.py`:

| Variable | Default | What |
|---|---|---|
| `BIPA_ASSETS` | `./assets` | `artifacts/` (recorded runs) and `cache/` (episode and analysis caches) |
| `HARNESS_BENCH_ROOT` | `../harness-bench` | the harness-bench checkout: task registry, benchmark data, `.env` |
| `BIPA_OUT` | `./out` | analysis records in `data/mech/`, figures in `fig/mech/` |
| `BIPA_SEED_FILE` | `skills/seed/SKILL.md` | the seed, byte-identical to the one the runs used |
| `BIPA_LEGACY_ROOTS` | none | `:`-separated roots of the paths in older records; `layout.relocate` maps them into `$BIPA_ASSETS` and `optimizers/` |
| `OPENAI_API_KEY_2` | harness-bench's `.env` | the optimizer's API key |
| `OPENAI_BASE_URL_2` | harness-bench's `.env` | the optimizer's endpoint |
| `OPENAI_BASE_URL_2_OPTIMIZER` | unset | replaces the endpoint for optimizer calls only |
| `PYTHON` | `python` | the interpreter of `scripts/run.sh` |

The optimizer URL must be `https`; the client refuses plain `http`. The agent's own model calls are configured in harness-bench. Host settings such as proxies go in `host.env`, which `scripts/run.sh` sources before `python -m src.run`.

## Execution backend

BiPA does not run agents itself. Every episode (one task, one skill text, one repetition) goes through `src/executor.py`, which calls `harness_bench.eval.run` and reads back one record per task. An executor takes a skill text (delivered inside the agent's first prompt), a benchmark (`terminal-bench-2-1`, `tau3-bench`, `automationbench`), a split, task ids, a model and a harness (`opencode`). Per task it returns a verdict (true, false, or unknown for provider and infrastructure failures, which never count as failures), every model request with its exact provider input-token count, its tool calls and whether each ran, the raw request and response of each call as JSON, and a record that the exact skill text reached the model input. Without provider-level request capture the cost columns are empty and every verdict is demoted to unknown. Unknown outcomes are re-run at most twice per evaluation; known outcomes never are.

Our runs used harness-bench, a wrapper around Harbor that runs each episode in a container on a sandbox service. It is not released with this code, so the commands that call the model (`--execute` of `run`, `repeat`, `transfer`, `suite`) do not work from a fresh clone. The task ids of each split are in every run's `native_protocol.json`. The analyses read only recorded episodes; they import `harness_bench` for its skill version ids and need no executor, model or network. A few `aux/` scripts also read the benchmark task folders or call the model; their READMEs say which.

To run elsewhere, pass an `executor` keyword to `harness_bench.eval.run` from `settings()` in `src/executor.py`: an object with `run(version, panel, split, output_dir, attempt)`, which returns the evaluation record, and `cached_episodes(version, rows, split)`. Point `HARNESS_BENCH_ROOT` at your checkout; we have run no other environment. Harbor's `docker` switch is rejected for these benchmarks: AutomationBench needs one container plus a network-less grading container, Terminal-Bench 2.1 one container per task from 89 prebuilt images, and τ3-bench Docker Compose with a tool-server sidecar, a separate grading service and model routes for a user simulator and a judge. Limit the agent's network to the model endpoint and package registries, since benchmark sources and hidden tests are public. Episodes from another executor are a different measurement, so give them their own `--cache-root`.

## Reproduction

Run everything from the repository root. `scripts/main.sh` and `scripts/aux.sh` go through `scripts/run.sh`, which sources `host.env` and runs `python -m src.run`, or any module with `-m MODULE`. Every target of both scripts prints its commands in order and runs nothing; `--execute` runs them, the model calls included, and `--execute --offline` runs only the steps that read the recorded runs. Caches go to `$BIPA_ASSETS/cache/analysis/`, records to `$BIPA_OUT/data/mech/`, figures to `$BIPA_OUT/fig/mech/`.

**Main experiment: `scripts/main.sh`.** `table1` reads the recorded runs only, so `--offline` changes nothing there.

```bash
scripts/main.sh train deepseek-flash                 # print the runs: every method and control on every benchmark, then the three test repetitions of each final skill
scripts/main.sh train deepseek-flash --execute
scripts/main.sh train gpt-6-luna --benches tau3 --methods skillopt seed --execute
scripts/main.sh run skillopt tau3 gpt-6-luna --tag my-run --execute            # one cell
scripts/main.sh repeat skillopt tau3 deepseek-flash --execute                  # avg@3 test repetitions of one run
scripts/main.sh transfer skillopt tau3 deepseek-flash --to gpt-6-luna --execute  # a returned skill, unchanged, under another model or on another benchmark
scripts/main.sh table1 --execute                     # Table 1, its paired statistics, the turn slopes, the four-term attribution
```

`train` is `python -m src.run suite`; `run`, `repeat` and `transfer` pass through to `src.run`; `table1` runs the steps of `python -m src.analysis`, the attribution and slopes caches first. Benchmarks are `tau3`, `autob`, `tb2` or their full names; methods are `seed`, `none` (No skill), `gepa`, `skillopt`, `skilladam`. `run` and `repeat` default to the cell's run in `configs/experiments.yaml`, and `--tag NAME` picks another; `transfer` takes the run pinned under `pins` in the same file unless `--tag` names the source, and writes `ood-from-<model>` or `ood-from-bench-<benchmark>` (the seed's reference: `ood-seed`, `ood-bench-seed`). A started run continues only with `--resume`; `run` on a finished run does nothing. A test slot that ran in all three repetitions but never got a verdict scores 0 (`adjudicated_failures`, never retried); a slot with no episode blocks the average (`collect` in `src/run.py`). The analyses resolve every run through the pinned tags; a cell with no pin and several finished runs is an error.

**Other analyses: `scripts/aux.sh`.** One target per folder of `aux/` ([aux/README.md](aux/README.md) lists them and what each reproduces), and `all`. `all` runs the steps of `scripts/main.sh table1`, then every folder, data steps before the figures that read them. With `--execute` a failing step ends its folder, the other folders still run, and a table of each folder's completed steps and failed step closes the run, which exits non-zero if any step failed.

```bash
scripts/aux.sh all                                   # print every step
scripts/aux.sh all --execute --offline               # run every step that needs no model call
scripts/aux.sh selection_replay --execute --offline  # one folder
```

**What needs the executor.** The steps that call the model go through the executor (Execution backend) and run only with `--execute` and without `--offline`: `train`, `run`, `repeat` and `transfer` of `main.sh`, the transfer evaluations and their repetitions in `aux.sh headline_numbers`, and the path-test passes in `aux.sh path_test` (`aux/path_test/launch.sh`). The delivery and trigger runs of `contamination_and_loading` (`run_native`, `run_trigger`) also need the SkillZip runtime, which is not in this repository: `aux.sh contamination_and_loading` lists them and never runs them, and `all` leaves them out with a note. Every other step reads the recorded runs, including the path-test passes under `$BIPA_ASSETS/artifacts/path-test`; it needs the `harness_bench` package but no executor or model, and the contamination check also reads the benchmark task folders under `HARNESS_BENCH_ROOT`.
