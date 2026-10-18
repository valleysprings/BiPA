# BiPA — Bipartite Attribution for Learned Skills

Code for the BiPA paper. BiPA verifies a learned skill by what it changes in the agent: it compares the skill with its seed over repeated runs, replays the optimizer's own selection with every candidate identical to the seed (a control that isolates the dev gain selection alone reports), and records each episode as a bipartite graph between turns and the context they read, splitting extra input tokens into four behaviors: more actions, the same actions spread over more turns, more content per turn, and longer skill text.

## Layout

| Path | Contents |
|---|---|
| `src/` | Experiment harness: scheduling (`run.sh` → `launch.py`), single-cell runs (`run.py`), repeated test evaluation (`repeat.py`, `repeat_transfer.py`), episode tracing (`native_protocol.py`, `native_trace.py`), fixed panels (`panels.py`), and adapters (`adapters/`) to the upstream optimizers |
| `optimizers/` | Canonical upstream sources: SkillOpt, SkillAdam, GEPA, EvoHarness |
| `configs/` | Run configuration: panels, budgets, and backends (`tasks.yaml`), experiment definitions (`experiments.yaml`) |
| `docs/` | Investigation notes (e.g. `luna-20261014.md`) |
| `src/analysis/` | Offline analysis of finished runs (read-only on the execution artifacts): the avg@3 main table (`src/analysis/table1.py`) and the appendix analyses (`src/analysis/mech/`, `src/analysis/mechanism/`) |
| `src/graph/` | Per-episode bipartite graph construction and metrics used by the analysis |
| `scripts/` | Scheduling and RQ-test drivers (transfer evaluation), the live status/progress table, and run-evidence audits |

The seed skill is read from the shared artifacts tree (`../assets/skills/seed/SKILL.md` in the harness-mitigation workspace), not vendored in this repo.

The adapters convert tasks, trajectories, and verifier scores into each upstream interface; the search, edit, acceptance, and return rules remain the upstream implementations. Adaptations relative to the original papers (benchmarks, budgets, starting skill, execution interface, caching, transport) are recorded in each adapter's `DEVIATIONS`.

## Requirements

- Python 3.12 with `harness-bench` (task execution, caching, retry, and raw traces) installed
- The dependencies of each upstream optimizer, see `optimizers/<name>/`

## Usage

From the repository root, everything previews by default (no model calls):

```bash
bash src/run.sh run tau3 <model> --arm skillopt      # one optimization cell
bash src/run.sh repeat tau3 <model> --arm skillopt   # repeated test evaluation
bash src/run.sh suite <model>                        # full suite for one model
```

Add `--execute` to actually run, or `--detach` to run the schedule in the background. `run.sh` sources an optional `host.env` in the repository root for host-specific settings (e.g. proxy); `BASELINE_PYTHON` selects the interpreter.

Raw execution artifacts and caches are not part of this repository.

The repeated test evaluation (`src/repeat.py`, three repetitions per arm) scores a task whose episodes ran but never produced a verifier outcome in any repetition as a failure (listed under `adjudicated_failures`); slots without episode evidence still block the average. `src/analysis/table1.py` applies the same rule when it closes an avg@3 cell.
