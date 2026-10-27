"""The settings every analysis shares, kept in one place: which run each cell reads, which skills count as learned, how a
cell's test episodes are read, how resampling and random draws are made, and where intermediate files go.

Runs come only from the experiments file (configs/experiments.yaml, or the file BIPA_EXPERIMENTS names): the pins for the
cells of Table 1, the aux_runs block for every other run. Test episodes are read only by Table 1's reader: the same files
per repetition and the same rule for a verified outcome.

Random draws. One stream per draw set: the resamples (or task-model draws) that a group of statistics is computed from
are drawn once, from rng(key), and every statistic of the group is computed on those same draws, so comparisons within the
group (rules a-d on one p, every skill of a benchmark on one task resample) are paired. Different draw sets use different
keys, so changing one cell or the order of the work moves no other number. The one exception is aux.success.stats (the
intervals beside Table 1), which keeps its single sequential generator with the same SEED and N_BOOT."""
import functools, json, os, zlib

import numpy as np

from src.utils import layout
from src.analysis.common import merged_episodes, merged_rows_of
from src.utils.dataloader import BENCHES, BN, EX, MODELS, SEED_TEXT, final_skill, out_dir, vid

DEEPSEEK, LUNA = "deepseek-flash", "gpt-6-luna"
ARMS = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")
OPTIMIZERS = ARMS[2:]
ARM_NAME = {"native-noseed": "No skill", "native-seed": "Seed", "gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
MODEL_NAME = {DEEPSEEK: "DeepSeek-V4.1-Flash", LUNA: "GPT-6-Luna"}
MODEL_SHORT = {DEEPSEEK: "DeepSeek", LUNA: "Luna"}
BENCH_SHORT = BN  # tau3, AutoB, TB2

# Resampling. Every interval resamples tasks N_BOOT times; every task-model replay draws N_DRAWS times.
SEED = 20261012
N_BOOT = 10_000
N_DRAWS = 100_000
LEVEL = 0.95
Z95 = 1.96  # the paper's two-sided 95% normal quantile (meta-analysis intervals, |z| >= 1.96 counts)
GAINS = (1, 2, 3)  # true gains, in dev tasks, at which an acceptance rule's power is reported

if os.environ.get("BIPA_MANIFEST"):
    raise RuntimeError("the analyses read the runs the experiments file pins; unset BIPA_MANIFEST")


# ---- runs and skills

def run(model, bench, method):
    """The run a cell reports, as the experiments file pins it (a cell without a pin is an error)."""
    try:
        tag = layout.PINS[model][method][bench]
    except KeyError:
        raise LookupError(f"no pin for {model} {method} {bench} in {layout.EXPERIMENTS}") from None
    path = EX / model / method / bench / tag
    if not (path / "summary.json").is_file():
        raise FileNotFoundError(f"pinned run {path} has no summary.json")
    return path


def extra(name, model, bench):
    """[run directory] of a named set of runs that no cell pins (the experiments file's aux_runs block), e.g.
    extra("seed_dev_pool", DEEPSEEK, "tau3-bench"). A listed run that is missing is an error."""
    if not layout.AUX_RUNS or name not in layout.AUX_RUNS:
        raise LookupError(f"no aux_runs entry {name!r} in {layout.EXPERIMENTS}")
    spec = layout.AUX_RUNS[name].get(model, [])
    entries = spec.get(bench, []) if isinstance(spec, dict) else spec
    paths = [EX / model / e.split("/")[0] / bench / e.split("/")[1] for e in entries]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"aux_runs {name}: missing {', '.join(missing)}")
    return paths


def skill(model, bench, method):
    """The cell's final skill text (None for No skill)."""
    return final_skill(method, run(model, bench, method))


def version(text):
    """The version id of a skill text; a skill's evaluation folders start with its first 12 characters."""
    return vid(text)


SEED_VERSION = version(SEED_TEXT)


def returned_seed(model, bench, method):
    """True for an optimizer whose final skill is the seed text unchanged."""
    return method in OPTIMIZERS and skill(model, bench, method) == SEED_TEXT


def learned(models=MODELS, benches=BENCHES):
    """[(model, bench, optimizer)] whose final skill differs from the seed: the skills every pooled number is over."""
    return [(m, b, o) for m in models for b in benches for o in OPTIMIZERS if not returned_seed(m, b, o)]


def learned_pairs(models=MODELS, benches=BENCHES):
    """[(model, bench)] with at least one learned skill."""
    return list(dict.fromkeys((m, b) for m, b, _ in learned(models, benches)))


def dev_gated(model, bench, method):
    """True when the run chose among its candidates on the dev panel: GEPA and SkillOpt always, SkillAdam unless it gated
    its edits on the training batch (arm option gate = train-batch). Only these runs enter a replay of dev selection."""
    if method not in OPTIMIZERS:
        return False
    summary = json.loads((run(model, bench, method) / "summary.json").read_text())
    return (summary.get("arm_options") or {}).get("gate", "dev") != "train-batch"


def domain_of(bench, task):
    """The domain of a task: the middle token of tau3-<domain>-<n>, the leading token of an AutomationBench id, and "all"
    on TB2."""
    if bench == "terminal-bench-2-1":
        return "all"
    return task.split("-")[1] if bench == "tau3-bench" else task.split("-")[0]


# ---- test episodes, read as Table 1 reads them

@functools.lru_cache(maxsize=None)
def _seed_files(model, bench):
    from src.analysis import table1
    return table1.seed_files(model, bench)[1]


def cell_files(model, bench, method):
    """{repetition: [evaluation.json]} of a row of Table 1. A skill equal to the seed reports the Seed row's runs."""
    from src.analysis import table1
    got = table1.cell_files(model, bench, method, _seed_files(model, bench))
    return got[2] if got else {}


def run_files(run_dir, text):
    """{repetition: [evaluation.json]} of a run that no cell pins (a transfer or from-scratch run), by the same rule."""
    from src.analysis import table1
    return table1.run_files(run_dir, text)


def slot(row):
    """(task, repetition) of a panel entry or an episode."""
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def known(episode):
    """True when an episode has a verified outcome (Table 1's rule)."""
    return type(episode.get("success")) is bool and episode.get("outcome_valid", True) is True


def episodes(files):
    """{repetition: {task: recorded episode}}: a repetition's files merged, where a verified outcome is never overridden."""
    return {k: {t: e for (t, _), e in merged_episodes(fs).items()} for k, fs in sorted(files.items())}


def rows(files, model):
    """{repetition: {task: row}} by Table 1's reader for the model. A row holds success (None without a verified outcome),
    executed calls N, input tokens X (None when not fully captured), c and the distinct-request counts nreq and R (turns)."""
    from src.analysis import table1
    reader = table1.row_reader(model)
    return {k: {t: r for (t, _), r in merged_rows_of(fs, reader).items()} for k, fs in sorted(files.items())}


def cell_episodes(model, bench, method):
    return episodes(cell_files(model, bench, method))


def cell_rows(model, bench, method):
    return rows(cell_files(model, bench, method), model)


def wins(reps):
    """{task: runs that solved it} over the tasks verified in all three repetitions of {rep: {task: row}} (pass@3 coverage
    is wins >= 1); {} unless there are three repetitions."""
    if len(reps) != 3:
        return {}
    tasks = set.intersection(*(set(r) for r in reps.values()))
    return {t: sum(r[t]["success"] is True for r in reps.values()) for t in sorted(tasks)
            if all(r[t]["success"] is not None for r in reps.values())}


# ---- random streams and intervals

def rng(*key):
    """The random stream of one named draw set, e.g. rng("pooled", "task resample"): independent of every other draw set
    and the same in every process. Draw once from it and compute every statistic of the set on those draws."""
    return np.random.default_rng([SEED, *(zlib.crc32(str(k).encode()) for k in key)])


def boot_index(n, *key, b=N_BOOT):
    """(b, n) indices of n tasks resampled with replacement: one draw set, named by key, shared by every statistic
    computed on these resamples."""
    return rng("boot", *key).integers(0, n, size=(b, n))


def interval(draws, axis=0):
    """The two-sided percentile interval at LEVEL."""
    lo, hi = np.percentile(draws, [50 * (1 - LEVEL), 50 * (1 + LEVEL)], axis=axis)
    return float(lo) if np.ndim(lo) == 0 else lo, float(hi) if np.ndim(hi) == 0 else hi


def upper(draws):
    """The one-sided upper bound at LEVEL."""
    return float(np.percentile(draws, 100 * LEVEL))


def lower(draws):
    """The one-sided lower bound at LEVEL."""
    return float(np.percentile(draws, 100 * (1 - LEVEL)))


def wilson(k, n, z=Z95):
    """Wilson interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


# ---- outputs

def cache(folder):
    """The directory of a folder's intermediate files, in the analysis cache (BIPA_ANALYSIS_CACHE)."""
    return out_dir(folder)


def runs_text(cells, extra_runs=()):
    """The Runs section of a record: each (model, bench, method) it reads and the run the pins give, then any other run
    (a path from extra())."""
    lines = []
    for model, bench, method in cells:
        r = run(model, bench, method)
        note = " (returns the seed: Table 1 reports the Seed row's runs)" if returned_seed(model, bench, method) else ""
        lines.append(f"- {MODEL_SHORT[model]} {BENCH_SHORT[bench]} {ARM_NAME[method]}: `{r.relative_to(layout.EXECUTION)}`{note}")
    lines += [f"- `{p.relative_to(layout.EXECUTION)}`" for p in extra_runs]
    return "\n".join(lines + ["", f"Pins: `{layout.EXPERIMENTS.name}`."])
