"""Task panels and training batches using sampling functions derived from slim.
Current candidate config differs from historical slim main (160/b16 versus 60/b6
on tau3/AutoB); shared sampling functions do not imply identical panels or protocol.
Arms within a configured cell receive the same scheduled panels before algorithmic adaptations."""

import collections, copy, functools, hashlib, json, math, re, warnings
from pathlib import Path

import yaml
from harness_bench.eval import normalize_panel, registry, registry_panel
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "configs/tasks.yaml").read_text(encoding="utf-8"))
STREAM_SCHEMA = "bach-training-stream-v1"  # slim's schema string: every sampling seed hashes it


def _seed(*token) -> int:
    return int(hashlib.sha256(json.dumps([STREAM_SCHEMA, *token], separators=(",", ":")).encode()).hexdigest()[:16], 16)


def _strata_labels(task_ids: list[str], pattern: str | None) -> list[str] | None:
    """Stratum per task (regex group 1, "" unmatched); lone tasks share one "rare" stratum; None if < 2 strata."""
    labels = [(m.group(1) if (m := re.match(pattern, task)) else "") if pattern else "" for task in task_ids]
    counts = collections.Counter(labels)
    labels = [label if counts[label] > 1 else "\x00rare" for label in labels]
    counts = collections.Counter(labels)
    return None if len(counts) < 2 or min(counts.values()) < 2 else labels


def sample_panel(panel: list[dict], cap: int, seed: int, strata: str | None) -> list[dict]:
    """The rows of ``cap`` tasks drawn by sklearn's seeded stratified split, in panel order."""
    tasks = sorted({row["task_id"] for row in panel})
    if cap >= len(tasks):
        return copy.deepcopy(panel)
    keep = set(train_test_split(tasks, train_size=cap, shuffle=True, stratify=_strata_labels(tasks, strata),
                                random_state=int(seed) % 2**32)[0])
    return [copy.deepcopy(row) for row in panel if row["task_id"] in keep]


def shard(panel: list[dict], batch_size: int, seed: int, strata: str | None) -> list[list[dict]]:
    """One epoch's batches: the folds of a shuffled (Stratified)KFold over the sorted panel."""
    ordered = sorted(panel, key=lambda row: (row["repetition"], row["task_id"]))
    splits = math.ceil(len(ordered) / batch_size)
    labels = _strata_labels([row["task_id"] for row in ordered], strata)
    if labels is not None and max(collections.Counter(labels).values()) < splits:
        labels = None
    splitter = (StratifiedKFold if labels is not None else KFold)(n_splits=splits, shuffle=True,
                                                                  random_state=int(seed) % 2**32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        folds = [test for _, test in splitter.split([[0]] * len(ordered), labels)]
    return [[ordered[index] for index in sorted(fold)] for fold in folds]


def plan_stream(panel: list[dict], epochs: int, stream: dict) -> list[list[dict]]:
    seeds = [_seed("train-batches", int(stream["batch_seed"]), epoch) for epoch in range(epochs)]
    return [batch for seed in seeds for batch in shard(panel, stream["batch_size"], seed, stream.get("batch_strata"))]


def sensitive(benchmark: str) -> tuple:
    """Train tasks the model gateway refuses (configs/sensitive-train.yaml)."""
    listed = yaml.safe_load((Path(__file__).resolve().parent / "configs/sensitive-train.yaml").read_text()) or {}
    return tuple(listed.get(benchmark) or ())


@functools.cache
def _panels(benchmark: str, scale: str, exclude: tuple = ()) -> dict:
    """Capped panels from harness-bench's registry and D_sel (seeds depend only on the benchmark); ``exclude`` drops
    train tasks after sampling, so the other splits and the remaining train tasks are unchanged."""
    spec = CONFIG["benchmarks"][benchmark]["scales"][scale]
    bench_root = CONFIG["backend"]["bench_root"]
    panels = {split: registry_panel(benchmark, split, bench_root)
              for split in registry(bench_root)["tasks"][benchmark]["panels"]}
    # drop_tasks: per-split regex of task ids removed before capping/sampling (e.g. a held-out domain:
    # train and dev drop it, so D_sel never sees it either; test keeps it to measure the transfer)
    for split, pattern in (spec.get("drop_tasks") or {}).items():
        panels[split] = [row for row in panels[split] if not re.match(pattern, row["task_id"])]
    sampling, strata = spec["panel_sampling"], spec["panel_sampling"].get("strata")
    for split, cap in spec["panel_caps"].items():
        panels[split] = (sample_panel(panels[split], cap, _seed("panel-sampling", benchmark, split, "subset"), strata)
                         if sampling.get(split) == "stratified" else panels[split][:cap])
    panels["sel"] = sample_panel(panels["dev"], spec["sel"], _seed("panel-sampling", benchmark, "dev", "selection"),
                                 strata)
    panels["train"] = [row for row in panels["train"] if row["task_id"] not in exclude]
    return panels


def panels(benchmark: str, scale: str = "main", exclude: tuple | None = None) -> dict:
    """{"train", "sel", "test"}: lists of task ids; ``exclude`` defaults to the gateway-sensitive train tasks."""
    exclude = sensitive(benchmark) if exclude is None else tuple(exclude)
    return {split: [row["task_id"] for row in _panels(benchmark, scale, exclude)[split]] for split in ("train", "sel", "test")}


@functools.cache
def batches(benchmark: str, scale: str = "main", arm: str = "default", exclude: tuple | None = None) -> list[list[str]]:
    """B_1 … B_T as lists of task ids: the arm's upstream batch size over the shared train panel, cut at the train
    episode budget (smoke: all batches of the given rounds)."""
    spec = CONFIG["benchmarks"][benchmark]["scales"][scale]
    size = spec["batch_size"] if isinstance(spec["batch_size"], int) else spec["batch_size"].get(arm, spec["batch_size"]["default"])
    stream = {"batch_seed": _seed("panel-sampling", benchmark, "train", "batches"), "batch_size": size,
              "batch_strata": spec["panel_sampling"].get("strata")}
    exclude = sensitive(benchmark) if exclude is None else tuple(exclude)
    planned, kept, total = plan_stream(_panels(benchmark, scale, exclude)["train"], spec["rounds"], stream), [], 0
    for batch in planned:
        if "train_episodes" in spec and total + len(batch) > spec["train_episodes"]:
            break
        kept.append([row["task_id"] for row in batch]); total += len(batch)
    steps = spec.get("steps")
    steps = steps.get(arm) if isinstance(steps, dict) else steps
    return kept[:steps]
