"""Task panels and training batches (configs/tasks.yaml), sampled from harness-bench's registry with seeded splits."""
import collections, functools, hashlib, json, math, random, re, warnings

import yaml
from harness_bench.eval import registry, registry_panel
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.layout import HARNESS_BENCH, ROOT

CONFIG = yaml.safe_load((ROOT / "configs/tasks.yaml").read_text(encoding="utf-8"))
BENCHMARKS = CONFIG["benchmarks"]


# Every panel and batch is drawn from seeds derived from this salt and the token strings below ("panel-sampling",
# "train-stream", "train-batches"). The recorded runs fixed them: changing any of them redraws every panel and batch.
# ("bach" is the harness project the runs were made in.)
SALT = "bach-training-stream-v1"


def _seed(*token):
    """A sampling seed derived from ``token``."""
    text = json.dumps([SALT, *token], separators=(",", ":"))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _strata(task_ids, pattern):
    """Stratum per task (regex group 1); lone tasks share one stratum; None when fewer than two strata."""
    labels = [(m.group(1) if (m := re.match(pattern, task)) else "") if pattern else "" for task in task_ids]
    counts = collections.Counter(labels)
    labels = [label if counts[label] > 1 else "\x00rare" for label in labels]
    counts = collections.Counter(labels)
    return None if len(counts) < 2 or min(counts.values()) < 2 else labels


def sample(rows, cap, seed, strata):
    """The rows of ``cap`` tasks drawn by a seeded stratified split, in panel order."""
    tasks = sorted({row["task_id"] for row in rows})
    if cap >= len(tasks):
        return rows
    keep = set(train_test_split(tasks, train_size=cap, shuffle=True, stratify=_strata(tasks, strata),
                                random_state=seed % 2**32)[0])
    return [row for row in rows if row["task_id"] in keep]


def shard(rows, size, seed, strata):
    """One pass in batches of about ``size``: the folds of a shuffled (stratified) k-fold over the sorted rows."""
    ordered = sorted(rows, key=lambda row: (row["repetition"], row["task_id"]))
    splits = math.ceil(len(ordered) / size)
    labels = _strata([row["task_id"] for row in ordered], strata)
    if labels is not None and max(collections.Counter(labels).values()) < splits:
        labels = None
    splitter = (StratifiedKFold if labels is not None else KFold)(n_splits=splits, shuffle=True, random_state=seed % 2**32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        folds = [test for _, test in splitter.split([[0]] * len(ordered), labels)]
    return [[ordered[index]["task_id"] for index in sorted(fold)] for fold in folds]


def plan_passes(tasks, passes, seed, budget, window):
    """``passes`` seeded reshuffles of the tasks cut at ``budget`` slots; each pass starts with tasks that were not
    among the last ``window`` of the previous one, so no ``window`` consecutive slots repeat a task."""
    planned = []
    for index in range(passes):
        order = random.Random(_seed("train-stream", seed, index)).sample(sorted(tasks), len(tasks))
        if planned:
            head = [task for task in order if task not in planned[-1][-window:]][:window]
            order = head + [task for task in order if task not in head]
        planned.append(order[:budget - sum(map(len, planned))])
    return [order for order in planned if order]


def slice_stream(passes, size):
    """Consecutive full batches over the concatenated passes (a batch may span two passes)."""
    stream = [task for order in passes for task in order]
    return [stream[start:start + size] for start in range(0, len(stream) - size + 1, size)]


def chunk_passes(passes, size):
    """Upstream SkillOpt's epochs: a full pass cut in order into batches of ``size``, a partial last pass split into
    as many near-equal batches as a full pass has steps."""
    steps, batches = math.ceil(len(passes[0]) / size), []
    for order in passes:
        if len(order) == len(passes[0]):
            batches += [order[start:start + size] for start in range(0, len(order), size)]
        else:
            cuts = [len(order) * step // steps for step in range(steps + 1)]
            batches += [order[cuts[step]:cuts[step + 1]] for step in range(steps)]
    return batches


@functools.cache
def _panels(benchmark, scale):
    spec = BENCHMARKS[benchmark]["scales"][scale]
    rows = {split: registry_panel(benchmark, split, str(HARNESS_BENCH))
            for split in registry(str(HARNESS_BENCH))["tasks"][benchmark]["panels"]}
    strata = spec["panel_sampling"].get("strata")
    for split, cap in spec["panel_caps"].items():
        rows[split] = (sample(rows[split], cap, _seed("panel-sampling", benchmark, split, "subset"), strata)
                       if spec["panel_sampling"].get(split) == "stratified" else rows[split][:cap])
    rows["sel"] = sample(rows["dev"], spec["sel"], _seed("panel-sampling", benchmark, "dev", "selection"), strata)
    return rows


def panels(benchmark, scale="main"):
    """{"train", "sel", "test"}: lists of task ids."""
    return {split: [row["task_id"] for row in _panels(benchmark, scale)[split]] for split in ("train", "sel", "test")}


@functools.cache
def batches(benchmark, scale="main", method="default"):
    """B_1 ... B_T, lists of task ids at the method's batch size, cut at the train episode budget.

    One pass (tau3, AutomationBench): stratified folds of the train panel. Several passes (TB2): one stream shared by
    every method; its no-repeat window is the largest batch size of the methods that slice it."""
    spec = BENCHMARKS[benchmark]["scales"][scale]
    sizes = spec["batch_size"] if isinstance(spec["batch_size"], dict) else {"default": spec["batch_size"]}
    size, seed, train = sizes.get(method, sizes["default"]), _seed("panel-sampling", benchmark, "train", "batches"), \
        _panels(benchmark, scale)["train"]
    if spec["rounds"] > 1:
        aligned = spec.get("pass_aligned", [])
        window = max(value for key, value in sizes.items() if key not in aligned)
        passes = plan_passes([row["task_id"] for row in train], spec["rounds"], seed, spec["train_episodes"], window)
        return (chunk_passes if method in aligned else slice_stream)(passes, size)
    kept, total = [], 0
    for batch in shard(train, size, _seed("train-batches", seed, 0), spec["panel_sampling"].get("strata")):
        if total + len(batch) > spec.get("train_episodes", math.inf):
            break
        kept.append(batch)
        total += len(batch)
    return kept
