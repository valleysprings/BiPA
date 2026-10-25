"""Per-episode readers of the avg@3 test runs of both models: load_cell(model, arm, bench) -> {rep: {task: episode fields}}."""
import collections
import json

from src.analysis.common import rep_files
from src.utils.dataloader import EX, SEED_TEXT, current_run, final_skill, vid

LUNA = EX / "gpt-6-luna"
LUNA_TAG = {"native-noseed": "native-20261011", "native-seed": "native-20261011",
            "gepa": "native-20261012", "skillopt": "native-20261012", "skilladam": "native-20261012"}


def ep_metrics(raw):
    """Success (None unless verified), input tokens X, executed calls N, completed requests R and the tool-name sequence."""
    seen, reqs = set(), []
    for r in raw.get("requests", []):
        if r["request_id"] not in seen:
            seen.add(r["request_id"])
            if r.get("input_tokens") is not None:  # a provider-failed request the client retried
                reqs.append(r)
    seq = [c["tool"] for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False]
    ok = type(raw.get("success")) is bool and raw.get("outcome_valid", True) is not False
    return {"success": bool(raw["success"]) if ok else None,
            "X": sum(r["input_tokens"] for r in reqs), "N": len(seq), "R": len(reqs), "seq": seq}


def rows_from_file(path):
    eps = {}
    for e in json.loads(path.read_text())["episodes"]:
        eps.setdefault(e["task_id"], e)
    return {t: ep_metrics(e) for t, e in eps.items()}


def deepseek_reps(arm, bench):
    """{rep: evaluation.json} of one DeepSeek arm, the files Table 1 averages."""
    run = current_run(arm, "deepseek-flash", bench)
    dirs = [run, run.parent / "rep-1", run.parent / "rep-2"] if arm == "native-seed" else \
        [run, EX / "deepseek-flash/test-reps" / f"{bench}__{arm}__{run.name}"]
    return rep_files(dirs, vid(final_skill(arm, run)))


def slot(row):
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def known(ep):
    return type(ep.get("success")) is bool and ep.get("outcome_valid", True) is True


def valid_slots(data):
    panel = [slot(r) for r in data.get("panel", [])]
    eps = [slot(r) for r in data.get("episodes", [])]
    return len(set(panel)) == len(panel) and len(set(eps)) == len(eps) and set(eps) <= set(panel)


def base_eval(dirs, version, tasks, rep):
    """The first test evaluation of the version whose panel is exactly the tasks at this repetition."""
    expected = collections.Counter((t, rep) for t in tasks)
    for d in dirs:
        if d and (d / "results/evaluations/test").is_dir():
            for path in sorted((d / "results/evaluations/test").glob("*/evaluation.json")):
                data = json.loads(path.read_text())
                if (data.get("version_id") == version and data.get("split") == "test" and valid_slots(data)
                        and collections.Counter(slot(r) for r in data.get("panel", [])) == expected):
                    return data
    return None


def collect(run, out, version, tasks, rep):
    """One repetition as repeat merges it: (base found, {task: episode}, known tasks); supplements fill unknown slots."""
    data = base_eval([run, out] if rep == 0 else [out, run.parent / f"rep-{rep}"], version, tasks, rep)
    merged = {slot(e)[0]: e for e in data.get("episodes", [])} if data else {}
    done = {t: None for t, e in merged.items() if known(e)}  # ordered: the base's known slots, then the fills
    expected = {(t, rep) for t in tasks}
    for path in sorted((out / "results/evaluations/test").glob("*/evaluation.json")) if out else []:
        extra = json.loads(path.read_text())
        panel = [slot(r) for r in extra.get("panel", [])]
        if extra.get("version_id") != version or extra.get("split") != "test" or not panel \
                or not valid_slots(extra) or not set(panel) <= expected:
            continue
        for e in extra.get("episodes", []):
            if slot(e) in panel and slot(e)[0] not in done:
                merged[slot(e)[0]] = e
                if known(e):
                    done[slot(e)[0]] = None
    return data is not None, merged, done


def luna_reps(arm, bench):
    """{rep: {task: ep_metrics}} of one Luna arm: rep 0 from the source run, rep 1/2 from its test-reps directory."""
    run = LUNA / arm / bench / LUNA_TAG[arm]
    out = LUNA / "test-reps" / f"{bench}__{arm}__{LUNA_TAG[arm]}"
    out = out if out.is_dir() else None
    version = vid(json.loads((run / "summary.json").read_text())["final_skill"])
    tasks = [slot(r)[0] for r in json.loads((run / "native_protocol.json").read_text())["panels"]["test"]]
    reps = {}
    for k in range(3):
        found, merged, _ = collect(run, out, version, tasks, k)
        if found:
            reps[k] = {t: ep_metrics(e) for t, e in merged.items()}
    return reps


def load_cell(model, arm, bench):
    if model == "gpt-6-luna":
        return luna_reps(arm, bench)
    return {rep: rows_from_file(path) for rep, path in deepseek_reps(arm, bench).items()}


def luna_seed_skill(arm, bench):
    """True when the Luna arm's skill is the seed (TB2's three optimizers returned it)."""
    if arm in ("native-noseed", "native-seed"):
        return arm == "native-seed"
    return json.loads((LUNA / arm / bench / LUNA_TAG[arm] / "summary.json").read_text())["final_skill"] == SEED_TEXT


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def editsim(a, b):
    return 1.0 if not a and not b else 1.0 - levenshtein(a, b) / max(len(a), len(b))
