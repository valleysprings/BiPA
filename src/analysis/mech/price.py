"""Price of the learned skills under cached-input pricing, recomputed on Table 1's data (replaces the first-run-only appC_price numbers).

Role: Appendix C's answer to "does the extra input cost real money once cached tokens are cheap?". For every benchmark and each of GEPA, SkillOpt, SkillAdam and No skill,
the price of a task is  uncached input + c * cached input + o * output  (token units), and the skill is compared to the Seed skill.
Data: identical to Table 1 / stats.py. The runs are the three test runs of each side (native-20261008-https + test-reps rep 1/2; seed native-seed + rep-1/2), the loaders
`stats.per_task_runs` and `stats.task_means` give the verified runs and the per-task means, `stats.common_tasks` gives the common task set.
Per task: the mean over the runs of each side of X (input), cached input and output tokens, then the price of the mean; ratio = price(skill) / price(seed).
Cached input per episode follows analysis.mechanism.appc: tokens served from cache from the second request on, min(cache_tokens, input_tokens) per distinct request.
Weights: main c = 0.1, o = 4; sensitivity c in {0.02, 0.1, 0.5} x o in {2, 4, 8} (nine settings, min / max of each ratio reported).
Outputs: assets/cache/analysis/mech/price3.json and paper/assets/data/mech/mech_price3.md. Run: .venv/bin/python -m src.analysis.mech.price
"""
import itertools
import json

import numpy as np

from src.analysis import table1 as T
from src.utils.dataloader import BENCHES, SEED_TEXT, current_run, final_skill, gmean, vid
from src.utils import logger, dataloader
from src.analysis.mech import stats as S

MODEL = S.MODEL
MAIN = (0.1, 4.0)
GRID = list(itertools.product((0.02, 0.1, 0.5), (2.0, 4.0, 8.0)))
SHORT = {"tau3-bench": "tau3-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}


def token_totals(path):
    """{(task, rep): dict(cached, out)} of one evaluation.json: cached input from request 2 on and output tokens over the distinct requests of the episode."""
    data = json.loads(path.read_text())
    episodes = {}
    for episode in data["episodes"]:
        episodes.setdefault((episode["task_id"], episode.get("repetition", 0)), episode)
    totals = {}
    for key, episode in episodes.items():
        seen = set()
        requests = []
        for request in episode.get("requests", []):
            if request["request_id"] in seen:
                continue
            seen.add(request["request_id"])
            requests.append(request)
        cached = 0.0
        for request in requests[1:]:
            usage = request.get("captured_usage") or {}
            cached += min(usage.get("cache_tokens") or 0, request["input_tokens"])
        totals[key] = dict(cached=cached, out=float(sum(r["output_tokens"] or 0 for r in requests)))
    return totals


def side_means(reps):
    """{task: dict(X, cached, out)}: mean over the verified runs of the side (the runs stats.per_task_runs keeps), plus the Table 1 means N, X for the common set."""
    sums = {}
    for _, path in sorted(reps.items()):
        totals = token_totals(path)
        for (task, rep), row in T.rows_of(path).items():
            # same run filter as stats.per_task_runs
            if row["success"] is None or row["N"] is None or row["X"] is None:
                continue
            tot = totals[(task, rep)]
            sums.setdefault(task, []).append((row["N"], row["X"], tot["cached"], tot["out"]))
    out = {}
    for task, runs in sums.items():
        n, x, cached, out_tok = np.mean(np.array(runs, float), axis=0)
        out[task] = dict(N=float(n), X=float(x), cached=float(cached), out=float(out_tok))
    return out


def price(side, task, c, o):
    """uncached input + c * cached input + o * output of the per-task mean."""
    r = side[task]
    return (r["X"] - r["cached"]) + c * r["cached"] + o * r["out"]


def ratios(ref, own, tasks, c, o):
    """(per-task geometric mean of price(own)/price(ref), ratio of summed prices)."""
    p_own = np.array([price(own, t, c, o) for t in tasks])
    p_ref = np.array([price(ref, t, c, o) for t in tasks])
    return float(gmean(p_own / p_ref)), float(p_own.sum() / p_ref.sum())


def block(ref, own, tasks):
    """All numbers of one (benchmark, method): shares, main-weight ratios, min/max over the nine settings."""
    geo, total = ratios(ref, own, tasks, *MAIN)
    grid = [ratios(ref, own, tasks, c, o) for c, o in GRID]
    x = sum(own[t]["X"] for t in tasks)
    out = sum(own[t]["out"] for t in tasks)
    return dict(
        n=len(tasks),
        cached_share=float(sum(own[t]["cached"] for t in tasks) / x),
        out_share=float(out / (out + x)),
        geo=geo, sum=total,
        geo_min=min(g for g, _ in grid), geo_max=max(g for g, _ in grid),
        sum_min=min(s for _, s in grid), sum_max=max(s for _, s in grid),
        seed_cached_share=float(sum(ref[t]["cached"] for t in tasks) / sum(ref[t]["X"] for t in tasks)),
    )


def compute():
    """{bench: {method: block}} on Table 1's runs and common task sets."""
    res = {}
    for bench in BENCHES:
        seed_run = current_run("native-seed", MODEL, bench)
        seed_reps = T.rep_files([seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"], vid(SEED_TEXT))
        ref = side_means(seed_reps)
        res[bench] = {}
        for name, method in S.METHODS:
            run = current_run(method, MODEL, bench)
            text = final_skill(method, run)
            reps = T.rep_files([run, T.EX / MODEL / "test-reps" / f"{bench}__{method}__{run.name}"], vid(text))
            own = side_means(reps)
            tasks = S.common_tasks(ref, own)
            res[bench][name] = block(ref, own, tasks)
            res[bench][name]["runs"] = len(reps)
    return res


def markdown(res):
    """One table of the main weights and sensitivity ranges."""
    head = ["Benchmark", "Method", "n", "cached share of input", "output share of tokens",
            "price, geo. mean", "[min, max] over 9", "price, ratio of sums", "[min, max] over 9"]
    rows = []
    for bench in BENCHES:
        for name, _ in S.METHODS:
            b = res[bench][name]
            rows.append([SHORT[bench], name, b["n"], f"{100 * b['cached_share']:.1f}%", f"{100 * b['out_share']:.2f}%",
                         f"x{b['geo']:.2f}", f"[{b['geo_min']:.2f}, {b['geo_max']:.2f}]",
                         f"x{b['sum']:.2f}", f"[{b['sum_min']:.2f}, {b['sum_max']:.2f}]"])
    return logger.md_table(head, rows)


def main():
    res = compute()
    dataloader.write_json("price3", dict(main=dict(c=MAIN[0], o=MAIN[1]), grid=GRID, results=res))
    commands = ".venv/bin/python -m src.analysis.mech.price"
    logger.write_record(
        "price3",
        "Price of each learned skill (and No skill) relative to the Seed skill when cached input is cheap: price = uncached input + c x cached input + o x output tokens.",
        "DeepSeek-V4.1-Flash in OpenCode, baseline runs only: GEPA, SkillOpt, SkillAdam (native-20261008-https + test-reps rep 1/2), No skill (native-noseed, same layout), "
        "seed native-seed/<bench>/native-20261003-trace-final + rep-1 + rep-2; tau3-bench, automationbench (AutoBench), terminal-bench-2-1 (TB2). Same runs and loaders as Table 1 / mech_stats.",
        "Per task the mean over the verified runs of each side of input tokens X, cached input (tokens served from cache from request 2 on, min(cache_tokens, input_tokens) per distinct request) "
        "and output tokens; price of the per-task mean; ratio skill / seed per task. Common task set = stats.common_tasks (Table 1). Main weights c = 0.1, o = 4; "
        "sensitivity c in {0.02, 0.1, 0.5} x o in {2, 4, 8}, [min, max] over the nine settings. Shares are of the skill side, pooled over the common tasks "
        "(cached / input; output / (input + output)).",
        markdown(res),
        "Machine-readable: assets/cache/analysis/mech/price3.json.",
        commands)
    print(markdown(res))


if __name__ == "__main__":
    main()
