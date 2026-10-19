"""Paper 1 (mechanism paper), reviewer P1-4 + X3: is the token tax larger on tasks that fail, and is it flailing?

Role in the argument: Appendix B (tab:paired) shows that on tasks both the skill and the seed fail in all three runs, the learned skills read 1.07-1.27 x the
seed's input tokens on tau3 / AutoBench and 1.90-2.55 x on TB2 (n = 3-4). A reviewer asks whether failures are expensive because the agent flails (more actions,
more turns) when it fails. This module splits every (skill, benchmark) cell of attribution.py into three task subsets and reports, on each:

  a) tasks that skill and seed both solve in all three runs,
  b) tasks that skill and seed both fail in all three runs,
  c) the rest (mixed outcomes),

 1. the four-term attribution of attribution.py on the 3-run means (same functions: run_files, load_runs, mean_metrics, four_terms; total = mean dlog X);
 2. per episode (mean over tasks of the per-task 3-run mean) turns R, executed calls N, input tokens X of seed and skill, calls per turn N / R, and the ratio of
    these means plus the geometric mean of per-task ratios;
 3. how the skill's and the seed's failed episodes end, from the recorded backend classification: task_failure (agent stopped / verifier says wrong, no limit hit),
    timeout (wall-clock limit of the harness: AgentTimeoutError), anything else; and whether the last request of the episode still issued tool calls.
    The harness records no turn cap, so a turn limit is not separable from the agent stopping; only the wall-clock timeout is recorded.
 4. the contrast (b) minus (a) of the total and of each term, pooled over the nine cells and per benchmark, with a paired-by-cell task bootstrap interval
    (resample tasks within a subset, 10,000 draws, rng seed 20261012). Question answered: is the tax on failing tasks larger than on solved ones, and which term carries it?

Task set: attribution.py's usable tasks (Table 1 common set with a graph episode and positive history in all six runs); n therefore can be below tab:paired's n.
No BACH, no model calls. Runs as attribution.py.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.failcost compute  -> assets/cache/analysis/mech/failcost.json
                                .venv/bin/python -m src.analysis.mech.failcost record   -> paper/assets/data/mech/mech_failcost.md
"""
import argparse
import json
import math

import numpy as np

from src.analysis.mech import attribution as A
from src.utils import logger, dataloader
SUBSETS = [("solve", "(a) both solve 3/3"), ("fail", "(b) both fail 3/3"), ("mixed", "(c) mixed")]
N_BOOT = 10000
SEED = 20261012


def episode_endings(path, tasks):
    """{(task, rep): (classification, exception_type, last request has tool calls)} for the episodes of an evaluation.json."""
    data = json.loads(path.read_text())
    out = {}
    for ep in data["episodes"]:
        if ep["task_id"] not in tasks:
            continue
        key = (ep["task_id"], ep.get("repetition", 0))
        if key in out:
            continue
        backend = ep.get("backend_outcome") or {}
        requests = ep.get("requests") or []
        last_has_calls = bool(requests and requests[-1].get("tool_calls"))
        out[key] = (backend.get("classification"), backend.get("exception_type"), last_has_calls)
    return out


def split_tasks(rows_own, rows_seed, usable, reps):
    """Subset name of each usable task from the per-run Table 1 outcomes of both sides."""
    subset = {}
    for task in usable:
        own = [rows_own[r][task]["success"] is True for r in reps]
        seed = [rows_seed[r][task]["success"] is True for r in reps]
        if all(own) and all(seed):
            subset[task] = "solve"
        elif not any(own) and not any(seed):
            subset[task] = "fail"
        else:
            subset[task] = "mixed"
    return subset


def per_episode_stats(pairs):
    """Mean over tasks of the per-task 3-run means R, N, X of seed and skill, N / R, ratio of means and geometric mean of per-task ratios."""
    out = {}
    for side, idx in (("seed", 0), ("skill", 1)):
        out[side] = {k: float(np.mean([p[idx][k] for p in pairs])) for k in ("R", "N", "X")}
        out[side]["N_per_R"] = float(np.mean([p[idx]["N"] / p[idx]["R"] for p in pairs]))
    for k in ("R", "N", "X"):
        out[f"ratio_{k}"] = out["skill"][k] / out["seed"][k]
        out[f"geo_{k}"] = float(np.exp(np.mean([math.log(p[1][k] / p[0][k]) for p in pairs])))
    return out


def ending_counts(files, subset, tasks_in, reps):
    """Counts of the recorded ending of the failed episodes (all three runs) of the given tasks in one run set."""
    counts = {"episodes": 0, "failed": 0, "task_failure": 0, "timeout": 0, "other": 0, "last_request_has_calls": 0}
    for rep in reps:
        endings = episode_endings(files[rep], set(tasks_in))
        for (task, _), (cls, _, has_calls) in endings.items():
            counts["episodes"] += 1
            if cls == "success":
                continue
            counts["failed"] += 1
            if cls == "task_failure":
                counts["task_failure"] += 1
            elif cls == "timeout":
                counts["timeout"] += 1
            else:
                counts["other"] += 1
            counts["last_request_has_calls"] += int(has_calls)
    return counts


def compute():
    """One entry per cell and subset: terms, per-episode stats, endings; plus per-task arrays for the bootstrap."""
    cells = []
    for method_disp, method in A.OPT:
        for bench, short in A.BENCH:
            own_files, seed_files = A.run_files(method, bench)
            own_eps, own_rows = A.load_runs(own_files)
            seed_eps, seed_rows = A.load_runs(seed_files)
            reps = sorted(own_files)
            _, own_mean = A.T.mean_rows(own_files)
            _, seed_mean = A.T.mean_rows(seed_files)
            t1_tasks = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
            per_side = {}
            for t in t1_tasks:
                rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                              for rows in (own_rows, seed_rows) for r in reps)
                if not rows_ok:
                    continue
                if not all(t in eps[r] and eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
                    continue
                a = A.mean_metrics([seed_eps[r][t][1] for r in reps])
                b = A.mean_metrics([own_eps[r][t][1] for r in reps])
                if a is None or b is None:
                    continue
                per_side[t] = (a, b)
            usable = sorted(per_side)
            subset_of = split_tasks(own_rows, seed_rows, usable, reps)
            cell = dict(method=method_disp, bench=bench, short=short, n=len(usable), subsets={})
            # all usable tasks as a reference row (equals attribution.py's A)
            groups = dict(all=usable)
            for key, _ in SUBSETS:
                groups[key] = [t for t in usable if subset_of[t] == key]
            for key, tasks in groups.items():
                if not tasks:
                    cell["subsets"][key] = dict(n=0)
                    continue
                pairs = [per_side[t] for t in tasks]
                terms, dX = A.four_terms(pairs)
                entry = dict(n=len(tasks), terms={k: float(v.mean()) for k, v in terms.items()}, total=float(dX.mean()),
                             per_task_terms={k: [float(x) for x in v] for k, v in terms.items()}, per_task_total=[float(x) for x in dX],
                             stats=per_episode_stats(pairs), tasks=tasks)
                entry["endings_skill"] = ending_counts(own_files, subset_of, tasks, reps)
                entry["endings_seed"] = ending_counts(seed_files, subset_of, tasks, reps)
                cell["subsets"][key] = entry
            cells.append(cell)
            print(method_disp, short, {k: v["n"] for k, v in cell["subsets"].items()},
                  {k: round(v["total"], 3) for k, v in cell["subsets"].items() if v["n"]})
    dataloader.write_json("failcost", dict(model=A.MODEL, cells=cells))
    return cells


def boot_contrast(cells, key_a, key_b, bench=None):
    """Mean over cells of (subset b mean) - (subset a mean) for the total and each term, with a bootstrap interval from resampling tasks within each cell and subset."""
    rng = np.random.default_rng(SEED)
    chosen = [c for c in cells if (bench is None or c["short"] == bench) and c["subsets"][key_a]["n"] and c["subsets"][key_b]["n"]]
    names = A.TERMS + ["total"]
    point = {}
    draws = {k: np.zeros(N_BOOT) for k in names}
    for name in names:
        diffs = []
        for c in chosen:
            sa = c["subsets"][key_a]
            sb = c["subsets"][key_b]
            xa = np.array(sa["per_task_total"] if name == "total" else sa["per_task_terms"][name])
            xb = np.array(sb["per_task_total"] if name == "total" else sb["per_task_terms"][name])
            diffs.append((xa, xb))
        point[name] = float(np.mean([xb.mean() - xa.mean() for xa, xb in diffs]))
        for i in range(N_BOOT):
            vals = []
            for xa, xb in diffs:
                ra = xa[rng.integers(0, len(xa), len(xa))].mean()
                rb = xb[rng.integers(0, len(xb), len(xb))].mean()
                vals.append(rb - ra)
            draws[name][i] = np.mean(vals)
    return dict(n_cells=len(chosen), point=point,
                lo={k: float(np.percentile(v, 2.5)) for k, v in draws.items()},
                hi={k: float(np.percentile(v, 97.5)) for k, v in draws.items()})


def f(x, nd=3):
    return f"{x:+.{nd}f}"


def record(cells):
    """paper/assets/data/mech/mech_failcost.md."""
    rows_terms, rows_stats, rows_end = [], [], []
    for c in cells:
        for key, label in [("all", "all usable")] + SUBSETS:
            s = c["subsets"][key]
            if not s["n"]:
                rows_terms.append([c["method"], c["short"], label, 0, "-", "-", "-", "-", "-", "-"])
                continue
            t = s["terms"]
            rows_terms.append([c["method"], c["short"], label, s["n"]] + [f(t[k]) for k in A.TERMS] + [f(s["total"]), f"{math.exp(s['total']):.2f}"])
            st = s["stats"]
            rows_stats.append([c["method"], c["short"], label, s["n"],
                               f"{st['seed']['R']:.1f} / {st['skill']['R']:.1f}", f"{st['seed']['N']:.1f} / {st['skill']['N']:.1f}", f"{st['seed']['X'] / 1000:.0f}k / {st['skill']['X'] / 1000:.0f}k",
                               f"{st['seed']['N_per_R']:.2f} / {st['skill']['N_per_R']:.2f}",
                               f"{st['ratio_R']:.2f} / {st['ratio_N']:.2f} / {st['ratio_X']:.2f}", f"{st['geo_R']:.2f} / {st['geo_N']:.2f} / {st['geo_X']:.2f}"])
        for key, label in [("fail", "(b) both fail 3/3"), ("mixed", "(c) mixed"), ("solve", "(a) both solve 3/3")]:
            s = c["subsets"][key]
            if not s["n"]:
                continue
            for side in ("skill", "seed"):
                e = s["endings_" + side]
                if e["failed"]:
                    rows_end.append([c["method"], c["short"], label, side, e["episodes"], e["failed"], e["task_failure"], e["timeout"], e["other"],
                                     f"{e['last_request_has_calls']}"])
    contrast_all = boot_contrast(cells, "solve", "fail")
    contrast_bench = {b: boot_contrast(cells, "solve", "fail", b) for _, b in A.BENCH}
    names = A.TERMS + ["total"]
    rows_con = []
    for label, con in [("pooled 9 cells", contrast_all)] + [(b, contrast_bench[b]) for _, b in A.BENCH]:
        rows_con.append([label, con["n_cells"]] + [f"{f(con['point'][k], 2)} [{f(con['lo'][k], 2)}, {f(con['hi'][k], 2)}]" for k in names])
    # pooled episode-level endings over the nine cells for the skill, failing subset
    pooled = {}
    for side in ("skill", "seed"):
        for key in ("fail", "mixed", "solve"):
            agg = dict(episodes=0, failed=0, task_failure=0, timeout=0, other=0, last_request_has_calls=0)
            for c in cells:
                s = c["subsets"][key]
                if s["n"]:
                    for k in agg:
                        agg[k] += s["endings_" + side][k]
            pooled[(side, key)] = agg
    rows_pool = []
    for bench_filter in [None] + [b for _, b in A.BENCH]:
        for side in ("skill", "seed"):
            agg = dict(episodes=0, failed=0, task_failure=0, timeout=0, other=0, last_request_has_calls=0)
            for c in cells:
                s = c["subsets"]["fail"]
                if s["n"] and (bench_filter is None or c["short"] == bench_filter):
                    for k in agg:
                        agg[k] += s["endings_" + side][k]
            rows_pool.append([bench_filter or "all benchmarks", side, agg["failed"], agg["task_failure"], agg["timeout"], agg["other"], agg["last_request_has_calls"]])
    # number of cells where the failing-subset total exceeds the solved-subset total, and the same for actions
    both = [c for c in cells if c["subsets"]["solve"]["n"] and c["subsets"]["fail"]["n"]]
    n_total_gt = sum(c["subsets"]["fail"]["total"] > c["subsets"]["solve"]["total"] for c in both)
    n_act_gt = sum(c["subsets"]["fail"]["terms"]["more_actions"] > c["subsets"]["solve"]["terms"]["more_actions"] for c in both)
    dataloader.write_json("failcost_contrast", dict(pooled=contrast_all, per_bench=contrast_bench, cells_total_fail_gt_solve=n_total_gt, cells_actions_fail_gt_solve=n_act_gt, n_cells=len(both)))
    what = ("Is the input-token tax larger on tasks that fail, and is it flailing? For each skill x benchmark, the usable tasks of attribution.py are split into (a) both skill and seed solve in all three runs, "
            "(b) both fail in all three runs, (c) mixed; each subset gets attribution.py's four-term split on 3-run means (actions, turn-splitting, content per turn, skill text; total = mean dlog X), "
            "the mean per-episode turns R, executed calls N and input tokens X of seed and skill, and, for the failed episodes, the recorded ending (task_failure = agent stopped without a limit, "
            "timeout = harness wall-clock limit; no turn cap is recorded).")
    runs = "As attribution.py: skills native-20261008-https + test-reps rep 1/2, seed native-seed native-20261003-trace-final + rep-1/2, DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1."
    agg = ("Per task, R, N, X, F are averaged over the three runs of each side (attribution.mean_metrics) and the four terms are computed per task (attribution.four_terms); subset values are task means. "
           "Subsets use the per-run verified outcome of Table 1 on both sides. Contrast = mean over cells of (b) minus (a) task-mean term, interval from 10,000 bootstrap draws resampling tasks within each cell and subset "
           f"(numpy default_rng seed {SEED}); cells lacking either subset are skipped. Endings come from evaluation.json episodes[].backend_outcome.classification (success / task_failure / timeout) and whether the last recorded request still issued tool calls.")
    table = ("### 1. Four terms by subset (log units; ratio = exp(total))\n\n"
             + logger.md_table(["Skill", "Bench", "Subset", "n", "actions", "turn-split", "content", "skill text", "total", "exp(total)"], rows_terms)
             + "\n\n### 2. Per-episode behaviour by subset (seed / skill; mean over tasks of 3-run means)\n\n"
             + logger.md_table(["Skill", "Bench", "Subset", "n", "turns R", "calls N", "input tokens X", "calls per turn", "ratio of means R / N / X", "geo-mean ratio R / N / X"], rows_stats)
             + "\n\n### 3. How failed episodes end (all runs of the subset's tasks)\n\n"
             + logger.md_table(["Skill", "Bench", "Subset", "Side", "episodes", "failed", "task_failure", "timeout", "other", "last request had calls"], rows_end)
             + "\n\nBoth-fail subset (b) pooled over the nine cells:\n\n"
             + logger.md_table(["Scope", "Side", "failed episodes", "task_failure", "timeout", "other", "last request had calls"], rows_pool)
             + "\n\n### 4. Contrast (b) both fail minus (a) both solve, mean over cells, 95% bootstrap interval\n\n"
             + logger.md_table(["Scope", "cells"] + ["actions", "turn-split", "content", "skill text", "total"], rows_con)
             + f"\n\nCells where the failing-subset total exceeds the solved-subset total: {n_total_gt} of {len(both)}; where the failing-subset actions term exceeds the solved one: {n_act_gt} of {len(both)}.")
    logger.write_record("failcost", what, runs, agg, table, ".venv/bin/python -m src.analysis.mech.failcost compute\n.venv/bin/python -m src.analysis.mech.failcost record")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["compute", "record"])
    step = ap.parse_args().step
    if step == "compute":
        compute()
        return
    record(json.loads((dataloader.CACHE / "failcost.json").read_text())["cells"])


if __name__ == "__main__":
    main()
