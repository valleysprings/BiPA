"""mech_devsize: how the dev gain reported for identical candidates and the MDE80 of a single-run comparison change with the
size of the dev panel, under the task model of aux.selection_replay.selection.

Run: python -m aux.selection_replay.devsize   -> CACHE/analysis/main/selection_devsize.json, data/mech/mech_devsize.md (reads selection.json)
"""
import json

import numpy as np

from aux.selection_replay.selection import (BENCHES, MODERATE, best_of, check_inputs, executions, fit_prior, mde80, outcomes, reseed,
                                            seed_files, task_probabilities)
from src.utils import dataloader, logger

DEVSIZE_DRAWS = 40_000
DEV_SIZES = (9, 15, 30, 60, 120)
DEVSIZE_RULES = (("SkillOpt", 4), ("GEPA", 7))
PRIORS = ("fitted", "moderate")


def devsize(dev, prior, index, j, baseline_quantile) -> dict:
    """Per dev size: E[announced gain] (points) of SkillOpt's best of 4 and GEPA's argmax over 7 with a random baseline and with the baseline
    at the run's own seed-draw mid-quantile, and the MDE80 of a single-run comparison (points)."""
    rows = {}
    for s, size in enumerate(DEV_SIZES):
        reseed(index, j, 100 + s)
        p = task_probabilities(dev, prior, size, DEVSIZE_DRAWS)
        base = outcomes(p)
        at_quantile = float(np.quantile(base, baseline_quantile, method="nearest"))
        row = {"baseline_at_quantile": at_quantile}
        for rule, k in DEVSIZE_RULES:
            row[rule] = {"E_gain_random_points": 100 * float(best_of(p, k, base).mean()) / size,
                         "E_gain_quantile_points": 100 * float(best_of(p, k, at_quantile).mean()) / size}
        critical, mde = mde80(p, step=0.5 if size > 30 else 0.25)
        row["MDE80_points"] = 100 * mde / size
        row["critical_net_tasks"] = critical
        rows[str(size)] = row
    return rows


def compute():
    """Per benchmark and prior, the dev-size rows from the seed-draw mid-quantile that selection.json recorded."""
    sel = json.loads((dataloader.CACHE / "selection.json").read_text())
    result = {}
    for index, (bench, _) in enumerate(BENCHES):
        dev_files, test_files = seed_files(bench)
        dev, test = executions(dev_files), executions(test_files)
        check_inputs(bench, dev, test)
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = fit_prior(pooled)
        mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        result[bench] = {"priors": {}, "devsize": {}}
        for j, name in enumerate(PRIORS):
            prior = (a, b) if name == "fitted" else (MODERATE * mean, MODERATE * (1 - mean))
            q = sel[bench]["priors"][name]["seed_draw_mid_quantile"]
            result[bench]["priors"][name] = {"seed_draw_mid_quantile": q}
            result[bench]["devsize"][name] = devsize(dev, prior, index, j, q)
    dataloader.write_json("selection_devsize", result)
    record_devsize(result)


def record_devsize(result):
    """The audit record data/mech/mech_devsize.md."""
    rows = []
    for bench, _ in BENCHES:
        for prior in PRIORS:
            q = result[bench]["priors"][prior]["seed_draw_mid_quantile"]
            for size in DEV_SIZES:
                r = result[bench]["devsize"][prior][str(size)]
                rows.append([bench, prior, size, f"{q:.2f} -> {r['baseline_at_quantile']:.0f}/{size}",
                             f"{r['SkillOpt']['E_gain_random_points']:.1f}", f"{r['SkillOpt']['E_gain_quantile_points']:.1f}",
                             f"{r['GEPA']['E_gain_random_points']:.1f}", f"{r['GEPA']['E_gain_quantile_points']:.1f}",
                             f"{r['MDE80_points']:.1f}", r["critical_net_tasks"]])
    logger.write_record(
        "devsize",
        "How the gain a success-only selector announces under no effect, and the smallest effect a single-run dev comparison detects, change "
        "with the size of the dev panel (SkillOpt's best of 4, GEPA's argmax over 7), in percentage points of the dev panel.",
        "The task model of mech_selection (the same seed executions and fitted / moderate priors).",
        f"Per dev size, {DEVSIZE_DRAWS} draws: the dev tasks are a random subset of the benchmark's dev tasks (posteriors) when the size is at most "
        "its dev panel, else all of them plus tasks drawn from the prior. Random baseline: the seed's score drawn like the candidates. Quantile "
        "baseline: the baseline fixed at the same mid-quantile of the seed-score distribution as the run's own seed draw at the real dev size "
        "(the quantile and the resulting score are listed). MDE80: the smallest true gain detected with probability 0.8 by a one-sided 0.05 "
        "comparison of two single runs. Random seed [20261012, benchmark index, prior index, 100 + size index].",
        logger.md_table(["bench", "prior", "dev size", "seed-draw quantile -> baseline", "SkillOpt E[gain] random (pts)", "SkillOpt E[gain] at quantile (pts)",
                    "GEPA E[gain] random (pts)", "GEPA E[gain] at quantile (pts)", "MDE80 (pts)", "critical net difference (tasks)"], rows),
        "python -m aux.selection_replay.devsize")


if __name__ == "__main__":
    compute()
