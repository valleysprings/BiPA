"""Learned skills against their seed, pooled over every learned skill of both models: the test success difference and the
input-token ratio, with cluster-bootstrap intervals and one-sided bounds.

A learned skill is one whose final text differs from the seed (settings.learned). Each is compared with the Seed row of its
model and benchmark on the tasks that the seed and every learned skill of that model and benchmark share. Per task, success
is the mean over the test runs with a verified outcome and input tokens the mean over those that read any (input tokens as in
aux.success.reruns); a GPT-6 Luna run that recorded no request, every request having failed at the provider, counts for
neither, while a DeepSeek run without input tokens still counts its verified outcome. The difference is in points, the ratio
is taken in logs. A pool
averages the per-task values of its skills with each (task, skill) counted once (task-weighted), or averages the skills' own
means (skill-weighted). The cluster bootstrap resamples the tasks of each model and benchmark N_BOOT times and keeps a task's
skills together, since they share the task and the seed run; the same resamples give every pool. The one-sided bound is
upper for the success difference and lower for the ratio."""
import json, math

import numpy as np

from aux.success.reruns import runs
from src.analysis import settings as S
from src.utils import logger

POOLS = (("DeepSeek", (S.DEEPSEEK,)), ("GPT-6-Luna", (S.LUNA,)), ("both models", (S.DEEPSEEK, S.LUNA)))


def task_means(model, bench, method):
    """{task: (success, input tokens)} over the runs the module docstring names; tasks without input tokens are left out."""
    per = {}
    for run in runs(model, bench, method):
        for t, (solved, x) in run.items():
            if x is not None or model == S.DEEPSEEK:
                per.setdefault(t, []).append((solved, x))
    return {t: (float(np.mean([s for s, _ in v])), float(np.mean([x for _, x in v if x]))) for t, v in per.items() if any(x for _, x in v)}


def blocks():
    """{(model, bench): {"skills": [method], "dq": (tasks, skills) points, "lr": (tasks, skills) log ratios}}."""
    out = {}
    for model, bench in S.learned_pairs():
        skills = [o for m, b, o in S.learned() if (m, b) == (model, bench)]
        seed = task_means(model, bench, "native-seed")
        own = {o: task_means(model, bench, o) for o in skills}
        tasks = sorted(set(seed).intersection(*own.values()))
        out[model, bench] = {"skills": skills,
                             "dq": np.array([[100 * (own[o][t][0] - seed[t][0]) for o in skills] for t in tasks]),
                             "lr": np.array([[math.log(own[o][t][1] / seed[t][1]) for o in skills] for t in tasks])}
    return out


def draws(bl):
    """Per block, the point and the bootstrap draws of the task sums and of each skill's mean: one task resample per block."""
    out = {}
    for key, b in bl.items():
        idx = S.boot_index(len(b["dq"]), "pooled", *key)
        out[key] = {v: {"sum": (b[v].sum(), b[v].sum(axis=1)[idx].sum(axis=1)), "mean": (b[v].mean(axis=0), b[v][idx].mean(axis=1))}
                    for v in ("dq", "lr")}
    return out


def pooled(bl, dr, keys):
    """Task- and skill-weighted means of a pool of blocks, each as (point, draws)."""
    pairs = sum(bl[k]["dq"].size for k in keys)
    out = {}
    for v in ("dq", "lr"):
        out[v, "task"] = tuple(sum(dr[k][v]["sum"][i] for k in keys) / pairs for i in (0, 1))
        out[v, "skill"] = tuple(np.concatenate([np.atleast_1d(dr[k][v]["mean"][i].T) for k in keys]).mean(axis=0) for i in (0, 1))
    return out


def numbers(bl, dr):
    rows = [[S.MODEL_SHORT[model], S.BENCH_SHORT[bench], S.ARM_NAME[o], len(b["dq"]), f"{b['dq'][:, j].mean():+.1f}",
             f"{math.exp(b['lr'][:, j].mean()):.2f}"] for (model, bench), b in bl.items() for j, o in enumerate(b["skills"])]
    out = [logger.md_table(["model", "bench", "skill", "tasks", "dQ", "rho"], rows), ""]
    rows = []
    for name, models in POOLS:
        keys = [k for k in bl if k[0] in models]
        res = pooled(bl, dr, keys)
        cells = []
        for v, w in (("dq", "task"), ("dq", "skill"), ("lr", "task"), ("lr", "skill")):
            point, boot = res[v, w]
            lo, hi = S.interval(boot)
            if v == "dq":
                cells += [f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{S.upper(boot):+.2f}"]
            else:
                cells += [f"{math.exp(point):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}]", f"{math.exp(S.lower(boot)):.3f}"]
        skills = sum(len(bl[k]["skills"]) for k in keys)
        rows.append([name, skills, sum(len(bl[k]["dq"]) for k in keys)] + cells)
    out += [logger.md_table(["pool", "skills", "tasks", "dQ, task-weighted [95%]", "upper bound, task-weighted", "dQ, skill-weighted [95%]",
                             "upper bound, skill-weighted", "rho, task-weighted [95%]", "lower bound, task-weighted",
                             "rho, skill-weighted [95%]", "lower bound, skill-weighted"], rows)]
    return "\n".join(out)


def main():
    bl = blocks()
    dr = draws(bl)
    (S.cache("success") / "pooled.json").write_text(json.dumps(
        {f"{m}|{b}": {"skills": v["skills"], "dq": v["dq"].tolist(), "lr": v["lr"].tolist()} for (m, b), v in bl.items()}))
    seeds = [(m, b, "native-seed") for m, b in bl]
    logger.record(__spec__.name, "pooled", numbers(bl, dr), S.runs_text(seeds + S.learned()))


if __name__ == "__main__":
    main()
