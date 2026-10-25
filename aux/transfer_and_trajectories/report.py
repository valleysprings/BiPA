"""Success and cost-ratio statistics of a run's test episodes (only the
parts aux.transfer_and_trajectories.transfer uses: stats() and ratios(), which need no graph allocator).

Episodes are objects with the task, the verified outcome `success` (None when unknown), the input tokens `X` and the
number of tool calls `N` (transfer.episode builds them from evaluation.json).
"""

import math


def stats(success: list, episodes: list) -> dict:
    """The statistics of one run.

    success: the per-task outcomes (1 / 0, or None when the outcome is unknown).  episodes: the run's workload episodes.
    Returns {success, known, tasks, input_tokens, episodes}: the mean success over the known outcomes (None without any), how many
    outcomes are known, how many tasks there are, the mean input tokens per episode (None without episodes), the episode count."""
    known = [y for y in success if y is not None]
    if known:
        mean_success = sum(known) / len(known)
    else:
        mean_success = None

    if episodes:
        mean_tokens = sum(e.X for e in episodes) / len(episodes)
    else:
        mean_tokens = None

    return {
        "success": mean_success,
        "known": len(known),
        "tasks": len(success),
        "input_tokens": mean_tokens,
        "episodes": len(episodes),
    }


def usable(episode) -> bool:
    """Whether an episode enters a ratio: a verified outcome and positive input tokens and tool calls (the logarithm needs both)."""
    return episode.success is not None and episode.X > 0 and episode.N > 0


def geometric_mean_ratio(runs_by_task: dict, seed_by_task: dict, field: str) -> float:
    """exp(mean over tasks of log(runs' mean of `field` / the seed's mean of `field`)); `field` is "X" or "N".

    Both dicts map a task to its episodes; the tasks of the result are those of runs_by_task (all of them are in seed_by_task)."""
    log_sum = 0.0
    for task, run_episodes in runs_by_task.items():
        run_mean = sum(getattr(e, field) for e in run_episodes) / len(run_episodes)
        seed_episodes = seed_by_task[task]
        seed_mean = sum(getattr(e, field) for e in seed_episodes) / len(seed_episodes)
        log_sum += math.log(run_mean / seed_mean)
    return math.exp(log_sum / len(runs_by_task))


def ratios(runs_episodes: list, seed_episodes: list) -> dict | None:
    """Table 1's rho (input tokens X) and rho^calls (tool calls N) of the runs against the Seed row.

    Per task that both sides have a usable episode on, the ratio of the runs' mean (the runs pooled per task) to the Seed row's mean
    over its episodes (one per task, or one per run); the geometric mean of the ratios over the common tasks.
    Returns {"input", "calls", "tasks"}, or None without common tasks."""
    seed_by_task = {}
    for e in seed_episodes:
        if usable(e):
            seed_by_task.setdefault(e.task, []).append(e)

    runs_by_task = {}
    for episodes in runs_episodes:
        for e in episodes:
            if usable(e) and e.task in seed_by_task:
                runs_by_task.setdefault(e.task, []).append(e)

    if not runs_by_task:
        return None
    return {
        "input": geometric_mean_ratio(runs_by_task, seed_by_task, "X"),
        "calls": geometric_mean_ratio(runs_by_task, seed_by_task, "N"),
        "tasks": len(runs_by_task),
    }
