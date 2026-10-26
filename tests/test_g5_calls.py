"""aux.calls: the vectorized bootstrap of b_R and the episode tally of the writes figure."""
import numpy as np

calls = __import__("aux.calls.calls", fromlist=["calls"])
figure = __import__("aux.calls.figure", fromlist=["figure"])


def records(seed=1, tasks=12):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(tasks):
        for skill in ("seed", "GEPA", "SkillOpt"):
            for _ in range(3):
                n, r = rng.integers(2, 40), rng.integers(2, 30)
                out.append(dict(task=f"t{t:02d}", skill=skill, N=n, R=r, X=1000 * n ** 0.3 * r ** 1.2 * rng.lognormal(0, 0.2)))
    return out


def test_resampled_b_r_is_the_fit_on_the_drawn_tasks():
    recs = records()
    names = sorted({r["task"] for r in recs})
    assert np.isclose(calls.resampled_b_r(recs, np.ones((1, len(names))))[0], calls.regressions(recs)["NR"]["coef"][1])
    # a task drawn twice enters as two clusters, a task not drawn not at all
    counts = np.ones(len(names))
    counts[0], counts[1] = 2, 0
    drawn = [dict(r, task=f"{r['task']}#{k}") for r in recs for k in range(int(counts[names.index(r["task"])]))]
    assert np.isclose(calls.resampled_b_r(recs, counts[None])[0], calls.regressions(drawn)["NR"]["coef"][1])


def test_episode_tally():
    e = figure.episode(["RRR", "-", "WW", "W", "RO"])
    assert (e["nturn"], e["N"], e["W"], e["Wp"], e["R"], e["Rp"]) == (5, 8, 3, 2, 4, 3)
    assert e["turns"][1] == []
