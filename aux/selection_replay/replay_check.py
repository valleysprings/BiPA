"""Does the selection replay's task model (the Beta prior of selection.py) match the seed's real reruns?

(1) fit: the fitted Beta(a, b) predicts, for a task run three times, the shares solved always / sometimes / never, and the SD of a run's total
    score; compared with the seed's three test runs and its distinct dev draws.
(2) model-free: the seed's real reruns used as identical candidates. On the test panel (3 seed runs) every ordered choice of a baseline run and
    K = 2 candidate runs gives one real best-of-2 gain (strictly above the baseline, else 0); the same on the dev draws. Compared with the model's
    E[G0] for K = 2 on the same panel, with a random baseline.
Run: python -m aux.selection_replay.replay_check
"""
import itertools
import json

import numpy as np

from src.analysis import selection as S


def main():
    rng = np.random.default_rng(7)
    out = {}
    for bench, _ in S.BENCHES:
        dev_files, test_files = S.seed_files(bench)
        dev, test = S.executions(dev_files), S.executions(test_files)
        a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        # (1a) always / sometimes / never over tasks with exactly three test executions
        three = [v for v in test.values() if len(v) == 3]
        obs = [np.mean([sum(v) == 3 for v in three]), np.mean([0 < sum(v) < 3 for v in three]), np.mean([sum(v) == 0 for v in three])]
        p = rng.beta(a, b, size=200_000)
        pred = [np.mean(p ** 3), 1 - np.mean(p ** 3) - np.mean((1 - p) ** 3), np.mean((1 - p) ** 3)]
        # (1b) SD of a run's total score: real (between the seed's runs on the same panel) vs model (posterior per task)
        def run_totals(files, n_panel):
            totals, seen = [], set()
            for path in files:
                eps = json.load(open(path))["episodes"]
                if len(eps) == n_panel and path not in seen:
                    seen.add(path)
                    ok = [e for e in eps if S.verified(e)]
                    totals.append(sum(bool(e["success"]) for e in ok) * n_panel / max(1, len(ok)))
            return totals
        n_test = len(test)
        test_totals = run_totals(test_files, n_test)
        draws = S.dev_draws(dev_files, {'terminal-bench-2-1': 9}.get(bench, 30))
        dev_totals = [d["solved"] * len(dev) / d["verified"] for d in draws]
        post_test = rng.beta(a + np.array([sum(v) for v in test.values()]), b + np.array([len(v) - sum(v) for v in test.values()]), size=(50_000, n_test))
        sd_model_test = float(np.sqrt((post_test * (1 - post_test)).sum(axis=1)).mean())
        post_dev = rng.beta(a + np.array([sum(v) for v in dev.values()]), b + np.array([len(v) - sum(v) for v in dev.values()]), size=(50_000, len(dev)))
        sd_model_dev = float(np.sqrt((post_dev * (1 - post_dev)).sum(axis=1)).mean())
        # (2) real best-of-2 over the seed's reruns vs the model's E[G0] for K = 2
        def real_best_of_2(totals):
            gains = []
            for base in range(len(totals)):
                others = [t for i, t in enumerate(totals) if i != base]
                for pair in itertools.combinations(others, 2):
                    gains.append(max(0.0, max(pair) - totals[base]))
            return gains
        def model_best_of_2(post):
            base = (rng.random(post.shape) < post).sum(axis=1)
            c1 = (rng.random(post.shape) < post).sum(axis=1)
            c2 = (rng.random(post.shape) < post).sum(axis=1)
            return float(np.maximum(np.maximum(c1, c2) - base, 0).mean())
        out[bench] = {"prior": (round(a, 3), round(b, 3)),
                      "always_sometimes_never_obs": [round(100 * x, 1) for x in obs], "pred": [round(100 * x, 1) for x in pred], "n3": len(three),
                      "test_totals": test_totals, "dev_totals": dev_totals,
                      "sd_obs_test": float(np.std(test_totals, ddof=1)) if len(test_totals) > 1 else None, "sd_model_test": sd_model_test,
                      "sd_obs_dev": float(np.std(dev_totals, ddof=1)) if len(dev_totals) > 1 else None, "sd_model_dev": sd_model_dev,
                      "best2_real_test": float(np.mean(real_best_of_2(test_totals))) if len(test_totals) >= 3 else None,
                      "best2_model_test": model_best_of_2(post_test),
                      "best2_real_dev": float(np.mean(real_best_of_2(dev_totals))) if len(dev_totals) >= 3 else None,
                      "best2_model_dev": model_best_of_2(post_dev)}
        print(bench, json.dumps(out[bench]))


if __name__ == "__main__":
    main()
