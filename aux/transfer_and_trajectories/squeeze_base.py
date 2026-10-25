"""Base rate for the white-squeeze statistic: no-gain share (dQ <= 0) among tasks a skill pushed to more turns (dR > 0)
versus among tasks it did not (dR <= 0), and the share of tasks with dQ == 0 (always / never solved on both sides)."""
import numpy as np

from src.analysis import dynamics as D


def main():
    for model in ("deepseek-flash", "gpt-6-luna"):
        for bench in ("tau3-bench", "automationbench"):
            seed = D.load_cell(model, "native-seed", bench)
            for arm in ("gepa", "skillopt", "skilladam"):
                own = D.load_cell(model, arm, bench)
                tasks = [t for t in set.intersection(*(set(own[k]) for k in range(3)), *(set(seed[k]) for k in range(3)))
                         if all(own[k][t]["success"] is not None and seed[k][t]["success"] is not None for k in range(3))]
                dr = np.array([np.mean([own[k][t]["R"] for k in range(3)]) - np.mean([seed[k][t]["R"] for k in range(3)]) for t in tasks])
                dq = np.array([np.mean([float(own[k][t]["success"]) for k in range(3)]) - np.mean([float(seed[k][t]["success"]) for k in range(3)]) for t in tasks])
                up = dr > 0
                print(f"{model[:8]} {bench[:5]} {arm:9s} n={len(tasks)} nogain|up={np.mean(dq[up] <= 0):.2f} nogain|notup={np.mean(dq[~up] <= 0):.2f} "
                      f"loss|up={np.mean(dq[up] < 0):.2f} loss|notup={np.mean(dq[~up] < 0):.2f} gain|up={np.mean(dq[up] > 0):.2f} gain|notup={np.mean(dq[~up] > 0):.2f} tie_all={np.mean(dq == 0):.2f}")


if __name__ == "__main__":
    main()
