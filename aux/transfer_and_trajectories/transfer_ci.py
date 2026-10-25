"""Luna tau3: success difference of each skill to the Luna seed, and of the pooled transferred and native skills, with bootstrap intervals.

Run:  python -m aux.transfer_and_trajectories.transfer_ci
"""
import random

from aux.headline_numbers import transfer as L


def ci(m, ref):
    tasks = [t for t in ref if t in m]
    d = [m[t]["S"] - ref[t]["S"] for t in tasks]
    boots = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
    return 100 * sum(d) / len(d), 100 * boots[125], 100 * boots[4874], len(tasks)


def main():
    random.seed(0)
    cells = {(a, t): L.cell_data(a, t, "tau3-bench") for a, t in L.ARMS}
    seed = L.task_means(cells[("native-seed", "native-20261011")]["reps"])
    for (a, t), c in cells.items():
        if (a, t) == ("native-seed", "native-20261011"):
            continue
        m = L.task_means(c["reps"])
        print(f"{a}@{t}: dQ vs seed = {ci(m, seed)[0]:+.1f} [{ci(m, seed)[1]:+.1f}, {ci(m, seed)[2]:+.1f}] n={ci(m, seed)[3]}")

    def pool(keys):
        """Three transferred or three Luna-optimized skills: per task mean over the skills."""
        ms = [L.task_means(cells[k]["reps"]) for k in keys]
        return {t: {"S": sum(m[t]["S"] for m in ms) / len(ms)} for t in seed if all(t in m for m in ms)}
    tr = pool([(a, "ood-from-deepseek-flash") for a in ("skillopt", "gepa", "skilladam")])
    nat = pool([(a, "native-20261012") for a in ("skillopt", "gepa", "skilladam")])
    print("pooled transferred vs seed", ci(tr, seed))
    print("pooled native vs seed", ci(nat, seed))
    print("pooled transferred vs native", ci(tr, nat))


if __name__ == "__main__":
    main()
