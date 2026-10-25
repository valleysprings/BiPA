"""(e, extra) Alternative operationalisations of the 'writes sharing a turn' share and of read-back, to see which one (if any) gives the quoted 62% / 60%.
Imports fingerprints.py (data loader).
Run: python -m aux.three_word_seed.fingerprints_variants"""
import json, numpy as np
from aux.three_word_seed import fingerprints as FP

OUT = FP.OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, tag in FP.AGENTS.items():
        reps = FP.load(tag)
        d = {}
        for rep, per in reps.items():
            d[f"rep{rep}_Wpar_all"] = FP.wpar({rep: per})["share"]
            d[f"rep{rep}_Wpar_complete"] = FP.wpar({rep: per}, only_complete=True)["share"]
        # per-episode mean of the share (episodes with >= 1 write), and macro over tasks
        ep_sh, task_sh = [], {}
        wt = mw = 0   # turns with >= 1 write, of which with >= 2
        for rep, per in reps.items():
            for t, ep in per.items():
                W = sum(sum(c == "W" for c in turn) for turn in ep["turns"])
                Wp = sum(sum(c == "W" for c in turn) for turn in ep["turns"] if sum(c == "W" for c in turn) >= 2)
                if W: ep_sh.append(Wp / W); task_sh.setdefault(t, []).append((Wp, W))
                for turn in ep["turns"]:
                    w = sum(c == "W" for c in turn); wt += w >= 1; mw += w >= 2
        d["mean_of_per_episode_share"] = float(np.mean(ep_sh))
        d["macro_over_tasks_share"] = float(np.mean([sum(a for a, _ in v) / sum(b for _, b in v) for v in task_sh.values()]))
        d["share_of_write_turns_with_2plus_writes"] = mw / wt
        # episodes with at least one multi-write turn
        d["share_eps_with_a_multi_write_turn"] = float(np.mean([any(sum(c == "W" for c in turn) >= 2 for turn in ep["turns"]) for per in reps.values() for ep in per.values() if any("W" in turn for turn in ep["turns"])]))
        # read-back variants on rep level
        for rep, per in reps.items():
            rb = FP.readback({rep: per})
            d[f"rep{rep}_last_write_read_any"] = rb["share_eps_last_write_followed_by_read_any"]
            d[f"rep{rep}_writes_read_same_service"] = rb["share_writes_read_back_same_service"]
        out[name] = d
        print(name); [print("   ", k, round(v, 4)) for k, v in d.items()]
    json.dump(out, open(OUT / "fingerprints_variants.json", "w"), indent=1)


if __name__ == "__main__":
    main()
