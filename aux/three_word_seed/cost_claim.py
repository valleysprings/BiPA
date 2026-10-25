"""(c, extra) Where do '-5% (SkillOpt) / +45% (GEPA)' on tau3 come from?  They equal the ratio of summary.json test.final.cost / test.seed.cost.
summary 'cost' = mean over the episodes whose capture is complete (dataloader.task_row: invocations_complete and requests_complete) of c = N + X/1000 (a composite of tool calls and
input tokens in thousands), taken over DIFFERENT episode subsets for the seed and for the skill (61/75 vs 56/75 for SkillOpt, 73/75 vs 71/75 for GEPA).
Below: reproduce it, and compute the matched-task versions.
Run: python -m aux.three_word_seed.cost_claim"""
import glob, json, math
import numpy as np
from src.utils import dataloader as C
from src import layout

EXE = layout.EXECUTION / "gpt-6-luna"
OUT = layout.OUT / "three_word_seed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for opt, ret in (("skillopt", "dfd18dc47eb0"), ("gepa", "d1ee2d3a774f")):
        run = EXE / opt / "tau3-bench/fromscratch"
        s = json.load(open(run / "summary.json"))["evaluated_panels"]["test"]
        f = {v: glob.glob(str(run / f"results/evaluations/test/{v}-*/evaluation.json"))[0] for v in ("cf7b932190ed", ret)}
        rows = {v: {e["task_id"]: C.task_row(e) for e in json.load(open(p))["episodes"]} for v, p in f.items()}
        cs = {v: [r["c"] for r in rows[v].values() if r["c"] is not None] for v in rows}
        mean_c = {v: float(np.mean(cs[v])) for v in cs}
        unmatched_ratio = mean_c[ret] / mean_c["cf7b932190ed"]
        # matched tasks where both have c (capture complete) -> ratio of means and geometric mean of ratios, for c and for X alone
        common = [t for t in rows[ret] if rows[ret][t]["c"] is not None and rows["cf7b932190ed"].get(t, {}).get("c") is not None]
        def gm(k): return float(math.exp(np.mean([math.log(rows[ret][t][k] / rows["cf7b932190ed"][t][k]) for t in common if rows[ret][t][k] and rows["cf7b932190ed"][t][k]])))
        res[opt] = dict(summary_cost_seed=s["seed"]["cost"], summary_cost_final=s["final"]["cost"], summary_ratio=s["final"]["cost"] / s["seed"]["cost"],
                        reproduced_mean_c=mean_c, episodes_with_c={v: len(cs[v]) for v in cs}, unmatched_ratio_of_means_c=unmatched_ratio,
                        matched_tasks_capture_complete_both=len(common), matched_ratio_of_means_c=float(np.mean([rows[ret][t]["c"] for t in common]) / np.mean([rows["cf7b932190ed"][t]["c"] for t in common])),
                        matched_geomean_c=gm("c"), matched_geomean_X=gm("X"), matched_geomean_N=gm("N"))
        print(opt, json.dumps(res[opt], indent=1))
    json.dump(res, open(OUT / "cost_claim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
