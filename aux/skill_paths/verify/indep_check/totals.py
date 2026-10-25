import sys
import json, os
from aux.skill_paths.verify.indep_check.vlib import *
from aux.skill_paths.verify.indep_check import derive_skilladam


if __name__ == "__main__":
    TAG = "native-20261008-https"
    out = {}
    for bench in ("tau3-bench", "automationbench", "terminal-bench-2-1"):
        # skillopt
        h = json.load(open(EX / f"deepseek-flash/skillopt/{bench}/{TAG}/skillopt/history.json"))
        acc = sum(1 for s in h if s["action"] != "reject"); rej = sum(1 for s in h if s["action"] == "reject")
        acts = sorted({s["action"] for s in h})
        # gepa
        c = json.load(open(EX / f"deepseek-flash/gepa/{bench}/{TAG}/gepa/candidates.json"))
        sm = json.load(open(EX / f"deepseek-flash/gepa/{bench}/{TAG}/summary.json"))["extra"]
        out[(bench, "skillopt")] = (acc, rej, acts)
        out[(bench, "gepa")] = (len(c) - 1, 0, sm["accepted"], len(sm["val_scores"]) - 1)
        print(bench, "SkillOpt accepted/rejected", acc, rej, acts, "| GEPA pool-1", len(c) - 1, "summary accepted", sm["accepted"], "val_scores-1", len(sm["val_scores"]) - 1)
        # skilladam
        run = EX / f"deepseek-flash/skilladam/{bench}/{TAG}"
        D = derive_skilladam.derive(run)
        steps = D["steps"]
        scored = [s for s in steps if s["patch_mode"] != "failed" and os.path.exists(run / f"skilladam/iterations/iter_{s['iter']:03d}/validation_candidate_evaluation.json")]
        differs = [s for s in scored if s["cand_vid"] != s["incumbent_vid"] and s["cand_vid"] is not None]
        ambiguous = [s["iter"] for s in steps if len(s["others_on_panel"]) > 1]
        nacc = sum(1 for s in differs if s["accepted"]); nrej = len(differs) - nacc
        print(bench, "SkillAdam iterations", len(steps), "dev-scored", len(scored), "text differs from incumbent", len(differs), "accepted", nacc, "rejected", nrej, "ambiguous panels", ambiguous,
              "| accepted total (checkpoint)", sum(1 for s in steps if s["accepted"]), "patch modes", {m: sum(1 for s in steps if s["patch_mode"] == m) for m in ("strict", "empty", "failed")},
              "better/worse/tied", sum(s["candidate_n_success"] > s["baseline_n_success"] for s in steps), sum(s["candidate_n_success"] < s["baseline_n_success"] for s in steps), sum(s["candidate_n_success"] == s["baseline_n_success"] for s in steps),
              "patch attempts", sum(s["patch_attempts"] for s in steps))
        out[(bench, "skilladam")] = (nacc, nrej)
    tot_acc = sum(out[(b, "skillopt")][0] for b in ("tau3-bench", "automationbench", "terminal-bench-2-1")) + sum(out[(b, "gepa")][0] for b in ("tau3-bench", "automationbench", "terminal-bench-2-1")) + sum(out[(b, "skilladam")][0] for b in ("tau3-bench", "automationbench"))
    tot_rej = sum(out[(b, "skillopt")][1] for b in ("tau3-bench", "automationbench", "terminal-bench-2-1")) + sum(out[(b, "skilladam")][1] for b in ("tau3-bench", "automationbench"))
    print("TOTAL accepted", tot_acc, "rejected", tot_rej, "sum", tot_acc + tot_rej)
