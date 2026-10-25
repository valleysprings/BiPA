"""(d, part 1) Dev and test scores of every version evaluated in the four from-scratch runs, the seed's dev executions, and K / baseline as each optimizer used them.

Counts follow selection.py: an episode is 'verified' if it has requests, outcome_valid and a boolean success; signature = sorted request ids (a cached episode
shared by two evaluation files has the same signature).
Run: python -m aux.three_word_seed.collect_dev
"""
import glob, json
from pathlib import Path
from src import layout
from src.analysis import selection as S

EXE = layout.EXECUTION / "gpt-6-luna"
OUT = layout.OUT / "three_word_seed"
SEED3 = "cf7b932190ed"
RETURNED = {("skillopt", "automationbench"): "bb509d254dad", ("skillopt", "tau3-bench"): "dfd18dc47eb0",
            ("gepa", "automationbench"): "52d485532050", ("gepa", "tau3-bench"): "d1ee2d3a774f"}


def ev(path):
    d = json.load(open(path))
    eps = d["episodes"]
    ver = [e for e in eps if S.verified(e)]
    return dict(panel=len(d["panel"]), episodes=len(eps), verified=len(ver), solved=sum(bool(e["success"]) for e in ver),
                solved_if_unverified_fail=sum(bool(e.get("success")) is True and S.verified(e) for e in eps),
                per_task={e["task_id"]: int(bool(e["success"])) for e in ver},
                signatures={e["task_id"]: list(S.signature(e)) for e in ver})


def main():
    OUT.mkdir(parents=True, exist_ok=True)


    out = {}
    for (opt, bench), ret in RETURNED.items():
        run = EXE / opt / bench / "fromscratch"
        s = json.load(open(run / "summary.json"))
        extra = s["extra"]
        n_dev = s["panels"]["sel"]
        dev = {}
        for d in sorted(glob.glob(str(run / "results/evaluations/dev/*/evaluation.json"))):
            vid = Path(d).parent.name.split("-")[0]
            dev.setdefault(vid, []).append(ev(d))
        rec = dict(run=str(run), n_dev=n_dev, returned=ret, dev_versions=len(dev),
                   dev={v: [{k: x[k] for k in ("panel", "episodes", "verified", "solved")} for x in xs] for v, xs in dev.items()})
        if opt == "skillopt":
            hist = json.load(open(run / "skillopt/history.json"))
            # the baseline of SkillOpt is the starting current_score: before step 1 it is the seed's dev score
            # step-1 record: if action == reject the seed stays current, so current_score after step 1 = baseline; if accept_new_best, baseline < candidate and is not recorded in history.
            rec["history"] = [dict(step=h["step"], selection_hard=h["selection_hard"], action=h["action"], current_score=h["current_score"], best_score=h["best_score"], best_origin=h["best_origin"],
                                   candidate_hash=h["candidate_hash"]) for h in hist]
            rec["K"] = extra["total_accepts"] + extra["total_rejects"]
            rec["candidate_dev_tasks"] = [round(h["selection_hard"] * n_dev) for h in hist]
            # baseline as the optimizer used it: seed's dev evaluation in the run (the selection_eval_baseline dir)
            rec["baseline_note"] = "SkillOpt summary.baseline_selection_hard is null; baseline = the run's own seed dev evaluation (evaluation.json of %s)" % SEED3
        else:
            vals = extra["val_scores"]
            rec["val_scores_tasks"] = [round(v * n_dev) for v in vals]
            rec["K"] = len(vals) - 1
            rec["best_idx"] = extra["best_idx"]
            rec["accepted_flag_in_summary"] = extra["accepted"]
            rec["baseline_tasks_val_scores0"] = round(vals[0] * n_dev)
        seed_dev = dev[SEED3]
        rec["seed_dev_evals"] = len(seed_dev)
        rec["seed_dev_solved"] = [x["solved"] for x in seed_dev]
        rec["returned_dev_solved"] = [x["solved"] for x in dev[ret]]
        rec["test"] = {v: [ev(f)["solved"] / ev(f)["verified"] for f in glob.glob(str(run / f"results/evaluations/test/{v}-*/evaluation.json"))] for v in {SEED3, ret}}
        out[f"{opt}|{bench}"] = rec
        out[f"{opt}|{bench}"]["_seed_dev_per_task"] = seed_dev[0]["per_task"]
        out[f"{opt}|{bench}"]["_seed_dev_signatures"] = seed_dev[0]["signatures"]

    # are the two optimizer runs' seed dev evaluations the same episodes (cache) or independent draws?
    for bench in ("automationbench", "tau3-bench"):
        a, b = out[f"skillopt|{bench}"], out[f"gepa|{bench}"]
        sa, sb = a["_seed_dev_signatures"], b["_seed_dev_signatures"]
        common = sorted(set(sa) & set(sb))
        same = sum(sa[t] == sb[t] for t in common)
        agree = sum(a["_seed_dev_per_task"][t] == b["_seed_dev_per_task"][t] for t in common)
        out[f"seed_dev_overlap|{bench}"] = dict(tasks_verified_in_both=len(common), identical_episode_signatures=same, same_outcome=agree,
                                                  skillopt_run_seed_solved=a["seed_dev_solved"], gepa_run_seed_solved=b["seed_dev_solved"])
    for k in list(out):
        if isinstance(out[k], dict):
            out[k].pop("_seed_dev_signatures", None)
            out[k].pop("_seed_dev_per_task", None)
    json.dump(out, open(OUT / "dev_collect.json", "w"), indent=1)
    for k, v in out.items():
        if k.startswith("seed_dev_overlap"):
            print(k, v); continue
        print(k, "K", v["K"], "seed dev", v["seed_dev_solved"], "returned dev", v["returned_dev_solved"], "n_dev", v["n_dev"], "test", v["test"])
        print("   ", v.get("candidate_dev_tasks") or v.get("val_scores_tasks"), "history" if "history" in v else "")
        if "history" in v:
            for h in v["history"]:
                print("     ", h)
        print("    all dev versions:", {a: [x["solved"] for x in b] for a, b in v["dev"].items()})


if __name__ == "__main__":
    main()
