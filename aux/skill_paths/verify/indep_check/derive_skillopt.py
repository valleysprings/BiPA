import sys, json, os
from pathlib import Path
from aux.skill_paths.verify.indep_check.vlib import *


def derive(run):
    run = Path(run)
    sd = run / "skillopt"
    hist = json.load(open(sd / "history.json"))
    seed = rd(sd / "seed.md")
    sv = vid(seed)[:12]
    out = {"run": str(run), "seed_vid": sv, "seed_chars": len(seed), "hist_len": len(hist)}
    # step dirs, skills
    out["step_dirs"] = sorted(x for x in os.listdir(sd / "steps")) if (sd / "steps").is_dir() else []
    out["skills"] = sorted(os.listdir(sd / "skills")) if (sd / "skills").is_dir() else []
    seed_dev = [summarize_eval(p / "evaluation.json") for _, p in eval_dirs(run, "dev", sv)]
    out["seed_dev_evals"] = [{k: s[k] for k in ("path", "n", "n_success", "n_unknown", "mean_tok", "mean_tok_known", "panel_len")} for s in seed_dev]
    steps = []
    current_text, current_vid = seed, sv
    current_name = "seed"
    for h in hist:
        n = h["step"]
        cpath = sd / f"steps/step_{n:04d}/candidate_skill.md"
        ctext = rd(cpath) if cpath.exists() else None
        cv = vid(ctext)[:12] if ctext is not None else None
        devs = [summarize_eval(p / "evaluation.json") for _, p in eval_dirs(run, "dev", cv)] if cv else []
        rec = {"step": n, "action": h["action"], "kept": h["action"] != "reject",
               "cand_vid": cv, "cand_chars": len(ctext) if ctext else None, "cand_words": words(ctext) if ctext else None,
               "parent_vid": current_vid, "parent_name": current_name,
               "hist_selection_hard": h.get("selection_hard"), "hist_candidate_gate_score": h.get("candidate_gate_score"),
               "hist_current_score_after": h.get("current_score"), "hist_skill_len": h.get("skill_len"),
               "hist_candidate_skill_len": h.get("candidate_skill_len"),
               "dev_evals": [{k: s[k] for k in ("path", "n", "n_success", "n_false", "n_unknown", "mean_tok", "mean_tok_known", "panel_len", "mtime")} for s in devs],
               "added": line_diff(current_text, ctext)[0] if ctext else None,
               "removed": line_diff(current_text, ctext)[1] if ctext else None,
               "same_text_as_parent": ctext == current_text}
        steps.append(rec)
        if rec["kept"]:
            current_text, current_vid, current_name = ctext, cv, f"step_{n:04d}"
    out["steps"] = steps
    best = rd(sd / "best_skill.md")
    out["best_vid"] = vid(best)[:12]
    out["best_chars"] = len(best)
    rs = json.load(open(sd / "runtime_state.json"))
    out["runtime_state"] = {k: v for k, v in rs.items() if "path" not in k}
    summ = json.load(open(run / "summary.json"))
    out["summary_extra"] = summ.get("extra")
    fs = summ.get("final_skill")
    out["summary_final_skill_vid"] = vid(fs)[:12] if fs else None
    out["summary_final_skill_eq_best"] = (fs == best)
    # test evals for the returned skill
    out["test_evals_returned"] = [{k: s[k] for k in ("path", "n", "n_success", "n_false", "n_unknown", "mean_tok", "panel_len")}
                                  for s in [summarize_eval(p / "evaluation.json") for _, p in eval_dirs(run, "test", out["best_vid"])]]
    # all dev evals in the run (vid -> list)
    alldev = {}
    for d in sorted(os.listdir(run / "results/evaluations/dev")):
        alldev[d] = None
    out["all_dev_dirs"] = sorted(alldev)
    return out


if __name__ == "__main__":
    r = derive(sys.argv[1])
    print(json.dumps(r, indent=1, default=str))
