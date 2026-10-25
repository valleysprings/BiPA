import sys
import json, os, re, glob
from pathlib import Path
from aux.skill_paths.verify.indep_check.vlib import *

KEEP = ("path", "n", "n_success", "n_false", "n_unknown", "mean_tok", "panel_len", "mtime")


def delivered_texts(run):
    """{vid12: text} from results/skills/deliveries/evaluations/<split>/<vid12>-<panel>/native/SKILL.md (the exact text delivered); guidance.md as fallback"""
    out = {}
    base = Path(run) / "results/skills/deliveries/evaluations"
    if not base.is_dir():
        return out
    for split in sorted(os.listdir(base)):
        for d in sorted(os.listdir(base / split)):
            v = d.split("-")[0]
            if v in out: continue
            nat = base / split / d / "native/SKILL.md"
            g = base / split / d / "guidance.md"
            if nat.exists():
                out[v] = rd(nat)
            elif g.exists():
                raw = rd(g)
                m = re.match(r"<bach-skill [^>]*>\n(.*)\n</bach-skill>\s*$", raw, re.S)
                out[v] = m.group(1) if m else raw
    return out


def derive(run):
    run = Path(run)
    sd = run / "skilladam"
    ck = json.load(open(sd / "checkpoint.json"))
    its = ck["iterations"]
    seed = rd(sd / "seed.md")
    sv = vid(seed)[:12]
    texts = delivered_texts(run)
    texts.setdefault(sv, seed)
    # vid check of delivered texts
    bad = [v for v, t in texts.items() if vid(t)[:12] != v]
    out = {"seed_vid": sv, "seed_chars": len(seed), "n_iter": len(its), "next_iteration": ck.get("next_iteration"),
           "delivered_text_vid_mismatch": bad, "n_delivered_texts": len(texts)}
    # dev eval index: panel taskset -> {vid12: (panel12, path)}
    ddir = run / "results/evaluations/dev"
    panels = {}
    for d in sorted(os.listdir(ddir)):
        v, pn = d.split("-", 1)
        s = summarize_eval(ddir / d / "evaluation.json")
        ts = frozenset(s["panel_tasks"])
        panels.setdefault(ts, {})[v] = (pn, ddir / d / "evaluation.json", s)
    out["n_dev_panels"] = len(panels)
    current_v = sv
    cur_text = seed
    steps = []
    for it in its:
        k = it["iteration"]
        vids = frozenset(it["validation_case_ids"])
        on = panels.get(vids, {})
        idir = sd / "iterations" / f"iter_{k:03d}"
        attempts = []
        for pa in sorted(glob.glob(str(idir / "patch_attempt_??.json"))):
            pj = json.load(open(pa))
            attempts.append({"file": os.path.basename(pa), "patch_head": pj["patch"][:30] if isinstance(pj["patch"], str) else None,
                             "unparseable": pj["patch"] == "(unparseable)", "empty": (pj["patch"] or "").strip() == ""})
        rec_b = {t: m["hard"] for t, m in it["baseline"]["case_metrics"].items()}
        rec_c = {t: m["hard"] for t, m in it["candidate"]["case_metrics"].items()}
        def vec(v):
            s = on[v][2]
            return dict(zip(s["ep_tasks"], [float(bool(x)) for x in s["ep_success"]]))
        base_match = [v for v in on if vec(v) == rec_b]
        cand_match = [v for v in on if vec(v) == rec_c]
        rec = {"iter": k, "accepted": bool(it["accepted"]), "patch_mode": it["patch_mode"], "patch_attempts": it["patch_attempts"],
               "edit_budget": it["edit_budget"], "reason": it["reason"],
               "baseline_primary": it["baseline"]["primary"], "candidate_primary": it["candidate"]["primary"],
               "baseline_n_success": sum(rec_b.values()), "candidate_n_success": sum(rec_c.values()), "n_val": len(rec_c),
               "incumbent_vid": current_v, "vids_on_panel": sorted(on), "panel12": sorted({on[v][0] for v in on}),
               "base_match": sorted(base_match), "cand_match": sorted(cand_match),
               "attempts": attempts, "n_train": len(it["training_case_ids"])}
        # candidate identification
        others = [v for v in on if v != current_v]
        rec["others_on_panel"] = sorted(others)
        cand_v = None
        if len(others) == 1:
            cand_v = others[0]
        elif len(others) == 0:
            cand_v = current_v  # identical text to incumbent
        else:
            cand_v = None  # ambiguous
        rec["cand_vid"] = cand_v
        rec["cand_identical_to_incumbent"] = (cand_v == current_v)
        if cand_v is not None and cand_v in on:
            s = on[cand_v][2]
            rec["cand_eval"] = {kk: s[kk] for kk in ("n", "n_success", "n_false", "n_unknown", "mean_tok")}
            rec["cand_eval_panel"] = on[cand_v][0]
        if current_v in on:
            s = on[current_v][2]
            rec["base_eval"] = {kk: s[kk] for kk in ("n", "n_success", "n_false", "n_unknown", "mean_tok")}
        # text/diff
        ctext = texts.get(cand_v) if cand_v else None
        rec["cand_text_found"] = ctext is not None
        if ctext is not None:
            rec["cand_chars"] = len(ctext); rec["cand_words"] = words(ctext)
            a, r = line_diff(cur_text, ctext)
            rec["added"], rec["removed"] = a, r
        steps.append(rec)
        if it["accepted"] and cand_v:
            current_v = cand_v; cur_text = texts.get(cand_v, cur_text)
    out["steps"] = steps
    fs = rd(sd / "final_skill.md")
    out["final_vid"] = vid(fs)[:12]; out["final_chars"] = len(fs)
    out["current_skill_eq_final"] = (rd(sd / "current_skill.md") == fs)
    out["checkpoint_current_skill_eq_final"] = (ck["current_skill"] == fs)
    summ = json.load(open(run / "summary.json"))
    out["summary_final_skill_eq_final"] = (summ.get("final_skill") == fs)
    out["summary_extra"] = summ.get("extra")
    tdir = run / "results/evaluations/test"
    out["test_evals"] = {}
    if tdir.is_dir():
        for d in sorted(os.listdir(tdir)):
            s = summarize_eval(tdir / d / "evaluation.json")
            out["test_evals"][d] = {kk: s[kk] for kk in ("n", "n_success", "n_false", "n_unknown", "mean_tok")}
    out["summary_test"] = summ.get("test")
    # seed dev evals (all panels)
    out["seed_dev_evals"] = [{"panel": pn, **{kk: summarize_eval(p / "evaluation.json")[kk] for kk in ("n", "n_success", "n_unknown", "mean_tok")}}
                             for pn, p in eval_dirs(run, "dev", sv)]
    out["_texts"] = texts
    return out


if __name__ == "__main__":
    r = derive(sys.argv[1])
    r.pop("_texts")
    for s in r["steps"]:
        s.pop("added", None); s.pop("removed", None)
    print(json.dumps(r, indent=1, default=str))
