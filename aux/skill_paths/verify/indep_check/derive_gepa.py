import sys
import json, os, re
from pathlib import Path
from aux.skill_paths.verify.indep_check.vlib import *

KEEP = ("path", "n", "n_success", "n_false", "n_unknown", "mean_tok", "panel_len", "mtime")


def parse_log(txt):
    """{iteration(1-based): dict(parent=int, proposed=text|None, outcome=str, line_idx)} from run_log.txt"""
    lines = txt.split("\n")
    its = {}
    cur = None
    for idx, ln in enumerate(lines):
        m = re.match(r"^Iteration (\d+): (.*)$", ln)
        if m:
            n = int(m.group(1)); rest = m.group(2)
            it = its.setdefault(n, {"proposed": None, "events": []})
            if rest.startswith("Proposed new text for current_candidate: "):
                it["proposed_start"] = idx
                it["proposed_first"] = rest[len("Proposed new text for current_candidate: "):]
            else:
                it["events"].append((idx, rest))
    # texts: from proposed_start up to the next 'Iteration n:' line of the same iteration
    for n, it in its.items():
        if "proposed_start" in it:
            start = it["proposed_start"]
            nxt = [i for i, _ in it["events"] if i > start]
            # also any iteration line after start that belongs to any iteration
            end = None
            for j in range(start + 1, len(lines)):
                if re.match(r"^Iteration \d+: ", lines[j]):
                    end = j; break
            body = [it["proposed_first"]] + lines[start + 1:end]
            it["proposed"] = "\n".join(body)
    return its


def derive(run):
    run = Path(run)
    gd = run / "gepa"
    cands = json.load(open(gd / "candidates.json"))
    rl = json.load(open(gd / "run_log.json"))
    txt = rd(gd / "run_log.txt")
    log = parse_log(txt)
    summ = json.load(open(run / "summary.json"))
    ex = summ["extra"]
    out = {"n_pool": len(cands), "n_iter": len(rl), "val_scores": ex["val_scores"], "parents_summary": ex["parents"], "best_idx": ex["best_idx"]}
    pool = []
    for k, c in enumerate(cands):
        t = c["current_candidate"]; v = vid(t)[:12]
        devs = [summarize_eval(p / "evaluation.json") for _, p in eval_dirs(run, "dev", v)]
        pool.append({"k": k, "vid": v, "chars": len(t), "words": words(t), "text": t,
                     "dev": [{kk: d[kk] for kk in KEEP} for d in devs]})
    out["pool"] = [{k: v for k, v in p.items() if k != "text"} for p in pool]
    # iterations
    its = []
    for e in rl:
        t = e["i"] + 1
        rec = {"t": t, "parent": e["selected_program_candidate"], "has_scores": "subsample_scores" in e,
               "parent_sum": sum(e["subsample_scores"]) if "subsample_scores" in e else None,
               "child_sum": sum(e["new_subsample_scores"]) if "new_subsample_scores" in e else None,
               "new_idx": e.get("new_program_idx"), "subsample_ids": e["subsample_ids"],
               "parent_scores": e.get("subsample_scores"), "child_scores": e.get("new_subsample_scores")}
        lg = log.get(t, {})
        rec["log_events"] = [r for _, r in lg.get("events", [])
                             if re.match(r"(New subsample|All subsample|Valset score|Reflective|New program candidate|Selected program)", r)]
        if rec["new_idx"] is None and rec["has_scores"]:
            ptxt = lg.get("proposed")
            rec["rej_text_found"] = ptxt is not None
            if ptxt is not None:
                # candidates for whitespace variants
                variants = {"raw": ptxt, "rstrip": ptxt.rstrip(), "strip": ptxt.strip(), "rstrip+nl": ptxt.rstrip() + "\n"}
                found = None
                for name, vt in variants.items():
                    v = vid(vt)[:12]
                    tr = eval_dirs(run, "train", v)
                    if tr:
                        found = (name, v, vt, tr); break
                if found:
                    name, v, vt, tr = found
                    rec["rej_variant"] = name; rec["rej_vid"] = v; rec["rej_chars"] = len(vt); rec["rej_words"] = words(vt); rec["rej_text"] = vt
                    rec["rej_train"] = [(panel, summarize_eval(p / "evaluation.json")) for panel, p in tr]
                    rec["rej_train"] = [(pn, {kk: d[kk] for kk in ("n", "n_success", "n_unknown", "mean_tok", "ep_tasks")}) for pn, d in rec["rej_train"]]
                else:
                    rec["rej_vid"] = None; rec["rej_variants_vids"] = {n: vid(v)[:12] for n, v in variants.items()}
                    rec["rej_chars_raw"] = len(ptxt)
        its.append(rec)
    out["its"] = its
    out["_pool_text"] = [p["text"] for p in pool]
    out["_rej_text"] = {r["t"]: r.get("rej_text") for r in its if "rej_text" in r}
    # accepted children train eval on B_t: child vid with train panels
    for r in its:
        if r["new_idx"] is not None:
            v = pool[r["new_idx"]]["vid"]
            tr = eval_dirs(run, "train", v)
            r["acc_train"] = [(pn, {kk: summarize_eval(p / "evaluation.json")[kk] for kk in ("n", "n_success", "n_unknown", "mean_tok")}) for pn, p in tr]
        # parent train eval on same panel
    # returned
    best = pool[ex["best_idx"]]
    fs = summ.get("final_skill")
    out["returned_vid"] = best["vid"]; out["returned_chars"] = best["chars"]
    out["summary_final_skill_eq_best_candidate"] = (fs == best["text"])
    out["test_evals_all"] = {}
    tdir = run / "results/evaluations/test"
    if tdir.is_dir():
        for d in sorted(os.listdir(tdir)):
            s = summarize_eval(tdir / d / "evaluation.json")
            out["test_evals_all"][d] = {kk: s[kk] for kk in ("n", "n_success", "n_false", "n_unknown", "mean_tok")}
    out["summary_test"] = summ.get("test")
    # all dev dirs
    out["all_dev_dirs"] = sorted(os.listdir(run / "results/evaluations/dev"))
    out["pool_vids"] = [p["vid"] for p in pool]
    return out


if __name__ == "__main__":
    r = derive(sys.argv[1])
    r.pop("_pool_text"); r.pop("_rej_text")
    for it in r["its"]:
        it.pop("rej_text", None)
    print(json.dumps(r, indent=1, default=str))
