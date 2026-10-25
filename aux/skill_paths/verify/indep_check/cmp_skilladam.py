import sys
import json, os
from aux.skill_paths.verify.indep_check.vlib import *
from aux.skill_paths.verify.indep_check import derive_skilladam

def relclose(a, b, tol=0.01):
    if a is None or b is None: return a is b
    return abs(a - b) <= tol * max(abs(b), 1e-9)

def train_eval_for(run, inc_vid, train_ids):
    ts = frozenset(train_ids)
    hits = []
    for pn, p in eval_dirs(run, "train", inc_vid):
        s = summarize_eval(p / "evaluation.json")
        if frozenset(s["panel_tasks"]) == ts:
            hits.append((pn, s))
    return hits

def compare(run, jfile):
    run = Path(run)
    D = derive_skilladam.derive(run)
    J = json.load(open(PATHS / jfile))
    ck = json.load(open(run / "skilladam/checkpoint.json"))["iterations"]
    texts = D["_texts"]
    disc = []; notes = []
    def chk(name, mine, theirs):
        if mine != theirs: disc.append(f"{name}: mine={mine!r} json={theirs!r}")
    jst = J["steps"]
    chk("n_steps incl seed", D["n_iter"] + 1, len(jst))
    # seed
    s0 = jst[0]
    first = D["steps"][0]
    se = first["base_eval"]
    chk("seed.dev (step-1 draw)", (se["n"], se["n_success"], se["n_unknown"]), (s0["dev"]["n"], s0["dev"]["n_success"], s0["dev"]["n_unknown"]))
    if not relclose(s0["dev"]["mean_input_tokens_per_episode"], se["mean_tok"]): disc.append(f"seed.tok mine={se['mean_tok']} json={s0['dev']['mean_input_tokens_per_episode']}")
    chk("top seed_dev", (J["seed_dev"]["n"], J["seed_dev"]["n_success"]), (se["n"], se["n_success"]))
    chk("seed.skill_chars", D["seed_chars"], s0["skill_chars"])
    # name of incumbents
    inc_name = {D["seed_vid"]: "seed"}
    inc_text = texts[D["seed_vid"]]
    inc_vid = D["seed_vid"]
    n_dev_scored = 0; n_dev_scored_differs = 0; accepted_steps = []
    for st, j in zip(D["steps"], jst[1:]):
        k = st["iter"]; step = k + 1; pfx = f"step{step}"
        chk(pfx + ".step", step, j["step"])
        chk(pfx + ".candidate_id", "step%02d" % step, j["candidate_id"])
        chk(pfx + ".parent_id", inc_name[st["incumbent_vid"]], j["parent_id"])
        chk(pfx + ".kept", st["accepted"], j["kept"])
        failed = (st["patch_mode"] == "failed")
        # check against whether validation_candidate_evaluation.json exists
        has_cand_file = os.path.exists(run / f"skilladam/iterations/iter_{k:03d}/validation_candidate_evaluation.json")
        chk(pfx + ".failed-iff-no-candidate-eval-file", failed, not has_cand_file)
        if failed:
            chk(pfx + ".dev null (failed patch)", None, j["dev"])
        else:
            ce = st["cand_eval"]; n_dev_scored += 1
            if st["cand_vid"] != st["incumbent_vid"]: n_dev_scored_differs += 1
            if j["dev"] is None: disc.append(f"{pfx}: json dev null but candidate was scored (mine {ce})")
            else:
                chk(pfx + ".dev", (ce["n"], ce["n_success"], ce["n_unknown"]), (j["dev"]["n"], j["dev"]["n_success"], j["dev"]["n_unknown"]))
                if not relclose(j["dev"]["mean_input_tokens_per_episode"], ce["mean_tok"]): disc.append(f"{pfx}.tok mine={ce['mean_tok']} json={j['dev']['mean_input_tokens_per_episode']}")
            chk(pfx + ".dev == recorded candidate", (st["candidate_n_success"], st["n_val"]), (ce["n_success"], ce["n"]))
        # gate
        g = j["gate"]["detail"]
        chk(pfx + ".gate.accepted", st["accepted"], g["accepted"])
        chk(pfx + ".gate.baseline", (st["baseline_n_success"], st["n_val"]), (g["baseline"]["n_success"], g["baseline"]["n"]))
        chk(pfx + ".gate.candidate", (st["candidate_n_success"], st["n_val"]), (g["candidate"]["n_success"], g["candidate"]["n"]))
        chk(pfx + ".gate.gain_tasks", st["candidate_n_success"] - st["baseline_n_success"], g["gain_tasks"])
        chk(pfx + ".gate.patch_mode", st["patch_mode"], g["patch_mode"]); chk(pfx + ".gate.patch_attempts", st["patch_attempts"], g["patch_attempts"])
        chk(pfx + ".gate.edit_budget", st["edit_budget"], g["edit_budget"]); chk(pfx + ".gate.reason", st["reason"], g["reason"])
        chk(pfx + ".gate.validation_case_ids", ck[k]["validation_case_ids"], g["validation_case_ids"])
        if not failed and g["baseline"].get("dev"):
            be = st["base_eval"]
            chk(pfx + ".gate.baseline.dev", (be["n"], be["n_success"], be["n_unknown"]), (g["baseline"]["dev"]["n"], g["baseline"]["dev"]["n_success"], g["baseline"]["dev"]["n_unknown"]))
            if not relclose(g["baseline"]["dev"]["mean_input_tokens_per_episode"], be["mean_tok"]): disc.append(f"{pfx}.baseline.tok mine={be['mean_tok']} json={g['baseline']['dev']['mean_input_tokens_per_episode']}")
        # train
        hits = train_eval_for(run, st["incumbent_vid"], ck[k]["training_case_ids"])
        if len(hits) < 1: disc.append(f"{pfx}: no train eval for incumbent on training ids")
        else:
            pn, ts = hits[0]
            if len({(h[1]['n'], h[1]['n_success'], h[1]['n_unknown']) for h in hits}) > 1: disc.append(f"{pfx}: train eval multiple differing {len(hits)}")
            tr = j["train"]
            chk(pfx + ".train", (ts["n"], ts["n_success"], ts["n_unknown"]), (tr["n"], tr["n_success"], tr["n_unknown"]))
            # cross-check with iteration training_evaluation.json
            te = json.load(open(run / f"skilladam/iterations/iter_{k:03d}/training_evaluation.json"))
            cm = te["metrics"]["overall"]["case_metrics"]
            chk(pfx + ".train vs iteration training_evaluation.json", ts["n_success"], int(sum(m["hard"] for m in cm.values())))
        # text/diff/chars
        if failed:
            ctext = inc_text  # no candidate: the extractor carries parent's text
            chk(pfx + ".skill_chars (failed: parent's)", len(ctext), j["skill_chars"])
        else:
            ctext = texts[st["cand_vid"]]
            chk(pfx + ".skill_chars", len(ctext), j["skill_chars"]); chk(pfx + ".skill_words", words(ctext), j["skill_words"])
            chk(pfx + ".skill_text eq", True, j["skill_text"] == ctext)
            a, r = line_diff(inc_text, ctext)
            chk(pfx + ".added_lines", a, j["added_lines"]); chk(pfx + ".removed_lines", r, j["removed_lines"])
        # advance incumbent
        if st["accepted"]:
            inc_name[st["cand_vid"]] = pfx_name = "step%02d" % step
            inc_vid = st["cand_vid"]; inc_text = texts[inc_vid]; accepted_steps.append(step)
        else:
            pass
    chk("n dev-scored non-seed", n_dev_scored, sum(1 for s in jst[1:] if s["dev"] is not None))
    chk("n dev-scored with text != parent", n_dev_scored_differs, J["verification"]["edges_table_crosscheck"]["mine_dev_scored_text_differs"])
    chk("accepted steps", accepted_steps, J["extra"]["accepted_steps"])
    # returned
    ret_step = accepted_steps[-1] if accepted_steps else None
    chk("returned_skill_id", ("step%02d" % ret_step) if ret_step else "seed", J["returned_skill_id"])
    chk("final vid == last accepted cand vid", D["final_vid"], inc_vid)
    tkey = [k for k in D["test_evals"] if k.startswith(D["final_vid"])]
    if not tkey: disc.append("no test eval for final vid")
    else:
        te = D["test_evals"][tkey[0]]; ft = J["final_test"]
        chk("final_test", (te["n"], te["n_success"], te["n_unknown"]), (ft["n_episodes"], ft["n_success"], ft["n_unknown"]))
        if not relclose(ft["mean_input_tokens_per_episode"], te["mean_tok"]): disc.append("final_test tok")
    chk("summary final_skill eq final_skill.md", True, D["summary_final_skill_eq_final"])
    chk("current_skill.md==final_skill.md", True, D["current_skill_eq_final"])
    # summary iterations cross-check
    sx = D["summary_extra"]["iterations"]
    for st, x in zip(D["steps"], sx):
        chk(f"summary.extra.iterations t={x['t']} accepted", st["accepted"], x["accepted"])
        chk(f"summary.extra.iterations t={x['t']} baseline/candidate", (st["baseline_primary"], st["candidate_primary"]), (x["baseline"], x["candidate"]))
    return D, J, disc

if __name__ == "__main__":
    D, J, disc = compare(sys.argv[1], sys.argv[2])
    print("DISCREPANCIES:", len(disc))
    for x in disc: print(" -", x[:700])
