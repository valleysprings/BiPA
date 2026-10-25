import sys, json
from aux.skill_paths.verify.indep_check.vlib import *
from aux.skill_paths.verify.indep_check.derive_skillopt import derive

def relclose(a, b, tol=0.01):
    if a is None or b is None: return a is b
    return abs(a - b) <= tol * max(abs(b), 1e-9)

def compare(run, jfile):
    D = derive(run)
    J = json.load(open(PATHS / jfile))
    disc = []
    def chk(name, mine, theirs):
        if mine != theirs:
            disc.append(f"{name}: mine={mine!r} json={theirs!r}")
    jsteps = J["steps"]
    chk("n_steps(json incl seed)", len(D["steps"]) + 1, len(jsteps))
    # seed
    s0 = jsteps[0]
    chk("seed.event", "seed", s0["event"])
    chk("seed.step", 0, s0["step"])
    sd = D["seed_dev_evals"][0]
    chk("seed.dev.n", sd["n"], s0["dev"]["n"]); chk("seed.dev.n_success", sd["n_success"], s0["dev"]["n_success"]); chk("seed.dev.n_unknown", sd["n_unknown"], s0["dev"]["n_unknown"])
    if not relclose(s0["dev"]["mean_input_tokens_per_episode"], sd["mean_tok"]): disc.append(f"seed.tok mine={sd['mean_tok']} json={s0['dev']['mean_input_tokens_per_episode']}")
    chk("top seed_dev", {k: J["seed_dev"][k] for k in ("n", "n_success", "n_unknown")}, {"n": sd["n"], "n_success": sd["n_success"], "n_unknown": sd["n_unknown"]})
    chk("seed.skill_chars", D["seed_chars"], s0["skill_chars"])
    chk("seed.version", D["seed_vid"], s0["version_id"][:12])
    chk("seed skill_text eq seed.md", True, s0["skill_text"] == rd(Path(run) / "skillopt/seed.md"))
    # name map: json ids by vid
    idmap = {s["version_id"][:12]: s["candidate_id"] for s in jsteps if s.get("version_id")}
    for i, m in enumerate(D["steps"]):
        j = jsteps[i + 1]
        p = f"step{m['step']}"
        chk(p + ".step", m["step"], j["step"])
        chk(p + ".event", "candidate", j["event"])
        chk(p + ".version", m["cand_vid"], j["version_id"][:12])
        chk(p + ".kept", m["kept"], j["kept"])
        # map the derived parent vid to its json candidate id and compare to json parent_id
        exp_parent = idmap.get(m["parent_vid"])
        chk(p + ".parent_id", exp_parent, j["parent_id"])
        ev = m["dev_evals"]
        if len(ev) != 1:
            disc.append(f"{p}: derived dev evals count {len(ev)}")
        if ev:
            e = ev[0]
            if j["dev"] is None: disc.append(f"{p}: json dev null but an eval was found")
            else:
                chk(p + ".dev.n", e["n"], j["dev"]["n"]); chk(p + ".dev.n_success", e["n_success"], j["dev"]["n_success"]); chk(p + ".dev.n_unknown", e["n_unknown"], j["dev"]["n_unknown"])
                if not relclose(j["dev"]["mean_input_tokens_per_episode"], e["mean_tok"]): disc.append(f"{p}.tok mine={e['mean_tok']} json={j['dev']['mean_input_tokens_per_episode']}")
            # cross-check with history
            if abs(e["n_success"] / e["n"] - m["hist_selection_hard"]) > 1e-9: disc.append(f"{p}: history selection_hard {m['hist_selection_hard']} != eval {e['n_success']}/{e['n']}")
        chk(p + ".skill_chars", m["cand_chars"], j["skill_chars"])
        chk(p + ".skill_words", m["cand_words"], j["skill_words"])
        chk(p + ".skill_text eq", True, j["skill_text"] == rd(Path(run) / f"skillopt/steps/step_{m['step']:04d}/candidate_skill.md"))
        chk(p + ".added_lines", m["added"], j["added_lines"]); chk(p + ".removed_lines", m["removed"], j["removed_lines"])
        # gate
        g = j["gate"]
        if g:
            gd = g["detail"]
            chk(p + ".gate.candidate_n_success", ev[0]["n_success"] if ev else None, gd.get("candidate_n_success"))

        # ---- train + gate checks (extended)
        hist = json.load(open(Path(run) / "skillopt/history.json"))[i]
        pv = m["parent_vid"]
        want_n = hist["rollout_n"]; want_s = round(hist["rollout_hard"] * hist["rollout_n"])
        tr = j["train"]
        chk(p + ".train.n vs history rollout_n", want_n, tr["n"] if tr else None)
        chk(p + ".train.n_success vs history rollout_hard*n", want_s, tr["n_success"] if tr else None)
        if tr:
            chk(p + ".train.n_unknown", 0, tr["n_unknown"])
        # find train eval of the parent whose success matches and whose panel size == n
        cands_tr = []
        for pn, pp in eval_dirs(run, "train", pv):
            se = summarize_eval(pp / "evaluation.json")
            if se["n"] == want_n and se["n_success"] == want_s:
                cands_tr.append((pn, se))
        if not cands_tr: disc.append(f"{p}: no train eval of parent {pv} with {want_s}/{want_n}")
        else:
            toks = {round(c[1]["mean_tok"], 3) for c in cands_tr}
            if tr and "mean_input_tokens_per_episode" in tr:
                if not any(relclose(tr["mean_input_tokens_per_episode"], c[1]["mean_tok"]) for c in cands_tr): disc.append(f"{p}.train tok json={tr['mean_input_tokens_per_episode']} mine={toks}")
        if g:
            gd = g["detail"]
            chk(p + ".gate.current_n_success_before", round(({0: None}.get(0) or 0) + 0) if False else None, None)  # placeholder no-op
            prev_current = D["steps"][i-1]["hist_current_score_after"] if i > 0 else None
            exp_before = (D["seed_dev_evals"][0]["n_success"] if i == 0 else round(prev_current * 30))
            chk(p + ".gate.current_n_success_before", exp_before, gd.get("current_n_success_before"))
            chk(p + ".gate.action", m["action"], gd.get("action"))
            chk(p + ".gate.margin", (ev[0]["n_success"] - exp_before) if ev else None, gd.get("margin_tasks_over_current"))
    # returned skill
    ret = J["returned_skill_id"]
    chk("returned_skill_id -> vid", D["best_vid"], idmap.get(D["best_vid"]) and D["best_vid"])
    chk("returned_skill_id name", idmap.get(D["best_vid"]), ret)
    ft = J["final_test"]
    te = D["test_evals_returned"]
    if te and ft:
        chk("final_test.n_episodes", te[0]["n"], ft["n_episodes"]); chk("final_test.n_success", te[0]["n_success"], ft["n_success"])
    else:
        disc.append(f"final_test mine={te} json={ft}")
    return D, J, disc

if __name__ == "__main__":
    D, J, disc = compare(sys.argv[1], sys.argv[2])
    print("DISCREPANCIES:", len(disc))
    for x in disc: print(" -", x[:600])
