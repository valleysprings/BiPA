import sys
import json
from aux.skill_paths.verify.indep_check.vlib import *
from aux.skill_paths.verify.indep_check import derive_gepa

def relclose(a, b, tol=0.01):
    if a is None or b is None: return a is b
    return abs(a - b) <= tol * max(abs(b), 1e-9)

def train_panel_for(run, parent_vid, child_vid):
    """panels (train) where both parent and child were evaluated"""
    pp = {pn: p for pn, p in eval_dirs(run, "train", parent_vid)}
    cp = {pn: p for pn, p in eval_dirs(run, "train", child_vid)}
    return sorted(set(pp) & set(cp)), pp, cp

def compare(run, jfile, expect_partial=False):
    run = Path(run)
    D = derive_gepa.derive(run)
    J = json.load(open(PATHS / jfile))
    disc = []
    def chk(name, mine, theirs):
        if mine != theirs: disc.append(f"{name}: mine={mine!r} json={theirs!r}")
    pool_text = D["_pool_text"]
    jst = J["steps"]
    # expected step list
    exp = [("seed", 0)]
    for it in D["its"]:
        if it["has_scores"]:
            exp.append((("c%d" % it["new_idx"]) if it["new_idx"] is not None else ("r%d" % it["t"]), it["t"]))
    chk("n_steps incl seed", len(exp), len(jst))
    chk("order of candidate ids", [e[0] for e in exp], [s["candidate_id"] for s in jst])
    chk("order of step numbers", [e[1] for e in exp], [s["step"] for s in jst])
    skipped = [it["t"] for it in D["its"] if not it["has_scores"]]
    chk("skipped iterations", skipped, [s["step"] for s in J.get("skipped_iterations", [])])
    # seed
    s0 = jst[0]; p0 = D["pool"][0]["dev"][0]
    chk("seed.dev", (p0["n"], p0["n_success"], p0["n_unknown"]), (s0["dev"]["n"], s0["dev"]["n_success"], s0["dev"]["n_unknown"]))
    if not relclose(s0["dev"]["mean_input_tokens_per_episode"], p0["mean_tok"]): disc.append(f"seed.tok mine={p0['mean_tok']} json={s0['dev']['mean_input_tokens_per_episode']}")
    chk("top seed_dev", (J["seed_dev"]["n"], J["seed_dev"]["n_success"], J["seed_dev"]["n_unknown"]), (p0["n"], p0["n_success"], p0["n_unknown"]))
    chk("seed.skill_chars", D["pool"][0]["chars"], s0["skill_chars"])
    chk("seed val_score_recorded", D["val_scores"][0], s0.get("val_score_recorded"))
    by_t = {it["t"]: it for it in D["its"]}
    n_dev_scored = 0
    for (cid, t), j in zip(exp[1:], jst[1:]):
        it = by_t[t]; pfx = f"t{t}/{cid}"
        chk(pfx + ".candidate_id", cid, j["candidate_id"])
        chk(pfx + ".parent_id", "c%d" % it["parent"], j["parent_id"])
        accepted = it["new_idx"] is not None
        chk(pfx + ".kept", accepted, j["kept"])
        ptxt = pool_text[it["parent"]]
        if accepted:
            k = it["new_idx"]; pe = D["pool"][k]; ctext = pool_text[k]
            n_dev_scored += 1
            e = pe["dev"][0] if pe["dev"] else None
            if e is None or j["dev"] is None: disc.append(f"{pfx}: dev missing mine={e} json={j['dev']}")
            else:
                chk(pfx + ".dev", (e["n"], e["n_success"], e["n_unknown"]), (j["dev"]["n"], j["dev"]["n_success"], j["dev"]["n_unknown"]))
                if not relclose(j["dev"]["mean_input_tokens_per_episode"], e["mean_tok"]): disc.append(f"{pfx}.tok mine={e['mean_tok']} json={j['dev']['mean_input_tokens_per_episode']}")
                if abs(e["n_success"] / e["n"] - D["val_scores"][k]) > 1e-9: disc.append(f"{pfx}: val_scores {D['val_scores'][k]} != eval {e['n_success']}/{e['n']}")
            chk(pfx + ".version", pe["vid"], (j.get("version_id") or "")[:12])
            chk(pfx + ".pool_index", k, j.get("pool_index"))
        else:
            ctext = D["_rej_text"].get(t)
            chk(pfx + ".dev null", None, j["dev"])
            chk(pfx + ".pool_index null", None, j.get("pool_index"))
            chk(pfx + ".version", it.get("rej_vid"), (j.get("version_id") or "")[:12])
            if ctext is None: disc.append(f"{pfx}: could not parse rejected text")
        # gate
        g = j["gate"]
        chk(pfx + ".gate.kind", "minibatch_sum_strict_improvement", g["kind"])
        gd = g["detail"]
        chk(pfx + ".gate.parent_sum", it["parent_sum"], gd["parent_sum"]); chk(pfx + ".gate.child_sum", it["child_sum"], gd["child_sum"])
        chk(pfx + ".gate.parent_scores", it["parent_scores"], gd["parent_scores"]); chk(pfx + ".gate.child_scores", it["child_scores"], gd["child_scores"])
        chk(pfx + ".gate.accept", it["child_sum"] > it["parent_sum"], gd["accept"])
        # train: child's outcome on B_t
        if ctext is not None:
            cvid = vid(ctext)[:12]; pvid = D["pool"][it["parent"]]["vid"]
            panels, pp, cp = train_panel_for(run, pvid, cvid)
            if len(panels) != 1:
                disc.append(f"{pfx}: train panel intersection ambiguous/empty: {panels}")
            if panels:
                # choose the panel whose sums agree with the log
                good = []
                for pn in panels:
                    cs = summarize_eval(cp[pn] / "evaluation.json"); ps = summarize_eval(pp[pn] / "evaluation.json")
                    if cs["n_success"] == it["child_sum"] and ps["n_success"] == it["parent_sum"]:
                        good.append((pn, cs, ps))
                if len(good) != 1:
                    disc.append(f"{pfx}: {len(good)} panels agree with log sums (panels {panels})")
                if good:
                    pn, cs, ps = good[0]
                    tr = j["train"]
                    chk(pfx + ".train", (cs["n"], cs["n_success"], cs["n_unknown"]), (tr["n"], tr["n_success"], tr["n_unknown"]))
                    if "mean_input_tokens_per_episode" in tr and not relclose(tr["mean_input_tokens_per_episode"], cs["mean_tok"]): disc.append(f"{pfx}.train tok mine={cs['mean_tok']} json={tr['mean_input_tokens_per_episode']}")
                    pm = gd.get("parent_minibatch")
                    if pm:
                        chk(pfx + ".parent_minibatch", (ps["n"], ps["n_success"], ps["n_unknown"]), (pm["n"], pm["n_success"], pm["n_unknown"]))
            # text
            chk(pfx + ".skill_chars", len(ctext), j["skill_chars"]); chk(pfx + ".skill_words", words(ctext), j["skill_words"])
            chk(pfx + ".skill_text eq", True, j["skill_text"] == ctext)
            a, r = line_diff(ptxt, ctext)
            chk(pfx + ".added_lines", a, j["added_lines"]); chk(pfx + ".removed_lines", r, j["removed_lines"])
    chk("n dev-scored non-seed", n_dev_scored, sum(1 for s in jst[1:] if s["dev"] is not None))
    # returned
    chk("returned_skill_id", "c%d" % D["best_idx"], J["returned_skill_id"])
    chk("returned vid", D["returned_vid"], (J["final_test"] or {}).get("version_id", "")[:12])
    te = D["test_evals_all"].get(D["returned_vid"] + "-" + [k for k in D["test_evals_all"] if k.startswith(D["returned_vid"])][0].split("-")[1]) if any(k.startswith(D["returned_vid"]) for k in D["test_evals_all"]) else None
    ft = J["final_test"]
    if te is None: disc.append(f"no test eval for returned vid; json final_test={ft}")
    else:
        chk("final_test.n_episodes", te["n"], ft["n_episodes"]); chk("final_test.n_success", te["n_success"], ft["n_success"]); chk("final_test.n_unknown", te["n_unknown"], ft.get("n_unknown"))
    chk("summary final_skill == candidates[best]", True, D["summary_final_skill_eq_best_candidate"])
    return D, J, disc

if __name__ == "__main__":
    D, J, disc = compare(sys.argv[1], sys.argv[2])
    print("DISCREPANCIES:", len(disc))
    for x in disc: print(" -", x[:700])
