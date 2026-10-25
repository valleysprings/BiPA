"""Independent verification of mech_paths.md numbers from paths/*.json.
Does not import analyze_paths.py / analyze_shared.py.  Run with python -I.  Read-only on everything but stdout."""
import json, glob, re, math, itertools
from pathlib import Path
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p))
        runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    print("n path files:", len(runs))

    MODEL = {"DeepSeek": "deepseek-flash", "Luna": "gpt-6-luna"}
    BENCH = {"tau3": "tau3-bench", "AutoB": "automationbench", "TB2": "terminal-bench-2-1"}
    OPT = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}
    def key(label, opt):
        m = label.split()
        model = MODEL[m[0]]; bench = BENCH[m[1]]
        rid = "native-20261008-https" if model == "deepseek-flash" else "native-20261012"
        if len(m) > 2 and m[2] == "near-empty":
            rid = "fromscratch"
        return (model, OPT[opt], bench, rid)

    def scored(d):
        return [s for s in d["steps"] if s["event"] == "candidate" and s.get("dev")]
    def cands(d):
        return [s for s in d["steps"] if s["event"] == "candidate"]

    # ---- parse the record's table 'Runs, steps and the returned skill'
    md = (SC / "mech_paths.md").read_text().splitlines()
    rows = []
    on = False
    for ln in md:
        if ln.startswith("| run | optimizer | dev panel"):
            on = True; continue
        if on:
            if not ln.startswith("|"): break
            if ln.startswith("|---"): continue
            c = [x.strip() for x in ln.strip("|").split("|")]
            rows.append(c)
    print("record rows:", len(rows))
    ok = 0
    for c in rows:
        k = key(c[0], c[1]); d = runs[k]
        ncand = len(cands(d)); nsc = len(scored(d)); nkept = sum(1 for s in cands(d) if s["kept"] is True)
        ft = d["final_test"]
        ts = f'{ft["n_success"]}/{ft["n_episodes"]}' + (f' ({ft["n_unknown"]} unknown)' if ft.get("n_unknown") else "")
        rec = (int(c[3]), int(c[4]), int(c[5]), c[6], c[7], c[8])
        mine = (ncand, nsc, nkept, d["returned_skill_id"] if d["returned_skill_id"] != "seed" else "seed", ts, "yes" if d["partial"] else "no")
        good = rec == mine
        ok += good
        if not good:
            print("MISMATCH", c[0], c[1], "record", rec, "mine", mine)
    print(f"Runs table: {ok}/{len(rows)} rows identical (steps, dev-scored, kept, returned, test, partial)")

    # ---- dev score sequences for GEPA / SkillOpt (record section 1 table)
    seqrows = []
    on = False
    for ln in md:
        if ln.startswith("| optimizer | run | seed draw"):
            on = True; continue
        if on:
            if not ln.startswith("|"): break
            if ln.startswith("|---"): continue
            seqrows.append([x.strip() for x in ln.strip("|").split("|")])
    okc = 0
    share_first = {"GEPA": [], "SkillOpt": []}
    pos_gain = {"GEPA": 0, "SkillOpt": 0}
    first_best = {"GEPA": 0, "SkillOpt": 0}
    for c in seqrows:
        opt, lab = c[0], c[1]
        k = key(lab, opt); d = runs[k]
        sc = scored(d)
        seq = [s["dev"]["n_success"] for s in sc]
        base = d["seed_dev"]["n_success"]
        best = max(seq); kbest = seq.index(best) + 1 if best > base else None
        gain = max(0, best - base)
        rec_seq = [int(x) for x in c[3].split()]
        rec_seed = int(c[2].split("/")[0])
        rec_gain = int(re.match(r"\+(\d+)", c[5]).group(1))
        after = (len(seq) - kbest) if kbest else len(seq)
        rec_after = int(c[7])
        # share of final gain at first scored candidate
        if gain > 0:
            sh = max(0, seq[0] - base) / gain
            share_first[opt].append(sh); pos_gain[opt] += 1
            first_best[opt] += (kbest == 1) or (seq[0] == best)
        rec_share = c[6]
        good = (seq == rec_seq and base == rec_seed and gain == rec_gain and after == rec_after)
        okc += good
        if not good:
            print("SEQ MISMATCH", opt, lab, seq, rec_seq, base, rec_seed, gain, rec_gain, after, rec_after)
    print(f"dev-score sequences / seed draw / final gain / scored-after-best: {okc}/{len(seqrows)} rows identical")
    for opt in ("SkillOpt", "GEPA"):
        print(opt, "hand-written+near-empty rows with positive gain:", pos_gain[opt], " mean share at first scored (incl near-empty row):", sum(share_first[opt]) / len(share_first[opt]))
    # hand-written only means
    for opt in ("SkillOpt", "GEPA"):
        vals = []; fb = 0; n_pos = 0
        for c in seqrows:
            if c[0] != opt or "near-empty" in c[1]: continue
            d = runs[key(c[1], opt)]; seq = [s["dev"]["n_success"] for s in scored(d)]; base = d["seed_dev"]["n_success"]
            g = max(0, max(seq) - base)
            if g > 0:
                n_pos += 1; vals.append(max(0, seq[0] - base) / g)
                # best already at first scored candidate: first scored candidate reaches the max
                fb += (seq[0] == max(seq))
        print(f"{opt} hand-written: runs {sum(1 for c in seqrows if c[0]==opt and 'near' not in c[1])}, positive {n_pos}, best at first scored {fb}, mean share {sum(vals)/len(vals):.3f}")
