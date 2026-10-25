"""Write data/mech/mech_tiebreak.md from the tiebreak.json of aux.edit_gate_and_design.tiebreak (no other inputs).
Test numbers of the returned skills and the seed are Table 1 (mech_table1.md, avg@3), copied below as TABLE1.
Section (d) holds the within-run counts for the accepted DeepSeek edges.
Run: python -m aux.edit_gate_and_design.tiebreak_record"""
import json, math

from src import layout
OUT = layout.OUT / "data/mech/mech_tiebreak.md"
SEED = "4fd5ca441c55"
BN = {"tau3-bench": "tau3", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
ON = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
# Table 1 (mech_table1.md): (Q, rho) of the returned skill; seed Q per benchmark
TABLE1 = {"deepseek-flash": {"seed": {"terminal-bench-2-1": 75.9, "tau3-bench": 76.4, "automationbench": 45.0},
                             "gepa": {"terminal-bench-2-1": (74.1, 1.34), "tau3-bench": (75.6, 1.15), "automationbench": (44.2, 1.21)},
                             "skillopt": {"terminal-bench-2-1": (64.8, 1.43), "tau3-bench": (79.1, 1.22), "automationbench": (41.7, 1.25)},
                             "skilladam": {"terminal-bench-2-1": (68.5, 1.37), "tau3-bench": (75.1, 1.09), "automationbench": (44.7, 1.08)}},
          "gpt-6-luna": {"seed": {"terminal-bench-2-1": 68.6, "tau3-bench": 69.3, "automationbench": 37.2},
                         "gepa": {"terminal-bench-2-1": (68.6, 1.00), "tau3-bench": (65.3, 1.04), "automationbench": (41.4, 1.07)},
                         "skillopt": {"terminal-bench-2-1": (68.6, 1.00), "tau3-bench": (63.1, 1.01), "automationbench": (42.8, 1.25)},
                         "skilladam": {"terminal-bench-2-1": (68.6, 1.00), "tau3-bench": (64.0, 0.98), "automationbench": (38.6, 0.93)}}}


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def frac(p):
    return f"{p[0]}/{p[1]} ({100 * p[0] / p[1]:.0f}%)"


def ratio(x):
    return "-" if not x or x["L"] != x["L"] else f"{math.exp(x['L']):.2f}"


def dtask(x):
    return "-" if not x else f"{int(x['D']):+d}/{x['n_y']}"


def zl(x):
    return "-" if not x or x["z_L"] != x["z_L"] else f"{x['z_L']:+.1f}"


def dcount(es):
    return [len(es), sum(abs(e["z_L"]) >= 1.96 for e in es), sum(e["z_L"] >= 1.96 for e in es),
            sum(abs(e["z_D"]) >= 1.96 for e in es), sum(e["D"] >= e["crit"] for e in es)]


def main():
    D = json.loads((layout.OUT / "edit_gate_and_design/tiebreak.json").read_text())


    lines = ["# mech_tiebreak", "",
             "Audit record for design principle 1 (success cannot separate a candidate from its parent; cost can). Baseline runs only, "
             "no new episodes, no LLM calls. Written by `aux/edit_gate_and_design/tiebreak.py` (numbers, `tiebreak.json`) and `tiebreak_record.py` (this file).", "",
             "## Runs", "",
             "DeepSeek: skillopt/gepa/skilladam/<bench>/native-20261008-https; edges = the 59 canonical dev edges of mech_edges (rebuilt with "
             "`edges.skillopt_edges / gepa_edges / skilladam_edges`, one shared dev panel each; SkillAdam x TB2 excluded as there). Luna: the same builders "
             "on native-20261012, the 62 dev-scored edges of mech_rules_luna (SkillAdam x TB2 excluded). Seed-vs-seed noise: distinct full-panel dev draws "
             "of the seed among `selection.seed_files` (paper_input: no 1017, no LODO runs) and the kept evaluations of `selection.kept_dev`. Test: Table 1 (mech_table1, avg@3).", "",
             "## Method", "",
             "- Per edge, on the shared dev tasks: success difference D = sum over tasks of (child - parent) with paired SE sqrt(n) sd(d); token change "
             "L = mean over tasks of log(X_child / X_parent), X = episode input tokens (DeepSeek `dataloader.task_row`; Luna: distinct completed requests, as Table 1), "
             "paired SE sd(l)/sqrt(n). Separable = |z| >= 1.96; for success also |D| >= the one-sided 0.05 critical value of aux.selection_replay.selection.critical_value "
             "(TB2 9 tasks: 2; tau3 30: 5, 10: 3; AutoBench 30: 4, 10: 3; the DeepSeek values are applied to Luna too).",
             "- D reproduces mech_edges' dQ x n_pair / 100 on all 59 edges; L correlates 0.999 with its dlogX (dlogX keeps only lever-valid tasks).",
             "- Pool rule (GEPA, SkillOpt, SkillAdam x TB2: every version the run scored on its full dev panel, seed included): eligible = versions whose dev "
             "success is less than one critical value below the pool best. P: return the eligible version with the lowest mean log input tokens. "
             "Z: return the anchor (the seed if eligible, else the optimizer's own pick) unless an eligible version is cheaper than it with token z <= -1.96 "
             "(then the cheapest such).",
             "- Sequential gate (SkillOpt chain; SkillAdam x tau3/AutoBench iterations on their 10-task subsets): against the current skill, accept iff "
             "D >= c, or D > -c and the candidate is cheaper (P: L < 0; Z: token z <= -1.96). A SkillAdam comparison is evaluable only while the replay's "
             "current skill was scored on that iteration's subset; later ones are skipped. The same gate per logged edge (against its logged parent) gives the "
             "refused / newly admitted counts. Proposals after a divergence were generated from a different parent: a selection counterfactual on logged "
             "proposals, not a rerun.", ""]

    # ---- (a)
    lines += ["## (a) What separates a candidate from its parent on dev", ""]
    rows = []
    for m in ("deepseek-flash", "gpt-6-luna"):
        for g, s in D[m]["summary"].items():
            rows.append([m, g, s["edges"], frac(s["success_z"]), frac(s["success_crit"]), frac(s["tokens_z"]), f"{s['tokens_z_up'][0]} / {s['tokens_z_down'][0]}",
                         f"{s['median_abs_zD']:.2f}", f"{s['median_abs_zL']:.2f}", f"{s['median_se_D']:.2f}", f"{s['median_se_L']:.3f}", f"{s['median_abs_L']:.3f}",
                         frac(s["tokens_z_given_success_not_crit"])])
    lines += [table(["model", "edges", "n", "success |z|>=1.96", "success |D|>=crit", "tokens |z|>=1.96", "tokens up / down", "median |z| success",
                     "median |z| tokens", "median SE of D (tasks)", "median SE of L", "median |L|", "tokens separable when success is not (|D|<crit)"], rows), ""]
    ds = [e for e in D["deepseek-flash"]["edges"] if e["parent"] not in ("seed", "c0")]
    lines += [f"DeepSeek edges whose parent is not the seed (the seed's dev episodes come from the earlier run native-20261004): tokens separable "
              f"{sum(abs(e['z_L']) >= 1.96 for e in ds)}/{len(ds)}, success {sum(abs(e['z_D']) >= 1.96 for e in ds)}/{len(ds)} (|z|) and "
              f"{sum(abs(e['D']) >= e['crit'] for e in ds)}/{len(ds)} (critical value). No edge has both separable.", ""]
    lines += ["Edges where success is separable:", ""]
    for m in ("deepseek-flash", "gpt-6-luna"):
        for e in D[m]["edges"]:
            if abs(e["z_D"]) >= 1.96 or abs(e["D"]) >= e["crit"]:
                lines.append(f"- {m} {ON[e['opt']]} {BN[e['bench']]} {e['parent'][:12]} -> {e['child'][:12]}: D {int(e['D']):+d}/{e['n_y']} (crit {e['crit']}, z {e['z_D']:+.1f}), "
                             f"token ratio {math.exp(e['L']):.2f} (z {e['z_L']:+.1f}), accepted={e['accepted']}")
    lines += [""]

    lines += ["### Seed-vs-seed noise floor (DeepSeek dev)", ""]
    rows = []
    for b, n in D["noise"].items():
        for p in n["pairs"]:
            rows.append([BN[b], " vs ".join(f.split("/results")[0] for f in p["files"]), f"{p['solved_p']} vs {p['solved_c']}", p["n_x"], f"{p['sd_l']:.2f}",
                         f"{p['L']:+.3f}", f"{p['se_L']:.3f}", f"{p['z_L']:+.1f}"])
    lines += [table(["bench", "seed draws", "solved", "tasks", "SD per-task log ratio", "mean log ratio", "SE", "z"], rows), ""]
    sd = D["deepseek-flash"]["by_bench_sd_l"]
    seed_pairs = [p for n in D["noise"].values() for p in n["pairs"]]
    separable = sum(abs(p["z_L"]) >= 1.96 for p in seed_pairs)
    lines += [f"Per-task SD of the log token ratio, seed vs seed: tau3 {D['noise']['tau3-bench']['sd_task_log_ratio']:.2f}, AutoBench "
              f"{D['noise']['automationbench']['sd_task_log_ratio']:.2f}, TB2 {D['noise']['terminal-bench-2-1']['sd_task_log_ratio']:.2f} (one pair); "
              f"median over the 59 edges: tau3 {sd['tau3-bench']:.2f}, AutoBench {sd['automationbench']:.2f}, TB2 {sd['terminal-bench-2-1']:.2f}. "
              "The edges' per-task spread equals run-to-run spread on tau3 and AutoBench, so the paired SE of an edge is the run-to-run noise of its mean: "
              "about 0.05 (tau3) and 0.11 (AutoBench) log points on 30 tasks. "
              f"{separable} of {len(seed_pairs)} seed pairs is separable on tokens (TB2, +0.31, two runs on different dates); "
              "the tau3 and AutoBench pairs are within 1.2 SE. Luna has one seed execution per dev task, so no Luna seed-vs-seed pair exists.", ""]

    # ---- (b)
    for m in ("deepseek-flash", "gpt-6-luna"):
        lines += [f"## (b) Tie-break replay, {m}", ""]
        rows = []
        t1 = TABLE1[m]
        for cell, c in D[m]["replay"].items():
            opt, bench = cell.split("|")
            q, rho = t1[opt][bench]
            for rule, r in c["picks"].items():
                same = r["same_as_returned"]
                name = r["name"] if r["name"] != "?" else r["pick"]
                if r["pick"] == SEED:
                    test = f"seed: Q {t1['seed'][bench]} vs returned {q} (rho {rho:.2f})"
                elif same:
                    test = f"= returned (Q {q}, rho {rho:.2f})"
                else:
                    test = "none (dev only)"
                rows.append([f"{ON[opt]} {BN[bench]}", c["returned_name"] if c["returned_name"] not in (None, "?") else c["returned"], rule, name, "yes" if same else "no",
                             dtask(r.get("dev_vs_returned")), ratio(r.get("dev_vs_returned")), zl(r.get("dev_vs_returned")),
                             dtask(r.get("dev_vs_seed")) if r["pick"] != SEED else "0", ratio(r.get("dev_vs_seed")) if r["pick"] != SEED else "1.00", test])
        lines += [table(["run", "returned", "rule", "rule returns", "same", "dev D vs returned", "dev tokens vs returned", "z", "dev D vs seed",
                         "dev tokens vs seed", "test (Table 1)"], rows), ""]
        for k in ("gate", "gate_z"):
            g = D[m][k]
            by = "; ".join(f"{ON[o]} {v['refused']}/{v['accepted']} refused, {v['would_accept']}/{v['rejected']} admitted" for o, v in g["by_opt"].items())
            lines.append(f"- Edge gate {k[-1].upper() if k == 'gate_z' else 'P'}: refuses {g['refused']} of {g['accepted']} accepted edges, admits "
                         f"{g['would_accept']} of {g['rejected']} rejected ({by}). Mean L of refused accepted edges {g['mean_L_refused']:+.3f}, of kept {g['mean_L_kept']:+.3f}.")
        lines += [""]
        lines += ["Pools (dev solved on the full panel; token change vs the seed in log points; * = eligible; uN = SkillOpt slow-update skill; a bare hash = "
                  "a version the run scored on dev that is in no lineage file, a re-proposal close to a listed one, e.g. Luna SkillOpt tau3 5e2026 = s3 plus one rule, "
                  "Luna GEPA tau3 262f23 a full rewrite near c7; kept in the pool because the run scored it):", ""]
        for cell, c in D[m]["replay"].items():
            if "pool" not in c:
                continue
            p = c["pool"]
            names = dict((v, n) for v, n in p["pool"])
            s0 = p["logx"].get(SEED, 0.0)
            items = ", ".join(f"{names[v] if names[v] != '?' else v[:6]} {p['solved'][v]} {p['logx'][v] - s0:+.2f}{'*' if v in p['eligible'] else ''}" for v, _ in p["pool"])
            lines.append(f"- {cell} (crit {p['crit']}, best {p['best']}): {items}")
        lines += [""]

    # ---- summaries for (c)
    ds_seed = [(o, b) for o in TABLE1["deepseek-flash"] if o != "seed" for b in TABLE1["deepseek-flash"][o]
               if D["deepseek-flash"]["replay"][f"{o}|{b}"]["picks"].get("pool_z", D["deepseek-flash"]["replay"][f"{o}|{b}"]["picks"].get("seq_z"))["pick"] == SEED]
    dq = [TABLE1["deepseek-flash"][o][b][0] - TABLE1["deepseek-flash"]["seed"][b] for o, b in ds_seed]
    rh = [TABLE1["deepseek-flash"][o][b][1] for o, b in ds_seed]
    lu_seed = [(o, b) for o in TABLE1["gpt-6-luna"] if o != "seed" for b in TABLE1["gpt-6-luna"][o] if b != "terminal-bench-2-1"
               and D["gpt-6-luna"]["replay"][f"{o}|{b}"]["picks"].get("pool_z", D["gpt-6-luna"]["replay"][f"{o}|{b}"]["picks"].get("seq_z"))["pick"] == SEED]
    lq = [TABLE1["gpt-6-luna"][o][b][0] - TABLE1["gpt-6-luna"]["seed"][b] for o, b in lu_seed]
    lines += ["## (c) Verdict", "",
              f"- Z rule, DeepSeek: returns the seed in {len(ds_seed)} of 9 runs ({', '.join(ON[o] + ' ' + BN[b] for o, b in ds_seed)}). On test (Table 1) the skills those runs "
              f"returned differ from the seed by {', '.join(f'{x:+.1f}' for x in dq)} points (mean {sum(dq) / len(dq):+.1f}) at rho {min(rh):.2f}-{max(rh):.2f}, "
              f"so the rule's choice reads {100 * (1 - 1 / min(rh)):.0f}-{100 * (1 - 1 / max(rh)):.0f}% fewer input tokens than the returned skill. GEPA tau3 would return c1 "
              "and SkillAdam tau3 1c1c98f79c3c (cheaper, no test evaluation); SkillAdam TB2 is unchanged.",
              f"- Z rule, Luna: returns the seed in {len(lu_seed)} of the 6 non-TB2 runs ({', '.join(ON[o] + ' ' + BN[b] for o, b in lu_seed)}); their returned skills differ from "
              f"the seed on test by {', '.join(f'{x:+.1f}' for x in lq)} points (mean {sum(lq) / len(lq):+.1f}): the rule gives up the two AutoBench gains that dev could not see "
              "and avoids the two tau3 losses. On Luna tokens separate 4 of 62 edges, so the rule mostly falls back to the parent.",
              "- P (point-estimate 'cheaper') flips on noise: it admits SkillAdam tau3 b227f46b80a4 for a 0.3% token difference and, on Luna, picks pool versions "
              "whose token advantage has |z| < 1.7. Cost must be judged against its own noise, like success.",
              "- Supported wording: keep the parent unless the candidate is better by more than selection noise or cheaper by more than cost noise. "
              "'Keep the cheaper one' as worded (point estimate) is not supported; half (ii) holds on DeepSeek (cost separates 21 of 59 edges, success 2) "
              "but not on Luna, where the skills add little cost (4 of 62 vs 5 of 62).", ""]
    # ---- (d) within-run counts (DeepSeek; within run = parent not in {seed, c0})
    E = D["deepseek-flash"]["edges"]
    W = [e for e in E if e["parent"] not in ("seed", "c0")]
    S = [e for e in E if e["parent"] in ("seed", "c0")]


    groups = [("within run, all", W), ("within run, accepted", [e for e in W if e["accepted"]]),
              ("within run, rejected", [e for e in W if not e["accepted"]]),
              ("seed parent (across runs), accepted", [e for e in S if e["accepted"]]), ("all accepted", [e for e in E if e["accepted"]])]
    dc = {g: dcount(es) for g, es in groups}
    wa, sa, aa = dc["within run, accepted"], dc["seed parent (across runs), accepted"], dc["all accepted"]
    tb2_seed_z = [p["z_L"] for p in D["noise"]["terminal-bench-2-1"]["pairs"]]
    lines += ["## (d) Within-run counts for the accepted edges", "",
              f"Recounted from `tiebreak.json` (deepseek-flash edges; within run = parent not in {{seed, c0}}, the restriction of the "
              f"{dc['within run, all'][1]}/{dc['within run, all'][0]} line above). Written by tiebreak_record.py.", "",
              table(["DeepSeek edges", "n", "tokens |z|>=1.96", "tokens up (z>=1.96)", "success |z|>=1.96", "success D >= crit"], [[g] + dc[g] for g, _ in groups]), "",
              f"In words: of the {wa[0]} within-run accepted edits, {wa[2]} raised tokens beyond noise and "
              f"{'none' if wa[4] == 0 else wa[4]} raised success beyond its critical value; the all-accepted {aa[2]}/{aa[0]} and {aa[4]}/{aa[0]} include {sa[2]} and {sa[4]} "
              f"seed-parent edges compared across runs (run-date drift; TB2 seed-vs-seed pair z={', '.join(f'{z:+.1f}' for z in tb2_seed_z)}).", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)
    print("deepseek seed-returning", ds_seed, dq, rh)
    print("luna seed-returning", lu_seed, lq)


if __name__ == "__main__":
    main()
