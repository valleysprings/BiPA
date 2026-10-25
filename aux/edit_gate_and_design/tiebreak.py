"""Principle 1 evidence: on the same dev evidence, can success separate a candidate from its parent, and can cost? And what would a
cost tie-break have returned from the optimizers' own candidate pools? Read-only; no episodes, no LLM calls.

Run: python -m aux.edit_gate_and_design.tiebreak
Writes OUT/edit_gate_and_design/tiebreak.json (all numbers) and prints a summary.

(a) per dev-scored edge (child vs its parent on their shared dev panel): success difference D (tasks) with paired SE sqrt(n) sd(d),
    mean per-task log input-token ratio L with paired SE sd(l)/sqrt(n); 'distinguishable' = |z| >= 1.96, and for success also the
    one-sided 0.05 critical value of selection.critical_value (TB2 9 -> 2, tau3 30 -> 5 / 10 -> 3, AutoBench 30 -> 4 / 10 -> 3; |D| >= c).
    Seed-vs-seed noise floor: per-task log token ratio between two distinct full-panel seed dev draws (selection.seed_files, and the kept draws of selection.kept_dev).
(b) tie-break replay: pool rule (GEPA, SkillOpt, SkillAdam TB2: every version scored on the run's full dev panel) = among versions whose dev
    success is not significantly below the pool best (best - s < c), the one with the lowest mean log dev input tokens (ties -> earliest);
    sequential gate (SkillOpt chain, SkillAdam iterations) = against the current skill on the panel of the comparison, accept iff
    gain >= c, or gain > -c and the candidate's mean per-task log token ratio < 0.
"""
import json, math, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

from aux.rule_coding import edges as E
from aux.selection_replay import selection as S
from src import layout
from src.utils import dataloader as C
from src.utils.dataloader import vid, SEED_TEXT

OUT = layout.OUT / "edit_gate_and_design/tiebreak.json"
TAGS = {"deepseek-flash": "native-20261008-https", "gpt-6-luna": "native-20261012"}
CRIT = {"terminal-bench-2-1": {9: 2}, "tau3-bench": {10: 3, 30: 5}, "automationbench": {10: 3, 30: 4}}
SEEDV = vid(SEED_TEXT)[:12]
Z = 1.96


def crit(bench, n):
    table = CRIT[bench]
    return table[min(table, key=lambda k: abs(k - n))]


_cache = {}


def tokens(e, path):
    """Episode input tokens. DeepSeek: src.utils.dataloader.task_row (requests complete). Luna: summed over the distinct completed requests
    (provider-failed requests carry no usage and are retried), as appC_luna / mech_attribution_luna."""
    if "gpt-6-luna" not in str(path):
        x = C.task_row(e)["X"]
        return x if x else None
    seen, x = set(), 0
    for r in e.get("requests", []):
        if r["request_id"] in seen or not isinstance(r.get("input_tokens"), (int, float)):
            continue
        seen.add(r["request_id"])
        x += r["input_tokens"]
    return x if x > 0 else None


def load_eval(path):
    """{task: (success 0/1 or None, input tokens or None)} of one evaluation.json (first repetition only; dev runs have one)."""
    path = str(path)
    if path in _cache:
        return _cache[path]
    d = json.loads(Path(path).read_text())
    eps = {}
    for e in d["episodes"]:
        eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    out = {}
    for t in d["panel"]:
        key = (t["task_id"], t.get("repetition", 0))
        if key[1] != 0:
            continue
        e = eps.get(key, {})
        ok = bool(e.get("outcome_valid")) and e.get("success") is not None and bool(e.get("requests"))
        out[t["task_id"]] = (int(bool(e["success"])) if ok else None, tokens(e, path))
    _cache[path] = out
    return out


def compare(ev_p, ev_c):
    """Paired success and token statistics of child vs parent on their shared tasks."""
    ty = [t for t in ev_p if t in ev_c and ev_p[t][0] is not None and ev_c[t][0] is not None]
    tx = [t for t in ev_p if t in ev_c and ev_p[t][1] and ev_c[t][1]]
    d = np.array([ev_c[t][0] - ev_p[t][0] for t in ty], float)
    l = np.array([math.log(ev_c[t][1] / ev_p[t][1]) for t in tx], float)
    D = float(d.sum())
    se_D = float(math.sqrt(len(d)) * d.std(ddof=1)) if len(d) > 1 else float("nan")
    L = float(l.mean()) if len(l) else float("nan")
    se_L = float(l.std(ddof=1) / math.sqrt(len(l))) if len(l) > 1 else float("nan")
    zD = D / se_D if se_D > 0 else (0.0 if D == 0 else float("inf") * np.sign(D))
    zL = L / se_L if se_L > 0 else float("nan")
    return {"n_y": len(ty), "D": D, "se_D": se_D, "z_D": float(zD), "n_x": len(tx), "L": L, "se_L": se_L, "z_L": float(zL),
            "sd_l": float(l.std(ddof=1)) if len(l) > 1 else float("nan"), "solved_p": int(sum(ev_p[t][0] for t in ty)),
            "solved_c": int(sum(ev_c[t][0] for t in ty))}


def lineage(model, opt, bench):
    run = C.EX / model / opt / bench / TAGS[model]
    index = E.eval_index(run)
    if opt == "skillopt":
        nodes, edges = E.skillopt_edges(run)
    elif opt == "gepa":
        nodes, edges = E.gepa_edges(run)
    else:
        nodes, edges = E.skilladam_edges(run, E.delivered_texts(run), index)
    return run, nodes, edges, index


# ---------------------------------------------------------------- (a) edges
def edge_rows(model):
    rows = []
    table = None
    if model == "deepseek-flash":
        table = {(r["opt"], r["bench"], r["parent"], r["child"], r["panel"]): r for r in E.load_dev_edges()}
    for opt in E.OPTIMIZERS:
        for bench in C.BENCHES:
            if (opt, bench) == E.EXCLUDED_CELL:
                continue
            run, nodes, edges, index = lineage(model, opt, bench)
            for e in edges:
                if e["parent"] == "?":
                    continue
                vp, vc = vid(nodes[e["parent"]])[:12], vid(nodes[e["child"]])[:12]
                if vp == vc:
                    continue
                panels = sorted(k for k in set(index.get(vp, {})) & set(index.get(vc, {})) if k[0] == "dev")
                if not panels:
                    continue
                if table is not None:
                    panels = [k for k in panels if (opt, bench, e["parent"], e["child"], k[1]) in table]
                    if not panels:
                        continue
                # one panel per edge: the largest shared one (DeepSeek: the panel of the canonical table)
                key = max(panels, key=lambda k: len(load_eval(index[vp][k])))
                st = compare(load_eval(index[vp][key]), load_eval(index[vc][key]))
                c = crit(bench, st["n_y"])
                st.update(model=model, opt=opt, bench=bench, parent=e["parent"], child=e["child"], panel=key[1], accepted=bool(e["accepted"]), crit=c)
                if table is not None:
                    t = table[(opt, bench, e["parent"], e["child"], key[1])]
                    st.update(table_dlogX=t["dlogX"], table_dQ=t["dQ"], table_n_pair=t["n_pair"])
                rows.append(st)
    return rows


def summarize_edges(rows):
    def share(f, rs):
        return [int(sum(map(f, rs))), len(rs)]
    out = {}
    groups = {"all": rows, "accepted": [r for r in rows if r["accepted"]], "rejected": [r for r in rows if not r["accepted"]]}
    for opt in E.OPTIMIZERS:
        groups[opt] = [r for r in rows if r["opt"] == opt]
    for g, rs in groups.items():
        if not rs:
            continue
        out[g] = {
            "edges": len(rs),
            "success_z": share(lambda r: abs(r["z_D"]) >= Z, rs),
            "success_crit": share(lambda r: abs(r["D"]) >= r["crit"], rs),
            "success_gain_crit": share(lambda r: r["D"] >= r["crit"], rs),
            "tokens_z": share(lambda r: abs(r["z_L"]) >= Z, rs),
            "tokens_z_up": share(lambda r: r["z_L"] >= Z, rs),
            "tokens_z_down": share(lambda r: r["z_L"] <= -Z, rs),
            "tokens_beyond_noise_sd": None,
            "median_abs_zD": float(np.median([abs(r["z_D"]) for r in rs])),
            "median_abs_zL": float(np.median([abs(r["z_L"]) for r in rs])),
            "median_se_D": float(np.nanmedian([r["se_D"] for r in rs])),
            "median_se_L": float(np.nanmedian([r["se_L"] for r in rs])),
            "median_abs_L": float(np.median([abs(r["L"]) for r in rs])),
            "mean_L": float(np.mean([r["L"] for r in rs])),
            "both": share(lambda r: abs(r["z_D"]) >= Z and abs(r["z_L"]) >= Z, rs),
            "tokens_only": share(lambda r: abs(r["z_D"]) < Z and abs(r["z_L"]) >= Z, rs),
            "success_only": share(lambda r: abs(r["z_D"]) >= Z and abs(r["z_L"]) < Z, rs),
            "neither": share(lambda r: abs(r["z_D"]) < Z and abs(r["z_L"]) < Z, rs),
            "tokens_z_given_success_not_crit": share(lambda r: abs(r["z_L"]) >= Z, [r for r in rs if abs(r["D"]) < r["crit"]]),
        }
    return out


# ---------------------------------------------------------------- noise floor
def noise_floor():
    out = {}
    for bench in C.BENCHES:
        dev_files, _ = S.seed_files(bench)
        n_dev = S.PAPER_INPUTS[bench][0]
        draws = {}
        for f in dev_files:
            d = json.loads(Path(f).read_text())
            if len(d["episodes"]) != n_dev:
                continue
            ok = [e for e in d["episodes"] if S.verified(e)]
            key = frozenset((e["task_id"], S.signature(e)) for e in ok)
            draws.setdefault(key, f)
        evals = [(f.split("deepseek-flash/")[1], load_eval(f)) for f in draws.values()]
        evals += [(runs[0], {t: (o, x) for t, o, x in rows}) for runs, rows in S.kept_dev(bench, dev_files).items()]
        pairs = []
        for i in range(len(evals)):
            for j in range(i + 1, len(evals)):
                st = compare(evals[i][1], evals[j][1])
                st["files"] = [evals[i][0], evals[j][0]]
                pairs.append(st)
        sd = [p["sd_l"] for p in pairs]
        out[bench] = {"draws": len(evals), "pairs": pairs, "sd_task_log_ratio": float(np.mean(sd)) if sd else None,
                      "mean_abs_L": float(np.mean([abs(p["L"]) for p in pairs])) if pairs else None,
                      "se_L_mean": float(np.mean([p["se_L"] for p in pairs])) if pairs else None}
    return out


# ---------------------------------------------------------------- (b) replay
def test_stats(model, v12):
    """{task: (mean success, geometric-mean input tokens)} over every test evaluation of a version (paper inputs only)."""
    files = [f for f in glob.glob(str(C.EX / model / f"*/*/*/results/evaluations/test/{v12}-*/evaluation.json")) if model != "deepseek-flash" or S.paper_input(f)]
    seen = defaultdict(dict)
    for f in files:
        bench = f.split(model + "/")[1].split("/")[1]
        for e in json.loads(Path(f).read_text())["episodes"]:
            if not S.verified(e):
                continue
            seen[(bench, e["task_id"])][S.signature(e)] = (int(bool(e["success"])), tokens(e, f))
    out = defaultdict(dict)
    for (bench, t), runs in seen.items():
        ys = [y for y, _ in runs.values()]
        xs = [x for _, x in runs.values() if x]
        out[bench][t] = (float(np.mean(ys)), float(np.exp(np.mean(np.log(xs)))) if xs else None, len(ys))
    return out, len(files)


def test_compare(a, b):
    """b vs a on shared test tasks: success points difference and geometric mean token ratio."""
    tasks = [t for t in a if t in b]
    tx = [t for t in tasks if a[t][1] and b[t][1]]
    return {"tasks": len(tasks), "dS_pts": 100 * float(np.mean([b[t][0] - a[t][0] for t in tasks])) if tasks else None,
            "rho_tokens": float(np.exp(np.mean([math.log(b[t][1] / a[t][1]) for t in tx]))) if tx else None,
            "S_a": 100 * float(np.mean([a[t][0] for t in tasks])) if tasks else None, "S_b": 100 * float(np.mean([b[t][0] for t in tasks])) if tasks else None}


def full_panel_pool(index, names):
    """Versions scored on the run's main dev panel (the panel most versions share), in proposal order."""
    count = defaultdict(int)
    for v, evs in index.items():
        for k in evs:
            if k[0] == "dev":
                count[k] += 1
    key = max(count, key=count.get)
    pool = [v for v in index if key in index[v]]

    def order(v):
        n = names.get(v, "")
        if v == SEEDV:
            return (-1, 0)
        if n[:1] in "cs" and n[1:].isdigit():
            return (0, int(n[1:]))
        return (1, v)
    return key, sorted(pool, key=order)


def pool_rule(bench, key, pool, index, returned):
    evs = {v: load_eval(index[v][key]) for v in pool}
    tasks = sorted(set.intersection(*[set(t for t in ev if ev[t][1] and ev[t][0] is not None) for ev in evs.values()]))
    solved = {v: sum(evs[v][t][0] for t in tasks) for v in pool}
    full_solved = {v: sum(y for y, _ in evs[v].values() if y is not None) for v in pool}
    logx = {v: float(np.mean([math.log(evs[v][t][1]) for t in tasks])) for v in pool}
    c = crit(bench, len(evs[pool[0]]))
    best = max(full_solved.values())
    eligible = [v for v in pool if best - full_solved[v] < c]
    pick = min(eligible, key=lambda v: (round(logx[v], 12), pool.index(v)))
    # Z: anchor = the seed when it is eligible, else the optimizer's own pick (returned); switch only to an eligible version whose paired
    # token change against the anchor is distinguishable (z <= -1.96), the cheapest such
    anchor = SEEDV if SEEDV in eligible else returned
    sig = [v for v in eligible if v != anchor and compare(evs[anchor], evs[v])["z_L"] <= -Z]
    pick_z = min(sig, key=lambda v: logx[v]) if sig else anchor
    return {"panel": key[1], "tasks_tokens": len(tasks), "crit": c, "best": best, "solved": full_solved, "logx": logx,
            "eligible": eligible, "pick": pick, "anchor_z": anchor, "pick_z": pick_z}


def seq_gate(bench, steps, index, start=SEEDV, mode="P"):
    """steps = [(candidate vid, actual accepted, panel key or None)]; returns the replay trace."""
    cur, trace = start, []
    for cand, acc, key in steps:
        keys = [key] if key else []
        if not key:
            keys = [k for k in set(index.get(cur, {})) & set(index.get(cand, {})) if k[0] == "dev"]
        if not keys or key and (key not in index.get(cur, {}) or key not in index.get(cand, {})):
            trace.append({"cand": cand, "actual_accept": acc, "evaluable": False, "current": cur})
            continue
        k = keys[0]
        st = compare(load_eval(index[cur][k]), load_eval(index[cand][k]))
        c = crit(bench, st["n_y"])
        cheaper = st["L"] < 0 if mode == "P" else st["z_L"] <= -Z
        accept = st["D"] >= c or (st["D"] > -c and cheaper)
        trace.append({"cand": cand, "vs": cur, "actual_accept": acc, "rule_accept": bool(accept), "D": st["D"], "L": st["L"], "crit": c,
                      "n": st["n_y"], "evaluable": True})
        if accept:
            cur = cand
    return cur, trace


def replay(model):
    out = {}
    tests_cache = {}

    def tests(v):
        if v not in tests_cache:
            tests_cache[v] = test_stats(model, v)[0]
        return tests_cache[v]

    for opt in E.OPTIMIZERS:
        for bench in C.BENCHES:
            run, nodes, edges, index = lineage(model, opt, bench)
            summary = json.loads((run / "summary.json").read_text())
            returned = vid(summary["final_skill"])[:12]
            names = {vid(t)[:12]: n for n, t in nodes.items()}
            for f in sorted((run / "skillopt/skills").glob("skill_v*.md")) if opt == "skillopt" else []:
                names.setdefault(vid(f.read_text())[:12], "u" + str(int(f.stem.split("_v")[1])) + " (slow update)")
            names[SEEDV] = "seed"
            cell = {"returned": returned, "returned_name": names.get(returned)}
            key, pool = full_panel_pool(index, names)
            n_panel = len(load_eval(index[pool[0]][key]))
            full = n_panel == {"terminal-bench-2-1": 9}.get(bench, 30)
            if full and len(pool) > 2:
                pr = pool_rule(bench, key, pool, index, returned)
                pr["pool"] = [(v, names.get(v, "?")) for v in pool]
                if returned not in pr["solved"]:
                    pr["returned_missing"] = True
                cell["pool"] = pr
            # sequential gate
            if opt == "skillopt":
                steps = [(vid(nodes[e["child"]])[:12], e["accepted"], key) for e in edges]
                cur, trace = seq_gate(bench, steps, index)
                cell["seq"] = {"final": cur, "trace": trace}
                cur, trace = seq_gate(bench, steps, index, mode="Z")
                cell["seq_z"] = {"final": cur, "trace": trace}
            if opt == "skilladam" and bench != "terminal-bench-2-1":
                steps = []
                for e in edges:
                    if e["parent"] == "?":
                        continue
                    vp, vc = vid(nodes[e["parent"]])[:12], vid(nodes[e["child"]])[:12]
                    steps.append((vc, e["accepted"], None))
                cur, trace = seq_gate(bench, steps, index)
                cell["seq"] = {"final": cur, "trace": trace}
                cur, trace = seq_gate(bench, steps, index, mode="Z")
                cell["seq_z"] = {"final": cur, "trace": trace}
            # compare picks against returned and seed, dev (on the pool panel when possible) and test
            picks = {}
            if "pool" in cell:
                picks["pool"] = cell["pool"]["pick"]
                picks["pool_z"] = cell["pool"]["pick_z"]
            if "seq" in cell:
                picks["seq"] = cell["seq"]["final"]
                picks["seq_z"] = cell["seq_z"]["final"]
            cmp = {}
            for rule, v in picks.items():
                r = {"pick": v, "name": names.get(v, "?"), "same_as_returned": v == returned}
                if "pool" in cell and v in cell["pool"]["solved"] and returned in cell["pool"]["solved"]:
                    ev = lambda x: load_eval(index[x][key])
                    r["dev_vs_returned"] = compare(ev(returned), ev(v))
                    r["dev_vs_seed"] = compare(ev(SEEDV), ev(v)) if SEEDV in cell["pool"]["solved"] else None
                    r["dev_returned_vs_seed"] = compare(ev(SEEDV), ev(returned)) if SEEDV in cell["pool"]["solved"] else None
                else:
                    shared = [k for k in set(index.get(v, {})) & set(index.get(returned, {})) if k[0] == "dev"]
                    r["dev_vs_returned"] = compare(load_eval(index[returned][shared[0]]), load_eval(index[v][shared[0]])) if shared and v != returned else None
                    shared_s = [k for k in set(index.get(v, {})) & set(index.get(SEEDV, {})) if k[0] == "dev"]
                    r["dev_vs_seed"] = compare(load_eval(index[SEEDV][shared_s[0]]), load_eval(index[v][shared_s[0]])) if shared_s and v != SEEDV else None
                tv, tr, ts = tests(v), tests(returned), tests(SEEDV)
                r["has_test"] = bench in tv and len(tv[bench]) > 0
                r["test_execs"] = {"pick": sum(x[2] for x in tv.get(bench, {}).values()), "returned": sum(x[2] for x in tr.get(bench, {}).values()),
                                   "seed": sum(x[2] for x in ts.get(bench, {}).values())}
                if r["has_test"]:
                    r["test_vs_returned"] = test_compare(tr[bench], tv[bench]) if v != returned else None
                    r["test_vs_seed"] = test_compare(ts[bench], tv[bench]) if v != SEEDV else None
                if bench in tr and bench in ts:
                    r["test_returned_vs_seed"] = test_compare(ts[bench], tr[bench])
                cmp[rule] = r
            cell["picks"] = cmp
            out[f"{opt}|{bench}"] = cell
    return out


def edge_gate_counts(rows, mode="P"):
    """Per logged edge, the gate against its parent: refused accepted / newly accepted rejected. P: cheaper = L < 0; Z: z_L <= -1.96."""
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    cheaper = (lambda r: r["L"] < 0) if mode == "P" else (lambda r: r["z_L"] <= -Z)
    gate = lambda r: r["D"] >= r["crit"] or (r["D"] > -r["crit"] and cheaper(r))
    by = {}
    for opt in E.OPTIMIZERS:
        a = [r for r in acc if r["opt"] == opt]
        j = [r for r in rej if r["opt"] == opt]
        by[opt] = {"accepted": len(a), "refused": sum(not gate(r) for r in a), "rejected": len(j), "would_accept": sum(gate(r) for r in j)}
    kept = [r for r in acc if gate(r)]
    refused = [r for r in acc if not gate(r)]
    return {"accepted": len(acc), "refused": len(refused), "rejected": len(rej), "would_accept": sum(gate(r) for r in rej), "by_opt": by,
            "mean_L_refused": float(np.mean([r["L"] for r in refused])) if refused else None,
            "mean_L_kept": float(np.mean([r["L"] for r in kept])) if kept else None,
            "kept": [(r["opt"], r["bench"], r["parent"], r["child"], r["D"], round(r["L"], 3)) for r in kept],
            "newly": [(r["opt"], r["bench"], r["parent"], r["child"], r["D"], round(r["L"], 3)) for r in rej if gate(r)]}


def main():
    res = {}
    for model in TAGS:
        rows = edge_rows(model)
        res[model] = {"edges": rows, "summary": summarize_edges(rows), "gate": edge_gate_counts(rows), "gate_z": edge_gate_counts(rows, "Z"),
                      "by_bench_sd_l": {b: float(np.median([r["sd_l"] for r in rows if r["bench"] == b])) for b in C.BENCHES}}
    res["noise"] = noise_floor()
    for model in TAGS:
        try:
            res[model]["replay"] = replay(model)
        except Exception as exc:  # report, keep the rest
            import traceback
            traceback.print_exc()
            res[model]["replay_error"] = repr(exc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    for model in TAGS:
        print(model, "edges", len(res[model]["edges"]))
        for g, s in res[model]["summary"].items():
            print(" ", g, {k: v for k, v in s.items() if v is not None})
        print("  gate", {k: v for k, v in res[model]["gate"].items() if k not in ("kept", "newly")})
        print("  gate_z", {k: v for k, v in res[model]["gate_z"].items() if k not in ("kept", "newly")})
        print("  edge per-task sd of log ratio by bench", res[model]["by_bench_sd_l"])
    for b, n in res["noise"].items():
        print("noise", b, n["draws"], n["sd_task_log_ratio"], n["mean_abs_L"], n["se_L_mean"])


if __name__ == "__main__":
    main()
