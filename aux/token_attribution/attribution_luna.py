"""GPT-6-Luna version of the input-token attribution: the four-term split of each Luna learned skill's input-token change against the Luna seed (avg@3),
each term's share of the summed log-token increase, task-bootstrap intervals for those shares (Luna and the DeepSeek 73% headline), and the
within-task turn slopes of input tokens (output tokens and calls alongside).

Reuses src.analysis.attribution (mean_metrics, four_terms, run_files, load_runs) and src.analysis.episodes (metrics, fe_reg).
Luna episodes are read the Luna way (src.analysis.common.luna_row, the extraction behind Table 1's Luna rows): provider-failed requests carry no
usage and were retried, so they are removed before the episode's graph is built; R = completed requests, X = their input tokens, N = calls
not marked unexecuted. Repetitions are merged as `src.run repeat` does (full-panel base evaluation + unknown-only supplements; only verified outcomes).
Per task and side, R, N, X, F are averaged over the runs Table 1 averages (verified outcome, cost known); a task is dropped if any of those
runs has no usable graph metrics (R < 2, no call, no positive history) or the side has no run.

Run:  python -m aux.token_attribution.attribution_luna [record]
Writes OUT/token_attribution/attribution_luna.json and OUT/data/mech/mech_attribution_luna.md. No model calls.
"""
import collections
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from aux.headline_numbers.dynamics import known, slot, valid_slots
from src import layout
from src.analysis import attribution as A
from src.analysis import table1 as T
from src.analysis.episodes import metrics
from src.analysis.episodes import fe_reg
from src.graph import episode as W
from src.utils.dataloader import SEED_TEXT, vid
from src.utils import logger

OUT = layout.OUT / "token_attribution"
LUNA = layout.EXECUTION / "gpt-6-luna"
TAG = {"native-seed": "native-20261011", "gepa": "native-20261012", "skillopt": "native-20261012", "skilladam": "native-20261012"}
OPT = A.OPT  # (display, method)
BENCH2 = [("tau3-bench", "tau3"), ("automationbench", "AutoB")]
TERMS = A.TERMS
B = 10000
RNG = np.random.default_rng(20261020)


def luna_side(arm, bench):
    """{rep: {task: (episode dict, evaluation path)}} of verified-outcome episodes, merged as src.run.collect does."""
    run = LUNA / arm / bench / TAG[arm]
    out = LUNA / "test-reps" / f"{bench}__{arm}__{TAG[arm]}"
    out = out if out.is_dir() else None
    text = json.loads((run / "summary.json").read_text())["final_skill"] if (run / "summary.json").is_file() else SEED_TEXT
    v = vid(text)
    proto = run if (run / "native_protocol.json").is_file() else LUNA / "native-seed" / bench / TAG["native-seed"]
    tasks = [slot(r)[0] for r in json.loads((proto / "native_protocol.json").read_text())["panels"]["test"]]
    reps = {}
    for k in range(3):
        dirs = [run, out] if k == 0 else [out, run.parent / f"rep-{k}"]
        expected = collections.Counter((t, k) for t in tasks)
        base = None
        for d in dirs:
            if not d or not (d / "results/evaluations/test").is_dir():
                continue
            for p in sorted((d / "results/evaluations/test").glob("*/evaluation.json")):
                data = json.loads(p.read_text())
                if data.get("version_id") == v and data.get("split") == "test" and collections.Counter(slot(r) for r in data.get("panel", [])) == expected and valid_slots(data):
                    base = (p, data)
                    break
            if base:
                break
        got = {}
        if base:
            for e in base[1]["episodes"]:
                if known(e):
                    got[slot(e)] = (e, base[0])
        if out:
            for p in sorted((out / "results/evaluations/test").glob("*/evaluation.json")):
                data = json.loads(p.read_text())
                panel = [slot(r) for r in data.get("panel", [])]
                if data.get("version_id") != v or data.get("split") != "test" or not panel or not valid_slots(data) or not set(panel) <= set(expected):
                    continue
                for e in data.get("episodes", []):
                    if slot(e) in panel and slot(e) not in got and known(e):
                        got[slot(e)] = (e, p)
        reps[k] = {t: ev for (t, _), ev in got.items()}
    return dict(run=str(run), vid=v, is_seed=text == SEED_TEXT, tasks=tasks, reps=reps)


def luna_row(ep):
    """src.analysis.common.luna_row: Table 1's Luna per-episode N, X, R (None if no completed request)."""
    reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r["input_tokens"] is not None]
    calls = [c for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False]
    ok = len(reqs) > 0
    return dict(N=len(calls) if ok else None, X=sum(r["input_tokens"] for r in reqs) if ok else None, R=len(reqs) if ok else None,
                out=float(sum(r.get("output_tokens") or 0 for r in reqs)) if ok else None)


def _graph(args):
    ep, path = args
    ep2 = dict(ep, requests=[r for r in ep["requests"] if r["input_tokens"] is not None])
    if not ep2["requests"]:
        return None
    return W._episode((ep2, path))


GCACHE = OUT / "graph_cache.pkl"


def graphs(items):
    """{(path, task): Episode or None} with a pickle cache keyed by evaluation mtime."""
    cache = pickle.loads(GCACHE.read_bytes()) if GCACHE.exists() else {}
    todo = []
    for (ep, path) in items:
        key = (str(path), ep["task_id"], path.stat().st_mtime_ns)
        if key not in cache:
            todo.append((key, ep, path))
    if todo:
        with ProcessPoolExecutor(24) as pool:
            res = list(pool.map(_graph, [(e, p) for _, e, p in todo], chunksize=4))
        for (key, _, _), g in zip(todo, res):
            cache[key] = g
        tmp = GCACHE.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(cache))
        os.replace(tmp, GCACHE)
    return {(str(p), e["task_id"]): cache[(str(p), e["task_id"], p.stat().st_mtime_ns)] for e, p in items}


# Luna four-term split
def luna_cells():
    sides = {}
    for bench, _ in BENCH2 + [("terminal-bench-2-1", "TB2")]:
        sides[("native-seed", bench)] = luna_side("native-seed", bench)
        if bench != "terminal-bench-2-1":
            for _, m in OPT:
                sides[(m, bench)] = luna_side(m, bench)
    items = [ev for s in sides.values() for rep in s["reps"].values() for ev in rep.values()]
    G = graphs(items)
    # per side, per task: list of (rep, luna_row, graph metrics or None, Episode or None) over Table 1's runs (verified outcome + cost known)
    runs = {}
    mismatch = collections.Counter()
    for key, s in sides.items():
        per = collections.defaultdict(list)
        for k, rep in s["reps"].items():
            for t, (e, p) in rep.items():
                row = luna_row(e)
                if row["X"] is None:
                    continue
                g = G[(str(p), t)]
                if g is not None and (g.R != row["R"] or abs(g.X - row["X"]) > 0.5 or g.N != row["N"]):
                    mismatch[key] += 1
                per[t].append(dict(rep=k, row=row, ep=g, m=metrics(g) if g is not None else None, succ=bool(e["success"])))
        runs[key] = per
    cells = []
    for bench, short in BENCH2:
        seed = runs[("native-seed", bench)]
        for disp, m in OPT:
            own = runs[(m, bench)]
            # Table 1 Luna common set: per-task means over runs, both sides N, X > 0 (aux.headline_numbers.transfer.rho_geo)
            mean = lambda rs, key: sum(r["row"][key] for r in rs) / len(rs)
            common = [t for t in sorted(seed) if t in own and all(x > 0 for x in (mean(seed[t], "N"), mean(seed[t], "X"), mean(own[t], "N"), mean(own[t], "X")))]
            t1 = {t: math.log(mean(own[t], "X") / mean(seed[t], "X")) for t in common}
            dropped = collections.Counter()
            usable, pairs = [], []
            for t in common:
                if not all(r["m"] for r in seed[t] + own[t]):
                    dropped["a run without usable graph metrics (R<2, no call, no history, or unsplittable request)"] += 1
                    continue
                a = A.mean_metrics([r["m"] for r in seed[t]])
                b = A.mean_metrics([r["m"] for r in own[t]])
                if a is None or b is None:
                    dropped["no positive history on the run mean"] += 1
                    continue
                usable.append(t)
                pairs.append((a, b))
            terms, dX = A.four_terms(pairs)
            summ = A.summarize(terms, dX)
            nfull = sum(len(seed[t]) == 3 and len(own[t]) == 3 for t in usable)
            # C) skill tokens per request, skill minus seed (per-task mean over runs, then over tasks)
            def col(rs, c):
                return float(np.mean([float(r["ep"].Xrg[:, r["ep"].comps.index(c)].mean()) if c in r["ep"].comps else 0.0 for r in rs]))
            dskill = float(np.mean([col(own[t], "skill") - col(seed[t], "skill") for t in usable]))
            seedskill = float(np.mean([col(seed[t], "skill") for t in usable]))
            cells.append(dict(model="gpt-6-luna", method=disp, bench=bench, short=short, n=len(usable), n_table1=len(common), n_3runs=nfull,
                              dropped=dict(dropped), A=summ, terms={k: v.tolist() for k, v in terms.items()}, dX=dX.tolist(), tasks=usable,
                              t1_log_rho_full=float(np.mean(list(t1.values()))), t1_log_rho_used=float(np.mean([t1[t] for t in usable])),
                              d_skill_per_req=dskill, seed_skill_per_req=seedskill))
            print("Luna", disp, short, "n", len(usable), "/", len(common), {k: round(v, 4) for k, v in summ.items() if k != "largest"}, summ["largest"],
                  "T1 rho full", round(math.exp(cells[-1]["t1_log_rho_full"]), 3))
    return cells, sides, runs, G, dict(mismatch)


# DeepSeek per-task terms (same rule as attribution.compute)
def deepseek_cells():
    cells = []
    for disp, m in OPT:
        for bench, short in A.BENCH:
            own_files, seed_files = A.run_files(m, bench)
            own_eps, own_rows = A.load_runs(own_files)
            seed_eps, seed_rows = A.load_runs(seed_files)
            reps = sorted(own_files)
            _, own_mean = T.mean_rows(own_files)
            _, seed_mean = T.mean_rows(seed_files)
            t1 = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
            usable, pairs = [], []
            for t in t1:
                if not all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None for rows in (own_rows, seed_rows) for r in reps):
                    continue
                if not all(t in eps[r] and eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
                    continue
                a = A.mean_metrics([seed_eps[r][t][1] for r in reps])
                b = A.mean_metrics([own_eps[r][t][1] for r in reps])
                if a is None or b is None:
                    continue
                usable.append(t)
                pairs.append((a, b))
            terms, dX = A.four_terms(pairs)
            cells.append(dict(model="deepseek-flash", method=disp, bench=bench, short=short, n=len(usable), A=A.summarize(terms, dX),
                              terms={k: v.tolist() for k, v in terms.items()}, dX=dX.tolist(), tasks=usable))
    return cells


# shares and bootstrap
def shares_point(cells, weighted=False):
    """Shares of the summed log-token increase: sum over skills of the per-skill task means (the paper's 73% / 21%), or, weighted=True,
    sum over all (skill, task) pairs (each task of each skill counts once)."""
    agg = (lambda c, k: float(np.sum(c["dX"] if k == "total" else c["terms"][k]))) if weighted else (lambda c, k: c["A"][k])
    tot = sum(agg(c, "total") for c in cells)
    s = {k: sum(agg(c, k) for c in cells) for k in TERMS}
    turns = s["more_actions"] + s["turn_splitting"]
    gross = sum(abs(agg(c, k)) for c in cells for k in TERMS)
    gross_turn = sum(abs(agg(c, k)) for c in cells for k in ("more_actions", "turn_splitting"))
    gross_text = sum(abs(agg(c, "skill_text")) for c in cells)
    return dict(total=tot, sums=s, turns=turns, share_turns=turns / tot, share_text=s["skill_text"] / tot, share_content=s["content"] / tot,
                share_actions=s["more_actions"] / tot, share_split=s["turn_splitting"] / tot,
                gross_share_turns=gross_turn / gross, gross_share_text=gross_text / gross)


def bootstrap(cells, weighted=False):
    """Resample tasks with replacement within each benchmark, the same draw for every skill of that benchmark (they share the seed);
    each cell averages its terms over the drawn tasks it uses. Returns percentile intervals of the summed shares."""
    by_bench = collections.defaultdict(list)
    for c in cells:
        by_bench[c["bench"]].append(c)
    keys = ["more_actions", "turn_splitting", "content", "skill_text", "total"]
    acc = {k: np.zeros(B) for k in keys}
    for bench, cs in by_bench.items():
        pool = sorted({t for c in cs for t in c["tasks"]})
        n = len(pool)
        counts = RNG.multinomial(n, np.full(n, 1 / n), size=B)  # B x n
        for c in cs:
            idx = np.array([pool.index(t) for t in c["tasks"]])
            w = counts[:, idx].astype(float)
            wsum = w.sum(1)
            wsum[wsum == 0] = np.nan
            for k in keys:
                v = np.array(c["dX"] if k == "total" else c["terms"][k])
                acc[k] += (w @ v) if weighted else (w @ v) / wsum
    turns = acc["more_actions"] + acc["turn_splitting"]
    tot = acc["total"]
    out = {}
    for name, num in (("share_turns", turns), ("share_text", acc["skill_text"]), ("share_content", acc["content"]),
                      ("share_actions", acc["more_actions"]), ("share_split", acc["turn_splitting"])):
        r = num / tot
        out[name] = [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]
    out["total"] = [float(np.nanpercentile(tot, 2.5)), float(np.nanpercentile(tot, 97.5))]
    out["turns_sum"] = [float(np.nanpercentile(turns, 2.5)), float(np.nanpercentile(turns, 97.5))]
    out["text_sum"] = [float(np.nanpercentile(acc["skill_text"], 2.5)), float(np.nanpercentile(acc["skill_text"], 97.5))]
    out["p_total_le_0"] = float(np.mean(tot <= 0))
    out["p_turns_gt_text"] = float(np.mean(turns > acc["skill_text"]))
    out["p_share_turns_gt_half"] = float(np.mean(turns / tot > 0.5))
    return out


# slopes
def luna_slopes(sides, runs, G):
    """Within-task FE slopes of log input tokens / output tokens / calls on log turns. Design (i) as DeepSeek's mech_slopes_input: run 0 of the
    seed and of each learned skill, one episode per arm and task; (ii) all three runs of every arm. TB2: the seed only (all optimizers returned it),
    so only (ii)."""
    res = {}
    for bench, short in BENCH2 + [("terminal-bench-2-1", "TB2")]:
        arms = ["native-seed"] + ([m for _, m in OPT] if bench != "terminal-bench-2-1" else [])
        for design in ("run0", "all3"):
            recs = []
            for arm in arms:
                for t, rs in runs[(arm, bench)].items():
                    for r in rs:
                        if design == "run0" and r["rep"] != 0:
                            continue
                        if r["m"]:
                            recs.append((t, r["m"]["R"], r["m"]["X"], r["row"]["out"], r["m"]["N"]))
            if design == "run0" and bench == "terminal-bench-2-1":
                continue
            row = {}
            for q, j in (("input", 2), ("output", 3), ("calls", 4)):
                its = [(t, R, y) for t, R, *ys in recs for y in [ys[j - 2]] if y and y > 0]
                # tasks with a single record carry no within-task information; fe_reg demeans them to 0, which only adds to dof
                cnt = collections.Counter(t for t, _, _ in its)
                its = [x for x in its if cnt[x[0]] > 1]
                r, _ = fe_reg(its)
                row[q] = r
            res[f"{short}|{design}"] = row
            print("slope", short, design, {q: (round(v["slope"], 3), round(v["se"], 3), v["n"], v["n_tasks"]) for q, v in row.items()})
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lc, sides, runs, G, mismatch = luna_cells()
    dc = deepseek_cells()
    # check DeepSeek against mech_attribution3 (A)
    for c in dc:
        print("DS", c["method"], c["short"], c["n"], {k: round(c["A"][k], 3) for k in TERMS + ["total"]})
    sl = luna_slopes(sides, runs, G)
    pos = [c for c in lc if c["A"]["total"] > 0]
    res = dict(luna=lc, deepseek=[{k: v for k, v in c.items()} for c in dc], mismatch=mismatch, slopes=sl,
               shares=dict(deepseek=shares_point(dc), luna=shares_point(lc), luna_pos=shares_point(pos),
                           deepseek_plus_luna=shares_point(dc + lc)),
               boot=dict(deepseek=bootstrap(dc), luna=bootstrap(lc), deepseek_plus_luna=bootstrap(dc + lc)),
               pos_cells=[(c["method"], c["short"]) for c in pos])
    for nm, cs in (("deepseek", dc), ("luna", lc)):
        res["shares"][nm + "_weighted"] = shares_point(cs, weighted=True)
        res["boot"][nm + "_weighted"] = bootstrap(cs, weighted=True)
        for bench in sorted({c["short"] for c in cs}):
            sub = [c for c in cs if c["short"] == bench]
            res["shares"][f"{nm}|{bench}"] = shares_point(sub)
            res["boot"][f"{nm}|{bench}"] = bootstrap(sub)
    # per-skill shares with bootstrap
    res["boot_cell"] = {f'{c["method"]}|{c["short"]}': bootstrap([c]) for c in lc}
    res["boot_cell_ds"] = {f'{c["method"]}|{c["short"]}': bootstrap([c]) for c in dc}
    (OUT / "attribution_luna.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["shares"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "sums"})
    for k, v in res["boot"].items():
        print(k, v)
    print("mismatch", mismatch)
    record()


# record
PAPER_RHO = {("GEPA", "tau3"): 1.04, ("SkillOpt", "tau3"): 1.01, ("SkillAdam", "tau3"): 0.98,
             ("GEPA", "AutoB"): 1.07, ("SkillOpt", "AutoB"): 1.25, ("SkillAdam", "AutoB"): 0.93}  # paper Table 1, GPT-6-Luna panel


def ci(x, nd=3, pct=False):
    if pct:
        return f"[{100 * x[0]:.0f}%, {100 * x[1]:.0f}%]"
    return f"[{x[0]:+.{nd}f}, {x[1]:+.{nd}f}]"


def record():
    r = json.loads((OUT / "attribution_luna.json").read_text())
    lab = A.TERM_LABEL
    f = lambda x: f"{x:+.3f}"
    rowsA = []
    for c in r["luna"]:
        a = c["A"]
        rowsA.append([c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f(a[k]) for k in TERMS] + [f(a["total"]), f"{math.exp(c['t1_log_rho_full']):.3f} ({PAPER_RHO[(c['method'], c['short'])]:.2f})",
                     f(a["residual_mean"]), f"{a['residual_max_abs']:.3f}", lab[a["largest"]]])
    rowsA2 = []
    for c in r["luna"]:
        b = r["boot_cell"][f"{c['method']}|{c['short']}"]
        a = c["A"]
        rowsA2.append([c["method"], c["short"], f"{f(a['total'])} {ci(b['total'])}", f"{f(a['more_actions'] + a['turn_splitting'])} {ci(b['turns_sum'])}",
                       f"{f(a['skill_text'])} {ci(b['text_sum'])}", f"{b['p_total_le_0']:.2f}", f"{c['d_skill_per_req']:+.0f} (seed {c['seed_skill_per_req']:.0f})"])
    S, Bt = r["shares"], r["boot"]
    order = [("deepseek", "DeepSeek, 9 skills (paper headline)"), ("deepseek_weighted", "DeepSeek, 9 skills, task-weighted"),
             ("deepseek|tau3", "DeepSeek, tau3"), ("deepseek|AutoB", "DeepSeek, AutoB"), ("deepseek|TB2", "DeepSeek, TB2"),
             ("luna", "Luna, 6 skills (tau3, AutoB)"), ("luna_weighted", "Luna, 6 skills, task-weighted"),
             ("luna|tau3", "Luna, tau3"), ("luna|AutoB", "Luna, AutoB"),
             ("luna_pos", "Luna, the 4 skills with total > 0 (no bootstrap)"),
             ("deepseek_plus_luna", "DeepSeek + Luna, 15 skills")]
    rowsB = []
    for key, name in order:
        s = S[key]
        b = Bt.get(key)
        tot = f"{s['total']:+.3f}" + (f" {ci(b['total'])}" if b and "weighted" not in key else "")
        tr = f"{s['turns']:+.3f}" + (f" {ci(b['turns_sum'])}" if b and "weighted" not in key else "")
        rowsB.append([name, tot, tr, f"{s['sums']['skill_text']:+.3f}", f"{s['sums']['content']:+.3f}",
                      f"{100 * s['share_turns']:.0f}%" + (f" {ci(b['share_turns'], pct=True)}" if b else ""),
                      f"{100 * s['share_text']:.0f}%" + (f" {ci(b['share_text'], pct=True)}" if b else ""),
                      f"{100 * s['share_content']:.0f}%", f"{100 * s['gross_share_turns']:.0f}% / {100 * s['gross_share_text']:.0f}%",
                      f"{b['p_turns_gt_text']:.3f}" if b else "-", f"{b['p_share_turns_gt_half']:.3f}" if b else "-", f"{b['p_total_le_0']:.3f}" if b else "-"])
    rowsBds = []
    for c in r["deepseek"]:
        b = r["boot_cell_ds"][f"{c['method']}|{c['short']}"]
        a = c["A"]
        rowsBds.append([c["method"], c["short"], c["n"], f"{f(a['total'])} {ci(b['total'])}", f"{f(a['more_actions'] + a['turn_splitting'])} {ci(b['turns_sum'])}",
                        f"{100 * (a['more_actions'] + a['turn_splitting']) / a['total']:.0f}% {ci(b['share_turns'], pct=True)}"])
    sl = r["slopes"]
    DS = {"tau3": (1.349, 0.046, 300, 75, 1.025, 0.840), "AutoB": (1.556, 0.040, 480, 120, 0.881, 0.853), "TB2": (1.584, 0.064, 71, 18, 0.498, 0.936)}
    rowsC = []
    for key, design in (("tau3|run0", "run 0 of seed + 3 skills"), ("tau3|all3", "3 runs of seed + 3 skills"), ("AutoB|run0", "run 0 of seed + 3 skills"),
                        ("AutoB|all3", "3 runs of seed + 3 skills"), ("TB2|all3", "3 runs of the seed (all optimizers returned it)")):
        v = sl[key]
        b = key.split("|")[0]
        d = DS[b]
        rowsC.append([b, design, f"{v['input']['slope']:.3f} ({v['input']['se']:.3f})", f"{v['output']['slope']:.3f} ({v['output']['se']:.3f})",
                      f"{v['calls']['slope']:.3f} ({v['calls']['se']:.3f})", f"{v['input']['n']} / {v['input']['n_tasks']}",
                      f"{d[0]:.3f} ({d[1]:.3f}); out {d[4]:.3f}; calls {d[5]:.3f}; {d[2]} / {d[3]}"])
    what = ("The input-token attribution extended to GPT-6-Luna. (A) Four-term split (Eq. split: more actions, turn-splitting, content per turn, skill text) of each Luna learned skill's change in "
            "log input tokens against the Luna seed, per task on avg@3 run means, exactly the computation of mech_attribution3 (src.analysis.attribution.mean_metrics / four_terms). "
            "(B) Each term's share of the summed log-token increase (sum over skills of the per-skill task means, the definition behind the paper's 73% / 21%), with task-bootstrap "
            "95% percentile intervals for DeepSeek and Luna. (C) Within-task turn slopes of input tokens (output tokens, tool calls alongside), as mech_slopes_input. "
            "Luna TB2 is excluded from A and B: all three optimizers returned the seed, so every term is 0.")
    runs = ("GPT-6-Luna: seed native-seed native-20261011; GEPA, SkillOpt, SkillAdam native-20261012; "
            "rep 0 from the run directory, reps 1-2 and unknown-only supplements from test-reps/<bench>__<arm>__<tag>, merged as src.run repeat does (only verified outcomes; a supplement "
            "fills a slot only if it is still unknown) -- the same merge and episode extraction as Table 1's Luna panel (aux.headline_numbers.transfer). "
            "DeepSeek-V4.1-Flash: as mech_attribution3 (skills native-20261008-https + test-reps; seed native-20261003-trace-final + rep-1/rep-2), recomputed here per task for the bootstrap.")
    agg = ("Luna episodes: provider-failed requests (no usage, retried) are removed before the episode graph is built, so R = completed requests, X = their input tokens, N = calls "
           "not marked unexecuted; graph R, N, X equal Table 1's Luna extraction on every episode (0 mismatches). Per task and side, R, N, X, F are averaged over the runs Table 1 "
           "averages (verified outcome, cost known); tasks: Table 1's Luna common set (both sides N, X > 0), all of which have usable graph metrics (R>=2, a call, positive history) "
           "in every run (no task dropped; 3 runs on both sides for every task except one GEPA AutoB task). exp(total) reproduces the paper's Luna rho. "
           f"Bootstrap: {B} draws, tasks resampled with replacement within each benchmark, the same draw for the three skills of a benchmark (they share the seed); each skill "
           "averages its per-task terms over the drawn tasks; shares = summed numerator / summed total per draw; percentile intervals. 'Task-weighted' sums over all (skill, task) "
           "pairs instead of over per-skill means, so AutoBench (120 tasks) dominates. 'Gross' = sum of |term| over skills (each term's absolute size, signs ignored). "
           "Slopes: within-task fixed-effects regression (src.analysis.episodes.fe_reg) of log y on log turns; tasks with one episode drop out. 'run 0' pools one episode per arm and task as "
           "DeepSeek's mech_slopes_input does; Luna output tokens are summed over completed requests.")
    table = ("### A. Luna four-term split, avg@3 run means (log units; n = tasks used / Table 1 common tasks)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "more actions", "turn-splitting", "content per turn", "skill text", "total", "rho = exp(total), full set (paper Table 1)",
                                "resid mean", "resid max |.|", "largest"], rowsA)
             + "\n\n### A2. Luna per skill: total, turn terms (more actions + turn-splitting) and skill text with 95% task-bootstrap intervals; skill tokens per request\n\n"
             + logger.md_table(["Skill", "Bench", "total", "turn terms", "skill text", "P(total <= 0)", "skill tokens / request, skill minus seed"], rowsA2)
             + "\n\n### B. Shares of the summed log-token increase (95% task-bootstrap intervals)\n\n"
             + logger.md_table(["Set", "summed total", "turn terms", "skill text", "content per turn", "turn share", "text share", "content share",
                                "gross share turns / text", "P(turns > text)", "P(turn share > 50%)", "P(total <= 0)"], rowsB)
             + "\n\nDeepSeek per skill (turn share of the skill's own total):\n\n"
             + logger.md_table(["Skill", "Bench", "n", "total", "turn terms", "turn share"], rowsBds)
             + "\n\n### C. Within-task turn slopes, GPT-6-Luna (slope (SE)); DeepSeek reference from mech_slopes / mech_slopes_input\n\n"
             + logger.md_table(["Bench", "episodes pooled", "input tokens", "output tokens", "tool calls", "episodes / tasks", "DeepSeek input (SE); output; calls; episodes / tasks"], rowsC))
    s_ds, s_lu = S["deepseek"], S["luna"]
    b_ds, b_lu = Bt["deepseek"], Bt["luna"]
    extra = "\n".join([
        "## Reading",
        "",
        f"- DeepSeek headline: turn terms {100 * s_ds['share_turns']:.0f}% {ci(b_ds['share_turns'], pct=True)}, skill text {100 * s_ds['share_text']:.0f}% {ci(b_ds['share_text'], pct=True)} of the summed increase; "
        f"turns exceed text in {100 * b_ds['p_turns_gt_text']:.0f}% of draws and carry more than half in {100 * b_ds['p_share_turns_gt_half']:.0f}%. The width comes from TB2 (15-17 tasks per skill, "
        f"per-task sd of the turn terms 0.54-0.66): AutoBench alone gives {100 * S['deepseek|AutoB']['share_turns']:.0f}% {ci(Bt['deepseek|AutoB']['share_turns'], pct=True)}, "
        f"the task-weighted share is {100 * S['deepseek_weighted']['share_turns']:.0f}% {ci(Bt['deepseek_weighted']['share_turns'], pct=True)}; on tau3 turns and text are even "
        f"({100 * S['deepseek|tau3']['share_turns']:.0f}% vs {100 * S['deepseek|tau3']['share_text']:.0f}%).",
        f"- Luna: the six skills raise log input tokens by {s_lu['total']:+.3f} summed {ci(b_lu['total'])} (two skills, SkillAdam on both benchmarks, lower them). Turn terms add "
        f"{s_lu['turns']:+.3f} {ci(b_lu['turns_sum'])}, skill text {s_lu['sums']['skill_text']:+.3f} {ci(b_lu['text_sum'])}, content per turn {s_lu['sums']['content']:+.3f}. "
        f"Because content per turn falls, the turn terms exceed the net increase ({100 * s_lu['share_turns']:.0f}% of it, {ci(b_lu['share_turns'], pct=True)}) and a share is a poor summary; "
        f"in absolute size the turn terms are {100 * s_lu['gross_share_turns']:.0f}% of all term movement (DeepSeek {100 * s_ds['gross_share_turns']:.0f}%) and the text "
        f"{100 * s_lu['gross_share_text']:.0f}% (DeepSeek {100 * s_ds['gross_share_text']:.0f}%). Turns exceed text in {100 * b_lu['p_turns_gt_text']:.1f}% of draws. "
        f"The net is carried by AutoBench ({S['luna|AutoB']['total']:+.3f}); on tau3 the summed change {S['luna|tau3']['total']:+.3f} {ci(Bt['luna|tau3']['total'])} is not distinguishable "
        f"from zero and its largest single term is GEPA's skill text.",
        "- Largest term per Luna skill: turn-splitting or more actions for all three AutoBench skills and SkillOpt on tau3 (whose terms are all below 0.02); skill text for GEPA on tau3 "
        "(the only Luna skill whose per-request skill text grows by more than the seed's own length); content per turn for SkillAdam on tau3 (net -0.019).",
        "- Turn slopes: Luna input tokens rise with turns at 1.44 (tau3), 1.56 (AutoB) and 1.52 (TB2, seed runs), inside DeepSeek's 1.35-1.58. Tool calls stay at or below 1.06. "
        "Output tokens do not all stay below 1.03: on Luna tau3 the output slope is 1.31 (run 0) / 1.13 (three runs).",
        "",
        "In one sentence each: DeepSeek -- \"together they carry 73% of the summed log-token increase (95% task-bootstrap interval "
        f"{ci(b_ds['share_turns'], pct=True)}), the skill text 21%\" -- and for Luna: \"Under GPT-6-Luna the tax is small (pooled 1.05), yet the extra turns again move it most: "
        f"turn terms add {s_lu['turns']:+.2f} to the summed log-token increase {ci(b_lu['turns_sum'], nd=2)} and the skill text {s_lu['sums']['skill_text']:+.2f}, while content per turn falls "
        f"({s_lu['sums']['content']:+.2f}); input tokens rise with turns at slopes of 1.44 to 1.56.\"",
    ])
    logger.write_record("attribution_luna", what, runs, agg, table,
                        "python -m aux.token_attribution.attribution_luna  # compute\n"
                        "python -m aux.token_attribution.attribution_luna record",
                        extra)


if __name__ == "__main__":
    record() if sys.argv[1:2] == ["record"] else main()
