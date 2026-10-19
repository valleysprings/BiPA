"""Within-task slopes of input tokens on turns and the Sec. 3 elasticity checks (DeepSeek test episodes, recorded, no model calls). Three subcommands:

  (bare)    episode-level slope tables (appendix slope tables): slope of log X on log R by sample and skill, the gap decomposition
            slope = e_R + s_H b_h + s_F b_F with a counterfactual at task-mean content, the share of input in the last quarter of turns,
            turns vs success, where optimized skills' extra turns come from (calls per turn), and the per-turn mechanism of the seed
            -> assets/cache/analysis/slopes/slopes.txt
  checks [--model deepseek-flash] [--out DIR]   (merged from slopes_checks.py)
            the Sec. 3 numbers Fig. 1 and the appendix costsplit text rest on: within-task slopes (Fig. 1 panel (a) values), 1 + mean s_H
            and the exact e_R, the eq:split decomposition per skill with R^2 (limit vs exact e_R, Fig. 1 panel (b) readouts), and a second
            per-skill fit of d log X by t_N, t_m, t_h, t_F with k = s_F + 2 s_H (supplementary)
            -> assets/cache/analysis/slopes_checks/checks.json, fit.json
  reruns [--out DIR]   (merged from reruns.py)
            rerun checks of the elasticity analysis: per-episode e_R ranges, within-task slopes (task FE on the main runs, task+skill FE,
            seed-only over its three runs, task x skill over all reruns), per-run and pooled success, and the seed's three-run spread in
            mean log input tokens (the lever-test criterion) -> assets/cache/analysis/reruns/check.json, seed_spread.json

This module also holds the shared episode library (merged from episodes.py): ep_rows (the former episodes.episode_rows, renamed so the
checks-stage episodes(path) loader below keeps its name), fe, samp, build_rows, derive, test_files; pricing.py imports ep_rows/fe/samp from
here. metrics, episodes and METHODS3 are the per-episode contract used by appc.py and the src.analysis.rq* modules. Episode quantities (per task):
R requests, N executed calls, X input tokens, F = mean over requests of the fixed components (system, tool schemas, task, skill),
h = (X - R F) / (R (R - 1) / 2) the content added per request, s_F = R F / X, e_R = 1 + (1 - s_F) R / (R - 1).
Run: <bach-harness venv python> -m src.analysis.appendix.slopes [checks|reruns] [flags]   (from bipa/)"""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np
from src.utils import dataloader as C
from src.utils.dataloader import BN, BENCHES, DS, EX, OPTIMIZERS, SEED_TEXT, current_run, eval_file, final_skill, gmean, out_dir, returned_skills, vid
from src.utils.logger import Report
from src.graph import episode as W
from src.analysis.appendix.pricing import price


# ---- episode rows (merged from episodes.py; used by the bare slope tables and by pricing.py) ----------------------------------------

FIXED = {"system", "tool_schemas", "task", "skill"}  # the components every request re-sends
OPTS = ("gepa", "skillopt", "skilladam")


def test_files(bench):
    """[(skill, rep, evaluation.json)] of the seed (main + two reruns) and of each optimizer's returned skill (main + reruns), returned seeds left out."""
    out = []
    sv = vid(SEED_TEXT)
    for m in ("native-seed",) + OPTS:
        run = current_run(m, "deepseek-flash", bench)
        if run is None: continue
        t = final_skill(m, run)
        f = eval_file(run, "test", vid(t))
        if m != "native-seed" and vid(t) == sv: continue
        if f: out.append(("seed" if m == "native-seed" else m, "main", str(f)))
    for r in ("rep-1", "rep-2"):
        f = eval_file(DS/"native-seed"/bench/r, "test", sv)
        if f: out.append(("seed", r, str(f)))
    for m in ("skillopt", "skilladam", "gepa"):
        for r in ("rep-1", "rep-2"):
            run = DS/m/bench/r
            if run.is_dir():
                f = eval_file(run, "test", vid(final_skill(m, current_run(m, "deepseek-flash", bench))))
                if f: out.append((m, r, str(f)))
    return out


def build_rows():
    rows = []
    for b in BENCHES:
        for skill, rep, f in test_files(b):
            d = json.load(open(f)); raw = {}
            for e in d["episodes"]: raw.setdefault((e["task_id"], e.get("repetition", 0)), e)
            for we in W.load(f):
                e = raw.get((we.task, 0)) or next((v for kk, v in raw.items() if kk[0] == we.task), None)
                seen = set(); reqs = []
                for r in e["requests"]:
                    if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
                if len(reqs) != we.R: continue
                idx = [i for i, c in enumerate(we.comps) if c in FIXED]
                Fr = we.Xrg[:, idx].sum(1) if idx else we.req_in[:1].repeat(we.R)
                cache = np.array([min((r.get("captured_usage") or {}).get("cache_tokens") or 0, r["input_tokens"]) for r in reqs], float)
                out = np.array([r["output_tokens"] for r in reqs], float)
                ncall = np.array([len(r.get("tool_calls", [])) for r in reqs])
                rows.append(dict(bench=b, skill=skill, rep=rep, task=we.task, succ=we.success, R=we.R, N=we.N, req_in=we.req_in, Fr=Fr, cache=cache, out=out, ncall=ncall))
    return rows


def derive(r):
    """X total input, F mean fixed prefix, h new content per turn, sH history share, eR = 1 + sH R/(R-1) structural elasticity, output total, cached tokens."""
    R = r["R"]; X = r["req_in"].sum(); F = r["Fr"].mean()
    r["X"] = X; r["F"] = F; r["h"] = 2 * (X - R * F) / (R * (R - 1)) if R > 1 else np.nan
    r["sH"] = 1 - R * F / X; r["eR"] = 1 + r["sH"] * R / (R - 1) if R > 1 else np.nan
    r["out_tot"] = r["out"].sum(); r["res"] = r["cache"][1:].sum()  # request 1 re-sends nothing
    r["res_all"] = r["cache"].sum()


def ep_rows():
    """Rows of all benches with derived fields; episodes with a single turn or no history are dropped.
    (The episode_rows of the former episodes.py, renamed here so the checks-stage episodes(path) loader below keeps its name.)"""
    rows = build_rows()
    for r in rows: derive(r)
    return [r for r in rows if r["R"] >= 2 and r["N"] >= 1 and r["X"] - r["R"] * r["F"] > 0 and r["F"] > 0]


def fe(rs, y, x):
    """within-task fixed-effect slope of y on x (and its se) over the episodes rs."""
    ys = collections.defaultdict(list)
    for r in rs: ys[r["task"]].append((y(r), x(r)))
    num = den = ssr = 0
    for v in ys.values():
        a = np.array(v); a = a - a.mean(0); num += (a[:, 0] * a[:, 1]).sum(); den += (a[:, 1] ** 2).sum()
    b = num / den
    for v in ys.values():
        a = np.array(v); a = a - a.mean(0); ssr += ((a[:, 0] - b * a[:, 1]) ** 2).sum()
    return b, np.sqrt(ssr / (sum(len(v) for v in ys.values()) - len(ys) - 1) / den)


def samp(rows, b, kind):
    """episodes of bench b: main = first runs of every skill (SkillAdam TB2 excluded), seed = seed runs, opt = optimized skills, all."""
    out = []
    for r in rows:
        if r["bench"] != b: continue
        if kind == "seed" and r["skill"] == "seed": out.append(r)
        elif kind == "main" and r["rep"] == "main" and not (r["skill"] == "skilladam" and b == "terminal-bench-2-1"): out.append(r)
        elif kind == "opt" and r["skill"] != "seed": out.append(r)
        elif kind == "all": out.append(r)
    return out


# ---- Sec. 3 per-skill decomposition and its checks (merged from slopes_checks.py; contract for appc.py and the src.analysis.rq* modules) ------------

METHODS3 = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}


def metrics(ep):
    """Per-episode R, N, X, F, h, m, s_F, e_R and success; None for episodes with R < 2, no calls, or no positive history/fixed part."""
    if ep.R < 2 or ep.N < 1: return None
    idx = [i for i, c in enumerate(ep.comps) if c in W.FIXED]
    F = float(ep.Xrg[:, idx].sum(1).mean()) if idx else float(ep.req_in[0])
    X = ep.X; Hh = X - ep.R * F
    if Hh <= 0 or F <= 0: return None
    sF = ep.R * F / X
    return dict(R=ep.R, N=ep.N, X=X, F=F, h=Hh / (ep.R * (ep.R - 1) / 2), m=ep.N / ep.R, sF=sF, eR=1 + (1 - sF) * ep.R / (ep.R - 1), succ=ep.success)


def episodes(path): return {e.task: e for e in W.load(path)}


def fe_slope(rs, ykey, groups):
    """Within-group (task fixed effects) slope of log y on log R over records (_, _, task, metrics); groups with one record drop out."""
    x = np.log([m["R"] for *_, m in rs]); y = np.log([m[ykey] for *_, m in rs])
    g = collections.defaultdict(list)
    for i, r in enumerate(rs): g[groups(r)].append(i)
    g = {k: i for k, i in g.items() if len(i) > 1}; idx = [i for v in g.values() for i in v]
    xs, ys = x.copy(), y.copy()
    for ii in g.values(): xs[ii] -= x[ii].mean(); ys[ii] -= y[ii].mean()
    xs, ys = xs[idx], ys[idx]
    return dict(slope=round(float((xs * ys).sum() / (xs * xs).sum()), 3), n=len(xs), groups=len(g))


def checks(model):
    """Sec. 3 checks per benchmark: slopes, 1 + mean s_H, e_R, and the per-skill decomposition (eq:split)."""
    out, skills = {}, []
    for bench in C.BENCHES:
        seedrun = C.current_run("native-seed", model, bench)
        seedf = C.eval_file(seedrun, "test", C.vid(C.SEED_TEXT))
        main = {"seed": seedf}; texts = {"seed": C.SEED_TEXT}; commons = {}
        rs = C.returned_skills(model, bench, METHODS3)
        for disp, (run, t) in rs.items():
            texts[disp] = t; main[disp] = C.eval_file(run, "test", C.vid(t))
            ra, rb = C.evaluation(run, "test", C.vid(t))[0], C.evaluation(seedrun, "test", C.vid(C.SEED_TEXT))[0]
            commons[disp] = {k[0] for k, r in ra.items() if k in rb and r["success"] is not None and rb[k]["success"] is not None and all(z and z > 0 for z in (r["N"], r["X"], rb[k]["N"], rb[k]["X"]))}
        reps = collections.defaultdict(list)
        for d, f in main.items():
            if f: reps[d].append(("main", f))
        for d in texts:  # the repetitions rep-1, rep-2 of the same skill (seed x3 slope)
            meth = "native-seed" if d == "seed" else METHODS3[d]
            for r in ("rep-1", "rep-2"):
                dd = C.EX / model / meth / bench / r
                if dd.is_dir():
                    f = C.eval_file(dd, "test", C.vid(texts[d]))
                    if f: reps[d].append((r, f))
        recs = []; E = {}
        for d, lst in reps.items():
            for rn, f in lst:
                ee = episodes(f); E[(d, rn)] = ee
                for t, e in ee.items():
                    m = metrics(e)
                    if m: recs.append((d, rn, t, m))
        mainrecs = [r for r in recs if r[1] == "main"]
        sH = np.array([1 - m["sF"] for *_, m in mainrecs]); eR = np.array([m["eR"] for *_, m in mainrecs]); Xw = np.array([m["X"] for *_, m in mainrecs])
        seedrecs = [r for r in recs if r[0] == "seed"]
        out[bench] = dict(
            slope_main=fe_slope(mainrecs, "X", lambda r: r[2]), slope_seed3=fe_slope(seedrecs, "X", lambda r: r[2]),
            slope_h_main=fe_slope(mainrecs, "h", lambda r: r[2]), slope_F_main=fe_slope(mainrecs, "F", lambda r: r[2]),
            one_plus_mean_sH=round(1 + float(sH.mean()), 3), mean_exact_eR=round(float(eR.mean()), 3),
            one_plus_Xweighted_sH=round(1 + float((sH * Xw).sum() / Xw.sum()), 3),
            eR_main_range=[round(float(eR.min()), 3), round(float(eR.max()), 3)],
            eR_all_range=[round(float(min(m["eR"] for *_, m in recs)), 3), round(float(max(m["eR"] for *_, m in recs)), 3)], n_all=len(recs),
            R_median=float(np.median([m["R"] for *_, m in mainrecs])),
            mean_R_over_Rminus1=round(float(np.mean([m["R"] / (m["R"] - 1) for *_, m in mainrecs])), 3),
            tasks_with_2_main=sum(1 for c in collections.Counter(r[2] for r in mainrecs).values() if c > 1))
        S = E[("seed", "main")]
        for d in METHODS3:
            if (d, "main") not in E: continue
            Cs = E[(d, "main")]; rows = []
            for t, ec in Cs.items():
                a = metrics(S[t]) if t in S else None; b = metrics(ec)
                if a and b and t in commons[d]: rows.append((a, b))
            L = lambda k: np.array([math.log(b[k] / a[k]) for a, b in rows])
            dR, dN, dh, dX, dF = L("R"), L("N"), L("h"), L("X"), L("F"); dm = dN - dR
            sF = np.array([(a["sF"] + b["sF"]) / 2 for a, b in rows]); sHH = 1 - sF
            Rm = np.array([(a["R"] + b["R"]) / 2 for a, b in rows])
            res = {}
            for lab, k in (("limit", 1 + sHH), ("exact", 1 + sHH * Rm / (Rm - 1))):
                seg = dict(act=float(np.mean(k * dN)), turns=float(np.mean(-k * dm)), cont=float(np.mean(sHH * dh)), text=float(np.mean(sF * dF)))
                res[lab] = {kk: round(vv, 6) for kk, vv in seg.items()}; res[lab]["sum"] = round(sum(seg.values()), 6)
                p = k * dN - k * dm + sHH * dh + sF * dF
                res[lab]["ep_R2"] = round(float(1 - ((dX - p) ** 2).sum() / ((dX - dX.mean()) ** 2).sum()), 4)
            skills.append(dict(bench=bench, method=d, n=len(rows), dlogX=round(float(dX.mean()), 6), **res))

    def r2(lab):
        y = np.array([s["dlogX"] for s in skills]); p = np.array([s[lab]["sum"] for s in skills])
        return dict(R2=round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4), rmse=round(float(np.sqrt(((y - p) ** 2).mean())), 4), maxgap=round(float(np.abs(y - p).max()), 4))
    out["skills"] = skills; out["fit"] = {lab: r2(lab) for lab in ("limit", "exact")}
    y = np.array([s["dlogX"] for s in skills]); p3 = np.array([s["exact"]["act"] + s["exact"]["turns"] + s["exact"]["cont"] for s in skills])
    out["fit"]["exact_without_text_R2"] = round(float(1 - ((y - p3) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)
    return out


def cost_table(c):
    """Per-skill four-term table: one row per optimized skill, columns as printed (Act., Split, Cont., Text, Total), from the eq:split decomposition with the exact e_R."""
    rows = [dict(bench=s["bench"], skill=s["method"], n=s["n"], act=s["exact"]["act"], split=s["exact"]["turns"], cont=s["exact"]["cont"], text=s["exact"]["text"],
                 total=s["dlogX"], sum_of_terms=s["exact"]["sum"]) for s in c["skills"]]
    return {"columns": {"act": "Act.", "split": "Split", "cont": "Cont.", "text": "Text", "total": "Total (measured mean d log X = ln rho)"},
            "rows": rows, "fit_four_terms": c["fit"]["exact"], "fit_without_text_R2": c["fit"]["exact_without_text_R2"]}


def fit(xs, ps):
    xs, ps = np.array(xs), np.array(ps)
    return dict(n=len(xs), r2=float(1 - np.sum((xs - ps) ** 2) / np.sum((xs - xs.mean()) ** 2)) if len(xs) > 1 and xs.var() > 0 else None, rms=float(np.sqrt(np.mean((xs - ps) ** 2))),
                corr=float(np.corrcoef(xs, ps)[0, 1]) if len(xs) > 2 else None, slope=float(np.polyfit(ps, xs, 1)[0]) if len(xs) > 2 else None, rms_obs=float(np.sqrt(np.mean(xs ** 2))))


def skill_fit(model):
    """Per optimized skill, on the test tasks both it and the seed completed: mean d log of R, N, m, h, F, X; the four terms t_N, t_m, t_h, t_F (k = s_F + 2 s_H);
    prediction k dR + s_H dh (+ s_F dF); residual; episode-level fit; success seed -> skill. Pooled fits and episode-level within-task slopes per benchmark."""
    jobs = []
    for bench in C.BENCHES:
        sf = C.eval_file(C.current_run("native-seed", model, bench), "test", C.vid(C.SEED_TEXT))
        for disp, (run, t) in C.returned_skills(model, bench).items():
            jobs.append((bench, disp, C.eval_file(run, "test", C.vid(t)), sf))
    seedcache, results, allep = {}, [], []
    for bench, disp, f, sf in jobs:
        label = f"{model}|{bench}|{disp}"
        if f is None: print("MISSING", label); continue
        S = seedcache.setdefault(str(sf), episodes(sf)); rows = []
        for t, ec in episodes(f).items():
            es = S.get(t)
            if es is None: continue
            a, b = metrics(es), metrics(ec)
            if a is None or b is None: continue
            rows.append((t, a, b))
        if not rows: print("EMPTY", label); continue
        L = lambda k: np.array([math.log(b[k] / a[k]) for _, a, b in rows])
        dR, dN, dm, dh, dX, dF = L("R"), L("N"), L("m"), L("h"), L("X"), L("F")
        sF = np.array([(a["sF"] + b["sF"]) / 2 for _, a, b in rows]); sH = 1 - sF
        k = sF + 2 * sH
        pred = k * dR + sH * dh; predF = pred + sF * dF
        tN, tm, th, tF = k * dN, -k * dm, sH * dh, sF * dF
        sd = lambda a_: float(np.mean(a_))
        res = dict(label=label, model=model, bench=bench, skill=disp, n=len(rows), dlogR=sd(dR), dlogN=sd(dN), dlogm=sd(dm), dlogh=sd(dh), dlogF=sd(dF), dlogX=sd(dX),
                   sF_seed=float(np.mean([a["sF"] for _, a, b in rows])), sF=sd(sF), sH=sd(sH), pred=sd(pred), predF=sd(predF), resid=sd(dX - pred),
                   turns=sd(k * dR), t_N=sd(tN), t_m=sd(tm), t_h=sd(th), t_F=sd(tF),
                   succ_seed=float(np.mean([bool(a["succ"]) for _, a, b in rows])), succ=float(np.mean([bool(b["succ"]) for _, a, b in rows])),
                   ep_rms=float(np.sqrt(np.mean((dX - pred) ** 2))), ep_r2=float(1 - np.sum((dX - pred) ** 2) / np.sum((dX - dX.mean()) ** 2)))
        results.append(res)
        for x, p in zip(dX, pred): allep.append((label, bench, model, x, p))
        print(label, {k_: round(v, 3) if isinstance(v, float) else v for k_, v in res.items() if k_ not in ("label", "model", "bench")})
    per = {}
    for r in results: per.setdefault(r["bench"], []).append(r)
    summary = {"pooled_skills": fit([r["dlogX"] for r in results], [r["pred"] for r in results]),
               "pooled_skills_withF": fit([r["dlogX"] for r in results], [r["predF"] for r in results]),
               "pooled_episodes": fit([e[3] for e in allep], [e[4] for e in allep]), "per_bench": {}, "per_setting": {}}
    mean = lambda rs, key, f=lambda v: v: float(np.mean([f(r[key]) for r in rs]))
    for b, rs in per.items():
        summary["per_bench"][b] = dict(skills=len(rs), fit=fit([r["dlogX"] for r in rs], [r["pred"] for r in rs]),
              mean_abs={k: mean(rs, "t_" + k, abs) for k in "N m h F".split()}, mean_signed=dict(N=mean(rs, "t_N"), m=mean(rs, "t_m"), h=mean(rs, "t_h"), dlogX=mean(rs, "dlogX")),
              mean_sH=mean(rs, "sH"), ep_fit=fit([e[3] for e in allep if e[1] == b], [e[4] for e in allep if e[1] == b]))
    for lab in sorted({r["model"] + "|" + r["bench"] for r in results}):
        rs = [r for r in results if r["model"] + "|" + r["bench"] == lab]
        summary["per_setting"][lab] = dict(skills=len(rs), fit=fit([r["dlogX"] for r in rs], [r["pred"] for r in rs]))
    # episode-level regression of log X on log R per benchmark over seed and skill episodes, raw and within task
    eps_all = {}
    for bench, disp, f, sf in jobs:
        label = f"{model}|{bench}|{disp}"
        for tag, p in ((disp, f), ("seed", sf)):
            if not p: continue
            for t, e in episodes(p).items():
                m_ = metrics(e)
                if m_: eps_all.setdefault((model, bench), {}).setdefault((label if tag != "seed" else "seed|" + model + "|" + bench, t), m_)
    reg = {}
    for b in C.BENCHES:
        recs = [(k[1], m_) for (mo, be), d in eps_all.items() if be == b for k, m_ in d.items()]
        if len(recs) < 5: continue
        y = np.log([m_["X"] for _, m_ in recs]); x = np.log([m_["R"] for _, m_ in recs]); tk = [t for t, _ in recs]
        slope_raw = float(np.polyfit(x, y, 1)[0])
        g = collections.defaultdict(list)
        for i, t in enumerate(tk): g[t].append(i)
        xs = x.copy(); ys = y.copy()
        for idx in g.values(): xs[idx] -= x[idx].mean(); ys[idx] -= y[idx].mean()
        slope_w = float((xs * ys).sum() / (xs * xs).sum()); r2w = float(((xs * ys).sum() ** 2) / ((xs * xs).sum() * (ys * ys).sum()))
        sH = float(np.mean([1 - m_["sF"] for _, m_ in recs]))
        reg[b] = dict(n=len(recs), slope_raw=slope_raw, slope_within_task=slope_w, r2_within=r2w, mean_sH=sH, predicted_slope=1 + sH,
                      sH_weighted_by_X=float(np.sum([(1 - m_["sF"]) * m_["X"] for _, m_ in recs]) / np.sum([m_["X"] for _, m_ in recs])))
    summary["regression"] = reg
    return {"skills": results, "summary": summary}


def raw_files(model):
    """The test evaluation files checks() reads: the seed (main run and reruns) and each returned skill (main run and reruns)."""
    out = []
    for bench in C.BENCHES:
        seedrun = C.current_run("native-seed", model, bench); texts = {"seed": (seedrun, "native-seed", C.SEED_TEXT)}
        for disp, (run, t) in C.returned_skills(model, bench, METHODS3).items(): texts[disp] = (run, METHODS3[disp], t)
        for d, (run, meth, t) in texts.items():
            out.append(C.eval_file(run, "test", C.vid(t)))
            for r in ("rep-1", "rep-2"):
                dd = C.EX / model / meth / bench / r
                if dd.is_dir(): out.append(C.eval_file(dd, "test", C.vid(t)))
    return [f for f in out if f]


def checks_cli(argv):
    ap = argparse.ArgumentParser(description="Sec. 3 numbers of the paper that Fig. 1 and the appendix costsplit text rest on")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--out", default=str(C.out_dir("slopes_checks")))
    a = ap.parse_args(argv)
    out = C.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = checks(a.model); (out / "checks.json").write_text(json.dumps(c, indent=1)); print(json.dumps(c, indent=1))
    f = skill_fit(a.model); (out / "fit.json").write_text(json.dumps(f, indent=1)); print(json.dumps(f["summary"], indent=1))


# ---- rerun checks of the elasticity analysis (merged from reruns.py) ----------------------------------------------------------------

MODEL = "deepseek-flash"
DS_ROOT = EX/MODEL


def fixed_part(we):
    """F: mean over requests of the input tokens in the fixed components (system, tool schemas, task, skill); first request's input if the graph has none."""
    idx = [i for i, c in enumerate(we.comps) if c in FIXED]
    return float(we.Xrg[:, idx].sum(1).mean()) if idx else float(we.req_in[0])


def rerun_files(bench, names=OPTIMIZERS):
    """{skill: [(run label, test evaluation file)]}: the main run of the seed and of each returned optimizer skill, plus its reruns
    (seed: rep-1/rep-2 beside the main run; optimizer skills: the repetitions under test-reps/<bench>__<method>__<run>, as analysis.main's rep_files)."""
    from src.analysis.main import rep_files as t1_rep_files
    texts = {"seed": SEED_TEXT}; runs = {}
    main = {"seed": eval_file(current_run("native-seed", MODEL, bench), "test", vid(SEED_TEXT))}
    for disp, (run, t) in returned_skills(MODEL, bench, names).items():
        texts[disp] = t; runs[disp] = run; main[disp] = eval_file(run, "test", vid(t))
    reps = collections.defaultdict(list)
    for disp, f in main.items(): reps[disp].append(("main", f))
    for disp in texts:
        meth = "native-seed" if disp == "seed" else names[disp]
        if disp == "seed":
            for r in ("rep-1", "rep-2"):
                d = DS_ROOT/meth/bench/r
                if d.is_dir():
                    f = eval_file(d, "test", vid(texts[disp]))
                    if f: reps[disp].append((r, f))
        else:
            found = t1_rep_files([runs[disp], DS_ROOT/"test-reps"/f"{bench}__{meth}__{runs[disp].name}"], vid(texts[disp]))
            for rep, f in sorted(found.items()):
                if rep != 0: reps[disp].append((f"rep-{rep}", f))
    return reps


def rerun_metrics(ep):
    """The rerun checks' per-episode metrics (the metrics of the former reruns.py; the checks-stage metrics above is the shared contract)."""
    if ep.R < 2 or ep.N < 1: return None
    F = fixed_part(ep); X = ep.X
    if X - ep.R*F <= 0 or F <= 0: return None
    sF = ep.R*F/X
    return dict(R=ep.R, N=ep.N, X=X, sF=sF, eR=1+(1-sF)*ep.R/(ep.R-1), succ=ep.success)


def slope(rs, groups):
    """Within-group slope of log X on log R (groups = fixed effects) with its standard error."""
    x = np.log([m["R"] for *_, m in rs]); y = np.log([m["X"] for *_, m in rs])
    g = collections.defaultdict(list)
    for i, r in enumerate(rs): g[groups(r)].append(i)
    g = {k: i for k, i in g.items() if len(i) > 1}
    idx = [i for v in g.values() for i in v]
    xs, ys = x.copy(), y.copy()
    for ii in g.values(): xs[ii] -= x[ii].mean(); ys[ii] -= y[ii].mean()
    xs, ys = xs[idx], ys[idx]
    sl = float((xs*ys).sum()/(xs*xs).sum()); res = ys - sl*xs
    dof = len(xs) - len(g) - 1; se = float(np.sqrt((res**2).sum()/dof/(xs*xs).sum()))
    return dict(slope=round(sl, 3), se=round(se, 3), n=len(xs), groups=len(g))


def twoway(rs):
    """Slope with task and skill fixed effects (alternating projections)."""
    x = np.log([m["R"] for *_, m in rs]); y = np.log([m["X"] for *_, m in rs])
    T = [r[2] for r in rs]; S = [r[0] for r in rs]
    def demean(v):
        v = v.copy()
        for _ in range(200):
            for key in (T, S):
                g = collections.defaultdict(list)
                for i, k in enumerate(key): g[k].append(i)
                for ii in g.values(): v[ii] -= v[ii].mean()
        return v
    xs, ys = demean(x), demean(y)
    return round(float((xs*ys).sum()/(xs*xs).sum()), 3)


def check(names):
    out = {}
    for bench in BENCHES:
        reps = rerun_files(bench, names); recs = []
        for disp, lst in reps.items():
            for rn, f in lst:
                for t, e in {e.task: e for e in W.load(f)}.items():
                    m = rerun_metrics(e)
                    if m: recs.append((disp, rn, t, m))
        mainrecs = [r for r in recs if r[1] == "main"]
        eR = np.array([m["eR"] for *_, m in recs]); eRm = np.array([m["eR"] for *_, m in mainrecs])
        Rs = np.array([m["R"] for *_, m in mainrecs])
        out[bench] = dict(
            runs={k: [r for r, _ in l] for k, l in reps.items()},
            slope_main_taskFE=slope(mainrecs, lambda r: r[2]),
            slope_main_task_plus_skillFE=twoway(mainrecs),
            slope_seed_only_taskFE=slope([r for r in recs if r[0] == "seed"], lambda r: r[2]),
            slope_taskxskill_FE_reps=slope(recs, lambda r: (r[2], r[0])),
            eR_main=dict(min=round(float(eRm.min()), 3), max=round(float(eRm.max()), 3), n=len(eRm), above2=int((eRm > 2).sum()), below1=int((eRm < 1).sum())),
            eR_all=dict(min=round(float(eR.min()), 3), max=round(float(eR.max()), 3), n=len(eR), above2=int((eR > 2).sum())),
            R_main=dict(min=int(Rs.min()), median=float(np.median(Rs)), max=int(Rs.max())),
            sH_mean_main=round(float(np.mean([1-m["sF"] for *_, m in mainrecs])), 3))
        ps = {}   # success (%) of each run, then pooled over the runs
        for disp, lst in reps.items():
            q = []
            for rn, f in lst:
                d = json.loads(f.read_text()); eps = {}
                for e in d["episodes"]: eps.setdefault(e["task_id"], e)
                ys = [eps[t["task_id"]].get("success") for t in d["panel"] if t["task_id"] in eps]
                q.append(100*np.mean([y for y in ys if y is not None]))
            ps[disp] = [round(float(z), 1) for z in q] + ["pooled", round(float(np.mean(q)), 1)]
        out[bench]["success_runs"] = ps
    return out


def seed_spread():
    out = {}
    for bench in BENCHES:
        files = [eval_file(current_run("native-seed", MODEL, bench), "test", vid(SEED_TEXT))]
        for r in ("rep-1", "rep-2"): files.append(eval_file(DS_ROOT/"native-seed"/bench/r, "test", vid(SEED_TEXT)))
        runs = [{e.task: e for e in W.load(f)} for f in files]
        common_ = set.intersection(*[{t for t, e in r.items() if e.X and e.X > 0} for r in runs])
        means = [float(np.mean([math.log(r[t].X) for t in common_])) for r in runs]
        calls = [float(np.mean([math.log(max(r[t].N, 1)) for t in common_])) for r in runs]
        out[bench] = dict(n=len(common_), mean_logX=[round(m, 4) for m in means], spread_logX=round(max(means)-min(means), 4), spread_logN=round(max(calls)-min(calls), 4))
    return out


def reruns_cli(argv):
    ap = argparse.ArgumentParser(description="Rerun checks of the elasticity analysis (DeepSeek-V4.1-Flash)")
    ap.add_argument("--out", default=str(out_dir("reruns")))
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = check(OPTIMIZERS)
    (out/"check.json").write_text(json.dumps(c, indent=1)); print(json.dumps(c, indent=1))
    s = seed_spread()
    (out/"seed_spread.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s, indent=1))


# ---- default subcommand: the episode-level slope tables (the original slopes.py bare run) ---------------------------------------------

lx = lambda r: np.log(r["X"]); lR = lambda r: np.log(r["R"])


def slope_tables():
    rows = ep_rows(); say = Report()
    say("== Q1 slope of log X on log R (within-task FE)")
    for kind in ("main", "seed", "opt", "all"):
        for b in BN:
            s = samp(rows, b, kind); bb, se = fe(s, lx, lR)
            eR = np.array([r["eR"] for r in s]); sH = np.array([r["sH"] for r in s])
            say(f"{kind:5s} {BN[b]:5s} n={len(s):4d} slope={bb:.2f}({se:.2f}) meanER={eR.mean():.2f} range=[{eR.min():.2f},{eR.max():.2f}] 1+meanSH={1+sH.mean():.2f}")
    say("-- per skill (within task FE, all reps for that skill)")
    for b in BN:
        for sk in ("seed", "gepa", "skillopt", "skilladam"):
            s = [r for r in rows if r["bench"] == b and r["skill"] == sk]
            if len(s) >= 17:
                bb, se = fe(s, lx, lR); say(f"  {BN[b]} {sk} n={len(s)} slope={bb:.2f}({se:.2f}) meanER={np.mean([r['eR'] for r in s]):.2f}")
    say("-- task x skill FE (within a skill-task pair across reruns)")
    for b in BN:
        s2 = [dict(r, task=r["task"] + "|" + r["skill"]) for r in rows if r["bench"] == b]
        c = collections.Counter(r["task"] for r in s2); s2 = [r for r in s2 if c[r["task"]] > 1]
        bb, se = fe(s2, lx, lR); say(f"  {BN[b]} n={len(s2)} slope={bb:.2f}({se:.2f})")
    say("-- gap decomposition: slope = E[e_R] + s_H*b_h + s_F*b_F  (within task, per-episode weights approximated by means)")
    for b in BN:
        s = samp(rows, b, "main")
        sl, _ = fe(s, lx, lR)
        bh, _ = fe(s, lambda r: np.log(r["h"]), lR); bF, _ = fe(s, lambda r: np.log(r["F"]), lR)
        eR = np.mean([r["eR"] for r in s]); sH = np.mean([r["sH"] for r in s]); sF = 1 - sH
        say(f"{BN[b]}: slope={sl:.2f}  mean eR={eR:.2f}  b_h(dlogh/dlogR)={bh:.2f} b_F={bF:.2f}  predicted={eR+sH*bh+sF*bF:.2f}  gap(slope-eR)={sl-eR:.2f}  sH*b_h={sH*bh:.2f} sF*bF={sF*bF:.2f}")
        g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(r)
        for r in s:
            gg = g[r["task"]]; r["h_bar"] = gmean([q["h"] for q in gg])
            r["F_bar"] = gmean([q["F"] for q in gg])
            r["Xcf"] = r["R"] * r["F_bar"] + r["R"] * (r["R"] - 1) / 2 * r["h_bar"]
        scf, _ = fe(s, lambda r: np.log(r["Xcf"]), lR)
        say(f"    counterfactual slope with h,F fixed at task means (pure structural, geometric means): {scf:.2f}")
    say("-- spread of R within task (sd log R) and e_R std")
    for b in BN:
        s = samp(rows, b, "main"); g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(np.log(r["R"]))
        say(BN[b], "median sd logR within task", np.median([np.std(v) for v in g.values()]), "median R", np.median([r["R"] for r in s]), "R range", min(r["R"] for r in s), max(r["R"] for r in s))
    say("== A: share of input tokens in last k turns / last 25% turns; marginal X_t / X_1")
    for b in BN:
        s = samp(rows, b, "main"); l4 = []; l3 = []
        for r in s:
            x = r["req_in"]; R = r["R"]; l4.append(x[-max(1, R // 4):].sum() / x.sum()); l3.append(x[-3:].sum() / x.sum())
        say(BN[b], "last quarter of turns carries", round(np.mean(l4), 2), "of input; last 3 turns", round(np.mean(l3), 2), "; median R", np.median([r['R'] for r in s]))
    say("== B: turns vs success within task (logit-free: success rate by R relative to task median)")
    for b in BN:
        s = samp(rows, b, "all"); g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(r)
        d = []
        for t, v in g.items():
            if len(v) < 3: continue
            m = np.median([q["R"] for q in v])
            for q in v:
                if q["succ"] is not None: d.append((q["R"] / m, int(q["succ"])))
        d = np.array(d); lo = d[d[:, 0] < 1]; hi = d[d[:, 0] > 1]
        say(BN[b], "success rate: shorter-than-task-median", round(lo[:, 1].mean(), 2), f"(n={len(lo)})", "longer", round(hi[:, 1].mean(), 2), f"(n={len(hi)})")
        sd = [r for r in s if r["succ"] is not None]
        say("   mean R success", round(np.mean([r['R'] for r in sd if r['succ']]), 1), "fail", round(np.mean([r['R'] for r in sd if not r['succ']]), 1), "; mean X success", round(np.mean([r['X'] for r in sd if r['succ']]) / 1e3), "fail", round(np.mean([r['X'] for r in sd if not r['succ']]) / 1e3), "k")
    say("== C: where extra turns come from: optimized vs seed paired by task (main), dlogR = dlogN - dlog m ; zero-call turn share")
    for b in BN:
        sd = {r["task"]: r for r in rows if r["bench"] == b and r["skill"] == "seed" and r["rep"] == "main"}
        for sk in ("gepa", "skillopt", "skilladam"):
            so = [r for r in rows if r["bench"] == b and r["skill"] == sk and r["rep"] == "main"]
            if not so: continue
            p = [(r, sd[r["task"]]) for r in so if r["task"] in sd]
            dR = np.mean([np.log(a["R"] / c["R"]) for a, c in p]); dN = np.mean([np.log(a["N"] / c["N"]) for a, c in p])
            z = lambda r: np.mean(r["ncall"] == 0)
            dX = np.mean([np.log(a["X"] / c["X"]) for a, c in p])
            dP = np.mean([np.log(price(a, .1, 4, "res_all") / price(c, .1, 4, "res_all")) for a, c in p])
            say(BN[b], sk, f"dlogX={dX:+.3f} dlogR={dR:+.3f} dlogN={dN:+.3f} dlog m={dN-dR:+.3f} dlogPrice(.1,4)={dP:+.3f}  zero-call turn share seed {np.mean([z(c) for a, c in p]):.2f} -> {np.mean([z(a) for a, c in p]):.2f}")
    say("== D: per-turn mechanism: calls/turn m, share of zero-call turns, mean new content per turn h (seed main)")
    for b in BN:
        s = [r for r in samp(rows, b, "main") if r["skill"] == "seed"]
        say(BN[b], "m", round(np.mean([r['N'] / r['R'] for r in s]), 2), "zero-call share", round(np.mean([np.mean(r['ncall'] == 0) for r in s]), 2), "h tokens", round(np.median([r['h'] for r in s])), "F tokens", round(np.median([r['F'] for r in s])), "sH", round(np.mean([r['sH'] for r in s]), 2))
    say.save("slopes", "slopes.txt")


def main():
    if sys.argv[1:2] == ["checks"]: checks_cli(sys.argv[2:])
    elif sys.argv[1:2] == ["reruns"]: reruns_cli(sys.argv[2:])
    else: slope_tables()


if __name__ == "__main__":
    main()
