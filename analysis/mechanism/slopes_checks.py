"""Sec. 3 numbers of the paper that Fig. 1 and the appendix costsplit text rest on, from recorded test episodes (no model calls): within-task slopes of log input tokens on log requests (Fig. 1 panel (a)
values and the appendix slopes text), 1 + mean s_H and the exact turn elasticity e_R, co-variation of the per-request content h with R, the decomposition of Eq. (split) with the limit (1 + s_H)
against the exact e_R (Fig. 1 panel (b) and the appendix worked readouts), and a second per-skill fit of d log X by the terms t_N, t_m, t_h, t_F with k = s_F + 2 s_H on all paired tasks (supplementary, not in the paper).
(slopes.py is the other group's episode-level slope tables; this module is the per-skill decomposition and its checks.)

Episode quantities (per task): R requests, N executed calls, X input tokens, F = mean over requests of the fixed components (system, tool schemas, task, skill),
h = (X - R F) / (R (R - 1) / 2) the content added per request, s_F = R F / X, e_R = 1 + (1 - s_F) R / (R - 1).

Outputs (assets/cache/analysis/slopes_checks/):
  checks.json  slopes (main, seed x3 repetitions, h, F), 1 + mean s_H, exact e_R, eq:split decomposition per skill with R^2 (limit vs exact e_R)
  fit.json     per skill dlogR/N/m/h/F, the four terms (t_N, t_m, t_h, t_F), prediction and residual; pooled fits; episode-level slopes per benchmark (supplementary)
Command: <bach-harness venv python> -m analysis.mechanism.slopes_checks [--model deepseek-flash] [--out DIR]   (from bipa/)"""
import argparse, collections, json, math
import numpy as np
from analysis import common as C
from graph import episode as W

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--out", default=str(C.out_dir("slopes_checks")))
    a = ap.parse_args()
    out = C.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = checks(a.model); (out / "checks.json").write_text(json.dumps(c, indent=1)); print(json.dumps(c, indent=1))
    f = skill_fit(a.model); (out / "fit.json").write_text(json.dumps(f, indent=1)); print(json.dumps(f["summary"], indent=1))


if __name__ == "__main__":
    main()
