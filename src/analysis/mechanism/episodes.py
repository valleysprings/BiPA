"""Per-episode rows (turns R, input tokens per request, fixed prefix F, cache, output, calls) of the DeepSeek test evaluations and the within-task fixed-effect
slope used by analysis/mechanism/slopes.py (Sec. 3 elasticity claims) and analysis/mechanism/pricing.py (price slope vs cache price). Loads through graph.episode; no outputs."""
import collections, json
import numpy as np
from src.analysis.common import BENCHES, DS, SEED_TEXT, current_run, eval_file, final_skill, vid  # first: puts bipa/ on sys.path
from src.graph import episode as W

FIXED = {"system", "tool_schemas", "task", "skill"}
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


def episode_rows():
    """Rows of all benches with derived fields; episodes with a single turn or no history are dropped."""
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
