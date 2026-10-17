"""Power / minimum detectable effect (MDE80) of the submission protocol, driven by the frozen manifest (analysis/submission_protocol.py); offline, read-only on artifacts.

Question: can the Table 1 comparison (candidate avg@3 vs Seed avg@3, paired over the same test tasks) detect gains of the size the original papers report (lit_gains.json)?
Inputs: the seed's three pinned test executions per bench (manifest_reps), loaded as in nullmodel.py (request-id dedup); lit gains are the verified extraction in lit_gains.md/.json, transcribed in GAINS below.
Model (per simulation): seed per-task success p_t ~ Beta(K_t + a, R - K_t + b) with (a, b) the beta-binomial fit on the seed's (K_t, R=3) counts (nullmodel.bb_fit); the test tasks are resampled with replacement (parametric
bootstrap over tasks), each execution is a fresh Bernoulli(p_t) draw. Both arms are simulated fresh (the observed Seed row is not held fixed). Candidate = Seed with mean shift delta (points):
  uniform  : p'_t = clip(p_t + c, 0, 1) on every task, c solved per simulation so that the panel's mean success rises by delta (saturates at c = 1 in a simulation whose own headroom is smaller);
  mixed    : same, but only on tasks the seed did not always solve (observed K_t < R, i.e. mixed or failed), always-solved tasks unchanged.
Test: paired t-test over tasks on the per-task difference of mean success (two-sided alpha 0.05); power = P(reject and candidate better). MDE80 = smallest delta with power >= 0.8 (grid step 2, then bisection on common
random numbers); none = delta is above the headroom (mean over simulations of 1 - p on the shiftable tasks) or power stays < 0.8 up to it. Settings: avg3 (3 executions per arm, test panel), single (1 vs 1 execution, test panel; what the original
papers do), dev (1 execution per arm on the selection panel of 30/30/9 tasks; tasks resampled from the test tasks' posterior, so the dev prior equals the test one). Fixed seed; Monte Carlo SE of a power value <= 0.8%.
Output (--out DIR, created): power.json, power.md. Run: <bach-harness venv python> -m src.analysis.mechanism.power --manifest FILE --out DIR   (from bipa/)"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist, ttest_rel
from src.analysis import common
from src.analysis.common import BENCHES
from src.analysis.mechanism.nullmodel import bb_fit, load_exec, seed_execs, tally

SEED, NS, ALPHA, TARGET = 20261008, 4000, 0.05, 0.8
CURVE = (2.5, 5, 7.5, 10, 15, 20)
LIT = Path(common.H)/"assets/cache/analysis/submission-20261008/lit_gains.json"
# reported gains (pp), transcribed from lit_gains.md (verified extraction). cls: a = vs starting skill/prompt, b = vs no skill, c = method vs method. ag = agentic multi-turn / tool-using.
# (cls, paper, label, gain, n_test, agentic)
GAINS = [
 ("a", "GEPA", "GEPA vs seed prompt, HotpotQA, Qwen3-8B", 20.00, 300, False), ("a", "GEPA", "GEPA vs seed prompt, IFBench, Qwen3-8B", 1.71, 294, False),
 ("a", "GEPA", "GEPA vs seed prompt, HoVer, Qwen3-8B", 17.00, 300, False), ("a", "GEPA", "GEPA vs seed prompt, PUPA, Qwen3-8B", 11.03, 221, False),
 ("a", "GEPA", "GEPA vs seed prompt, AIME-2025, Qwen3-8B", 4.67, 150, False), ("a", "GEPA", "GEPA vs seed prompt, LiveBench-Math, Qwen3-8B", 3.25, 123, False),
 ("a", "GEPA", "GEPA vs seed prompt, aggregate of 6, Qwen3-8B", 9.62, None, False), ("a", "GEPA", "GEPA vs seed prompt, HotpotQA, GPT-4.1 mini", 31.00, 300, False),
 ("a", "GEPA", "GEPA vs seed prompt, IFBench, GPT-4.1 mini", 4.93, 294, False), ("a", "GEPA", "GEPA vs seed prompt, HoVer, GPT-4.1 mini", 5.34, 300, False),
 ("a", "GEPA", "GEPA vs seed prompt, PUPA, GPT-4.1 mini", 15.90, 221, False), ("a", "GEPA", "GEPA vs seed prompt, AIME-2025, GPT-4.1 mini", 10.00, 150, False),
 ("a", "GEPA", "GEPA vs seed prompt, LiveBench-Math, GPT-4.1 mini", 5.93, 123, False), ("a", "GEPA", "GEPA vs seed prompt, aggregate of 6, GPT-4.1 mini", 12.19, None, False),
 ("a", "GEPA", "GEPA vs plain Sequential10 agent, NPUEval (vector utilisation), GPT-4o", 26.27, None, True), ("a", "GEPA", "GEPA vs no optimisation, KernelBench fast_1 (figure-read), GPT-4o", 20.0, 35, True),
 ("a", "SkillOpt", "SkillOpt vs Human skill, ALFWorld, GPT-5.5", 3.7, 134, True), ("a", "SkillOpt", "SkillOpt vs Human skill, ALFWorld, GPT-5.4", 16.4, 134, True),
 ("a", "SkillOpt", "SkillOpt vs Human skill, SpreadsheetBench, GPT-5.5", 7.8, None, False), ("a", "SkillOpt", "SkillOpt vs Human skill, OfficeQA, GPT-5.5", 5.2, None, False),
 ("a", "SkillAdam", "SkillOpt vs Human skill, SearchQA, GPT-5.5", 5.5, None, False), ("a", "SkillAdam", "SkillOpt vs Human skill, DocVQA, GPT-5.5", 1.1, None, False),
 ("a", "SkillAdam", "SkillAdam vs seed skill, DeepPlanning Shopping L1 (25 cases, one run), Sonnet 4.5", 16.0, 25, True),
 ("b", "SkillOpt", "SkillOpt vs no skill, ALFWorld, GPT-5.5", 11.9, 134, True), ("b", "SkillOpt", "SkillOpt vs no skill, ALFWorld, GPT-5.4", 15.6, 134, True),
 ("b", "SkillOpt", "SkillOpt vs no skill, ALFWorld, GPT-5.4-mini", 12.7, 134, True), ("b", "SkillOpt", "SkillOpt vs no skill, ALFWorld, GPT-5.4-nano", 35.1, 134, True),
 ("b", "SkillOpt", "SkillOpt vs no skill, ALFWorld, Qwen3.5-4B", 50.7, 134, True), ("b", "SkillOpt", "SkillOpt vs no skill, SpreadsheetBench Codex harness, GPT-5.5", 57.5, None, True),
 ("b", "SkillOpt", "SkillOpt vs no skill, SpreadsheetBench Claude Code harness, GPT-5.5", 58.3, None, True), ("b", "SkillOpt", "SkillOpt vs no skill, SearchQA direct chat, GPT-5.5", 9.6, None, False),
 ("b", "SkillOpt", "SkillOpt vs no skill, DocVQA direct chat, GPT-5.5", 12.4, None, False), ("b", "SkillOpt", "SkillOpt vs no skill, OfficeQA direct chat, GPT-5.5", 39.0, None, False),
 ("b", "SkillOpt", "SkillOpt vs no skill, SearchQA direct chat, GPT-5.4 (smallest)", 6.2, None, False), ("b", "SkillOpt", "SkillOpt vs no skill, six-benchmark average, GPT-5.5", 23.5, None, False),
 ("b", "SkillOpt", "GEPA vs no skill, ALFWorld, GPT-5.5", 2.2, 134, True), ("b", "SkillOpt", "GEPA vs no skill, ALFWorld, GPT-5.4", -0.8, 134, True),
 ("b", "SkillOpt", "GEPA vs no skill, ALFWorld, GPT-5.4-mini", 8.2, 134, True),
 ("b", "SkillAdam", "SkillAdam vs no skill, ALFWorld, GPT-5.5", 6.0, None, True), ("b", "SkillAdam", "SkillAdam vs no skill, DeepPlanning Shopping (60 cases), Sonnet 4.5", 13.3, 60, True),
 ("b", "SkillAdam", "SkillAdam vs no skill, DeepPlanning Travel (60 cases), Sonnet 4.5", 11.7, 60, True), ("b", "SkillAdam", "SkillAdam vs no skill, DeepPlanning average, Sonnet 4.5", 12.5, 120, True),
 ("b", "SkillAdam", "SkillAdam vs no skill, SearchQA, GPT-5.5", 9.8, None, False),
 ("c", "SkillAdam", "SkillAdam vs SkillOpt, ALFWorld, GPT-5.5", 2.3, None, True), ("c", "SkillAdam", "SkillAdam vs SkillOpt, DeepPlanning Shopping, Sonnet 4.5", 3.3, 60, True),
 ("c", "SkillAdam", "SkillAdam vs SkillOpt, DeepPlanning Travel, Sonnet 4.5", 10.0, 60, True), ("c", "SkillAdam", "SkillAdam vs SkillOpt, DeepPlanning average, Sonnet 4.5", 6.7, 120, True),
 ("c", "SkillAdam", "SkillAdam vs SkillOpt, SearchQA, GPT-5.5", 0.2, None, False), ("c", "SkillAdam", "SkillAdam vs SkillOpt, SpreadsheetBench, GPT-5.5", 0.4, None, False),
 ("c", "SkillAdam", "SkillAdam vs SkillOpt, OfficeQA, GPT-5.5", 0.0, None, False), ("c", "SkillAdam", "SkillAdam vs SkillOpt, DocVQA, GPT-5.5", 1.1, None, False),
 ("c", "SkillAdam", "SkillAdam vs SkillOpt, LiveMathematicianBench, GPT-5.5", 0.8, None, False),
 ("c", "SkillOpt", "SkillOpt vs GEPA, ALFWorld, GPT-5.5", 9.7, 134, True), ("c", "SkillOpt", "SkillOpt vs GEPA, ALFWorld, GPT-5.4", 16.4, 134, True),
 ("c", "SkillOpt", "SkillOpt vs GEPA, ALFWorld, GPT-5.4-mini", 4.5, 134, True), ("c", "SkillOpt", "SkillOpt vs GEPA, ALFWorld, GPT-5.4-nano (smallest)", 0.7, 134, True),
 ("c", "SkillOpt", "SkillOpt vs GEPA, SearchQA, GPT-5.5", 2.5, None, False), ("c", "SkillOpt", "SkillOpt vs GEPA, SpreadsheetBench, GPT-5.5", 7.1, None, False),
 ("c", "SkillOpt", "SkillOpt vs GEPA, DocVQA, GPT-5.5", 2.1, None, False), ("c", "SkillOpt", "SkillOpt vs GEPA, LiveMathematicianBench, GPT-5.5", 23.7, None, False),
]
CLS = {"a": "vs starting skill/prompt", "b": "vs no skill", "c": "method vs method"}


# ---- model ------------------------------------------------------------------------------------------------------------------------------------------

def seed_stats(ex):
    """per-task counts K (of R seed executions) with the descriptive within / between-task statistics of the seed's test executions."""
    tasks, K, n, mult = tally(ex); R = int(n.min()); assert (n == R).all() and (mult == 1).all()
    p = K / R; s2 = K * (R - K) / (R * (R - 1)); tot = np.array([sum(e["succ"].values()) for e in ex]) / len(tasks) * 100
    a, b = bb_fit(K, n)
    return tasks, K, R, a, b, dict(n_tasks=len(tasks), R=R, seed_avg=100 * float(p.mean()), exec_rates=[float(x) for x in tot], exec_sd_pts=float(tot.std(ddof=1)), always_pass=int((K == R).sum()), always_fail=int((K == 0).sum()),
        mixed=int(((K > 0) & (K < R)).sum()), within_var=float(s2.mean()), between_var=float(p.var(ddof=1)), icc=float(1 - s2.mean() / (p.var(ddof=1) * R / (R - 1) + 1e-12)) if p.var(ddof=1) > 0 else None,
        bb_ab=[float(a), float(b)], headroom_pts=100 * float(1 - p.mean()), headroom_mixed_pts=100 * float(((1 - p) * (K < R)).mean()),
        se_paired_avg3_pts=100 * float((2 * s2.mean() / (R * len(tasks))) ** .5), se_paired_single_pts=100 * float((2 * (p * (1 - p)).mean() / len(tasks)) ** .5))


class Setting:
    """common random numbers for one (bench, panel size, executions per arm): resampled tasks, posterior p_t, uniforms of both arms."""
    def __init__(self, rng, K, R, a, b, n_panel, reps):
        self.reps, self.mask = reps, None
        P = rng.beta(K + a, R - K + b, size=(NS, len(K))); self.idx = rng.integers(0, len(K), size=(NS, n_panel)); self.mix = (K < R)[self.idx]
        self.p = np.take_along_axis(P, self.idx, 1); self.u0 = rng.random((NS, n_panel, reps)); self.u1 = rng.random((NS, n_panel, reps)); self.base = (self.u0 < self.p[..., None]).sum(2) / reps
        self.crit = tdist.ppf(1 - ALPHA / 2, n_panel - 1)
    def headroom(self, mixed): w = self.mix if mixed else 1; return 100 * float((((1 - self.p) * w).mean(1)).mean())
    def shifted(self, delta, mixed):
        """candidate p'_t: clip(p_t + c) on the shiftable tasks, c by bisection so the panel mean rises by delta points (per simulation); nan when delta exceeds the mean headroom over simulations."""
        w = self.mix if mixed else np.ones_like(self.p, bool); lo, hi = np.zeros(NS), np.ones(NS); tgt = delta / 100
        gain = lambda c: (np.where(w, np.clip(self.p + c[:, None], 0, 1), self.p) - self.p).mean(1)
        if 100 * tgt > self.headroom(mixed): return None
        for _ in range(30): mid = (lo + hi) / 2; ok = gain(mid) >= tgt; hi = np.where(ok, mid, hi); lo = np.where(ok, lo, mid)
        return np.where(w, np.clip(self.p + hi[:, None], 0, 1), self.p)
    def power(self, delta, mixed=False):
        """P(two-sided paired t over tasks rejects at alpha and the candidate is better) for a mean shift of delta points; nan if the shift is infeasible."""
        q = self.shifted(delta, mixed) if delta > 0 else self.p
        if q is None: return float("nan")
        d = (self.u1 < q[..., None]).sum(2) / self.reps - self.base; m, s = d.mean(1), d.std(1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"): z = np.where(s > 0, m / (s / d.shape[1] ** .5), 0.0)
        return float((z > self.crit).mean())
    def mde(self, mixed=False):
        """smallest delta (points) with power >= TARGET (grid step 2, then bisection); None when not reached below the headroom."""
        prev, d = 0.0, 2.0
        while True:
            pw = self.power(d, mixed)
            if pw != pw: return None
            if pw >= TARGET: break
            prev, d = d, d + 2.0
        lo, hi = prev, d
        for _ in range(8):
            mid = (lo + hi) / 2
            if self.power(mid, mixed) >= TARGET: hi = mid
            else: lo = mid
        return float(hi)


def nosk_gap(m, b, seed_ex):
    """No skill vs Seed on the available repetitions (pending reps noted): per-task means, same-repetition Seed reference and the Seed avg@3."""
    c = m["cells"][b]["native-noseed"]; reps = sorted(int(r) for r in (c.get("executions") or {}))
    if not reps: return dict(status=c["status"], reps=[])
    ns = [load_exec(c["executions"][str(r)]["file"]) for r in reps]
    mean = lambda ex: {t: np.mean([e["succ"][k] for e in ex for k in e["succ"] if k[0] == t]) for t in {k[0] for e in ex for k in e["succ"]}}
    N, S3 = mean(ns), mean(seed_ex); S1 = mean(seed_ex[:len(ns)]); ts = sorted(set(N) & set(S3))
    f = lambda X, Y: (100 * float(np.mean([N[t] - Y[t] for t in ts])), float(ttest_rel([N[t] for t in ts], [Y[t] for t in ts]).pvalue))
    return dict(status=c["status"], reps=reps, pending_reps=[r for r in (0, 1, 2) if r not in reps], n_tasks=len(ts), noskill_avg=100 * float(np.mean([N[t] for t in ts])),
                seed_avg3=100 * float(np.mean([S3[t] for t in ts])), gap_vs_seed_avg3=f(N, S3)[0], p_vs_seed_avg3=f(N, S3)[1], seed_same_reps=[int(i) for i in range(len(ns))],
                gap_vs_seed_rep_matched=f(N, S1)[0], p_rep_matched=f(N, S1)[1], note="gap = No skill - Seed (negative: the seed skill helps); paired t over tasks, informational")


# ---- main -------------------------------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.environ.get("BACH_SUBMISSION_MANIFEST")); ap.add_argument("--out", required=True)
    args = ap.parse_args(); os.environ["BACH_SUBMISSION_MANIFEST"] = args.manifest; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m = common.manifest(); log = []; res = dict(seed=SEED, sims=NS, alpha=ALPHA, target=TARGET, manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), lit_gains_sha256=hashlib.sha256(LIT.read_bytes()).hexdigest(),
                                               model=__doc__.split("Model (per simulation):")[1].split("Test:")[0].strip(), benches={})
    for bi, b in enumerate(BENCHES):
        ex = seed_execs(m, b, "test", log); tasks, K, R, a, bb, st = seed_stats(ex); n = len(tasks); n_dev = m["cells"][b]["native-seed"]["panels"]["sel"]["n"]
        cfg = dict(avg3=(n, R), single=(n, 1), dev=(n_dev, 1)); r = dict(stats=st, n_dev=n_dev, settings={}, nosk=nosk_gap(m, b, ex))
        for si, (name, (npan, reps)) in enumerate(cfg.items()):
            S = Setting(np.random.default_rng([SEED, bi, si]), K, R, a, bb, npan, reps); r["settings"][name] = dict(n_tasks=npan, executions_per_arm=reps, **{
                f"mde80_{mod}": S.mde(mod == "mixed") for mod in ("uniform", "mixed")}, headroom_uniform=S.headroom(False), headroom_mixed=S.headroom(True),
                curve_uniform={str(d): S.power(d) for d in CURVE}, curve_mixed={str(d): S.power(d, True) for d in CURVE})
            r["settings"][name]["_S"] = S
        res["benches"][b] = r
    rep = []  # reported gains vs our per-bench power (uniform model; avg3 and single)
    for cls, paper, label, g, nt, ag in GAINS:
        row = dict(cls=cls, paper=paper, label=label, gain_pp=g, n_test=nt, agentic=ag, per_bench={})
        for b in BENCHES:
            o = {}
            for name in ("avg3", "single"):
                S = res["benches"][b]["settings"][name]["_S"]; mde = res["benches"][b]["settings"][name]["mde80_uniform"]
                pw = S.power(g) if g > 0 else None
                o[name] = dict(power=None if pw is None or pw != pw else pw, above_headroom=bool(g > res["benches"][b]["settings"][name]["headroom_uniform"]), detectable80=bool(mde is not None and g >= mde))
            row["per_bench"][b] = o
        rep.append(row)
    for r in res["benches"].values():
        for s in r["settings"].values(): s.pop("_S")
    res["reported_gains"] = rep
    (out/"power.json").write_text(json.dumps(res, indent=1)); (out/"power.md").write_text(render(res)); print((out/"power.md").read_text())


def f1(x): return "-" if x is None else f"{x:.1f}"
def cell(mde, head): return f"{mde:.1f}" if mde is not None else f"none (headroom {head:.0f})"


def render(res):
    B, L = res["benches"], []; P = lambda s: " | ".join(s)
    L += ["# Power of the submission protocol", "", f"Seed {res['seed']}, {res['sims']} simulations per setting, two-sided alpha {res['alpha']}, power target {res['target']}. Manifest sha256 {res['manifest_sha256'][:12]}, lit_gains.json sha256 {res['lit_gains_sha256'][:12]}.", "",
          "Model: " + res["model"].replace("\n", " "), "", "## Seed test executions (the variance input)", "",
          "| bench | tasks | seed avg@3 | exec rates (%) | exec SD (pts) | always-pass / always-fail / mixed | within-task var | between-task var | SE of paired avg@3 diff (pts) | SE single-run diff (pts) | headroom (pts) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in B.items():
        s = r["stats"]; L.append(f"| {b} | {s['n_tasks']} | {s['seed_avg']:.1f} | {', '.join(f1(x) for x in s['exec_rates'])} | {s['exec_sd_pts']:.1f} | {s['always_pass']} / {s['always_fail']} / {s['mixed']} | {s['within_var']:.3f} | {s['between_var']:.3f} | {s['se_paired_avg3_pts']:.1f} | {s['se_paired_single_pts']:.1f} | {s['headroom_pts']:.1f} (mixed-only {s['headroom_mixed_pts']:.1f}) |")
    L += ["", "## MDE80 (points; two-sided paired t over tasks)", "", "Uniform shift model; the mixed-task shift model in parentheses. `none` = power stays below 80% up to the headroom.", "",
          "| bench | MDE80 avg@3 paired (test) | MDE80 single run (test) | MDE80 dev panel (1 exec) |", "|---|---|---|---|"]
    for b, r in B.items():
        c = lambda k: f"{cell(r['settings'][k]['mde80_uniform'], r['settings'][k]['headroom_uniform'])} ({cell(r['settings'][k]['mde80_mixed'], r['settings'][k]['headroom_mixed'])})"
        L.append(f"| {b} | {c('avg3')} | {c('single')} | {c('dev')} [{r['n_dev']} tasks] |")
    L += ["", "Power at fixed shifts (uniform model; avg@3 / single / dev):", "", "| bench | " + P(f"{d} pts" for d in CURVE) + " |", "|---|" + "---|" * len(CURVE)]
    for b, r in B.items(): L.append(f"| {b} | " + P(" / ".join(f"{100*r['settings'][k]['curve_uniform'][str(d)]:.0f}" if r['settings'][k]['curve_uniform'][str(d)] == r['settings'][k]['curve_uniform'][str(d)] else "n/a" for k in ("avg3", "single", "dev")) for d in CURVE) + " |")
    L += ["", "## Our No skill vs Seed gap (current data)", "", "| bench | status | reps used | pending reps | tasks | No skill avg | Seed avg@3 | gap vs avg@3 (p) | gap vs same-rep Seed (p) |", "|---|---|---|---|---|---|---|---|---|"]
    for b, r in B.items():
        g = r["nosk"]
        L.append(f"| {b} | {g['status']} | {g['reps']} | {g.get('pending_reps')} | " + (f"{g['n_tasks']} | {g['noskill_avg']:.1f} | {g['seed_avg3']:.1f} | {g['gap_vs_seed_avg3']:+.1f} ({g['p_vs_seed_avg3']:.2f}) | {g['gap_vs_seed_rep_matched']:+.1f} ({g['p_rep_matched']:.2f}) |" if g["reps"] else "- | - | - | - | - |"))
    L += ["", "## Reported gains vs our MDE80", "", "Cell = power of the avg@3 paired test at that shift (uniform model), then `yes` if gain >= MDE80 avg@3, `no` otherwise; `>head` = gain exceeds the seed's headroom on that bench (unreachable), `gain<=0` = no gain to detect. Single-run column in power.json. `ag` marks agentic multi-turn / tool-using entries; gain in points on the paper's own scale and model, not transferred.", ""]
    for cls in "abc":
        L += [f"### ({cls}) {CLS[cls]}", "", "| paper | entry | gain (pp) | test n | ag | " + P(BENCHES) + " |", "|---|---|---|---|---|---|---|---|"]
        for ag in (True, False):
            for e in res["reported_gains"]:
                if e["cls"] != cls or e["agentic"] != ag: continue
                cs = []
                for b in BENCHES:
                    o = e["per_bench"][b]["avg3"]; cs.append("gain<=0" if e["gain_pp"] <= 0 else ">head" if o["above_headroom"] else "n/a" if o["power"] is None else ( f"{100*o['power']:.0f}% {'yes' if o['detectable80'] else 'no'}"))
                L.append(f"| {e['paper']} | {e['label']} | {e['gain_pp']:+.1f} | {e['n_test'] or 'n/s'} | {'ag' if ag else ''} | {' | '.join(cs)} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
