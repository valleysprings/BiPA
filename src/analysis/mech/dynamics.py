"""Paper 1 (mechanism paper): trajectory dynamics behind the H1 / H2 / H4 hypotheses, on the avg@3 test runs of both models.

H1 (skills lower the number of attractors): the three test runs of one task should look more alike under a learned skill than under
the seed or no skill. Signature of an episode = the sequence of executed tool-call names (turn order). Per task: mean pairwise
similarity over the 3 run pairs, with two similarity measures
  - jacc: Jaccard index of the tool-name sets (1.0 when both episodes make no call);
  - editsim: 1 - Levenshtein(a, b) / max(len(a), len(b)) on the name sequences (1.0 when both are empty);
and the number of distinct signatures among the 3 runs (behavioural modes, 1..3). Also h = X / R (mean input tokens per turn),
the per-turn content the hypothesis says should shrink.

H4 (skills are knowledge triggers, pass@k diversity): pass@1 = per-task success rate over the 3 runs averaged over tasks
(= Table 1's Q when every run covers every task), pass@3 = share of tasks solved in at least one of the 3 runs, gap = pass@3 - pass@1.

H2 (skills just squeeze more thinking; test-time scaling): the model's own test-time curve = seed-episode success rate binned by
turns R (and by input tokens X), quantile bins over the pooled seed episodes of one (model, benchmark). Against it, per learned
skill and task: dR = mean R (skill) - mean R (seed), dQ likewise; "wasted squeeze" = tasks with dR > 0 whose dQ <= 0, as a share of
the dR > 0 tasks and of all tasks. For tasks with dR > 0: the seed curve's success rate at the bin of the task's seed mean R
(origin) and at the bin of its skill mean R (destination) — whether the extra turns buy curve segments that still pay.

Runs: DeepSeek-V4.1-Flash = Table 1's own runs (seed native-seed + rep-1/rep-2; skills and no-skill native-* + test-reps rep 1/2,
picked by analysis.table1.rep_files exactly as Table 1). GPT-6 Luna = the avg@3 repeat protocol of bipa/src/repeat.py re-implemented
here (rep 0 = the source run's first evaluation; rep 1/2 = assets/artifacts/execution/gpt-6-luna/test-reps/<bench>__<arm>__<tag>,
unknown-only supplement evaluations merged per slot; TB2's three optimizer arms returned the seed text and report the seed row).
Episode fields are read straight from evaluation.json as analysis.mech.transfer / analysis.mech.callsperturn do: requests deduplicated
by request_id, provider-failed requests (no usage) dropped, X = summed input tokens, N = executed tool calls, R = completed requests;
an outcome is verified when success is a boolean and outcome_valid is not False (repeat.py's "known").

TB2 has 18 tasks and one permanently unknown task (torch-pipeline-parallelism); its cells are reported but the evidence is tau3 (75)
and AutoBench (120). No BACH, no model calls.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.dynamics   -> assets/cache/analysis/mech/dynamics.json, data/mech/mech_dynamics.md
"""
import collections
import json
import math

import numpy as np

from src.analysis import table1 as T
from src.analysis.common import SEED_TEXT, current_run, final_skill, vid, EX
from src.analysis.mech import common as MC

MODELS = (("deepseek-flash", "DeepSeek"), ("gpt-6-luna", "Luna"))
ARMS = [("native-noseed", "No skill"), ("native-seed", "Seed skill"),
        ("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
LUNA = EX / "gpt-6-luna"
LUNA_TAG = {"native-noseed": "native-20261011", "native-seed": "native-20261011",
            "gepa": "native-20261012", "skillopt": "native-20261012", "skilladam": "native-20261012"}
N_BINS = 6


# ---------- episode extraction (shared by both models) ----------

def ep_metrics(raw):
    """One episode's dynamics fields: success (None unless verified), X, N, R and the executed tool-name sequence."""
    seen, reqs = set(), []
    for r in raw.get("requests", []):
        if r["request_id"] in seen:
            continue
        seen.add(r["request_id"])
        if r.get("input_tokens") is None:  # provider-failed request (Luna); the client retried it
            continue
        reqs.append(r)
    seq = [c["tool"] for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False]
    ok = type(raw.get("success")) is bool and raw.get("outcome_valid", True) is not False
    return {"success": bool(raw["success"]) if ok else None,
            "X": sum(r["input_tokens"] for r in reqs), "N": len(seq), "R": len(reqs), "seq": seq}


def rows_from_file(path):
    """{task_id: ep_metrics} of one evaluation.json (first episode per task slot, as table1.rows_of)."""
    data = json.loads(open(path).read())
    eps = {}
    for e in data["episodes"]:
        eps.setdefault(e["task_id"], e)
    return {t: ep_metrics(e) for t, e in eps.items()}


# ---------- DeepSeek: Table 1's own run selection ----------

def deepseek_reps(arm, bench):
    """{rep: evaluation.json} of one arm on one benchmark, exactly the files Table 1 averages."""
    run = current_run(arm, "deepseek-flash", bench)
    text = final_skill(arm, run)
    if arm == "native-seed":
        dirs = [run, run.parent / "rep-1", run.parent / "rep-2"]
    else:
        dirs = [run, EX / "deepseek-flash" / "test-reps" / f"{bench}__{arm}__{run.name}"]
    return T.rep_files(dirs, vid(text))


# ---------- Luna: repeat.py's avg@3 merge, re-implemented (importing bipa/src/repeat.py pulls the executor stack) ----------

def _slot(row):
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def _valid_slots(data):
    panel = [_slot(r) for r in data.get("panel", [])]
    episodes = [_slot(r) for r in data.get("episodes", [])]
    return len(set(panel)) == len(panel) and len(set(episodes)) == len(episodes) and set(episodes) <= set(panel)


def _base_eval(directories, v, tasks, repetition):
    expected = collections.Counter((t, repetition) for t in tasks)
    for directory in directories:
        if not directory or not (directory / "results/evaluations/test").is_dir():
            continue
        for path in sorted((directory / "results/evaluations/test").glob("*/evaluation.json")):
            data = json.loads(path.read_text())
            if (data.get("version_id") == v and data.get("split") == "test"
                    and collections.Counter(_slot(r) for r in data.get("panel", [])) == expected
                    and _valid_slots(data)):
                return path, data
    return None, None


def luna_reps(arm, bench):
    """{rep: {task_id: ep_metrics}} of one Luna arm: rep 0 from the source run, rep 1/2 from test-reps,
    unknown-only supplement evaluations overriding the slot's earlier episode (repeat.py collect())."""
    run = LUNA / arm / bench / LUNA_TAG[arm]
    out = LUNA / "test-reps" / f"{bench}__{arm}__{LUNA_TAG[arm]}"
    out = out if out.is_dir() else None
    summary = json.loads((run / "summary.json").read_text())
    v = vid(summary["final_skill"])
    tasks = [_slot(r)[0] for r in json.loads((run / "native_protocol.json").read_text())["panels"]["test"]]
    reps = {}
    for k in range(3):
        path, data = _base_eval([run, out] if k == 0 else [out, run.parent / f"rep-{k}"], v, tasks, k)
        if data is None:
            continue
        merged, known = {}, set()
        for e in data.get("episodes", []):
            merged[_slot(e)[0]] = e
            if type(e.get("success")) is bool and e.get("outcome_valid", True) is True:
                known.add(_slot(e)[0])
        expected = {(t, k) for t in tasks}
        if out:
            for supplement in sorted((out / "results/evaluations/test").glob("*/evaluation.json")):
                extra = json.loads(supplement.read_text())
                panel = [_slot(r) for r in extra.get("panel", [])]
                if (extra.get("version_id") != v or extra.get("split") != "test" or not panel
                        or not _valid_slots(extra) or not set(panel) <= expected):
                    continue
                # repeat.py keeps the first verified verdict of a slot; a supplement only takes over slots still unknown
                for e in extra.get("episodes", []):
                    if _slot(e) in panel and _slot(e)[0] not in known:
                        merged[_slot(e)[0]] = e
                        if type(e.get("success")) is bool and e.get("outcome_valid", True) is True:
                            known.add(_slot(e)[0])
        reps[k] = {t: ep_metrics(e) for t, e in merged.items()}
    return reps


def load_cell(model, arm, bench):
    """{rep: {task: ep_metrics}} for one (model, arm, bench)."""
    if model == "gpt-6-luna":
        return luna_reps(arm, bench)
    return {rep: rows_from_file(path) for rep, path in deepseek_reps(arm, bench).items()}


# ---------- H1: trajectory convergence ----------

def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def editsim(a, b):
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def h1_cell(reps):
    """H1 numbers of one arm's 3 runs: per task mean pairwise similarity, modes, h; task sets need 3 episodes with R > 0."""
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in sorted(reps)))
             if all(reps[k][t]["R"] > 0 for k in sorted(reps))] if len(reps) == 3 else []
    jac, edit, modes, hh, rr, xx = [], [], [], [], [], []
    for t in tasks:
        seqs = [reps[k][t]["seq"] for k in sorted(reps)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        jac.append(float(np.mean([jaccard(seqs[i], seqs[j]) for i, j in pairs])))
        edit.append(float(np.mean([editsim(seqs[i], seqs[j]) for i, j in pairs])))
        modes.append(len({tuple(s) for s in seqs}))
        hh.append(float(np.mean([reps[k][t]["X"] / reps[k][t]["R"] for k in sorted(reps)])))
        rr.append(float(np.mean([reps[k][t]["R"] for k in sorted(reps)])))
        xx.append(float(np.mean([reps[k][t]["X"] for k in sorted(reps)])))
    if not tasks:
        return None
    return {"n": len(tasks), "jacc": float(np.mean(jac)), "editsim": float(np.mean(edit)),
            "modes": float(np.mean(modes)), "one_mode_share": float(np.mean([m == 1 for m in modes])),
            "h": float(np.mean(hh)), "R": float(np.mean(rr)), "X": float(np.mean(xx))}


# ---------- H4: pass@k ----------

def h4_cell(reps):
    """pass@1 / pass@3 over the tasks with a verified outcome in all 3 runs."""
    if len(reps) != 3:
        return None
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in sorted(reps)))
             if all(reps[k][t]["success"] is not None for k in sorted(reps))]
    if not tasks:
        return None
    wins = {t: sum(reps[k][t]["success"] is True for k in sorted(reps)) for t in tasks}
    return {"n": len(tasks), "pass1": float(100 * np.mean([wins[t] / 3 for t in tasks])),
            "pass3": float(100 * np.mean([wins[t] >= 1 for t in tasks]))}


# ---------- H2: seed test-time curve and the squeeze ----------

def seed_curve(seed_reps):
    """The model's own test-time curve: seed episodes (verified outcomes) binned by R and by X, N_BINS quantile bins."""
    eps = [seed_reps[k][t] for k in sorted(seed_reps) for t in seed_reps[k] if seed_reps[k][t]["success"] is not None]
    out = {}
    for key in ("R", "X"):
        vals = np.array([e[key] for e in eps], dtype=float)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS + 1)))
        bins = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = [e for e in eps if (e[key] >= lo if i == 0 else e[key] > lo) and e[key] <= hi]
            if sel:
                bins.append({"lo": float(lo), "hi": float(hi), "n": len(sel),
                             "success": float(100 * np.mean([e["success"] for e in sel]))})
        out[key] = bins
    return out


def bin_of(bins, value):
    """Index of the curve bin holding value (last bin when above the top edge)."""
    for i, b in enumerate(bins):
        if value <= b["hi"]:
            return i
    return len(bins) - 1


def h2_cell(own_reps, seed_reps, curve_bins):
    """Per-task dR / dQ against the seed on tasks with 3 verified runs on both sides; wasted-squeeze shares; curve segments."""
    if len(own_reps) != 3 or len(seed_reps) != 3:
        return None
    tasks = [t for t in set.intersection(*(set(own_reps[k]) for k in range(3)), *(set(seed_reps[k]) for k in range(3)))
             if all(own_reps[k][t]["success"] is not None and seed_reps[k][t]["success"] is not None for k in range(3))]
    if not tasks:
        return None
    rows = []
    for t in tasks:
        ro = float(np.mean([own_reps[k][t]["R"] for k in range(3)]))
        rs = float(np.mean([seed_reps[k][t]["R"] for k in range(3)]))
        qo = float(np.mean([float(own_reps[k][t]["success"]) for k in range(3)]))
        qs = float(np.mean([float(seed_reps[k][t]["success"]) for k in range(3)]))
        rows.append({"task": t, "dR": ro - rs, "dQ": qo - qs, "R_own": ro, "R_seed": rs})
    dr = np.array([r["dR"] for r in rows])
    dq = np.array([r["dQ"] for r in rows])
    up = dr > 0
    wasted = up & (dq <= 0)
    origin, dest = [], []
    for r in rows:
        if r["dR"] > 0 and curve_bins:
            origin.append(curve_bins[bin_of(curve_bins, r["R_seed"])]["success"])
            dest.append(curve_bins[bin_of(curve_bins, r["R_own"])]["success"])
    return {"n": len(tasks), "dR": float(dr.mean()), "dQ": float(100 * dq.mean()),
            "share_dR_up": float(up.mean()), "n_up": int(up.sum()), "n_wasted": int(wasted.sum()),
            "wasted_of_up": float(wasted.sum() / up.sum()) if up.sum() else None,
            "wasted_of_all": float(wasted.mean()),
            "curve_at_origin": float(np.mean(origin)) if origin else None,
            "curve_at_dest": float(np.mean(dest)) if dest else None}


# ---------- driver ----------

def luna_seed_skill(arm, bench):
    """True when the Luna optimizer run returned the seed text (TB2): the row reports the seed runs."""
    if arm in ("native-noseed", "native-seed"):
        return arm == "native-seed"
    run = LUNA / arm / bench / LUNA_TAG[arm]
    return json.loads((run / "summary.json").read_text())["final_skill"] == SEED_TEXT


def compute():
    res = {}
    for model, mshort in MODELS:
        for bench, bshort in BENCH:
            arms = {}
            for arm, label in ARMS:
                if model == "gpt-6-luna" and arm not in ("native-noseed", "native-seed") and luna_seed_skill(arm, bench):
                    arms[arm] = {"label": label, "shared_seed": True}
                    continue
                reps = load_cell(model, arm, bench)
                arms[arm] = {"label": label, "shared_seed": False, "reps": reps,
                             "h1": h1_cell(reps), "h4": h4_cell(reps)}
            seed_reps = arms["native-seed"].get("reps")
            curve = seed_curve(seed_reps) if seed_reps else None
            for arm, label in ARMS:
                cell = arms[arm]
                if cell.get("shared_seed") or arm in ("native-noseed", "native-seed") or not cell.get("reps"):
                    cell["h2"] = None
                    continue
                cell["h2"] = h2_cell(cell["reps"], seed_reps, curve["R"] if curve else None)
            res[f"{model}|{bench}"] = {"bench": bench, "short": bshort, "model": model, "curve": curve, "arms": arms}
            print(model, bench, "loaded")
    return res


# ---------- record ----------

def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def record(res):
    h1_rows, h1_delta, h4_rows, c1_rows, c2_rows = [], [], [], [], []
    for key, cell in res.items():
        model, bench = key.split("|")
        mshort = dict(MODELS)[model]
        for arm, label in ARMS:
            a = cell["arms"][arm]
            if a.get("shared_seed"):
                h1_rows.append([mshort, cell["short"], label, "= Seed row (returned the seed)"])
                h4_rows.append([mshort, cell["short"], label, "= Seed row"])
                continue
            h1, h4 = a["h1"], a["h4"]
            if h1:
                h1_rows.append([mshort, cell["short"], label, h1["n"], fmt(h1["jacc"], 3), fmt(h1["editsim"], 3),
                                fmt(h1["modes"]), f"{100 * h1['one_mode_share']:.0f}%", f"{h1['h']:.0f}", fmt(h1["R"], 1), f"{h1['X']:.0f}"])
            if h4:
                h4_rows.append([mshort, cell["short"], label, h4["n"], f"{h4['pass1']:.1f}", f"{h4['pass3']:.1f}", f"{h4['pass3'] - h4['pass1']:+.1f}"])
        seed_h1 = cell["arms"]["native-seed"]["h1"]
        for arm, label in ARMS[2:]:
            a = cell["arms"][arm]
            if a.get("shared_seed") or not a.get("h1") or not seed_h1:
                continue
            h1_delta.append([mshort, cell["short"], label,
                             f"{a['h1']['jacc'] - seed_h1['jacc']:+.3f}", f"{a['h1']['editsim'] - seed_h1['editsim']:+.3f}",
                             f"{a['h1']['modes'] - seed_h1['modes']:+.2f}", f"{a['h1']['h'] / seed_h1['h'] - 1:+.2f}"])
        if cell["curve"]:
            for i, b in enumerate(cell["curve"]["R"]):
                c1_rows.append([mshort, cell["short"], "turns R", i, f"{b['lo']:.0f}-{b['hi']:.0f}", b["n"], f"{b['success']:.1f}"])
            for i, b in enumerate(cell["curve"]["X"]):
                c1_rows.append([mshort, cell["short"], "input tokens X", i, f"{b['lo'] / 1000:.0f}k-{b['hi'] / 1000:.0f}k", b["n"], f"{b['success']:.1f}"])
        for arm, label in ARMS[2:]:
            a = cell["arms"][arm]
            if a.get("h2"):
                h = a["h2"]
                wasted = f"{100 * h['wasted_of_up']:.0f}% ({h['n_wasted']}/{h['n_up']})" if h["wasted_of_up"] is not None else "-"
                c2_rows.append([mshort, cell["short"], label, h["n"], f"{h['dR']:+.2f}", f"{100 * h['share_dR_up']:.0f}%",
                                f"{h['dQ']:+.1f}", wasted,
                                f"{100 * h['wasted_of_all']:.0f}%", fmt(h["curve_at_origin"], 1), fmt(h["curve_at_dest"], 1)])
    what = ("Trajectory dynamics for three mechanism hypotheses, on the avg@3 test runs of DeepSeek-V4.1-Flash (Table 1's runs) and GPT-6 Luna "
            "(the repeat-protocol runs). H1: convergence of a task's 3 runs — pairwise tool-name-sequence similarity (Jaccard on name sets; "
            "editsim = 1 - normalized Levenshtein) and behavioural modes (distinct call-name signatures among the 3 runs), plus h = X/R (input "
            "tokens per turn). H4: pass@1 (per-task success rate averaged over tasks) vs pass@3 (tasks solved in >= 1 of 3 runs). "
            "H2: the seed's own test-time curve (success binned by turns / input tokens) and per-task dR vs dQ of each learned skill against the "
            "seed; 'wasted squeeze' = tasks with more turns but no success gain.")
    runs = ("DeepSeek: seed = native-seed + rep-1/rep-2; no-skill and skills = their native runs + test-reps rep 1/2 (analysis.table1.rep_files, "
            "Table 1's selection). Luna: rep 0 = source run (native-20261011 native arms, native-20261012 optimizers), rep 1/2 = "
            "assets/artifacts/execution/gpt-6-luna/test-reps/<bench>__<arm>__<tag>, unknown-only supplements merged per slot (bipa/src/repeat.py's "
            "protocol). Luna TB2's GEPA/SkillOpt/SkillAdam returned the seed text: their rows are the Seed row and are marked, not recomputed.")
    agg = ("Episode fields from evaluation.json: requests deduplicated by request_id, provider-failed requests (no usage) dropped; X = summed input "
           "tokens, N = executed tool calls, R = completed requests, signature = executed tool names in turn order. Verified outcome = boolean "
           "success with outcome_valid not False. H1 task set: 3 runs with R > 0. H4/H2 task set: verified outcome in all 3 runs (both sides for H2). "
           "Curve bins: 6 quantile bins over the pooled seed episodes of the (model, benchmark); origin/dest = the curve success at the bin of the "
           "task's seed mean R / skill mean R, averaged over the tasks with dR > 0. TB2 has 18 tasks and one permanently unknown task "
           "(torch-pipeline-parallelism); the evidence is tau3 (75) and AutoBench (120).")
    table = ("### A. H1: run-to-run trajectory similarity, behavioural modes, per-turn content h = X/R\n\n"
             + MC.md_table(["Model", "Bench", "Arm", "n tasks", "jacc", "editsim", "modes (of 3)", "1-mode %", "h = X/R", "R", "X"], h1_rows)
             + "\n\n### A2. H1: learned skill minus Seed (same benchmark)\n\n"
             + MC.md_table(["Model", "Bench", "Arm", "d jacc", "d editsim", "d modes", "d h (relative)"], h1_delta)
             + "\n\n### B. H4: pass@1 vs pass@3 over the 3 runs\n\n"
             + MC.md_table(["Model", "Bench", "Arm", "n tasks", "pass@1 %", "pass@3 %", "gap (pts)"], h4_rows)
             + "\n\n### C1. H2: the seed's own test-time curve (success % by quantile bin)\n\n"
             + MC.md_table(["Model", "Bench", "x", "bin", "range", "n episodes", "success %"], c1_rows)
             + "\n\n### C2. H2: learned skill vs seed, per task (dR in turns, dQ in points; wasted = dR > 0 and dQ <= 0)\n\n"
             + MC.md_table(["Model", "Bench", "Arm", "n tasks", "mean dR", "dR>0 %", "mean dQ", "wasted of dR>0", "wasted of all",
                            "seed curve % at origin bin", "seed curve % at dest bin"], c2_rows))
    MC.write_json("dynamics", {"cells": {k: {"curve": v["curve"],
                                             "arms": {a: {"label": x["label"], "shared_seed": x.get("shared_seed", False),
                                                          "h1": x.get("h1"), "h4": x.get("h4"), "h2": x.get("h2")}
                                                      for a, x in v["arms"].items()}} for k, v in res.items()}})
    MC.write_record("dynamics", what, runs, agg, table, ".venv/bin/python -m src.analysis.mech.dynamics")


def main():
    record(compute())


if __name__ == "__main__":
    main()
