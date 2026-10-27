"""Where a learned skill's extra input tokens come from: the four-term split of its change in log input tokens against
the seed under both models, and the same split on the tasks both solve or both fail.

The split (aux.tokens.split) is per task, on the means of R, N, X and the prefix F over the test runs of each
side (DeepSeek: all three runs; GPT-6-Luna: the runs Table 1 averages): more actions e_R dlogN, turn-splitting
-e_R dlog(N/R), content per turn s_H dlogh and skill text s_F dlogF, with midpoint shares and the exact
e_R = 1 + s_H R / (R - 1). A task is usable when every averaged run has a graph episode with R >= 2, an executed call and
positive history. The residual is the measured total minus the sum of the four terms. mech_attribution3 also splits each
run pair (run k of the skill against run k of the seed, k = 0, 1, 2) and gives the per-request change of each prefix column.

Shares of the summed log-token increase over a set of skills sum the per-skill task means (task-weighted: every task of
every skill); gross shares sum each term's absolute size. Intervals resample the tasks of a benchmark with replacement,
one draw set per model and benchmark, shared by every skill of that benchmark (they share the seed); each skill averages
its per-task terms over the drawn tasks it uses. The Luna turn slopes are fitted within tasks (src.graph.metrics.fe_reg)
on the first test run of the seed and of each learned skill, as mech_slopes is for DeepSeek, and on all runs; a task with
one episode carries no within-task information and drops out.

mech_failcost splits each DeepSeek skill's usable tasks by Table 1's verified outcomes into those that the skill and the
seed both solve in every run, both fail in every run, and the rest. How a failed episode ends is its recorded backend
classification (task_failure: the agent stopped; timeout: the harness's wall-clock limit; no turn cap is recorded). The
contrast is the mean over cells of the both-fail minus the both-solve task mean; its interval resamples the tasks within
each cell and subset, one draw set per cell and subset."""
import json, math

import numpy as np

from src.analysis import settings as S
from aux.tokens.split import OPT, TERM_LABEL, TERMS, four_terms, side, split, summarize
from src.graph.metrics import fe_reg
from aux.tokens.slopes import output_tokens
from src.utils.logger import md_table, record

SHORT = S.BENCH_SHORT
SUBSETS = [("solve", "(a) both solve 3/3"), ("fail", "(b) both fail 3/3"), ("mixed", "(c) mixed")]
f3 = "{:+.3f}".format


def ci(x, nd=3, pct=False):
    return f"[{100 * x[0]:.0f}%, {100 * x[1]:.0f}%]" if pct else f"[{x[0]:+.{nd}f}, {x[1]:+.{nd}f}]"


def cell_list(model):
    """The cells a record of a model's learned skills reads: the skills and their benchmarks' seed."""
    learned = S.learned([model])
    return learned + [(model, b, "native-seed") for b in dict.fromkeys(b for _, b, _ in learned)]


def attribution3():
    cells = json.loads((S.cache("main") / "attribution.json").read_text())["cells"]
    rows_a = [[c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f3(c["A"][k]) for k in TERMS + ["total"]]
              + [f3(c["table1_log_rho_usable"]), f3(c["A"]["residual_mean"]), f"{c['A']['residual_max_abs']:.3f}", TERM_LABEL[c["A"]["largest"]]] for c in cells]
    rows_b = [[c["method"], c["short"], b["run"]] + [f3(b[k]) for k in TERMS + ["total", "residual_mean"]] + [TERM_LABEL[b["largest"]]] for c in cells for b in c["B"]]
    rows_s = [[c["method"], c["short"], " / ".join(TERM_LABEL[b["largest"]] for b in c["B"]), "yes" if c["same_largest_all_runs"] else "no"] for c in cells]
    rows_c = [[c["method"], c["short"]] + [f"{v['delta']:+.0f} (seed {v['seed']:.0f})" for v in c["C"].values()] for c in cells]
    rows_t = [[c["method"], c["short"], f"{c['table1_rho_full']:.3f}", f"{math.exp(c['table1_log_rho_usable']):.3f}", f"{math.exp(c['A']['total']):.3f}"] for c in cells]
    rows_d = [[c["method"], c["short"], c["n_table1"] - c["n"], "; ".join(f"{k}: {v}" for k, v in c["dropped"].items() if v) or "-"] for c in cells]
    terms = ["actions", "turn-split", "content", "skill text"]
    numbers = "\n\n".join([
        "### A. 3-run means (log units; n = tasks used / Table 1 common tasks)", md_table(
            ["Skill", "Bench", "n", *terms, "total", "Table 1 log rho (same tasks)", "resid mean", "resid max |.|", "largest"], rows_a),
        "### B. Per run pair (skill rep k vs seed rep k)", md_table(["Skill", "Bench", "run", *terms, "total", "resid mean", "largest"], rows_b),
        "Stability of the largest term (runs 0 / 1 / 2):", md_table(["Skill", "Bench", "largest per run", "same in all three (and in A)"], rows_s),
        "### C. Fixed-prefix columns, change per request in tokens (skill minus seed)", md_table(["Skill", "Bench", *cells[0]["C"]], rows_c),
        "### Table 1 consistency (rho_trj)",
        md_table(["Skill", "Bench", "Table 1 rho (full common set)", "rho on used tasks (Table 1 rows)", "exp(total) of A"], rows_t),
        "### Dropped tasks (Table 1 common set minus used)", md_table(["Skill", "Bench", "dropped", "reason"], rows_d)])
    what = ("Four-term split of the change in log input tokens of each DeepSeek learned skill against the seed, on the same data as "
            "Table 1: per task the 3-run means of both sides, so the total is Table 1's log rho on the usable tasks.")
    record(__spec__.name, "attribution3", numbers, S.runs_text(cell_list(S.DEEPSEEK)), what=what)


# ---- the split under both models, the shares of the summed increase and their intervals

def order(c):
    """Optimizer, then benchmark, as aux.tokens.split orders its cells."""
    return [d for d, _ in OPT].index(c["method"]), list(SHORT).index(c["bench"])


def cells_of(model):
    """Per learned skill of a model: the split, its per-task terms, and (GPT-6-Luna) the check of the episode graphs."""
    out = []
    for _, bench, method in S.learned([model]):
        c = split(model, bench, method)
        terms, dX = four_terms([u["pair"] for u in c["usable"].values()])

        def level(eps, runs, task):
            return float(np.mean([e.Xrg[:, e.comps.index("skill")].mean() if "skill" in e.comps else 0.0 for e in (eps[r][task][0] for r in runs)]))
        skill = [(level(c["own"], u["runs"][1], t), level(c["seed"], u["runs"][0], t)) for t, u in c["usable"].items()]
        mismatch = sum(abs(e.X - rows[r][t]["X"]) > 0.5 or (e.R, e.N) != (rows[r][t]["R"], rows[r][t]["N"])
                       for eps, rows in zip((c["seed"], c["own"]), c["rows"]) for r, rep in eps.items() for t, (e, _) in rep.items()
                       if model == S.LUNA and rows[r].get(t) and rows[r][t]["X"] is not None)
        out.append(dict(model=model, method=S.ARM_NAME[method], bench=bench, short=SHORT[bench], n=len(c["usable"]), n_table1=len(c["common"]),
                        n_full=sum(len(a) == len(b) == len(c["reps"]) for a, b in (u["runs"] for u in c["usable"].values())),
                        A=summarize(terms, dX), terms=terms, dX=dX, tasks=list(c["usable"]), mismatch=mismatch,
                        log_rho_full=float(np.mean(list(c["log_rho"].values()))),
                        d_skill=float(np.mean([a - b for a, b in skill])), seed_skill=float(np.mean([b for _, b in skill]))))
    return out


def draw_counts(cells):
    """{(model, bench): (task pool, N_BOOT x pool counts)}: one resample of each benchmark's tasks per model."""
    out = {}
    for model, bench in dict.fromkeys((c["model"], c["bench"]) for c in cells):
        pool = sorted({t for c in cells if (c["model"], c["bench"]) == (model, bench) for t in c["tasks"]})
        idx = S.boot_index(len(pool), "attribution shares", model, bench)
        counts = np.zeros(idx.shape)
        np.add.at(counts, (np.arange(len(idx))[:, None], idx), 1)
        out[(model, bench)] = pool, counts
    return out


def shares(cells, draws=None, weighted=False):
    """The shares of the summed log-token increase over a set of skills, or (given draws) their percentile intervals."""
    keys = TERMS + ["total"]
    if draws is None:
        agg = {k: [float(np.sum(c["dX"] if k == "total" else c["terms"][k])) if weighted else c["A"][k] for c in cells] for k in keys}
    else:
        agg = {k: [] for k in keys}
        for c in cells:
            pool, counts = draws[(c["model"], c["bench"])]
            w = counts[:, [pool.index(t) for t in c["tasks"]]]
            wsum = np.where(w.sum(1) == 0, np.nan, w.sum(1))
            for k in keys:
                v = w @ (c["dX"] if k == "total" else c["terms"][k])
                agg[k].append(v if weighted else v / wsum)
    s = {k: np.sum(v, axis=0) for k, v in agg.items()}
    turns, tot = s["more_actions"] + s["turn_splitting"], s["total"]
    out = dict(total=tot, turns=turns, text=s["skill_text"], content=s["content"], share_turns=turns / tot, share_text=s["skill_text"] / tot,
               share_content=s["content"] / tot)
    if draws is None:
        gross = sum(abs(x) for k in TERMS for x in agg[k])
        return out | dict(gross_turns=sum(abs(x) for k in TERMS[:2] for x in agg[k]) / gross, gross_text=sum(map(abs, agg["skill_text"])) / gross)
    q = {k: S.interval(v[~np.isnan(v)]) for k, v in out.items()}
    return q | dict(p_total_le_0=float(np.mean(tot <= 0)), p_turns_gt_text=float(np.mean(turns > s["skill_text"])),
                    p_turns_gt_half=float(np.mean(turns / tot > 0.5)))


def share_sets(by_model):
    """[(label, cells, with intervals, task-weighted)] in print order."""
    out = []
    for model, cells in by_model.items():
        name, benches = S.MODEL_SHORT[model], list(dict.fromkeys(c["short"] for c in cells))
        full = [c for c in cells if sum(d["bench"] == c["bench"] for d in cells) == len(S.OPTIMIZERS)]
        out.append((f"{name}, {len(cells)} skills" + (f" ({', '.join(benches)})" if len(benches) < len(SHORT) else ""), cells, True, False))
        if 0 < len(full) < len(cells):
            out.append((f"{name}, {len(full)} skills ({', '.join(dict.fromkeys(c['short'] for c in full))})", full, True, False))
        out.append((f"{name}, {len(cells)} skills, task-weighted", cells, True, True))
        out += [(f"{name}, {b}", [c for c in cells if c["short"] == b], True, False) for b in benches]
        if 0 < len(pos := [c for c in cells if c["A"]["total"] > 0]) < len(cells):
            out.append((f"{name}, the {len(pos)} skills with total > 0 (no bootstrap)", pos, False, False))
    every = [c for cells in by_model.values() for c in cells]
    return out + [(" + ".join(S.MODEL_SHORT[m] for m in by_model) + f", {len(every)} skills", every, True, False)]


def luna_slopes():
    """{(bench, episodes pooled): {quantity: fe_reg}} of the Luna test episodes: run 0 and all runs of the seed and the learned
    skills of a benchmark, and all runs of the seed alone."""
    out = {}
    for bench in SHORT:
        arms = ["native-seed"] + [m for _, b, m in S.learned([S.LUNA]) if b == bench]
        recs, n = [], len(side(S.LUNA, bench, "native-seed")[1])
        for arm in arms:
            (eps, rows), files = side(S.LUNA, bench, arm), S.cell_files(S.LUNA, bench, arm)
            outs = {(k, t): output_tokens(e) for k, rep in S.episodes(files).items() for t, e in rep.items()}
            recs += [(arm, r, t, m["R"], m["X"], outs[(r, t)], m["N"]) for r, rep in eps.items() for t, (_, m) in rep.items()
                     if m and rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["X"] is not None]
        k = len(arms) - 1
        designs = [(f"run 0 of seed + {k} skills", lambda a, r: r == 0), (f"{n} runs of seed + {k} skills", lambda a, r: True)] if k else []
        designs.append((f"{n} runs of the seed" + ("" if k else " (all optimizers returned it)"), lambda a, r: a == "native-seed"))
        for label, keep in designs:
            fits = {}
            for q, j in (("input", 4), ("output", 5), ("calls", 6)):
                items = [(rec[2], rec[3], rec[j]) for rec in recs if keep(rec[0], rec[1]) and rec[j] and rec[j] > 0]
                count = {t: sum(i[0] == t for i in items) for t in {i[0] for i in items}}
                fits[q] = fe_reg([i for i in items if count[i[0]] > 1])[0]
            out[(SHORT[bench], label)] = fits
    return out


def attribution_luna():
    by_model = {m: cells_of(m) for m in (S.DEEPSEEK, S.LUNA)}
    draws = draw_counts([c for cells in by_model.values() for c in cells])
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    luna = sorted(by_model[S.LUNA], key=lambda c: order(c)[::-1])
    rows_a = [[c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f3(c["A"][k]) for k in TERMS + ["total"]]
              + [f"{math.exp(c['log_rho_full']):.3f} ({table1[S.LUNA + '|' + c['bench']][c['method']]['rho_trj']:.2f})",
                 f3(c["A"]["residual_mean"]), f"{c['A']['residual_max_abs']:.3f}", TERM_LABEL[c["A"]["largest"]]] for c in luna]
    rows_k = [[c["method"], c["short"], c["n_full"], c["mismatch"]] for c in luna]
    cell_ci = {(c["model"], c["method"], c["bench"]): shares([c], draws) for cells in by_model.values() for c in cells}
    rows_a2 = []
    for c in luna:
        b, a = cell_ci[(c["model"], c["method"], c["bench"])], c["A"]
        rows_a2.append([c["method"], c["short"], f"{f3(a['total'])} {ci(b['total'])}", f"{f3(a['more_actions'] + a['turn_splitting'])} {ci(b['turns'])}",
                        f"{f3(a['skill_text'])} {ci(b['text'])}", f"{b['p_total_le_0']:.2f}", f"{c['d_skill']:+.0f} (seed {c['seed_skill']:.0f})"])
    rows_b = []
    for label, cells, boot, weighted in share_sets(by_model):
        s, b = shares(cells, weighted=weighted), shares(cells, draws, weighted) if boot else None
        sums = b and not weighted
        rows_b.append([label, f"{s['total']:+.3f}" + (f" {ci(b['total'])}" if sums else ""), f"{s['turns']:+.3f}" + (f" {ci(b['turns'])}" if sums else ""),
                       f"{s['text']:+.3f}", f"{s['content']:+.3f}", f"{100 * s['share_turns']:.0f}%" + (f" {ci(b['share_turns'], pct=True)}" if b else ""),
                       f"{100 * s['share_text']:.0f}%" + (f" {ci(b['share_text'], pct=True)}" if b else ""), f"{100 * s['share_content']:.0f}%",
                       f"{100 * s['gross_turns']:.0f}% / {100 * s['gross_text']:.0f}%"]
                      + ([f"{b['p_turns_gt_text']:.3f}", f"{b['p_turns_gt_half']:.3f}", f"{b['p_total_le_0']:.3f}"] if b else ["-"] * 3))
    rows_ds = []
    for c in sorted(by_model[S.DEEPSEEK], key=order):
        b, a = cell_ci[(c["model"], c["method"], c["bench"])], c["A"]
        turns = a["more_actions"] + a["turn_splitting"]
        rows_ds.append([c["method"], c["short"], c["n"], f"{f3(a['total'])} {ci(b['total'])}", f"{f3(turns)} {ci(b['turns'])}",
                        f"{100 * turns / a['total']:.0f}% {ci(b['share_turns'], pct=True)}"])
    ref = json.loads((S.cache("main") / "slopes.json").read_text())["reg"]
    rows_c = []
    for (short, label), v in luna_slopes().items():
        d = ref[next(b for b, s in SHORT.items() if s == short)]
        rows_c.append([short, label] + [f"{v[q]['slope']:.3f} ({v[q]['se']:.3f})" for q in ("input", "output", "calls")] + [f"{v['input']['n']} / {v['input']['n_tasks']}",
                      f"{d['input']['slope']:.3f} ({d['input']['se']:.3f}); out {d['output']['slope']:.3f}; calls {d['calls']['slope']:.3f}; "
                      f"{d['input']['n']} / {d['input']['n_tasks']}"])
    numbers = "\n\n".join([
        "### A. Luna four-term split, avg@3 run means (log units; n = tasks used / Table 1 common tasks)",
        md_table(["Skill", "Bench", "n", "more actions", "turn-splitting", "content per turn", "skill text", "total",
                  "rho = exp(total), full set (paper Table 1)", "resid mean", "resid max |.|", "largest"], rows_a),
        md_table(["Skill", "Bench", "tasks with every run on both sides", "episodes whose graph R, N, X differ from Table 1's row"], rows_k),
        "### A2. Luna per skill: total, turn terms (more actions + turn-splitting) and skill text with 95% task-bootstrap intervals; skill tokens per request",
        md_table(["Skill", "Bench", "total", "turn terms", "skill text", "P(total <= 0)", "skill tokens / request, skill minus seed"], rows_a2),
        "### B. Shares of the summed log-token increase (95% task-bootstrap intervals)",
        f"Intervals: two-sided {100 * S.LEVEL:.0f}% percentile, {S.N_BOOT} task resamples per model and benchmark.",
        md_table(["Set", "summed total", "turn terms", "skill text", "content per turn", "turn share", "text share", "content share",
                  "gross share turns / text", "P(turns > text)", "P(turn share > 50%)", "P(total <= 0)"], rows_b),
        "DeepSeek per skill (turn share of the skill's own total):", md_table(["Skill", "Bench", "n", "total", "turn terms", "turn share"], rows_ds),
        "### C. Within-task turn slopes, GPT-6-Luna (slope (SE)); DeepSeek reference from mech_slopes / mech_slopes_input",
        md_table(["Bench", "episodes pooled", "input tokens", "output tokens", "tool calls", "episodes / tasks",
                  "DeepSeek input (SE); output; calls; episodes / tasks"], rows_c)])
    what = ("The input-token attribution extended to GPT-6-Luna: the four-term split of every skill learned under Luna (A), each "
            "term's share of the summed log-token increase under both models with task-bootstrap intervals (B), and the Luna turn "
            "slopes of input tokens, output tokens and tool calls (C).")
    record(__spec__.name, "attribution_luna", numbers, S.runs_text(cell_list(S.LUNA) + cell_list(S.DEEPSEEK)), what=what)


# ---- the split on tasks both solve or both fail (DeepSeek)

def endings(episodes):
    """Counts of how the failed episodes of a set end, and whether their last request still issued tool calls."""
    out = dict.fromkeys(("episodes", "failed", "task_failure", "timeout", "other", "last_request_has_calls"), 0)
    for e in episodes:
        cls = (e.get("backend_outcome") or {}).get("classification")
        out["episodes"] += 1
        if cls != "success":
            kind = cls if cls in ("task_failure", "timeout") else "other"
            out |= {"failed": out["failed"] + 1, kind: out[kind] + 1,
                    "last_request_has_calls": out["last_request_has_calls"] + bool(e.get("requests") and e["requests"][-1].get("tool_calls"))}
    return out


def failcost_cells():
    cells = []
    for _, bench, method in S.learned([S.DEEPSEEK]):
        c = split(S.DEEPSEEK, bench, method)
        (seed_rows, own_rows), reps = c["rows"], c["reps"]
        recorded = [S.episodes(S.cell_files(S.DEEPSEEK, bench, m)) for m in ("native-seed", method)]

        def subset(t):
            won = [rows[r][t]["success"] is True for rows in (own_rows, seed_rows) for r in reps]
            return "solve" if all(won) else "fail" if not any(won) else "mixed"
        groups = {"all": list(c["usable"])} | {k: [t for t in c["usable"] if subset(t) == k] for k, _ in SUBSETS}
        cell = dict(method=S.ARM_NAME[method], bench=bench, short=SHORT[bench], subsets={})
        for key, tasks in groups.items():
            pairs = [c["usable"][t]["pair"] for t in tasks]
            if not pairs:
                cell["subsets"][key] = dict(n=0)
                continue
            terms, dX = four_terms(pairs)
            st = {side: {k: float(np.mean([p[i][k] for p in pairs])) for k in "RNX"} | {"N_per_R": float(np.mean([p[i]["N"] / p[i]["R"] for p in pairs]))}
                  for i, side in enumerate(("seed", "skill"))}
            st |= {f"ratio_{k}": st["skill"][k] / st["seed"][k] for k in "RNX"} | {f"geo_{k}": float(np.exp(np.mean([math.log(b[k] / a[k]) for a, b in pairs]))) for k in "RNX"}
            cell["subsets"][key] = dict(n=len(tasks), terms=terms, total=dX, stats=st, **{
                f"endings_{side}": endings([eps[r][t] for r in reps for t in tasks if t in eps[r]]) for side, eps in zip(("seed", "skill"), recorded)})
        cells.append(cell)
    return cells


def contrast(cells, short=None):
    """Mean over cells of the (b) minus (a) task mean of each term and the total, with its percentile interval."""
    chosen = [c for c in cells if (short is None or c["short"] == short) and c["subsets"]["solve"]["n"] and c["subsets"]["fail"]["n"]]
    point, lim = {}, {}
    for name in TERMS + ["total"]:
        diffs, sims = [], []
        for c in chosen:
            xa, xb = (c["subsets"][k]["total"] if name == "total" else c["subsets"][k]["terms"][name] for k in ("solve", "fail"))
            ia, ib = (S.boot_index(len(x), "failcost", c["method"], c["bench"], k) for x, k in ((xa, "solve"), (xb, "fail")))
            diffs.append(xb.mean() - xa.mean())
            sims.append(xb[ib].mean(1) - xa[ia].mean(1))
        point[name], lim[name] = float(np.mean(diffs)), S.interval(np.mean(sims, axis=0))
    return len(chosen), point, lim


def failcost():
    cells = failcost_cells()
    rows_t, rows_s, rows_e = [], [], []
    for c in cells:
        for key, label in [("all", "all usable")] + SUBSETS:
            s = c["subsets"][key]
            head = [c["method"], c["short"], label, s["n"]]
            if not s["n"]:
                rows_t.append(head + ["-"] * 6)
                continue
            rows_t.append(head + [f3(s["terms"][k].mean()) for k in TERMS] + [f3(s["total"].mean()), f"{math.exp(s['total'].mean()):.2f}"])
            a, b = s["stats"]["seed"], s["stats"]["skill"]
            rows_s.append(head + [f"{a['R']:.1f} / {b['R']:.1f}", f"{a['N']:.1f} / {b['N']:.1f}", f"{a['X'] / 1000:.0f}k / {b['X'] / 1000:.0f}k",
                                  f"{a['N_per_R']:.2f} / {b['N_per_R']:.2f}"] + [" / ".join(f"{s['stats'][f'{p}_{k}']:.2f}" for k in "RNX") for p in ("ratio", "geo")])
        for key, label in (("fail", SUBSETS[1][1]), ("mixed", SUBSETS[2][1]), ("solve", SUBSETS[0][1])):
            for side in ("skill", "seed") if c["subsets"][key]["n"] else ():
                if (e := c["subsets"][key][f"endings_{side}"])["failed"]:
                    rows_e.append([c["method"], c["short"], label, side] + list(e.values()))
    rows_p, rows_c = [], []
    for short in [None] + list(SHORT.values()):
        for side in ("skill", "seed"):
            got = [c["subsets"]["fail"][f"endings_{side}"] for c in cells if c["subsets"]["fail"]["n"] and short in (None, c["short"])]
            rows_p.append([short or "all benchmarks", side] + [sum(e[k] for e in got) for k in ("failed", "task_failure", "timeout", "other", "last_request_has_calls")])
        n, point, lim = contrast(cells, short)
        rows_c.append([short or f"pooled {n} cells", n] + [f"{point[k]:+.2f} [{lim[k][0]:+.2f}, {lim[k][1]:+.2f}]" for k in TERMS + ["total"]])
    both = [c["subsets"] for c in cells if c["subsets"]["solve"]["n"] and c["subsets"]["fail"]["n"]]
    more = {k: sum((s["fail"]["total"] if x == "total" else s["fail"]["terms"][x]).mean() > (s["solve"]["total"] if x == "total" else s["solve"]["terms"][x]).mean()
                   for s in both) for k, x in (("total", "total"), ("actions", "more_actions"))}
    numbers = "\n\n".join([
        "### 1. Four terms by subset (log units; ratio = exp(total))",
        md_table(["Skill", "Bench", "Subset", "n", "actions", "turn-split", "content", "skill text", "total", "exp(total)"], rows_t),
        "### 2. Per-episode behaviour by subset (seed / skill; mean over tasks of 3-run means)",
        md_table(["Skill", "Bench", "Subset", "n", "turns R", "calls N", "input tokens X", "calls per turn", "ratio of means R / N / X",
                  "geo-mean ratio R / N / X"], rows_s),
        "### 3. How failed episodes end (all runs of the subset's tasks)",
        md_table(["Skill", "Bench", "Subset", "Side", "episodes", "failed", "task_failure", "timeout", "other", "last request had calls"], rows_e),
        f"Both-fail subset (b) pooled over the {len(cells)} cells:",
        md_table(["Scope", "Side", "failed episodes", "task_failure", "timeout", "other", "last request had calls"], rows_p),
        "### 4. Contrast (b) both fail minus (a) both solve, mean over cells, 95% bootstrap interval",
        f"Intervals: two-sided {100 * S.LEVEL:.0f}% percentile, {S.N_BOOT} task resamples per cell and subset.",
        md_table(["Scope", "cells", "actions", "turn-split", "content", "skill text", "total"], rows_c),
        f"Cells where the failing-subset total exceeds the solved-subset total: {more['total']} of {len(both)}; where the failing-subset "
        f"actions term exceeds the solved one: {more['actions']} of {len(both)}."])
    what = ("Is the input-token tax larger on tasks that fail, and is it flailing? Each DeepSeek skill's usable tasks split into those "
            "both sides solve in all three runs, both fail in all three, and the rest; per subset the four-term split on 3-run means, "
            "the turns, calls and input tokens per episode, and how the failed episodes end.")
    record(__spec__.name, "failcost", numbers, S.runs_text(cell_list(S.DEEPSEEK)), what=what)


if __name__ == "__main__":
    attribution3()
    attribution_luna()
    failcost()
