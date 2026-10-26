"""What the extra input tokens cost: the learned skills' price under cached-input pricing, their token tax under a harness
that compacts its history, and the cost of the optimization itself, all from recorded data.

mech_price3 (DeepSeek): the price of a task is its uncached input + c x cached input + o x output tokens, from the means
over the test runs Table 1 averages; cached input is what the provider served from cache from request 2 on. The
ratio to the seed is the per-task geometric mean on Table 1's common tasks (price, geo. mean) or the ratio of summed
prices, at the main weights and over a grid of weights. Shares are of the skill side.

mech_compaction (DeepSeek): every recorded test episode is re-billed as if the harness had sent a shorter history; the
agent's turns are held fixed, so this is the accounting effect of compaction only. A request's prefix F (system prompt,
tool schemas, task, skill text) is always paid; its history H = X - F is cut to what the last k turns added (a window) or
to at most C tokens (a cap). Tasks: Table 1's common tasks with a graph episode in all six runs. rho is the geometric mean
over tasks of the 3-run means, skill against seed, the same policy on both sides; rho_turns the same on turns; the slope is
fitted within tasks on the six episodes of each task (src.analysis.episodes.fe_reg). The window-1 intervals resample tasks.

mech_optcost: the optimizer's own model calls of every pinned optimizer run of both models (llm/call-*.json): tokens, the
calls whose prompt the client cut to its context limit (the request holds the client's cut marker; summary.json's count
is printed beside it), the longest prompts and the replies that stopped at the output cap. Then, for DeepSeek, whose
test episodes record their input tokens in full, the agent rollouts of the optimization (the train and dev evaluations; an
episode served from the shared evaluation cache counts as nominal, not executed), their total in units of the seed's median
test episode, the tax per test episode (the mean over Table 1's common tasks of skill minus seed input tokens) and the
number of episodes after which the tax exceeds the optimization's input tokens."""
import itertools, json, math
from statistics import median

import numpy as np

from src.analysis import settings as S
from src.analysis.attribution import OPT, split
from src.analysis.episodes import fe_reg
from src.graph import episode as W
from src.utils.dataloader import BENCHES, gmean, task_row
from src.utils.fig import BENCH_LABEL
from src.utils.logger import md_table, record

MODEL = S.DEEPSEEK
MAIN = (0.1, 4.0)
GRID = list(itertools.product((0.02, 0.1, 0.5), (2.0, 4.0, 8.0)))
WINDOWS = (1, 3, 5, 10, None)
CAPS = (8000, 32000, 64000, 128000, None)
CUT = "[... earlier content truncated to fit the context ...]"
SPLITS = ("train", "dev")
SHORT = S.BENCH_SHORT


def skills():
    """(name, method, bench) of the DeepSeek learned skills, optimizer by optimizer as src.analysis.attribution orders them."""
    learned = set(S.learned([MODEL]))
    return [(d, m, b) for d, m in OPT for b in BENCHES if (MODEL, b, m) in learned]


def runs(*rows):
    """The Runs section of a DeepSeek record: the learned skills and the given rows of every benchmark."""
    return S.runs_text([(MODEL, b, m) for _, m, b in skills()] + [(MODEL, b, r) for r in rows for b in BENCHES])


def requests(episode):
    """The distinct requests of a recorded episode, in order."""
    return list({r["request_id"]: r for r in episode.get("requests", [])}.values())


def side_means(method, bench):
    """{task: mean N, X, cached input and output tokens} over the test runs with a verified outcome and both counts."""
    files = S.cell_files(MODEL, bench, method)
    per = {}
    for rows, eps in zip(S.rows(files, MODEL).values(), S.episodes(files).values()):
        for t, r in rows.items():
            if None not in (r["success"], r["N"], r["X"]):
                reqs = requests(eps[t])
                cached = sum(min((q.get("captured_usage") or {}).get("cache_tokens") or 0, q["input_tokens"]) for q in reqs[1:])
                per.setdefault(t, []).append((r["N"], r["X"], cached, sum(q.get("output_tokens") or 0 for q in reqs)))
    return {t: dict(zip(("N", "X", "cached", "out"), np.mean(v, axis=0))) for t, v in per.items()}


def common(ref, own):
    return [t for t in sorted(ref) if t in own and min(ref[t]["N"], ref[t]["X"], own[t]["N"], own[t]["X"]) > 0]


def price3():
    head = ["Benchmark", "Method", "n", "cached share of input", "output share of tokens", "price, geo. mean", f"[min, max] over {len(GRID)}",
            "price, ratio of sums", f"[min, max] over {len(GRID)}"]
    rows = []
    for bench in BENCHES:
        ref = side_means("native-seed", bench)
        for method in ["native-noseed"] + [m for m in S.OPTIMIZERS if (MODEL, bench, m) in S.learned([MODEL])]:
            own = side_means(method, bench)
            tasks = common(ref, own)

            def ratios(c, o):
                p_own, p_ref = (np.array([s[t]["X"] - (1 - c) * s[t]["cached"] + o * s[t]["out"] for t in tasks]) for s in (own, ref))
                return gmean(p_own / p_ref), float(p_own.sum() / p_ref.sum())
            (geo, tot), grid = ratios(*MAIN), [ratios(c, o) for c, o in GRID]
            x, out = (sum(own[t][k] for t in tasks) for k in ("X", "out"))
            rows.append([BENCH_LABEL[bench], S.ARM_NAME[method], len(tasks), f"{100 * sum(own[t]['cached'] for t in tasks) / x:.1f}%",
                         f"{100 * out / (out + x):.2f}%", f"x{geo:.2f}", f"[{min(g for g, _ in grid):.2f}, {max(g for g, _ in grid):.2f}]",
                         f"x{tot:.2f}", f"[{min(s for _, s in grid):.2f}, {max(s for _, s in grid):.2f}]"])
    weights = (f"Main weights c = {MAIN[0]}, o = {MAIN[1]:g}; the grid: c in {{{', '.join(f'{c:g}' for c in dict.fromkeys(c for c, _ in GRID))}}} "
               f"x o in {{{', '.join(f'{o:g}' for o in dict.fromkeys(o for _, o in GRID))}}}.")
    record(__spec__.name, "price3", weights + "\n\n" + md_table(head, rows), runs("native-seed", "native-noseed"),
           what="Price of each learned skill (and of No skill) relative to the seed when cached input is cheap (DeepSeek).")


# ---- compaction

def billed(e, window=None, cap=None):
    """Input tokens of an episode when each request pays its prefix and only the history the last `window` turns added, or
    at most `cap` tokens of history (None: the recorded history)."""
    fixed = [i for i, c in enumerate(e.comps) if c in W.FIXED]
    F = e.Xrg[:, fixed].sum(axis=1) if fixed else np.zeros(e.R)
    H = np.maximum(e.req_in - F, 0.0)
    if window is not None:
        H = np.maximum(H - np.concatenate([np.zeros(min(window, len(H))), H[:-window]]), 0.0)
    return float((F + (H if cap is None else np.minimum(H, cap))).sum())


def compaction_cell(name, method, bench):
    c = split(MODEL, bench, method)
    reps, own, seed = c["reps"], c["own"], c["seed"]
    tasks = [t for t in c["common"] if all(t in eps[r] for eps in (own, seed) for r in reps)]

    def logs(fn):
        """Per task: log of the skill's 3-run mean over the seed's, and the six (task, turns, value) points."""
        out, pts = [], []
        for t in tasks:
            a, b = ([fn(eps[r][t][0]) for r in reps] for eps in (seed, own))
            out.append(math.log(np.mean(b) / np.mean(a)))
            pts += [(t, eps[r][t][0].R, v) for eps, vs in ((seed, a), (own, b)) for r, v in zip(reps, vs)]
        return np.array(out), pts
    turns = logs(lambda e: e.R)[0]
    results, w1 = [], {}
    for policy, params in (("window", WINDOWS), ("cap", CAPS)):
        for p in params:
            lr, pts = logs(lambda e: billed(e, **{policy: p}))
            elasticity = float(lr.mean() / turns.mean()) if abs(turns.mean()) > 1e-9 else None
            results.append(dict(policy=policy, param="inf" if p is None else str(p), rho=math.exp(lr.mean()), slope=fe_reg(pts)[0]["slope"],
                                elasticity=elasticity))
            if policy == "window" and p in (1, None):
                w1[p] = lr
    idx = S.boot_index(len(tasks), "compaction window 1", method, bench)
    lim = dict(rho_inf=S.interval(np.exp(w1[None][idx].mean(1))), rho_w1=S.interval(np.exp(w1[1][idx].mean(1))),
               gap=S.interval(w1[1][idx].mean(1) - turns[idx].mean(1)))
    return dict(method=name, bench=bench, short=SHORT[bench], n=len(tasks), n_table1=len(c["common"]), rho_turns=math.exp(turns.mean()),
                gap=float(w1[1].mean() - turns.mean()), ci=lim, results=results)


def compaction():
    cells = [compaction_cell(*s) for s in skills()]
    (S.cache("tokens") / "compaction.json").write_text(json.dumps(dict(
        model=MODEL, windows=["inf" if k is None else str(k) for k in WINDOWS], caps=["inf" if c is None else str(c) for c in CAPS],
        cells=[{k: v for k, v in c.items() if k not in ("ci", "gap")} for c in cells]), indent=1))
    pick = lambda c, policy, p: next(r for r in c["results"] if r["policy"] == policy and r["param"] == p)
    sections = []
    for policy, params, title in (("window", WINDOWS, "(i) sliding window: history of the last k turns"), ("cap", CAPS, "(ii) token cap C on history per request")):
        params = ["inf" if p is None else str(p) for p in params]
        head = ["Skill", "Bench", "n"]
        cols = [f"{policy} {p}" for p in params]
        tab = lambda key, nd, extra=False: md_table(head + (["rho_turns"] if extra else []) + cols, [
            [c["method"], c["short"], c["n"]] + ([f"{c['rho_turns']:.3f}"] if extra else []) + [f"{pick(c, policy, p)[key]:.{nd}f}" for p in params] for c in cells])
        sections += [f"### {title}", "rho (geometric mean over tasks of per-task 3-run means, skill vs seed)", tab("rho", 3, True),
                     "within-task slope of log input on log turns", tab("slope", 3),
                     "log rho / log rho_turns (1 = the whole token tax is the turn ratio)", tab("elasticity", 2)]
    trend = [[policy, p, f"{np.mean([pick(c, policy, p)['slope'] for c in cells]):.3f}",
              f"{np.mean([abs(math.log(pick(c, policy, p)['rho'] / c['rho_turns'])) for c in cells]):.3f}"]
             for policy, params in (("window", WINDOWS), ("cap", CAPS)) for p in ("inf" if p is None else str(p) for p in params)]
    sections += [f"### Trend over the {len(cells)} skills", md_table(["policy", "parameter", "mean slope", "mean |log(rho / rho_turns)|"], trend)]
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    sections += ["### The recorded run (window inf) against Table 1", md_table(
        ["Skill", "Bench", "tasks used", "Table 1 tasks", "rho, window inf (graph-episode input)", "Table 1 rho_trj"],
        [[c["method"], c["short"], c["n"], c["n_table1"], f"{pick(c, 'window', 'inf')['rho']:.3f}",
          f"{table1[MODEL + '|' + c['bench']][c['method']]['rho_trj']:.3f}"] for c in cells])]
    ci = lambda x, nd=3, sign="": f"[{x[0]:{sign}.{nd}f}, {x[1]:{sign}.{nd}f}]"
    rows = [[c["method"], c["short"], c["n"], f"{pick(c, 'window', 'inf')['rho']:.3f} {ci(c['ci']['rho_inf'])}",
             f"{pick(c, 'window', '1')['rho']:.3f} {ci(c['ci']['rho_w1'])}", f"{c['rho_turns']:.3f}", f"{c['gap']:+.3f} {ci(c['ci']['gap'], sign='+')}"] for c in cells]
    lo, hi = (min(cells, key=lambda c: f(pick(c, "window", "1")["rho"])) for f in (lambda x: x, lambda x: -x))
    w1 = lambda c: f"{pick(c, 'window', '1')['rho']:.2f} {ci(c['ci']['rho_w1'], 2)} ({c['method']} {c['short']})"
    sections += [f"### Compaction to the last turn of history (window 1), {100 * S.LEVEL:.0f}% intervals over {S.N_BOOT} task resamples",
                 md_table(["skill", "bench", "n", "rho as recorded", "rho, last turn of history kept (window 1)", "rho^turns",
                           "log(rho window 1 / rho^turns)"], rows),
                 f"Range of the window-1 tax: {w1(lo)} to {w1(hi)}; lower bound above 1 for {sum(c['ci']['rho_w1'][0] > 1 for c in cells)} "
                 f"of {len(cells)} skills. The gap to rho^turns excludes 0 for {sum(c['ci']['gap'][0] > 0 for c in cells)} of {len(cells)}."]
    record(__spec__.name, "compaction", "\n\n".join(sections), runs("native-seed"),
           what="A counterfactual: how large would the token tax be if the harness compacted its history? (DeepSeek)")


# ---- the cost of the optimization

def calls(run):
    """The optimizer's model calls of a run: counts, tokens, cut prompts, the longest prompt and replies at the output cap."""
    got = dict(calls=0, prompt=0, completion=0, cut=0, cut_tokens=[], max_tokens=0, max_chars=0, at_cap=0)
    for f in sorted((run / "llm").glob("call-*.json")):
        d = json.loads(f.read_text())
        usage, text = d["response"].get("usage", {}), "".join(m["content"] if isinstance(m["content"], str) else
                                                              "".join(p.get("text", "") for p in m["content"]) for m in d["request"]["messages"])
        tokens = usage.get("prompt_tokens", 0)
        got |= dict(calls=got["calls"] + 1, prompt=got["prompt"] + tokens, completion=got["completion"] + usage.get("completion_tokens", 0),
                    max_tokens=max(got["max_tokens"], tokens), max_chars=max(got["max_chars"], len(text)),
                    at_cap=got["at_cap"] + (d["response"]["choices"][0].get("finish_reason") == "length"))
        if CUT in text:
            got["cut"] += 1
            got["cut_tokens"].append(tokens)
    attempts = run / "llm/attempts.jsonl"
    got["errors"] = (sum(r.get("event") == "result" and r.get("outcome") == "error" for r in map(json.loads, attempts.read_text().splitlines()))
                     if attempts.is_file() else None)
    got["summary"] = json.loads((run / "summary.json").read_text()).get("llm") or {}
    return got


def rollouts(run):
    """The agent episodes of the optimization (train and dev evaluations), nominal and executed, per split."""
    out = {}
    for split_ in SPLITS:
        cell = dict.fromkeys(("episodes", "x_in", "x_out", "unknown_x", "exec_episodes", "exec_x_in"), 0)
        for f in sorted((run / "results/evaluations" / split_).glob("*/evaluation.json")):
            for e in json.loads(f.read_text())["episodes"]:
                reqs, x = requests(e), task_row(e)["X"]
                if x is None:
                    cell["unknown_x"] += 1
                    x = sum(r["input_tokens"] for r in reqs if isinstance(r["input_tokens"], (int, float)))
                executed = e.get("collection_mode") != "evaluation_cache_hit"
                cell |= dict(episodes=cell["episodes"] + 1, x_in=cell["x_in"] + x, x_out=cell["x_out"] + sum(r.get("output_tokens") or 0 for r in reqs),
                             exec_episodes=cell["exec_episodes"] + executed, exec_x_in=cell["exec_x_in"] + executed * x)
        out[split_] = cell
    return out


def optcost():
    M = lambda x: f"{x / 1e6:.1f}M"
    k = lambda x: f"{x / 1e3:.0f}k"
    runs = {(m, b, o): calls(S.run(m, b, o)) for m in (S.DEEPSEEK, S.LUNA) for o in S.OPTIMIZERS for b in BENCHES}
    head_a = ["run", "optimizer calls", "of which cut at 512k", "failed attempts", "opt. in tok", "opt. out tok", "summary.json llm.calls",
              "cut (request holds the cut marker)", "longest prompt, tokens", "longest prompt, characters", "replies at the output cap"]
    sections = []
    for model in (S.DEEPSEEK, S.LUNA):
        rows, order = [], [(o, b) for b in BENCHES for o in S.OPTIMIZERS]
        for o, b in order:
            c = runs[(model, b, o)]
            rows.append([f"{S.ARM_NAME[o]} {SHORT[b]}", c["calls"], c["summary"].get("truncated_calls", 0), "n/a" if c["errors"] is None else c["errors"],
                         M(c["prompt"]), M(c["completion"]), c["summary"].get("calls", "empty ({})"), c["cut"], k(c["max_tokens"]), k(c["max_chars"]), c["at_cap"]])
        cut = {o: sum(runs[(model, b, o)]["summary"].get("truncated_calls", 0) for b in BENCHES) for o in S.OPTIMIZERS}
        rows.append([f"all {S.MODEL_SHORT[model]} runs", sum(runs[(model, b, o)]["calls"] for o, b in order),
                     ", ".join(f"{S.ARM_NAME[o]} {n}" for o, n in cut.items()), "", "", "", ""] + [""] * 4)
        sections += [f"**(a) Optimizer LLM calls, {S.MODEL_NAME[model]}** (llm/call-*.json; usage = prompt / completion tokens)", md_table(head_a, rows)]
    summary = []
    for model in (S.DEEPSEEK, S.LUNA):
        for o in S.OPTIMIZERS:
            cs = [runs[(model, b, o)] for b in BENCHES]
            cut_tokens = [t for c in cs for t in c["cut_tokens"]]
            summary.append([S.MODEL_SHORT[model], S.ARM_NAME[o], len(cs), sum(c["calls"] for c in cs), sum(c["cut"] for c in cs), max(c["cut"] for c in cs),
                            f"{k(min(cut_tokens))} to {k(max(cut_tokens))}" if cut_tokens else "-", k(max(c["max_tokens"] for c in cs)),
                            k(max(c["max_chars"] for c in cs)), sum(c["at_cap"] for c in cs)])
    sections += ["**Prompt lengths per model and optimizer** (over its pinned runs)", md_table(
        ["model", "optimizer", "runs", "calls", "cut calls", "most cut calls in one run", "cut prompts, tokens", "longest prompt, tokens",
         "longest prompt, characters", "replies at the output cap"], summary)]
    rows_b, rows_c = [], []
    for bench in BENCHES:
        seed = S.rows(S.cell_files(MODEL, bench, "native-seed"), MODEL)
        median_x = float(median(r["X"] for rep in seed.values() for r in rep.values() if None not in (r["success"], r["N"], r["X"])))
        ref = side_means("native-seed", bench)
        for o in (o for o in S.OPTIMIZERS if (MODEL, bench, o) in S.learned([MODEL])):
            run, c = S.run(MODEL, bench, o), runs[(MODEL, bench, o)]
            ro, own = rollouts(run), side_means(o, bench)
            tasks = common(ref, own)
            tax = float(np.mean([own[t]["X"] - ref[t]["X"] for t in tasks]))
            nominal, executed = (sum(ro[s][key] for s in SPLITS) + c["prompt"] for key in ("x_in", "exec_x_in"))
            name = f"{S.ARM_NAME[o]} {SHORT[bench]}"
            rows_b.append([name, f"{ro['train']['episodes']} / {ro['train']['exec_episodes']}", f"{ro['dev']['episodes']} / {ro['dev']['exec_episodes']}",
                           M(sum(ro[s]["x_in"] for s in SPLITS)), M(sum(ro[s]["exec_x_in"] for s in SPLITS)), M(sum(ro[s]["x_out"] for s in SPLITS)),
                           sum(ro[s]["unknown_x"] for s in SPLITS)])
            even = lambda total: f"{total / tax:,.0f}" if tax > 0 else "never (tax <= 0)"
            rows_c.append([name, f"{median_x:,.0f}", M(nominal), f"{nominal / median_x:.0f}", M(executed), f"{executed / median_x:.0f}", f"{tax:+,.0f}",
                           f"{100 * tax / np.mean([ref[t]['X'] for t in tasks]):+.0f}%", even(nominal), even(executed)])
    sections += [f"**(b) Agent rollouts of the optimization, {S.MODEL_NAME[MODEL]}** (results/evaluations/train+dev; test split excluded)", md_table(
        ["run", "train eps (nominal / executed)", "dev eps (nominal / executed)", "rollout in tok nominal", "rollout in tok executed",
         "rollout out tok nominal", "eps with incomplete capture"], rows_b),
        "**(c) Cost against the seed test episode and the per-episode tax**", md_table(
        ["run", "seed test ep. median X", "total in tok (rollouts + optimizer calls), nominal", "= seed test eps", "executed only", "= seed test eps",
         "tax / episode (tok)", "tax / seed X", "break-even episodes (nominal)", "break-even (executed)"], rows_c)]
    cells = [(m, b, o) for m in (S.DEEPSEEK, S.LUNA) for b in BENCHES for o in S.OPTIMIZERS] + [(MODEL, b, "native-seed") for b in BENCHES]
    record(__spec__.name, "optcost", "\n\n".join(sections), S.runs_text(cells),
           what=("The optimizer-side cost of each optimizer run: its own model calls (both models), and for DeepSeek the training and dev "
                 "rollouts in units of one seed test episode, and the number of deployment episodes after which the skill's input-token "
                 "tax exceeds that cost."))


if __name__ == "__main__":
    price3()
    compaction()
    optcost()
