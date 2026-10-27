"""What the learned skills change in the agent's tool calls, turn by turn, read off the recorded test episodes: how many
calls a turn carries and which of them only read (mech_callsperturn), whether the serialize rules put AutoBench writes
one per turn while reads stay grouped (mech_calls_rw), and how much of each skill's token tax would go if it kept the
seed's calls per turn (mech_batching). Accounting on recorded episodes; nothing is re-run.

Calls per turn (DeepSeek, three benchmarks). Each cell uses the task set of the four-term split (aux.tokens.split):
Table 1's common tasks with a verified outcome, N and X in all six runs and a usable graph episode with positive history.
A side pools its three runs; the seed row is the seed on that cell's tasks. A turn is a model request and its calls the
calls the harness executed (checked against the graph episode); m = calls / turns, m_task the mean of the episodes' own m.
Read-only follows aux.calls.classify.read_only; the lenient variant also accepts echo, printf, sort, uniq, cut, tr, pwd,
which, stat, file, tree, du and diff inside an otherwise pure-read command. A run is a maximal stretch of two or more
consecutive turns that each hold one read-only call; merging it into one turn saves its length minus one.

Writes and reads (AutoBench, both models). Each executed call is a write, a read or other (aux.calls.classify.classify)
in Table 1's AutoBench rows. Each row is compared with the Seed row on Table 1's common task set, using only complete
episodes (outcome, calls and input tokens captured); for DeepSeek n, rho^calls and rho are checked against Table 1.
Writes in a multi-write turn are the writes that share their turn with another write, as a share of all writes (likewise
reads), so they do not move when a skill simply writes less. rho^calls and rho^turns are geometric means over tasks of
three-run means. Intervals: paired bootstrap over tasks (both sides resampled together), two-sided percentile. Packing
gives the skill's writes the seed's writes per write-carrying turn. Sensitivities: local file writes (write, edit,
apply_patch) counted as writes; every recorded episode with its completed requests, on all tasks.

Batching (DeepSeek). Every graph episode of the seed and the three skills with R >= 2, a call and positive history is
fitted per benchmark and pooled: log X = a_task + c_skill + b_N log N + b_R log R, by least squares within tasks with
skill dummies and standard errors clustered by task; the fit on log N and log N/R is the same model (g_m = -b_R). If a
skill kept its calls at the seed's calls per turn on every task, log X would fall by b_R (dlog m), averaged over the
cell's split tasks; this is set beside the split's turn-splitting term. The AutoBench interval resamples tasks, re-fits
b_R on each draw (a task drawn k times enters as k clusters) and recomputes every skill's numbers on that draw."""
import collections, functools, itertools, json, math

import numpy as np

from aux.calls import classify as C
from aux.tokens import split as A
from src.analysis import settings as S
from src.utils.dataloader import BENCHES, MODELS, task_row
from src.utils.logger import md_table, record

AUTOB = "automationbench"
OPT = ("skillopt", "skilladam", "gepa")  # the order of the four-term split
ROWS = ("native-noseed", "gepa", "skillopt", "skilladam")  # each against the Seed row
TASK = "finance-vendor-payment-approval"  # the teaser's task, drawn as one episode in the writes figure
SIDES = {"seed": "native-seed", "skill": None}
# the statistics of mech_calls_rw printed with an interval
INTERVALS = ("d_W2", "d_R2", "d_Wpar", "d_Rpar", "did_par", "d_m", "log_rho_calls", "log_rho_turns", "log_W_ratio", "log_R_ratio",
             "pack_saved_turns_ep", "log_turns_pooled", "pack_log_turns_removed")


def pct(x):
    return f"{100 * x:.1f}"


def ci(point, bounds, scale=100, nd=1, sign="+"):
    return f"{scale * point:{sign}.{nd}f} [{scale * bounds[0]:{sign}.{nd}f}, {scale * bounds[1]:{sign}.{nd}f}]"


# ---- DeepSeek cells: the four-term split's task set, calls per turn, batching

def split_pairs(bench, method):
    """{task: (seed metrics, skill metrics)} on the four-term split's task set of one DeepSeek cell."""
    return {t: u["pair"] for t, u in A.split(S.DEEPSEEK, bench, method)["usable"].items()}


@functools.lru_cache(maxsize=None)
def turn_lists(bench, method):
    """{rep: {task: [[(tool, arguments) of each executed call] per turn]}} of one DeepSeek row."""
    files = S.cell_files(S.DEEPSEEK, bench, method)
    return {rep: {t: C.turn_calls(e) for t, e in S.episodes({rep: fs})[rep].items() if e.get("requests")} for rep, fs in files.items()}


def side_stats(bench, method, tasks, extra):
    """Calls per turn, single read-only turns, their runs and the bash calls of one side, its runs pooled."""
    k, m_task, runs, lengths, heads = collections.Counter(), collections.defaultdict(list), [], [], collections.Counter()
    for rep, eps in sorted(turn_lists(bench, method).items()):
        for t in tasks:
            turns, e = eps[t], A.side(S.DEEPSEEK, bench, method)[0][rep][t][0]
            assert (len(turns), sum(map(len, turns))) == (e.R, e.N), (bench, method, rep, t)
            flags = [len(c) == 1 and C.read_only(*c[0], extra) for c in turns]
            k.update(f"calls{min(len(c), 2)}" for c in turns)
            k.update(episodes=1, turns=len(turns), calls=sum(map(len, turns)), single=sum(len(c) == 1 for c in turns), single_ro=sum(flags))
            m_task[t].append(sum(map(len, turns)) / len(turns))
            got = [n for flag, g in itertools.groupby(flags) if flag and (n := len(list(g))) >= 2]
            runs.append((len(got), sum(got), sum(got) - len(got)))
            lengths += got
            for tool, arguments in itertools.chain(*turns):
                if tool == "bash":
                    command = C.bash_command(arguments)
                    pure, words = C.bash_pure_read(command, extra), command.split()
                    k.update(bash=1, bash_pure=int(pure))
                    if not pure:
                        heads[" ".join(words[:2]) if words and words[0] in ("python", "python3", "uv") else (words[0] if words else "")] += 1
    turns, runs = k["turns"], np.array(runs, float)
    return dict(episodes=k["episodes"], turns=turns, calls=k["calls"], m=k["calls"] / turns,
                m_task_mean=float(np.mean([np.mean(v) for v in m_task.values()])),
                share0=k["calls0"] / turns, share1=k["calls1"] / turns, share2p=k["calls2"] / turns,
                single_readonly_of_turns=k["single_ro"] / turns, single_share=k["single"] / turns,
                single_readonly_of_single=k["single_ro"] / k["single"] if k["single"] else math.nan,
                runs_per_episode=runs[:, 0].mean(), run_turns_per_episode=runs[:, 1].mean(), run_turns_share=runs[:, 1].sum() / turns,
                saved_per_episode=runs[:, 2].mean(), mean_run_length=float(np.mean(lengths)) if lengths else math.nan,
                bash_calls=k["bash"], bash_pure_read_share=k["bash_pure"] / k["bash"] if k["bash"] else math.nan,
                non_pure_bash_top=heads.most_common(6))


def calls_per_turn(pairs):
    cells = []
    for method in OPT:
        for bench in BENCHES:
            tasks = list(pairs[bench, method])
            cell = dict(method=S.ARM_NAME[method], bench=bench, short=S.BENCH_SHORT[bench], n=len(tasks))
            for suffix, extra in (("", frozenset()), ("_lenient", C.LENIENT_EXTRA)):
                cell |= {side + suffix: side_stats(bench, arm or method, tasks, extra) for side, arm in SIDES.items()}
            cells.append(cell)
    return cells


def within(records, columns, skill=True):
    """log X and the columns plus skill dummies (first level dropped), demeaned inside each task; each row's task index."""
    task = np.unique([r["task"] for r in records], return_inverse=True)[1]
    labels = [r["skill"] for r in records]
    x = np.column_stack([columns] + [[float(s == level) for s in labels] for level in sorted(set(labels))[1:] if skill])
    y = np.log([r["X"] for r in records])
    for t in range(task.max() + 1):
        x[task == t] -= x[task == t].mean(axis=0)
        y[task == t] -= y[task == t].mean()
    return x, y, task


def regressions(records):
    """The (N, R) fit with skill effects, the same on (N, N/R), and (N, R) without skill effects."""
    log_n, log_r = np.log([r["N"] for r in records]), np.log([r["R"] for r in records])
    out = {}
    for name, columns, skill in (("NR", np.column_stack([log_n, log_r]), True), ("Nm", np.column_stack([log_n, log_n - log_r]), True),
                                 ("NR_noskill", np.column_stack([log_n, log_r]), False)):
        x, y, task = within(records, columns, skill)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        score = np.array([x[task == t].T @ (y - x @ beta)[task == t] for t in range(task.max() + 1)])
        bread, n, g = np.linalg.inv(x.T @ x), len(y), task.max() + 1
        se = np.sqrt(np.diag(bread @ score.T @ score @ bread) * g / (g - 1) * (n - 1) / (n - x.shape[1]))
        out[name] = dict(coef=beta[:2].tolist(), se=se[:2].tolist(), n=n, tasks=int(g))
    return out


def resampled_b_r(records, counts):
    """b_R re-fitted on each row of counts (draws x tasks in sorted order; a task drawn k times enters as k clusters): the
    within-task cross products of each task, summed with the draw's counts."""
    x, y, task = within(records, np.log([[r["N"], r["R"]] for r in records]))
    xx = np.array([x[task == t].T @ x[task == t] for t in range(counts.shape[1])])
    xy = np.array([x[task == t].T @ y[task == t] for t in range(counts.shape[1])])
    return np.linalg.solve(np.einsum("bt,tij->bij", counts, xx), np.einsum("bt,ti->bi", counts, xy)[..., None])[:, 1, 0]


def batching(pairs):
    """Fits per benchmark and pooled, each cell's counterfactual, and the AutoBench bootstrap."""
    fits, cells, pooled, auto = {}, [], [], []
    for bench in BENCHES:
        records = [dict(task=f"{bench}:{t}", skill=label, N=x["N"], R=x["R"], X=x["X"])
                   for method, label in [("native-seed", "seed")] + [(m, S.ARM_NAME[m]) for m in OPT]
                   for eps in A.side(S.DEEPSEEK, bench, method)[0].values() for t, (_, x) in eps.items() if x]
        fits[bench], pooled = regressions(records), pooled + records
        auto = records if bench == AUTOB else auto
        for method in OPT:
            p = pairs[bench, method]
            terms, d_x = A.four_terms(list(p.values()))
            d_m = np.array([math.log((b["N"] / b["R"]) / (a["N"] / a["R"])) for a, b in p.values()])
            cells.append(dict(method=S.ARM_NAME[method], bench=bench, n=len(p), log_rho=float(d_x.mean()),
                              turn_splitting_term=float(terms["turn_splitting"].mean()), d_log_m=d_m.tolist(), dlog_x=d_x.tolist(),
                              e_r_dlog_m=(-terms["turn_splitting"]).tolist(), tasks=list(p)))
    fits["pooled"] = regressions(pooled)
    for cell in cells:
        for source, suffix in ((cell["bench"], ""), ("pooled", "_pooled_slope")):
            cell["removed" + suffix] = float(np.mean(-fits[source]["NR"]["coef"][1] * np.array(cell["d_log_m"])))
            cell["log_rho_cf" + suffix] = cell["log_rho"] - cell["removed" + suffix]
    names = sorted({r["task"] for r in auto})
    counts = np.zeros((S.N_BOOT, len(names)))
    np.add.at(counts, (np.arange(S.N_BOOT)[:, None], S.boot_index(len(names), "batching", AUTOB)), 1)
    b_r, draws = resampled_b_r(auto, counts), {}
    for cell in (c for c in cells if c["bench"] == AUTOB):
        w = counts[:, [names.index(f"{AUTOB}:{t}") for t in cell["tasks"]]]
        lr, rm = w @ cell["dlog_x"] / w.sum(1), -b_r * (w @ cell["d_log_m"]) / w.sum(1)
        draws[cell["method"]] = dict(log_rho=lr, removed=rm, log_rho_cf=lr - rm)
    mean = {k: np.mean([d[k] for d in draws.values()], axis=0) for k in ("log_rho", "removed", "log_rho_cf")}
    boot = dict(b_R=S.interval(b_r), share_removed=S.interval(mean["removed"] / mean["log_rho"]), tasks=len(names),
                **{f"{k}.{c}": S.interval(v) for c, d in [("mean", mean)] + list(draws.items()) for k, v in d.items()})
    for cell in cells:
        del cell["tasks"]
    return dict(model=S.DEEPSEEK, fits=fits, cells=cells), boot


# ---- AutoBench writes and reads, both models

@functools.lru_cache(maxsize=None)
def classified(model, method):
    """{rep: {task: episode}} of a Table 1 AutoBench row: the class of each executed call per turn, and N, X, turns and
    success; complete when the outcome, calls and input tokens are all captured. An incomplete episode keeps the requests
    that completed."""
    out = {}
    for rep, fs in sorted(S.cell_files(model, AUTOB, method).items()):
        out[rep] = {}
        for task, e in S.episodes({rep: fs})[rep].items():
            row, seen, turns = task_row(e), set(), []
            complete = None not in (row["success"], row["N"], row["X"])
            for r in e.get("requests", []):
                if r["request_id"] not in seen:
                    seen.add(r["request_id"])
                    if complete or r.get("input_tokens") is not None:
                        turns.append([C.classify(c["tool"], c.get("arguments") or "") for c in r.get("tool_calls", []) if c.get("executed") is not False])
            assert not complete or (sum(map(len, turns)), len(turns)) == (row["N"], row["nreq"]), (model, method, task)
            if turns:
                out[rep][task] = dict(turns=turns, N=row["N"], X=row["X"], R=row["nreq"] if complete else len(turns),
                                      success=row["success"], complete=complete)
    return out


def flat(res):
    """A stats_of result with its seed and skill parts as 'seed.m', 'skill.m', ..."""
    out = {}
    for k, v in res.items():
        out |= {f"{k}.{j}": w for j, w in v.items()} if isinstance(v, dict) else {k: v}
    return out


def writes_and_reads():
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    cells = []
    for model in MODELS:
        seed = classified(model, "native-seed")
        for method in ROWS:
            own = classified(model, method)
            tasks, a, b = C.common_tasks(seed, own)
            ref = table1[f"{model}|{AUTOB}"][S.ARM_NAME[method]]
            check = dict(n=len(tasks), n_table1=ref["n"], rho_calls=math.exp(np.mean([math.log(b[t][0] / a[t][0]) for t in tasks])),
                         rho_calls_table1=ref["rho_inv"], rho_x=math.exp(np.mean([math.log(b[t][1] / a[t][1]) for t in tasks])),
                         rho_x_table1=ref["rho_trj"])
            assert model == S.LUNA or (len(tasks) == ref["n"] and math.isclose(check["rho_calls"], ref["rho_inv"])
                                       and math.isclose(check["rho_x"], ref["rho_trj"])), check
            loose = C.common_tasks(seed, own, only_complete=False)[0]
            for variant, local_w, complete, tset in (("main", False, True, tasks), ("local_writes_as_W", True, True, tasks),
                                                     ("all_recorded_turns", False, False, loose)):
                s, o = C.task_arrays(seed, tset, local_w, complete), C.task_arrays(own, tset, local_w, complete)
                draws = [[C.stats_of(s, o, i)[k] for k in INTERVALS] for i in S.boot_index(len(tset), "calls_rw", model, method, complete)]
                lo, hi = S.interval(np.array(draws))
                cells.append(dict(model=model, row=S.ARM_NAME[method], variant=variant, n=len(tset), check=check,
                                  point=flat(C.stats_of(s, o, np.arange(len(tset)))), ci=dict(zip(INTERVALS, zip(lo, hi))),
                                  turns_seed=int(s["turns"].sum()), turns_skill=int(o["turns"].sum()),
                                  eps_seed=int(s["eps"].sum()), eps_skill=int(o["eps"].sum())))
            print(model, method, "n", len(tasks), flush=True)
    episode = {S.ARM_NAME[m]: ["".join(c if c in "WR" else "O" for c in turn) or "-" for turn in classified(S.DEEPSEEK, m)[0][TASK]["turns"]]
               for m in ("native-seed",) + ROWS[1:]}
    return dict(ambig=dict(C.AMBIG), cells=cells), episode


# ---- records

def record_calls_per_turn(cells):
    mix, ro, runs, delta, lenient, bash = [], [], [], [], [], []
    for c in cells:
        for side in SIDES:
            s, n, label = c[side], c[side + "_lenient"], [c["method"], c["short"], c["n"], side]
            mix.append(label + [s["turns"], pct(s["share0"]), pct(s["share1"]), pct(s["share2p"]), f"{s['m']:.3f}", f"{s['m_task_mean']:.3f}"])
            ro.append(label + [pct(s["single_share"]), pct(s["single_readonly_of_turns"]), pct(s["single_readonly_of_single"])])
            runs.append(label + [f"{s[k]:.2f}" for k in ("runs_per_episode", "run_turns_per_episode")] + [pct(s["run_turns_share"])]
                        + [f"{s[k]:.2f}" for k in ("saved_per_episode", "mean_run_length")])
            lenient.append(label + [pct(n["single_readonly_of_turns"]), pct(n["single_readonly_of_single"]), f"{n['runs_per_episode']:.2f}",
                                    f"{n['saved_per_episode']:.2f}", pct(n["bash_pure_read_share"]) if c["bench"] != "tau3-bench" else "-"])
            if c["bench"] != "tau3-bench":
                bash.append(label + [s["bash_calls"], pct(s["bash_pure_read_share"]), "; ".join(f"{h} {v}" for h, v in s["non_pure_bash_top"][:4])])
        a, b = c["seed"], c["skill"]
        delta.append([c["method"], c["short"], f"{b['m'] - a['m']:+.3f}"] + [f"{100 * (b[k] - a[k]):+.1f}" for k in
                     ("share0", "share1", "share2p", "single_readonly_of_turns")] + [f"{b['saved_per_episode'] - a['saved_per_episode']:+.2f}"])
    head = ["Skill", "Bench", "n", "side"]
    numbers = "\n\n".join([
        "### A. Calls per turn", md_table(head + ["turns", "0 calls %", "1 call %", "2+ calls %", "m", "m_task"], mix),
        "### B. Single read-only-call turns", md_table(head + ["single-call turns % of turns", "single read-only % of turns",
                                                                "read-only % of single-call turns"], ro),
        "### C. Runs of >= 2 consecutive single read-only turns (per episode)",
        md_table(head + ["runs / episode", "turns in runs / episode", "turns in runs % of turns", "turns saved / episode", "mean run length"], runs),
        "### D. Change skill minus seed (m; percentage points; turns saved per episode)",
        md_table(["Skill", "Bench", "d m", "d 0-call pp", "d 1-call pp", "d 2+-call pp", "d single read-only pp", "d saved / episode"], delta),
        "### F. Sensitivity: lenient read-only rule", md_table(head + ["single read-only % of turns", "read-only % of single-call turns",
                                                                       "runs / episode", "turns saved / episode", "bash pure-read %"], lenient),
        "### E. Bash calls: share classified as pure reads, and the most frequent non-read commands",
        md_table(head + ["bash calls", "pure-read %", "top non-read command heads"], bash)])
    record(__spec__.name, "callsperturn", numbers, split_runs(), what=(
        "Opens up m, the executed tool calls per turn behind the turn-splitting term of mech_attribution3: how calls spread over "
        "turns, how many turns hold a single read-only call, and how many turns merging consecutive such turns would save, for "
        "the seed and each learned skill."))


def split_runs():
    return S.runs_text([(S.DEEPSEEK, b, m) for b in BENCHES for m in ("native-seed",) + OPT])


def record_writes_and_reads(rw, episode, b_r):
    by = collections.defaultdict(list)
    for c in rw["cells"]:
        by[c["variant"]].append(c)
    name = lambda c: [S.MODEL_SHORT[c["model"]], c["row"]]
    head = ["model", "skill", "n", "d >=2-write turns pp", "d >=2-read turns pp", "d writes in multi-write turn pp",
            "d reads in multi-read turn pp", "difference (writes - reads) pp", "d calls/turn"]

    def changes(variant):
        return md_table(head, [name(c) + [c["n"]] + [ci(c["point"][k], c["ci"][k]) for k in ("d_W2", "d_R2", "d_Wpar", "d_Rpar", "did_par")]
                               + [ci(c["point"]["d_m"], c["ci"]["d_m"], 1, 3)] for c in by[variant]])

    def dose(variant):
        e = lambda c, k: f"{math.exp(c['point'][k]):.3f} [{math.exp(c['ci'][k][0]):.3f}, {math.exp(c['ci'][k][1]):.3f}]"
        return md_table(["model", "skill", "n", "rho^calls", "rho^turns", "log rho^calls / log rho^turns", "writes/episode ratio",
                         "reads/episode ratio"],
                        [name(c) + [c["n"], e(c, "log_rho_calls"), e(c, "log_rho_turns"), ci(c["point"]["log_rho_calls"], c["ci"]["log_rho_calls"], 1, 3)
                                    + " / " + ci(c["point"]["log_rho_turns"], c["ci"]["log_rho_turns"], 1, 3), e(c, "log_W_ratio"), e(c, "log_R_ratio")]
                         for c in by[variant]])
    levels = []
    for c in by["main"]:
        p = c["point"]
        for side, eps, turns in (("seed", c["eps_seed"], c["turns_seed"]), ("skill", c["eps_skill"], c["turns_skill"])):
            levels.append([S.MODEL_SHORT[c["model"]], c["row"] + " vs Seed", c["n"], "Seed" if side == "seed" else c["row"], eps, turns]
                          + [f"{p[f'{side}.{k}']:.2f}" for k in ("m", "w_turn", "r_turn", "o_turn")]
                          + [pct(p[f"{side}.{k}"]) for k in ("W2", "R2", "Wpar", "Rpar")] + [f"{p[f'{side}.{k}']:.2f}" for k in ("W_ep", "R_ep")])
    packing = [name(c) + [c["n"], f"{c['point']['writes_per_write_turn_seed']:.2f}", f"{c['point']['writes_per_write_turn_skill']:.2f}",
                          ci(c["point"]["pack_saved_turns_ep"], c["ci"]["pack_saved_turns_ep"], 1, 2)]
               + [ci(c["point"][k], c["ci"][k], 1, 3) for k in ("log_turns_pooled", "pack_log_turns_removed")] for c in by["main"]]
    removed = [b_r * c["point"]["pack_log_turns_removed"] for c in by["main"] if c["model"] == S.DEEPSEEK and c["row"] != "No skill"]

    def tally(turns, kind):  # calls of a kind, and those sharing their turn with another of the kind
        return sum(t.count(kind) for t in turns), sum(t.count(kind) for t in turns if t.count(kind) >= 2)
    shown = [[m, len(turns), sum(len(t) for t in turns if t != "-"), *tally(turns, "W"), *tally(turns, "R"), " ".join(turns)]
             for m, turns in episode.items()]
    numbers = "\n\n".join([
        "Ambiguous cases and how they were resolved (calls over every extracted episode of both models and all five rows):",
        "\n".join(f"- {k}: {v}" for k, v in sorted(rw["ambig"].items(), key=lambda kv: -kv[1])),
        "### A. Levels (Table 1 task sets)",
        md_table(["model", "pair", "n tasks", "side", "episodes", "turns", "calls/turn", "writes/turn", "reads/turn", "other/turn",
                  ">=2 writes % of turns", ">=2 reads % of turns", "writes in a multi-write turn %", "reads in a multi-read turn %",
                  "writes/episode", "reads/episode"], levels),
        f"### B. Change, skill minus seed, with paired bootstrap {100 * S.LEVEL:.0f}% intervals (percentage points; calls per turn in calls)",
        changes("main"),
        "### C. Dose contrast: rho^calls, rho^turns and the write/read split (Table 1 task sets)", dose("main"),
        "### D. Packing counterfactual: the skill's writes packed at the seed's writes per write-carrying turn",
        md_table(["model", "skill", "n", "writes per write-carrying turn, seed", "same, skill", "turns saved / episode if packed like the seed",
                  "log turn ratio (pooled episodes)", "log turns removed by packing"], packing),
        f"Times AutoBench's b_R = {b_r:.2f} (mech_batching), the log turns packing removes lower DeepSeek's log rho by "
        f"{min(removed):.3f} to {max(removed):.3f} across the three learned skills.",
        "### E. Sensitivity: local file writes (write, edit, apply_patch) counted as writes", changes("local_writes_as_W"),
        "### F. Sensitivity: all recorded turns (every episode, completed requests)", changes("all_recorded_turns"), dose("all_recorded_turns"),
        f"### H. The episode of the writes figure: DeepSeek, {TASK}, first test run (W write, R read, O other, - no call)",
        md_table(["skill", "turns", "calls", "writes", "writes sharing a turn", "reads", "reads sharing a turn", "calls per turn"], shown)])
    record(__spec__.name, "calls_rw", numbers, S.runs_text([(m, AUTOB, r) for m in MODELS for r in ("native-seed",) + ROWS]), what=(
        "Do the serialize rules the optimizers add on AutoBench (issue mutating calls one at a time; parallel is safe only for "
        "reads) change exactly the behaviour they name? If they act, the writes that share a turn with another write fall under "
        "every skill with such a rule, and grouped reads do not. Also the contrast across skills against rho^calls, rho^turns and "
        "the write / read split, and how many of the skills' extra turns the serialized writes account for."))


def record_batching(data, boot):
    fits, cells = data["fits"], data["cells"]
    f = lambda fit, j: f"{fit['coef'][j]:.3f} ({fit['se'][j]:.3f})"
    rows = [[n, fits[k]["NR"]["n"], fits[k]["NR"]["tasks"]] + [f(fits[k][m], j) for m in ("NR", "Nm", "NR_noskill") for j in (0, 1)]
            for k, n in [("pooled", "pooled"), ("tau3-bench", "tau3"), (AUTOB, "AutoBench"), ("terminal-bench-2-1", "TB2")]]
    mean = lambda cs, k: f"{np.mean([c[k] for c in cs]):+.3f}"
    means = [[b, *(mean([c for c in cells if c["bench"] == b], k) for k in ("log_rho", "removed", "turn_splitting_term"))] for b in BENCHES]
    b = lambda key, nd=3: f"[{boot[key][0]:.{nd}f}, {boot[key][1]:.{nd}f}]"
    auto = [c for c in cells if c["bench"] == AUTOB]
    keys = ("log_rho", "removed", "log_rho_cf")
    boot_rows = [["three skills (mean)"] + [f"{np.mean([c[k] for c in auto]):.3f} {b(k + '.mean')}" for k in keys]
                 + [f"{np.mean([c['removed'] for c in auto]) / np.mean([c['log_rho'] for c in auto]):.2f} {b('share_removed', 2)}"]]
    boot_rows += [[c["method"]] + [f"{c[k]:.3f} {b(k + '.' + c['method'])}" for k in keys] + [""] for c in auto]
    numbers = "\n\n".join([
        "### Within-task regression log X = a_task + c_skill + b_N log N + b_R log R (task-clustered SE in brackets)",
        md_table(["pool", "episodes", "tasks", "b_N", "b_R", "g_N (log N)", "g_m (log N/R)", "b_N no skill FE", "b_R no skill FE"], rows),
        "### Counterfactual: each skill keeps the seed's calls per turn on every task",
        md_table(["Skill", "Bench", "tasks", "log rho", "removed by seed's m (b_R)", "log rho after", "turn-splitting term (mech_attribution3 rule)",
                  "removed with pooled b_R"], [[c["method"], S.BENCH_SHORT[c["bench"]], c["n"]] + [f"{c[k]:+.3f}" for k in
                  ("log_rho", "removed", "log_rho_cf", "turn_splitting_term", "removed_pooled_slope")] for c in cells]),
        "### Benchmark means against the turn-splitting term",
        md_table(["Bench", "mean log rho", "mean removed (b_R)", "mean turn-splitting term"],
                 means + [["all nine"] + [mean(cells, k) for k in ("log_rho", "removed", "turn_splitting_term")]]),
        f"### AutoBench, {100 * S.LEVEL:.0f}% bootstrap intervals over its {boot['tasks']} tasks (b_R re-fitted on each draw: "
        f"{fits[AUTOB]['NR']['coef'][1]:.3f} {b('b_R')})",
        md_table(["AutoBench", "log rho as recorded", "removed at the seed's calls per turn", "log rho after", "share removed"], boot_rows)])
    record(__spec__.name, "batching", numbers, split_runs(), what=(
        "Design principle 'issue independent calls in one turn'. Within a task, how many more input tokens an episode bills when "
        "it spreads the same calls over more turns (b_R), and how much of each learned skill's token tax would go if it kept the "
        "seed's calls per turn on every task, set beside the turn-splitting term of mech_attribution3."))


def main():
    out = S.cache("calls")
    pairs = {(b, m): split_pairs(b, m) for b in BENCHES for m in OPT}
    per_turn = calls_per_turn(pairs)
    (out / "callsperturn.json").write_text(json.dumps(dict(model=S.DEEPSEEK, cells=per_turn), indent=1))
    record_calls_per_turn(per_turn)
    data, boot = batching(pairs)
    (out / "batching.json").write_text(json.dumps(data, indent=1))
    record_batching(data, boot)
    rw, episode = writes_and_reads()
    (out / "calls_rw.json").write_text(json.dumps(rw, indent=1))
    (out / "episode.json").write_text(json.dumps(episode, indent=1))
    record_writes_and_reads(rw, episode, data["fits"][AUTOB]["NR"]["coef"][1])


if __name__ == "__main__":
    main()
