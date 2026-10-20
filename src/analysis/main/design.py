"""Paper 1 (mechanism paper), Section 5: the evidence behind the design principles for a cost-aware skill optimizer
(merged from batching.py, designev.py, selrules.py and implications_fig.py).  Four subcommands:

  batching   design principle "issue independent calls in one turn": what does spreading the same tool calls over more turns cost in input
             tokens?  The harness re-sends its history every turn, so input tokens X grow faster than turns R; an editor that writes rules
             which cut cost should therefore batch independent calls (higher calls per turn m = N / R bill fewer tokens).  (1) Per benchmark
             the test episodes of the seed and of the three learned skills (three runs each) are pooled and fitted with task fixed effects
             (and a skill fixed effect in the headline fit): log X = a_task [+ c_skill] + b_N log N + b_R log R, SEs clustered by task; the
             re-parameterised fit on log N and log m is the identical model (g_m = -b_R).  (2) Counterfactual inside the measurement: a skill
             keeps its calls N but is given the seed's calls per turn m_seed on that task, so log X falls by -b_R (log m_skill - log m_seed);
             compared with the turn-splitting term of mech_attribution3.  Accounting, not an experiment.
             -> assets/cache/analysis/main/batching.json, data/mech/mech_batching.md
  designev   empirical support of design principles P2, P3 and P4 from the logged baseline proposals (no new runs).
             A (P2, price the behavior a rule induces): over the 59 canonical dev edges, R^2 and Spearman of dlogX on the change in log skill
             length, dlogR or dlogN; at test level (nine learned skills) log rho of input tokens on the skill's word ratio vs the seed and on
             log rho_turns.  B (P4, make proposals cost-aware): share of all 59 proposals with dlogX > 0, dlogX of kept vs rejected, and a
             counterfactual gate (accept iff dQ >= 0 and dlogX <= 0) applied to the logged proposals.  C (P3, give the editor operators that
             cut cost): edges that remove text or shorten the skill, chars added vs removed, mean dlogX of removing/shortening edges vs pure
             additions.  Counterfactual, not an experiment.
             -> assets/cache/analysis/main/designev.json, data/mech/mech_designev.md
  selrules   design principle "select against the noise of selection": how often four acceptance rules accept ONE candidate against its
             parent when the candidate is identical to the parent (false acceptance) and when it truly solves g more dev tasks in expectation
             (power), on selection.py's beta-binomial task model of the seed's dev tasks; then the detectability gate applied to the logged
             accepted edges of the Table 1 runs (edge dev gain in tasks = dQ x n_pair / 100 against the one-sided 0.05 critical value of an
             n_pair-task comparison), with refused vs surviving edges compared on dlogX.  Rules: (a) better on one run, (b) gain >= the
             one-sided 0.05 critical value of a single-run comparison under no effect, (c) better then better again on
             a fresh draw, (d) mean of 3 runs.
             -> assets/cache/analysis/main/selrules.json, data/mech/mech_selrules.md
  draw       the Section 5 evidence figure: one panel per principle, laid out from the three caches —
             (a) selrules panel_bars (what each rule keeps), (b) designev panel_predict (what explains token change), (c) batching panel
             (the tax with the seed's calls per turn restored), (d) designev panel_gate (kept vs rejected proposals and the cost-aware gate)
             -> fig/mech/mech_implications.{pdf,svg}

Commands (from bipa):  .venv/bin/python -m src.analysis.main.design {batching|designev|selrules|draw}
"""
import argparse
import json
import math

import numpy as np
from scipy import stats

from src.analysis.main import table1 as T
from src.analysis.main import attribution as A
from src.analysis.main import edges as E
from src.analysis.main import selection as S
from src.utils import fig, logger, dataloader


# ---------- batching (from batching.py) ----------

BENCH_NAME = {"tau3-bench": r"$\tau^3$", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}


def fe_fit(task, skill, columns, y):
    """Within-task OLS of y on the columns plus skill dummies, task-clustered standard errors.

    task, skill: label arrays; columns: (n, k) array of regressors; y: (n,).  Task fixed effects are removed by demeaning every variable inside its task (Frisch-Waugh);
    the skill fixed effect enters as dummies (first level dropped) that are demeaned the same way.  Returns the coefficients and clustered SEs of the k regressors."""
    skill_levels = sorted(set(skill))[1:]
    dummies = np.array([[1.0 if s == level else 0.0 for level in skill_levels] for s in skill]).reshape(len(y), len(skill_levels))
    design = np.hstack([columns, dummies]) if skill_levels else columns
    task_ids = np.unique(task, return_inverse=True)[1]
    design = design.astype(float)
    y = y.astype(float)
    for t in np.unique(task_ids):
        rows = task_ids == t
        design[rows] -= design[rows].mean(axis=0)
        y[rows] -= y[rows].mean()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for t in np.unique(task_ids):
        rows = task_ids == t
        score = design[rows].T @ resid[rows]
        meat += np.outer(score, score)
    n_tasks = len(np.unique(task_ids))
    # small-sample factor of the usual cluster-robust estimator
    factor = n_tasks / (n_tasks - 1) * (len(y) - 1) / (len(y) - design.shape[1])
    se = np.sqrt(np.diag(bread @ meat @ bread) * factor)
    k = columns.shape[1]
    return beta[:k], se[:k], dict(n=len(y), tasks=n_tasks)


def regressions(records):
    """Fits of one episode pool: [(N, R) fit with skill FE, (N, m) fit with skill FE, (N, R) fit without skill FE] as dicts."""
    task = np.array([r["task"] for r in records])
    skill = np.array([r["skill"] for r in records])
    log_n = np.log([r["N"] for r in records])
    log_r = np.log([r["R"] for r in records])
    log_x = np.log([r["X"] for r in records])
    log_m = log_n - log_r
    out = {}
    for name, columns, with_skill in (("NR", np.column_stack([log_n, log_r]), True),
                                      ("Nm", np.column_stack([log_n, log_m]), True),
                                      ("NR_noskill", np.column_stack([log_n, log_r]), False)):
        labels = skill if with_skill else np.zeros(len(task))
        beta, se, size = fe_fit(task, labels, columns, log_x)
        out[name] = dict(coef=beta.tolist(), se=se.tolist(), **size)
    return out


def bench_numbers(bench):
    """Episode records of the seed and the three skills on one benchmark, and the counterfactual of each skill."""
    records = []
    cells = []
    seed_added = False
    for method_disp, method in A.OPT:
        own_files, seed_files = A.run_files(method, bench)
        own_eps, own_rows = A.load_runs(own_files)
        seed_eps, seed_rows = A.load_runs(seed_files)
        reps = sorted(own_files)
        # Table 1's common task set, then tasks with usable metrics in all six runs (as mech_attribution3)
        _, own_mean = T.mean_rows(own_files)
        _, seed_mean = T.mean_rows(seed_files)
        common = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
        usable = []
        pairs = {}
        for t in common:
            rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                          for rows in (own_rows, seed_rows) for r in reps)
            if not rows_ok:
                continue
            if not all(t in eps[r] and eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
                continue
            a = A.mean_metrics([seed_eps[r][t][1] for r in reps])
            b = A.mean_metrics([own_eps[r][t][1] for r in reps])
            if a is None or b is None:
                continue
            usable.append(t)
            pairs[t] = (a, b)
        # episode records for the regression: every graph episode with usable metrics
        for eps, label in ((own_eps, method_disp), (seed_eps, "seed")):
            if label == "seed" and seed_added:
                continue
            for r in reps:
                for t, (_, m) in eps[r].items():
                    if m:
                        records.append(dict(task=f"{bench}:{t}", skill=label, N=m["N"], R=m["R"], X=m["X"]))
        seed_added = True
        terms, d_x = A.four_terms([pairs[t] for t in usable])
        d_log_m = np.array([math.log((pairs[t][1]["N"] / pairs[t][1]["R"]) / (pairs[t][0]["N"] / pairs[t][0]["R"])) for t in usable])
        cells.append(dict(method=method_disp, bench=bench, n=len(usable), log_rho=float(d_x.mean()), turn_splitting_term=float(terms["turn_splitting"].mean()),
                          d_log_m=d_log_m.tolist(), dlog_x=d_x.tolist(), e_r_dlog_m=(-terms["turn_splitting"]).tolist()))
        print(method_disp, bench, "tasks", len(usable), "log rho", round(float(d_x.mean()), 3), "turn-splitting", round(float(terms["turn_splitting"].mean()), 3))
    return records, cells


def compute_batching():
    """Fit per benchmark and pooled, apply the counterfactual with the benchmark's b_R, write batching.json."""
    fits = {}
    cells = []
    pooled_records = []
    for bench, _ in A.BENCH:
        records, bench_cells = bench_numbers(bench)
        pooled_records += records
        fits[bench] = regressions(records)
        cells += bench_cells
    fits["pooled"] = regressions(pooled_records)
    for cell in cells:
        for source in (cell["bench"], "pooled"):
            b_r = fits[source]["NR"]["coef"][1]
            # fall of log X when the skill keeps the seed's calls per turn: -b_R dlog m, task mean
            removed = float(np.mean(-b_r * np.array(cell["d_log_m"])))
            key = "removed" if source == cell["bench"] else "removed_pooled_slope"
            cell[key] = removed
            cell["log_rho_cf" if source == cell["bench"] else "log_rho_cf_pooled_slope"] = cell["log_rho"] - removed
    data = dict(model=A.MODEL, fits=fits, cells=cells)
    dataloader.write_json("batching", data)
    for name, fit in fits.items():
        print(name, {k: [round(c, 3) for c in v["coef"]] for k, v in fit.items()}, "n", fit["NR"]["n"])
    return data


def mean_over(cells, key):
    """Mean of a cell field over a list of cells."""
    return float(np.mean([c[key] for c in cells]))


def panel(ax, data):
    """Compact panel: per benchmark (and all nine skills) the mean token tax log rho of the three skills as recorded and with the seed's calls per turn restored."""
    groups = [(BENCH_NAME[b], [c for c in data["cells"] if c["bench"] == b]) for b, _ in A.BENCH]
    groups.append(("all", data["cells"]))
    x = np.arange(len(groups))
    width = 0.38
    measured = [mean_over(cells, "log_rho") for _, cells in groups]
    counterfactual = [mean_over(cells, "log_rho_cf") for _, cells in groups]
    ax.bar(x - width / 2, measured, width, color="#9e9e9e", label="as recorded")
    ax.bar(x + width / 2, counterfactual, width, color="#2166ac", label="at seed's calls per turn")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in groups], fontsize=7)
    ax.set_ylabel(r"token tax ($\log\rho$)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylim(min(0, min(counterfactual)) - 0.01, max(measured) * 1.75)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.1, labelspacing=0.2, handletextpad=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def record_batching(data):
    """Audit record mech_batching.md."""
    fits = data["fits"]
    names = {"pooled": "pooled"}
    names.update({b: n for b, n in (("tau3-bench", "tau3"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2"))})
    rows = []
    for key, name in names.items():
        f = fits[key]
        rows.append([name, f["NR"]["n"], f["NR"]["tasks"], "%.3f (%.3f)" % (f["NR"]["coef"][0], f["NR"]["se"][0]), "%.3f (%.3f)" % (f["NR"]["coef"][1], f["NR"]["se"][1]),
                     "%.3f (%.3f)" % (f["Nm"]["coef"][0], f["Nm"]["se"][0]), "%.3f (%.3f)" % (f["Nm"]["coef"][1], f["Nm"]["se"][1]),
                     "%.3f (%.3f)" % (f["NR_noskill"]["coef"][0], f["NR_noskill"]["se"][0]), "%.3f (%.3f)" % (f["NR_noskill"]["coef"][1], f["NR_noskill"]["se"][1])])
    t1 = logger.md_table(["pool", "episodes", "tasks", "b_N", "b_R", "g_N (log N)", "g_m (log N/R)", "b_N no skill FE", "b_R no skill FE"], rows)
    cell_rows = []
    for c in data["cells"]:
        cell_rows.append([c["method"], BENCH_NAME[c["bench"]].replace("$", "").replace("\\", ""), c["n"], "%+.3f" % c["log_rho"], "%+.3f" % c["removed"], "%+.3f" % c["log_rho_cf"],
                          "%+.3f" % c["turn_splitting_term"], "%+.3f" % c["removed_pooled_slope"]])
    t2 = logger.md_table(["Skill", "Bench", "tasks", "log rho", "removed by seed's m (b_R)", "log rho after", "turn-splitting term (mech_attribution3 rule)", "removed with pooled b_R"], cell_rows)
    summary = []
    for b, _ in A.BENCH:
        cs = [c for c in data["cells"] if c["bench"] == b]
        summary.append([b, "%+.3f" % mean_over(cs, "log_rho"), "%+.3f" % mean_over(cs, "removed"), "%+.3f" % mean_over(cs, "turn_splitting_term")])
    summary.append(["all nine", "%+.3f" % mean_over(data["cells"], "log_rho"), "%+.3f" % mean_over(data["cells"], "removed"), "%+.3f" % mean_over(data["cells"], "turn_splitting_term")])
    t3 = logger.md_table(["Bench", "mean log rho", "mean removed (b_R)", "mean turn-splitting term"], summary)
    table = ("### Within-task regression log X = a_task + c_skill + b_N log N + b_R log R (task-clustered SE in brackets)\n\n" + t1 +
             "\n\n### Counterfactual: each skill keeps the seed's calls per turn on every task\n\n" + t2 + "\n\n### Benchmark means against the turn-splitting term\n\n" + t3)
    logger.write_record(
        "batching",
        "Design principle 'issue independent calls in one turn'. Within a task, how many more input tokens does an episode bill when it spreads the same calls over more turns (b_R), and how much "
        "of each learned skill's token tax would disappear if it kept the seed's calls per turn on every task.  The counterfactual is compared with the turn-splitting term of mech_attribution3.",
        "DeepSeek-V4.1-Flash in OpenCode, test split: seed (3 runs) and GEPA, SkillOpt, SkillAdam (3 runs each) on tau3-bench, automationbench, terminal-bench-2-1; same run files as mech_attribution3.",
        "Episodes with R>=2, a call and positive history (slopes_checks.metrics). Regression: task fixed effects by within-task demeaning, skill fixed effect (seed + three skills) as dummies, SE clustered by task. "
        "The N, N/R fit is the same model re-parameterised (g_m = -b_R). Counterfactual: R_cf = N / m_seed on 3-run task means, removed = mean over tasks of -b_R * (log m_skill - log m_seed), "
        "using the benchmark's own b_R; tasks and 3-run means are those of mech_attribution3. Accounting only, no episode is re-run.",
        table,
        ".venv/bin/python -m src.analysis.main.design batching")


# ---------- designev (from designev.py) ----------

OPTS = E.OPTS
NAMES = E.NAMES
SELF_CMD = ".venv/bin/python -m src.analysis.main.design designev"


def fit(x, y):
    """R^2 of the simple regression y ~ x and the Spearman correlation, with the sample size."""
    reg = stats.linregress(x, y)
    rho = stats.spearmanr(x, y)
    return dict(n=len(x), r2=float(reg.rvalue ** 2), slope=float(reg.slope), r=float(reg.rvalue),
                spearman=float(rho.statistic), spearman_p=float(rho.pvalue))


def part_a(edges):
    """What predicts dlogX: edge level (59 dev edges) and test level (nine learned skills)."""
    dlogx = np.array([e["dlogX"] for e in edges])
    dloglen = np.array([np.log(e["len_c"] / e["len_p"]) for e in edges])
    dlogr = np.array([e["dlogR"] for e in edges])
    dlogn = np.array([e["dlogN"] for e in edges])
    edge_level = {
        "skill length": fit(dloglen, dlogx),
        "turns": fit(dlogr, dlogx),
        "actions": fit(dlogn, dlogx),
    }
    # test level: log rho of input tokens vs log word ratio of the skill and log rho_turns
    attribution = json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"]
    compaction = json.loads((dataloader.CACHE / "compaction.json").read_text())["cells"]
    sizes = json.loads((dataloader.CACHE / "skilltext.json").read_text())["sizes"]
    turns = {(c["method"], c["bench"]): c["rho_turns"] for c in compaction}
    cells = []
    for c in attribution:
        key = c["method"].lower() + "|" + c["bench"]
        cells.append(dict(method=c["method"], bench=c["bench"], log_rho=c["A"]["total"],
                          log_len=float(np.log(sizes[key]["words_ratio"])), log_rho_turns=float(np.log(turns[(c["method"], c["bench"])]))))
    log_rho = np.array([c["log_rho"] for c in cells])
    test_level = {
        "skill length": fit(np.array([c["log_len"] for c in cells]), log_rho),
        "turns": fit(np.array([c["log_rho_turns"] for c in cells]), log_rho),
    }
    return dict(edge_level=edge_level, test_level=test_level, test_cells=cells)


def part_b(edges):
    """Proposals above zero, kept vs rejected, and the cost-aware gate (accept iff dQ >= 0 and dlogX <= 0)."""
    by_opt = {}
    for opt in OPTS:
        group = [e for e in edges if e["opt"] == opt]
        by_opt[opt] = dict(n=len(group), n_above=int(sum(e["dlogX"] > 0 for e in group)),
                           share_above=float(np.mean([e["dlogX"] > 0 for e in group])) if group else None)
    accepted = [e for e in edges if e["accepted"]]
    rejected = [e for e in edges if not e["accepted"]]
    dq_ok = [e for e in edges if e["dQ"] >= 0]
    passing = [e for e in edges if e["dQ"] >= 0 and e["dlogX"] <= 0]
    kept_acc = [e for e in passing if e["accepted"]]
    kept_rej = [e for e in passing if not e["accepted"]]
    gate_set = kept_acc + kept_rej
    return dict(
        n=len(edges), n_above=int(sum(e["dlogX"] > 0 for e in edges)), share_above=float(np.mean([e["dlogX"] > 0 for e in edges])),
        by_opt=by_opt,
        accepted=dict(n=len(accepted), mean=float(np.mean([e["dlogX"] for e in accepted])), median=float(np.median([e["dlogX"] for e in accepted])),
                      share_above=float(np.mean([e["dlogX"] > 0 for e in accepted])), values=[e["dlogX"] for e in accepted]),
        rejected=dict(n=len(rejected), mean=float(np.mean([e["dlogX"] for e in rejected])), median=float(np.median([e["dlogX"] for e in rejected])),
                      share_above=float(np.mean([e["dlogX"] > 0 for e in rejected])), values=[e["dlogX"] for e in rejected]),
        mannwhitney_p=float(stats.mannwhitneyu([e["dlogX"] for e in accepted], [e["dlogX"] for e in rejected]).pvalue),
        n_dq_ok=len(dq_ok),
        gate=dict(n_pass=len(passing), accepted_kept=len(kept_acc), accepted_total=len(accepted),
                  rejected_taken=len(kept_rej), rejected_total=len(rejected),
                  mean_dlogX_kept=float(np.mean([e["dlogX"] for e in gate_set])) if gate_set else None,
                  mean_dlogX_accepted_kept=float(np.mean([e["dlogX"] for e in kept_acc])) if kept_acc else None,
                  passing=[dict(opt=e["opt"], bench=e["bench"], child=e["child"], accepted=e["accepted"], dlogX=e["dlogX"], dQ=e["dQ"]) for e in passing]),
        dlogx_min=float(min(e["dlogX"] for e in edges)),
    )


def part_c(edges):
    """Edges that remove text or shorten the skill, chars added vs removed, and their dlogX against pure additions."""
    removing = [e for e in edges if e["rem_chars"] > 0]
    shortening = [e for e in edges if e["len_c"] < e["len_p"]]
    additions = [e for e in edges if e["rem_chars"] == 0]
    by_opt = {}
    for opt in OPTS:
        group = [e for e in edges if e["opt"] == opt]
        by_opt[opt] = dict(n=len(group), added=int(sum(e["add_chars"] for e in group)), removed=int(sum(e["rem_chars"] for e in group)),
                           n_removing=int(sum(e["rem_chars"] > 0 for e in group)), n_shortening=int(sum(e["len_c"] < e["len_p"] for e in group)))

    def mean_of(group):
        return float(np.mean([e["dlogX"] for e in group])) if group else None

    return dict(n=len(edges), n_removing=len(removing), n_shortening=len(shortening), n_additions=len(additions),
                added=int(sum(e["add_chars"] for e in edges)), removed=int(sum(e["rem_chars"] for e in edges)),
                by_opt=by_opt, mean_dlogX_removing=mean_of(removing), mean_dlogX_shortening=mean_of(shortening),
                mean_dlogX_additions=mean_of(additions),
                shortening=[dict(opt=e["opt"], bench=e["bench"], accepted=e["accepted"], dlogX=e["dlogX"], dQ=e["dQ"]) for e in shortening])


def compute_designev():
    """Parts A, B, C on the 59 canonical dev edges and the nine test-level skills."""
    edges = E.load_dev_edges()
    out = dict(n_edges=len(edges), A=part_a(edges), B=part_b(edges), C=part_c(edges))
    dataloader.write_json("designev", out)
    print("A edge", {k: round(v["r2"], 3) for k, v in out["A"]["edge_level"].items()}, "test", {k: round(v["r2"], 3) for k, v in out["A"]["test_level"].items()})
    print("B", out["B"]["share_above"], out["B"]["gate"]["n_pass"], out["B"]["gate"]["accepted_kept"], out["B"]["gate"]["rejected_taken"])
    print("C", out["C"]["n_removing"], out["C"]["n_shortening"], out["C"]["added"], out["C"]["removed"])
    return out


def panel_predict(ax, data):
    """R^2 of dlogX on skill-length change, turns and actions: dev edges (dark bars) and, for the length and turn ratios, the nine test-level skills (light bars)."""
    edge = data["A"]["edge_level"]
    test = data["A"]["test_level"]
    # two populations, drawn as two labelled groups: the dev-scored edits (top) and the nine learned skills (bottom)
    labels = ["skill length", "turns", "tool calls", "skill length", "turns"]
    values = [edge["skill length"]["r2"], edge["turns"]["r2"], edge["actions"]["r2"], test["skill length"]["r2"], test["turns"]["r2"]]
    colors = ["#999999", "#2166ac", "#2166ac", "#cccccc", "#92c5de"]
    ypos = np.array([5.6, 4.6, 3.6, 1.4, 0.4])
    ax.barh(ypos, values, color=colors, height=0.68)
    ax.text(-0.02, 6.45, "59 dev-scored edits", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.text(-0.02, 2.25, "9 learned skills", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.set_ylim(-0.3, 6.8)
    for y, v in zip(ypos, values):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.5, linespacing=0.9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"token change explained ($R^2$)", fontsize=7, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_gate(ax, data):
    """dlogX (x100 log points) of rejected and accepted proposals as a strip with a zero line; the share above zero and the cost-aware gate result are written on the panel."""
    b = data["B"]
    rng = np.random.default_rng(0)
    groups = [("rejected", b["rejected"], "#b2182b"), ("accepted", b["accepted"], "#2166ac")]
    for i, (name, g, color) in enumerate(groups):
        values = 100 * np.array(g["values"])
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(i + jitter, values, s=7, color=color, alpha=0.75, linewidths=0)
        ax.hlines(100 * g["mean"], i - 0.3, i + 0.3, color="k", lw=1.0)
        ax.text(i, 112, f"{100 * g['share_above']:.0f}% add", ha="center", va="top", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"rejected\n({b['rejected']['n']} edits)", f"kept\n({b['accepted']['n']} edits)"], fontsize=6.5, linespacing=0.9)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(-95, 115)
    ax.set_ylabel(r"change in log input tokens ($\times$100)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", length=0)
    g = b["gate"]
    ax.text(0.5, 0.02, f"a cost gate passes\n{g['accepted_kept']} of {g['accepted_total']} and {g['rejected_taken']} of {g['rejected_total']}",
            transform=ax.transAxes, fontsize=5.8, va="bottom", ha="center", linespacing=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fmt(x, nd=2):
    """Number with nd decimals, 'n/a' for None."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def record_designev(data):
    """paper/assets/data/mech/mech_designev.md."""
    a, b, c = data["A"], data["B"], data["C"]
    rows_a = []
    for level, fits in (("dev edges (59), dlogX on", a["edge_level"]), ("nine skills, log rho on", a["test_level"])):
        for name, f in fits.items():
            rows_a.append([level, name, f["n"], fmt(f["r2"]), fmt(f["spearman"]), fmt(f["spearman_p"], 3), fmt(f["slope"])])
    table_a = logger.md_table(["level", "predictor", "n", "R^2", "Spearman", "p", "slope"], rows_a)
    rows_b = [[NAMES[o], v["n"], v["n_above"], fmt(v["share_above"])] for o, v in b["by_opt"].items()]
    rows_b.append(["all", b["n"], b["n_above"], fmt(b["share_above"])])
    table_b = logger.md_table(["optimizer", "proposals", "dlogX > 0", "share"], rows_b)
    rows_k = [[n, g["n"], fmt(100 * g["mean"], 1), fmt(100 * g["median"], 1), fmt(g["share_above"])] for n, g in (("accepted", b["accepted"]), ("rejected", b["rejected"]))]
    table_k = logger.md_table(["status", "n", "mean dlogX x100", "median x100", "share > 0"], rows_k)
    g = b["gate"]
    gate_lines = (f"Gate accept iff dQ >= 0 and dlogX <= 0: {g['n_pass']} of {b['n']} proposals pass (min dlogX over all proposals = {fmt(100 * b['dlogx_min'], 1)} x100). "
                  f"It keeps {g['accepted_kept']} of {g['accepted_total']} accepted edges and would accept {g['rejected_taken']} of {g['rejected_total']} rejected ones; "
                  f"mean dlogX of the kept set = {fmt(None if g['mean_dlogX_kept'] is None else 100 * g['mean_dlogX_kept'], 1)} x100. "
                  f"{b['n_dq_ok']} proposals have dQ >= 0. Mann-Whitney p (accepted vs rejected dlogX) = {fmt(b['mannwhitney_p'], 3)}.\n\n"
                  + logger.md_table(["optimizer", "child", "accepted", "dlogX x100", "dQ"], [[NAMES[p["opt"]] + " " + p["bench"], p["child"], p["accepted"], fmt(100 * p["dlogX"], 1), fmt(p["dQ"], 1)] for p in g["passing"]]))
    rows_c = [[NAMES[o], v["n"], v["n_removing"], v["n_shortening"], v["added"], v["removed"]] for o, v in c["by_opt"].items()]
    rows_c.append(["all", c["n"], c["n_removing"], c["n_shortening"], c["added"], c["removed"]])
    table_c = logger.md_table(["optimizer", "edges", "remove text", "shorten skill", "chars added", "chars removed"], rows_c)
    c_lines = (f"Mean dlogX x100: edges that remove text ({c['n_removing']}) = {fmt(None if c['mean_dlogX_removing'] is None else 100 * c['mean_dlogX_removing'], 1)}; "
               f"edges that shorten the skill ({c['n_shortening']}) = {fmt(None if c['mean_dlogX_shortening'] is None else 100 * c['mean_dlogX_shortening'], 1)}; "
               f"pure additions ({c['n_additions']}) = {fmt(100 * c['mean_dlogX_additions'], 1)}.")
    table = ("### A. what predicts dlogX\n\n" + table_a + "\n\n### B. proposals and the cost-aware gate\n\n" + table_b + "\n\n" + table_k + "\n\n" + gate_lines
             + "\n\n### C. operators: does the editor ever remove text?\n\n" + table_c + "\n\n" + c_lines)
    what = ("Empirical support for three design principles. P2: skill length is a poor price of a candidate's token tax; turns and actions explain it. P4: nearly all logged proposals add tokens, so a success-only gate "
            "has nothing cost-free to choose from, and a cost-aware gate applied to the same proposals has almost nothing to keep. P3: the editors almost only add text. A counterfactual on logged proposals, not a new run.")
    runs = "The 59 canonical dev edges of native-20261008-https (edges_table.json, SkillAdam x terminal-bench-2-1 excluded); test level: the nine learned skills of mech_attribution3 / mech_compaction / mech_skilltext."
    agg = ("A: scipy linregress R^2 and spearmanr of dlogX on log(len_c/len_p), dlogR, dlogN over edges; test level: log rho of input tokens (attribution total, Table 1 log rho_trj) on log(words of skill / words of seed) and on log rho_turns "
           "(compaction rho_turns). B: counts over all 59 edges, gate accept iff dQ >= 0 and dlogX <= 0. C: rem_chars > 0, len_c < len_p, sums of add_chars and rem_chars. No intervals; the sample is 59 edges and 9 skills.")
    logger.write_record("designev", what, runs, agg, table, SELF_CMD)


# ---------- selrules (from selrules.py) ----------

RULES = [("better", "(a) better on one run"), ("critical", "(b) gain >= one-sided 0.05 critical value"), ("confirm", "(c) better, then better again"),
         ("mean3", "(d) mean of 3 runs")]
TRUE_GAINS = (1, 2, 4)       # tasks; the panel shows PLOT_GAIN
PLOT_GAIN = 2
DRAWS = 100_000
COLORS = {"terminal-bench-2-1": "#D55E00", "tau3-bench": "#0072B2", "automationbench": "#009E73"}


# ---- the simulation ----------------------------------------------------------------------------------------------------------------

def runs_total(p, repeats):
    """Tasks solved summed over `repeats` independent runs of a skill with task probabilities p (draws, tasks)."""
    total = np.zeros(p.shape[0], dtype=int)
    for _ in range(repeats):
        total += S.outcomes(p)
    return total


def acceptance(p, q, critical) -> dict:
    """{rule: acceptance rate} of one candidate with task probabilities q against a parent with p, on the same draws of the task model."""
    parent, child = S.outcomes(p), S.outcomes(q)
    net = child - parent
    # (c) a fresh draw of both after a first win
    net_again = S.outcomes(q) - S.outcomes(p)
    # (d) three runs each, summed
    net_mean3 = runs_total(q, 3) - runs_total(p, 3)
    return {"better": float((net > 0).mean()),
            "critical": float((net >= critical).mean()),
            "confirm": float(((net > 0) & (net_again > 0)).mean()),
            "mean3": float((net_mean3 > 0).mean())}


def panel_threshold(dev, prior, size, index) -> int:
    """One-sided 0.05 critical net difference (tasks) of a single-run comparison on `size` dev tasks (the panel, or a random subset of it)
    when the candidate is identical to its parent."""
    S.reseed(index, 31, size)
    p = S.task_probabilities(dev, prior, size, S.DRAWS)
    return S.critical_value(p)


def analyse(bench) -> dict:
    """Null acceptance and power per rule for one benchmark, plus the critical values of the panel sizes used by the logged edges."""
    dev_files, test_files = S.seed_files(bench)
    dev, test = S.executions(dev_files), S.executions(test_files)
    S.check_inputs(bench, dev, test)
    n_dev = len(dev)
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    prior = S.fit_prior(pooled)
    index = [x for x, _ in S.BENCHES].index(bench)
    S.reseed(index, 30)
    p = S.task_probabilities(dev, prior, n_dev, DRAWS)
    critical = panel_threshold(dev, prior, n_dev, index)
    S.reseed(index, 32)
    out = {"dev_tasks": n_dev, "critical_tasks": critical, "null": acceptance(p, p, critical), "power": {}}
    for gain in TRUE_GAINS:
        out["power"][str(gain)] = acceptance(p, S.raised(p, gain), critical)
    # thresholds of the smaller comparisons logged (SkillAdam's validation minibatches)
    sizes = sorted({r["n_pair"] for r in logged_edges() if r["bench"] == bench})
    out["critical_by_size"] = {str(s): (critical if s == n_dev else panel_threshold(dev, prior, s, index)) for s in sizes}
    return out


# ---- the logged edges --------------------------------------------------------------------------------------------------------------

def logged_edges() -> list:
    """The canonical dev edges (59) of the Table 1 runs."""
    return E.load_dev_edges()


def weighted_mean(values, weights) -> float:
    """Mean of values with weights."""
    return float(np.average(values, weights=weights))


def edge_gate(result) -> dict:
    """Of the accepted dev edges, those a rule-(b) gate refuses (dev gain in tasks below the critical value of the edge's comparison size) and their mean
    dlogX (x100 log points), unweighted and weighted by n_pair, against the surviving edges."""
    accepted = [r for r in logged_edges() if r["accepted"]]
    gain_tasks = np.array([r["dQ"] * r["n_pair"] / 100 for r in accepted])
    threshold = np.array([result[r["bench"]]["critical_by_size"][str(r["n_pair"])] for r in accepted])
    refused = gain_tasks < threshold
    dlogx = 100 * np.array([r["dlogX"] for r in accepted])
    weight = np.array([r["n_pair"] for r in accepted], float)
    out = {"accepted": len(accepted), "refused": int(refused.sum()), "survive": int((~refused).sum()),
           "gain_tasks": gain_tasks.tolist(), "threshold_tasks": threshold.tolist(), "per_bench": {}}
    for name, mask in (("refused", refused), ("survive", ~refused)):
        out[name + "_dlogX_mean"] = float(dlogx[mask].mean()) if mask.any() else float("nan")
        out[name + "_dlogX_weighted"] = weighted_mean(dlogx[mask], weight[mask]) if mask.any() else float("nan")
    for bench, _ in S.BENCHES:
        in_bench = np.array([r["bench"] == bench for r in accepted])
        out["per_bench"][bench] = {"accepted": int(in_bench.sum()), "refused": int((refused & in_bench).sum())}
    return out


# ---- outputs -----------------------------------------------------------------------------------------------------------------------

def record_selrules(result):
    """The audit record data/mech/mech_selrules.md."""
    rows = []
    for bench, _ in S.BENCHES:
        r = result[bench]
        for key, label in RULES:
            rows.append([bench, label, f"{100 * r['null'][key]:.1f}"] + [f"{100 * r['power'][str(g)][key]:.1f}" for g in TRUE_GAINS])
    head = ["benchmark", "rule", "false acceptance %"] + [f"accepted at true gain {g} tasks %" for g in TRUE_GAINS]
    gate = result["edge_gate"]
    crit_rows = [[bench, result[bench]["dev_tasks"], result[bench]["critical_tasks"], json.dumps(result[bench]["critical_by_size"])]
                 for bench, _ in S.BENCHES]
    bench_rows = [[bench, v["accepted"], v["refused"]] for bench, v in gate["per_bench"].items()]
    extra = "\n".join([
        "One-sided 0.05 critical net difference of a single-run comparison under no effect (selection.critical_value, fitted prior):", "",
        logger.md_table(["benchmark", "dev tasks", "critical value (tasks)", "critical value by comparison size (tasks)"], crit_rows), "",
        f"Logged accepted edges refused by rule (b): {gate['refused']} of {gate['accepted']} (survive {gate['survive']}).", "",
        logger.md_table(["benchmark", "accepted edges", "refused by (b)"], bench_rows), "",
        logger.md_table(["group", "edges", "mean dlogX (x100)", "n_pair-weighted dlogX (x100)"],
                   [["refused", gate["refused"], f"{gate['refused_dlogX_mean']:.1f}", f"{gate['refused_dlogX_weighted']:.1f}"],
                    ["survive", gate["survive"], f"{gate['survive_dlogX_mean']:.1f}", f"{gate['survive_dlogX_weighted']:.1f}"]])])
    logger.write_record(
        "selrules",
        "How often four acceptance rules accept one candidate against its parent: a candidate identical to the parent (false acceptance) and "
        f"one that truly solves g more dev tasks in expectation (g = {', '.join(map(str, TRUE_GAINS))}; the figure shows g = {PLOT_GAIN}). "
        "Also, how many of the 30 accepted logged dev edges a critical-value gate (rule b: single-run gain at least the one-sided 0.05 critical "
        "value under no effect) would have refused, and the token cost they carried.",
        "Seed skill dev and test executions of Table 1 (native-20261008-https), the beta-binomial task model of mech_selection; logged edges: the 59 "
        "canonical dev edges (edges_table.json, SkillAdam x terminal-bench-2-1 excluded), 30 accepted.",
        f"{S.DRAWS} draws of the task probabilities per benchmark; each rule compares one candidate with its parent on the same draw; rules: "
        + "; ".join(label for _, label in RULES) + ". Seeds fixed per benchmark. Edge gain in tasks = dQ x n_pair / 100 against the critical value of an "
        "n_pair-task comparison.",
        logger.md_table(head, rows),
        ".venv/bin/python -m src.analysis.main.design selrules",
        extra)


def compute_selrules():
    """Simulate every benchmark, gate the logged edges, and write the cache and the audit record."""
    result = {bench: analyse(bench) for bench, _ in S.BENCHES}
    result["edge_gate"] = edge_gate(result)
    result["plot_gain"] = PLOT_GAIN
    dataloader.write_json("selrules", result)
    record_selrules(result)


def panel_bars(ax, data):
    """One axes for the design-principle figure: per acceptance rule, how often it keeps a candidate identical to its parent (light bar) and
    one that truly solves PLOT_GAIN more dev tasks (dark bar); bar = mean over the three benchmarks, whisker = their range."""
    gain = str(data["plot_gain"])
    benches = list(COLORS)
    labels = ["one win\n(today)", "one win at\n5% level", "two wins\nin a row", "3-run\ntotal"]
    width = 0.38
    for i, (key, _) in enumerate(RULES):
        null = [100 * data[b]["null"][key] for b in benches]
        power = [100 * data[b]["power"][gain][key] for b in benches]
        for offset, values, color in ((-width / 2, null, "0.72"), (width / 2, power, "#0072B2")):
            mean = float(np.mean(values))
            ax.bar(i + offset, mean, width=width, color=color, edgecolor="none")
            ax.plot([i + offset] * 2, [min(values), max(values)], color="0.15", lw=0.7)
    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(labels, fontsize=6, linespacing=0.95)
    ax.set_xlim(-0.6, len(RULES) - 0.4)
    # headroom above 100 for the legend
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("edits kept (%)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 100)


# ---------- the Section 5 figure (from implications_fig.py) ----------

TITLES = ["(a) What each rule keeps", "(b) What explains token change", "(c) Seed's calls per turn", "(d) Kept vs. rejected"]


def legend_selrules(ax):
    """Panel (a) legend above its bars: which candidate a bar counts."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="0.72", label="unchanged edit"), Patch(facecolor="#0072B2", label="+2 dev tasks")]
    ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.8, frameon=False, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0.0)


def draw():
    """Load the three caches and draw the 1x4 figure at full text width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.6})
    sel = json.loads((dataloader.CACHE / "selrules.json").read_text())
    dev = json.loads((dataloader.CACHE / "designev.json").read_text())
    bat = json.loads((dataloader.CACHE / "batching.json").read_text())

    figure, axes = plt.subplots(1, 4, figsize=(6.75, 1.72), gridspec_kw=dict(width_ratios=[1.35, 1, 0.9, 0.85]))
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.235, top=0.875, wspace=0.62)

    # (a) the four acceptance rules on the fitted seed task model
    panel_bars(axes[0], sel)
    legend_selrules(axes[0])
    # (b) what predicts an edit's change in input tokens
    panel_predict(axes[1], dev)
    # (c) the tax with the seed's calls per turn restored
    panel(axes[2], bat)
    # (d) rejected vs accepted proposals and the cost-aware gate
    panel_gate(axes[3], dev)

    for ax, title in zip(axes, TITLES):
        ax.set_title(title, fontsize=7, loc="left", pad=3, fontweight="bold")
    fig.save_figure(figure, "implications")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["batching", "designev", "selrules", "draw"])
    step = ap.parse_args().step
    if step == "batching":
        record_batching(compute_batching())
    elif step == "designev":
        record_designev(compute_designev())
    elif step == "selrules":
        compute_selrules()
    else:
        draw()


if __name__ == "__main__":
    main()
