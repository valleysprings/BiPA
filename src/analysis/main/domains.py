"""Per-domain test success of the baselines on tau3-bench and AutomationBench (reviewer point P0-4 / X8).

Role: split Table 1's Q (avg@3) by task domain, to compare our absolute success rates with the published ones (the gap itself is explained in the record).
Domain of a task: tau3 ids are `tau3-<domain>-<n>` (airline, retail, telecom, banking_knowledge); AutomationBench ids are `<domain>-<slug>`
(finance, hr, marketing, operations, sales, support; the benchmark's own domain prefix).
Rows: No skill, Seed skill, GEPA, SkillOpt, SkillAdam on DeepSeek-V4.1-Flash (same runs as Table 1 / main.stats: seed = native-seed run + rep-1/2,
optimizers = native-20261008-https + test-reps, No skill = native-noseed), and GPT-6 Luna No skill / Seed (one test run, native-20261011).
Aggregation: per domain, the success rate among tasks with a verified outcome inside each run, averaged over the runs (the formula of Table 1's Q, restricted to the domain).
The per-domain cells therefore average to Table 1's Q only with task-count weights; the `all` column is Table 1's Q itself.
Outputs: assets/cache/analysis/main/domains.json and paper/assets/data/mech/mech_domains.md.

Bare run computes; the draw subcommand draws from the cache (absorbed from domains_fig.py):
  .venv/bin/python -m src.analysis.main.domains draw   -> fig/mech/mech_domains.{pdf,svg} and fig/mech/mech_domains_col.{pdf,svg}
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.analysis.main import table1 as T
from src.utils.dataloader import BENCHES, EX, SEED_TEXT, current_run, final_skill, vid
from src.utils import fig, logger, dataloader
from src.utils.fig import save_figure
MODEL = "deepseek-flash"
LUNA = "gpt-6-luna"
LUNA_RUN = "native-20261011"
ROWS = (("No skill", "native-noseed"), ("Seed skill", "native-seed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
DOMAINS = {
    "tau3-bench": ("airline", "retail", "telecom", "banking_knowledge"),
    "automationbench": ("finance", "hr", "marketing", "operations", "sales", "support"),
}

EXPLANATION = """## Why our success rates are above the published ones (P0-4 / X8)

Evidence paths: HB = /mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-bench. arXiv numbers are from the papers' PDFs (text extracted locally).

### What the published numbers are

| | tau-Knowledge (2603.04370) | AutomationBench (2604.18934) |
|---|---|---|
| Headline | 25.52 pass^1 = best of all model x retrieval cells (GPT-5.2 high, terminal-use); BM25 column 24.48 (GPT-5.2 high), Claude-4.5-Opus 17.78, Gemini-3-Flash 18.56; gold-document column 32.73-39.69 | best leaderboard score 9.9% (Opus 4.7 max), Gemini 3.1 Pro 9.6%, GPT 5.4 high 7.6%, Sonnet 4.6 5.3%, Haiku 4.5 1.5% |
| Tasks | tau-Banking, all 97 tasks | leaderboard on the PRIVATE held-out set (600+ tasks, 'additional hardening techniques'); the public set has 600 tasks (100 per domain) |
| Metric | pass^k over k<=4 trials; pass^1 = per-task success probability averaged over tasks and trials (the same quantity as our avg@k) | single run; no partial credit; success = end-state assertions all pass |
| Models | GPT-5.2, Claude-4.5-Opus/Sonnet, Gemini-3-Pro/Flash (Mar 2026) | Opus 4.7, Gemini 3.1 Pro, GPT 5.4, Sonnet 4.6, Haiku 4.5 (Apr 2026) |
| Tools | KB retrieval (BM25 / dense / terminal / gold); user simulator GPT-5.2 low | Search (BM25 over API schemas, top-5) + Execute (curl-like) tool calls, max 50 steps |

### What our setup is, per benchmark

AutomationBench (HB/configs/registry/benchmarks.lock.json:46-67, HB/configs/tasks.yaml:2466-2481, HB/assets/benchmarks/automationbench/README.md:1-5,34-39).
- Success rule is the paper's: harbor.py:1406 sets success = rewards['task_completed_correctly'] == 1; in the benchmark wheel (automation_bench 1.0.5, rubric/__init__.py:157-164) task_completed_correctly = 1.0 iff partial_credit == 1.0, i.e. every scored assertion passes (assertions already true in the initial state are excluded from the count, and breaking one counts as a failure; negative assertions are included). No partial credit enters success. Programmatic only, no LLM judge. So scoring is NOT the cause.
- Data is the PUBLIC task set, upstream commit 66cc02f0 (zapier/AutomationBench, adapter harbor PR 2236 c1d9675), 600 scored tasks of which our fixed test split is 120 (20 per domain, tasks.yaml:2476). The paper says leaderboard scores are on the private set, 'purposely harder' (paper Sec. 2; repo README: 'local scores may not match the official leaderboard 1:1'). The repo README (as fetched 2026-10-12) lists public-set scores of 26-50% for the current frontier models (e.g. Opus 5 max 50.3%, Gemini 3.5 Flash high 38.3%, Opus 4.8 max 41.0%, Sonnet 5 max 34.7%); our 36.4% No skill / 45.0% Seed on the 120-task test split is inside that band. The published '<10%' and our 45.0% are therefore different test sets (private vs public) and different model generations (Apr 2026 vs DeepSeek-V4.1-Flash).
- Interface differs from the paper's Search+Execute function calls: the agent is OpenCode and uses the upstream adapter's shell CLI (instruction.md: 'python /workspace/automationbench_cli.py init'; toolset api, as the leaderboard; tool calls in our seed run are bash 3618 / read 668 / grep 40 over 120 episodes). Same API simulator and world state, but a bash CLI instead of two typed tools, with a 30-minute agent budget (task.toml agent timeout 1800 s) and a prompt saying ~50 tool turns, not a hard 50-step cap.

tau3-bench (HB/configs/registry/benchmarks.lock.json:3-27, HB/configs/tasks.yaml:1939-1954, HB/src/harness/harbor.py:1406-1408).
- Text half-duplex mode (adapter evaluate.py:161,170: CommunicationMode.HALF_DUPLEX; adapter README:8,23), tau2-bench v1.0.1 (commit fc0055dc), all four domains, banking_knowledge with the BM25 retrieval variant (config.json retrieval_variant 'bm25'; evaluate.py:90; the agent uses the KB_search MCP tool, 384 calls in the seed run). Success = rewards['reward'] == 1 (harbor.py:1408) of tau2's evaluate_simulation(EvaluationType.ALL), i.e. all reward_basis components of the task (banking: DB 87 tasks, ACTION 9, DB+NL_ASSERTION 1; retail DB+NL_ASSERTION; airline DB+COMMUNICATE; telecom ENV_ASSERTION). A run that ends without agent_stop or user_stop scores 0 (evaluate.py:250). This is the official binary reward, the same as tau-Knowledge's reward on the final database state.
- Our 'tau3 = 76.4' is a mixture: airline 86.7, retail 84.1, telecom 98.6 (the three non-knowledge domains, which the tau-Knowledge paper does not evaluate) and banking_knowledge only 35.1. It is the banking column that corresponds to the published 25.5%.
- Banking test slice: 19 of the 97 tasks (task-084..102, the fixed split of tasks.yaml:1949-1954), SE over tasks 8.6 points; the 75-task tau3 test panel has 10 airline / 23 retail / 23 telecom / 19 banking tasks. pass^1 in the paper is a mean over 4 trials, our avg@3 is a mean over 3 runs: same estimand.
- Banking No skill is 24.6, i.e. in the published range (BM25 cells 9.5-24.5 for GPT-5.2, 16.8-17.8 for Claude-4.5, 18.6 for Gemini-3-Flash; terminal 11.6-25.5). The Seed skill adds +10.5 points (35.1), a level the paper reaches only with gold documents in context (32.2 average, 39.7 best). With 19 tasks the SE is 8-9 points, so 24.6 vs 35.1 is about one SE apart per cell; the same direction holds on GEPA 36.8, SkillOpt 38.6, SkillAdam 38.6.
- Possible additional differences, not isolated here: the paper predates the tau2-bench v1.0.1 banking grading update (repo README says scores before v1.0.1 are not comparable); the simulated user/NL judge is the cell's own model (TAU2_USER_MODEL = openai/deepseek-flash in HB/.env:29, and configs/models.yaml:21-30 for Luna) versus GPT-5.2 low in the paper; DeepSeek-V4.1-Flash is a later model than the Mar 2026 ones; the harness (OpenCode, MCP tools) differs from the paper's agent loop (which truncates old retrieval outputs, in 1-3% of its runs).

### Summary of the gap

| | published | ours | difference that explains it |
|---|---|---|---|
| tau-Banking pass^1 | <= 25.5 (frontier, Mar 2026) | No skill 24.6, Seed 35.1 (19 tasks, SE 8.6) | same scoring and BM25 retrieval; ours equals the published range without a skill, +10.5 points with the generic seed skill; newer model; self-simulated user; v1.0.1 grading; 19-task slice |
| tau3 overall | none published as one number | 76.4 | dominated by airline/retail/telecom (86.7/84.1/98.6), which are easy for current models; banking is 25% of the panel |
| AutomationBench | < 10% (private set, Apr 2026 frontier) | No skill 36.4, Seed 45.0 (120 tasks) | same success rule (all assertions); public vs private set; later model; bash CLI instead of Search+Execute; public-set scores in the repo README are 26-50% |

Nothing in the scoring path (harbor.py:1403-1411) credits partial success or skips failed episodes (HARNESS_STRICT_OUTCOMES=1: an episode without a verified outcome is unknown and enters neither numerator nor denominator; Luna banking has such cells, e.g. 5.6 = 1 of 18 known).
"""


def domain_of(bench, task_id):
    """Domain of a task id: the middle token of tau3-<domain>-<n>, the leading token of an AutomationBench id."""
    if bench == "tau3-bench":
        return task_id.split("-")[1]
    return task_id.split("-")[0]


def cell_runs(row, method, bench):
    """{repetition: evaluation.json} of the DeepSeek test runs of one row, exactly the files Table 1 averages."""
    seed_run = current_run("native-seed", MODEL, bench)
    if method == "native-seed":
        return T.rep_files([seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"], vid(SEED_TEXT))
    run = current_run(method, MODEL, bench)
    text = final_skill(method, run)
    return T.rep_files([run, T.EX / MODEL / "test-reps" / f"{bench}__{method}__{run.name}"], vid(text))


def luna_runs(arm, bench):
    """The single Luna test run of an arm (the file main/transfer.py also reads)."""
    pattern = str(EX / LUNA / arm / bench / LUNA_RUN / "results/evaluations/test/*/evaluation.json")
    return {0: Path(sorted(glob.glob(pattern))[0])}


def by_domain(bench, files):
    """{domain or 'all': {'q': percent, 'tasks': number of tasks, 'known': verified outcomes per run, 'run_q': per-run percent}} over the runs {rep: file}."""
    per_run = []
    for _, f in sorted(files.items()):
        rows = T.rows_of(f)
        counts = {}
        for (task, _), row in rows.items():
            if row["success"] is None:
                continue
            for key in (domain_of(bench, task), "all"):
                wins, known = counts.get(key, (0, 0))
                counts[key] = (wins + (row["success"] is True), known + 1)
        total = {}
        for (task, _) in rows:
            for key in (domain_of(bench, task), "all"):
                total[key] = total.get(key, 0) + 1
        per_run.append((counts, total))
    # per-task mean success over the runs (verified outcomes only), for the standard error over tasks
    per_task = {}
    for _, f in sorted(files.items()):
        for (task, _), row in T.rows_of(f).items():
            if row["success"] is not None:
                per_task.setdefault(task, []).append(float(row["success"] is True))
    out = {}
    for key in list(DOMAINS[bench]) + ["all"]:
        means = [np.mean(v) for t, v in per_task.items() if key == "all" or domain_of(bench, t) == key]
        se = float(100 * np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None
        rates = [c[key][0] / c[key][1] for c, _ in per_run if key in c]
        out[key] = {"q": float(100 * np.mean(rates)),
                    "se_tasks": se,
                    "run_q": [float(100 * r) for r in rates],
                    "tasks": per_run[0][1].get(key, 0),
                    "known": [c[key][1] if key in c else 0 for c, _ in per_run]}
    return out


def compute():
    """{bench: {row: by_domain}} for DeepSeek rows and Luna rows (keys 'Luna No skill', 'Luna Seed')."""
    res = {}
    for bench in ("tau3-bench", "automationbench"):
        res[bench] = {}
        for name, method in ROWS:
            res[bench][name] = by_domain(bench, cell_runs(name, method, bench))
        for name, arm in (("Luna No skill", "native-noseed"), ("Luna Seed", "native-seed")):
            res[bench][name] = by_domain(bench, luna_runs(arm, bench))
    return res


def table(res, bench):
    """Markdown table of one benchmark: rows = methods, columns = domains (tasks) and all."""
    doms = list(DOMAINS[bench]) + ["all"]
    first = next(iter(res[bench].values()))
    head = ["Row"] + [f"{d} (n={first[d]['tasks']})" for d in doms]
    rows = []
    for name, cell in res[bench].items():
        rows.append([name] + [f"{cell[d]['q']:.1f}" for d in doms])
    return logger.md_table(head, rows)


def se_table(res, bench):
    """Markdown table of the standard error over tasks (SD of per-task mean success / sqrt(tasks)) for the DeepSeek rows."""
    doms = list(DOMAINS[bench]) + ["all"]
    rows = [[name] + [f"{cell[d]['se_tasks']:.1f}" for d in doms] for name, cell in res[bench].items()]
    return logger.md_table(["Row"] + doms, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figure", nargs="?", choices=["draw"], help="draw mech_domains and mech_domains_col from domains.json instead of computing")
    args = ap.parse_args()
    if args.figure == "draw":
        draw()
        draw_col()
        return
    res = compute()
    dataloader.write_json("domains", res)
    text = "\n\n".join(f"### {b}\n\n{table(res, b)}\n\nStandard error over tasks (points):\n\n{se_table(res, b)}" for b in res)
    extra = EXPLANATION
    logger.write_record(
        "domains",
        "Test success (avg@3, percent) per task domain of tau3-bench and AutomationBench for No skill, Seed, GEPA, SkillOpt, SkillAdam (DeepSeek-V4.1-Flash, OpenCode) and for GPT-6 Luna No skill / Seed (one run), and why our absolute success is above the published numbers (reviewer P0-4 / X8).",
        "Same runs as Table 1 (main.stats): seed = native-seed native-20261003-trace-final + rep-1 + rep-2; optimizers = native-20261008-https + test-reps; No skill = latest native-noseed; Luna = native-20261011 test run of native-noseed / native-seed.",
        "Per domain: success rate among tasks with a verified outcome inside each run, averaged over the runs (Table 1's Q restricted to the domain); the all column is Table 1's Q. Standard error = SD over tasks of the per-task mean success / sqrt(tasks), a lower bound of the uncertainty (no run noise).",
        text, ".venv/bin/python -m src.analysis.main.domains", extra)
    print(text)


# ---- figures (absorbed from domains_fig.py) ----------------------------------------------------------------------------------------

FIG_ROWS = ["No skill", "Seed skill", "GEPA", "SkillOpt", "SkillAdam"]
FIG_LABEL = {"Seed skill": "Seed"}
FIG_STYLE = {"No skill": dict(m="o", fc="white", ec="0.5"), "Seed skill": dict(m="o", fc="k", ec="k"),
             "GEPA": dict(m="^", fc="#009E73", ec="#009E73"), "SkillOpt": dict(m="o", fc="#0072B2", ec="#0072B2"),
             "SkillAdam": dict(m="s", fc="#D55E00", ec="#D55E00")}
FIG_PANELS = [("tau3-bench", r"$\tau^3$-bench", ["airline", "retail", "telecom", "banking_knowledge"], ["airline", "retail", "telecom", "banking"]),
              ("automationbench", "AutoBench", ["finance", "hr", "marketing", "operations", "sales", "support"], ["finance", "HR", "marketing", "ops", "sales", "support"])]
PUBLISHED = 25.5


def draw():
    """mech_domains: test avg@3 success per task domain for No skill, Seed, GEPA, SkillOpt, SkillAdam (tau3-bench and AutoBench panels).
    One marker per row and domain, side by side inside each domain slot; the grey tick on banking_knowledge is the published best pass^1 of
    tau-Knowledge (25.52).  Reads assets/cache/analysis/main/domains.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((dataloader.CACHE / "domains.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), gridspec_kw=dict(width_ratios=[4, 6], wspace=0.08), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.17, top=0.80)
    step = 0.15
    offs = [(i - 2) * step for i in range(5)]
    numbers = []
    for ax, (bk, disp, keys, names) in zip(axes, FIG_PANELS):
        for j, k in enumerate(keys):
            for r, o in zip(FIG_ROWS, offs):
                q = d[bk][r][k]["q"]
                s = FIG_STYLE[r]
                ax.scatter([j + o], [q], s=14, marker=s["m"], facecolor=s["fc"], edgecolor=s["ec"], lw=0.9, zorder=3)
                numbers.append((disp, k, r, q))
            if k == "banking_knowledge":
                ax.plot([j - 0.42, j + 0.42], [PUBLISHED] * 2, color="0.3", lw=1.3, zorder=2, solid_capstyle="butt")
                ax.text(j, PUBLISHED - 3, "published best", ha="center", va="top", fontsize=7, color="0.25")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(0, 105)
        ax.set_xlabel(disp)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("test success (%)")
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker=FIG_STYLE[r]["m"], mfc=FIG_STYLE[r]["fc"], mec=FIG_STYLE[r]["ec"], ms=4.5, mew=0.9) for r in FIG_ROWS]
    h.append(plt.Line2D([], [], color="0.3", lw=1.3))
    fig.legend(h, [FIG_LABEL.get(r, r) for r in FIG_ROWS] + [r"best published (25.5%)"], loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=6,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    save_figure(fig, "domains")
    heads = ["bench", "domain"] + [FIG_LABEL.get(r, r) for r in FIG_ROWS]
    tab = []
    for bk, disp, keys, _ in FIG_PANELS:
        for k in keys:
            tab.append([bk, k] + [f"{d[bk][r][k]['q']:.1f}" for r in FIG_ROWS])
    logger.write_record("domains_fig", "Test avg@3 success (%) per domain for No skill (hollow grey), Seed (black), GEPA, SkillOpt, SkillAdam; grey tick = best published tau-Knowledge pass^1 (25.52) on the banking domain.",
                   "Same runs as Table 1 (see mech_domains.md tables from main.domains); DeepSeek-V4.1-Flash, OpenCode.",
                   "domains.json q: success among verified tasks per run, averaged over runs, per domain.",
                   logger.md_table(heads, tab), ".venv/bin/python -m src.analysis.main.domains draw")
    return numbers


def draw_col():
    """Single-column variant (3.25 x 2.5 in): tau3-bench panel above AutoBench panel, shared legend on top in two rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((dataloader.CACHE / "domains.json").read_text())
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.3))
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.075, top=0.835)
    offs = [(i - 2) * 0.15 for i in range(5)]
    for ax, (bk, disp, keys, names) in zip(axes, FIG_PANELS):
        for j, k in enumerate(keys):
            for r, o in zip(FIG_ROWS, offs):
                s = FIG_STYLE[r]
                ax.scatter([j + o], [d[bk][r][k]["q"]], s=12, marker=s["m"], facecolor=s["fc"], edgecolor=s["ec"], lw=0.8, zorder=3)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 50, 100])
        ax.text(0.01, 0.97, disp, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=2)
    fig.text(0.02, 0.455, "test success (%)", rotation=90, ha="center", va="center", fontsize=7)
    h = [plt.Line2D([], [], ls="", marker=FIG_STYLE[r]["m"], mfc=FIG_STYLE[r]["fc"], mec=FIG_STYLE[r]["ec"], ms=4.5, mew=0.9) for r in FIG_ROWS]
    # the published tau-Banking score is a setup check, not part of this figure's comparison; it stays in Appendix A only
    fig.legend(h, [FIG_LABEL.get(r, r) for r in FIG_ROWS], loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=3,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.0, handletextpad=0.3, labelspacing=0.25)
    save_figure(fig, "domains_col")


if __name__ == "__main__":
    main()
