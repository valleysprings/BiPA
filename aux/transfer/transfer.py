"""Transfer: a learned skill run unchanged under the other model or on the other benchmark scores below the seed, as it does
in its own setting, and most of its extra input tokens travel with it (records mech_transfer and mech_ood, figure
mech_transfer3).

Runs. GPT-6-Luna on tau3-bench: the main table's seed, no-skill and Luna-learned cells, then the three tau3 skills learned
under DeepSeek-V4.1-Flash and a rerun of the seed (aux_runs transfer_model), each with three test runs except the seed rerun.
DeepSeek: each TB2 skill run once on AutoBench and each AutoBench skill once on TB2, beside a rerun of the seed made on the
same days (aux_runs transfer_bench). A transfer run is matched to the DeepSeek skill whose version it tested.

An episode counts what it recorded: a request the provider rejected has no count and is dropped, and the episode keeps its
other turns (common.luna_row, the main table's reader of GPT-6-Luna, used here for every run outside that table too).

Success difference to the seed, in points: per task, the share of the skill's runs that solved it minus the share of the
seed's three test runs (the main table's Seed row), averaged over the tasks verified in every run on both sides; a transfer
that ran once is compared with the seed's three runs. The pooled tau3 numbers first average each task over the three transferred
skills and over the three skills learned under GPT-6-Luna; their intervals are percentile intervals over N_BOOT resamples
of the tasks, one draw set for the three numbers.

Input-token ratio rho, and the same ratio of turns and of calls: the main table's geometric mean, over the tasks both sides
have with calls and input tokens, of the per-task means over the runs with a verified outcome. DeepSeek's skills in their own
setting take the main table's rho. A skill moved to the other benchmark is compared with the seed rerun of the same days
rather than with the seed's three runs, against whose mean a single run reads low.

The transfer runs (they call the model; provenance):
    python -m src.run transfer METHOD tau3 deepseek-flash --to gpt-6-luna --execute          METHOD: skillopt gepa skilladam seed
    python -m src.run repeat METHOD tau3 gpt-6-luna --tag ood-from-deepseek-flash --execute   METHOD: skillopt gepa skilladam
    python -m src.run repeat seed tau3 gpt-6-luna --tag ood-seed --execute
    python -m src.run transfer METHOD SOURCE deepseek-flash --to TARGET --execute            SOURCE, TARGET: TB2 and AutoBench"""
import json

import numpy as np

from src.analysis import settings as S
from src.analysis.common import accumulate, geo, luna_row
from src.utils import fig
from src.utils.logger import md_table, record

TAU, AUTOB, TB2 = "tau3-bench", "automationbench", "terminal-bench-2-1"
SEED, NOSKILL, OPT = "native-seed", "native-noseed", S.OPTIMIZERS
MORE = 1.05  # a transferred skill "carries the extra input tokens" above this ratio to the seed


def recorded(files):
    """{repetition: {task: row}} of test files, each episode counted as recorded (common.luna_row)."""
    return {k: {t: luna_row(e) for t, e in eps.items()} for k, eps in S.episodes(files).items()}


def tested(run):
    """(source benchmark, rows) of a run that tested the seed (source None) or a DeepSeek skill outside its setting."""
    method = run.parts[-3]
    if method == SEED:
        return None, recorded(S.run_files(run, S.SEED_TEXT))
    for bench in S.BENCHES:
        files = S.run_files(run, S.skill(S.DEEPSEEK, bench, method))
        if files:
            return bench, recorded(files)
    raise LookupError(f"{run}: no test evaluation of a DeepSeek {method} skill")


def solved(reps):
    """{task: share of the runs that solved it}, over the tasks verified in every run."""
    tasks = set.intersection(*({t for t, r in rows.items() if r["success"] is not None} for rows in reps.values()))
    return {t: np.mean([rows[t]["success"] for rows in reps.values()]) for t in tasks}


def difference(own, ref):
    """(points, tasks): the mean per-task difference of two solved() maps over their common tasks."""
    d = [own[t] - ref[t] for t in own if t in ref]
    return 100 * np.mean(d), len(d)


def success(reps):
    """Test success (%) of each run over its verified outcomes."""
    return [100 * np.mean([r["success"] for r in rows.values() if r["success"] is not None]) for _, rows in sorted(reps.items())]


def ratios(reps, ref):
    """(tasks, {X, R, N}): the main table's rho of input tokens, turns and calls against a reference."""
    means = lambda rr: accumulate([{(t, k): r for t, r in rows.items()} for k, rows in rr.items()])
    n, r = geo(means(reps), means(ref), turns=True)
    return n, {"X": r["rho_trj"], "R": r["rho_turns"], "N": r["rho_inv"]}


def compute():
    """Every row of the records and the figure: GPT-6-Luna on tau3, the cross-benchmark runs and the success differences."""
    luna = {("own", m): recorded(S.cell_files(S.LUNA, TAU, m)) for m in S.ARMS}
    runs = S.extra("transfer_model", S.LUNA, TAU)
    for run in runs:
        source, rows = tested(run)
        luna[("transferred" if source else "rerun", run.parts[-3])] = rows
    seed = luna[("own", SEED)]
    luna = {k: dict(Q=success(v), rho=ratios(v, seed), solved=solved(v)) for k, v in luna.items()}
    seed = luna[("own", SEED)]["solved"]
    dq = {}  # (group, setting): {optimizer: (points, tasks)}
    for bench in (TB2, AUTOB, TAU):
        ref = solved(recorded(S.cell_files(S.DEEPSEEK, bench, SEED)))
        dq["own", (S.DEEPSEEK, bench)] = {o: difference(solved(recorded(S.cell_files(S.DEEPSEEK, bench, o))), ref) for o in OPT}
    dq["own", (S.LUNA, TAU)] = {o: difference(luna[("own", o)]["solved"], seed) for o in OPT}
    bench_rows = []  # (source, target, method, Q, rho)
    for target in (AUTOB, TB2):
        found = [(run, *tested(run)) for run in S.extra("transfer_bench", S.DEEPSEEK, target)]
        rerun = next(rows for _, source, rows in found if source is None)
        ref = solved(recorded(S.cell_files(S.DEEPSEEK, target, SEED)))
        bench_rows.append((None, target, SEED, success(rerun), None))
        for run, source, rows in found:
            if source:
                method = run.parts[-3]
                bench_rows.append((source, target, method, success(rows), ratios(rows, rerun)))
                dq.setdefault(("transferred", (source, target)), {})[method] = difference(solved(rows), ref)
        runs += [run for run, *_ in found]
    dq["transferred", (S.DEEPSEEK, S.LUNA)] = {o: difference(luna[("transferred", o)]["solved"], seed) for o in OPT}
    # the three transferred skills against the three learned under GPT-6-Luna, each task averaged over the three
    tasks = sorted(set(seed).intersection(*(luna[(g, o)]["solved"] for g in ("own", "transferred") for o in OPT)))
    mean = lambda g: np.array([np.mean([luna[(g, o)]["solved"][t] for o in OPT]) for t in tasks])
    base = np.array([seed[t] for t in tasks])
    pooled = {"transferred": mean("transferred") - base, "learned under GPT-6-Luna": mean("own") - base}
    pooled["transferred minus learned under GPT-6-Luna"] = pooled["transferred"] - pooled["learned under GPT-6-Luna"]
    idx = S.boot_index(len(tasks), "transfer", "pooled tau3")
    pooled = {k: (100 * d.mean(), *(100 * np.array(S.interval(d[idx].mean(1))))) for k, d in pooled.items()}
    return dict(luna=luna, dq=dq, bench=bench_rows, pooled=pooled, runs=runs)


def setting(key):
    group, (a, b) = key
    if group == "own":
        return f"{S.MODEL_SHORT[a]}, {fig.BENCH_LABEL[b]}"
    if b == S.LUNA:
        return f"DeepSeek → Luna, {fig.BENCH_LABEL[TAU]}"
    return f"DeepSeek, {fig.BENCH_LABEL[a]} → {fig.BENCH_LABEL[b]}"


def token_ratios(res):
    """{(group, setting): {optimizer: rho}} as drawn: the main table's for DeepSeek in its own setting."""
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    rho = {("own", (S.DEEPSEEK, b)): {o: table1[f"{S.DEEPSEEK}|{b}"][S.ARM_NAME[o]]["rho_trj"] for o in OPT}
           for b in (TB2, AUTOB, TAU)}
    rho["own", (S.LUNA, TAU)] = {o: res["luna"][("own", o)]["rho"][1]["X"] for o in OPT}
    for source, target, method, _, r in res["bench"]:
        if source:
            rho.setdefault(("transferred", (source, target)), {})[method] = r[1]["X"]
    rho["transferred", (S.DEEPSEEK, S.LUNA)] = {o: res["luna"][("transferred", o)]["rho"][1]["X"] for o in OPT}
    return rho


def counts(res, rho):
    keys = list(res["dq"])
    below = lambda group: [res["dq"][k][o][0] < 0 for k in keys if k[0] == group for o in OPT]
    more = [rho[k][o] > MORE for k in keys if k[0] == "transferred" for o in OPT]
    return {g: (sum(below(g)), len(below(g))) for g in ("own", "transferred")} | {"more": (sum(more), len(more))}


def draw(res, rho, n):
    """(a) success difference and (b) input-token ratio to the seed, one row per setting, one marker per optimizer."""
    plt = fig.pyplot(8, tick=7.5)
    from matplotlib.transforms import blended_transform_factory as blend
    keys = list(res["dq"])
    heads = [("Run where it was learned", [k for k in keys if k[0] == "own"]),
             ("Transferred (learned → run)", [k for k in keys if k[0] != "own"])]
    y, rows, ys = 0.0, [], []
    for b, (head, items) in enumerate(heads):
        y -= 0.55 if b else 0
        ys.append((y, head))
        y -= 1.0
        for k in items:
            rows.append((y, k))
            y -= 1.0
    figure, (axq, axr) = plt.subplots(1, 2, figsize=(fig.TEXT_W, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    figure.subplots_adjust(left=0.29, right=0.985, bottom=0.125, top=0.76)
    sep = (rows[len(heads[0][1]) - 1][0] + ys[1][0]) / 2
    dodge = dict(zip(OPT, (0.22, 0.0, -0.22)))
    for ax, ref, value in ((axq, 0.0, lambda k, o: res["dq"][k][o][0]), (axr, 1.0, lambda k, o: rho[k][o])):
        for yy, k in rows:
            ax.axhline(yy, color=fig.GRID, lw=0.5, zorder=0)
            for o in OPT:
                ax.plot(value(k, o), yy + dodge[o], marker=fig.MARKER[o], ms=4.6, color=fig.ARM[o], markeredgecolor="white",
                        markeredgewidth=0.4, lw=0, zorder=3)
        ax.axvline(ref, color=fig.INK, lw=0.7, ls="--", zorder=1)
        ax.text(ref, 1.0, "seed", transform=blend(ax.transData, ax.transAxes), ha="center", va="bottom", fontsize=fig.SMALL,
                color=fig.MUTED)
        ax.axhline(sep, color=fig.MID, lw=0.5, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", pad=1.5)
    axq.set_ylim(y + 0.45, ys[0][0] + 0.45)
    axq.set_yticks([yy for yy, _ in rows])
    once = lambda k: k[0] == "transferred" and k[1][1] != S.LUNA  # moved to the other benchmark: one test run
    axq.set_yticklabels([setting(k) + (" (1 run)" if once(k) else "") for _, k in rows], fontsize=8)
    for yy, head in ys:
        axq.text(0.012, yy, head, transform=blend(figure.transFigure, axq.transData), ha="left",
                 va="center", fontweight="bold")
    axq.set_xlim(-12.5, 4.5)
    axq.set_xticks([-10, -5, 0])
    axq.set_xticklabels(["$-$10", "$-$5", "0"])
    axq.set_xlabel("test success minus the seed's (points)", labelpad=2)
    axr.set_xlim(0.94, 1.56)
    axr.set_xticks([1.0, 1.2, 1.4])
    axr.set_xlabel(r"input-token ratio $\rho$ to the seed", labelpad=2)
    axq.set_title("(a) {} of {} transferred skills score\nbelow the seed".format(*n["transferred"]), loc="left", pad=11,
                  linespacing=1.15)
    axr.set_title("(b) Most of the extra input tokens travel\nwith the skill", loc="left", pad=11, linespacing=1.15)
    handles = [plt.Line2D([], [], marker=fig.MARKER[o], color=fig.ARM[o], markeredgecolor="white", markeredgewidth=0.4, lw=0,
                          ms=5, label=S.ARM_NAME[o]) for o in OPT]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.625, 1.01), ncol=3, frameon=False, handletextpad=0.2,
                  columnspacing=1.4)
    fig.save_figure(figure, "transfer3")


def main():
    res = compute()
    rho = token_ratios(res)
    n = counts(res, rho)
    draw(res, rho, n)
    f1, f3 = lambda v: f"{v:.1f}", lambda v: f"{v:.3f}"
    names = {("own", SEED): "Seed", ("own", NOSKILL): "No skill", ("rerun", SEED): "Seed, rerun (one run)"}
    names |= {(g, o): f"{S.ARM_NAME[o]}, learned under {'GPT-6-Luna' if g == 'own' else 'DeepSeek'}"
              for g in ("own", "transferred") for o in OPT}
    luna = [[names[k], f1(np.mean(v["Q"])), " / ".join(map(f1, v["Q"])), *(f3(v["rho"][1][x]) for x in "XRN"), v["rho"][0]]
            for k, v in sorted(res["luna"].items(), key=lambda kv: list(names).index(kv[0]))]
    dq = [[setting(k), *(f"{v[o][0]:+.1f}" for o in OPT), " / ".join(sorted({str(v[o][1]) for o in OPT}))]
          for k, v in res["dq"].items()]
    pooled = [[k, f"{v[0]:+.1f}", f"[{v[1]:+.1f}, {v[2]:+.1f}]"] for k, v in res["pooled"].items()]
    numbers = "\n".join([
        f"GPT-6-Luna on {fig.BENCH_LABEL[TAU]}: test success and ratios to the seed (three test runs per skill).", "",
        md_table(["skill", "test success (%)", "per run", "rho (input tokens)", "rho (turns)", "rho (calls)",
                  "tasks in the ratios"], luna), "",
        "Test success minus the seed's (points) and the tasks it is averaged over.", "",
        md_table(["setting", *map(S.ARM_NAME.get, OPT), "tasks"], dq), "",
        f"GPT-6-Luna on {fig.BENCH_LABEL[TAU]}, each task averaged over the three transferred skills and over the three "
        "learned under GPT-6-Luna.", "",
        md_table(["skills", "difference to the seed (points)", "95% interval"], pooled), "",
        "Below the seed: {} of {} skills in their own setting, {} of {} transferred. Transferred skills reading more than "
        "{} times the seed's input tokens: {} of {}.".format(*n["own"], *n["transferred"], MORE, *n["more"])])
    cells = [(S.LUNA, TAU, m) for m in S.ARMS] + [(S.DEEPSEEK, b, m) for b in (TB2, AUTOB, TAU) for m in (SEED, *OPT)]
    record(__spec__.name, "transfer", numbers, S.runs_text(cells, res["runs"]))
    ood = [[f"{fig.BENCH_LABEL[s]} → {fig.BENCH_LABEL[t]}", S.ARM_NAME[m], f1(q[0]), *(f3(r[1][x]) for x in "XRN"), r[0]]
           for s, t, m, q, r in res["bench"] if s]
    record(__spec__.name, "ood", md_table(["source → target", "skill", "test success (%)", "rho (input tokens)", "rho (turns)",
                                           "rho (calls)", "tasks in the ratios"], ood),
           S.runs_text([(S.DEEPSEEK, b, m) for b in (TB2, AUTOB) for m in OPT], S.extra("transfer_bench", S.DEEPSEEK, AUTOB)
                       + S.extra("transfer_bench", S.DEEPSEEK, TB2)),
           what="DeepSeek-V4.1-Flash's TB2 and AutoBench skills, each run once on the other benchmark: test success and the ratios "
                "to the seed rerun of the same days.")


if __name__ == "__main__":
    main()
