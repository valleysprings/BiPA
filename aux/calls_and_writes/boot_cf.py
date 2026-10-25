"""Bootstrap intervals for two accounting counterfactuals of the mechanism paper (DeepSeek-V4.1-Flash, recorded test episodes, no re-runs).

(1) Re-billing at the seed's calls per turn (design.py batching; mech_batching.md): AutoBench mean log rho 0.166 -> 0.083.  Task-cluster bootstrap over
    AutoBench tasks: each draw resamples tasks, re-fits b_R (within-task regression of log X on log N, log R with skill dummies, design.fe_fit) on the drawn
    tasks' episodes, and recomputes each skill's log rho and removed = mean(-b_R dlog m) on the drawn tasks of its usable set.
(2) Compaction to the last turn of history (compaction.py, window 1; mech_compaction.md): rho per skill (nine cells) 1.05 to 1.36.  Per cell, bootstrap over
    tasks of the window-1 rho and of log(rho_w1 / rho_turns).
Task sets and per-task quantities are rebuilt with the modules' own code; the point estimates are checked against batching.json / compaction.json.
Run:  python -m aux.calls_and_writes.boot_cf   -> OUT/calls_and_writes/boot_cf.json
"""
import json
import math

import numpy as np

from src.analysis import table1 as T
from src.analysis import attribution as A
from aux.edit_gate_and_design import design as D
from aux.cost_appendix import compaction as C
from src import layout
from src.utils import dataloader

OUT = layout.OUT / "calls_and_writes"
B = 2000
rng = np.random.default_rng(20261020)


def load_cell(method, bench):
    own_files, seed_files = A.run_files(method, bench)
    own_eps, own_rows = A.load_runs(own_files)
    seed_eps, seed_rows = A.load_runs(seed_files)
    reps = sorted(own_files)
    _, own_mean = T.mean_rows(own_files)
    _, seed_mean = T.mean_rows(seed_files)
    common = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
    return dict(own_eps=own_eps, seed_eps=seed_eps, own_rows=own_rows, seed_rows=seed_rows, reps=reps, common=common)


def batching_cell(c):
    """design.bench_numbers for one cell, keeping task ids: usable tasks, per-task dlog_x and dlog_m, and the episode records."""
    reps, own_eps, seed_eps = c["reps"], c["own_eps"], c["seed_eps"]
    usable, dx, dm = [], [], []
    for t in c["common"]:
        rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                      for rows in (c["own_rows"], c["seed_rows"]) for r in reps)
        if not rows_ok or not all(t in eps[r] and eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
            continue
        a = A.mean_metrics([seed_eps[r][t][1] for r in reps])
        b = A.mean_metrics([own_eps[r][t][1] for r in reps])
        if a is None or b is None:
            continue
        usable.append(t)
        dx.append(math.log(b["X"] / a["X"]))
        dm.append(math.log((b["N"] / b["R"]) / (a["N"] / a["R"])))
    return usable, np.array(dx), np.array(dm)


def records_of(eps_by_rep, label):
    out = []
    for r, eps in eps_by_rep.items():
        for t, (_, m) in eps.items():
            if m:
                out.append(dict(task=t, skill=label, N=m["N"], R=m["R"], X=m["X"]))
    return out


def fit_bR(records, tasks_draw):
    """b_R of design.regressions 'NR' on the episodes of the drawn tasks (a task drawn k times enters as k distinct clusters)."""
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    task, skill, ln, lr, lx = [], [], [], [], []
    for k, t in enumerate(tasks_draw):
        for r in by_task.get(t, []):
            task.append(f"{t}#{k}"); skill.append(r["skill"]); ln.append(math.log(r["N"])); lr.append(math.log(r["R"])); lx.append(math.log(r["X"]))
    beta, _, _ = D.fe_fit(np.array(task), np.array(skill), np.column_stack([ln, lr]), np.array(lx))
    return float(beta[1])


def main():
    batching_ref = {(c["method"], c["bench"]): c for c in json.loads((dataloader.CACHE / "batching.json").read_text())["cells"]}
    bfit = json.loads((dataloader.CACHE / "batching.json").read_text())["fits"]["automationbench"]["NR"]["coef"][1]
    comp_ref = {(c["method"], c["bench"]): c for c in json.loads((dataloader.CACHE / "compaction.json").read_text())["cells"]}
    out = dict(batching={}, compaction=[])
    auto_cells, auto_records, seed_added = {}, [], False
    for bench, short in A.BENCH:
        for disp, method in A.OPT:
            c = load_cell(method, bench)
            # ---- compaction, window 1 (compaction.cell_numbers task set: common tasks with a graph episode in all six runs)
            reps = c["reps"]
            tasks = [t for t in c["common"] if all(t in eps[r] for eps in (c["own_eps"], c["seed_eps"]) for r in reps)]
            lw1 = np.array([math.log(np.mean([C.window_input(c["own_eps"][r][t][0], 1) for r in reps]) / np.mean([C.window_input(c["seed_eps"][r][t][0], 1) for r in reps])) for t in tasks])
            linf = np.array([math.log(np.mean([C.window_input(c["own_eps"][r][t][0], None) for r in reps]) / np.mean([C.window_input(c["seed_eps"][r][t][0], None) for r in reps])) for t in tasks])
            lturn = np.array([math.log(np.mean([c["own_eps"][r][t][0].R for r in reps]) / np.mean([c["seed_eps"][r][t][0].R for r in reps])) for t in tasks])
            ref = comp_ref[(disp, bench)]
            w1_ref = [x for x in ref["results"] if x["policy"] == "window" and x["param"] == "1"][0]["rho"]
            assert len(tasks) == ref["n"] and abs(math.exp(lw1.mean()) - w1_ref) < 1e-9 and abs(math.exp(lturn.mean()) - ref["rho_turns"]) < 1e-9, (disp, bench)
            draws = []
            for _ in range(B):
                i = rng.integers(0, len(tasks), len(tasks))
                draws.append((math.exp(lw1[i].mean()), lw1[i].mean() - lturn[i].mean(), math.exp(linf[i].mean()), linf[i].mean() - lw1[i].mean()))
            d = np.array(draws)
            q = lambda col: [float(np.percentile(d[:, col], 2.5)), float(np.percentile(d[:, col], 97.5))]
            out["compaction"].append(dict(method=disp, bench=short, n=len(tasks), rho_w1=math.exp(lw1.mean()), rho_w1_ci=q(0), rho_turns=math.exp(lturn.mean()),
                                          log_gap_w1_turns=float(lw1.mean() - lturn.mean()), log_gap_ci=q(1), rho_inf=math.exp(linf.mean()), rho_inf_ci=q(2),
                                          log_drop_inf_to_w1=float(linf.mean() - lw1.mean()), log_drop_ci=q(3)))
            print("compaction", disp, short, round(math.exp(lw1.mean()), 3), q(0), flush=True)
            # ---- batching, AutoBench only
            if bench == "automationbench":
                usable, dx, dm = batching_cell(c)
                r = batching_ref[(disp, bench)]
                assert len(usable) == r["n"] and np.allclose(dx, r["dlog_x"]) and np.allclose(dm, r["d_log_m"]), disp
                auto_cells[disp] = (usable, dx, dm)
                auto_records += records_of(c["own_eps"], disp)
                if not seed_added:
                    auto_records += records_of(c["seed_eps"], "seed"); seed_added = True
            del c
    # ---- batching bootstrap
    all_tasks = sorted({r["task"] for r in auto_records})
    b0 = fit_bR(auto_records, all_tasks)
    assert abs(b0 - bfit) < 1e-9, (b0, bfit)

    def summary(task_draw, bR):
        counts = {}
        for t in task_draw:
            counts[t] = counts.get(t, 0) + 1
        lr, rm = [], []
        for disp, (usable, dx, dm) in auto_cells.items():
            w = np.array([counts.get(t, 0) for t in usable], dtype=float)
            if w.sum() == 0:
                return None
            lr.append(float((w * dx).sum() / w.sum()))
            rm.append(float((w * -bR * dm).sum() / w.sum()))
        return dict(per_cell_log_rho=lr, per_cell_removed=rm, log_rho=float(np.mean(lr)), removed=float(np.mean(rm)), after=float(np.mean(lr) - np.mean(rm)),
                    share_removed=float(np.mean(rm) / np.mean(lr)), bR=bR)

    point = summary(all_tasks, b0)
    draws = []
    for k in range(B):
        td = list(rng.choice(all_tasks, len(all_tasks), replace=True))
        s = summary(td, fit_bR(auto_records, td))
        if s:
            draws.append(s)
        if k % 200 == 0:
            print("batching draw", k, flush=True)
    ci = {}
    for key in ("log_rho", "removed", "after", "share_removed", "bR"):
        v = np.array([d[key] for d in draws])
        ci[key] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for j, disp in enumerate(auto_cells):
        for key in ("per_cell_log_rho", "per_cell_removed"):
            v = np.array([d[key][j] for d in draws])
            ci[f"{key}.{disp}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        v = np.array([d["per_cell_log_rho"][j] - d["per_cell_removed"][j] for d in draws])
        ci[f"after.{disp}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["batching"] = dict(point=point, ci=ci, draws=len(draws), tasks=len(all_tasks), cells={k: len(v[0]) for k, v in auto_cells.items()}, episodes=len(auto_records))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "boot_cf.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["batching"], indent=1))


if __name__ == "__main__":
    main()
