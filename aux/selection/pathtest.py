"""The path test: the test success of the skills each optimizer run kept in turn on dev, which the runs never tested, and
whether the update of DeepSeek-V4.1-Flash after the runs moved the seed (mech_pathtest).

A run's path is the seed, then each SkillOpt or GEPA candidate whose score on the whole dev set beats the seed and every
earlier candidate, then the returned skill. A path with a skill between the seed and the returned one is tested once on
the test set, in a private episode cache. GPT-6-Luna did not change after its runs, so only the path skills its runs did
not test are run. DeepSeek-V4.1-Flash was updated by its provider after the runs (launch --new-release): every skill of
the path is tested on the update, against a rerun of the seed on it, on the dev and test sets of every benchmark.

Release: the system fingerprint of an episode's first request; the record checks that the passes ran on a release that
the pinned runs never recorded. Release shift: per benchmark the task model of the replay is fitted to the seed's
executions in the pinned runs, the mean success of a split is moved on the logit scale by one shift with the prior's
concentration held fixed, and the shift is fitted by maximum likelihood to the seed's executions on the update, per split
and once for both; a likelihood-ratio test asks whether it differs from zero. Paired sensitivity: each task keeps its own
posterior, and its rate moves by the shift on the logit scale.

Figure: along each path tested on the update, the dev gain the run reported over its starting score, and the test
difference to the seed rerun with one paired standard error and the tasks gained and lost.

  python -m aux.selection.pathtest plan [--model M] [--new-release]    the path skills and the evaluations to run
  python -m aux.selection.pathtest launch --model gpt-6-luna           runs them (calls the model; rerun to resume)
  python -m aux.selection.pathtest launch --model deepseek-flash --new-release
  python -m aux.selection.pathtest                                     record and figure from the passes

The passes on disk were launched in parts, before the plan kept only paths with an intermediate skill; the returned
skills of the other DeepSeek paths were left out by version id (aux/path_test/launch.sh):
  run_path_test --model gpt-6-luna --execute
  run_path_test --model deepseek-flash [--only automationbench | --only terminal-bench-2-1] --execute
      --skip f87d1b8897ab --skip 032508e0e6e1 --skip 5382240a88f3 --skip ee4f029e6a98 --skip ece3e407af64
      --skip fcba9d0bbc84 --skip bee6ffb67dfa [--skip 43226f3d6a15 --skip 9fc436ada6e6 --skip 1e1cc8c08336]
  run_path_test --model deepseek-flash --only gepa__tau3-bench --execute
  run_path_test --model deepseek-flash --only skillopt__tau3-bench --no-reuse --skip 55f8a4a38828 --skip 45ca0b86e2b2 --execute"""
import argparse, json, math, os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

from aux.selection import taskmodel as T
from src.utils import layout
from src.analysis import settings as S
from src.utils import fig as F
from src.utils.logger import md_table, record

PASSES = layout.ARTIFACTS / "path-test"
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
DS = S.DEEPSEEK
QUANTILES = (np.arange(4000) + 0.5) / 4000  # equal-weight quadrature of a task's posterior


def load(path):
    return json.loads(Path(path).read_text())


def known(data):
    return {S.slot(e): e["success"] for e in data["episodes"] if S.known(e)}


# ---- paths

def evaluations(run, split, rows):
    """{version id: evaluation} of a run's first evaluation of each version on the whole panel `rows`."""
    out, want = {}, Counter(S.slot(r) for r in rows)
    for path in sorted((run / "results/evaluations" / split).glob("*/evaluation.json")):
        data = load(path)
        if Counter(S.slot(r) for r in data["panel"]) == want:
            out.setdefault(data["version_id"], data)
    return out


def path(model, opt, bench):
    """The skills a run kept in turn on dev: [{round, role, vid, text, dev (solved, panel), test (the run's evaluation)}]."""
    run, cell = S.run(model, bench, opt), layout.RESULTS / model / opt / bench
    protocol = load(run / "native_protocol.json")
    rows = {s: [{"task_id": S.slot(r)[0], "repetition": 0} for r in protocol["panels"][s]] for s in ("sel", "test")}
    dev, test = evaluations(run, "dev", rows["sel"]), evaluations(run, "test", rows["test"])
    if load(cell / "rounds.json")["run_id"] != run.name:
        raise ValueError(f"{cell}/rounds.json is not the pinned run's")
    score = lambda v: (sum(known(dev[v]).values()), len(dev[v]["panel"])) if v in dev else None
    seed = S.skill(model, bench, "native-seed")
    entries = [{"round": 0, "role": "seed", "text": seed, "vid": S.version(seed), "dev": score(S.version(seed))}]
    for n in sorted(int(p.parent.name.split("-")[1]) for p in cell.glob("skills/round-*/candidate.md")):
        text = (cell / f"skills/round-{n:02d}/candidate.md").read_text()
        got = score(S.version(text))
        if got and got[0] > entries[-1]["dev"][0]:
            entries.append({"round": n, "role": "path", "text": text, "vid": S.version(text), "dev": got})
    final = S.skill(model, bench, opt)
    if entries[-1]["vid"] != S.version(final):
        entries.append({"round": None, "role": "path", "text": final, "vid": S.version(final), "dev": score(S.version(final))})
    entries[-1]["role"] = "returned"
    for e in entries:
        e["test"] = test.get(e["vid"])
    return {"opt": opt, "bench": bench, "protocol": protocol, "rows": rows, "entries": entries}


def paths(model):
    return [path(model, opt, b) for opt in ("skillopt", "gepa") for b in BENCHES if not S.returned_seed(model, b, opt)]


def passed(model, folder, vid, split="test"):
    """{task: success} of a pass's evaluation of a version, verified outcomes only (None when there is none)."""
    files = sorted((PASSES / "out" / model / folder / "results/evaluations" / split).glob(f"{vid[:12]}-*/evaluation.json"))
    return {t: e["success"] for t, e in S.episodes({0: files})[0].items() if S.known(e)} if files else None


# ---- plan and launch

def plan(model, new_release):
    """[(path, entry, split, out directory, panel rows, rows still to run)]."""
    items, seeds = [], set()
    for p in paths(model):
        if new_release and p["bench"] not in seeds:
            seeds.add(p["bench"])
            items += [(p, p["entries"][0], split, f"seed__{p['bench']}") for split in ("dev", "test")]
        if any(e["role"] == "path" for e in p["entries"]):
            items += [(p, e, "test", f"{p['opt']}__{p['bench']}") for e in p["entries"][1:] if new_release or e["test"] is None]
    out = []
    for p, e, split, folder in items:
        done, rows = passed(model, folder, e["vid"], split) or {}, p["rows"]["sel" if split == "dev" else "test"]
        out.append((p, e, split, PASSES / "out" / model / folder, rows, [r for r in rows if r["task_id"] not in done]))
    return out


def launch(model, items, slots):
    from src.utils.executor import Cell, evaluate
    from src.run import ENV
    os.environ.update(ENV)  # the capture protocol is part of the episode cache key
    for p, e, split, out, rows, missing in items:
        if missing:
            settings = p["protocol"]
            print(f"evaluate {p['opt']}/{p['bench']} round {e['round']} {e['vid'][:12]} {split}: {len(missing)} of {len(rows)} tasks", flush=True)
            cell = Cell(settings["benchmark"], settings["model"], settings["harness"], out.resolve(), slots=slots,
                        cache_root=(PASSES / "cache" / model).resolve())
            evaluate(cell, e["text"], rows, split)


# ---- what the passes show

def release(files, limit=8):
    """Counter of the system fingerprints of the first requests of up to `limit` executed episodes per evaluation file."""
    seen = Counter()
    for f in files:
        run = Path(f).parents[4]
        for e in [e for e in load(f)["episodes"] if not e.get("cache_hit")][:limit]:
            for r in e.get("requests", [])[:1]:
                head, sep, tail = r["record_path"].partition(f"/{run.name}/")
                try:
                    seen[(load(r["record_path"] if Path(r["record_path"]).exists() or not sep else run / tail).get("response") or {}).get("system_fingerprint")] += 1
                except OSError:
                    seen["unreadable"] += 1
    return seen


def loglik(counts, m, c):
    a, b = m * c, (1 - m) * c
    return float(sum(special.gammaln(k + a) + special.gammaln(n - k + b) - special.gammaln(n + a + b) - special.gammaln(a) - special.gammaln(b)
                     + special.gammaln(a + b) for k, n in counts))


def fit_shift(ll):
    """(shift, LR, p) of the maximum-likelihood shift against no shift."""
    best = optimize.minimize_scalar(lambda d: -ll(d), bounds=(-6, 6), method="bounded", options={"xatol": 1e-5})
    lr = max(0.0, 2 * (-best.fun - ll(0.0)))
    return float(best.x), lr, float(stats.chi2.sf(lr, 1))


def shift(old, new, prior):
    """{(model, split): (pinned mean %, shift, points, LR, p)} of the marginal model and of the paired sensitivity model."""
    a, b = prior
    base = {s: float(special.expit(optimize.minimize_scalar(lambda x: -loglik(old[s].values(), special.expit(x), a + b), bounds=(-9, 9),
                                                            method="bounded").x)) for s in new}
    grids = {s: [(np.clip(stats.beta.ppf(QUANTILES, a + ko, b + no - ko), 1e-12, 1 - 1e-12), k, n)
                 for t, (k, n) in new[s].items() for ko, no in [old[s].get(t, (0, 0))]] for s in new}
    moved = lambda q, d: special.expit(special.logit(q) + d)
    models = {"marginal": ({s: (lambda d, s=s: loglik(new[s].values(), moved(base[s], d), a + b)) for s in new},
                           lambda s, d: moved(base[s], d), base),
              "paired": ({s: (lambda d, s=s: sum(math.log(max(np.mean(moved(q, d) ** k * (1 - moved(q, d)) ** (n - k)), 1e-300)) + math.log(math.comb(n, k))
                                                 for q, k, n in grids[s])) for s in new},
                         lambda s, d: np.mean([moved(q, d).mean() for q, _, _ in grids[s]]), {s: np.mean([q.mean() for q, _, _ in grids[s]]) for s in new})}
    out = {}
    for name, (lls, after, before) in models.items():
        for s in new:
            d, lr, p = fit_shift(lls[s])
            out[name, s] = (100 * before[s], d, 100 * (after(s, d) - before[s]), lr, p)
        d, lr, p = fit_shift(lambda d: sum(f(d) for f in lls.values()))
        out[name, "dev+test"] = (None, d, None, lr, p)
    return out


def collect():
    """The DeepSeek paths tested on the update against the seed rerun, the GPT-6-Luna paths, the release check and the shift."""
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())
    ds = []
    for p in paths(DS):
        seed = passed(DS, f"seed__{p['bench']}", p["entries"][0]["vid"])
        tested = [(e, passed(DS, f"{p['opt']}__{p['bench']}", e["vid"])) for e in p["entries"][1:]]
        if not seed or len(tested) < 2 or not all(s for _, s in tested):
            continue
        start = T.observed(DS, p["bench"], p["entries"][0]["dev"][1])[S.ARM_NAME[p["opt"]]]["baseline"]
        earlier = table1[f"{DS}|{p['bench']}"]["Seed skill"]["Q"]
        rows = []
        for e, s in tested:
            common = [t for t in seed if t in s]
            g, l, n = sum(s[t] and not seed[t] for t in common), sum(seed[t] and not s[t] for t in common), len(common)
            rows.append({"round": e["round"], "returned": e["role"] == "returned", "dev_solved": e["dev"][0], "n_dev": e["dev"][1],
                         "dev_gain": 100 * (e["dev"][0] - start) / e["dev"][1], "diff": 100 * (g - l) / n, "gained": g, "lost": l, "n": n,
                         "se": 100 * math.sqrt(g + l - (g - l) ** 2 / n) / n, "solved": sum(s.values()), "vs_earlier": 100 * sum(s.values()) / len(s) - earlier})
        same = [sum(a[t] != b[t] for t in a if t in b) for (_, a), (_, b), x, y in zip(tested, tested[1:], rows, rows[1:]) if x["solved"] == y["solved"]]
        ds.append({"opt": p["opt"], "bench": p["bench"], "rows": rows, "seed": (sum(seed.values()), len(seed)), "same_score_differ": same, "earlier": earlier})
    luna = []
    for p in paths(S.LUNA):
        if not any(e["role"] == "path" for e in p["entries"]) or not p["entries"][0]["test"]:
            continue
        rate = lambda k: sum(k.values()) / len(k)
        seed = rate(known(p["entries"][0]["test"]))
        for e in p["entries"][1:]:
            got = passed(S.LUNA, f"{p['opt']}__{p['bench']}", e["vid"]) or (e["test"] and {t: v for (t, _), v in known(e["test"]).items()})
            if got:
                luna.append([S.ARM_NAME[p["opt"]], F.BENCH_TICK[p["bench"]], e["round"] if e["role"] == "path" else f"{e['round']}, returned",
                             f"{e['dev'][0]}/{e['dev'][1]}", f"{100 * (rate(got) - seed):+.1f}"])
    shifts, releases = {}, {}
    for bench in BENCHES:
        dev, test, _ = T.seed_executions(DS, bench)
        root = PASSES / "out" / DS / f"seed__{bench}/results/evaluations"
        files = {s: sorted((root / s).glob(f"{S.SEED_VERSION[:12]}-*/evaluation.json")) for s in ("dev", "test")}
        new = {s: {t: (sum(v), len(v)) for t, v in T.executions(fs).items()} for s, fs in files.items() if fs}
        if new:
            prior = T.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
            old = {s: {t: (sum(v), len(v)) for t, v in ex.items()} for s, ex in (("dev", dev), ("test", test))}
            shifts[bench] = {"prior": prior, "executions": sum(n for s in new.values() for _, n in s.values()), "fit": shift(old, new, prior)}
            pinned = [f for fs in S.cell_files(DS, bench, "native-seed").values() for f in fs][:3]
            releases[bench] = (release([f for fs in files.values() for f in fs]), release(pinned))
    return ds, luna, shifts, releases


def draw(ds):
    from matplotlib.lines import Line2D
    plt = F.pyplot(F.SMALL, title=F.TEXT)
    fig, axes = plt.subplots(1, len(ds), figsize=(F.TEXT_W, 2.25), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.725, wspace=0.12)
    titles, dx, axes = {}, 0.11, np.atleast_1d(axes)
    for ax, p in zip(axes, ds):
        rows, c, x = p["rows"], F.ARM[p["opt"]], np.arange(len(p["rows"]))
        dev, dif = [r["dev_gain"] for r in rows], [r["diff"] for r in rows]
        ax.axhline(0, color=F.MUTED, lw=0.6, zorder=1)
        ax.plot(x - dx, dev, color=c, lw=1.1, zorder=3)
        ax.scatter(x - dx, dev, s=22, marker="o", facecolor=c, edgecolor=c, lw=0.8, zorder=4)
        for i, v in zip(x, dev):
            ax.text(i - dx, v + 1.0, f"+{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=c)
        ax.errorbar(x + dx, dif, yerr=[r["se"] for r in rows], fmt="none", ecolor=c, elinewidth=0.9, capsize=3.0, capthick=0.9, zorder=3)
        ax.scatter(x + dx, dif, s=22, marker="o", facecolor="white", edgecolor=c, lw=1.0, zorder=4)
        for i, r in zip(x, rows):
            y = 5 * math.floor(r["diff"] / 5) + 2.5  # the middle of the grid gap that holds the point
            # test labels in italic grey, so they do not read as the dev gains (upright, in the optimizer's color)
            ax.annotate(f"+{r['diff']:.1f}", (i + dx, y), xytext=(5, 0.5), textcoords="offset points", ha="left", va="bottom", fontsize=6.8, color=F.MUTED, fontstyle="italic")
            ax.annotate(f"{r['gained']} gained, {r['lost']} lost", (i + dx, y), xytext=(5, -0.5), textcoords="offset points", ha="left", va="top", fontsize=6.8, color=F.MUTED, fontstyle="italic")
        ax.set_xticks(x, [f"round {r['round']}" + (", returned" if r["returned"] else "") for r in rows])
        ax.set(xlim=(-0.45, len(rows) - 0.05), ylim=(-5, 20.5), yticks=[0, 5, 10, 15])
        ax.grid(axis="y", color=F.GRID, lw=0.4)
        ax.set_axisbelow(True)
        F.despine(ax)
        ax.tick_params(axis="x", length=0, pad=3)
        climb = 100 * (rows[-1]["dev_solved"] - rows[0]["dev_solved"]) / rows[0]["n_dev"]
        within = max(abs(d - dif[0]) for d in dif)
        titles[p["opt"]] = f"{S.ARM_NAME[p['opt']]}: dev gain climbs {climb:.1f} points,\ntest difference stays within {within:.1f} points of its first value"
        first = f"round {rows[0]['round']}"  # the first kept skill; the paper's words for the same two numbers
        ax.set_title(f"{S.ARM_NAME[p['opt']]}: from {first} on, dev gain climbs {climb:.1f} points,\n"
                     f"test difference stays within {within:.1f} points of its {first.replace(' ', '-')} value", loc="left", color=F.INK, pad=4, linespacing=1.15)
    axes[0].set_ylabel("points", labelpad=2)
    test = axes[0].errorbar([np.nan], [np.nan], yerr=[1], fmt="o", ms=4, mfc="white", mec=F.INK, mew=1.0, ecolor=F.INK, elinewidth=0.9, capsize=3.0, capthick=0.9)
    handles = [Line2D([], [], color=F.INK, lw=1.1, marker="o", ms=4, mfc=F.INK, mec=F.INK), test]
    legend = fig.legend(handles, ["reported dev gain over the starting score", "test difference to the seed rerun"],
                        loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False, handlelength=2.2, columnspacing=2.0, handletextpad=0.5)
    legend.get_texts()[1].set(color=F.MUTED, fontstyle="italic")  # as the test labels
    F.save_figure(fig, "pathtest")
    return titles


def record_pathtest(ds, luna, shifts, releases, titles):
    rows = [[S.ARM_NAME[p["opt"]], F.BENCH_TICK[p["bench"]], r["round"] if not r["returned"] else f"{r['round']} (returned)", f"+{r['dev_gain']:.1f}",
             f"{r['solved']}/{r['n']}", f"{r['diff']:+.1f}", f"{r['gained']} / {r['lost']}", f"{r['se']:.1f}"] for p in ds for r in p["rows"]]
    later = [r["vs_earlier"] for p in ds for r in p["rows"]]
    fits = [[F.BENCH_TICK[b], name, s, "-" if v[0] is None else f"{v[0]:.1f}", f"{v[1]:+.3f}", "-" if v[2] is None else f"{v[2]:+.1f}", f"{v[3]:.2f}", f"{v[4]:.3f}"]
            for b, x in shifts.items() for (name, s), v in x["fit"].items()]
    priors = "; ".join(f"{F.BENCH_TICK[b]} Beta({x['prior'][0]:.3f}, {x['prior'][1]:.3f}), {x['executions']} executions on the update" for b, x in shifts.items())
    other = all(not (set(new) & set(old)) - {"unreadable"} for new, old in releases.values())
    seeds = {p["bench"]: p for p in ds}
    numbers = "\n\n".join(
        [f"Seed rerun on the update, {F.BENCH_TICK[b]} test: {p['seed'][0]} of {p['seed'][1]} solved, "
         f"{p['earlier'] - 100 * p['seed'][0] / p['seed'][1]:.1f} points below its {p['earlier']:.1f}% over the three runs of Table 1." for b, p in seeds.items()]
        + [titles[p["opt"]].replace("\n", " ") + "." + "".join(f" Two of its path skills solve as many tasks and differ on {k}." for k in p["same_score_differ"])
           for p in ds]
        + ["### Path skills on the update, against the seed rerun\n\n"
           + md_table(["optimizer", "bench", "round", "reported dev gain", "test solved", "difference to the rerun", "gained / lost", "paired SE"], rows)
           + f"\n\nAgainst the seed's Table 1 runs the path skills lie at {min(later):+.1f} to {max(later):+.1f} points, {np.mean(later):+.1f} on average.",
           "### Release\n\nThe seed's passes ran on " + ("a release the pinned runs never recorded." if other else "a release the pinned runs also recorded."),
           "### Shift of the seed's mean success on the update (logit scale)\n\n" + f"Prior of the pinned runs: {priors}.\n\n"
           + md_table(["bench", "model", "split", "pinned runs, %", "shift", "points", "LR", "p"], fits),
           "### GPT-6-Luna path skills (its own seed and returned evaluations)\n\n"
           + md_table(["optimizer", "bench", "round", "dev solved", "test difference to the run's seed, points"], luna)])
    cells = [(m, b, o) for m in (DS, S.LUNA) for b in BENCHES for o in ("native-seed", "skillopt", "gepa")]
    record(__spec__.name, "pathtest", numbers, S.runs_text(cells) + f"\n\nPasses: `{PASSES.relative_to(layout.ASSETS)}/out`.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", choices=("plan", "launch", "record"), default="record")
    ap.add_argument("--model", choices=S.MODEL_NAME, action="append")
    ap.add_argument("--new-release", action="store_true", help="the model changed after its runs: test every path skill and rerun the seed")
    ap.add_argument("--slots", type=int, default=8)
    args = ap.parse_args(argv)
    if args.action == "record":
        ds, luna, shifts, releases = collect()
        record_pathtest(ds, luna, shifts, releases, draw(ds))
        return
    for model in args.model or S.MODEL_NAME:
        items = plan(model, args.new_release)
        for p, e, split, out, rows, missing in items:
            print(f"{model} {p['opt']}/{p['bench']} round {e['round']} {e['role']} {e['vid'][:12]} {split}: {len(missing)} of {len(rows)} tasks to run"
                  f" -> {out.relative_to(PASSES)}")
        if args.action == "launch":
            launch(model, items, args.slots)


if __name__ == "__main__":
    main()
