"""One-pass TEST evaluation of the optimization-path skills of the Table 1 runs, modeled on the repeat action of src/run.py.

Path skills of a run: the seed, then every round's candidate whose full-dev-panel success (its existing dev evaluation in the run) is strictly
above the best of the seed and all earlier candidates (= the skill the optimizer would have returned had it stopped after that round), and
the returned skill (summary.json final_skill).  SkillAdam on tau3/AutoBench never scored a candidate on the full dev panel (10-task
minibatches only); there the strict path is seed + returned, and --skilladam-accept-chain adds its accepted candidates (its own
returned-if-stopped-here skills; dev = union of minibatch slots, a different estimator).

What gets evaluated (repetition 0, one pass):
  gpt-6-luna      path skills without a test evaluation in their run (seed and returned are already tested in every Luna run).
  deepseek-flash  on the CURRENT model release (FINGERPRINT['new']): path skills without a test evaluation; the returned skill of each
                  run unless a new-model test of it exists on the same panel; the seed once per benchmark on dev (sel) and on test
                  (calibration of the old->new model gap, see estimate_gap.py).  Path skills are not re-evaluated on dev.

Default is --dry-run (prints the plan, calls nothing, imports nothing from harness-bench).  --execute runs it.  Output goes to
ARTIFACTS/path-test/out/<model>/<opt>__<bench>/ (seed calibration: seed__<bench>/); the episode cache defaults to a private
root ARTIFACTS/path-test/cache/<model>/ because the slot cache key carries the route alias ("deepseek-flash") and no model
fingerprint: the frozen DeepSeek cache root would serve old-model (aeb564...) episodes for the seed's dev and the returned skills' test.

Run (from the repo root, host.env sourced, with the interpreter that runs src/run.py):
  python -u -m aux.path_test.run_path_test --model gpt-6-luna --execute
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from aux.headline_numbers.dynamics import slot
from src import layout

EXECUTION, RESULTS, SEED_FILE = layout.EXECUTION, layout.RESULTS, layout.SEED_FILE
PATH_TEST = layout.ARTIFACTS / "path-test"   # the raw test passes
OUT = PATH_TEST / "out"
PRIVATE_CACHE = PATH_TEST / "cache"
RUNS = {"deepseek-flash": "native-20261008-https", "gpt-6-luna": "native-20261012"}
OPTIMIZERS = ("skillopt", "gepa", "skilladam")
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
SKIP_CELLS = {("gpt-6-luna", b) for b in ("terminal-bench-2-1",)}  # Luna TB2: all three optimizers returned the seed
TEST_SIZE = {"tau3-bench": 75, "automationbench": 120, "terminal-bench-2-1": 18}
DEV_SIZE = {"tau3-bench": 30, "automationbench": 30, "terminal-bench-2-1": 9}
# per-task mean input tokens of the seed (millions), and the learned skills' measured input-token ratio range
TOKENS_M = {"deepseek-flash": {"tau3-bench": 0.70, "automationbench": 1.10, "terminal-bench-2-1": 1.38},
            "gpt-6-luna": {"tau3-bench": 0.49, "automationbench": 0.69, "terminal-bench-2-1": 0.90}}
SKILL_TOKEN_RATIO = (1.08, 1.43)  # README RQ2 (DeepSeek learned skills); used as the range for non-seed skills
FINGERPRINT = {"old": "aeb56401ca74e127821c4f9126dcb669", "new": "e72d3c7f663d3e1383b00b3884cb4501"}  # deepseek-flash releases


def version(text):
    files = {} if text is None else {"SKILL.md": text}
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalized(panel):
    return sorted(({"task_id": slot(r)[0], "repetition": 0, "weight": 1.0} for r in panel), key=lambda r: (r["task_id"], r["repetition"]))


def digest(rows):  # harness-bench eval/evaluate.py Evaluator.directory
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def known(episodes):
    return {slot(e): e["success"] for e in episodes if type(e.get("success")) is bool and e.get("outcome_valid", True) is True}


def load(path):
    return json.loads(Path(path).read_text())


def evaluations(run, split):
    """{version_id: [(path, data)]} of one run's evaluations of a split."""
    out = collections.defaultdict(list)
    for path in sorted((run / "results/evaluations" / split).glob("*/evaluation.json")):
        data = load(path)
        out[data["version_id"]].append((path, data))
    return out


def full(data, rows):
    return collections.Counter(slot(r) for r in data["panel"]) == collections.Counter(slot(r) for r in rows)


def delivered_text(run, vid):
    """The delivered bytes of a version (results/skills/deliveries/evaluations/*/<vid12>-*/native/SKILL.md), checked against vid."""
    for path in sorted((run / "results/skills/deliveries/evaluations").glob(f"*/{vid[:12]}-*/native/SKILL.md")):
        text = path.read_text()
        if version(text) == vid:
            return text, path
    return None, None


def first_fingerprints(data, limit=8):
    """system_fingerprint (or response.model) of the first request of up to `limit` executed episodes."""
    seen = collections.Counter()
    for e in [e for e in data.get("episodes", []) if not e.get("cache_hit")][:limit]:
        for r in e.get("requests", [])[:1]:
            try:
                resp = load(r["record_path"]).get("response") or {}
            except (OSError, ValueError, KeyError):
                seen["unreadable"] += 1
                continue
            seen[resp.get("system_fingerprint") or resp.get("model")] += 1
    return dict(seen)


# ---- the path of one run -----------------------------------------------------------------------------------------------------------

def run_cell(model, opt, bench):
    run = EXECUTION / model / opt / bench / RUNS[model]
    protocol, summary = load(run / "native_protocol.json"), load(run / "summary.json")
    for key in ("model", "benchmark", "arm", "harness", "scale"):
        if summary.get(key) != protocol.get(key):
            raise ValueError(f"{run}: summary/protocol mismatch on {key}")
    if protocol.get("protocol") != "native-v1":
        raise ValueError(f"{run}: not a frozen native-v1 run")
    if (run / "final/SKILL.md").read_text() != (summary["final_skill"] or ""):
        raise ValueError(f"{run}: final/SKILL.md differs from summary.final_skill")
    seed = SEED_FILE.read_text()
    dev_rows, test_rows = normalized(protocol["panels"]["sel"]), normalized(protocol["panels"]["test"])
    dev = {vid: next(((p, d) for p, d in items if full(d, dev_rows)), None) for vid, items in evaluations(run, "dev").items()}
    dev = {vid: item for vid, item in dev.items() if item}
    test = {vid: next(((p, d) for p, d in items if full(d, test_rows)), None) for vid, items in evaluations(run, "test").items()}
    test = {vid: item for vid, item in test.items() if item}
    rounds = load(RESULTS / model / opt / bench / "rounds.json")
    if rounds["run_id"] != RUNS[model]:
        raise ValueError(f"{model}/{opt}/{bench}: results/ rounds.json belongs to {rounds['run_id']}")
    return run, protocol, summary, seed, dev_rows, test_rows, dev, test, rounds


def dev_score(dev, vid):
    if vid not in dev:
        return None
    k = known(dev[vid][1]["episodes"])
    return sum(k.values()), len(k), len(dev[vid][1]["panel"])


def union_dev(run, vid, dev_rows):
    """SkillAdam: the union of a version's dev-minibatch slots (cache-consistent: same version, task and repetition -> same episode)."""
    union = {}
    for _, data in evaluations(run, "dev").get(vid, []):
        union.update(known(data["episodes"]))
    union = {k: v for k, v in union.items() if k in {slot(r) for r in dev_rows}}
    return sum(union.values()), len(union), len(dev_rows)


def path_of(model, opt, bench, accept_chain=False):
    run, protocol, summary, seed, dev_rows, test_rows, dev, test, rounds = run_cell(model, opt, bench)
    seed_vid, final_text = version(seed), summary["final_skill"]
    final_vid = version(final_text)
    entries = [{"round": 0, "role": "seed", "vid": seed_vid, "text": seed, "dev": dev_score(dev, seed_vid)}]
    best = entries[0]["dev"][0] if entries[0]["dev"] else None
    strict_dev = best is not None
    candidates = []
    for record in rounds["rounds"][1:]:
        n = record["round"]
        path = RESULTS / model / opt / bench / f"skills/round-{n:02d}/candidate.md"
        if not path.exists():
            continue  # no proposal (e.g. SkillAdam patch_failed)
        text = path.read_text()
        candidates.append((n, record, text, version(text)))
    if strict_dev:
        for n, record, text, vid in candidates:
            score = dev_score(dev, vid)
            if score and score[0] > best:
                entries.append({"round": n, "role": "path", "vid": vid, "text": text, "dev": score})
                best = score[0]
    elif accept_chain:
        for n, record, text, vid in candidates:
            if record.get("decision") == "accepted" and vid != entries[-1]["vid"]:
                entries.append({"round": n, "role": "accept", "vid": vid, "text": text, "dev": None,
                                "dev_union": union_dev(run, vid, dev_rows)})
    if entries[-1]["vid"] == final_vid:
        entries[-1]["role"] = "seed=returned" if final_vid == seed_vid else "returned"
    else:
        entries.append({"round": None, "role": "returned", "vid": final_vid, "text": final_text, "dev": dev_score(dev, final_vid)})
    if not strict_dev:
        entries[0]["dev_union"] = union_dev(run, seed_vid, dev_rows)
        entries[-1].setdefault("dev_union", union_dev(run, final_vid, dev_rows))
    for e in entries:
        e["test"] = test.get(e["vid"])
        if e["role"] == "path" and not delivered_text(run, e["vid"])[0]:
            raise ValueError(f"{model}/{opt}/{bench}: no delivered text for path skill {e['vid'][:12]}")
    return {"model": model, "opt": opt, "bench": bench, "run": run, "protocol": protocol, "dev_rows": dev_rows, "test_rows": test_rows,
            "entries": entries, "strict_dev": strict_dev}


def new_model_test(bench, vid, test_rows):
    """A test evaluation of `vid` on the same panel under execution/deepseek-flash, outside RUNS['deepseek-flash'], complete and on the new model."""
    name = f"{vid[:12]}-{digest(test_rows)[:12]}"
    # run dirs are execution/deepseek-flash/<method>/<bench>/<run>/ and test-reps/<bench>__<method>__<run>/ (fixed depths, no ** walk)
    paths = list(EXECUTION.glob(f"deepseek-flash/*/{bench}/*/results/evaluations/test/{name}/evaluation.json"))
    paths += list(EXECUTION.glob(f"deepseek-flash/test-reps/{bench}__*/results/evaluations/test/{name}/evaluation.json"))
    for path in sorted(paths):
        if RUNS["deepseek-flash"] in str(path):
            continue
        data = load(path)
        if data.get("version_id") != vid or not full(data, test_rows) or len(known(data["episodes"])) != len(test_rows):
            continue
        prints = first_fingerprints(data)
        if prints and set(prints) == {FINGERPRINT["new"]}:
            return path
    return None


# ---- the evaluation plan ----------------------------------------------------------------------------------------------------------

def out_dir(model, opt, bench):
    return OUT / model / f"{opt}__{bench}"


def done_in(out, vid, rows, split):
    """Known outcomes of an earlier execution of this script for (vid, split, panel), for resume and counting."""
    for path in sorted((out / "results/evaluations" / split).glob(f"{vid[:12]}-*/evaluation.json")):
        data = load(path)
        if data.get("version_id") == vid and full(data, rows):
            return known(data["episodes"])
    return {}


def plan(model, only=(), accept_chain=False, skip=(), no_reuse=False):
    cells, items = [], []
    seen_seed = set()
    for opt in OPTIMIZERS:
        for bench in BENCHES:
            name = f"{opt}__{bench}"
            if (model, bench) in SKIP_CELLS or (only and name not in only and bench not in only and opt not in only):
                continue
            cell = path_of(model, opt, bench, accept_chain)
            cells.append(cell)
            for e in cell["entries"]:
                key = (opt, bench, e["vid"][:12])
                if any(s in key or f"{opt}__{bench}__{e['vid'][:12]}" == s for s in skip):
                    continue
                if model == "gpt-6-luna":
                    if e["test"] is None:  # seed and returned already have one in every Luna run
                        items.append(dict(cell=cell, kind=e["role"], entry=e, split="test", rows=cell["test_rows"],
                                          out=out_dir(model, opt, bench)))
                    continue
                if e["role"] in ("seed", "seed=returned"):
                    if bench not in seen_seed:  # seed calibration once per benchmark, dev and test
                        seen_seed.add(bench)
                        for split, rows in (("dev", cell["dev_rows"]), ("test", cell["test_rows"])):
                            items.append(dict(cell=cell, kind="seed-calibration", entry=e, split=split, rows=rows,
                                              out=out_dir(model, "seed", bench)))
                    if e["role"] == "seed":
                        continue
                if e["role"] in ("returned", "seed=returned"):
                    existing = None if no_reuse else new_model_test(bench, e["vid"], cell["test_rows"])
                    e["new_model_test"] = existing
                    if existing is None and e["role"] == "returned":
                        items.append(dict(cell=cell, kind="returned", entry=e, split="test", rows=cell["test_rows"],
                                          out=out_dir(model, opt, bench)))
                    continue
                if e["test"] is None:
                    items.append(dict(cell=cell, kind=e["role"], entry=e, split="test", rows=cell["test_rows"],
                                      out=out_dir(model, opt, bench)))
    for item in items:
        done = done_in(item["out"], item["entry"]["vid"], item["rows"], item["split"])
        item["missing"] = [r for r in item["rows"] if (r["task_id"], 0) not in done]
    return cells, items


def fmt_dev(e):
    if e.get("dev"):
        s, k, n = e["dev"]
        return f"{s}/{n}" + ("" if k == n else f" ({k} known)")
    if e.get("dev_union"):
        s, k, n = e["dev_union"]
        return f"[{s}/{k} of {n} via minibatches]"
    return "-"


def report(model, cells, items):
    print(f"# {model}: path skills (round, version, dev success on the full dev panel, test evaluation in the run)")
    for cell in cells:
        flag = "" if cell["strict_dev"] else "   (no full-dev evaluation of any candidate: SkillAdam minibatches only)"
        print(f"## {cell['opt']}/{cell['bench']}  run={cell['run'].name}  path length={len(cell['entries'])}{flag}")
        for e in cell["entries"]:
            test = e["test"]
            tq = f"test {sum(known(test[1]['episodes']).values())}/{len(test[1]['panel'])}" if test else "no test"
            extra = f"  new-model test: {e['new_model_test'].relative_to(EXECUTION)}" if e.get("new_model_test") else ""
            print(f"   r{e['round'] if e['round'] is not None else '-':>2}  {e['role']:<13} {e['vid'][:12]}  dev {fmt_dev(e):<28} {tq}{extra}")
    print(f"\n# {model}: planned evaluations (repetition 0, one pass)")
    totals = collections.Counter()
    tokens = collections.defaultdict(lambda: [0.0, 0.0])
    for it in items:
        bench, n = it["cell"]["bench"], len(it["missing"])
        ratio = (1.0, 1.0) if it["kind"] == "seed-calibration" else SKILL_TOKEN_RATIO
        per = TOKENS_M[model][bench]
        tokens[bench][0] += n * per * ratio[0]
        tokens[bench][1] += n * per * ratio[1]
        totals[(bench, it["kind"], it["split"])] += n
        print(f"   {it['kind']:<16} {it['cell']['opt']:<9} {bench:<19} {it['split']:<4} {it['entry']['vid'][:12]}  "
              f"episodes {n:>3}/{len(it['rows'])}  -> {it['out'].relative_to(PATH_TEST)}")
    print(f"\n# {model}: episodes by benchmark, kind and split")
    for key, n in sorted(totals.items()):
        print(f"   {key[0]:<19} {key[1]:<16} {key[2]:<4} {n}")
    total = sum(totals.values())
    lo, hi = sum(v[0] for v in tokens.values()), sum(v[1] for v in tokens.values())
    for bench, (a, b) in sorted(tokens.items()):
        print(f"   tokens {bench:<19} {a/1000:.2f}-{b/1000:.2f} B input tokens")
    print(f"   TOTAL {total} episodes, {lo/1000:.2f}-{hi/1000:.2f} B input tokens "
          f"(seed per-task means x episodes; learned skills x{SKILL_TOKEN_RATIO[0]}-{SKILL_TOKEN_RATIO[1]})")
    return total


# ---- checks and execution ---------------------------------------------------------------------------------------------------------

def selfcheck(model, cells):
    problems = 0
    def check(ok, text):
        nonlocal problems
        problems += not ok
        print(f"   {'ok ' if ok else 'FAIL'} {text}")
    print(f"# {model}: selfcheck")
    panels = collections.defaultdict(set)
    for cell in cells:
        tag = f"{cell['opt']}/{cell['bench']}"
        run, test_rows, dev_rows = cell["run"], cell["test_rows"], cell["dev_rows"]
        check(len(test_rows) == TEST_SIZE[cell["bench"]] and len({r["task_id"] for r in test_rows}) == len(test_rows),
              f"{tag}: frozen test panel loads, {len(test_rows)} unique tasks")
        check(len(dev_rows) == DEV_SIZE[cell["bench"]], f"{tag}: frozen dev (sel) panel loads, {len(dev_rows)} tasks")
        panels[(cell["bench"], "test")].add(json.dumps(test_rows, sort_keys=True))
        panels[(cell["bench"], "dev")].add(json.dumps(dev_rows, sort_keys=True))
        for e in cell["entries"]:
            if e["role"] == "path":
                text, where = delivered_text(run, e["vid"])
                check(text == e["text"], f"{tag}: r{e['round']} candidate.md == delivered {where.parent.parent.name}/native/SKILL.md")
            elif e["role"] == "accept":
                text, where = delivered_text(run, e["vid"])
                check(text == e["text"], f"{tag}: r{e['round']} accepted candidate matches its delivery")
            if e["dev"] is not None:
                check(e["dev"][1] == e["dev"][2], f"{tag}: {e['vid'][:12]} dev evaluation complete ({e['dev'][1]}/{e['dev'][2]} known)")
    for (bench, split), values in sorted(panels.items()):
        check(len(values) == 1, f"{bench} {split}: the three optimizers share one frozen panel")
    print(f"   {problems} problem(s)")
    return problems


def execute(model, cells, items, slots, cache_root):
    from src.executor import Cell, evaluate
    from src.layout import relocate
    from src.run import ENV
    os.environ.update(ENV)  # native-v1 env: part of the slot identity, read at call time
    # returned skills skipped because a new-model test on the same panel exists: listed so results.json alone drives the figure
    summary = [{"kind": "existing-new-model", "opt": c["opt"], "bench": c["bench"], "split": "test", "version_id": e["vid"],
                "round": e["round"], "dev": e.get("dev"), "out": str(e["new_model_test"])}
               for c in cells for e in c["entries"] if e.get("new_model_test")]
    for it in items:
        cell, e, out = it["cell"], it["entry"], it["out"].resolve()
        if not it["missing"]:
            print(f"skip (complete in {out}): {it['kind']} {e['vid'][:12]} {it['split']}", flush=True)
        else:
            protocol = cell["protocol"]
            root = Path(relocate(protocol["cache_root"])).resolve() if cache_root == "frozen" else Path(cache_root).resolve()
            print(f"evaluate {it['kind']} {cell['opt']}/{cell['bench']} {e['vid'][:12]} {it['split']}: {len(it['missing'])} slots, "
                  f"cache {root}", flush=True)
            target = Cell(protocol["benchmark"], protocol["model"], protocol["harness"], out, slots=slots, cache_root=root)
            # always the full panel: one evaluation directory per (vid, split) that harness-bench resumes from disk (known slots are
            # kept, the private cache serves finished ones), so done_in() sees everything after an interruption
            evaluate(target, e["text"], [dict(r, repetition=0) for r in it["rows"]], it["split"])
        done = done_in(out, e["vid"], it["rows"], it["split"])
        summary.append({"kind": it["kind"], "opt": cell["opt"], "bench": cell["bench"], "split": it["split"], "version_id": e["vid"],
                        "round": e["round"], "dev": e.get("dev"), "known": len(done), "tasks": len(it["rows"]),
                        "Q_known": sum(done.values()) / len(done) if done else None, "out": str(out)})
        (OUT / model).mkdir(parents=True, exist_ok=True)
        (OUT / model / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(RUNS), action="append", help="default: both")
    parser.add_argument("--only", action="append", default=[], help="opt__bench, bench or opt (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="version id prefix (12) or opt__bench__vid12 to leave out")
    parser.add_argument("--skilladam-accept-chain", action="store_true",
                        help="SkillAdam tau3/AutoBench: add the accepted candidates (no full-dev score exists for them)")
    parser.add_argument("--no-reuse", action="store_true", help="do not reuse an existing new-model test of a returned skill from another run (e.g. delivery-check, whose harness differs)")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--cache-root", default=None,
                        help="default: private ARTIFACTS/path-test/cache/<model>; 'frozen' = the run's protocol cache_root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.slots < 1:
        parser.error("slots must be positive")
    problems = 0
    for model in args.model or sorted(RUNS):
        cells, items = plan(model, tuple(args.only), args.skilladam_accept_chain, tuple(args.skip), args.no_reuse)
        report(model, cells, items)
        if args.selfcheck:
            problems += selfcheck(model, cells)
        elif args.execute:
            if model == "deepseek-flash" and any(o in OPTIMIZERS for o in args.only):
                parser.error("split DeepSeek processes by benchmark, not by optimizer: the seed calibration of a benchmark would run "
                             "in several processes into the same seed__<bench> directory")
            execute(model, cells, items, args.slots, args.cache_root or str(PRIVATE_CACHE / model))
        else:
            print(f"\n(dry run: nothing was called; add --execute to run {model})\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
