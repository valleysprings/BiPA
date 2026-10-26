"""What the GPT-6-Luna runs from the three-word seed "Solve the task." recorded: the seed and the skills SkillOpt and GEPA
learn from it, their test runs, every candidate's dev score and length, and the sentences of the returned skills.

The runs are the experiments file's aux_runs three_word_seed. On AutoBench the three-word seed has three test runs of its
own (native-seed) and each learned skill three; on tau3-bench each optimizer run tested the three-word seed once beside the
skill it returned. They are set against Table 1's GPT-6-Luna cells, whose optimizers start from the hand-written seed. A
skill's evaluations are found by its version (settings.version of its text), and test runs are read as Table 1 reads them.
Words are whitespace tokens of the whole skill file, front matter included. A sentence is a piece of the skill's body
(front matter and headings removed) that ends in . ! ? or ; and has at least three words; each carries one reader's label in
sentence_labels.json: S0 good practice anywhere, S1 a convention of the whole benchmark, S2 names a specific tool, app,
field or command, S3 holds for one class of tasks; S2 and S3 are particular to a benchmark."""
import json, re
from pathlib import Path

from aux.selection import taskmodel as TM
from src.analysis import settings as S

BENCHES = ("automationbench", "tau3-bench")
OPTIMIZERS = ("skillopt", "gepa")
LABELS = Path(__file__).parents[1] / "data" / "sentence_labels.json"
SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9`\"'(*\-])")


def runs(bench):
    """{method: run directory} of the finished runs from the three-word seed on a benchmark."""
    return {p.parent.parent.name: p for p in S.extra("three_word_seed", S.LUNA, bench) if (p / "summary.json").is_file()}


def summary(run):
    return json.loads((run / "summary.json").read_text())


def seed(bench):
    """The three-word seed, as the SkillOpt run started from it."""
    return (runs(bench)["skillopt"] / "skillopt/seed.md").read_text()


def returned(run):
    return summary(run)["final_skill"]


def last_round(run):
    """The skill SkillOpt holds after its last round (it returns the one with the best dev score)."""
    return sorted((run / "skillopt/skills").glob("skill_v*.md"))[-1].read_text()


def words(text):
    return len(text.split())


def agents(bench):
    """{name: {repetition: [evaluation.json]}}: the three-word seed and the skills learned from it, then Table 1's cells."""
    r, s = runs(bench), seed(bench)
    if "native-seed" in r:
        out = {"three-word seed": S.run_files(r["native-seed"], s)}
    else:
        out = {f"three-word seed ({S.ARM_NAME[o]} run)": S.run_files(r[o], s) for o in OPTIMIZERS}
    out |= {f"{S.ARM_NAME[o]} from three words": S.run_files(r[o], returned(r[o])) for o in OPTIMIZERS}
    if "native-seed" not in r:
        out["SkillOpt from three words, last round"] = S.run_files(r["skillopt"], last_round(r["skillopt"]))
    out |= {"no skill": S.cell_files(S.LUNA, bench, "native-noseed"), "hand-written seed": S.cell_files(S.LUNA, bench, "native-seed")}
    return out | {f"{S.ARM_NAME[o]} from the hand-written seed": S.cell_files(S.LUNA, bench, o) for o in S.OPTIMIZERS}


def success(files):
    """Test success (%) of each run: the share of its tasks with a verified outcome that it solves."""
    return [100 * sum(e["success"] for e in eps.values() if S.known(e)) / sum(S.known(e) for e in eps.values())
            for eps in S.episodes(files).values()]


def dev_files(run, text):
    return sorted((run / "results/evaluations/dev").glob(f"{S.version(text)[:12]}-*/evaluation.json"))


def dev_solved(run, text):
    """Dev tasks a version solved in a run, one count per dev evaluation of it."""
    return [sum(bool(e["success"]) for e in json.loads(f.read_text())["episodes"] if TM.verified(e)) for f in dev_files(run, text)]


def selection(bench, method):
    """K candidates, the baseline and the returned skill's dev tasks solved, and the dev panel size, as the run recorded them.
    SkillOpt's baseline is its own dev evaluation of the seed; GEPA's is its first validation score."""
    run = runs(bench)[method]
    n, extra = summary(run)["panels"]["sel"], summary(run)["extra"]
    if method == "skillopt":
        k, base = extra["total_accepts"] + extra["total_rejects"], dev_solved(run, seed(bench))[0]
    else:
        k, base = len(extra["val_scores"]) - 1, round(extra["val_scores"][0] * n)
    return {"K": k, "baseline": base, "returned": dev_solved(run, returned(run))[0], "n_dev": n}


def skillopt_candidates(bench):
    """Every SkillOpt candidate: dev tasks solved, words, depth (1 = a rewrite of the seed), accepted, returned; then the
    skill after the last round."""
    run = runs(bench)["skillopt"]
    out, depth, n = [], 0, summary(run)["panels"]["sel"]
    for x in json.loads((run / "skillopt/history.json").read_text()):
        text = (run / "skillopt/steps" / f"step_{x['step']:04d}" / "candidate_skill.md").read_text()
        out.append(dict(dev=round(n * x["selection_hard"]), words=words(text), depth=depth + 1, accepted=x["action"].startswith("accept"),
                        returned=x["step"] == x["best_step"] and x["action"] == "accept_new_best", tag=f"step {x['step']}"))
        depth += x["action"].startswith("accept")
    out.append(dict(dev=round(n * summary(run)["extra"]["final_selection_hard"]), words=words(last_round(run)), depth=depth + 1,
                    accepted=False, returned=False, tag="last round"))
    return out


def gepa_candidates(bench):
    """(seed dev score, every GEPA candidate: dev tasks solved, words, depth, parent, returned), from its candidate log."""
    g = runs(bench)["gepa"] / "gepa"
    n = summary(g.parent)["panels"]["sel"]
    cand, score, last = json.loads((g / "candidates.json").read_text()), {}, None
    for line in (g / "run_log.txt").read_text().splitlines():
        if m := re.match(r"Iteration \d+: Base program full valset score: ([0-9.]+)", line):
            score[0] = float(m.group(1))
        if m := re.match(r"Iteration \d+: Valset score for new program: ([0-9.]+)", line):
            last = float(m.group(1))
        if m := re.match(r"Iteration \d+: New program candidate index: (\d+)", line):
            score[int(m.group(1))], last = last, None
    parent = {e["new_program_idx"]: e["tasks"][0]["parent_idx"] for e in json.loads((g / "run_log.json").read_text())
              if e.get("new_program_idx") is not None}
    depth = {0: 0}
    for i in range(1, len(cand)):
        depth[i] = depth[parent[i]] + 1
    best = max(range(1, len(cand)), key=lambda i: (score[i], -i))
    return round(n * score[0]), [dict(dev=round(n * score[i]), words=words(cand[i]["current_candidate"]), depth=depth[i], parent=parent[i],
                                      returned=i == best, tag=f"candidate {i}") for i in range(1, len(cand))]


def sentences(text):
    body = text.split("---", 2)[2] if text.startswith("---") else text
    out = []
    for line in body.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            line = re.sub(r"^(#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)", "", line.strip()).strip()
            out += [s.strip() for s in SPLIT.split(line) if len(s.strip().split()) >= 3]
    return out


def labelled(bench, method):
    """[(sentence, level)] of a returned skill; a sentence without a hand label is an error."""
    labels = {s: level for s, level, _ in json.loads(LABELS.read_text())[f"{method}|{bench}"]}
    found = sentences(returned(runs(bench)[method]))
    if set(found) != set(labels):
        raise ValueError(f"{method} {bench}: sentences without a label: {sorted(set(found) - set(labels))}")
    return [(s, labels[s]) for s in found]
