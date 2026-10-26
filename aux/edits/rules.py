"""What the optimizers write into the skill: the rules each learned skill adds to the seed, by category, and the released
skills.

A rule (unit) is one added bullet, numbered item, table row or example line, with wrapped lines joined to the line they
continue; a line is added when the whole-line diff against the seed inserts or replaces it. Front matter, headings, fences
and table separators are structure and are not coded. Every unit was read by hand, and its category comes from
aux/data/rule_labels.json by model and text. A keyword pass proposes a category (the first of serialize, parallelize,
limit_read, ask_stop, verify, domain, format whose pattern matches, else other); a unit without a label stops the run with
its proposal, to be read and added to the file. Size is counted in whitespace-separated words and whole lines.

The second coder, a language model given only the category definitions, labelled every DeepSeek unit
(aux/data/coder2_labels.json, keyed <optimizer>|<benchmark>#<unit index>). It saw the labels of GEPA's tau3-bench skill
before coding, so agreement is also reported without that skill. Agreement is the share of equal labels and Cohen's kappa,
over the eight categories and, for each group of categories the paper relies on, over in-group or not.

An edit (aux.edits.lineage) lengthens the skill when its child has more characters than its parent; it counts when parent
and child share a dev panel. OUT/skills holds the seed and the skill each optimizer returned, each checked to be the run's
final/SKILL.md and the skill delivered at test."""
import collections, difflib, json, re

import numpy as np

from aux.edits import lineage as L
from src import layout
from src.analysis import settings as S
from src.utils.dataloader import BENCHES, SEED_TEXT
from src.utils.fig import BENCH_LABEL
from src.utils.logger import md_table, record

LABELS = layout.ROOT / "aux/data/rule_labels.json"
CODER2 = layout.ROOT / "aux/data/coder2_labels.json"
CATS = ("serialize", "parallelize", "limit_read", "ask_stop", "verify", "domain", "format", "other")
GROUPS = {"verify": ["verify"], "serialize": ["serialize"], "limit_read + parallelize": ["limit_read", "parallelize"]}
EXPOSED = "gepa|tau3-bench#"  # the skill whose labels the second coder saw
PROPOSE = {k: re.compile(v, re.I) for k, v in {
    "serialize": r"one at a time|\bsequential|one step at a time|\bONE candidate|single step|never (?:in )?parallel|not in parallel|without (?:batching|overlap)",
    "parallelize": r"in parallel|paralleli[sz](?:e|es|ed|ing)\b|parallel(?:ly| calls| tool)|batch(?:ed|ing)? (?:the )?(?:calls|reads|requests)|concurrent|simultaneous|same turn|multiple (?:tool )?calls at once",
    "limit_read": r"read (?:it |each |the file )?once|don'?t re-?read|do not re-?read|\bhead\b|\btail\b|\bgrep\b|truncat|only the (?:relevant|needed)|concise|limit (?:the )?output",
    "ask_stop": r"explicit (?:\"?yes|confirmation|consent)|ask the user|obtain (?:explicit )?confirmation|consent|\bdeny\b|\bdenied\b|escalat|transfer to|hand ?off|\bstop\b|state (?:it|that) cannot be done|the user can|performed by the user|user-run|wait for the (?:user|result)|\bblocker\b|cannot (?:be done|run it yourself)",
    "verify": r"verif|re-?read|read (?:it )?back|re-?query|re-?run|re-?check|re-?measure|re-?apply|confirm|evidence|\bfresh|audit|validat|diff\b|compare|\bcheck|proof|prove|\btests?\b|oracle|reference|assert|\binspect|count",
    "domain": r"polic|eligib|refund|\border|record|account|\bfee|price|\brate\b|balance|dispute|\bSIM\b|roaming|\bAPI|tool|\bfile|path|binary|compile|\bgcc\b|python|clone|commit|push|curl|directory|volume|schema|sheet|email|recipient|window|deadline|clock|time|authenticat|identity|identif",
    "format": r"format|wording|summary|quote|paraphrase|phrase|report (?:the|exactly)|filename|tone|\bsay\b|\bstate\b",
}.items()}
MARKER = re.compile(r"^\s*(?:[-*]\s|\d+\.\s|\||✅|❌)")
FENCE, HEADING, RULE = re.compile(r"^\s*```"), re.compile(r"^#{1,6}\s"), re.compile(r"^---\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
# the folders and benchmark names of the released skills
RELEASE = {"tau3-bench": "tau3-bench", "automationbench": "automationbench", "terminal-bench-2-1": "terminal-bench-2.1"}
BENCH_NAME = {"tau3-bench": "tau3-bench", "automationbench": "AutomationBench", "terminal-bench-2-1": "Terminal-Bench 2.1"}
SUSPECT = re.compile(r"/mnt/|/home/|sk-[A-Za-z0-9]{16,}|api[_-]?key|token\s*[:=]|password|[\w.+-]+@[\w-]+\.[a-z]{2,}|BEGIN [A-Z ]*PRIVATE", re.I)


def units(text):
    """Logical units of a skill: {kind, lines, text}; wrapped lines and indented fence continuations join the rule before them."""
    out, in_fence, in_front, prev, blank = [], False, False, None, True
    for i, line in enumerate(text.splitlines()):
        s, kind = line.strip(), None
        if (i == 0 and RULE.match(line)) or in_front:
            in_front, kind = i == 0 or not RULE.match(line), "frontmatter"
        elif not s:
            prev, blank = None, True
            continue
        elif FENCE.match(line):
            in_fence, kind = not in_fence, "fence"
        elif TABLE_SEP.match(line):
            kind = "separator"
        elif not in_fence and HEADING.match(line):
            kind = "heading"
        if kind:
            out.append({"kind": kind, "lines": [i], "text": s})
        elif out and out[-1]["kind"] == "rule" and prev == "rule" and not blank and (
                line[:1] in (" ", "\t") and not MARKER.match(line) if in_fence
                else not MARKER.match(line) and not s.startswith(("**", "|"))):
            out[-1]["lines"].append(i)
            out[-1]["text"] += " " + s
        else:
            out.append({"kind": "rule", "lines": [i], "text": s})
        prev, blank = kind or "rule", False
    return out


def propose(text):
    return next((cat for cat, rx in PROPOSE.items() if rx.search(text)), "other")


def front_matter(text):
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    return m.group(1) if m else None


def skill(model, bench, method, labels):
    """Coded units, added structure lines by kind and the size against the seed of one learned skill."""
    text = S.skill(model, bench, method)
    seed, lines = SEED_TEXT.splitlines(), text.splitlines()
    ops = difflib.SequenceMatcher(None, seed, lines, autojunk=False).get_opcodes()
    added = {j for t, _, _, j1, j2 in ops if t in ("insert", "replace") for j in range(j1, j2)}
    coded, structure, missing = [], collections.Counter(), []
    for u in units(text):
        if added.intersection(u["lines"]) and u["kind"] != "rule":
            structure[u["kind"]] += 1
        elif added.intersection(u["lines"]):
            coded.append({"idx": len(coded), "text": u["text"], "proposal": propose(u["text"]), "category": labels.get(u["text"])})
            missing += [f"  {model} {method} {bench} unit {coded[-1]['idx']}, proposed {coded[-1]['proposal']}: {u['text']}"] if coded[-1]["category"] is None else []
    if missing:
        raise SystemExit(f"rule units without a label in {LABELS} (read each and add it):\n" + "\n".join(missing))
    lost = {tag: sum(i2 - i1 for t, i1, i2, _, _ in ops if t == tag) for tag in ("replace", "delete")}
    return {"model": model, "bench": bench, "method": method, "units": coded, "structure": dict(structure),
            "counts": dict(collections.Counter(u["category"] for u in coded)), "words": len(text.split()),
            "words_ratio": len(text.split()) / len(SEED_TEXT.split()), "lines": len(lines),
            "lines_added": sum(j2 - j1 for t, _, _, j1, j2 in ops if t in ("insert", "replace")),
            "lines_removed": lost["replace"] + lost["delete"], "replaced": lost["replace"], "deleted": lost["delete"],
            "keeps_front_matter": front_matter(text) == front_matter(SEED_TEXT)}


def total(rows, cat=None):
    return sum(r["counts"].get(cat, 0) for r in rows) if cat else sum(len(r["units"]) for r in rows)


def hand_changed(rows):
    return sum(u["proposal"] != u["category"] for r in rows for u in r["units"])


def category_rows(rows, label):
    return [[label(r)] + [r["counts"].get(c, 0) for c in CATS] + [len(r["units"])] for r in rows]


def skilltext(ds, every):
    name = lambda r: f"{S.ARM_NAME[r['method']]} {S.BENCH_SHORT[r['bench']]}"  # noqa: E731
    sizes = md_table(["Optimizer", "Bench", "words", "x seed", "lines", "lines added", "lines removed", "seed lines deleted outright",
                      "seed lines replaced in place"],
                     [[S.ARM_NAME[r["method"]], S.BENCH_SHORT[r["bench"]], r["words"], f"{r['words_ratio']:.2f}", r["lines"], r["lines_added"],
                       r["lines_removed"], r["deleted"], r["replaced"]] for r in ds])
    rows = category_rows(ds, name)
    rows += [[f"**{S.ARM_NAME[o]} total**"] + [total(g, c) for c in CATS] + [total(g)]
             for o in S.OPTIMIZERS if (g := [r for r in ds if r["method"] == o])]
    rows.append(["**total**"] + [total(ds, c) for c in CATS] + [total(ds)])
    structure = "; ".join(f"{r['method']} {r['bench']}: {sum(r['structure'].values())}" for r in ds if r["structure"])
    front = {g: [r for r in every if (r["method"] == "gepa") == (g == "GEPA")] for g in ("GEPA", "SkillOpt and SkillAdam")}
    keep = {g: sum(r["keeps_front_matter"] for r in rs) for g, rs in front.items()}
    numbers = "\n\n".join([
        f"The {len(ds)} learned skills of {S.MODEL_NAME[S.DEEPSEEK]} on {', '.join(BENCH_NAME[b] for b in BENCHES)}. Size against the seed "
        f"({len(SEED_TEXT.split())} words, {len(SEED_TEXT.splitlines())} lines; whole-line diff):", sizes,
        f"Added rules by category: {total(ds)} units, {hand_changed(ds)} keyword proposals changed by hand. Added structure lines, not "
        f"coded: {structure}.", md_table(["Skill", *CATS, "units"], rows),
        f"Front matter (the seed's name and description), over the learned skills of both models: SkillOpt and SkillAdam keep it in "
        f"{keep['SkillOpt and SkillAdam']} of their {len(front['SkillOpt and SkillAdam'])} skills, GEPA in {keep['GEPA']} of its "
        f"{len(front['GEPA'])}."])
    record(__spec__.name, "skilltext", numbers, S.runs_text([(r["model"], r["bench"], r["method"]) for r in ds]),
           what=f"What GEPA, SkillOpt and SkillAdam add to the seed skill under {S.MODEL_NAME[S.DEEPSEEK]}, coded into {len(CATS)} rule categories.")


def rules_luna(luna, ds):
    edges = {}
    for model in (S.DEEPSEEK, S.LUNA):
        rows = L.lengths(model)
        kept, apart = [r for r in rows if not L.excluded(r)], [r for r in rows if L.excluded(r)]
        edges[model] = (kept, sum(r["chars_c"] > r["chars_p"] for r in kept), apart, sum(r["chars_c"] > r["chars_p"] for r in apart))
    rows = [[S.ARM_NAME[r["method"]], BENCH_LABEL[r["bench"]]] + [r["counts"].get(c, 0) for c in CATS]
            + [len(r["units"]), hand_changed([r]), r["words"], f"{r['words_ratio']:.2f}"] for r in luna]
    rows.append(["**total**", ""] + [f"**{total(luna, c)}**" for c in CATS] + [f"**{total(luna)}**", "", "", ""])
    n, dn = total(luna), total(ds)
    share = lambda k, m: f"{k}/{m} = {k / m:.0%}"  # noqa: E731
    seed_tb2 = [S.ARM_NAME[o] for m, b, o in returned(S.LUNA) if b == "terminal-bench-2-1"]
    lines = [f"The {len(luna)} learned skills of {S.MODEL_NAME[S.LUNA]}" + (f"; on Terminal-Bench 2.1 {', '.join(seed_tb2)} return the seed." if seed_tb2 else "."),
             "", md_table(["optimizer", "benchmark", *CATS, "units", "changed by hand", "words", "words / seed"], rows), "",
             f"verify share: {share(total(luna, 'verify'), n)} (DeepSeek: {share(total(ds, 'verify'), dn)}). "
             + ". ".join(f"{c}: {total(luna, c)} (DeepSeek {total(ds, c)})" for c in ("serialize", "parallelize", "limit_read")) + ".", ""]
    lines += [f"- parallelize unit: `{u['text']}`" for r in luna for u in r["units"] if u["category"] == "parallelize"]
    lines += ["", "Seed text removed (lines replaced in place / deleted outright): "
              + "; ".join(f"{r['method']}|{r['bench']}: {r['replaced']}/{r['deleted']}" for r in luna), "",
              "Dev edges that lengthen the skill (parent and child share a dev panel; SkillAdam on Terminal-Bench 2.1 is counted apart, "
              "its parents being ambiguous):", "",
              md_table(["model", "dev-scored edges", "lengthen", "SkillAdam TB2 (excluded): lengthen / edges"],
                       [[m, len(k), lk, f"{la} / {len(a)}"] for m, (k, lk, a, la) in edges.items()]), ""]
    lines += [f"- {m} shorter edge: {r['opt']} {r['bench']} {r['parent']} -> {r['child']}, {r['chars_p']} -> {r['chars_c']} chars, "
              f"accepted={r['accepted']}" for m, (k, _, _, _) in edges.items() for r in k if r["chars_c"] <= r["chars_p"]]
    cells = sorted({(r["model"], r["bench"], r["method"]) for r in luna} | {(m, b, o) for m in (S.DEEPSEEK, S.LUNA) for b in BENCHES
                                                                            for o in L.METHODS if S.dev_gated(m, b, o)})
    record(__spec__.name, "rules_luna", "\n".join(lines), S.runs_text(cells),
           what=f"The rules {S.MODEL_NAME[S.LUNA]}'s optimizers add, coded as the DeepSeek ones, and how many dev-scored edits lengthen "
                "the skill under each model.")


def kappa(a, b):
    """(agreement, Cohen's kappa) of two label sequences."""
    cats = sorted(set(a) | set(b))
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        table[cats.index(x), cats.index(y)] += 1
    n = table.sum()
    observed, expected = np.trace(table) / n, (table.sum(0) * table.sum(1)).sum() / n ** 2
    return observed, 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def coder2(ds):
    final = {f"{r['method']}|{r['bench']}#{u['idx']}": u["category"] for r in ds for u in r["units"]}
    second = json.loads(CODER2.read_text())
    if set(final) != set(second):
        raise SystemExit(f"{CODER2} labels other units than the coded ones")
    ids = sorted(final)
    blind = [i for i in ids if not i.startswith(EXPOSED)]

    def agree(subset):
        a, b = [final[i] for i in subset], [second[i] for i in subset]
        groups = [kappa([x in g for x in a], [x in g for x in b]) for g in GROUPS.values()]
        return [f"{100 * kappa(a, b)[0]:.1f}%", f"{kappa(a, b)[1]:.3f}"] + [f"{100 * x:.1f}% / {k:.3f}" for x, k in groups]

    n = len(ids)
    count = lambda labels, c: sum(labels[i] == c for i in ids)  # noqa: E731
    numbers = "\n".join([
        md_table(["Units", "agreement", "kappa (8 categories)"] + [f"{g}: agreement / kappa (binary)" for g in GROUPS],
                 [[f"all {n} units"] + agree(ids), [f"{len(blind)} blind units (without gepa tau3)"] + agree(blind)]), "",
        "### Counts per category", "",
        md_table(["category", "final", "final %", "coder 2", "coder 2 %"],
                 [[c, count(final, c), f"{100 * count(final, c) / n:.1f}", count(second, c), f"{100 * count(second, c) / n:.1f}"] for c in CATS]), "",
        "### Units per relied-on group", "",
        md_table(["group", "units in final", "units in coder 2"],
                 [[g, sum(final[i] in m for i in ids), sum(second[i] in m for i in ids)] for g, m in GROUPS.items()]), "",
        "### Confusion matrix (rows: final labels, columns: coder 2)", "",
        md_table(["final \\ coder 2", *CATS], [[f] + [sum(final[i] == f and second[i] == s for i in ids) for s in CATS] for f in CATS]), "",
        f"The {n} unit texts went to coder 2 without labels. The labels of GEPA's tau3-bench skill ({n - len(blind)} units) were shown to "
        f"coder 2 before coding, so agreement (percent and Cohen's kappa between coder 2 and the final labels) is reported on all {n} units "
        f"and on the {len(blind)} blind units. Coder 2 is a model, not a human.",
        f"Units with different labels: {sum(final[i] != second[i] for i in ids)} of {n}."])
    record(__spec__.name, "coder2", numbers, S.runs_text([(r["model"], r["bench"], r["method"]) for r in ds]),
           what=f"Inter-coder agreement for the {len(CATS)}-category coding of the {n} added rule units.")


def returned(model):
    return [(m, b, o) for m, b, o in ((model, b, o) for b in BENCHES for o in S.OPTIMIZERS) if S.returned_seed(m, b, o)]


def release():
    """OUT/skills: the seed and every returned skill with a README table; each checked against the run's final/SKILL.md, its
    summary and the skill delivered at test, and scanned for paths, keys and addresses."""
    out, seed = layout.OUT / "skills", layout.SEED_FILE.read_bytes()
    (out / "seed").mkdir(parents=True, exist_ok=True)
    (out / "seed/SKILL.md").write_bytes(seed)
    rows, problems = [("—", "—", "[Seed (hand-written)](seed/SKILL.md)", len(seed.decode().split()), "skills/seed", "yes (every run's seed)")], []
    for model in (S.DEEPSEEK, S.LUNA):
        for bench in BENCHES:
            for method in S.OPTIMIZERS:
                run, folder = S.run(model, bench, method), S.MODEL_NAME[model].lower()
                final = (run / "final/SKILL.md").read_bytes()
                if final.decode().strip() != S.skill(model, bench, method).strip():
                    problems.append(f"{model} {method} {bench}: final/SKILL.md != summary.final_skill")
                if m := SUSPECT.search(final.decode()):
                    problems.append(f"{model} {method} {bench}: suspicious text {m.group(0)!r}")
                tested = {p.read_bytes() for p in (run / "results/skills/deliveries/evaluations/test").glob("*/native/SKILL.md")}
                check = "yes" if final in tested else f"NO ({len(tested)} test deliveries)"
                path = "seed/SKILL.md" if final == seed else f"{folder}/{RELEASE[bench]}/{method}/SKILL.md"
                if final != seed:
                    (out / path).parent.mkdir(parents=True, exist_ok=True)
                    (out / path).write_bytes(final)
                rows.append((folder, RELEASE[bench], f"[{S.ARM_NAME[method]}]({path})", len(final.decode().split()), run.name,
                             check + ("; returned the seed, see seed/" if final == seed else "")))
    seed_cells = collections.defaultdict(list)
    for m, b, o in returned(S.DEEPSEEK) + returned(S.LUNA):
        seed_cells[(m, b)].append(S.ARM_NAME[o])
    every = [(m, b, o, S.skill(m, b, o)) for m, b, o in S.learned()]
    keep = {o: [front_matter(t) == front_matter(SEED_TEXT) for _, _, x, t in every if x == o] for o in S.OPTIMIZERS}
    front = ("GEPA rewrites the whole skill, including its `name` and `description` frontmatter; SkillOpt and SkillAdam edit the body "
             "and keep the seed's name and description." if not any(keep["gepa"]) and all(keep["skillopt"] + keep["skilladam"]) else
             "Skills that keep the seed's name and description: " + ", ".join(f"{S.ARM_NAME[o]} {sum(k)} of {len(k)}" for o, k in keep.items()) + ".")
    who = lambda names: "all three optimizers return" if len(names) == 3 else " and ".join(names) + (" returns" if len(names) == 1 else " return")  # noqa: E731
    seeds = " ".join(f"With {S.MODEL_NAME[m]} on {BENCH_NAME[b]} {who(names)} the seed unchanged (no dev-set gain separated a candidate "
                     "from it), so those cells point to `seed/` instead of a copy." for (m, b), names in seed_cells.items())
    text = ["# Skills", "",
            "The hand-written seed skill and the skill each optimizer returned in the paper's runs (OpenCode harness; three optimizers x three "
            "benchmarks x two agent models). Each `SKILL.md` is byte-identical to the run's `final/SKILL.md` and to the skill delivered to "
            "the agent in the test evaluations (checked against the delivered `native/SKILL.md`).", "", f"{front} {seeds}".strip(), "",
            "| model | benchmark | optimizer | words | source run tag | delivered at test |", "|---|---|---|---|---|---|"]
    text += [f"| {' | '.join(str(c) for c in r)} |" for r in rows]
    text += ["", "Words are whitespace-separated tokens of the whole file (frontmatter included).", ""]
    (out / "README.md").write_text("\n".join(text))
    print("released skills:", out, "problems:", problems or "none")


def main():
    labels = json.loads(LABELS.read_text())
    order = lambda r: (S.OPTIMIZERS.index(r["method"]), BENCHES.index(r["bench"]))  # noqa: E731
    every = [skill(m, b, o, labels[m]) for m, b, o in S.learned()]
    ds = sorted([r for r in every if r["model"] == S.DEEPSEEK], key=order)
    luna = sorted([r for r in every if r["model"] == S.LUNA], key=order)
    (S.cache("edits") / "skilltext.json").write_text(json.dumps(every, indent=1, ensure_ascii=False))
    skilltext(ds, every)
    rules_luna(luna, ds)
    coder2(ds)
    release()


if __name__ == "__main__":
    main()
