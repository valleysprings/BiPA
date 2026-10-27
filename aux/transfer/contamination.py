"""Contamination check: no learned skill carries the answers of the dev tasks its optimizer selected on (record
mech_cheatsheet).

Each learned skill (settings.learned) is compared with the dev tasks of its benchmark, the same tasks in every pinned
optimizer run of both models. Per dev task, from the task materials in harness-bench: the prompt corpus is what the agent
sees (instruction.md and task.toml; on tau3 the instruction holds the domain policy, so overlap with it is expected), the
answer corpus the oracle and the checker (solution/ and tests/, and on tau3 the runtime task_config.json with the expected
actions). Binary and undecodable files are skipped.

Per skill: the share of its word n-grams (lowercased alphanumerics; five, eight and thirteen words) found in a corpus; the
dev task ids it names; the paths and the long numbers it shares with the answer corpus, specific when they occur in the
answers of only a few dev tasks (wider ones are environment conventions); and its command lines found verbatim in the
answers. The seed is the control. Every rule unit the editors added to the nine DeepSeek skills (the coded units of
aux/edits/data/rule_labels.json, found in each skill's text) is tagged task-specific when it names a dev task or carries a
specific path, a specific number or an eight-word run of the answers; every tagged unit is quoted for reading by hand. The
verdict's thresholds are heuristic: strong when a skill names a dev task, has more than ten specific hits, or shares more
than two percent of its eight-word runs (five percent of its five-word runs) with the answers; weak when there is any
specific hit or tagged unit, or more than twice the seed's eight-word share; none otherwise."""
import functools, json, re

from src.utils import layout
from src.analysis import settings as S
from src.utils.logger import md_table, record

TASKS = layout.HARNESS_BENCH / "assets/benchmarks"
LABELS = layout.ROOT / "aux/edits/data/rule_labels.json"
BINARY = {".whl", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".so", ".zip", ".gz", ".pdf", ".ico", ".lock", ".wasm"}
MAX_BYTES = 5_000_000
SPECIFIC = 3  # a path or number in the answers of at most this many dev tasks is specific to them
DIGITS = 3  # a number in a skill is matched against the answers from this many digits on
WORD = re.compile(r"[a-z0-9]+")
PATH = re.compile(r"(?:^|(?<=[\s`\"'(=]))/?[\w.\-]+(?:/[\w.\-]+)+"
                  r"|[\w\-]+\.(?:py|sh|json|toml|md|txt|csv|ya?ml|sql|js|ts|go|rs|c|cc|cpp|h|hpp|html|css|xml|ini|cfg)\b")
VALUE = re.compile(rf"\b\d{{{DIGITS},}}(?:\.\d+)?\b")
COMMAND = ("automationbench_cli", "python", "python3", "bash ", "curl ", "kubectl", "grep ", "cat >", "cd ", "./", "pytest",
           "make ", "pip ", "cargo ", "gcc", "g++")
GRAMS = (("prompt", 5), ("answer", 5), ("answer", 8), ("answer", 13))


def read(path):
    """A file's text; '' when missing, binary, too large or undecodable."""
    try:
        ok = path.is_file() and path.suffix.lower() not in BINARY and path.stat().st_size <= MAX_BYTES
        return path.read_text() if ok else ""
    except (UnicodeDecodeError, ValueError):
        return ""


def read_tree(root):
    return "\n".join(t for t in map(read, sorted(root.rglob("*"))) if t) if root.is_dir() else ""


def grams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def names(text, task):
    return re.search(r"(?<![\w-])" + re.escape(task.lower()) + r"(?![\w-])", text.lower())


def dev_tasks(bench):
    """The dev panel of a benchmark, the same in every pinned optimizer run of both models (checked)."""
    panels = {tuple(t if isinstance(t, str) else t["task_id"] for t in json.loads((S.run(m, bench, o) / "native_protocol.json")
                                                                                   .read_text())["panels"]["sel"])
              for m in S.MODELS for o in S.OPTIMIZERS}
    if len(panels) != 1:
        raise ValueError(f"{bench}: the pinned optimizer runs select on different dev panels")
    return panels.pop()


@functools.cache
def corpora(bench):
    """The prompt and answer corpora of a benchmark's dev tasks, their n-grams, each task's answers and every task id."""
    base = TASKS / bench / "tasks"
    prompt, answer = {}, {}
    for t in dev_tasks(bench):
        prompt[t] = "\n".join(x for x in (read(base / t / "instruction.md"), read(base / t / "task.toml")) if x)
        answer[t] = read_tree(base / t / "solution") + "\n" + read_tree(base / t / "tests")
        if bench == "tau3-bench":
            answer[t] += "\n" + read(base / t / "environment/runtime-server/task_config.json")
    text = {"prompt": "\n".join(prompt.values()), "answer": "\n".join(answer.values()).lower()}
    return dict(ids=sorted(p.name for p in base.iterdir() if p.is_dir()), tasks=len(prompt), text=text,
                answers={t: a.lower() for t, a in answer.items()},
                grams={(k, n): grams(WORD.findall(text[k].lower()), n) for k, n in GRAMS})


def audit(text, c):
    """Overlap shares, dev task ids named, answer paths and numbers ({text, tasks}) and verbatim command lines of one skill."""
    words = WORD.findall(text.lower())
    out = {"words": len(words), "ids": [t for t in c["ids"] if names(text, t)]}
    for k, n in GRAMS:
        mine = grams(words, n)
        out[k, n] = len(mine & c["grams"][k, n]) / len(mine) if mine else None
    for kind, found in (("paths", PATH.findall(text)), ("values", VALUE.findall(text))):
        keys = [(x, x.lower().strip(".,:;)")) for x in sorted(set(found))]
        out[kind] = [dict(text=x, tasks=[t for t, a in c["answers"].items() if key in a]) for x, key in keys
                     if len(key) >= 4 and key in c["text"]["answer"]]
    lines = {ln.strip() for ln in c["text"]["answer"].splitlines() if len(ln.strip()) >= 20}
    out["commands"] = [s for s in (ln.strip().strip("`") for ln in text.splitlines())
                       if len(s) >= 20 and any(h in s for h in COMMAND) and s.lower() in lines]
    return out


def specific(a, kind):
    return [h for h in a[kind] if len(h["tasks"]) <= SPECIFIC]


def tag(unit, c, a):
    """(task-specific evidence, environment conventions) of one rule unit, against its skill's audit."""
    hits, generic = ["task-id"] if any(names(unit, t) for t in a["ids"]) else [], []
    for kind, found in (("path", lambda h: h["text"].lower() in unit.lower()), ("value", lambda h: h["text"] in unit)):
        for h in filter(found, a[kind + "s"]):
            (hits if len(h["tasks"]) <= SPECIFIC else generic).append(f"{kind}:{h['text']}")
    if grams(WORD.findall(unit.lower()), 8) & c["grams"]["answer", 8]:
        hits.append("8gram-answer")
    return hits, generic


def units(model, bench, method):
    """[(unit, category)] the editors added to a skill: the coded units found in its text (whitespace ignored)."""
    flat = lambda s: " ".join(s.split())
    text = flat(S.skill(model, bench, method))
    return [(u, c) for u, c in json.loads(LABELS.read_text())[model].items() if flat(u) in text]


def main():
    learned = S.learned()
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    row = lambda a: [a["words"], *(pct(a[g]) for g in GRAMS), len(a["ids"]),
                     *(f"{len(a[k])} ({len(specific(a, k))})" for k in ("paths", "values")), len(a["commands"])]
    rows, audits, evidence = [], {}, []
    for model in S.MODELS:
        for bench in S.BENCHES:
            for o in S.OPTIMIZERS:
                name = f"{S.MODEL_SHORT[model]} {o}"
                if (model, bench, o) not in learned:
                    rows.append([name, S.BENCH_SHORT[bench], "returned the seed text — row = seed control"] + [""] * 8)
                    continue
                a = audits[model, bench, o] = audit(S.skill(model, bench, o), corpora(bench))
                rows.append([name, S.BENCH_SHORT[bench], *row(a)])
                where = f"{name} {S.BENCH_SHORT[bench]}"
                evidence += [f"- {where}: names dev task id `{t}`" for t in a["ids"]]
                evidence += [f"- {where}: {k[:-1]} `{h['text']}` in the answers of {len(h['tasks'])} dev task(s) {h['tasks'][:3]}"
                             for k in ("paths", "values") for h in specific(a, k)]
                evidence += [f"- {where}: command line `{s[:120]}` verbatim in the answers" for s in a["commands"]]
    seed = {b: audit(S.SEED_TEXT, corpora(b)) for b in S.BENCHES}
    rows += [["seed (control)", S.BENCH_SHORT[b], *row(seed[b])] for b in S.BENCHES]
    rule_rows, tagged, conventions = [], [], []
    for (model, bench, o), a in sorted(audits.items(), key=lambda kv: f"{kv[0][2]}|{kv[0][1]}"):
        if model != S.DEEPSEEK:
            continue
        coded = [(u, cat, *tag(u, corpora(bench), a)) for u, cat in units(model, bench, o)]
        hit = [x for x in coded if x[2]]
        by_cat = {cat: sum(x[1] == cat for x in hit) for cat in sorted({x[1] for x in coded})}
        rule_rows.append([o, S.BENCH_SHORT[bench], len(coded), len(hit), sum(bool(x[3]) and not x[2] for x in coded),
                          "; ".join(f"{k}:{v}" for k, v in by_cat.items() if v) or "-"])
        tagged += [f"- [{o}|{bench}] ({cat}; hits: {', '.join(h)}) {u[:220]}" for u, cat, h, _ in hit]
        conventions += [f"- [{o}|{bench}] ({cat}; generic: {', '.join(g)}) {u[:220]}" for u, cat, h, g in coded if g and not h]
    total = [sum(r[k] for r in rule_rows) for k in (2, 3, 4)]
    rule_rows.append(["all", "", *total, "-"])
    shown = list(audits.values())
    worst = {g: max(a[g] for a in shown) for g in GRAMS[1:3]}
    seed8 = max(seed[b][GRAMS[2]] for b in S.BENCHES)
    ids = sum(len(a["ids"]) for a in shown)
    hits = sum(len(specific(a, "paths")) + len(specific(a, "values")) + len(a["commands"]) for a in shown)
    strength = ("strong" if ids or hits > 10 or worst[GRAMS[2]] > 0.02 or worst[GRAMS[1]] > 0.05
                else "weak" if hits or total[1] or worst[GRAMS[2]] > 2 * seed8 else "none")
    count = {m: sum(k[0] == m for k in audits) for m in S.MODELS}
    numbers = "\n".join([
        "Dev tasks: " + ", ".join(f"{S.BENCH_SHORT[b]} {corpora(b)['tasks']}" for b in S.BENCHES)
        + f". A path or number ({DIGITS} or more digits) is specific when it occurs in the answers of at most {SPECIFIC} dev tasks.",
        "",
        md_table(["skill", "bench", "words", "5g overlap prompt", "5g overlap answer", "8g overlap answer", "13g overlap answer",
                  "dev task ids named", "path hits (specific)", "value hits (specific)", "verbatim command lines"], rows), "",
        "Specific hits, quoted:", "", "\n".join(evidence) or "(none)", "",
        "Rule units the editors added to the DeepSeek skills:", "",
        md_table(["arm", "bench", "coded units", "task-specific (heuristic)", "generic-convention hits",
                  "task-specific by behaviour category"], rule_rows), "",
        "Units tagged task-specific, quoted:", "", "\n".join(tagged) or "(none)", "",
        "Units sharing only an environment convention (in the answers of more than three dev tasks), quoted:", "",
        "\n".join(conventions) or "(none)", "",
        f"Verdict: cheatsheet tendency {strength.upper()}. Over the {len(audits)} learned skills ({count[S.DEEPSEEK]} DeepSeek + "
        f"{count[S.LUNA]} Luna) the largest share found in the answers is {pct(worst[GRAMS[1]])} of a skill's 5-grams and "
        f"{pct(worst[GRAMS[2]])} of its 8-grams (seed control at most {pct(seed8)} of its 8-grams); dev task ids named: {ids}; "
        f"specific hits (paths, values, command lines): {hits}; rule units tagged task-specific: {total[1]} of {total[0]}."])
    record(__spec__.name, "cheatsheet", numbers, S.runs_text(learned))


if __name__ == "__main__":
    main()
