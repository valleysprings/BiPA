"""Cheatsheet (answer-leakage) audit of the learned skills: static text audit against the dev-task materials.

Question: does a learned skill's TEXT carry answer-key information about the
tasks (task ids, file paths, expected output values, command sequences — a cheatsheet), or is it generic procedure (pondering:
the gain, if any, comes from thinking/searching more)?

L0 — static text audit, per learned skill, against its benchmark's DEV panel task materials (the selection panel `sel` of the
frozen native-v1 protocol, 30 / 30 / 9 tasks for tau3 / AutoBench / TB2; identical across arms and models, verified here):
  corpora per dev task (harness-bench assets/benchmarks/<bench>/tasks/<id>/):
    prompt = instruction.md + task.toml (what the agent legitimately sees; tau3's instruction.md embeds the domain policy,
             so overlap with the prompt corpus is expected and is NOT leakage);
    answer = solution/** + tests/** (+ tau3 environment/runtime-server/task_config.json, which holds expected_actions) —
             the oracle and the checker; overlap with this corpus is the leakage signal. Binary files (.whl, .pyc, images)
             and undecodable files are skipped.
  metrics:
    (a) n-gram overlap: share of the skill's word 8-grams / 13-grams (lowercased alphanumerics) present in each corpus;
    (b) key-info hits, quoted for manual review: dev task ids named in the skill; path-like tokens that occur in the answer
        corpus (specific = in <= 3 dev tasks); numeric literals (>= 3 digits) that occur in the answer corpus (same);
        command-looking skill lines (>= 20 chars, CLI verbs / automationbench_cli.py) found verbatim in the answer corpus;
    (c) rule level: the 337 hand-coded added units of the nine DeepSeek skills (assets/cache/analysis/main/skilltext.json,
        aux/rule_coding/skilltext.py) get a heuristic task-specificity tag — a unit is tagged when it names a task id, or carries
        a path / value / 8-gram hit against the answer corpus; tagged units are printed verbatim with their evidence for the
        manual spot-check pass. Luna skills have no hand coding; they are audited at text level only.
  control: the seed skill against the same corpora (a generic text's background overlap).

No model calls, no benchmark execution. Command:
  python -m aux.contamination_and_loading.cheatsheet   -> assets/cache/analysis/main/cheatsheet.json, data/mech/mech_cheatsheet.md
"""
import json
import re

from src import layout
from src.utils.dataloader import SEED_TEXT, EX
from src.utils import logger, dataloader
HB = layout.HARNESS_BENCH  # task materials (see module docstring)
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
SHORT = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
OPT = ("gepa", "skillopt", "skilladam")
DS_RUN = "native-20261008-https"
LUNA_RUN = {"native": "native-20261011", "opt": "native-20261012"}
SKIP_SUFFIX = {".whl", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".so", ".zip", ".gz", ".pdf", ".ico", ".lock", ".wasm"}
MAX_FILE = 5_000_000
WORD = re.compile(r"[a-z0-9]+")
PATH_RE = re.compile(r"(?:^|(?<=[\s`\"'(=]))/?[\w.\-]+(?:/[\w.\-]+)+|[\w\-]+\.(?:py|sh|json|toml|md|txt|csv|ya?ml|sql|js|ts|go|rs|c|cc|cpp|h|hpp|html|css|xml|ini|cfg)\b")
VALUE_RE = re.compile(r"\b\d{3,}(?:\.\d+)?\b")
CMD_HINT = ("automationbench_cli", "python", "python3", "bash ", "curl ", "kubectl", "grep ", "cat >", "cd ", "./", "pytest", "make ", "pip ", "cargo ", "gcc", "g++")


def words(text):
    return WORD.findall(text.lower())


def ngrams(tok, n):
    return {tuple(tok[i:i + n]) for i in range(len(tok) - n + 1)}


def read_file(path):
    try:
        return path.read_text() if path.is_file() and path.stat().st_size <= MAX_FILE else ""
    except (UnicodeDecodeError, ValueError):
        return ""


def read_dir_text(root):
    """Concatenated text of the decodable non-binary files under a directory ('' when absent)."""
    if not root.is_dir():
        return ""
    parts = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() in SKIP_SUFFIX or f.stat().st_size > MAX_FILE:
            continue
        try:
            parts.append(f.read_text())
        except (UnicodeDecodeError, ValueError):
            continue
    return "\n".join(parts)


def load_bench(bench, dev_ids):
    """Per bench: dev corpora (prompt / answer), n-gram sets, per-task answer text, all task ids of the registry."""
    base = HB / "assets/benchmarks" / bench / "tasks"
    all_ids = sorted(p.name for p in base.iterdir() if p.is_dir())
    per_task, prompt_all, answer_all = {}, [], []
    for t in dev_ids:
        d = base / t
        prompt = "\n".join(x for x in (read_file(d / "instruction.md"), read_file(d / "task.toml")) if x)
        answer = read_dir_text(d / "solution") + "\n" + read_dir_text(d / "tests")
        if bench == "tau3-bench":  # the tau3 checker reads expected_actions from the runtime task config
            answer += "\n" + read_file(d / "environment/runtime-server/task_config.json")
        per_task[t] = {"prompt": prompt, "answer": answer}
        prompt_all.append(prompt)
        answer_all.append(answer)
    corp = {"prompt": "\n".join(prompt_all), "answer": "\n".join(answer_all)}
    sets = {kind: {n: ngrams(words(text), n) for n in (8, 13)} for kind, text in corp.items()}
    return {"ids": all_ids, "per_task": per_task, "text": corp, "ngrams": sets}


def final_skill_text(model, arm, bench):
    """(text, run name, returned_seed) of a run's final skill; None when the run is missing."""
    tag = DS_RUN if model == "deepseek-flash" else LUNA_RUN["opt"]
    run = EX / model / arm / bench / tag
    if not (run / "summary.json").is_file():
        return None
    text = json.loads((run / "summary.json").read_text())["final_skill"]
    return {"text": text, "run": run.name, "returned_seed": text == SEED_TEXT}


def dev_panel(bench):
    """The frozen selection panel (dev set) of the campaign; identical across arms and models (checked on both)."""
    panels = []
    for model, arm, tag in (("deepseek-flash", "skillopt", DS_RUN), ("gpt-6-luna", "skillopt", LUNA_RUN["opt"])):
        proto = json.loads((EX / model / arm / bench / tag / "native_protocol.json").read_text())
        panels.append([r if isinstance(r, str) else r["task_id"] for r in proto["panels"]["sel"]])
    assert panels[0] == panels[1], f"dev panel differs between models on {bench}"
    return panels[0]


def audit_text(skill_text, bench_data):
    """(a) + (b) of the module docstring for one skill text against one benchmark's dev corpora."""
    tok = words(skill_text)
    out = {"words": len(tok)}
    for kind in ("prompt", "answer"):
        for n in (5, 8, 13):
            mine = ngrams(tok, n)
            if n == 5:
                corpus = ngrams(words(bench_data["text"][kind]), n)
            else:
                corpus = bench_data["ngrams"][kind][n]
            hits = mine & corpus
            out[f"ov{n}_{kind}"] = {"share": len(hits) / len(mine) if mine else None, "hits": len(hits), "total": len(mine)}
    # task ids named in the skill
    low = skill_text.lower()
    out["task_ids"] = [t for t in bench_data["ids"] if re.search(r"(?<![\w-])" + re.escape(t.lower()) + r"(?![\w-])", low)]
    # path-like tokens and numeric literals of the skill found in the answer corpus; specific = in <= 3 dev tasks
    per_task_answer = {t: d["answer"].lower() for t, d in bench_data["per_task"].items()}
    answer_all = bench_data["text"]["answer"].lower()
    for label, cands in (("paths", sorted(set(PATH_RE.findall(skill_text)))), ("values", sorted(set(VALUE_RE.findall(skill_text))))):
        hits = []
        for c in cands:
            cl = c.lower().strip(".,:;)")
            if len(cl) < 4 or cl not in answer_all:
                continue
            where = [t for t, txt in per_task_answer.items() if cl in txt]
            hits.append({"text": c, "n_tasks": len(where), "tasks": where[:3]})
        out[label] = hits
        out[f"{label}_specific"] = [h for h in hits if h["n_tasks"] <= 3]
    # command-looking lines found verbatim in the answer corpus
    answer_lines = {ln.strip() for ln in answer_all.splitlines() if len(ln.strip()) >= 20}
    cmds = []
    for ln in skill_text.splitlines():
        s = ln.strip().strip("`")
        if len(s) >= 20 and any(h in s for h in CMD_HINT) and s.lower() in answer_lines:
            cmds.append(s)
    out["commands"] = cmds
    return out


def unit_hits(unit_text, bench_data, skill_audit):
    """Rule-level tag of one coded unit: task-specific evidence only (a generic environment convention shared by many
    dev tasks — a path like /workspace/world_state.json — is not task-specific and is reported separately)."""
    kinds, generic = [], []
    low = unit_text.lower()
    if any(re.search(r"(?<![\w-])" + re.escape(t.lower()) + r"(?![\w-])", low) for t in skill_audit["task_ids"]):
        kinds.append("task-id")
    for h in skill_audit["paths"]:
        if h["text"].lower() in low:
            (kinds if h["n_tasks"] <= 3 else generic).append(f"path:{h['text']}")
    for h in skill_audit["values"]:
        if h["text"] in unit_text:
            (kinds if h["n_tasks"] <= 3 else generic).append(f"value:{h['text']}")
    grams = ngrams(words(unit_text), 8)
    if grams & bench_data["ngrams"]["answer"][8]:
        kinds.append("8gram-answer")
    return kinds, generic


def compute():
    benches = {b: load_bench(b, dev_panel(b)) for b in BENCHES}
    skills = []
    for model, mshort in (("deepseek-flash", "DeepSeek"), ("gpt-6-luna", "Luna")):
        for bench in BENCHES:
            for arm in OPT:
                s = final_skill_text(model, arm, bench)
                if s is None:
                    continue
                skills.append({"model": model, "mshort": mshort, "bench": bench, "arm": arm, **s})
    for s in skills:
        if s["returned_seed"]:
            s["audit"] = None
            continue
        s["audit"] = audit_text(s["text"], benches[s["bench"]])
    seed_audit = {b: audit_text(SEED_TEXT, benches[b]) for b in BENCHES}
    # rule level on the 337 hand-coded units (DeepSeek nine)
    coded = json.loads((dataloader.CACHE / "skilltext.json").read_text())["coded_units"]
    rules = {}
    for key, units in coded.items():
        arm, bench = key.split("|")
        au = next(s["audit"] for s in skills if s["model"] == "deepseek-flash" and s["arm"] == arm and s["bench"] == bench)
        tagged = []
        for u in units:
            kinds, generic = unit_hits(u["text"], benches[bench], au)
            tagged.append({"idx": u["idx"], "category": u["category"], "text": u["text"], "hits": kinds, "generic": generic})
        rules[key] = {"n": len(tagged), "specific": [t for t in tagged if t["hits"]],
                      "generic": [t for t in tagged if t["generic"] and not t["hits"]],
                      "by_category": {c: sum(1 for t in tagged if t["category"] == c and t["hits"]) for c in {u["category"] for u in tagged}}}
    res = {"skills": [{k: v for k, v in s.items() if k != "text"} for s in skills], "seed": seed_audit, "rules": rules,
           "dev_sizes": {b: len(benches[b]["per_task"]) for b in BENCHES}}
    dataloader.write_json("cheatsheet", res)
    return res


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


def record(res):
    rows = []
    for s in res["skills"]:
        label = f"{s['mshort']} {s['arm']}"
        if s["returned_seed"]:
            rows.append([label, SHORT[s["bench"]], "returned the seed text — row = seed control", "", "", "", "", "", "", "", ""])
            continue
        a = s["audit"]
        rows.append([label, SHORT[s["bench"]], a["words"],
                     pct(a["ov5_prompt"]["share"]), pct(a["ov5_answer"]["share"]), pct(a["ov8_answer"]["share"]), pct(a["ov13_answer"]["share"]),
                     len(a["task_ids"]) or 0, f"{len(a['paths'])} ({len(a['paths_specific'])})", f"{len(a['values'])} ({len(a['values_specific'])})",
                     len(a["commands"]) or 0])
    for b in BENCHES:
        a = res["seed"][b]
        rows.append(["seed (control)", SHORT[b], a["words"],
                     pct(a["ov5_prompt"]["share"]), pct(a["ov5_answer"]["share"]), pct(a["ov8_answer"]["share"]), pct(a["ov13_answer"]["share"]),
                     len(a["task_ids"]) or 0, f"{len(a['paths'])} ({len(a['paths_specific'])})", f"{len(a['values'])} ({len(a['values_specific'])})",
                     len(a["commands"]) or 0])
    table_a = logger.md_table(["skill", "bench", "words", "5g overlap prompt", "5g overlap answer", "8g overlap answer", "13g overlap answer",
                           "dev task ids named", "path hits (specific)", "value hits (specific)", "verbatim command lines"], rows)
    # evidence quotes: every task-id hit, every specific path/value, every command line
    ev = []
    for s in res["skills"]:
        a = s["audit"]
        if not a:
            continue
        for t in a["task_ids"]:
            ev.append(f"- {s['mshort']} {s['arm']} {SHORT[s['bench']]}: names dev task id `{t}`")
        for h in a["paths_specific"]:
            ev.append(f"- {s['mshort']} {s['arm']} {SHORT[s['bench']]}: path `{h['text']}` in answer corpus of {h['n_tasks']} dev task(s) {h['tasks']}")
        for h in a["values_specific"]:
            ev.append(f"- {s['mshort']} {s['arm']} {SHORT[s['bench']]}: value `{h['text']}` in answer corpus of {h['n_tasks']} dev task(s) {h['tasks']}")
        for c in a["commands"]:
            ev.append(f"- {s['mshort']} {s['arm']} {SHORT[s['bench']]}: command line `{c[:120]}` verbatim in answer corpus")
    table_b = "\n".join(ev) if ev else "(no key-info hits)"
    rrows = []
    for key, r in sorted(res["rules"].items()):
        arm, bench = key.split("|")
        rrows.append([arm, SHORT[bench], r["n"], len(r["specific"]), len(r["generic"]),
                      "; ".join(f"{c}:{n}" for c, n in sorted(r["by_category"].items()) if n) or "-"])
    table_c = logger.md_table(["arm", "bench", "coded units", "task-specific (heuristic)", "generic-convention hits", "task-specific by behaviour category"], rrows)
    rex = []
    for key, r in sorted(res["rules"].items()):
        for t in r["specific"]:
            rex.append(f"- [{key}] ({t['category']}; hits: {', '.join(t['hits'])}) {t['text'][:220]}")
    table_cx = "\n".join(rex) if rex else "(no unit tagged task-specific)"
    rgx = []
    for key, r in sorted(res["rules"].items()):
        for t in r["generic"]:
            rgx.append(f"- [{key}] ({t['category']}; generic: {', '.join(t['generic'])}) {t['text'][:220]}")
    table_cg = "\n".join(rgx) if rgx else "(none)"
    what = ("Cheatsheet (answer-leakage) audit of the learned skills. L0: static overlap of each skill's text with its benchmark's dev-panel "
            "task materials (prompt corpus = instruction.md + task.toml, what the agent legitimately sees; answer corpus = solution/** + tests/** "
            "+ tau3 runtime task_config.json with expected_actions). Metrics: word 5-gram/8-gram/13-gram overlap shares, key-info hits (dev task "
            "ids, answer-corpus paths / numeric values / verbatim command lines; specific = occurring in <= 3 dev tasks, wider-shared ones are "
            "generic environment conventions and reported separately), and a heuristic task-specificity tag on the 337 hand-coded skill units. "
            "Control: the seed skill on the same corpora, then the verdict.")
    runs = ("Skills: DeepSeek nine = final_skill of the native-20261008-https optimizer runs (gepa/skillopt/skilladam x tau3/AutoBench/TB2); "
            "Luna = native-20261012 optimizer runs (tau3, AutoBench; TB2's three returned the seed text and are marked). Dev panel = the frozen "
            "native-v1 selection panel `sel` (30/30/9 tasks; identical between the DeepSeek and Luna protocols, checked). Task materials: "
            "harness-bench/assets/benchmarks/<bench>/tasks/<id>/. Seed: skills/seed/SKILL.md.")
    agg = ("Tokens = lowercased alphanumerics; overlap = share of the skill's n-grams present in the corpus. Specific path/value = occurs in the "
           "answer corpus of at most 3 dev tasks (out of 30/30/9). Binary/undecodable task files skipped. The rule-level tag is heuristic "
           "(task id, path, value or 8-gram hit against the answer corpus); every tagged unit is printed verbatim with its evidence for the "
           "manual spot-check.")
    table = ("### L0-a. n-gram overlap and key-info hits per skill (seed = control)\n\n" + table_a
             + "\n\n### L0-b. Key-info hit evidence (all hits quoted)\n\n" + table_b
             + "\n\n### L0-c. Rule level: task-specificity tag on the 337 coded units (DeepSeek nine)\n\n" + table_c
             + "\n\n### L0-c2. Tagged units, verbatim, for the manual spot-check\n\n" + table_cx
             + "\n\n### L0-c3. Generic-convention hits (shared by > 3 dev tasks — environment conventions, NOT task-specific)\n\n" + table_cg
             + "\n\n### Verdict\n\n" + verdict(res))
    logger.write_record("cheatsheet", what, runs, agg, table, "python -m aux.contamination_and_loading.cheatsheet")


def verdict(res):
    """One-paragraph verdict computed from the L0 numbers."""
    learned = [s for s in res["skills"] if s["audit"]]
    n_luna = sum(1 for s in learned if s["model"] == "gpt-6-luna")
    max_ov = max(s["audit"]["ov8_answer"]["share"] for s in learned)
    max_ov5 = max(s["audit"]["ov5_answer"]["share"] for s in learned)
    seed_max = max(res["seed"][b]["ov8_answer"]["share"] for b in BENCHES)
    ids = sum(len(s["audit"]["task_ids"]) for s in learned)
    spec = sum(len(s["audit"]["paths_specific"]) + len(s["audit"]["values_specific"]) + len(s["audit"]["commands"]) for s in learned)
    tagged = sum(len(r["specific"]) for r in res["rules"].values())
    units = sum(r["n"] for r in res["rules"].values())
    strength = "strong" if (ids or spec > 10 or max_ov > 0.02 or max_ov5 > 0.05) else "weak" if (spec or tagged or max_ov > 2 * seed_max) else "none"
    return ("**Cheatsheet tendency: %s** (heuristic thresholds, see the L0 tables). Over the %d learned skills (9 DeepSeek + %d Luna; Luna TB2's "
            "three optimizer runs returned the seed text and are not audited separately): the largest answer-corpus overlap of any skill is "
            "%s at 5-grams and %s at 8-grams (seed control max %s at 8-grams), dev task ids named in skill texts: %d, specific key-info hits "
            "(paths + values + verbatim commands): %d, rule units tagged task-specific: %d of %d; the audit finds %s answer leakage in the skill texts."
            % (strength.upper(), len(learned), n_luna, pct(max_ov5), pct(max_ov), pct(seed_max), ids, spec, tagged, units,
               {"none": "no", "weak": "only weak", "strong": "strong"}[strength]))


def main():
    record(compute())


if __name__ == "__main__":
    main()
