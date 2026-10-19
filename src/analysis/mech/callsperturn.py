"""Paper 1 (mechanism paper): how many tool calls a turn carries, seed against each learned skill (the m in the turn-splitting term).

Role in the argument: attribution.py shows that turn-splitting (a lower calls-per-turn m = N / R, hence more requests R that each re-send the whole
prompt) is the largest term on AutoBench. This module opens m up: is the drop in m a move from turns with several calls to turns with one call, and
are the single calls read-only ones that could have been batched into one turn?

Per benchmark and cell (skill, benchmark), pooled over the three runs of each side, on attribution.py's task set (Table 1's common tasks with a
usable graph episode in all six runs; the set is rebuilt here with the same rules and its size is checked against attribution.json):
  - turns = the model requests of the episode (Episode.R); a turn's calls = the calls the harness really executed in it (Episode.calls)
  - share of turns with 0, 1 and 2+ executed calls; m = executed calls / turns (pooled over all turns; the mean of per-task m is given too)
  - share of turns that carry exactly one call and that call is read-only; the same as a share of the single-call turns
  - runs of >= 2 consecutive turns that each carry exactly one read-only call (candidates for batching), per episode: number of runs, turns inside
    runs, and turns saved if each run were one turn (sum of length - 1)

Read-only rule (the recorded arguments decide):
  - read, grep, glob, list / ls, webfetch, websearch                                  -> read-only
  - tau3 tools whose name starts get / find / search / list / check / lookup, KB_search, and a call_discoverable_agent_tool whose wrapped tool
    name does (the same vocabulary as graph.episode LOOKUP)                           -> read-only; every other tau3 tool (messages, conversation
    control, unlock, calculate, state changes) is not
  - bash -> read-only only if the command is a pure read: it parses (shlex), contains no command substitution, heredoc or redirect (except to
    /dev/null or 2>&1), and every pipeline stage / ; && || segment (leading cd / set / export / VAR=value setup segments dropped) starts with
    cat, head, tail, ls, grep, find (without -delete, -exec, -execdir, -ok, -fprint*, -fls) or wc, or sed with -n and without -i. Python,
    CLI tools (automationbench_cli.py), echo, awk, diff and everything unparsable are not pure reads.
  - edit, write, todowrite and the rest -> not read-only
A turn with several calls is never "a single read-only call", whatever the calls are.

No BACH, no model calls. Runs as attribution.py.
Commands (from bipa):           .venv/bin/python -m src.analysis.mech.callsperturn   -> assets/cache/analysis/mech/callsperturn.json, data/mech/mech_callsperturn.md
                                .venv/bin/python -m src.analysis.mech.callsperturn draw   (merged from calls_fig.py) -> fig/mech/mech_calls.{pdf,svg}, data/mech/mech_calls.md
"""
import argparse
import collections
import json
import re
import shlex

import numpy as np

from src.analysis import main as T
from src.analysis.mech import attribution as A
from src.utils import fig, logger, dataloader
from src.graph.episode import LOOKUP

READ_PROGRAMS = {"cat", "head", "tail", "ls", "grep", "find", "wc"}
# sensitivity variant "lenient": also accepts the harmless separators and filters agents put between reads (echo "---", sort, ...); the strict rule is the headline
LENIENT_EXTRA = frozenset({"echo", "printf", "sort", "uniq", "cut", "tr", "pwd", "which", "stat", "file", "tree", "du", "diff"})
FIND_WRITE = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
READ_TOOLS = {"read", "grep", "glob", "list", "ls", "webfetch", "websearch"}
SETUP = re.compile(r"^(cd|set|export)$|^\w+=")
SEPARATORS = {";", "&&", "||", "|", "&", ";;"}
SHORT = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}


def bash_pure_read(command, extra=frozenset()):
    """True if the shell command is a pure read by the rule in the module docstring; False for anything else, including unparsable commands."""
    # command substitution and heredocs hide what is really run
    if "`" in command or "$(" in command or "<<" in command:
        return False
    segments = []
    # each physical line is parsed on its own; a quote spanning lines fails the parse and the command counts as not read-only
    for line in command.splitlines():
        if not line.strip():
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            return False
        current = []
        for position, token in enumerate(tokens):
            if token in SEPARATORS:
                segments.append(current)
                current = []
            elif set(token) <= set("<>&|;()") and token not in SEPARATORS:
                # a redirect operator: only to /dev/null or a file descriptor (2>&1) is harmless, input redirects and everything else are rejected
                target = tokens[position + 1] if position + 1 < len(tokens) else ""
                if token.startswith("<") or not (token.endswith("&") and target.isdigit() or target == "/dev/null"):
                    return False
                current.append(token)
            else:
                current.append(token)
        segments.append(current)
    # drop the setup segments (cd dir, set -e, export X, VAR=value) and the redirect targets that were kept as plain words
    program_segments = [s for s in segments if s and not SETUP.match(s[0])]
    if not program_segments:
        return False
    for seg in program_segments:
        program, rest = seg[0], seg[1:]
        if program in READ_PROGRAMS or program in extra:
            if program == "find" and FIND_WRITE & set(rest):
                return False
        elif program == "sed":
            flags = [t for t in rest if t.startswith("-") and not t.startswith("--")]
            if not any("n" in f[1:] for f in flags) or any("i" in f[1:] for f in flags) or "--in-place" in rest:
                return False
        else:
            return False
    return True


def bash_command(arguments):
    """The command string of a bash call's recorded arguments; "" if the arguments are truncated or not JSON (such a call is never a pure read)."""
    try:
        return str(json.loads(arguments).get("command", ""))
    except (ValueError, AttributeError):
        return ""


def read_only(tool, arguments, extra=frozenset()):
    """True if one executed call is read-only by the module-docstring rule; tool and arguments as recorded in the request."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    if tool.startswith("tau3-runtime_"):
        name = tool.split("_", 1)[1]
        if name == "KB_search":
            return True
        if name == "call_discoverable_agent_tool":
            return bool(LOOKUP.match(str(args.get("agent_tool_name", ""))))
        return bool(LOOKUP.match(name))
    if tool == "bash":
        return bash_pure_read(bash_command(arguments), extra)
    return tool in READ_TOOLS


def turn_calls(raw_episode):
    """Per turn of one recorded episode (requests deduplicated by id as graph.episode does): the list of executed calls as (tool, arguments)."""
    seen, turns = set(), []
    for request in raw_episode["requests"]:
        if request["request_id"] in seen:
            continue
        seen.add(request["request_id"])
        turns.append([(c["tool"], c.get("arguments") or "") for c in request.get("tool_calls", []) if c.get("executed") is not False])
    return turns


def usable_tasks(own_files, seed_files, own_eps, seed_eps, own_rows, seed_rows):
    """The task list attribution.compute uses for one cell: Table 1's common tasks with outcome, N, X and a usable graph episode in all six runs and a positive 3-run history."""
    reps = sorted(own_files)
    _, own_mean = T.mean_rows(own_files)
    _, seed_mean = T.mean_rows(seed_files)
    common = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
    usable = []
    for t in common:
        rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                      for rows in (own_rows, seed_rows) for r in reps)
        if not rows_ok:
            continue
        if not all(t in eps[r] for eps in (own_eps, seed_eps) for r in reps):
            continue
        if not all(eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
            continue
        if A.mean_metrics([seed_eps[r][t][1] for r in reps]) is None or A.mean_metrics([own_eps[r][t][1] for r in reps]) is None:
            continue
        usable.append(t)
    return usable


def side_stats(files, tasks, episodes, extra=frozenset()):
    """Pooled statistics of one side (seed or skill) of one cell over its three runs and the given tasks."""
    turn_counts = collections.Counter()          # executed calls in a turn: 0, 1, 2 (= 2+)
    calls = turns = single = single_ro = 0
    per_task_m = collections.defaultdict(list)
    run_count, run_turns, run_saved, run_lengths = [], [], [], []
    bash_total = bash_pure = 0
    non_pure_programs = collections.Counter()
    n_episodes = 0
    for rep, path in sorted(files.items()):
        raw = {e["task_id"]: e for e in json.loads(open(path).read())["episodes"] if e["requests"]}
        for task in tasks:
            graph_episode = episodes[rep][task][0]
            per_turn = turn_calls(raw[task])
            # the raw record and the graph episode must describe the same turns and calls
            assert len(per_turn) == graph_episode.R and sum(len(t) for t in per_turn) == graph_episode.N, (path, task)
            n_episodes += 1
            flags = []
            for turn in per_turn:
                turn_counts[min(len(turn), 2)] += 1
                if len(turn) == 1:
                    single += 1
                    ro = read_only(*turn[0], extra)
                    single_ro += ro
                    flags.append(ro)
                else:
                    flags.append(False)
                for tool, arguments in turn:
                    if tool == "bash":
                        bash_total += 1
                        command = bash_command(arguments)
                        if bash_pure_read(command, extra):
                            bash_pure += 1
                        else:
                            words = command.split()
                            non_pure_programs[" ".join(words[:2]) if words and words[0] in ("python", "python3", "uv") else (words[0] if words else "")] += 1
            turns += len(per_turn)
            calls += sum(len(t) for t in per_turn)
            per_task_m[task].append(sum(len(t) for t in per_turn) / len(per_turn))
            # runs of >= 2 consecutive turns that each carry one read-only call
            lengths, length = [], 0
            for flag in flags + [False]:
                if flag:
                    length += 1
                else:
                    if length >= 2:
                        lengths.append(length)
                    length = 0
            run_count.append(len(lengths))
            run_turns.append(sum(lengths))
            run_saved.append(sum(x - 1 for x in lengths))
            run_lengths.extend(lengths)
    return dict(
        episodes=n_episodes, turns=turns, calls=calls, m=calls / turns, m_task_mean=float(np.mean([np.mean(v) for v in per_task_m.values()])),
        share0=turn_counts[0] / turns, share1=turn_counts[1] / turns, share2p=turn_counts[2] / turns,
        single_readonly_of_turns=single_ro / turns, single_readonly_of_single=single_ro / single if single else float("nan"), single_share=single / turns,
        runs_per_episode=float(np.mean(run_count)), run_turns_per_episode=float(np.mean(run_turns)), run_turns_share=sum(run_turns) / turns,
        saved_per_episode=float(np.mean(run_saved)), mean_run_length=float(np.mean(run_lengths)) if run_lengths else float("nan"),
        bash_calls=bash_total, bash_pure_read_share=bash_pure / bash_total if bash_total else float("nan"),
        non_pure_bash_top=non_pure_programs.most_common(6))


def compute():
    """All nine cells: seed and skill statistics on the cell's attribution task set."""
    attribution = {(c["method"], c["bench"]): c["n"] for c in json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"]}
    cells = []
    for method_disp, method in A.OPT:
        for bench, short in A.BENCH:
            own_files, seed_files = A.run_files(method, bench)
            own_eps, own_rows = A.load_runs(own_files)
            seed_eps, seed_rows = A.load_runs(seed_files)
            tasks = usable_tasks(own_files, seed_files, own_eps, seed_eps, own_rows, seed_rows)
            # same task set as attribution.json
            assert len(tasks) == attribution[(method_disp, bench)], (method_disp, bench, len(tasks))
            cell = dict(method=method_disp, bench=bench, short=short, n=len(tasks),
                        seed=side_stats(seed_files, tasks, seed_eps), skill=side_stats(own_files, tasks, own_eps),
                        seed_lenient=side_stats(seed_files, tasks, seed_eps, LENIENT_EXTRA), skill_lenient=side_stats(own_files, tasks, own_eps, LENIENT_EXTRA))
            cells.append(cell)
            print(method_disp, short, len(tasks), "m seed/skill", round(cell["seed"]["m"], 3), round(cell["skill"]["m"], 3))
    dataloader.write_json("callsperturn", dict(model=A.MODEL, cells=cells))
    return cells


def pct(x):
    return f"{100 * x:.1f}"


def record(cells):
    """paper/assets/data/mech/mech_callsperturn.md."""
    rows_mix, rows_ro, rows_runs, rows_bash = [], [], [], []
    for c in cells:
        for side in ("seed", "skill"):
            s = c[side]
            label = [c["method"], c["short"], c["n"], side]
            rows_mix.append(label + [s["turns"], pct(s["share0"]), pct(s["share1"]), pct(s["share2p"]), f"{s['m']:.3f}", f"{s['m_task_mean']:.3f}"])
            rows_ro.append(label + [pct(s["single_share"]), pct(s["single_readonly_of_turns"]), pct(s["single_readonly_of_single"])])
            rows_runs.append(label + [f"{s['runs_per_episode']:.2f}", f"{s['run_turns_per_episode']:.2f}", pct(s["run_turns_share"]), f"{s['saved_per_episode']:.2f}", f"{s['mean_run_length']:.2f}"])
            if c["bench"] != "tau3-bench":
                rows_bash.append(label + [s["bash_calls"], pct(s["bash_pure_read_share"]), "; ".join(f"{k} {v}" for k, v in s["non_pure_bash_top"][:4])])
    rows_len = []
    for c in cells:
        for side in ("seed", "skill"):
            s = c[side + "_lenient"]
            rows_len.append([c["method"], c["short"], c["n"], side, pct(s["single_readonly_of_turns"]), pct(s["single_readonly_of_single"]), f"{s['runs_per_episode']:.2f}", f"{s['saved_per_episode']:.2f}", pct(s["bash_pure_read_share"]) if c["bench"] != "tau3-bench" else "-"])
    delta = []
    for c in cells:
        a, b = c["seed"], c["skill"]
        delta.append([c["method"], c["short"], f"{b['m'] - a['m']:+.3f}", f"{100 * (b['share0'] - a['share0']):+.1f}", f"{100 * (b['share1'] - a['share1']):+.1f}", f"{100 * (b['share2p'] - a['share2p']):+.1f}",
                      f"{100 * (b['single_readonly_of_turns'] - a['single_readonly_of_turns']):+.1f}", f"{b['saved_per_episode'] - a['saved_per_episode']:+.2f}"])
    what = ("Opens up m = executed tool calls per turn, the quantity behind the turn-splitting term of mech_attribution3: the distribution of calls per turn, how many turns carry a single read-only call, "
            "and how many runs of consecutive single read-only turns (batching candidates) an episode has, for the seed and for each learned skill.")
    runs = "As mech_attribution3: skills native-20261008-https (rep 0) + test-reps rep 1/2; seed native-seed native-20261003-trace-final + rep-1, rep-2; DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x three benchmarks."
    agg = ("Per cell (skill, benchmark): attribution.py's task set (n tasks), the three runs of each side pooled; the seed row is the seed on that cell's tasks, so seed rows differ between the three skills of a benchmark. "
           "Turn = model request; calls = executed calls only (Episode.calls, checked equal to the raw record). m = pooled calls / turns (m_task = mean over tasks of the per-task m on 3-run pooled turns, the unit of the attribution term). "
           "Shares are of all turns, in %. Read-only rule: read, grep, glob, list/ls, webfetch, websearch; tau3 tools get/find/search/list/check/lookup*, KB_search, and discoverable calls wrapping them; bash only when the whole command "
           "is a pure read (cat, head, tail, ls, grep, find without -delete/-exec/-ok/-fprint, wc, sed -n without -i, joined by | ; && ||, no redirect except /dev/null and 2>&1, no substitution or heredoc, leading cd/set/export/VAR= dropped); "
           "python, CLI tools, echo, awk, diff, unparsable commands are not read-only. A multi-call turn is never a single read-only call. Runs = maximal sequences of >= 2 consecutive turns each with exactly one read-only call; "
           "'saved' = sum over runs of (length - 1), the turns removed if each run were one turn.")
    table = ("### A. Calls per turn\n\n" + logger.md_table(["Skill", "Bench", "n", "side", "turns", "0 calls %", "1 call %", "2+ calls %", "m", "m_task"], rows_mix)
             + "\n\n### B. Single read-only-call turns\n\n" + logger.md_table(["Skill", "Bench", "n", "side", "single-call turns % of turns", "single read-only % of turns", "read-only % of single-call turns"], rows_ro)
             + "\n\n### C. Runs of >= 2 consecutive single read-only turns (per episode)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "side", "runs / episode", "turns in runs / episode", "turns in runs % of turns", "turns saved / episode", "mean run length"], rows_runs)
             + "\n\n### D. Change skill minus seed (m; percentage points; turns saved per episode)\n\n"
             + logger.md_table(["Skill", "Bench", "d m", "d 0-call pp", "d 1-call pp", "d 2+-call pp", "d single read-only pp", "d saved / episode"], delta)
             + "\n\n### F. Sensitivity: lenient read-only rule (also echo, printf, sort, uniq, cut, tr, pwd, which, stat, file, tree, du, diff inside an otherwise pure-read command)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "side", "single read-only % of turns", "read-only % of single-call turns", "runs / episode", "turns saved / episode", "bash pure-read %"], rows_len)
             + "\n\n### E. Bash calls: share classified as pure reads, and the most frequent non-read commands\n\n"
             + logger.md_table(["Skill", "Bench", "n", "side", "bash calls", "pure-read %", "top non-read command heads"], rows_bash))
    logger.write_record("callsperturn", what, runs, agg, table, ".venv/bin/python -m src.analysis.mech.callsperturn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["draw"])
    if ap.parse_args().cmd == "draw":
        run_draw()
    else:
        record(compute())


# ---------------------------------------------------------------------------------------------------------------------------------
# mech_calls figure (merged from calls_fig.py; the `draw` subcommand)
#
# Share of AutoBench turns with 0, 1 and 2+ executed calls, seed vs learned skill, per optimizer (stacked 100% bars); calls per turn
# (seed -> skill) printed at the right. Reads assets/cache/analysis/mech/callsperturn.json (cell 'AutoB').
# ---------------------------------------------------------------------------------------------------------------------------------
FIG_OPTS = ["GEPA", "SkillOpt", "SkillAdam"]
COLORS = ["#D9D9D9", "#9ECAE1", "#2171B5"]
LABELS = ["0 calls", "1 call", "2+ calls"]


def draw_calls():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = {c["method"]: c for c in json.loads((dataloader.CACHE / "callsperturn.json").read_text())["cells"] if c["short"] == "AutoB"}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7})
    f, ax = plt.subplots(figsize=(4.0, 1.6))
    f.subplots_adjust(left=0.23, right=0.79, bottom=0.04, top=0.84)
    y, ys, numbers = 0.0, [], []
    for o in FIG_OPTS:
        ytop = y
        for kind, lab in (("seed", "seed"), ("skill", "skill")):
            c = cells[o][kind]
            sh = [c["share0"], c["share1"], c["share2p"]]
            left = 0.0
            for s, col in zip(sh, COLORS):
                ax.barh(y, s * 100, left=left, height=0.8, color=col, lw=0)
                if s >= 0.12:
                    ax.text(left + s * 50, y, f"{s * 100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if col == COLORS[2] else "black")
                left += s * 100
            ax.text(-1.5, y, lab, ha="right", va="center", fontsize=7)
            numbers.append((o, kind, round(sh[0] * 100, 1), round(sh[1] * 100, 1), round(sh[2] * 100, 1), round(c["m"], 2), c["turns"]))
            y += 1
        ax.text(102, ytop + 0.5, f"{cells[o]['seed']['m']:.2f} → {cells[o]['skill']['m']:.2f}", ha="left", va="center", fontsize=7)
        ax.text(-15, ytop + 0.5, o, ha="right", va="center", fontsize=7, fontweight="bold")
        y += 0.55
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.55, -0.6)
    ax.axis("off")
    ax.text(102, -0.6, "calls / turn", ha="left", va="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    f.legend(handles, LABELS, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.45, 1.0),
             handlelength=1.0, columnspacing=1.5)
    fig.save_figure(f, "calls")
    return numbers


def run_draw():
    numbers = draw_calls()
    table = logger.md_table(["optimizer", "skill", "0 calls %", "1 call %", "2+ calls %", "calls/turn", "turns"], numbers)
    logger.write_record("calls", "Share of AutoBench turns with 0, 1, 2+ executed calls, seed vs learned skill; calls per turn at the right.",
                   "deepseek-flash, AutoBench, each optimizer's own seed and learned-skill AutoBench runs (callsperturn.json).", "Pooled over episodes and turns.", table,
                   ".venv/bin/python -m src.analysis.mech.callsperturn draw")


if __name__ == "__main__":
    main()
