"""Live status of the comparison grid, read straight from the run directories (no manifest, no certificates).

One row per model x benchmark x arm. The current run of a cell is layout.current_run(). Prints a markdown block; with --weekly FILE
it replaces the block between the AUTO markers in that file."""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.layout import current_run

MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = (("tau3-bench", "τ³"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2"))
ARMS = (("none", "no-skill"), ("seed", "seed"), ("skillopt", "SkillOpt"), ("gepa", "GEPA"))
BEGIN, END = "<!-- BEGIN AUTO: status -->", "<!-- END AUTO: status -->"


def health(run):
    """Problem flags for a run without a summary: process gone, no new output for 45 min, LLM errors in the last hour."""
    import subprocess, time
    from datetime import datetime, timezone
    flags, now = [], time.time()
    alive = str(run) in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    if not alive:
        flags.append("⚠ 进程已退出、未完成")
    probes = [run / "launch.log", run / "llm/attempts.jsonl", run / "gepa/run_log.txt",
              *run.glob("logs/evaluations/*/*/*/worker.log"),  # per-task worker logs move while episodes run
              *run.glob("logs/evaluations/*/*/retry-*/*/worker.log")]
    newest = max((p.stat().st_mtime for p in probes if p.is_file()), default=0)
    if alive and now - newest > 45 * 60:
        flags.append(f"⚠ {int((now - newest) / 60)} 分钟无新输出")
    ledger = run / "llm/attempts.jsonl"
    if ledger.is_file():
        errors = 0
        for line in ledger.read_text(errors="ignore").splitlines()[-400:]:
            event = json.loads(line)
            if event.get("outcome") == "error" and now - datetime.fromisoformat(event["recorded_at"]).timestamp() < 3600:
                errors += 1
        if errors:
            flags.append(f"⚠ 近 1 小时 LLM 调用失败 {errors} 次")
    return ("；" + "；".join(flags)) if flags else ""


def progress(run, arm):
    """Optimizer step k/T or evaluated episodes, for a run without a summary."""
    if run is None:
        return "未启动"
    return _progress(run, arm) + health(run)


def _progress(run, arm):
    protocol = run / "native_protocol.json"
    from src.panels import batches  # the current per-arm cap (a resumed run keeps its first recorded schedule)
    total = len(batches(run.parent.name, "main", arm)) if arm in ("gepa", "skillopt", "skilladam") else None
    if arm == "gepa" and (run / "gepa/run_log.txt").is_file():
        steps = re.findall(r"^Iteration (\d+)", (run / "gepa/run_log.txt").read_text(errors="ignore"), re.M)
        return f"运行中 第 {max(map(int, steps), default=0)}/{total} 轮"
    if arm == "skillopt" and (run / "launch.log").is_file():
        steps = re.findall(r"\[STEP (\d+)/(\d+)\]", (run / "launch.log").read_text(errors="ignore"))
        return f"运行中 第 {steps[-1][0]}/{steps[-1][1]} 步" if steps else "运行中 评测 seed"
    summary = json.loads(protocol.read_text()) if protocol.is_file() else {}
    sizes = {"train": len(summary.get("panels", {}).get("train", [])), "dev": len(summary.get("panels", {}).get("sel", [])),
             "test": len(summary.get("panels", {}).get("test", []))}
    parts = []
    for split in (("train", "dev", "test") if arm == "seed" else ("test",)):
        done = sum(1 for f in run.glob(f"results/evaluations/{split}/*/backend/*/*/results.jsonl")
                   if "retry" not in str(f) for _ in f.open(errors="ignore"))
        parts.append(f"{split} {min(done, sizes[split]) if sizes[split] else done}/{sizes[split] or '?'}")
    return "运行中 " + "，".join(parts)


def cell_row(model, bench, label, arm, arm_label, seed_test):
    run = current_run(arm, model, bench)
    summary = json.loads((run / "summary.json").read_text()) if run and (run / "summary.json").is_file() else None
    if summary is None:
        return f"| {model} | {label} | {arm_label} | {progress(run, arm)} | — | — | — | {run.name if run else '—'} |"
    test = summary["test"]
    final, seed = test["final"], test.get("seed") or seed_test
    q = f"{100 * final['success']:.1f} ({final['known']}/{final['tasks']})" if final["success"] is not None else "—"
    dq = (f"{100 * (final['success'] - seed['success']):+.1f}" if seed and seed.get("success") is not None
          and final["success"] is not None and arm != "seed" else "—")
    cost = (f"{final['cost'] / seed['cost']:.2f}" if seed and seed.get("cost") and final.get("cost")
            and arm != "seed" else "—")
    return f"| {model} | {label} | {arm_label} | 完成 | {q} | {dq} | {cost} | {run.name} |"


def block():
    lines = [BEGIN, "", "| 模型 | 基准 | 方法 | 状态 | test Q%（已知/总数） | ΔQ vs seed（pp） | 成本比 vs seed | run |",
             "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for bench, label in BENCHES:
            seed_run = current_run("seed", model, bench)
            seed_test = (json.loads((seed_run / "summary.json").read_text())["test"]["final"]
                         if seed_run and (seed_run / "summary.json").is_file() else None)
            lines += [cell_row(model, bench, label, arm, arm_label, seed_test) for arm, arm_label in ARMS]
    lines += ["", "ΔQ 与成本比：优化器取同一 run 内 seed 与 final 在同一 test 面板上的结果；no-skill 对比同模型同基准的 seed run。"
              "成本比为平均成本 c=N+X/1000 之比（论文表用逐题几何平均）。Q 只计结果已知的题。", "", END]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", type=Path)
    args = parser.parse_args()
    text = block()
    if args.weekly is None:
        print(text)
        return
    doc = args.weekly.read_text()
    start, end = doc.index(BEGIN), doc.index(END) + len(END)
    args.weekly.write_text(doc[:start] + text + doc[end:])


if __name__ == "__main__":
    main()
