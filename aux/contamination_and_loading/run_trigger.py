"""Skill-trigger check: on-demand delivery WITHOUT the activation instruction.

Usage: python -m aux.contamination_and_loading.run_trigger <bundle_dir> <benchmark> <out_root> [slots]
(calls the model; needs the SkillZip runtime `baselines.utils.skillzip_runtime`, which is not part of this repository
and is imported only when the script runs)

Same route as run_native.py (opencode native --skill, name/description resident, body loaded
through the skill tool), with two changes so that the trigger rate is a measurement and not a precondition:
  1. no activation directive: the guidance file is empty, so the task prompt carries no instruction to load the skill;
  2. no delivery gate: an episode in which the agent never loads the skill keeps its verdict (skillzip_runtime would
     withhold it as delivery-unverified and retry the slot). Whether and when the skill was loaded is recorded per
     episode under skill_delivery / skillzip_loading.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

INJECTION = "native-skill-on-demand-no-directive-v1"


def on_demand_executor(Z):
    """Z.NativeSkillExecutor without the activation directive and without the delivery gate."""
    from harness_bench.eval.slots import HarnessBench
    from harness_bench.eval.worker import write_json

    class OnDemandExecutor(Z.NativeSkillExecutor):
        def __init__(self, config, *, layout, bundle, runtime=None):
            super().__init__(config, layout=layout, bundle=bundle, runtime=runtime)
            self.activation = ""
            self.activation_sha256 = hashlib.sha256(b"").hexdigest()  # the runtime records the directive by its sha256

        def _slot_environment(self, slot, root):
            environment = super()._slot_environment(slot, root)
            empty = root / "no-directive.fragment"
            empty.parent.mkdir(parents=True, exist_ok=True)
            empty.write_text("", encoding="utf-8")
            environment["HARNESS_GUIDANCE_FILE"] = str(empty)  # empty file: guidance_arguments() adds no prompt template
            return environment

        def run(self, version, panel, split, output_dir, attempt=0):
            result = HarnessBench.run(self, version, panel, split, output_dir, attempt)
            for episode in result.get("episodes", []):
                audit = Z.audit_episode(episode, version["files"], None)
                episode["skillzip_loading"] = audit
                gate = episode.pop("delivery_gate", None) or {}
                if type(gate.get("withheld_success")) is bool:  # keep the verdict: loading is measured, not required
                    episode["success"] = gate["withheld_success"]
                    episode["outcome_valid"] = gate["withheld_outcome_valid"]
                    episode["outcome_reason"] = gate["withheld_outcome_reason"]
                first = next((e for e in audit["events"] if e["path"] == "SKILL.md"), None)
                delivery = episode.setdefault("skill_delivery", {})
                delivery.update(
                    verified=True, mode=INJECTION, triggered=first is not None,
                    first_trigger_request=first["request_index"] if first else None,
                    semantics=("on demand without an activation instruction: the skill's name and description are offered by "
                               "opencode's skill tool; whether the agent loads the body is measured (triggered), not required"),
                    files={e["path"]: [e["evidence_ref"]] for e in audit["events"]})
            episodes = result.get("episodes", [])
            if "slot_accounting" in result:
                result["slot_accounting"].update(unknown_outcomes=sum(type(e.get("success")) is not bool for e in episodes),
                                                 delivery_unverified=0, delivery_unverified_fraction=0.0)
            write_json(Path(output_dir) / "evaluation.json", result)
            return result

    return OnDemandExecutor


def evaluate(cell, bundle, task_ids, split):
    from baselines.utils import skillzip_runtime as Z
    from harness_bench.eval import run
    from harness_bench.eval.api import backend_config, layout as evaluation_layout
    from src import executor

    root = Path(bundle).expanduser().resolve()
    version = Z.bundle_version(root)
    settings = dict(executor.settings(cell))
    settings["injection"] = INJECTION
    config = backend_config(cell.benchmark, model=cell.model, harness=cell.harness, bench_root=settings["bench_root"],
                            timeout_seconds=settings["timeout_seconds"], injection=INJECTION,
                            infrastructure_retries=settings["infrastructure_retries"])
    native = on_demand_executor(Z)(config, layout=evaluation_layout(cell.root, cache_root=cell.cache_root), bundle=root,
                                   runtime={"slot_parallelism": cell.slots, "cache": "off"})
    return run(version, cell.benchmark, split, tasks=list(task_ids), executor=native, **settings)


def main():
    os.environ.setdefault("HARNESS_PROTOCOL", "native-v1")
    from harness_bench.eval import panels
    from src.executor import Cell

    bundle, bench, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    slots = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    cell = Cell(benchmark=bench, model="deepseek-flash", harness="opencode", root=out,
                cache_root=out / "cache", slots=slots)
    task_ids = [r["task_id"] for r in panels.registry_panel(bench, "test")]
    if os.environ.get("TRIGGER_LIMIT"):  # smoke runs only
        task_ids = task_ids[:int(os.environ["TRIGGER_LIMIT"])]
    result = evaluate(cell, bundle, task_ids, "test")
    eps = [result.episode(r["task_id"], r["repetition"]) for r in result.rows]
    known = [e for e in eps if type(e.get("success")) is bool]
    trig = [e for e in eps if (e.get("skill_delivery") or {}).get("triggered")]
    summary = {"bundle": bundle, "benchmark": bench, "delivery": INJECTION, "tasks": len(eps), "known": len(known),
               "Q_known": sum(e["success"] for e in known) / max(1, len(known)),
               "triggered": len(trig), "trigger_rate": len(trig) / max(1, len(eps)),
               "evaluation_path": str(result.path)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "trigger-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
