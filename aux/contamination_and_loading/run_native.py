"""Delivery-paradigm check: evaluate a single-file skill bundle through opencode's native --skill route.

Usage: python -m aux.contamination_and_loading.run_native <bundle_dir> <out_root> <cache_root>   (calls the model; needs the SkillZip runtime
`baselines.utils.skillzip_runtime`, which is not part of this repository and is imported only when the script runs)
Paradigm B = progressive disclosure (name/description resident, body loaded via the skill tool),
with the same explicit activation directive as the SkillZip runs. Counterpart of the default
guidance-flattening route (paradigm A) used by harness_bench.eval run --skill.
"""

import json
import os
import sys
from pathlib import Path


def main():
    os.environ.setdefault("HARNESS_PROTOCOL", "native-v1")
    bundle, out, cache = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    from baselines.utils import skillzip_runtime
    from harness_bench.eval import panels
    from src.executor import Cell

    cell = Cell(benchmark="tau3-bench", model="deepseek-flash", harness="opencode",
                root=out, cache_root=cache, slots=8)
    task_ids = [r["task_id"] for r in panels.registry_panel("tau3-bench", "test")]
    rows = skillzip_runtime.evaluate(cell, bundle, task_ids, "test")
    known = [r for r in rows if r["success"] is not None]
    summary = {"bundle": bundle, "delivery": "native-skill", "tasks": len(rows), "known": len(known),
               "Q_known": sum(r["success"] for r in known) / max(1, len(known)),
               "cost": sum(r["cost"] or 0 for r in rows),
               "evaluation_path": known[0]["evaluation_path"] if known else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "delivery-check-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
