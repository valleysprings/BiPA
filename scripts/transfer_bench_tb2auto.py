"""TB2 <-> AutoB cross-benchmark transfer scan, rep0 (subset driver of transfer_bench.py).

Only the two task-type-similar directions: skills trained on terminal-bench-2-1 evaluated on
automationbench's test panel, and the reverse. Seed reference rows for both benches are included
(they are cache hits of the Table 1 seed test rows). Default is a dry run; --execute calls the model.
Rep 1/2 completion is handled separately by bipa's repeat_transfer.py once rep0 lands.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("transfer_bench", HERE.parent / "transfer_bench.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

PAIR = {("terminal-bench-2-1", "automationbench"), ("automationbench", "terminal-bench-2-1")}
BENCHES = {"terminal-bench-2-1", "automationbench"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    slots = 16
    if "--slots" in argv:
        i = argv.index("--slots")
        slots = int(argv[i + 1])
    seeds, transfers, skipped = tb.build_cells(["deepseek-flash"], ["skillopt", "gepa", "skilladam"])
    transfers = [c for c in transfers if (c["train"], c["bench"]) in PAIR]
    seeds = [c for c in seeds if c["bench"] in BENCHES]
    for c in seeds + transfers:
        c["cached"] = tb.cached(c, slots)
    if "--execute" in argv:
        tb.execute(seeds, transfers, slots)
    else:
        tb.dry_run(seeds, transfers, skipped)


if __name__ == "__main__":
    main()
