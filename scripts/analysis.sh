#!/usr/bin/env bash
# Composition of the paper's analyses: per target, compute first, then figures.
# Everything runs under the one venv interpreter (it has numpy/scipy/matplotlib).
# Only "transfer" can call the model, and only with --execute; every other target
# is offline.
#
# Usage:
#   bash scripts/analysis.sh rq1|rq2|rq3|rq4|mainfigs|table1|refresh|all
#   bash scripts/analysis.sh transfer model|bench|tb2auto [args...]   # dry-run unless --execute
#   bash scripts/analysis.sh audit native-run|trace|skill-history [args...]
#
# Cross-target cache dependencies (assets/cache/analysis/): rq2's paired figure
# needs rq1's stats.json; rq3 needs rq2's appc/attr.json and main/attribution.json;
# cheatsheet (rq3) also needs rq4's dynamics.json and the executed transfer results;
# mainfigs needs rq1-rq3. "all" runs rq1 rq2 rq4 rq3 mainfigs, satisfying every one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${BASELINE_PYTHON:-$ROOT/bach-harness/.venv/bin/python}"
CACHE="$ROOT/assets/cache/analysis"
export BACH_SUBMISSION_MANIFEST="${BACH_SUBMISSION_MANIFEST:-$CACHE/submission-20261008/protocol.json}"
cd "$ROOT/bipa"

usage() { sed -n '6,10p' "$0"; exit 1; }

need_manifest() {  # rq1's nullmodel/power are driven by the frozen submission manifest
  if [ ! -f "$BACH_SUBMISSION_MANIFEST" ]; then
    echo "submission manifest not found: $BACH_SUBMISSION_MANIFEST" >&2
    echo "freeze it first: $PY -m src.analysis.submission.protocol" >&2
    exit 1
  fi
}

table1() {  # Table 1 (both models) -> ../paper/assets/data/mech/mech_table1.md
  "$PY" -m src.analysis.main.table1
}

rq1() {  # the success gains do not hold
  need_manifest
  "$PY" -m src.analysis.aux.nullmodel --out "$CACHE/nullmodel"
  "$PY" -m src.analysis.aux.nullmodel power --out "$CACHE/power"
  "$PY" -m src.analysis.aux.appb compute
  "$PY" -m src.analysis.aux.stability pooled
  "$PY" -m src.analysis.aux.stability reproducibility
  "$PY" -m src.analysis.main.stats
  "$PY" -m src.analysis.main.selection compute
  "$PY" -m src.analysis.main.selection records
  "$PY" -m src.analysis.main.domains
  "$PY" -m src.analysis.main.reruns compute
  "$PY" -m src.analysis.aux.appb draw
  "$PY" -m src.analysis.main.selection draw
  "$PY" -m src.analysis.main.reruns draw
  "$PY" -m src.analysis.main.stats devtest
  "$PY" -m src.analysis.main.domains draw
}

rq2() {  # every learned skill levies a token tax (the paired figure needs rq1's stats.json)
  "$PY" -m src.analysis.aux.appc compute
  "$PY" -m src.analysis.aux.slopes checks
  "$PY" -m src.analysis.aux.slopes
  "$PY" -m src.analysis.aux.pricing
  "$PY" -m src.analysis.aux.slopes reruns
  "$PY" -m src.analysis.aux.optcost
  "$PY" -m src.analysis.aux.facts
  "$PY" -m src.analysis.main.attribution compute
  "$PY" -m src.analysis.main.slopes compute
  "$PY" -m src.analysis.main.failcost compute
  "$PY" -m src.analysis.main.failcost record
  "$PY" -m src.analysis.main.price
  "$PY" -m src.analysis.main.compaction compute
  "$PY" -m src.analysis.main.optcost
  "$PY" -m src.analysis.main.attribution draw
  "$PY" -m src.analysis.main.slopes draw
  "$PY" -m src.analysis.main.slopes draw-input
  "$PY" -m src.analysis.main.price draw
  "$PY" -m src.analysis.main.compaction draw
  "$PY" -m src.analysis.main.stats paired
}

rq4() {  # the findings generalize (offline analysis of the executed transfers)
  "$PY" -m src.analysis.main.transfer
  "$PY" -m src.analysis.main.dynamics
  "$PY" -m src.analysis.main.transfer draw
  "$PY" -m src.analysis.main.dynamics draw
}

rq3() {  # the optimizers accumulate procedure (needs rq2's caches; cheatsheet also rq4's)
  "$PY" -m src.analysis.main.edges build
  "$PY" -m src.analysis.main.edges
  "$PY" -m src.analysis.main.edges holm
  "$PY" -m src.analysis.main.skilltext compute
  "$PY" -m src.analysis.main.skilltext record
  "$PY" -m src.analysis.main.coder2
  "$PY" -m src.analysis.main.callsperturn
  "$PY" -m src.analysis.main.cheatsheet
  "$PY" -m src.analysis.main.skilltext rules
  "$PY" -m src.analysis.main.edges draw
  "$PY" -m src.analysis.main.callsperturn draw
}

mainfigs() {  # main-text figures: Fig. 1, Sec. 4.3, the Sec. 5 design evidence (needs rq1-rq3 caches)
  "$PY" -m src.analysis.main.teaser compute
  "$PY" -m src.analysis.main.design batching
  "$PY" -m src.analysis.main.design designev
  "$PY" -m src.analysis.main.design selrules
  "$PY" -m src.analysis.main.teaser draw
  "$PY" -m src.analysis.main.design draw
  "$PY" -m src.analysis.main.skilltext skills
}

refresh() {  # derived tables under assets/artifacts/results/ (skill export, main table, status)
  "$PY" -m src.dynamics.render_native_skill_history
  "$PY" -m src.dynamics.main_table
  "$PY" -m src.dynamics.weekly_status
}

[ $# -ge 1 ] || usage
t="$1"; shift
case "$t" in
  rq1|rq2|rq3|rq4|mainfigs|table1|refresh) "$t" ;;
  all) rq1; rq2; rq4; rq3; mainfigs ;;
  transfer)
    case "${1:-}" in
      model)   m=transfer_model ;;
      bench)   m=transfer_bench ;;
      tb2auto) m=transfer_bench_tb2auto ;;
      *)       echo "usage: bash scripts/analysis.sh transfer model|bench|tb2auto [args...]" >&2; exit 2 ;;
    esac
    shift; exec "$PY" -m "src.dynamics.$m" "$@" ;;
  audit)
    case "${1:-}" in
      native-run)    m=audit_native_run ;;
      trace)         m=audit_trace_integrity ;;
      skill-history) m=audit_skill_history ;;
      *)             echo "usage: bash scripts/analysis.sh audit native-run|trace|skill-history [args...]" >&2; exit 2 ;;
    esac
    shift; exec "$PY" -m "src.dynamics.$m" "$@" ;;
  *) usage ;;
esac
