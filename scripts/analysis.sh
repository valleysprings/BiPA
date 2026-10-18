#!/usr/bin/env bash
# Composition of the paper's analyses: per target, compute first (venv python),
# then figures (system python3, matplotlib). Only "transfer" can call the model,
# and only with --execute; every other target is offline.
#
# Usage:
#   bash scripts/analysis.sh rq1|rq2|rq3|rq4|mainfigs|table1|refresh|all
#   bash scripts/analysis.sh transfer model|bench|tb2auto [args...]   # dry-run unless --execute
#   bash scripts/analysis.sh audit native-run|trace|skill-history [args...]
#
# Cross-target cache dependencies (assets/cache/analysis/): rq2's paired_fig needs
# rq1's stats.json; rq3 needs rq2's appc/attr.json and mech/attribution.json;
# cheatsheet (rq3) also needs rq4's dynamics.json and the executed transfer results;
# mainfigs needs rq1-rq3. "all" runs rq1 rq2 rq4 rq3 mainfigs, satisfying every one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${BASELINE_PYTHON:-$ROOT/bach-harness/.venv/bin/python}"
FIG="${FIG_PYTHON:-python3}"
CACHE="$ROOT/assets/cache/analysis"
export BACH_SUBMISSION_MANIFEST="${BACH_SUBMISSION_MANIFEST:-$CACHE/submission-20261008/protocol.json}"
cd "$ROOT/bipa"

usage() { sed -n '5,10p' "$0"; exit 1; }

need_manifest() {  # rq1's nullmodel/power are driven by the frozen submission manifest
  if [ ! -f "$BACH_SUBMISSION_MANIFEST" ]; then
    echo "submission manifest not found: $BACH_SUBMISSION_MANIFEST" >&2
    echo "freeze it first: $PY -m src.analysis.submission_protocol" >&2
    exit 1
  fi
}

table1() {  # Table 1 (both models) -> ../paper/assets/data/mech/mech_table1.md
  "$PY" -m src.analysis.table1
}

rq1() {  # the success gains do not hold
  need_manifest
  "$PY" -m src.analysis.mechanism.nullmodel --out "$CACHE/nullmodel"
  "$PY" -m src.analysis.mechanism.power --out "$CACHE/power"
  "$PY" -m src.analysis.mechanism.appb compute
  "$PY" -m src.analysis.mechanism.pooled
  "$PY" -m src.analysis.mechanism.reproducibility
  "$PY" -m src.analysis.mech.stats
  "$PY" -m src.analysis.mech.selection compute
  "$PY" -m src.analysis.mech.selection records
  "$PY" -m src.analysis.mech.domains
  "$PY" -m src.analysis.mech.reruns compute
  "$FIG" -m src.analysis.mechanism.appb draw
  "$FIG" -m src.analysis.mech.selection draw
  "$FIG" -m src.analysis.mech.reruns draw
  "$FIG" -m src.analysis.mech.devsize_fig
  "$FIG" -m src.analysis.mech.devtest_fig
  "$FIG" -m src.analysis.mech.domains_fig
}

rq2() {  # every learned skill levies a token tax (paired_fig needs rq1's stats.json)
  "$PY" -m src.analysis.mechanism.appc compute
  "$PY" -m src.analysis.mechanism.slopes_checks
  "$PY" -m src.analysis.mechanism.slopes
  "$PY" -m src.analysis.mechanism.pricing
  "$PY" -m src.analysis.mechanism.reruns
  "$PY" -m src.analysis.mechanism.optcost
  "$PY" -m src.analysis.mechanism.facts
  "$PY" -m src.analysis.mech.attribution compute
  "$PY" -m src.analysis.mech.slopes compute
  "$PY" -m src.analysis.mech.failcost compute
  "$PY" -m src.analysis.mech.failcost record
  "$PY" -m src.analysis.mech.price
  "$PY" -m src.analysis.mech.compaction compute
  "$PY" -m src.analysis.mech.optcost
  "$FIG" -m src.analysis.mech.attribution draw
  "$FIG" -m src.analysis.mech.slopes draw
  "$FIG" -m src.analysis.mech.slopes_input_fig
  "$FIG" -m src.analysis.mech.price_fig
  "$FIG" -m src.analysis.mech.compaction draw
  "$FIG" -m src.analysis.mech.paired_fig
}

rq4() {  # the findings generalize (offline analysis of the executed transfers)
  "$PY" -m src.analysis.mech.transfer
  "$PY" -m src.analysis.mech.dynamics
  "$FIG" -m src.analysis.mech.transfer_fig
  "$FIG" -m src.analysis.mech.dynamics_fig
}

rq3() {  # the optimizers accumulate procedure (needs rq2's caches; cheatsheet also rq4's)
  "$PY" -m src.analysis.mech.edges_build
  "$PY" -m src.analysis.mech.edges
  "$PY" -m src.analysis.mech.edges_holm
  "$PY" -m src.analysis.mech.skilltext compute
  "$PY" -m src.analysis.mech.skilltext record
  "$PY" -m src.analysis.mech.coder2
  "$PY" -m src.analysis.mech.callsperturn
  "$PY" -m src.analysis.mech.cheatsheet
  "$FIG" -m src.analysis.mech.rules_fig
  "$FIG" -m src.analysis.mech.edges_fig
  "$FIG" -m src.analysis.mech.calls_fig
}

mainfigs() {  # main-text figures: Fig. 1, Sec. 4.3, the Sec. 5 design evidence (needs rq1-rq3 caches)
  "$PY" -m src.analysis.mech.teaser compute
  "$PY" -m src.analysis.mech.batching compute
  "$PY" -m src.analysis.mech.designev compute
  "$PY" -m src.analysis.mech.selrules compute
  "$FIG" -m src.analysis.mech.teaser draw
  "$FIG" -m src.analysis.mech.implications_fig
  "$FIG" -m src.analysis.mech.skills_fig
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
