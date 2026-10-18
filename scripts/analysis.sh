#!/usr/bin/env bash
# Offline composition of the paper's analyses: per RQ, compute first (venv python),
# then figures (system python3, matplotlib). No model calls anywhere in this script.
#
# Usage: bash scripts/analysis.sh rq1|rq2|rq3|rq4|paper|table1|all [more targets...]
#
# Cross-target cache dependencies (assets/cache/analysis/): rq3 needs rq2's
# appc/attr.json and mech/attribution.json; cheatsheet (rq3) also needs rq4's
# mech/dynamics.json and the executed transfer results; paper needs rq1-rq3.
# "all" runs rq1 rq2 rq4 rq3 paper in that order, which satisfies every constraint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${BASELINE_PYTHON:-$ROOT/bach-harness/.venv/bin/python}"
FIG="${FIG_PYTHON:-python3}"
CACHE="$ROOT/assets/cache/analysis"
export BACH_SUBMISSION_MANIFEST="${BACH_SUBMISSION_MANIFEST:-$CACHE/submission-20261008/protocol.json}"
cd "$ROOT/bipa"

table1() {  # Table 1 (both models) -> ../paper/assets/data/mech/mech_table1.md
  "$PY" -m src.analysis.table1
}

need_manifest() {  # rq1's nullmodel/paired/power are driven by the frozen submission manifest
  if [ ! -f "$BACH_SUBMISSION_MANIFEST" ]; then
    echo "submission manifest not found: $BACH_SUBMISSION_MANIFEST" >&2
    echo "freeze it first: $PY -m src.analysis.submission_protocol" >&2
    exit 1
  fi
}

rq1() {  # the success gains do not hold
  need_manifest
  "$PY" -m src.analysis.mechanism.nullmodel --out "$CACHE/nullmodel"
  "$PY" -m src.analysis.mechanism.paired --out "$CACHE/paired"
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
  "$FIG" -m src.analysis.mech.paired_fig
}

rq2() {  # every learned skill levies a token tax
  "$PY" -m src.analysis.mechanism.fig1 compute
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
  "$FIG" -m src.analysis.mechanism.appc draw
  "$FIG" -m src.analysis.mech.attribution draw
  "$FIG" -m src.analysis.mech.slopes draw
  "$FIG" -m src.analysis.mech.slopes_input_fig
  "$FIG" -m src.analysis.mech.failcost_fig
  "$FIG" -m src.analysis.mech.price_fig
  "$FIG" -m src.analysis.mech.compaction draw
  "$FIG" -m src.analysis.mech.optcost_fig
  "$FIG" -m src.analysis.mech.runs_fig
}

rq4() {  # the findings generalize (analysis only; the model-calling drivers are scripts/rq4_transfer.sh)
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

paper() {  # paper-wide pieces (needs rq1-rq3 caches and the appb/appc figures)
  "$PY" -m src.analysis.mech.teaser compute
  "$PY" -m src.analysis.mech.batching compute
  "$PY" -m src.analysis.mech.designev compute
  "$PY" -m src.analysis.mech.selrules compute
  "$FIG" -m src.analysis.mech.teaser draw
  "$FIG" -m src.analysis.mech.batching draw
  "$FIG" -m src.analysis.mech.designev draw
  "$FIG" -m src.analysis.mech.selrules draw
  "$FIG" -m src.analysis.mech.implications_fig
  "$FIG" -m src.analysis.mech.skills_fig
  "$FIG" -m src.analysis.mech.copies
}

usage() { sed -n '2,9p' "$0"; exit 1; }
[ $# -ge 1 ] || usage
for t in "$@"; do
  case "$t" in
    rq1|rq2|rq3|rq4|paper|table1) "$t" ;;
    all) rq1; rq2; rq4; rq3; paper ;;
    *) usage ;;
  esac
done
