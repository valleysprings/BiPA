#!/usr/bin/env bash
# Run-evidence audits (offline, read-only on the original runs).
#   bash scripts/audit.sh native-run --run RUN_DIR [--output FILE]   completion certificate
#   bash scripts/audit.sh trace --audit-dir DIR                       trace integrity of selected test episodes
#   bash scripts/audit.sh skill-history [--output-dir DIR]            evidence-linked per-round skill history
set -euo pipefail
root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="${BASELINE_PYTHON:-$root/../bach-harness/.venv/bin/python}"
case "${1:-}" in
    native-run)    module=audit_native_run ;;
    trace)         module=audit_trace_integrity ;;
    skill-history) module=audit_skill_history ;;
    *) echo "usage: bash scripts/audit.sh native-run|trace|skill-history [args...]" >&2; exit 2 ;;
esac
shift
cd "$root"
exec "$PY" -m "src.dynamics.$module" "$@"
