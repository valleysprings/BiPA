#!/usr/bin/env bash
# RQ4 (generality) transfer drivers. Preview by default; the drivers' own --execute calls the model.
#   bash scripts/rq4_transfer.sh model   [--methods skillopt,gepa,skilladam] [--slots 16] [--execute]
#   bash scripts/rq4_transfer.sh bench   [--models deepseek-flash] [--methods ...] [--slots 16] [--execute]
#   bash scripts/rq4_transfer.sh tb2auto [--slots 16] [--execute]
set -euo pipefail
root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="${BASELINE_PYTHON:-$root/../bach-harness/.venv/bin/python}"
case "${1:-}" in
    model)   module=transfer_model ;;
    bench)   module=transfer_bench ;;
    tb2auto) module=transfer_bench_tb2auto ;;
    *) echo "usage: bash scripts/rq4_transfer.sh model|bench|tb2auto [args...]" >&2; exit 2 ;;
esac
shift
cd "$root"
exec "$PY" -m "src.dynamics.$module" "$@"
