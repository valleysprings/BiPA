#!/usr/bin/env bash
# scripts/run.sh ARGS...             python -m src.run ARGS..., with host.env sourced
# scripts/run.sh -m MODULE ARGS...   python -m MODULE ARGS..., with host.env sourced
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f host.env ]]; then
    set -a; source host.env; set +a
fi
module=src.run
if [[ "${1:-}" == -m ]]; then
    module="${2:?usage: scripts/run.sh -m MODULE [ARGS...]}"
    shift 2
fi
exec "${PYTHON:-python}" -m "$module" "$@"
