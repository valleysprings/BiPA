#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
set -a; [[ -f host.env ]] && source host.env; set +a
exec "${PYTHON:-python}" -m src.run "$@"
