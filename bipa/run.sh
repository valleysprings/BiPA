#!/usr/bin/env bash
set -euo pipefail
root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$root/../host.env" ]]; then
  source "$root/../host.env"
fi
export PYTHONPATH="$root/..${PYTHONPATH:+:$PYTHONPATH}"
exec "${BASELINE_PYTHON:-python}" "$root/launch.py" "$@"
