#!/usr/bin/env bash
# One offline refresh of results/: per-cell current-run skills, main table, weekly status. No API calls, no loop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${BASELINE_PYTHON:-$ROOT/bach-harness/.venv/bin/python}"
cd "$ROOT/bipa"
"$PY" -m src.dynamics.render_native_skill_history
"$PY" -m src.dynamics.main_table
"$PY" -m src.dynamics.weekly_status
