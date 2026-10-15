#!/usr/bin/env bash
# One offline refresh of results/: per-cell current-run skills, main table, weekly status. No API calls, no loop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$ROOT/bach-harness/.venv/bin/python"
A="$ROOT/bipa/scripts/analysis"
cd "$ROOT"
"$PY" "$A/render_native_skill_history.py"
"$PY" "$A/main_table.py"
"$PY" "$A/weekly_status.py" --weekly paper/weekly/2026-10-07.md
