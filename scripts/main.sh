#!/usr/bin/env bash
# The main experiment.
#
#   scripts/main.sh train MODEL [--benches B ...] [--methods M ...] [--execute]
#       the three optimizers and the two controls on each benchmark, each followed by its three test repetitions
#   scripts/main.sh run|repeat|transfer ARGS... [--execute]
#       one cell, as python -m src.run run|repeat|transfer
#   scripts/main.sh table1 [--execute]
#       Table 1 for both models, including rho_turns (python -m src.analysis.table1)
#
# Without --execute a target prints what it would run.
#
# Examples:
#   scripts/main.sh train deepseek-flash                      # print every run of the DeepSeek experiment
#   scripts/main.sh run skillopt tau3 gpt-6-luna --execute    # optimize one cell (calls the model)
#   scripts/main.sh table1 --execute
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
env_file="${BIPA_HOST_ENV:-host.env}"
[[ -f $env_file ]] && { set -a; source "$env_file"; set +a; }
py() { exec "${PYTHON:-python}" -m "$@"; }

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage
target="$1"; shift
case "$target" in
    -h|--help) usage ;;
    train) py src.run suite "$@" ;;
    run|repeat|transfer) py src.run "$target" "$@" ;;
    table1)
        execute=0
        for arg; do
            case "$arg" in
                --execute) execute=1 ;;
                --offline) ;;
                *) usage 2 ;;
            esac
        done
        if (( execute )); then
            "${PYTHON:-python}" -m src.analysis.table1
        else
            echo "python -m src.analysis.table1"
        fi ;;
    *) usage 2 ;;
esac
