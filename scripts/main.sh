#!/usr/bin/env bash
# The main experiment.
#
#   scripts/main.sh train MODEL [--benches B ...] [--methods M ...] [--execute]
#       the three optimizers and the two controls on each benchmark, each followed by its three test repetitions
#   scripts/main.sh run|repeat|transfer ARGS... [--execute]
#       one cell, as python -m src.run run|repeat|transfer
#   scripts/main.sh table1 [--execute] [--offline]
#       Table 1, its paired statistics, the turn slopes and the four-term attribution (python -m src.analysis)
#
# What a target prints and what --execute and --offline run: README.md, Reproduction.
#
# Examples:
#   scripts/main.sh train deepseek-flash                      # print every run of the DeepSeek experiment
#   scripts/main.sh run skillopt tau3 gpt-6-luna --execute    # optimize one cell (calls the model)
#   scripts/main.sh table1 --execute
set -euo pipefail
here="$(dirname "${BASH_SOURCE[0]}")"

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage
target="$1"; shift
case "$target" in
    -h|--help) usage ;;
    train) exec "$here/run.sh" suite "$@" ;;
    run|repeat|transfer) exec "$here/run.sh" "$target" "$@" ;;
    table1)
        execute=0
        for arg; do
            case "$arg" in
                --execute) execute=1 ;;
                --offline) ;;
                *) usage 2 ;;
            esac
        done
        for module in attribution slopes stats table1; do  # the graph and attribution caches first
            if (( execute )); then
                "$here/run.sh" -m src.analysis "$module"
            else
                echo "python -m src.analysis $module"
            fi
        done ;;
    *) usage 2 ;;
esac
