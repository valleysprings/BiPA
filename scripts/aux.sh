#!/usr/bin/env bash
# Every analysis outside Table 1, one target per folder of aux/ (aux/README.md).
#
#   scripts/aux.sh FOLDER|all [--execute] [--offline]
#
# Without --execute a target prints its steps. --execute runs them, including the steps that call the model;
# --execute --offline runs only the steps that read the recorded runs. all runs scripts/main.sh table1 first.
#
# Examples:
#   scripts/aux.sh all                          # print every step
#   scripts/aux.sh all --execute --offline      # every step that needs no model call
#   scripts/aux.sh selection --execute --offline
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
env_file="${BIPA_HOST_ENV:-host.env}"
[[ -f $env_file ]] && { set -a; source "$env_file"; set +a; }

# in this order: edits reads the outputs of tokens and calls
FOLDERS=(tokens calls edits success selection transfer three_word_seed)

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; echo "FOLDER: ${FOLDERS[*]}"; exit "${1:-0}"; }

execute=0 offline=0 status= folder=
attempt() {  # SHOWN COMMAND...: runs COMMAND; a failure ends the folder
    local shown="$1"; shift
    echo "+ $shown"
    "$@" || { echo "$shown" > "$status/$folder.failed"; exit 1; }
    echo "$shown" >> "$status/$folder.ran"
}
step() {  # a module that reads the recorded runs
    if (( execute )); then attempt "python -m $*" "${PYTHON:-python}" -m "$@"; else echo "  python -m $*"; fi
}
model() {  # a module that calls the model through the executor
    if (( execute && !offline )); then
        attempt "python -m $*" "${PYTHON:-python}" -m "$@"
    elif (( execute )); then
        echo "- python -m $*    # calls the model: not run with --offline"
    else
        echo "  python -m $*    # calls the model"
    fi
}

table1() {
    local steps line
    mapfile -t steps < <(scripts/main.sh table1)
    for line in "${steps[@]}"; do
        step ${line#python -m }
    done
}

tokens() {
    step aux.tokens.split compute
    step aux.tokens.split attr
    step aux.tokens.slopes compute
    step aux.tokens.slopes draw-input
    for module in attribution accounting cumulative teaser method; do step "aux.tokens.$module"; done
}

calls() {
    step aux.calls.calls
    step aux.calls.figure
}

edits() {
    for module in lineage rules gate figures; do step "aux.edits.$module"; done
}

success() {
    step aux.success.stats
    for module in reruns pooled domains; do step "aux.success.$module"; done
}

selection() {
    step aux.selection.replay
    step aux.selection.figures
    model aux.selection.pathtest launch --model gpt-6-luna
    model aux.selection.pathtest launch --model deepseek-flash --new-release
    step aux.selection.pathtest
}

transfer() {
    # DeepSeek's tau3 skills under GPT-6-Luna, and DeepSeek's skills across TB2 and AutoBench
    local method pair
    for method in skillopt gepa skilladam seed; do
        model src.run transfer "$method" tau3 deepseek-flash --to gpt-6-luna --execute
    done
    for method in skillopt gepa skilladam; do
        model src.run repeat "$method" tau3 gpt-6-luna --tag ood-from-deepseek-flash --execute
    done
    model src.run repeat seed tau3 gpt-6-luna --tag ood-seed --execute
    for pair in terminal-bench-2-1:automationbench automationbench:terminal-bench-2-1; do
        for method in skillopt gepa skilladam seed; do
            model src.run transfer "$method" "${pair%:*}" deepseek-flash --to "${pair#*:}" --execute
        done
        for method in skillopt gepa skilladam; do
            model src.run repeat "$method" "${pair#*:}" deepseek-flash --tag "ood-from-bench-${pair%:*}" --execute
        done
    done
    for module in transfer coverage contamination loading; do step "aux.transfer.$module"; done
}

three_word_seed() {
    step aux.three_word_seed.analysis
    step aux.three_word_seed.figure
}

count() { if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi; }

[[ $# -ge 1 ]] || usage
target="$1"; shift
for arg in "$@"; do
    case "$arg" in
        --execute) execute=1 ;;
        --offline) offline=1 ;;
        *) usage 2 ;;
    esac
done
case " -h --help " in *" $target "*) usage ;; esac
case " all ${FOLDERS[*]} " in
    *" $target "*) ;;
    *) usage 2 ;;
esac
targets=("$target")
[[ $target == all ]] && targets=(table1 "${FOLDERS[@]}")

if (( !execute )); then
    for folder in "${targets[@]}"; do
        [[ $target == all ]] && echo "### $folder"
        "$folder"
    done
    exit 0
fi

status="$(mktemp -d)"
trap 'rm -rf "$status"' EXIT
failed=0
set +e
for folder in "${targets[@]}"; do
    echo "### $folder"
    (set -e; "$folder")
    (( $? == 0 )) || failed=1
done
set -e
if [[ $target == all ]]; then
    printf '\n%-16s %5s  %s\n' folder steps "failed step"
    for folder in "${targets[@]}"; do
        printf '%-16s %5s  %s\n' "$folder" "$(count "$status/$folder.ran")" "$(cat "$status/$folder.failed" 2>/dev/null || echo -)"
    done
fi
exit "$failed"
