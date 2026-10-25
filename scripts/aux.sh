#!/usr/bin/env bash
# Every analysis outside the main experiment, one target per folder of aux/ (aux/README.md).
#
#   scripts/aux.sh FOLDER|all [--execute] [--offline]
#
# What a target prints and what --execute and --offline run: README.md, Reproduction.
#
# Examples:
#   scripts/aux.sh all                                    # print every step
#   scripts/aux.sh all --execute --offline                # every step that needs no model call
#   scripts/aux.sh selection_replay --execute --offline
#   scripts/aux.sh headline_numbers --execute             # the transfer evaluations too (calls the model)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/.."
if [[ -f host.env ]]; then
    set -a; source host.env; set +a
fi

FOLDERS=(rule_coding cost_appendix token_attribution selection_replay edit_gate_and_design calls_and_writes
         noise_and_domains headline_numbers transfer_and_trajectories three_word_seed figures
         contamination_and_loading path_test)

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; echo "FOLDER: ${FOLDERS[*]}"; exit "${1:-0}"; }

execute=0 offline=0 status= folder=
attempt() {  # SHOWN COMMAND...: runs COMMAND; a failure ends the folder
    local shown="$1"; shift
    echo "+ $shown"
    "$@" || { echo "$shown" > "$status/$folder.failed"; exit 1; }
    echo "$shown" >> "$status/$folder.ran"
}
step() {  # a module that reads the recorded runs
    if (( execute )); then attempt "python -m $*" "$here/run.sh" -m "$@"; else echo "  python -m $*"; fi
}
model() {  # a module, or a launch script, that calls the model through the executor
    local shown="python -m $*" command=("$here/run.sh" -m "$@")
    [[ $1 == *.sh ]] && shown="$*" command=(bash "$@")
    if (( execute && !offline )); then
        attempt "$shown" "${command[@]}"
    elif (( execute )); then
        echo "- $shown    # calls the model: not run with --offline"
    else
        echo "  $shown    # calls the model"
    fi
}
note() { echo "! $*"; }

table1() {  # the steps of scripts/main.sh table1
    local steps line
    mapfile -t steps < <("$here/main.sh" table1)
    for line in "${steps[@]}"; do
        step ${line#python -m }
    done
}

rule_coding() {
    step aux.rule_coding.edges build
    step aux.rule_coding.skilltext compute
    step aux.rule_coding.skilltext record
    step aux.rule_coding.skilltext luna
    step aux.rule_coding.coder2
    step aux.rule_coding.release_skills
}

cost_appendix() {
    step aux.cost_appendix.price
    step aux.cost_appendix.compaction compute
    step aux.cost_appendix.compaction record
    step aux.cost_appendix.optcost
    step aux.cost_appendix.training_cost
}

token_attribution() {
    step aux.token_attribution.attribution_draw
    step aux.token_attribution.failcost compute
    step aux.token_attribution.failcost record
    step aux.token_attribution.attribution_luna
    step aux.token_attribution.bipdata
}

selection_replay() {
    step aux.selection_replay.selection compute
    step aux.selection_replay.selection draw
    step aux.selection_replay.selection_fig
    step aux.selection_replay.luna
    step aux.selection_replay.devsize
    step aux.selection_replay.collect
    step aux.selection_replay.fig_devpertask
    step aux.selection_replay.fig_priorfit
}

edit_gate_and_design() {
    step aux.edit_gate_and_design.edges_contrast
    step aux.edit_gate_and_design.edges_contrast holm
    step aux.edit_gate_and_design.design batching
    step aux.edit_gate_and_design.design designev
    step aux.edit_gate_and_design.design selrules
    step aux.edit_gate_and_design.implications_fig
    step aux.edit_gate_and_design.tiebreak
    step aux.edit_gate_and_design.tiebreak_record
}

calls_and_writes() {
    step aux.calls_and_writes.callsperturn
    step aux.calls_and_writes.calls_rw extract
    step aux.calls_and_writes.calls_rw stats
    step aux.calls_and_writes.boot_cf
    step aux.calls_and_writes.write_record
}

noise_and_domains() {
    step aux.noise_and_domains.appb
    step aux.noise_and_domains.load
    step aux.noise_and_domains.a2_exch
    step aux.noise_and_domains.replay_models
    step aux.noise_and_domains.domains
    step aux.noise_and_domains.reruns_fig
    step aux.noise_and_domains.domains_fig
}

headline_numbers() {
    # the transfer evaluations: DeepSeek's tau3 skills under GPT-6-Luna, and DeepSeek's skills across TB2 and AutoBench
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
    step aux.headline_numbers.pooled
    step aux.headline_numbers.transfer
}

transfer_and_trajectories() {
    step aux.transfer_and_trajectories.transfer_dq
    step aux.transfer_and_trajectories.transfer_ci
    step aux.transfer_and_trajectories.transfer_fig
    step aux.transfer_and_trajectories.explore_strat
    step aux.transfer_and_trajectories.gains_fig
}

three_word_seed() {
    local name
    for name in collect_dev selection_replay rep_stats rho tau3_seed_overlap wordcounts sentences rules_audit \
                fingerprints make_numbers scratch_fig; do
        step "aux.three_word_seed.$name"
    done
}

figures() {
    step aux.figures.teaser compute
    step aux.figures.teaser draw
    step aux.figures.cumulative
    step aux.figures.method
    step aux.figures.skills
    step aux.figures.bipcum
    step aux.figures.writes
}

contamination_and_loading() {
    step aux.contamination_and_loading.cheatsheet
    step aux.contamination_and_loading.trigger_fig
    skillzip
}
skillzip() {
    if [[ $target == all ]]; then
        note "left out: the delivery and trigger runs need the SkillZip runtime (scripts/aux.sh contamination_and_loading lists them)"
        return
    fi
    note "python -m aux.contamination_and_loading.run_native BUNDLE OUT CACHE_ROOT    # needs the SkillZip runtime: never run"
    note "python -m aux.contamination_and_loading.run_trigger BUNDLE BENCH OUT [SLOTS]    # needs the SkillZip runtime: never run"
}

path_test() {
    local passes="${BIPA_ASSETS:-assets}/artifacts/path-test" launch
    # the passes on the updated DeepSeek release, as launched for the recorded ones
    for launch in deepseek deepseek-auto deepseek-supp deepseek-gepa; do
        model aux/path_test/launch.sh "$launch"
    done
    if (( execute )) && [[ ! -d "$passes" ]]; then
        note "not run: no path-test passes under $passes"
        return
    fi
    step aux.path_test.collect_path_test
    step aux.path_test.estimate_gap
    step aux.path_test.pathtest_fig
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
    printf '\n%-27s %5s  %s\n' folder steps "failed step"
    for folder in "${targets[@]}"; do
        printf '%-27s %5s  %s\n' "$folder" "$(count "$status/$folder.ran")" "$(cat "$status/$folder.failed" 2>/dev/null || echo -)"
    done
fi
exit "$failed"
