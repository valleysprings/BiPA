#!/bin/bash
# Launch the trajectory test passes (resumable: rerun the same command after an interruption).
# usage: launch.sh luna | deepseek | deepseek-auto | deepseek-tb2 | deepseek-main2 | deepseek-gepa | deepseek-supp
cd "$(dirname "$0")/../.." || exit 1
set -a; [[ -f host.env ]] && source host.env; set +a
PY=${PYTHON:-python}
RUN="$PY -u -m aux.path_test.run_path_test"
LOG=${BIPA_ASSETS:-assets}/artifacts/path-test/logs
mkdir -p "$LOG"
SKIP="--skip f87d1b8897ab --skip 032508e0e6e1 --skip 5382240a88f3 --skip ee4f029e6a98 --skip ece3e407af64 --skip fcba9d0bbc84 --skip bee6ffb67dfa"
case "$1" in
  luna) exec $RUN --model gpt-6-luna --execute >> $LOG/luna.log 2>&1 ;;
  deepseek) exec $RUN --model deepseek-flash --execute $SKIP >> $LOG/deepseek.log 2>&1 ;;
  deepseek-auto) exec $RUN --model deepseek-flash --only automationbench --execute $SKIP >> $LOG/deepseek-auto.log 2>&1 ;;
  deepseek-tb2) exec $RUN --model deepseek-flash --only terminal-bench-2-1 --execute $SKIP >> $LOG/deepseek-tb2.log 2>&1 ;;
  deepseek-main2) exec $RUN --model deepseek-flash --execute $SKIP \
      --skip 43226f3d6a15 --skip 9fc436ada6e6 --skip 1e1cc8c08336 >> $LOG/deepseek.log 2>&1 ;;
  deepseek-gepa) exec $RUN --model deepseek-flash --only gepa__tau3-bench --execute >> $LOG/deepseek-gepa.log 2>&1 ;;
  deepseek-supp) exec $RUN --model deepseek-flash --only skillopt__tau3-bench --no-reuse --skip 55f8a4a38828 --skip 45ca0b86e2b2 --execute >> $LOG/deepseek-supp.log 2>&1 ;;
  *) echo "usage: $0 luna|deepseek|deepseek-auto|deepseek-tb2|deepseek-main2|deepseek-gepa|deepseek-supp"; exit 2 ;;
esac
