#!/usr/bin/env bash
# Run every runs/tf_*_strict cell (parallel, staggered). PARALLEL=0 for sequential.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$(cd "${HERE}/../../runs" && pwd)"
for d in "${RUNS}"/tf_*_strict; do
  n="$(basename "$d")"
  if [[ "${PARALLEL:-1}" == "1" ]]; then
    (cd "$d" && RUN_TITLE="Phase-T $n" bash run.sh > run_stdout.log 2>&1; echo "exit=$?" >> run_stdout.log) &
    sleep 10
  else
    (cd "$d" && RUN_TITLE="Phase-T $n" bash run.sh > run_stdout.log 2>&1; echo "exit=$?" >> run_stdout.log)
  fi
done
wait
"${WSP_PYTHON:-python3}" "${HERE}/summarize_results.py"
