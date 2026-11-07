#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run nd_test, and nd_fault_test when present, on one node with one process per
# GPU. No MPI or job scheduler is needed; ranks exchange the NCCL unique ID
# through a temporary file.
#
# Usage: run_tests.sh [ranks ...]   (default: every visible GPU)
# Environment: NCCL_ND_TEST (default build/test/nccl_nd/nd_test);
# NCCL_ND_FAULT_TEST (default: nd_fault_test next to nd_test, skipped if
# absent); LD_LIBRARY_PATH must resolve libnccl_nd and libnccl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BIN="${NCCL_ND_TEST:-${SCRIPT_DIR}/../../build/test/nccl_nd/nd_test}"
[[ -x "${TEST_BIN}" ]] || { echo "nd_test not found at ${TEST_BIN}; set NCCL_ND_TEST" >&2; exit 1; }
FAULT_BIN="${NCCL_ND_FAULT_TEST:-$(dirname "${TEST_BIN}")/nd_fault_test}"
TEST_BINS=("${TEST_BIN}")
if [[ -x "${FAULT_BIN}" ]]; then
  TEST_BINS+=("${FAULT_BIN}")
elif [[ -n "${NCCL_ND_FAULT_TEST:-}" ]]; then
  echo "nd_fault_test not found at ${FAULT_BIN}" >&2
  exit 1
else
  echo "nd_fault_test not found next to nd_test; skipping fault injection" >&2
fi
if (( $# == 0 )); then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra visible <<< "${CUDA_VISIBLE_DEVICES}"
    set -- "${#visible[@]}"
  else
    set -- "$(nvidia-smi -L | wc -l)"
  fi
fi

WORK_DIR="$(mktemp -d)"
pids=()
# Ranks of a failed or interrupted run would otherwise wait in NCCL forever.
cleanup() {
  kill "${pids[@]+"${pids[@]}"}" 2> /dev/null || true
  wait 2> /dev/null || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# run_one BIN RANKS TAG: start one process per rank and stop all on the first failure.
run_one() {
  local bin="$1" ranks="$2" tag="$3" rank pid failed=0
  echo "=== $(basename "${bin}"), ${ranks} ranks ==="
  pids=()
  for ((rank = 0; rank < ranks; ++rank)); do
    "${bin}" --rank="${rank}" --nranks="${ranks}" --uid-file="${WORK_DIR}/uid-${tag}" \
      > "${WORK_DIR}/${tag}-${rank}.log" 2>&1 &
    pids+=($!)
  done
  # Poll so that the first failed rank stops the others.
  while (( ${#pids[@]} > 0 && ! failed )); do
    local running=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2> /dev/null; then
        running+=("${pid}")
      elif ! wait "${pid}"; then
        failed=1
      fi
    done
    pids=("${running[@]+"${running[@]}"}")
    (( ${#pids[@]} == 0 || failed )) || sleep 1
  done
  if (( failed )); then
    kill "${pids[@]+"${pids[@]}"}" 2> /dev/null || true
    wait 2> /dev/null || true
  fi
  pids=()
  cat "${WORK_DIR}/${tag}-0.log"
  if (( failed )); then
    for ((rank = 1; rank < ranks; ++rank)); do
      echo "--- rank ${rank} ---"
      cat "${WORK_DIR}/${tag}-${rank}.log"
    done
    return 1
  fi
}

status=0
for run in $(seq 1 $#); do
  ranks="${!run}"
  [[ "${ranks}" =~ ^[1-9][0-9]*$ ]] && (( ranks >= 2 )) || { echo "invalid rank count '${ranks}'" >&2; exit 1; }
  for index in "${!TEST_BINS[@]}"; do
    run_one "${TEST_BINS[${index}]}" "${ranks}" "${run}-${index}" || status=1
  done
done
exit "${status}"
