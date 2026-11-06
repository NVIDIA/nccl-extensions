#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run nd_test on one node with one process per GPU. No MPI or job scheduler is
# needed; ranks exchange the NCCL unique ID through a temporary file.
#
# Usage: run_tests.sh [ranks ...]   (default: every visible GPU)
# Environment: NCCL_ND_TEST (default build/test/nccl_nd/nd_test); LD_LIBRARY_PATH
# must resolve libnccl_nd and libnccl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BIN="${NCCL_ND_TEST:-${SCRIPT_DIR}/../../build/test/nccl_nd/nd_test}"
[[ -x "${TEST_BIN}" ]] || { echo "nd_test not found at ${TEST_BIN}; set NCCL_ND_TEST" >&2; exit 1; }
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

status=0
for run in $(seq 1 $#); do
  ranks="${!run}"
  [[ "${ranks}" =~ ^[1-9][0-9]*$ ]] && (( ranks >= 2 )) || { echo "invalid rank count '${ranks}'" >&2; exit 1; }
  echo "=== nd_test, ${ranks} ranks ==="
  pids=()
  for ((rank = 0; rank < ranks; ++rank)); do
    "${TEST_BIN}" --rank="${rank}" --nranks="${ranks}" --uid-file="${WORK_DIR}/uid-${run}" \
      > "${WORK_DIR}/${run}-${rank}.log" 2>&1 &
    pids+=($!)
  done
  # Poll so that the first failed rank stops the others.
  failed=0
  while (( ${#pids[@]} > 0 && ! failed )); do
    running=()
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
  cat "${WORK_DIR}/${run}-0.log"
  if (( failed )); then
    status=1
    for ((rank = 1; rank < ranks; ++rank)); do
      echo "--- rank ${rank} ---"
      cat "${WORK_DIR}/${run}-${rank}.log"
    done
  fi
done
exit "${status}"
