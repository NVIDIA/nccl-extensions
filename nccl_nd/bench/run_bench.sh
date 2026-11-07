#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run qkv_bench on one node with one process per GPU. No MPI or job scheduler is
# needed; ranks exchange the NCCL unique ID through a temporary file.
#
# Usage: run_bench.sh RANKS [--smoke]
# Environment: NCCL_ND_BENCH (default build/bench/nccl_nd/qkv_bench);
# LD_LIBRARY_PATH must resolve libnccl_nd and libnccl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_BIN="${NCCL_ND_BENCH:-${SCRIPT_DIR}/../../build/bench/nccl_nd/qkv_bench}"
[[ -x "${BENCH_BIN}" ]] || { echo "qkv_bench not found at ${BENCH_BIN}; set NCCL_ND_BENCH" >&2; exit 1; }
ranks="${1:-}"
[[ "${ranks}" =~ ^[1-9][0-9]*$ ]] && (( ranks >= 2 )) || { echo "usage: run_bench.sh RANKS [--smoke]" >&2; exit 1; }
shift

WORK_DIR="$(mktemp -d)"
pids=()
cleanup() {
  kill "${pids[@]+"${pids[@]}"}" 2> /dev/null || true
  wait 2> /dev/null || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for ((rank = 0; rank < ranks; ++rank)); do
  if (( rank == 0 )); then
    "${BENCH_BIN}" --rank=0 --nranks="${ranks}" --uid-file="${WORK_DIR}/uid" "$@" &
  else
    "${BENCH_BIN}" --rank="${rank}" --nranks="${ranks}" --uid-file="${WORK_DIR}/uid" "$@" \
      > "${WORK_DIR}/${rank}.log" 2>&1 &
  fi
  pids+=($!)
done
# The first failed rank stops the others, which would otherwise wait in NCCL.
status=0
for _ in "${pids[@]}"; do
  if ! wait -n; then
    status=1
    break
  fi
done
if (( status )); then
  kill "${pids[@]}" 2> /dev/null || true
  wait 2> /dev/null || true
  for ((rank = 1; rank < ranks; ++rank)); do
    echo "--- rank ${rank} ---"
    cat "${WORK_DIR}/${rank}.log"
  done
fi
pids=()
exit "${status}"
