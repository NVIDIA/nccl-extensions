#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Run the no-MPI basic API matrix within an existing Slurm allocation. The
# binary creates one local pthread per GPU, so srun launches one process only.

set -euo pipefail

: "${NCCL_HOME:?NCCL_HOME must be set}"
numGpus="${NCCL_M2N_TEST_GPUS:-${NGPUS:-}}"

if [[ -z "${numGpus}" ]]; then
  numGpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if ! [[ "${numGpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NCCL_M2N_TEST_GPUS/NGPUS must name at least one GPU; got '${numGpus}'." >&2
  exit 2
fi

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

if [[ "${1:-}" != "--inside-srun" && -n "${SLURM_JOB_ID:-}" ]]; then
  exec srun -N 1 -n 1 --exclusive --cpu-bind=none --export=ALL \
    bash "$0" --inside-srun
fi

binary="${NCCL_HOME}/bin/basic_api_test_local"
if [[ ! -x "${binary}" ]]; then
  echo "Missing executable basic API test: ${binary}" >&2
  exit 1
fi

log="$(mktemp)"
trap 'rm -f "${log}"' EXIT

"${binary}" -N "${numGpus}" 2>&1 | tee "${log}"

if ! grep -Eq "^worldSize=${numGpus}, devices=[1-9][0-9]*," "${log}"; then
  echo "basic_api_test_local did not run with ${numGpus} local ranks." >&2
  exit 1
fi

if ! grep -Eq '^\[  PASSED  \] [1-9][0-9]* tests?\.' "${log}"; then
  echo "basic_api_test_local completed without any passing test cases." >&2
  exit 1
fi
