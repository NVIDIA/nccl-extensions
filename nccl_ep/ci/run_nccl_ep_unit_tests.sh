#!/usr/bin/env bash
# NCCL EP CI: run the contrib/nccl_ep gtest unit binaries inside a Slurm allocation.
# These tests do not require MPI; contrib/nccl_ep/tests/run_tests.sh launches one
# process per GPU on the allocated node.

set -euo pipefail

: "${NCCL_HOME:?NCCL_HOME must be set}"
NCCL_EP_BUILDDIR="${NCCL_EP_BUILDDIR:-${NCCL_EP_HOME:-${NCCL_HOME}}}"
NUM_GPUS="${NCCL_EP_UNIT_GPUS:-${NGPUS:-}}"

if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi

export NCCL_HOME
export NCCL_EP_HOME="${NCCL_EP_HOME:-${NCCL_EP_BUILDDIR}}"
export NCCL_EP_BUILDDIR
export NCCL_EP_JIT_SOURCE_DIR="${NCCL_EP_JIT_SOURCE_DIR:-${NCCL_EP_BUILDDIR}/include/nccl_ep}"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="${NCCL_EP_JIT_BUILD_INCLUDE_DIR:-${NCCL_HOME}/include}"
export LD_LIBRARY_PATH="${NCCL_EP_BUILDDIR}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${NCCL_EP_JIT_CUDA_INCLUDE_DIR:-${CUDA_HOME}/include}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

RUN_TESTS="./contrib/nccl_ep/tests/run_tests.sh"

if [[ "${NCCL_EP_UNIT_USE_SRUN:-1}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  exec srun \
    -N 1 \
    -n 1 \
    --exclusive \
    --cpu-bind=none \
    --export=ALL \
    bash "$RUN_TESTS" "$NUM_GPUS"
fi

exec bash "$RUN_TESTS" "$NUM_GPUS"
