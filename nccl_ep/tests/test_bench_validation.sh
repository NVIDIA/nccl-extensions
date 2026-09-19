#!/usr/bin/env bash
# Argument-validation regression; exits before CUDA setup and needs no GPU.
set -euo pipefail
bench=${1:?usage: test_bench_validation.sh /path/to/ep_bench}
export HWLOC_COMPONENTS=-gl,-opencl
for hidden in -1 0 4294967296 18446744073709551616 256x ''; do
    rc=0
    output=$("$bench" --hidden "$hidden" 2>&1) || rc=$?
    [[ $rc == 1 && $output == *'hidden must be a positive unsigned integer'* ]]
    echo "PASS rejected hidden='$hidden'"
done
for hidden in 127 128 129 256 4096 4294967295; do
    # This incompatible option supplies a second parser error after the width
    # check, so supported widths can be checked without initializing CUDA.
    rc=0
    output=$("$bench" --validate --hidden "$hidden" \
        --scales-forward-token-dtype fp32 2>&1) || rc=$?
    [[ $rc == 1 ]]
    if (( hidden <= 128 )); then
        [[ $output == *'NONE validation requires hidden > 128'* ]]
    else
        [[ $output == *'scales-forward dtype/block options require'* ]]
    fi
    echo "PASS hidden=$hidden"
done
rc=0
output=$("$bench" --hidden 128 --scales-forward-token-dtype fp32 2>&1) || rc=$?
[[ $rc == 1 && $output == *'scales-forward dtype/block options require'* ]]
echo 'PASS validation disabled'
