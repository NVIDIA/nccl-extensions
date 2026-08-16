#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ ! -r "$1" ]; then
    echo "usage: $0 LOG_FILE [bucket_order|scaled]" >&2
    exit 2
fi

log_file=$1

# The test name is a parameter so the same guard covers every path that is
# expected to fuse. A coupled ncclReshardScaled submits its payload and scale
# planes as one two-entry group, so it produces the same fusion signature as an
# explicit two-entry group; without this check the feature could silently
# degrade to two sequential reshards while every correctness test still passes.
case "${2:-bucket_order}" in
    bucket_order) test_name_pattern='M2nGroupMpiTest\.OverlappingCommunicatorsPreserveBucketOrder' ;;
    scaled)       test_name_pattern='M2nGroupMpiTest\.ScaledReshardMovesBothPlanes' ;;
    *) echo "usage: $0 LOG_FILE [bucket_order|scaled]" >&2; exit 2 ;;
esac

[ "$(grep -Ec "^([0-9]+: )?\\[       OK \\] ${test_name_pattern}( \\(|$)" "$log_file")" -eq 1 ] &&
    ! grep -Eq '^([0-9]+: )?\[  SKIPPED \]' "$log_file" &&
    grep -Eq 'ncclM2nGroupEnd: entries=2 bins=[0-9]+ fusedBins=1 maxBinEntries=2 budget=[0-9]+' "$log_file"
