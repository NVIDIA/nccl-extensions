/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Internal test-only helpers — NOT part of the installed public API.
// Include this header only from unit test sources (C++ only), never from library code.
#pragma once
#include "nccl_ep.h"
#include <stdint.h>

// Returns the device pointer to the unified sparse-to-dense map for a HT handle.
// Rank-major layout: int32_t[nNodes * max_tokens_per_rank][n_ranks_per_node]
//   s2d[token][dest] = recv slot at dest, or -1.
// Expert-major layout: int32_t[nNodes * max_tokens_per_rank][num_topk]
//   s2d[token][k] = packed(rank_id<<24 | slot), or -1.
const int32_t* ncclEpHandle_test_getSparseToDenseMap(ncclEpHandle_t handle);

// FWD inline weights + IDs: selected payload width in words, or zero on fallback.
int ncclEpHandle_test_getCompactProbWireFloats(ncclEpHandle_t handle);

enum ncclEpTestMetadataMode {
    NCCL_EP_TEST_META_SCAN = 0,
    NCCL_EP_TEST_META_UNFUSED = 1,
    NCCL_EP_TEST_META_FUSED = 2
};

// Resolved metadata mode after UpdateHandle, including handle-specific fallbacks.
int ncclEpHandle_test_getMetadataMode(ncclEpHandle_t handle);

// Returns the HT scan-only workspace pointers wired into a handle. These buffers
// are group-owned and may be reused after ncclEpUpdateHandle completes.
void ncclEpHandle_test_getScanScratch(
    ncclEpHandle_t handle,
    const void** token_rank_mask,
    const void** preprocessing_scan_tmp,
    const int32_t** token_to_recv_slot);

// Returns num_topk stored in the handle.
int ncclEpHandle_test_getNumTopk(ncclEpHandle_t handle);

// Number of rows in the S2D: config.max_tokens_per_rank
int ncclEpHandle_test_getMaxTokensPerRank(ncclEpHandle_t handle);

// Number of ranks per NVLink node (== nRanks for single-node)
int ncclEpHandle_test_getNRanksPerNode(ncclEpHandle_t handle);

// Number of local experts per rank (num_experts / nRanks)
int ncclEpHandle_test_getExpertsPerRank(ncclEpHandle_t handle);

// HT only: total number of received tokens across all local experts.
// Reads handle->ht.num_tokens_for_experts via cudaMemcpy.
// Returns ncclInvalidUsage for non-HT algorithms.
ncclResult_t ncclEpHandle_test_getNumRecvTokens(ncclEpHandle_t handle, unsigned int* num_recv_tokens);

// Clear the handle's cached topk_idx pointer so the next ncclEpDispatch runs in
// scatter (backward, no-routing) mode.  HT only.
void ncclEpHandle_test_clearTopkIdx(ncclEpHandle_t handle);

// Returns the HT dispatch_tokens_per_chunk selected for this handle.
int ncclEpHandle_test_getDispatchTokensPerChunk(ncclEpHandle_t handle);

// Override the cached dynamic shared-memory limit for an internally-created
// test group. This is test-only and lets unit tests exercise launch fitting on
// devices whose physical limit is larger than the target budget.
ncclResult_t ncclEpGroup_test_setMaxDynamicSmem(ncclEpGroup_t group, int max_dynamic_smem);

// Returns the last LL combine launch's resolved warps-per-group, or zero when
// that group has not launched LL combine.
int ncclEpGroup_test_getLastLlCombineWarpsPerGroup(ncclEpGroup_t group);
// Return the independently resolved dispatch and combine SM budgets.
void ncclEpGroup_test_getSmBudgets(
    ncclEpGroup_t group, unsigned int* dispatch_num_sms, unsigned int* combine_num_sms);
