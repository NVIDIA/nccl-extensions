/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// Auto configuration constant for dynamic/automatic sizing
#define NCCL_EP_AUTO 0

// Communication algorithm (mode)
typedef enum {
    // Low-Latency (LL) mode
    NCCL_EP_ALGO_LOW_LATENCY = 0,
    // High-Throughput (HT) mode
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1
} ncclEpAlgorithm_t;

/**
 * Receive-buffer layout for the Low-Latency (LL) dispatch path.
 *
 * Controls the shape of the user-visible dispatch output tensor (recv_x) and
 * determines which side performs the expert weighted reduction before combine.
 *
 * The value is usable directly as a CUDA non-type template parameter.
 */
typedef enum {
    /**
     * Auto-select layout based on algorithm (zero-init default).
     * ncclEpCreateGroup resolves this to EXPERT_MAJOR for LL and RANK_MAJOR for HT.
     */
    NCCL_EP_LAYOUT_AUTO = NCCL_EP_AUTO,

    /**
     * Expert-major layout.
     *
     * Dispatch output:
     *   recv_x shape: [num_local_experts, max_tokens_per_rank * num_ranks, hidden]
     *
     * Combine input is the post-expert activation in the same shape.
     * Each expert rank sends its post-expert activation back to the originating
     * rank as a separate message; the combine kernel there accumulates up to
     * num_topk per-expert contributions, weighted by their topk weights, as
     * they arrive (reduction on the receive side).
     */
    NCCL_EP_LAYOUT_EXPERT_MAJOR,

    /**
     * Rank-major layout.
     *
     * Dispatch output:
     *   recv_x shape:            [max_tokens_per_rank * num_ranks, hidden]
     *   recv_topk_weights shape: [max_tokens_per_rank * num_ranks, num_topk]
     *   recv_topk_idx shape:     [max_tokens_per_rank * num_ranks, num_topk]
     *
     * Tokens arrive in rank-major order with no expert dimension.
     * The caller is responsible for running expert computation on each token
     * slot and pre-reducing across local experts using the per-expert weights
     * from recv_topk_weights, producing one weighted output vector per slot.
     *
     * ncclEpCombine sends these pre-reduced vectors back to each token's home
     * rank. The home rank still performs a receive-side reduction, but it
     * accumulates one contribution per source expert rank (not per expert),
     * weighted by that rank's combined weight (sum of its top-k weights for
     * the token). This is less work than expert-major, which reduces over
     * individual per-expert contributions.
     */
    NCCL_EP_LAYOUT_RANK_MAJOR,
} ncclEpLayout_t;

// Tensor tags required to identify the type of tensors in `ncclEpDispatch` and `ncclEpCombine`
typedef enum {
    NCCL_EP_TENSOR_TAG_NONE = 0,
    // Tensor containing tokens
    NCCL_EP_TENSOR_TAG_TOKENS = 1,
    // Tensor containing top-k expert indices
    NCCL_EP_TENSOR_TAG_TOPK_IDX = 2,
    // Tensor containing top-k weights
    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS = 3,
    // Tensor containing scales
    NCCL_EP_TENSOR_TAG_SCALES = 4,
    // Tensor containing tokens received per expert (device memory)
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE = 5,
    // Tensor containing tokens received per expert (pinned host memory)
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST = 6,
    // Tensor containing per-expert token counts
    NCCL_EP_TENSOR_TAG_TOKENS_PER_EXPERTS = 7,
    // LL rank-major dispatch outputs: topk indices/weights received from source ranks
    NCCL_EP_TENSOR_TAG_RECV_TOPK_IDX     = 8,
    NCCL_EP_TENSOR_TAG_RECV_TOPK_WEIGHTS = 9,
} ncclEpTensorTag_t;
