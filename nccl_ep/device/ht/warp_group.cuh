/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_HT_WARP_GROUP_CUH_
#define NCCL_EP_HT_WARP_GROUP_CUH_

namespace ht_gin_unordered {

// Generic warp group for warp-specializaion.
template <int NUM_WARPS, int STARTING_WARPS>
struct warp_group {
    __host__ __device__ static constexpr int size() {
        return 32 * NUM_WARPS;
    }
    __host__ __device__ static constexpr int warp_size() {
        return NUM_WARPS;
    }

    __host__ __device__ static int thread_rank() {
        return threadIdx.x - (32 * STARTING_WARPS);
    }
    __host__ __device__ static int warp_rank() {
        return thread_rank() / 32;
    }
};

}  // namespace ht_gin_unordered

#endif  // NCCL_EP_HT_WARP_GROUP_CUH_
