/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace nccl_ep {

inline constexpr unsigned int kHtFlatDefaultSms = 16;
inline constexpr unsigned int kHtExpertMajorDefaultSms = 24;
inline constexpr unsigned int kHtRubinExpertMajorSms = 32;
inline constexpr unsigned int kHtFlatCombineLargeBatchSms = 32;
inline constexpr unsigned int kHtLargeBatchTokens = 8192;

// Rubin is SM107. A numeric >= comparison also matches Blackwell SM110/120/121.
constexpr unsigned int ht_default_comm_sms(unsigned int device_sm, bool expert_major) {
    if (expert_major && device_sm == 107) {
        return kHtRubinExpertMajorSms;
    }
    return expert_major ? kHtExpertMajorDefaultSms : kHtFlatDefaultSms;
}

constexpr unsigned int ht_flat_combine_default_sms(unsigned int max_dispatch_tokens_per_rank) {
    if (max_dispatch_tokens_per_rank >= kHtLargeBatchTokens) {
        return kHtFlatCombineLargeBatchSms;
    } else {
        return kHtFlatDefaultSms;
    }
}

} // namespace nccl_ep
