/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_ENV_H_
#define NCCL_EP_ENV_H_

#include <cstdint>

// ============================================================================
// Per-group NCCL EP environment configuration.
//
// All std::getenv() parsing for the NCCL EP runtime lives in nccl_ep_env.cc.
// A configuration struct is populated once per EP group by nccl_ep_env_init()
// at group creation; handles then read their group's copy (handle->group->env)
// fields directly, rather than touching the environment. The struct only reads
// and types the environment; it makes no assumptions about what the values mean
// — range checking, defaulting, and clamping are the caller's responsibility.
//
// Set NCCL_EP_ENV_VERBOSE=[1|true|yes|on] to have nccl_ep_env_print() echo the
// resolved configuration to stderr, prefixed with "[nccl_ep][env]".
// ============================================================================

struct ncclEpEnvConfig {
    bool     verbose = false;                 // NCCL_EP_ENV_VERBOSE
    bool     debug = false;                   // NCCL_EP_DEBUG (non-empty)
    bool     ht_em_local_dup = false;         // NCCL_EP_HT_EM_LOCAL_DUP
    bool     ht_em_nvlink_dup = false;        // NCCL_EP_HT_EM_NVLINK_DUP

    bool     timeout_ms_set = false;          // NCCL_EP_TIMEOUT_MS present and > 0
    uint64_t timeout_ms = 0;

    bool     comm_num_sms_set = false;        // NCCL_EP_COMM_SMS present
    long     comm_num_sms = 0;                // raw atol() value

    bool     prolog_epilog_sms_set = false;   // NCCL_EP_PROLOG_EPILOG_SMS present
    long     prolog_epilog_sms = 0;           // raw atol() value

    bool     preprocess_num_sms_set = false;  // NCCL_EP_PREPROCESS_NUM_SMS present, non-empty
    long     preprocess_num_sms = 0;          // raw atol() value
};

// Read every NCCL EP environment variable into *cfg (resetting it first).
// Called once per group at creation time. No printing happens here.
void nccl_ep_env_init(ncclEpEnvConfig* cfg);

// Echo the resolved configuration to stderr — but only when cfg.verbose is set.
// Call right after nccl_ep_env_init() so the dump reflects the group's state.
void nccl_ep_env_print(const ncclEpEnvConfig& cfg);

#endif  // NCCL_EP_ENV_H_
