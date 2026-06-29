/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_ep_env.h"

#include <cstdio>
#include <cstdlib>
#include <strings.h>  // strcasecmp

namespace {

// Truthy values for boolean-style flags: "1"/"true"/"yes"/"on"
// (case-insensitive), or any non-zero integer. Empty/unset => false.
bool parse_truthy(const char* v) {
    if (v == nullptr || v[0] == '\0') return false;
    if (strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0 ||
        strcasecmp(v, "on") == 0)
        return true;
    if (strcasecmp(v, "false") == 0 || strcasecmp(v, "no") == 0 ||
        strcasecmp(v, "off") == 0)
        return false;
    return std::atol(v) != 0;
}

}  // namespace

void nccl_ep_env_init(ncclEpEnvConfig* cfg) {
    if (cfg == nullptr) return;
    *cfg = ncclEpEnvConfig{};  // reset to defaults before (re)reading

    cfg->verbose = parse_truthy(std::getenv("NCCL_EP_ENV_VERBOSE"));

    const char* dbg = std::getenv("NCCL_EP_DEBUG");
    cfg->debug = (dbg != nullptr && dbg[0] != '\0');

    if (const char* e = std::getenv("NCCL_EP_HT_EM_LOCAL_DUP"))
        cfg->ht_em_local_dup = std::atol(e) != 0;
    if (const char* e = std::getenv("NCCL_EP_HT_EM_NVLINK_DUP"))
        cfg->ht_em_nvlink_dup = std::atol(e) != 0;

    if (const char* e = std::getenv("NCCL_EP_TIMEOUT_MS")) {
        const uint64_t ms = std::strtoull(e, nullptr, 10);
        if (ms > 0) {
            cfg->timeout_ms_set = true;
            cfg->timeout_ms = ms;
        }
    }

    if (const char* e = std::getenv("NCCL_EP_PROLOG_EPILOG_SMS")) {
        cfg->prolog_epilog_sms_set = true;
        cfg->prolog_epilog_sms = std::atol(e);
    }

    if (const char* e = std::getenv("NCCL_EP_PREPROCESS_NUM_SMS");
        e != nullptr && e[0] != '\0') {
        cfg->preprocess_num_sms_set = true;
        cfg->preprocess_num_sms = std::atol(e);
    }
}

void nccl_ep_env_print(const ncclEpEnvConfig& cfg) {
    if (!cfg.verbose) return;

    std::fprintf(stderr, "[nccl_ep][env] NCCL EP environment configuration:\n");
    std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_ENV_VERBOSE      = enabled\n");
    std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_DEBUG            = %s\n",
                 cfg.debug ? "enabled" : "unset");

    if (cfg.timeout_ms_set)
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_TIMEOUT_MS       = %llu ms\n",
                     (unsigned long long)cfg.timeout_ms);
    else
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_TIMEOUT_MS       = unset "
                             "(config / compile-time default)\n");

    if (cfg.prolog_epilog_sms_set)
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_PROLOG_EPILOG_SMS = %ld\n",
                     cfg.prolog_epilog_sms);
    else
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_PROLOG_EPILOG_SMS = unset\n");

    if (cfg.preprocess_num_sms_set)
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_PREPROCESS_NUM_SMS = %ld\n",
                     cfg.preprocess_num_sms);
    else
        std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_PREPROCESS_NUM_SMS = unset\n");

    std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_HT_EM_LOCAL_DUP  = %s\n",
                 cfg.ht_em_local_dup ? "on" : "off");
    std::fprintf(stderr, "[nccl_ep][env]   NCCL_EP_HT_EM_NVLINK_DUP = %s\n",
                 cfg.ht_em_nvlink_dup ? "on" : "off");
}
