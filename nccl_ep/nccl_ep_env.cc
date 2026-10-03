/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_ep_env.h"

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <strings.h>  // strcasecmp

namespace {

// Parse a boolean flag environment variable into var (is_set + value.flag).
// Accepts only "1"/"on"/"true" (=> true) or "0"/"off"/"false" (=> false),
// case-insensitive. Unset/empty leaves it not-set; any other value is warned
// about and also left not-set.
void parse_flag(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;  // unset -> is_set stays false
    if (strcasecmp(v, "1") == 0 || strcasecmp(v, "on") == 0 || strcasecmp(v, "true") == 0) {
        var.is_set = true;
        var.value.flag = true;
    } else if (strcasecmp(v, "0") == 0 || strcasecmp(v, "off") == 0 || strcasecmp(v, "false") == 0) {
        var.is_set = true;
        var.value.flag = false;
    } else {
        std::fprintf(stderr, "[nccl_ep] %s=%s ignored (expected 1/on/true or 0/off/false)\n", var.name, v);
    }
}

// Parse an unsigned-long environment variable into var (is_set + value.ul).
// Unset/empty leaves it not-set; otherwise the raw strtoul() value is stored
// without interpretation — validity/range is the consumer's responsibility.
void parse_ulong(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;  // unset/empty -> not set
    var.is_set = true;
    var.value.ul = std::strtoul(v, nullptr, 10);
}

// Reject invalid timeout settings instead of turning negative or overflowing
// values into huge unsigned timeout budgets.
void parse_timeout_ms(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;  // unset/empty -> not set
    const char* number = v;
    while (std::isspace(static_cast<unsigned char>(*number))) ++number;
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(number, &end, 10);
    if (*number == '-' || end == number || *end != '\0' || errno == ERANGE) {
        std::fprintf(stderr, "[nccl_ep] %s=%s ignored (expected an unsigned long integer)\n", var.name, v);
        return;
    }
    var.is_set = true;
    var.value.ul = parsed;
}

void parse_choice(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;
    for (unsigned long index = 0; var.choices[index] != nullptr; ++index) {
        if (strcasecmp(v, var.choices[index]) == 0) {
            var.is_set = true;
            var.value.ul = index;
            return;
        }
    }
    std::fprintf(stderr, "[nccl_ep] %s=%s ignored (expected", var.name, v);
    for (const char* const* name = var.choices; *name != nullptr; ++name)
        std::fprintf(stderr, "%s%s", name == var.choices ? " " : ", ", *name);
    std::fprintf(stderr, "); using default\n");
}

}  // namespace

void nccl_ep_env_init(ncclEpEnvConfig* cfg) {
    if (cfg == nullptr) return;
    *cfg = ncclEpEnvConfig{};  // reset to defaults (re-binds the names) before reading

    // Boolean flags: is_set means present-and-valid, value.flag holds the bool.
    parse_flag(cfg->verbose);
    parse_flag(cfg->debug);
    parse_flag(cfg->ht_em_local_dup);
    parse_flag(cfg->ht_em_nvlink_dup);
    parse_flag(cfg->ht_em_pull_push);
    parse_flag(cfg->ht_unfused_sync);
    parse_flag(cfg->ht_em_ag_scan_mode);
    parse_flag(cfg->ht_em_count_unfused);
    parse_flag(cfg->disable_guard);

    parse_timeout_ms(cfg->timeout_ms);
    parse_choice(cfg->dispatch_copy_mode);

    // Numeric (ulong) vars: is_set means present, value.ul holds the raw integer
    // (no range checks here — consumers in nccl_ep.cc validate per their needs).
    parse_ulong(cfg->comm_num_sms);
    parse_ulong(cfg->dispatch_num_sms);
    parse_ulong(cfg->combine_num_sms);
    parse_ulong(cfg->shuffle_sms);
    parse_ulong(cfg->preprocess_num_sms);
    parse_ulong(cfg->tokens_per_chunk);
    parse_ulong(cfg->dispatch_num_stages);
    parse_ulong(cfg->dispatch_num_pipelines);
    parse_ulong(cfg->combine_num_stages_g2s);
    parse_ulong(cfg->combine_num_stages_s2g);
    parse_ulong(cfg->combine_num_pipelines);
}

void nccl_ep_env_print(const ncclEpEnvConfig& cfg) {
    // Every variable, so the user can see exactly what was provided. Adding a
    // field to ncclEpEnvConfig only requires adding it to this list.
    const ncclEpEnvVar* vars[] = {
        &cfg.verbose,
        &cfg.debug,
        &cfg.ht_em_local_dup,
        &cfg.ht_em_nvlink_dup,
        &cfg.ht_em_pull_push,
        &cfg.ht_unfused_sync,
        &cfg.ht_em_ag_scan_mode,
        &cfg.ht_em_count_unfused,
        &cfg.dispatch_copy_mode,
        &cfg.disable_guard,
        &cfg.timeout_ms,
        &cfg.comm_num_sms,
        &cfg.dispatch_num_sms,
        &cfg.combine_num_sms,
        &cfg.shuffle_sms,
        &cfg.preprocess_num_sms,
        &cfg.tokens_per_chunk,
        &cfg.dispatch_num_stages,
        &cfg.dispatch_num_pipelines,
        &cfg.combine_num_stages_g2s,
        &cfg.combine_num_stages_s2g,
        &cfg.combine_num_pipelines,
    };

    std::fprintf(stderr, "[nccl_ep][env] NCCL EP environment configuration:\n");
    for (const ncclEpEnvVar* v : vars) {
        if (!v->is_set) {
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = unset\n", v->name);
            continue;
        }
        switch (v->type) {
        case ncclEpEnvType::flag:
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = %s\n", v->name, v->value.flag ? "enabled" : "disabled");
            break;
        case ncclEpEnvType::ulong:
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = %lu\n", v->name, v->value.ul);
            break;
        case ncclEpEnvType::choice:
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = %s\n", v->name, v->choices[v->value.ul]);
            break;
        }
    }
}
