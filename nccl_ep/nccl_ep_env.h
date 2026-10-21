/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_ENV_H_
#define NCCL_EP_ENV_H_

// ============================================================================
// Per-group NCCL EP environment configuration.
//
// All std::getenv() parsing for the NCCL EP runtime lives in nccl_ep_env.cc.
// Each knob is a ncclEpEnvVar that carries its own environment-variable name,
// so callers never hardcode the name: they read <field>.set / <field>.value and
// (for diagnostics) <field>.name. A configuration struct is populated once per
// EP group by nccl_ep_env_init() at group creation; handles then read their
// group's copy (handle->group->env) directly.
//
// nccl_ep_env_print() dumps every variable (name + provided value, or "unset")
// when NCCL_EP_ENV_VERBOSE is truthy, so the user can inspect exactly what was
// provided. The struct only reads and types the environment — range checking,
// defaulting, and clamping are the caller's responsibility.
// ============================================================================

// The kind of value an environment variable carries; selects which union
// member of ncclEpEnvVar::value is active and how it is dumped.
enum class ncclEpEnvType {
    flag,   // boolean toggle stored in value.flag
    ulong,  // unsigned long integer stored in value.ul
    choice, // index into the variable's choices stored in value.ul
};

// A single environment variable bound to its name and type (both set at
// construction). `is_set` indicates that the parser supplied a value. Most
// invalid settings are warned about and left unset; numeric knobs may preserve
// an invalid value for the consumer to reject. `value` holds the parsed result
// in the union member selected by `type`; callers must
// check `is_set` before reading it. The union lets more representations
// (double, string, ...) be added later without churn.
struct ncclEpEnvVar {
    const char* name;
    ncclEpEnvType type;
    const char* const* choices = nullptr; // Null-terminated names for choice variables.
    bool is_set = false;
    union {
        unsigned long ul = 0;
        bool flag;
    } value;
};

struct ncclEpEnvConfig {
    int rank = 0;
    ncclEpEnvVar verbose{"NCCL_EP_ENV_VERBOSE", ncclEpEnvType::flag};
    ncclEpEnvVar debug{"NCCL_EP_DEBUG", ncclEpEnvType::flag};
    ncclEpEnvVar ht_em_local_dup{"NCCL_EP_HT_EM_LOCAL_DUP", ncclEpEnvType::flag};
    ncclEpEnvVar ht_em_nvlink_dup{"NCCL_EP_HT_EM_NVLINK_DUP", ncclEpEnvType::flag};
    ncclEpEnvVar ht_pull_push{"NCCL_EP_HT_PULL_PUSH", ncclEpEnvType::flag};
    ncclEpEnvVar nucs{"NCCL_EP_ENABLE_NUCS", ncclEpEnvType::flag};
    // Launch the intra-LSA head and tail sync as separate kernels instead of
    // fusing them into the dispatch/combine kernels. Pull-push only.
    ncclEpEnvVar ht_unfused_sync{"NCCL_EP_HT_UNFUSED_SYNC", ncclEpEnvType::flag};
    // Count exchange is the default HT path for EM local-permute and Flat handles; set to fall
    // back to the routing-map AllGather + scan path instead. Honored only under
    // meta_exchange_policy AUTO; see
    // meta_exchange_force_scan.
    ncclEpEnvVar ht_ag_scan_mode{"NCCL_EP_HT_AG_SCAN_MODE", ncclEpEnvType::flag};
    // Count mode fuses the recv layout into dispatch by default; set to force the unfused
    // AllGather + compute_layout_info path (used to compare fused vs. unfused performance).
    // Applies to EM local-permute and Flat handles alike. Honored only under
    // meta_exchange_policy AUTO; see meta_exchange_force_unfused.
    ncclEpEnvVar ht_count_unfused{"NCCL_EP_HT_COUNT_UNFUSED", ncclEpEnvType::flag};
    // Compact receive-row order: source-compact (default) keeps source top-k order
    // without gaps; expert-ascend sorts by ascending expert ID, also without gaps.
    static constexpr const char* ht_recv_topk_orders[] = {"source-compact", "expert-ascend", nullptr};
    ncclEpEnvVar ht_recv_topk_order{"NCCL_EP_HT_RECV_TOPK_ORDER", ncclEpEnvType::choice, ht_recv_topk_orders};
    static constexpr const char* dispatch_copy_modes[] = {"CE", "SIMT", "TMA", nullptr};
    ncclEpEnvVar dispatch_copy_mode{"NCCL_EP_DISPATCH_COPY_MODE", ncclEpEnvType::choice, dispatch_copy_modes};
    ncclEpEnvVar disable_guard{"NCCL_EP_DISABLE_GUARD", ncclEpEnvType::flag};
    ncclEpEnvVar timeout_ms{"NCCL_EP_TIMEOUT_MS", ncclEpEnvType::ulong};
    ncclEpEnvVar comm_num_sms{"NCCL_EP_COMM_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar dispatch_num_sms{"NCCL_EP_DISPATCH_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar combine_num_sms{"NCCL_EP_COMBINE_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar shuffle_sms{"NCCL_EP_SHUFFLE_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar preprocess_num_sms{"NCCL_EP_PREPROCESS_NUM_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar tokens_per_chunk{"NCCL_EP_TOKENS_PER_CHUNK", ncclEpEnvType::ulong};
    ncclEpEnvVar dispatch_tokens_per_chunk{"NCCL_EP_DISPATCH_TOKENS_PER_CHUNK", ncclEpEnvType::ulong};
    ncclEpEnvVar dispatch_num_stages{"NCCL_EP_DISPATCH_NUM_STAGES", ncclEpEnvType::ulong};
    ncclEpEnvVar dispatch_num_pipelines{"NCCL_EP_DISPATCH_NUM_PIPELINES", ncclEpEnvType::ulong};
    ncclEpEnvVar combine_num_stages_g2s{"NCCL_EP_COMBINE_NUM_STAGES_G2S", ncclEpEnvType::ulong};
    ncclEpEnvVar combine_num_stages_s2g{"NCCL_EP_COMBINE_NUM_STAGES_S2G", ncclEpEnvType::ulong};
    ncclEpEnvVar combine_num_pipelines{"NCCL_EP_COMBINE_NUM_PIPELINES", ncclEpEnvType::ulong};
    // Opt-in counted fabric transport for LL dispatch payloads within one
    // NVLink domain. Requires Blackwell+, CUDA 13.3+ (build, runtime and JIT),
    // driver 610.43.02+ (API 13030), fabric endpoints and Fabric Manager.
    // All ranks must agree; any platform/setup failure rejects the group
    // collectively with ncclInvalidUsage. Unset to use the store path.
    ncclEpEnvVar ll_cft{"NCCL_EP_LL_CFT", ncclEpEnvType::flag};
    // Debug-only driver API floor override for testing platform rejection.
    ncclEpEnvVar ll_cft_debug_min_driver{"NCCL_EP_LL_CFT_DEBUG_MIN_DRIVER", ncclEpEnvType::ulong};
    // Selects which LL dispatch kernel family an otherwise LSA-eligible call (nvlinkOnly,
    // rank-major) uses: "simt" = dispatch_kernel_impl_2sided_rm_lsa, "tma" (default when unset) =
    // dispatch_kernel_impl_2sided_rm_lsa_tma, "universal" = the general dispatch_kernel_impl in
    // ll_ep.cuh regardless of LSA eligibility. See NcclEpDispatchLsaMode's doc comment in
    // device/ll_ep_adapter.cuh.
    static constexpr const char* dispatch_lsa_modes[] = {"simt", "tma", "universal", nullptr};
    ncclEpEnvVar dispatch_lsa{"NCCL_EP_DISPATCH_LSA", ncclEpEnvType::choice, dispatch_lsa_modes};
    // Quantization workers in addition to the TMA/control warp, LL DSFP8 only.
    // Default is two workers. Invalid settings are rejected when launching
    // DSFP8 TMA dispatch rather than silently restoring the default.
    ncclEpEnvVar ll_tma_quant_warps{"NCCL_EP_LL_TMA_QUANT_WARPS", ncclEpEnvType::ulong};
    // Selects the LL combine protocol: "2sided" always uses combine_kernel_impl_2sided_rm_lsa and
    // "1sided" always uses combine_kernel_impl_1sided_rm_lsa, both requiring LSA eligibility
    // (errors otherwise); "universal" always uses the general kDefault kernel regardless of LSA
    // eligibility, mirroring dispatch_lsa's "universal"; UNSET (default) is "auto" -- the pick is
    // made by dispatch (not combine), on-device, from whether any active peer's published count
    // exceeds combine_lsa_threshold below, and consumed as-is by combine at kernel entry. See
    // combine_lsa handling in nccl_ep.cc and LlCombineKernel in jit/ll_combine_jit.cuh.
    static constexpr const char* combine_lsa_modes[] = {"2sided", "1sided", "universal", nullptr};
    ncclEpEnvVar combine_lsa{"NCCL_EP_COMBINE_LSA", ncclEpEnvType::choice, combine_lsa_modes};
    // Threshold above which combine_lsa's auto mode picks 1-sided over 2-sided -- read and applied
    // by dispatch (dispatch_kernel_args_t::combineLsaThreshold), not combine. Unset defaults to 512
    // at the read site (nccl_ep.cc), not here.
    ncclEpEnvVar combine_lsa_threshold{"NCCL_EP_COMBINE_LSA_1SIDED_THRESH", ncclEpEnvType::ulong};
};

// True iff a flag variable was explicitly set to an "on" value (1/on/true).
// Unset, invalid, or explicitly-off all yield false.
inline bool nccl_ep_env_flag_on(const ncclEpEnvVar& var) {
    return var.is_set && var.value.flag;
}

// Verbose diagnostics was requested by the user
inline bool nccl_ep_env_verbose(const ncclEpEnvConfig& cfg) {
    return nccl_ep_env_flag_on(cfg.verbose) && cfg.rank == 0;
}

// Stamp this process's rank into *cfg.
inline void nccl_ep_env_set_rank(ncclEpEnvConfig* cfg, int rank) {
    if (cfg != nullptr) cfg->rank = rank;
}

// Read every NCCL EP environment variable into *cfg (resetting it first).
// Called once per group at creation time. No printing happens here.
void nccl_ep_env_init(ncclEpEnvConfig* cfg);

// Dump every environment variable (name + provided value, or "unset") to
// stderr. Always prints — the caller decides when (e.g. gated on verbose at
// group init, or unconditionally when reporting a misconfiguration).
void nccl_ep_env_print(const ncclEpEnvConfig& cfg);

#endif  // NCCL_EP_ENV_H_
