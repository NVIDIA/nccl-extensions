/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for the group-owned low-latency communication epoch ring.
 *
 * Two expert-major handles share one LL group.  Handle A uses top-k 1.
 * Handle B uses top-k 2, routing each token to its base expert and the next
 * expert with weights 0.5/0.5.  This makes B's LL payload slot stride larger
 * than A's while identity combine still recovers each original token.
 *
 * The test first executes the sequential lifecycle
 *
 *     dispatch(A), combine(A), dispatch(B), combine(B)
 *
 * which catches implementations that advance dispatch and combine on
 * independent operation-kind epochs.  It then executes the phase-batched
 * lifecycle with the larger-stride handle first:
 *
 *     dispatch(A), dispatch(B), combine(A), combine(B)
 *
 * This catches old handle-local parity and missing mixed-stride rebasing:
 * without rebasing, A's slot 1 begins inside B's larger slot 0.  Finally, it
 * executes a split dispatch for each handle:
 *
 *     dispatch(A, send_only), complete(A), combine(A)
 *     dispatch(B, send_only), complete(B), combine(B)
 *
 * This verifies that a send-only operation stages one group epoch and its
 * matching completion consumes it.  The LL RDMA buffers are group resources,
 * so a group-owned epoch must advance once for every operation, independent of
 * which handle issued it.  Each ordering is repeated and validates both
 * dispatch buffers and both identity-combine round trips on exactly four ranks.
 */

#include "test_common.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include <vector>

namespace {

constexpr int kRequiredRanks = 4;
constexpr int kLlHidden = 512;
constexpr int kLlLocalExperts = kNumExperts / kRequiredRanks;
constexpr int kLlSlotsPerExpert = kRequiredRanks * kNumTokens;
constexpr int kIterations = 8;
constexpr std::array<int, 2> kLayerTopK{1, 2};

ncclEpGroup_t g_ll_group = nullptr;

struct LayerState {
    int top_k = 0;
    ncclEpHandle_t handle = nullptr;

    int64_t* d_topk = nullptr;
    nv_bfloat16* d_input = nullptr;
    nv_bfloat16* d_dispatched = nullptr;
    nv_bfloat16* d_combined = nullptr;
    float* d_weights = nullptr;
    int32_t* d_expert_counts = nullptr;

    ncclEpTensor_t* topk = nullptr;
    ncclEpTensor_t* input = nullptr;
    ncclEpTensor_t* dispatched = nullptr;
    ncclEpTensor_t* combined = nullptr;
    ncclEpTensor_t* weights = nullptr;
    ncclEpTensor_t* expert_counts = nullptr;
};

float token_value(int layer, int iteration, int source_rank, int token) {
    // All values stay in [1, 256], where integer values are exactly representable
    // as BF16.  The disjoint A/B ranges make cross-handle corruption obvious.
    return static_cast<float>(1 + iteration * 32 + layer * 16 + source_rank * kNumTokens + token);
}

enum class BootstrapStatus {
    kReady,
    kSkip,
    kError,
};

bool exchange_uid_checked(ncclUniqueId* uid) {
    const size_t size = sizeof(*uid);
    if (g_rank == 0) {
        const ncclResult_t ret = ncclGetUniqueId(uid);
        if (ret != ncclSuccess) {
            fprintf(stderr, "Rank 0: ncclGetUniqueId failed (err=%d)\n", ret);
            return false;
        }

        FILE* file = fopen(g_uid_file.c_str(), "wb");
        if (!file) {
            fprintf(stderr, "Rank 0: failed to open UID file %s for writing\n", g_uid_file.c_str());
            return false;
        }
        const size_t written = fwrite(uid, 1, size, file);
        const int close_ret = fclose(file);
        if (written != size || close_ret != 0) {
            fprintf(stderr, "Rank 0: failed to write UID file %s\n", g_uid_file.c_str());
            return false;
        }
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        FILE* file = fopen(g_uid_file.c_str(), "rb");
        if (file) {
            if (fseek(file, 0, SEEK_END) != 0) {
                fclose(file);
                fprintf(stderr, "Rank %d: failed to seek UID file %s\n", g_rank, g_uid_file.c_str());
                return false;
            }
            const long file_size = ftell(file);
            if (file_size < 0) {
                fclose(file);
                fprintf(stderr, "Rank %d: failed to size UID file %s\n", g_rank, g_uid_file.c_str());
                return false;
            }
            if (static_cast<size_t>(file_size) >= size) {
                rewind(file);
                const size_t read = fread(uid, 1, size, file);
                const int close_ret = fclose(file);
                if (read != size || close_ret != 0) {
                    fprintf(stderr, "Rank %d: failed to read UID file %s\n", g_rank, g_uid_file.c_str());
                    return false;
                }
                return true;
            }
            fclose(file);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fprintf(stderr, "Rank %d: timed out waiting for UID file %s\n", g_rank, g_uid_file.c_str());
    return false;
}

BootstrapStatus ll_bootstrap(int argc, char* argv[]) {
    ep_parse_args(argc, argv, "nccl_ep_test_ll_group_epoch_uid");
    ::testing::InitGoogleTest(&argc, argv);

    if (g_rank >= g_nranks) {
        fprintf(stderr, "Invalid rank %d for %d ranks\n", g_rank, g_nranks);
        return BootstrapStatus::kError;
    }
    if (g_nranks != kRequiredRanks) {
        if (g_rank == 0) {
            printf("SKIP: LL group epoch regression requires exactly %d ranks (got %d)\n", kRequiredRanks, g_nranks);
        }
        return BootstrapStatus::kSkip;
    }

    int device_count = 0;
    cudaError_t cuda_ret = cudaGetDeviceCount(&device_count);
    if (cuda_ret != cudaSuccess || device_count == 0) {
        fprintf(
            stderr,
            "Rank %d: cudaGetDeviceCount failed or found no devices (err=%s, count=%d)\n",
            g_rank,
            cudaGetErrorString(cuda_ret),
            device_count);
        return BootstrapStatus::kError;
    }

    cuda_ret = cudaSetDevice(g_rank % device_count);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaSetDevice failed: %s\n", g_rank, cudaGetErrorString(cuda_ret));
        return BootstrapStatus::kError;
    }

    int device = 0;
    int major = 0;
    cuda_ret = cudaGetDevice(&device);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaGetDevice failed: %s\n", g_rank, cudaGetErrorString(cuda_ret));
        return BootstrapStatus::kError;
    }
    cuda_ret = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaDeviceGetAttribute failed: %s\n", g_rank, cudaGetErrorString(cuda_ret));
        return BootstrapStatus::kError;
    }
    if (major < 9) {
        if (g_rank == 0) printf("SKIP: SM_90+ required (this device is SM_%d0)\n", major);
        return BootstrapStatus::kSkip;
    }

    ncclUniqueId uid{};
    if (!exchange_uid_checked(&uid)) return BootstrapStatus::kError;

    ncclResult_t ret = ncclCommInitRank(&g_comm, g_nranks, uid, g_rank);
    if (ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclCommInitRank failed (err=%d)\n", g_rank, ret);
        return BootstrapStatus::kError;
    }

    cuda_ret = cudaStreamCreate(&g_stream);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaStreamCreate failed: %s\n", g_rank, cudaGetErrorString(cuda_ret));
        return BootstrapStatus::kError;
    }

    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    config.num_experts = kNumExperts;
    config.max_dispatch_tokens_per_rank = kNumTokens;
    config.max_token_bytes = kLlHidden * sizeof(nv_bfloat16);
    config.rdma_buffer_size = NCCL_EP_AUTO;
    config.num_qp_per_rank = NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;

    ret = ncclEpCreateGroup(&g_ll_group, g_comm, &config);
    if (ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup(LL) failed (err=%d)\n", g_rank, ret);
        return BootstrapStatus::kError;
    }

    cuda_ret = cudaStreamSynchronize(g_stream);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "Rank %d: initial stream synchronization failed: %s\n", g_rank, cudaGetErrorString(cuda_ret));
        return BootstrapStatus::kError;
    }
    return BootstrapStatus::kReady;
}

void ll_teardown() {
    if (g_ll_group) ncclEpGroupDestroy(g_ll_group);
    if (g_stream) cudaStreamDestroy(g_stream);
    if (g_comm) ncclCommDestroy(g_comm);
    if (g_rank == 0) remove(g_uid_file.c_str());
}

void destroy_layer(LayerState& layer) {
    if (layer.handle) ncclEpHandleDestroy(layer.handle);
    if (layer.topk) ncclEpTensorDestroy(layer.topk);
    if (layer.input) ncclEpTensorDestroy(layer.input);
    if (layer.dispatched) ncclEpTensorDestroy(layer.dispatched);
    if (layer.combined) ncclEpTensorDestroy(layer.combined);
    if (layer.weights) ncclEpTensorDestroy(layer.weights);
    if (layer.expert_counts) ncclEpTensorDestroy(layer.expert_counts);

    if (layer.d_topk) cudaFree(layer.d_topk);
    if (layer.d_input) cudaFree(layer.d_input);
    if (layer.d_dispatched) cudaFree(layer.d_dispatched);
    if (layer.d_combined) cudaFree(layer.d_combined);
    if (layer.d_weights) cudaFree(layer.d_weights);
    if (layer.d_expert_counts) cudaFree(layer.d_expert_counts);
}

class LayerArrayGuard {
public:
    LayerArrayGuard() = default;
    ~LayerArrayGuard() {
        for (LayerState& layer : layers) destroy_layer(layer);
    }

    LayerArrayGuard(const LayerArrayGuard&) = delete;
    LayerArrayGuard& operator=(const LayerArrayGuard&) = delete;

    std::array<LayerState, 2> layers{};
};

}  // namespace

TEST(LlGroupEpochTest, TwoHandlesFullAndSplitDispatchLifecycles) {
    LayerArrayGuard layer_guard;
    auto& layers = layer_guard.layers;
    const size_t input_elems = static_cast<size_t>(kNumTokens) * kLlHidden;
    const size_t dispatched_elems =
        static_cast<size_t>(kLlLocalExperts) * kLlSlotsPerExpert * kLlHidden;

    for (int layer_idx = 0; layer_idx < static_cast<int>(layers.size()); ++layer_idx) {
        LayerState& layer = layers[layer_idx];
        layer.top_k = kLayerTopK[layer_idx];
        const size_t routing_elems = static_cast<size_t>(kNumTokens) * layer.top_k;

        CUDA_ASSERT(cudaMalloc(&layer.d_topk, routing_elems * sizeof(int64_t)));
        CUDA_ASSERT(cudaMalloc(&layer.d_input, input_elems * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&layer.d_dispatched, dispatched_elems * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&layer.d_combined, input_elems * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&layer.d_weights, routing_elems * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&layer.d_expert_counts, kLlLocalExperts * sizeof(int32_t)));

        std::vector<int64_t> topk(routing_elems);
        std::vector<float> weights(routing_elems);
        for (int token = 0; token < kNumTokens; ++token) {
            const int base_expert = (g_rank * kNumTokens + token) % kNumExperts;
            for (int route = 0; route < layer.top_k; ++route) {
                const size_t idx = static_cast<size_t>(token) * layer.top_k + route;
                topk[idx] = (base_expert + route) % kNumExperts;
                weights[idx] = 1.0f / static_cast<float>(layer.top_k);
            }
        }
        CUDA_ASSERT(cudaMemcpy(
            layer.d_topk, topk.data(), routing_elems * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            layer.d_weights, weights.data(), routing_elems * sizeof(float), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&layer.topk, 2, ncclInt64, layer.d_topk, kNumTokens, layer.top_k));
        NCCL_ASSERT(epTensorCreate(&layer.input, 2, ncclBfloat16, layer.d_input, kNumTokens, kLlHidden));
        NCCL_ASSERT(epTensorCreate(
            &layer.dispatched,
            3,
            ncclBfloat16,
            layer.d_dispatched,
            kLlLocalExperts,
            kLlSlotsPerExpert,
            kLlHidden));
        NCCL_ASSERT(epTensorCreate(&layer.combined, 2, ncclBfloat16, layer.d_combined, kNumTokens, kLlHidden));
        NCCL_ASSERT(epTensorCreate(
            &layer.weights, 2, ncclFloat32, layer.d_weights, kNumTokens, layer.top_k));
        NCCL_ASSERT(epTensorCreate(
            &layer.expert_counts, 1, ncclInt32, layer.d_expert_counts, kLlLocalExperts));

        NCCL_ASSERT(ncclEpCreateHandle(
            &layer.handle,
            g_ll_group,
            NCCL_EP_LAYOUT_EXPERT_MAJOR,
            layer.topk,
            nullptr,
            nullptr,
            g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        ASSERT_NE(layer.handle, nullptr);
    }

    for (int ordering_idx = 0; ordering_idx < 3; ++ordering_idx) {
        const bool phase_batched = ordering_idx == 1;
        const bool split_dispatch = ordering_idx == 2;
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            SCOPED_TRACE(::testing::Message() << "ordering="
                         << (phase_batched ? "D(A),D(B),C(A),C(B)"
                                           : split_dispatch ? "D_send(A),Complete(A),C(A),D_send(B),Complete(B),C(B)"
                                                            : "D(A),C(A),D(B),C(B)"));
            std::array<std::vector<nv_bfloat16>, 2> host_inputs;
            for (int layer_idx = 0; layer_idx < static_cast<int>(layers.size()); ++layer_idx) {
                LayerState& layer = layers[layer_idx];
                host_inputs[layer_idx].resize(input_elems);
                for (int token = 0; token < kNumTokens; ++token) {
                    nv_bfloat16 value = __float2bfloat16(token_value(layer_idx, iteration, g_rank, token));
                    std::fill_n(host_inputs[layer_idx].begin() + token * kLlHidden, kLlHidden, value);
                }
                CUDA_ASSERT(cudaMemcpy(
                    layer.d_input,
                    host_inputs[layer_idx].data(),
                    input_elems * sizeof(nv_bfloat16),
                    cudaMemcpyHostToDevice));
                CUDA_ASSERT(cudaMemset(layer.d_dispatched, 0, dispatched_elems * sizeof(nv_bfloat16)));
                CUDA_ASSERT(cudaMemset(layer.d_combined, 0, input_elems * sizeof(nv_bfloat16)));
                CUDA_ASSERT(cudaMemset(layer.d_expert_counts, 0, kLlLocalExperts * sizeof(int32_t)));
            }

            auto dispatch = [](LayerState& layer, bool send_only = false) {
                ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
                ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
                ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
                ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
                config.send_only = send_only;
                inputs.tokens = layer.input;
                outputs.tokens = layer.dispatched;
                layout_info.expert_counters = layer.expert_counts;
                return ncclEpDispatch(layer.handle, &inputs, &outputs, &layout_info, &config, g_stream);
            };

            auto combine = [](LayerState& layer) {
                ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
                ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
                ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;
                inputs.tokens = layer.dispatched;
                outputs.tokens = layer.combined;
                outputs.topk_weights = layer.weights;
                return ncclEpCombine(layer.handle, &inputs, &outputs, &config, g_stream);
            };

            auto complete = [](LayerState& layer) {
                return ncclEpComplete(layer.handle, nullptr, g_stream);
            };

            if (phase_batched) {
                NCCL_ASSERT(dispatch(layers[0]));
                NCCL_ASSERT(dispatch(layers[1]));
                NCCL_ASSERT(combine(layers[0]));
                NCCL_ASSERT(combine(layers[1]));
            } else if (split_dispatch) {
                NCCL_ASSERT(dispatch(layers[0], true));
                NCCL_ASSERT(complete(layers[0]));
                NCCL_ASSERT(combine(layers[0]));
                NCCL_ASSERT(dispatch(layers[1], true));
                NCCL_ASSERT(complete(layers[1]));
                NCCL_ASSERT(combine(layers[1]));
            } else {
                NCCL_ASSERT(dispatch(layers[0]));
                NCCL_ASSERT(combine(layers[0]));
                NCCL_ASSERT(dispatch(layers[1]));
                NCCL_ASSERT(combine(layers[1]));
            }
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));

            for (int layer_idx = 0; layer_idx < static_cast<int>(layers.size()); ++layer_idx) {
                LayerState& layer = layers[layer_idx];
                std::array<int32_t, kLlLocalExperts> counts{};
                std::vector<nv_bfloat16> dispatched(dispatched_elems);
                std::vector<nv_bfloat16> combined(input_elems);
                CUDA_ASSERT(cudaMemcpy(
                    counts.data(),
                    layer.d_expert_counts,
                    sizeof(counts),
                    cudaMemcpyDeviceToHost));
                CUDA_ASSERT(cudaMemcpy(
                    dispatched.data(),
                    layer.d_dispatched,
                    dispatched_elems * sizeof(nv_bfloat16),
                    cudaMemcpyDeviceToHost));
                CUDA_ASSERT(cudaMemcpy(
                    combined.data(),
                    layer.d_combined,
                    input_elems * sizeof(nv_bfloat16),
                    cudaMemcpyDeviceToHost));

                for (int local_expert = 0; local_expert < kLlLocalExperts; ++local_expert) {
                    const int global_expert = g_rank * kLlLocalExperts + local_expert;
                    std::multiset<float> expected;
                    for (int source_rank = 0; source_rank < kRequiredRanks; ++source_rank) {
                        for (int token = 0; token < kNumTokens; ++token) {
                            const int base_expert =
                                (source_rank * kNumTokens + token) % kNumExperts;
                            for (int route = 0; route < layer.top_k; ++route) {
                                if ((base_expert + route) % kNumExperts == global_expert) {
                                    expected.insert(
                                        token_value(layer_idx, iteration, source_rank, token));
                                }
                            }
                        }
                    }

                    EXPECT_EQ(counts[local_expert], static_cast<int32_t>(expected.size()))
                        << "rank=" << g_rank << " layer=" << layer_idx << " iteration=" << iteration
                        << " local_expert=" << local_expert;

                    std::multiset<float> actual;

                    const int valid_slots =
                        std::max(0, std::min(counts[local_expert], kLlSlotsPerExpert));
                    for (int slot = 0; slot < valid_slots; ++slot) {
                        const size_t base =
                            (static_cast<size_t>(local_expert) * kLlSlotsPerExpert + slot) * kLlHidden;
                        const float value = __bfloat162float(dispatched[base]);
                        actual.insert(value);
                        for (int hidden = 1; hidden < kLlHidden; ++hidden) {
                            EXPECT_EQ(__bfloat162float(dispatched[base + hidden]), value)
                                << "rank=" << g_rank << " layer=" << layer_idx << " iteration=" << iteration
                                << " expert=" << local_expert << " slot=" << slot << " hidden=" << hidden;
                        }
                    }
                    EXPECT_EQ(actual, expected)
                        << "rank=" << g_rank << " layer=" << layer_idx << " iteration=" << iteration
                        << " local_expert=" << local_expert;
                }

                for (int token = 0; token < kNumTokens; ++token) {
                    const float expected = token_value(layer_idx, iteration, g_rank, token);
                    for (int hidden = 0; hidden < kLlHidden; ++hidden) {
                        EXPECT_EQ(__bfloat162float(combined[token * kLlHidden + hidden]), expected)
                            << "rank=" << g_rank << " layer=" << layer_idx << " iteration=" << iteration
                            << " token=" << token << " hidden=" << hidden;
                    }
                }
            }
        }
    }

}

int main(int argc, char* argv[]) {
    const BootstrapStatus bootstrap = ll_bootstrap(argc, argv);
    if (bootstrap == BootstrapStatus::kSkip) return EXIT_SUCCESS;
    if (bootstrap == BootstrapStatus::kError) {
        ll_teardown();
        return EXIT_FAILURE;
    }

    const int result = RUN_ALL_TESTS();
    ll_teardown();
    return result;
}
