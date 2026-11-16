/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// Toolkit feature tests shared by host (common.hpp) and device JIT (mxfp8_quant.cuh).
// cuda_fp8.h has shipped E4M3 since CUDA 11.8; the E8M0 storage type used by
// MXFP8 reciprocal-scale conversion requires CUDA 12.8.

#include <cuda_runtime.h>

#ifndef NCCL_EP_HAS_CUDA_E8M0_TYPE
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define NCCL_EP_HAS_CUDA_E8M0_TYPE 1
#else
#define NCCL_EP_HAS_CUDA_E8M0_TYPE 0
#endif
#endif

// MXFP8 E8M0 block (elements per scale byte). Packed row H + H/block is
// 16-byte aligned iff H is a multiple of (16 * block) = 512.
#ifndef NCCL_EP_MXFP8_SCALE_BLOCK
#define NCCL_EP_MXFP8_SCALE_BLOCK 32
#endif
#ifndef NCCL_EP_MXFP8_HIDDEN_ALIGN
#define NCCL_EP_MXFP8_HIDDEN_ALIGN (16 * NCCL_EP_MXFP8_SCALE_BLOCK)
#endif
