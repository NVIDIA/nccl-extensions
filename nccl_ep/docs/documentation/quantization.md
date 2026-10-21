# Quantization

This document defines the public tensor contracts for NCCL EP quantization
support. The support is realized via quantization recipes (separate for Dispatch and Combine).

Attention-FFN Disaggregation (AFD, enabled by
`ncclEpGroupConfig_t::num_attn_ranks > 0`) currently supports unquantized
communication only. AFD callers must select `NCCL_EP_DISP_QUANT_NONE` for
dispatch and `NCCL_EP_COMB_QUANT_NONE` for combine.

In the remainder of this document,
* `B` is the batch size,
* `H` is the logical hidden dimension, and
* `S` is the caller-provided number of scale elements per token.

## `NCCL_EP_DISP_QUANT_NONE` and `NCCL_EP_COMB_QUANT_NONE` recipes

The unquantized recipes transport tokens in their declared dtype.
They are the default and require no scale tensors.

## `NCCL_EP_DISP_QUANT_FWD`

This recipe is available for **Dispatch** only.
It forwards the physical bytes of two 2D inputs: tokens `[B x H]` and
scales `[B x S]`.
Accepted token element data types are: FP32, FP16, BF16, FP8, or FP4x2 (see [FP4x2 data type](#fp4x2-data-type) below);
scales may use FP32, FP16, BF16, FP8, or `ncclUint8` raw byte storage.
`S` is taken directly from the scale tensor; the recipe does not prescribe a scale-block size.
Each physical row and storage base (or window offset) must be 16B aligned.

Dispatch scale outputs have the same leading layout dimensions as token outputs and `S` as their final dimension. In LL rank-major mode, token and scale output descriptors can independently be backed by NCCL windows.


`NCCL_EP_DISP_QUANT_FWD` applies to both HT and LL, and to both pass directions; `FWD`
means that scales are forwarded, not that the operation is a forward pass.
Both token and scale outputs are required, must match their respective input
dtype and physical row width, and `round_scales` must be zero. HT outputs are
2D and are either both window-backed or neither is; expert-major permutation
may stage before writing those windows. With `zero_copy = NCCL_EP_ZERO_COPY_ON`, windows for both tokens and scales tensors must be provided.

## `NCCL_EP_DISP_QUANT_DS_FP8E3M4`

This LL-only recipe performs DeepSeek FP8 E3M4 quantization internally. Supply
BF16 token inputs and no input scale tensor. Dispatch produces E4M3 token
bytes and required FP32 output scales, one per 128 token elements. The hidden
dimension must be divisible by 512.

On an NVLink-only, rank-major group, all three LL dispatch families
(`NCCL_EP_DISPATCH_LSA=universal|simt|tma`) support writing generated tokens
and scales directly into their independently registered output windows.
See [Zero-Copy](zero_copy.md) for window registration and `zero_copy` semantics.

## `NCCL_EP_COMB_QUANT_NVFP4` (EXPERIMENTAL)

This experimental LL-only recipe transports BF16 expert outputs using NVFP4 and requires CUDA 12.9+ with `cuda_fp4.h` and an E2M1 FP4-family GPU target.
The current usage API is temporary and **will likely change in the future**.
The hidden dimension must be a multiple of 512; multiples of 1024 additionally
enable a wider send unroll.

**Pre-requisites**: Before calling Combine, the application must calculate global scales (a single value per token) as `2688 / amax(abs(token))`, using `0` when `amax` is `0`.

To enable NVFP4 quantization:
* set `combine_config.quant_recipe = NCCL_EP_COMB_QUANT_NVFP4`
* pass global scales via `combine_inputs.scales` as a FP32 3D
tensor with the same leading dimensions as `combine_inputs.tokens` and a final
dimension of one (`[*][*][1]`).
* combine produces no scale output: NCCL EP performs the quantization internally and returns tokens in the BF16 format.

## Sizing of NCCL EP staging buffers (`max_token_bytes`)

`ncclEpGroupConfig_t::max_token_bytes` is the physical-byte budget for one
token row, including any attached scale data.
For example, packed FP4 with `S` scale elements requires
`H/2 + S * sizeof(scale_dtype)` bytes. In HT, configure
this bound as a multiple of 16 bytes.

## FP4x2 data type

FP4x2 is a pair of logical FP4 values packed in a single `uint8_t` data type.
For logical hidden size `H`, use physical token shape `[B x H/2]` (`sizes[1]` is `H/2`).
To ensure 16B tensor alignment, `H` must be a multiple of 32 (so that `H/2` in `[B x H/2]` is 16B aligned).
NCCL EP byte-forwards this reserved type (note that NCCL collectives do not support it).
