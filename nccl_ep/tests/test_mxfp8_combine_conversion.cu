/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ===========================================================================
 * MXFP8 combine — numerical conversion coverage
 * ===========================================================================
 *
 * WHAT THIS FILE IS FOR
 *
 * test_mxfp8_combine.cu covers the API contract (layout rejection, scales
 * rejection, wire geometry) and runs one end-to-end smoke case. It does NOT
 * exercise the quantizer: the data it feeds is piecewise constant and lands
 * exactly on the E4M3 grid, so it round-trips bit-exactly and a wrong rounding
 * mode, a wrong scale index, or a broken block-amax reduction would all pass.
 *
 * This file targets the conversion itself. Every case below is chosen so that
 * a specific implementation mistake produces a visibly wrong number.
 *
 *
 * THE CONTRACT UNDER TEST
 *
 * Quantization is fused into the combine prologue (local_permute_reduce):
 * contributions that share a rank are summed in FP32 and the *sum* is
 * quantized once; contributions on different ranks are quantized separately
 * and summed in FP32 after dequantization. In model form:
 *
 *     same rank      ->  Q(a + b)
 *     different rank ->  Q(a) + Q(b)
 *
 * That distinction is the single most important property here and is asserted
 * directly by CrossRankPreservesWhatSameRankLoses.
 *
 *
 * HOW EXPECTATIONS ARE COMPUTED
 *
 * Section 1 is an independent host model of MXFP8, written from the format
 * definition rather than ported from device code, so a bug in the device path
 * cannot hide behind an identical bug in the reference. Section 2 pins that
 * model against hand-computed golden values before it is used to judge
 * anything on the GPU. If Section 2 fails, distrust Section 1 first.
 *
 *
 * HIDDEN SIZES
 *
 * Every H is a separately compiled JIT variant and lands on a different shape of
 * the quantizing epilogue, so the whole suite runs over kHiddenSizes rather than
 * one H. See the table next to that constant for what each size covers. The case
 * tables cycle across the row, so a larger H runs the same cases at many more
 * scale-byte offsets instead of padding with zeros.
 *
 *
 * WHAT THIS FILE DOES NOT COVER
 *
 *  - The inter-node RDMA leg. Unit tests launch all ranks on one host, so
 *    "remote" here means a different rank reached over NVLink, which carries
 *    the packed MXFP8 row. The RDMA leg carries plain BF16 (it is scale-free
 *    by design) and needs a genuine multi-node launch to exercise.
 *  - Blocks whose amax is small enough to drive the E8M0 byte to 0 with a
 *    non-zero payload (amax <= 448 * 2^-127). The decode helper bit-casts the
 *    scale byte, so e=0 yields 0.0f rather than the spec's 2^-127, and such a
 *    block would decode to all zeros. Unreachable from realistic expert
 *    outputs; recorded here so the gap is known.
 */

#include "test_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ===========================================================================
// Section 1 — independent host model of MXFP8
//
// Derived from the format definition, not from nccl_ep/device/mxfp8_quant.cuh.
// Keep it that way: the value of this model is that it fails differently from
// the device code.
// ===========================================================================

// E4M3: 1 sign / 4 exponent (bias 7) / 3 mantissa. Only S.1111.111 is NaN, so
// the maximum finite value is 1.110 * 2^8 = 448 and there are no infinities.
static constexpr double kE4M3Max        = 448.0;
static constexpr double kE4M3MinNormal  = 0.015625;      // 2^-6  = 1.000 * 2^(1-7)
static constexpr double kE4M3MinSubnorm = 0.001953125;   // 2^-9  = 0.001 * 2^(1-7)

// Round a value onto the E4M3 grid, round-half-to-even, saturating to +/-448.
//
// Within a binade [2^e, 2^(e+1)) the 3 mantissa bits give 8 equal steps, so the
// step is 2^e / 8 = 2^(e-3). Below the smallest normal the exponent stops
// decreasing and the mantissa erodes instead, so the step is fixed at 2^-9
// throughout the subnormal range — that is what makes the ladder continuous
// across the normal/subnormal boundary.
static float host_e4m3_round(float v) {
    if (v == 0.0f) return 0.0f;
    if (std::isnan(v)) return v;

    const double sign = (v < 0.0f) ? -1.0 : 1.0;
    const double a = std::fabs(static_cast<double>(v));

    // SATFINITE: anything at or above the top of the range clamps to 448.
    if (a >= kE4M3Max) return static_cast<float>(sign * kE4M3Max);

    double step;
    if (a < kE4M3MinNormal) {
        step = kE4M3MinSubnorm;
    } else {
        // frexp gives a = m * 2^e2 with m in [0.5,1), so floor(log2(a)) = e2-1.
        // Using frexp rather than log2 keeps exact powers of two exact.
        int e2 = 0;
        std::frexp(a, &e2);
        step = std::ldexp(1.0, (e2 - 1) - 3);
    }

    // nearbyint under the default FE_TONEAREST is round-half-to-even, which is
    // what the hardware convert does. Rounding may carry into the next binade
    // (e.g. 31 -> 32); that lands on the grid, so no fixup is needed.
    double q = std::nearbyint(a / step) * step;
    if (q > kE4M3Max) q = kE4M3Max;
    return static_cast<float>(sign * q);
}

// E8M0 exponent byte for a block amax: the smallest e with 2^(e-127) >= amax/448,
// i.e. e = 127 + ceil(log2(amax/448)).
//
// Ceiling rather than rounding, because rounding down would push the largest
// element past 448 and clip precisely the value the block is scaled around.
// The cost is that the scaled amax only lands in (224, 448] instead of exactly
// at 448 — up to one bit of range given away.
static int host_e8m0_from_amax(float amax) {
    if (amax == 0.0f) return 0;
    const double t = static_cast<double>(amax) / kE4M3Max;
    int e2 = 0;
    const double m = std::frexp(t, &e2);       // t = m * 2^e2, m in [0.5,1)
    // ceil(log2(t)) is e2-1 exactly when t is a power of two (m == 0.5), else e2.
    const int unbiased = (m == 0.5) ? (e2 - 1) : e2;
    return std::min(0xFE, std::max(0, 127 + unbiased));
}

// Quantize + dequantize one 32-element block, i.e. what one contributor's data
// looks like after a round trip through the wire.
//
// FINITE INPUTS ONLY. This model deliberately does not reproduce the device's
// non-finite behaviour: the device derives the exponent from the raw float bits,
// so Inf lands on the 0xFE cap, whereas frexp here leaves the exponent
// unspecified for Inf and the model would return 127. Encoding that quirk into
// the reference would defeat the point of the reference being independent, so
// non-finite handling is asserted directly against the device in
// run_nonfinite_case instead. Every case table below is finite, and
// bf16_exact() guards the inputs.
static void host_mxfp8_block(const float* in, float* out, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(in[i]));

    if (amax == 0.0f) {
        // e = 0 and an all-zero payload. The device decodes the zero scale byte
        // to 0.0f, so the block comes back as zeros either way.
        for (int i = 0; i < n; ++i) out[i] = 0.0f;
        return;
    }

    const int e = host_e8m0_from_amax(amax);
    const double scale     = std::ldexp(1.0, e - 127);   // decode multiplier
    const double scale_inv = std::ldexp(1.0, 127 - e);   // encode multiplier

    // Both are exact powers of two, so the scaling contributes no error at all;
    // every bit of loss below comes from the 3-bit E4M3 mantissa.
    for (int i = 0; i < n; ++i) {
        const float scaled = static_cast<float>(static_cast<double>(in[i]) * scale_inv);
        out[i] = static_cast<float>(static_cast<double>(host_e4m3_round(scaled)) * scale);
    }
}

// Values fed to the device must survive BF16 unchanged, otherwise the test is
// measuring BF16 rounding instead of MXFP8 rounding.
static bool bf16_exact(float v) {
    return __bfloat162float(__float2bfloat16(v)) == v;
}

// ===========================================================================
// Section 2 — pin the host model against hand-computed values
//
// These run without any collective, so they still execute (and still catch a
// broken model) when the GPU portion is skipped.
// ===========================================================================

// E4M3 keeps 4 significant bits (implicit 1 + 3 mantissa), so every integer in
// [17,48] that needs a 5th or 6th bit is rounded, ties to even.
//
// Note the ties resolving in BOTH directions: 21->20 rounds down while 27->28
// rounds up. An implementation that always rounded down, or always away from
// zero, would satisfy half this table and fail the other half.
struct IntRoundGolden { int in; int expected; };
static const IntRoundGolden kIntRoundGolden[32] = {
    {17, 16}, {18, 18}, {19, 20}, {20, 20}, {21, 20}, {22, 22}, {23, 24}, {24, 24},
    {25, 24}, {26, 26}, {27, 28}, {28, 28}, {29, 28}, {30, 30}, {31, 32}, {32, 32},
    {33, 32}, {34, 32}, {35, 36}, {36, 36}, {37, 36}, {38, 40}, {39, 40}, {40, 40},
    {41, 40}, {42, 40}, {43, 44}, {44, 44}, {45, 44}, {46, 48}, {47, 48}, {48, 48},
};

TEST(Mxfp8HostModel, IntegerRoundingMatchesGolden) {
    float in[32], out[32];
    for (int i = 0; i < 32; ++i) in[i] = static_cast<float>(kIntRoundGolden[i].in);
    host_mxfp8_block(in, out, 32);
    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(kIntRoundGolden[i].expected))
            << "input " << kIntRoundGolden[i].in;
    }
}

// The block scale is a power of two, so it shifts the exponent without touching
// the significand. Quantizing the same pattern scaled by 2^k must therefore
// give the same answer scaled by 2^k — no better, no worse.
//
// This is why "amax just above a power of two" does NOT cost mantissa
// precision: it costs one binade of underflow headroom instead (see
// UnderflowFloorMovesWithAmax below).
TEST(Mxfp8HostModel, ScaleInvarianceOfRounding) {
    float base[32], base_out[32];
    for (int i = 0; i < 32; ++i) base[i] = static_cast<float>(kIntRoundGolden[i].in);
    host_mxfp8_block(base, base_out, 32);

    for (int k = -6; k <= 6; ++k) {
        const float f = std::ldexp(1.0f, k);
        float in[32], out[32];
        for (int i = 0; i < 32; ++i) in[i] = base[i] * f;
        host_mxfp8_block(in, out, 32);
        for (int i = 0; i < 32; ++i)
            EXPECT_FLOAT_EQ(out[i], base_out[i] * f) << "k=" << k << " i=" << i;
    }
}

// The ceiling rule puts the scaled amax in (224, 448]. When amax is a hair over
// a power of two the scale doubles, which halves the smallest magnitude the
// block can still represent. Same small element, two amax values, opposite fate.
TEST(Mxfp8HostModel, UnderflowFloorMovesWithAmax) {
    const float small = std::ldexp(1.5f, -10);   // 1.5 * 2^-10, BF16-exact

    float in[32], out[32];

    // amax = 448 exactly -> scale 1 -> `small` rounds up to the 2^-9 subnormal.
    for (int i = 0; i < 32; ++i) in[i] = small;
    in[0] = 448.0f;
    host_mxfp8_block(in, out, 32);
    EXPECT_FLOAT_EQ(out[0], 448.0f);
    EXPECT_FLOAT_EQ(out[1], std::ldexp(1.0f, -9));

    // amax = 512 -> scale 2 -> the same element now falls below half a step.
    for (int i = 0; i < 32; ++i) in[i] = small;
    in[0] = 512.0f;
    host_mxfp8_block(in, out, 32);
    EXPECT_FLOAT_EQ(out[0], 512.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
}

// E4M3 carries its own 4 exponent bits, so a block spans roughly 2^18 before
// small elements are lost. A 1000x spread costs nothing; 2^20 wipes the tail.
TEST(Mxfp8HostModel, DynamicRangeWithinBlock) {
    float in[32], out[32];

    for (int i = 0; i < 32; ++i) in[i] = 1.0f;
    in[0] = 1000.0f;
    host_mxfp8_block(in, out, 32);
    EXPECT_FLOAT_EQ(out[1], 1.0f) << "a 1000x spread must not disturb small elements";

    for (int i = 0; i < 32; ++i) in[i] = 1.0f;
    in[0] = std::ldexp(1.0f, 20);
    host_mxfp8_block(in, out, 32);
    EXPECT_FLOAT_EQ(out[0], std::ldexp(1.0f, 20));
    EXPECT_FLOAT_EQ(out[1], 0.0f) << "2^20 below amax is past the underflow floor";
}

TEST(Mxfp8HostModel, SignSymmetry) {
    float pos[32], neg[32], pout[32], nout[32];
    for (int i = 0; i < 32; ++i) {
        pos[i] =  static_cast<float>(kIntRoundGolden[i].in);
        neg[i] = -static_cast<float>(kIntRoundGolden[i].in);
    }
    host_mxfp8_block(pos, pout, 32);
    host_mxfp8_block(neg, nout, 32);
    for (int i = 0; i < 32; ++i) EXPECT_FLOAT_EQ(nout[i], -pout[i]) << "i=" << i;
}

// ===========================================================================
// Section 3 — block case tables
//
// One combine call carries many independent cases: the hidden dimension is cut
// into 32-element MXFP8 blocks and each block holds one case. Blocks are scaled
// independently, so cases cannot contaminate each other.
//
// Adjacent cases deliberately use DIFFERENT amax values. That matters: if every
// block in a row shared a scale, a wrong scale index (j>>2), a wrong 4-lane
// subgroup leader, or a misplaced E8M0 tail would read a neighbouring byte
// holding the same value and stay invisible. With distinct scales any indexing
// error shows up as a power-of-two error.
// ===========================================================================

static constexpr int kBlk      = 32;           // elements per MXFP8 block
static constexpr int kMxTopK   = 2;
static constexpr int kMxTokens = kNumTokens;   // 4
static constexpr int kMxSlots  = 64;           // recv-slot budget, generous for topk=2

// Hidden sizes under test. Every H is a separately compiled JIT variant
// (local_permute_reduce_topk<K>_h<HiddenInt4>_b<blocksPerSm>), and each lands on
// a different shape of the quantizing epilogue, so testing one H tests one
// kernel:
//
//   H      HiddenInt4  kElemsPerThread  kHiddenVec  nn_base iters  blocks/SM
//   512        64            1              1            1             2
//   4096      512            4              4            1             1
//   8192     1024            8              4            2             1
//
//   512  host-gate minimum; degenerate kHiddenVec == 1, and only 64 of the 128
//        lanes in a slot are active, so it covers the partial-row tail.
//   4096 production shape: acc[4][4] (32 float registers) and one block per SM,
//        i.e. the 64-register budget rather than the 32-register one.
//   8192 same as 4096 but drives the nn_base loop round more than once.
//
// Legal H is a multiple of 512: the packed row is H + H/32 bytes, the host
// requires that to be 16 B aligned, and 33H/32 % 16 == 0 implies H % 512 == 0.
static constexpr int kHiddenSizes[] = {512, 4096, 8192};

// ---- single-contributor cases (Section 5, SingleContributor) ----------------

struct BlockCase {
    const char* name;
    void (*fill)(float* v);   // writes kBlk floats
};

// amax 256. Powers of two inside one block: the scale is a power of two and so
// is every element, so the round trip must be bit-exact. Anything but equality
// here means the scale or the payload is wrong, not that precision was lost.
static void fill_powers_of_two(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = std::ldexp(1.0f, i % 9);   // 1..256
}

// amax 48. Integers 17..48 exercise 4-significant-bit rounding with ties in
// both directions. Golden values in kIntRoundGolden.
static void fill_int_rounding(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = static_cast<float>(kIntRoundGolden[i].in);
}

static void fill_neg_powers_of_two(float* v) { fill_powers_of_two(v); for (int i = 0; i < kBlk; ++i) v[i] = -v[i]; }
static void fill_neg_int_rounding(float* v)  { fill_int_rounding(v);  for (int i = 0; i < kBlk; ++i) v[i] = -v[i]; }

// amax 2^20. Everything 2^20 below the block amax must flush to zero; the amax
// itself must survive exactly.
static void fill_underflow_far(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = 1.0f;
    v[0] = std::ldexp(1.0f, 20);
}

// amax 0. Drives the E8M0 byte to 0, whose decode multiplier is 0.0f. Correct
// only because an all-zero amax implies an all-zero payload.
static void fill_all_zeros(float* v) { for (int i = 0; i < kBlk; ++i) v[i] = 0.0f; }

// amax 192 with interleaved exact zeros: zero must stay zero without a special
// case, and must not perturb the block amax.
//
// 192 rather than 256 on purpose. amax 256 and amax 448 both yield E8M0 byte 127
// (scale 1), which would make this block share a scale with its neighbour and
// weaken the distinct-scale property the table relies on. 192 lands on 126.
// AdjacentCasesUseDistinctScales enforces this.
static void fill_zero_inside_nonzero(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = (i % 2 == 0) ? 192.0f : 0.0f;
}

// amax 448 / amax 512 — the pair from UnderflowFloorMovesWithAmax, run on the
// device. The same small element survives in one block and vanishes in the
// other purely because amax crossed a power of two.
static void fill_floor_amax448(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = std::ldexp(1.5f, -10);
    v[0] = 448.0f;
}
static void fill_floor_amax512(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = std::ldexp(1.5f, -10);
    v[0] = 512.0f;
}

// A 32-element block is 4 consecutive BF16 int4s, i.e. 4 adjacent lanes, and the
// block amax is built with __shfl_xor at lane distances 1 and 2. Placing the
// maximum in each of the four sub-lanes in turn proves the reduction spans all
// four. If it were truncated to distance 1, lanes 0-1 and 2-3 would disagree:
// the scale byte written by lane 0 would be too small for the half holding the
// maximum, and that element would clamp to 448 instead of coming back exact.
//
// Each variant uses a different amax so the four blocks also have distinct
// scale bytes.
template <int SubLane, int AmaxExp>
static void fill_amax_in_sublane(float* v) {
    for (int i = 0; i < kBlk; ++i) v[i] = 1.0f;
    v[SubLane * 8] = std::ldexp(1.0f, AmaxExp);   // 8 BF16 elements per int4
}

// Three more distinct scales, to keep every adjacent pair of blocks on a
// different E8M0 byte through the end of the row.
template <int Exp>
static void fill_uniform_pow2(float* v) { for (int i = 0; i < kBlk; ++i) v[i] = std::ldexp(1.0f, Exp); }

static constexpr int kNumSingleCases = 16;
static const BlockCase kSingleCases[kNumSingleCases] = {
    {"PowersOfTwoExact        (amax 256)", fill_powers_of_two},
    {"IntegerRounding17to48   (amax 48)",  fill_int_rounding},
    {"NegPowersOfTwoExact     (amax 256)", fill_neg_powers_of_two},
    {"NegIntegerRounding      (amax 48)",  fill_neg_int_rounding},
    {"UnderflowFar2pow20      (amax 2^20)",fill_underflow_far},
    {"AllZeros                (amax 0)",   fill_all_zeros},
    {"ZeroInsideNonZero       (amax 192)", fill_zero_inside_nonzero},
    {"UnderflowFloorAmax448   (amax 448)", fill_floor_amax448},
    {"UnderflowFloorAmax512   (amax 512)", fill_floor_amax512},
    {"AmaxInSubLane0          (amax 2^4)", fill_amax_in_sublane<0, 4>},
    {"AmaxInSubLane1          (amax 2^5)", fill_amax_in_sublane<1, 5>},
    {"AmaxInSubLane2          (amax 2^6)", fill_amax_in_sublane<2, 6>},
    {"AmaxInSubLane3          (amax 2^7)", fill_amax_in_sublane<3, 7>},
    {"ScaleLadderLow          (amax 2^1)", fill_uniform_pow2<1>},
    {"ScaleLadderHigh         (amax 2^9)", fill_uniform_pow2<9>},
    {"ScaleLadderMid          (amax 2^3)", fill_uniform_pow2<3>},
};

// Compute the E8M0 byte a case will end up with, for the invariant checks below.
static int case_e8m0(const BlockCase& c) {
    float v[kBlk];
    c.fill(v);
    float amax = 0.0f;
    for (int i = 0; i < kBlk; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    return host_e8m0_from_amax(amax);
}

// Enforce the property the table is built around: no two neighbouring blocks may
// share an E8M0 byte. If they did, a wrong scale index would read a neighbouring
// byte holding the same value and the whole suite would go blind to that class
// of bug.
//
// This is not a hypothetical. The first draft of this table put amax 256 next to
// amax 448; both map to byte 127 (scale 1) because the ceiling rule only cares
// about amax/448. Keep this test green when adding cases.
TEST(Mxfp8HostModel, AdjacentCasesUseDistinctScales) {
    for (int b = 1; b < kNumSingleCases; ++b) {
        EXPECT_NE(case_e8m0(kSingleCases[b]), case_e8m0(kSingleCases[b - 1]))
            << "block " << b << " (" << kSingleCases[b].name << ") shares a scale with block "
            << (b - 1) << " (" << kSingleCases[b - 1].name << ")";
    }
}

// Non-degeneracy: a case with a non-zero amax must not quantize to an all-zero
// block. Catches a case whose values were chosen so small that the whole block
// underflows, which would silently stop testing anything.
TEST(Mxfp8HostModel, NonZeroCasesDoNotVanish) {
    for (int b = 0; b < kNumSingleCases; ++b) {
        float in[kBlk], out[kBlk];
        kSingleCases[b].fill(in);
        float amax = 0.0f;
        for (int i = 0; i < kBlk; ++i) amax = std::fmax(amax, std::fabs(in[i]));
        if (amax == 0.0f) continue;   // AllZeros is meant to vanish
        host_mxfp8_block(in, out, kBlk);
        bool any = false;
        for (int i = 0; i < kBlk; ++i) any = any || (out[i] != 0.0f);
        EXPECT_TRUE(any) << kSingleCases[b].name << " quantizes to an all-zero block";
    }
}

// ---- two-contributor cases (Sections 5, SameRank / CrossRank) ---------------
//
// Each case supplies the two contributions A and B. The expected output differs
// by placement: Q(A+B) when both land on one rank, Q(A)+Q(B) when they land on
// different ranks. The same table is used for both so the contrast is exact.

struct PairCase {
    const char* name;
    void (*fill)(float* a, float* b);
};

// Two equal powers of two sum to the next power of two — exact under either
// placement, so this is the control case.
static void pair_equal_powers(float* a, float* b) {
    for (int i = 0; i < kBlk; ++i) { a[i] = 64.0f; b[i] = 64.0f; }
}

// THE headline case. 48 and 2 are each exactly representable, but their sum 50
// needs 6 significant bits and rounds to 48.
//   same rank      -> Q(48+2) = Q(50) = 48   (the +2 is destroyed)
//   different ranks -> Q(48)+Q(2) = 48+2 = 50 (both survive)
static void pair_48_and_2(float* a, float* b) {
    for (int i = 0; i < kBlk; ++i) { a[i] = 48.0f; b[i] = 2.0f; }
}

// Both contributors carry the same value, and it is one that does not survive
// quantization. Cross-rank FP32 reduction prevents error from COMPOUNDING; it
// does not undo the per-contributor error. 50+50 comes back as 96, not 100.
static void pair_50_and_50(float* a, float* b) {
    for (int i = 0; i < kBlk; ++i) { a[i] = 50.0f; b[i] = 50.0f; }
}

// The sum, not the input, sets the block amax. Summed block is {80, 2, 2, ...}.
// If amax were taken before accumulation it would be 40, the scale would be one
// binade too small, and 80 would clamp to 448*2^-3 = 56.
static void pair_sum_changes_amax(float* a, float* b) {
    for (int i = 0; i < kBlk; ++i) { a[i] = 1.0f; b[i] = 1.0f; }
    a[0] = 40.0f; b[0] = 40.0f;
}

// Exact cancellation. On one rank the summed block is all zeros and takes the
// e=0 path from a non-trivial route; across ranks each side quantizes exactly
// and the zero appears only in the final FP32 reduction.
static void pair_cancellation(float* a, float* b) {
    for (int i = 0; i < kBlk; ++i) { a[i] = 48.0f; b[i] = -48.0f; }
}

// Order matters: the summed blocks must not put two equal E8M0 bytes side by
// side, for the same reason the single-contributor table must not. Summed amax
// per case is 128 / 50 / 100 / 0 / 80, giving E8M0 126 / 124 / 125 / 0 / 125,
// and the wrap from the last back to the first is 125 -> 126. FiftyPlusFifty
// and SumChangesAmax both land on 125, so Cancellation sits between them.
static const PairCase kPairCases[] = {
    {"EqualPowersOfTwo",  pair_equal_powers},      // summed amax 128 -> e8m0 126
    {"FortyEightPlusTwo", pair_48_and_2},          // summed amax  50 -> e8m0 124
    {"FiftyPlusFifty",    pair_50_and_50},         // summed amax 100 -> e8m0 125
    {"Cancellation",      pair_cancellation},      // summed amax   0 -> e8m0   0
    {"SumChangesAmax",    pair_sum_changes_amax},  // summed amax  80 -> e8m0 125
};
static constexpr int kNumPairCases = static_cast<int>(sizeof(kPairCases) / sizeof(kPairCases[0]));

// Block b of a row uses case b % N: at H > 512 the tables simply repeat, so a
// longer row runs the same cases at many more scale-byte offsets rather than
// padding with zeros. The wrap preserves the distinct-adjacent-scale property
// in both tables (single: e8m0 122 -> 127; pair: 125 -> 126).
static const BlockCase& single_case(int b) { return kSingleCases[b % kNumSingleCases]; }
static const PairCase&  pair_case(int b)   { return kPairCases[b % kNumPairCases]; }

// Same invariant as AdjacentCasesUseDistinctScales, for the same-rank (summed)
// view of the pair table. Checked with wraparound, because the table now cycles.
TEST(Mxfp8HostModel, AdjacentPairCasesUseDistinctScales) {
    auto summed_e8m0 = [](const PairCase& c) {
        float a[kBlk], b[kBlk];
        c.fill(a, b);
        float amax = 0.0f;
        for (int i = 0; i < kBlk; ++i) amax = std::fmax(amax, std::fabs(a[i] + b[i]));
        return host_e8m0_from_amax(amax);
    };
    for (int b = 0; b < kNumPairCases; ++b) {
        const int prev = (b + kNumPairCases - 1) % kNumPairCases;
        EXPECT_NE(summed_e8m0(kPairCases[b]), summed_e8m0(kPairCases[prev]))
            << kPairCases[b].name << " shares a summed scale with " << kPairCases[prev].name;
    }
}

// ===========================================================================
// Section 4 — harness
// ===========================================================================

// Expert e lives on rank e / experts_per_rank.
static int experts_per_rank() { return kNumExperts / g_nranks; }
static int expert_on_rank(int rank, int local_idx) { return rank * experts_per_rank() + local_idx; }
static int next_rank() { return (g_rank + 1) % g_nranks; }

static ncclEpGroup_t make_group(int hidden) {
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kMxTokens;
    gcfg.max_token_bytes = static_cast<unsigned int>(hidden) * static_cast<unsigned int>(sizeof(nv_bfloat16));
    gcfg.rdma_buffer_size = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank = NCCL_EP_AUTO;
    gcfg.num_channels = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank = kMxSlots;
    ncclEpGroup_t g = nullptr;
    EXPECT_EQ(ncclEpCreateGroup(&g, g_comm, &gcfg), ncclSuccess);
    return g;
}

// One discovered recv slot: which (source rank, token) it holds, and its
// ordinal among the slots on this rank holding that same token.
struct SlotInfo {
    int slot;
    int src_rank;
    int token;
    int ordinal;   // 0 or 1 when both top-k entries of a token land on this rank
};

// Everything the harness needs for one routing configuration.
struct Fixture {
    int            hidden = 0;      // this fixture's H; every buffer below is sized by it
    ncclEpGroup_t  group  = nullptr;
    ncclEpHandle_t handle = nullptr;
    int64_t*       d_topk = nullptr;
    ncclEpTensor_t* t_topk = nullptr;
    std::vector<SlotInfo> slots;   // valid slots on THIS rank, ascending
};

// Dispatch once with a tagged payload. This does double duty: it warms the
// handle (combine requires a completed dispatch) and it reveals the expert-major
// slot mapping, which is not otherwise queryable.
//
// Token i of rank r is stamped with (r+1, i+1) in elements 0 and 1. Padding
// slots stay zero, so a zero tag marks a slot that carries no token. The mapping
// learned here also applies to the combine input, because both are indexed by
// the same em_slot for a given handle.
static void dispatch_and_discover_slots(Fixture& fx) {
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float *d_w = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,    static_cast<size_t>(kMxTokens) * fx.hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,   static_cast<size_t>(kMxSlots)  * fx.hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,      static_cast<size_t>(kMxTokens) * kMxTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, static_cast<size_t>(kMxSlots)  * kMxTopK * sizeof(float)));

    std::vector<nv_bfloat16> h_tok(static_cast<size_t>(kMxTokens) * fx.hidden, __float2bfloat16(0.0f));
    for (int t = 0; t < kMxTokens; ++t) {
        h_tok[static_cast<size_t>(t) * fx.hidden + 0] = __float2bfloat16(static_cast<float>(g_rank + 1));
        h_tok[static_cast<size_t>(t) * fx.hidden + 1] = __float2bfloat16(static_cast<float>(t + 1));
    }
    std::vector<float> h_w(static_cast<size_t>(kMxTokens) * kMxTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w,   h_w.data(),   h_w.size()   * sizeof(float),        cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv,   0, static_cast<size_t>(kMxSlots) * fx.hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0, static_cast<size_t>(kMxSlots) * kMxTopK * sizeof(float)));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr, *t_recv_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,    2, ncclBfloat16, d_tok,    kMxTokens, fx.hidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,   2, ncclBfloat16, d_recv,   kMxSlots,  fx.hidden));
    NCCL_ASSERT(epTensorCreate(&t_w,      2, ncclFloat32,  d_w,      kMxTokens, kMxTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32,  d_recv_w, kMxSlots));

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t  dcfg  = NCCL_EP_DISPATCH_CONFIG_INIT;
    d_in.tokens        = t_tok;
    d_in.topk_weights  = t_w;
    d_out.tokens       = t_recv;
    d_out.topk_weights = t_recv_w;
    EXPECT_EQ(ncclEpDispatch(fx.handle, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(fx.handle, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    std::vector<nv_bfloat16> h_recv(static_cast<size_t>(kMxSlots) * fx.hidden);
    CUDA_ASSERT(cudaMemcpy(h_recv.data(), d_recv, h_recv.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    fx.slots.clear();
    for (int s = 0; s < kMxSlots; ++s) {
        const float tag_rank = __bfloat162float(h_recv[static_cast<size_t>(s) * fx.hidden + 0]);
        const float tag_tok  = __bfloat162float(h_recv[static_cast<size_t>(s) * fx.hidden + 1]);
        const int src = static_cast<int>(std::lround(tag_rank)) - 1;
        const int tok = static_cast<int>(std::lround(tag_tok))  - 1;
        if (src < 0 || src >= g_nranks || tok < 0 || tok >= kMxTokens) continue;   // padding
        SlotInfo si{s, src, tok, 0};
        // Ordinal: how many earlier slots already hold this same (src, token).
        for (const SlotInfo& prev : fx.slots)
            if (prev.src_rank == src && prev.token == tok) ++si.ordinal;
        fx.slots.push_back(si);
    }

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv_w);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_w); cudaFree(d_recv_w);
}

// Build a fixture whose top-k table sends every token to `experts` (entries < 0
// are dropped), then dispatch to warm the handle and learn the slot map.
static void make_fixture(Fixture& fx, int hidden, const int (&experts)[kMxTopK]) {
    fx.hidden = hidden;
    fx.group = make_group(hidden);
    ASSERT_NE(fx.group, nullptr);

    std::vector<int64_t> h_topk(static_cast<size_t>(kMxTokens) * kMxTopK);
    for (int t = 0; t < kMxTokens; ++t)
        for (int k = 0; k < kMxTopK; ++k)
            h_topk[static_cast<size_t>(t) * kMxTopK + k] = experts[k];

    CUDA_ASSERT(cudaMalloc(&fx.d_topk, h_topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(fx.d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    NCCL_ASSERT(epTensorCreate(&fx.t_topk, 2, ncclInt64, fx.d_topk, kMxTokens, kMxTopK));

    EXPECT_EQ(ncclEpCreateHandle(&fx.handle, fx.group, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                 fx.t_topk, nullptr, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    ASSERT_NE(fx.handle, nullptr);

    dispatch_and_discover_slots(fx);
}

static void destroy_fixture(Fixture& fx) {
    if (fx.handle) ncclEpHandleDestroy(fx.handle);
    if (fx.t_topk) ncclEpTensorDestroy(fx.t_topk);
    if (fx.d_topk) cudaFree(fx.d_topk);
    if (fx.group)  ncclEpGroupDestroy(fx.group);
    fx = Fixture{};
}

// Run combine with MXFP8 over caller-provided expert rows and return the output.
// `rows` is [kMxSlots][fx.hidden] in host order; only discovered slots matter.
static void run_combine(const Fixture& fx, const std::vector<float>& rows, std::vector<float>& out) {
    nv_bfloat16 *d_in = nullptr, *d_out = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_in,  static_cast<size_t>(kMxSlots)  * fx.hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out, static_cast<size_t>(kMxTokens) * fx.hidden * sizeof(nv_bfloat16)));

    std::vector<nv_bfloat16> h_in(static_cast<size_t>(kMxSlots) * fx.hidden);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = __float2bfloat16(rows[i]);
    CUDA_ASSERT(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_out, 0, static_cast<size_t>(kMxTokens) * fx.hidden * sizeof(nv_bfloat16)));

    ncclEpTensor_t *t_in = nullptr, *t_out = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_in,  2, ncclBfloat16, d_in,  kMxSlots,  fx.hidden));
    NCCL_ASSERT(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kMxTokens, fx.hidden));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t  ccfg  = NCCL_EP_COMBINE_CONFIG_INIT;
    c_in.tokens  = t_in;     // BF16 in; the library quantizes internally
    c_in.scales  = nullptr;
    c_out.tokens = t_out;
    ccfg.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;
    EXPECT_EQ(ncclEpCombine(fx.handle, &c_in, &c_out, &ccfg, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    std::vector<nv_bfloat16> h_out(static_cast<size_t>(kMxTokens) * fx.hidden);
    CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    out.resize(h_out.size());
    for (size_t i = 0; i < h_out.size(); ++i) out[i] = __bfloat162float(h_out[i]);

    ncclEpTensorDestroy(t_in);
    ncclEpTensorDestroy(t_out);
    cudaFree(d_in); cudaFree(d_out);
}

// Compare one token's row against a reference row, block by block, with the
// case name attached so a failure names the case rather than an element index.
template <typename NameFn>
static void expect_row_matches(const std::vector<float>& out, int hidden, int token,
                               const std::vector<float>& ref, NameFn name_of_block) {
    for (int b = 0; b < hidden / kBlk; ++b) {
        SCOPED_TRACE(std::string("H=") + std::to_string(hidden) +
                     " token=" + std::to_string(token) +
                     " block=" + std::to_string(b) + " case=" + name_of_block(b));
        for (int i = 0; i < kBlk; ++i) {
            const int h = b * kBlk + i;
            EXPECT_FLOAT_EQ(out[static_cast<size_t>(token) * hidden + h], ref[h])
                << "element " << i << " of block";
        }
    }
}

// MXFP8 needs THREE things, not two: the HT algorithm, the expert-major layout,
// and the expert-major mode resolved to kLocalPermute. The first two are set
// explicitly (make_group / make_fixture). The third is a group property
// decided inside ncclEpCreateGroup and is not settable through the public
// config, so it can only be relied on -- and it can be taken away:
//
//   NCCL_EP_HT_EM_NVLINK_DUP=1 -> kNvlinkDup
//   NCCL_EP_HT_EM_LOCAL_DUP=1  -> kLocalDup
//   zero_copy == ON            -> kNvlinkDup / kLocalDup
//   otherwise                  -> kLocalPermute   <- what these tests need
//
// The quantizing prologue only exists on the local-permute path, so under either
// env var every combine here would return ncclInvalidArgument. Skip loudly rather
// than failing with a misleading "expected ncclSuccess".
//
// run_tests.sh re-runs suites under LOCAL_DUP and NVLINK_DUP when a suite's third
// registration field is 1; this suite is registered with 0 so the sweep excludes
// it. This guard covers running the binary directly with those vars set.
static const char* em_mode_override() {
    for (const char* v : {"NCCL_EP_HT_EM_NVLINK_DUP", "NCCL_EP_HT_EM_LOCAL_DUP"}) {
        const char* s = std::getenv(v);
        if (s && *s && std::strcmp(s, "0") != 0) return v;
    }
    return nullptr;
}

class Mxfp8ConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const char* v = em_mode_override())
            GTEST_SKIP() << v << " forces a non-local-permute expert-major mode; "
                            "MXFP8 has no quantizing prologue there";
        if (kNumExperts % g_nranks != 0)
            GTEST_SKIP() << "needs experts divisible by ranks (have " << kNumExperts
                         << " experts, " << g_nranks << " ranks)";
    }
};

// ===========================================================================
// Section 5 — tests
// ===========================================================================

// --- Group 1: one contributor per token -----------------------------------
//
// top-k = {expert, -1}: the second entry is dropped, so no summation happens and
// the output is exactly one quantize/dequantize round trip. This isolates the
// conversion from the reduction.
//
// Run twice: once routing to a local expert (the contribution never leaves the
// GPU) and once to a remote rank (it crosses the NVLink leg as a packed MXFP8
// row). Both must produce identical numbers.
static void run_single_contributor_case(int target_rank, int hidden, const char* what) {
    SCOPED_TRACE(std::string(what) + " (H=" + std::to_string(hidden) + ")");

    const int experts[kMxTopK] = { expert_on_rank(target_rank, 0), -1 };
    Fixture fx;
    make_fixture(fx, hidden, experts);
    if (::testing::Test::HasFatalFailure()) return;

    // Build the reference row once: every contributor carries the same case
    // table, so every output token has the same expectation.
    std::vector<float> ref(hidden);
    std::vector<float> rows(static_cast<size_t>(kMxSlots) * fx.hidden, 0.0f);
    for (int b = 0; b < hidden / kBlk; ++b) {
        float in[kBlk];
        single_case(b).fill(in);
        for (int i = 0; i < kBlk; ++i)
            ASSERT_TRUE(bf16_exact(in[i])) << single_case(b).name << " element " << i
                                           << " is not BF16-exact (" << in[i] << ")";
        host_mxfp8_block(in, ref.data() + b * kBlk, kBlk);
        for (const SlotInfo& s : fx.slots)
            for (int i = 0; i < kBlk; ++i)
                rows[static_cast<size_t>(s.slot) * fx.hidden + b * kBlk + i] = in[i];
    }

    std::vector<float> out;
    run_combine(fx, rows, out);
    for (int t = 0; t < kMxTokens; ++t)
        expect_row_matches(out, hidden, t, ref, [](int b) { return single_case(b).name; });

    destroy_fixture(fx);
}

TEST_F(Mxfp8ConversionTest, SingleContributorLocal) {
    for (int h : kHiddenSizes)
        run_single_contributor_case(g_rank, h, "contribution stays on the local GPU");
}

TEST_F(Mxfp8ConversionTest, SingleContributorRemote) {
    for (int h : kHiddenSizes)
        run_single_contributor_case(next_rank(), h, "contribution crosses to a remote rank");
}

// --- Group 2: two contributors on the SAME rank ----------------------------
//
// Both top-k entries route to two different experts on one rank, so
// local_permute_reduce sums them in FP32 and quantizes the SUM once.
// Expected output is Q(A + B).
static void run_same_rank_case(int target_rank, int hidden, const char* what) {
    SCOPED_TRACE(std::string(what) + " (H=" + std::to_string(hidden) + ")");

    if (experts_per_rank() < 2)
        GTEST_SKIP() << "needs >= 2 experts per rank (have " << experts_per_rank() << ")";

    const int experts[kMxTopK] = { expert_on_rank(target_rank, 0), expert_on_rank(target_rank, 1) };
    Fixture fx;
    make_fixture(fx, hidden, experts);
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<float> ref(hidden);
    std::vector<float> rows(static_cast<size_t>(kMxSlots) * fx.hidden, 0.0f);
    for (int b = 0; b < hidden / kBlk; ++b) {
        float a[kBlk], bb[kBlk], sum[kBlk];
        pair_case(b).fill(a, bb);
        for (int i = 0; i < kBlk; ++i) {
            ASSERT_TRUE(bf16_exact(a[i]) && bf16_exact(bb[i]));
            sum[i] = a[i] + bb[i];
        }
        host_mxfp8_block(sum, ref.data() + b * kBlk, kBlk);   // Q(A+B)

        // The two slots holding the same token are distinguished by ordinal.
        // Only their sum is asserted, so which slot gets A and which gets B
        // does not affect the expectation.
        for (const SlotInfo& s : fx.slots) {
            const float* src = (s.ordinal == 0) ? a : bb;
            for (int i = 0; i < kBlk; ++i)
                rows[static_cast<size_t>(s.slot) * fx.hidden + b * kBlk + i] = src[i];
        }
    }

    std::vector<float> out;
    run_combine(fx, rows, out);
    for (int t = 0; t < kMxTokens; ++t)
        expect_row_matches(out, hidden, t, ref, [](int b) { return pair_case(b).name; });

    destroy_fixture(fx);
}

TEST_F(Mxfp8ConversionTest, SameRankAccumulationLocal) {
    for (int h : kHiddenSizes)
        run_same_rank_case(g_rank, h, "both experts on the local GPU");
}

TEST_F(Mxfp8ConversionTest, SameRankAccumulationRemote) {
    for (int h : kHiddenSizes)
        run_same_rank_case(next_rank(), h, "both experts on one remote rank");
}

// --- Group 3: two contributors on DIFFERENT ranks --------------------------
//
// Routing: top-k[0] -> an expert on this rank, top-k[1] -> an expert on the next
// rank. Each side quantizes its own contribution independently, and the two are
// summed in FP32 only after dequantization. Expected output is Q(A) + Q(B).
//
// Because every rank uses the same routing rule, a receiving rank can tell which
// contribution a slot represents from the source-rank tag alone: a token from
// itself arrived via top-k[0] (contribution A), a token from the previous rank
// arrived via top-k[1] (contribution B).
static void run_cross_rank_case(int hidden) {
    SCOPED_TRACE(std::string("H=") + std::to_string(hidden));

    const int experts[kMxTopK] = { expert_on_rank(g_rank, 0), expert_on_rank(next_rank(), 0) };
    Fixture fx;
    make_fixture(fx, hidden, experts);
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<float> ref(hidden);
    std::vector<float> rows(static_cast<size_t>(kMxSlots) * fx.hidden, 0.0f);
    for (int b = 0; b < hidden / kBlk; ++b) {
        float a[kBlk], bb[kBlk], qa[kBlk], qb[kBlk];
        pair_case(b).fill(a, bb);
        for (int i = 0; i < kBlk; ++i) ASSERT_TRUE(bf16_exact(a[i]) && bf16_exact(bb[i]));

        host_mxfp8_block(a,  qa, kBlk);
        host_mxfp8_block(bb, qb, kBlk);
        for (int i = 0; i < kBlk; ++i) ref[b * kBlk + i] = qa[i] + qb[i];   // Q(A) + Q(B)

        for (const SlotInfo& s : fx.slots) {
            const float* src = (s.src_rank == g_rank) ? a : bb;
            for (int i = 0; i < kBlk; ++i)
                rows[static_cast<size_t>(s.slot) * fx.hidden + b * kBlk + i] = src[i];
        }
    }

    std::vector<float> out;
    run_combine(fx, rows, out);
    for (int t = 0; t < kMxTokens; ++t)
        expect_row_matches(out, hidden, t, ref, [](int b) { return pair_case(b).name; });

    destroy_fixture(fx);
}

TEST_F(Mxfp8ConversionTest, CrossRankAccumulation) {
    for (int h : kHiddenSizes) run_cross_rank_case(h);
}

// --- Group 4: non-finite propagation ---------------------------------------
//
// Two behaviours here are counter-intuitive enough to be worth pinning, and
// both follow from the code rather than from guesswork:
//
//  1. An Inf DESTROYS ITS WHOLE BLOCK. amax = Inf sends float_to_e8m0 to its
//     0xFE cap, so the encode multiplier is 2^-127 and every OTHER element in
//     the block underflows to zero. The Inf itself saturates to 448 and decodes
//     as 448 * 2^127, which overflows FP32 back to Inf. The interesting
//     assertion is about the neighbours, not about Inf surviving.
//
//  2. A lone NaN DOES NOT AFFECT THE SCALE. fmaxf returns the non-NaN operand,
//     so the block amax skips it and the other elements are untouched. If NaN
//     did reach the amax, e8m0 would hit the same 0xFE cap and those elements
//     would come back as zeros instead of their exact values -- which is what
//     makes this a real discriminator rather than a tautology.
//
// Blocks are quantized independently, so a poisoned block must not disturb its
// neighbours. Benign blocks are interleaved to check that, each with a distinct
// amax so a stray scale index cannot be masked by a repeated scale byte.
//
// Exactly ONE assertion below is toolkit-dependent and is flagged inline:
// whether cvt.rn.satfinite.e4m3x2.f32 emits the E4M3 NaN encoding or clamps to
// 448. If that is the only failure here, the quantizer is fine and the
// toolkit's NaN policy changed.
static void run_nonfinite_case(int hidden) {
    SCOPED_TRACE(std::string("H=") + std::to_string(hidden));
    ASSERT_GE(hidden / kBlk, 5) << "needs at least five blocks";

    const int experts[kMxTopK] = { expert_on_rank(g_rank, 0), -1 };
    Fixture fx;
    make_fixture(fx, hidden, experts);
    if (::testing::Test::HasFatalFailure()) return;

    // Benign fillers. Each is a power of two whose block quantizes exactly
    // (v/scale == 256) and whose E8M0 byte differs from both neighbours:
    //   8 -> e8m0 122,  32 -> 124,  2 -> 120,  and the Inf block sits at 254.
    constexpr float kBenign[3] = {8.0f, 32.0f, 2.0f};
    constexpr int   kBenignBlk[3] = {0, 2, 4};
    constexpr int   kInfBlk = 1;
    constexpr int   kNanBlk = 3;

    std::vector<float> rows(static_cast<size_t>(kMxSlots) * fx.hidden, 0.0f);
    auto put_block = [&](int b, const float* v) {
        for (const SlotInfo& s : fx.slots)
            for (int i = 0; i < kBlk; ++i)
                rows[static_cast<size_t>(s.slot) * fx.hidden + b * kBlk + i] = v[i];
    };

    float blk[kBlk];
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < kBlk; ++i) blk[i] = kBenign[k];
        put_block(kBenignBlk[k], blk);
    }
    // Inf alongside ordinary values: the ordinary values are what we watch.
    for (int i = 0; i < kBlk; ++i) blk[i] = 1.0f;
    blk[0] = std::numeric_limits<float>::infinity();
    put_block(kInfBlk, blk);
    // NaN alongside a value that quantizes exactly, so any change to the block
    // scale is immediately visible.
    for (int i = 0; i < kBlk; ++i) blk[i] = kBenign[0];
    blk[0] = std::numeric_limits<float>::quiet_NaN();
    put_block(kNanBlk, blk);

    std::vector<float> out;
    run_combine(fx, rows, out);

    for (int t = 0; t < kMxTokens; ++t) {
        const float* row = out.data() + static_cast<size_t>(t) * hidden;
        SCOPED_TRACE(std::string("token=") + std::to_string(t));

        // Blocks are independent: a poisoned block must not leak into its
        // neighbours on either side.
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < kBlk; ++i)
                EXPECT_FLOAT_EQ(row[kBenignBlk[k] * kBlk + i], kBenign[k])
                    << "benign block " << kBenignBlk[k] << " element " << i
                    << " was disturbed by a neighbouring non-finite block";

        // 1. Inf wipes out the rest of its block.
        EXPECT_TRUE(std::isinf(row[kInfBlk * kBlk + 0]))
            << "Inf should saturate to 448 and decode as 448*2^127, which "
               "overflows FP32 back to Inf; got " << row[kInfBlk * kBlk + 0];
        for (int i = 1; i < kBlk; ++i)
            EXPECT_FLOAT_EQ(row[kInfBlk * kBlk + i], 0.0f)
                << "element " << i << " shares a block with Inf: the 2^-127 encode "
                   "multiplier must underflow it to zero";

        // 2. NaN leaves the block scale alone. This is the assertion that
        //    proves fmaxf skipped it -- if it had not, these would all be zero.
        for (int i = 1; i < kBlk; ++i)
            EXPECT_FLOAT_EQ(row[kNanBlk * kBlk + i], kBenign[0])
                << "element " << i << " shares a block with NaN; the block amax "
                   "must ignore NaN so the scale is unchanged";

        // TOOLKIT-DEPENDENT (see the comment above this function).
        EXPECT_TRUE(std::isnan(row[kNanBlk * kBlk + 0]))
            << "expected NaN to propagate through its own payload byte; got "
            << row[kNanBlk * kBlk + 0]
            << " (a value of 14 would mean satfinite clamped to 448 instead of "
               "emitting the E4M3 NaN encoding)";
    }

    destroy_fixture(fx);
}

// Behaviour is per-block, so one hidden size is enough here.
TEST_F(Mxfp8ConversionTest, NonFinitePropagation) {
    run_nonfinite_case(kHiddenSizes[0]);
}

// --- The paired contrast ---------------------------------------------------
//
// Same inputs, different placement, provably different answer. This is the
// assertion that pins down WHERE quantization happens; if it ever fails while
// the group 2 and 3 tests pass, the reduction and the prologue have been
// reordered relative to each other.
//
//   48 and 2 on one rank        -> Q(48 + 2) = Q(50) = 48   (the +2 is lost)
//   48 and 2 on separate ranks  -> Q(48) + Q(2) = 48 + 2 = 50
TEST(Mxfp8ContractModel, CrossRankPreservesWhatSameRankLoses) {
    float a[kBlk], b[kBlk];
    pair_48_and_2(a, b);

    float sum[kBlk], q_of_sum[kBlk];
    for (int i = 0; i < kBlk; ++i) sum[i] = a[i] + b[i];
    host_mxfp8_block(sum, q_of_sum, kBlk);
    EXPECT_FLOAT_EQ(q_of_sum[0], 48.0f) << "50 needs 6 significant bits; E4M3 keeps 4";

    float qa[kBlk], qb[kBlk];
    host_mxfp8_block(a, qa, kBlk);
    host_mxfp8_block(b, qb, kBlk);
    EXPECT_FLOAT_EQ(qa[0], 48.0f) << "48 is exactly representable";
    EXPECT_FLOAT_EQ(qb[0], 2.0f)  << "2 is exactly representable";
    EXPECT_FLOAT_EQ(qa[0] + qb[0], 50.0f);

    EXPECT_NE(q_of_sum[0], qa[0] + qb[0])
        << "placement must change the result; if these are equal the test has "
           "lost its discriminating power and needs new values";
}

// Cross-rank reduction stops error from COMPOUNDING; it does not remove the
// per-contributor error. Documented here so nobody reads the test above as
// "cross-rank is lossless".
TEST(Mxfp8ContractModel, CrossRankStillCarriesPerContributorError) {
    float a[kBlk], b[kBlk], qa[kBlk], qb[kBlk];
    pair_50_and_50(a, b);
    host_mxfp8_block(a, qa, kBlk);
    host_mxfp8_block(b, qb, kBlk);
    EXPECT_FLOAT_EQ(qa[0], 48.0f);
    EXPECT_FLOAT_EQ(qa[0] + qb[0], 96.0f) << "not 100 — each side lost the same 2";
}

/*
 * Inf/NaN is covered by NonFinitePropagation (group 4). Exactly one assertion
 * there is toolkit-dependent -- whether cvt.rn.satfinite.e4m3x2.f32 emits the
 * E4M3 NaN encoding or clamps to 448 -- and it is flagged inline so a CUDA
 * upgrade that changes the policy fails one line with a self-explanatory
 * message rather than looking like a quantizer bug.
 */

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_mxfp8_combine_conversion_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
