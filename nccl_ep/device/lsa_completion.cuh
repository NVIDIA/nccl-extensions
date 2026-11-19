// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

namespace nccl_ep {

// Each rank owns one slot: PCIe peer writes do not require peer atomic support.
__device__ __forceinline__ void publish_lsa_completion(uint32_t* slot, uint32_t epoch) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(slot), "r"(epoch) : "memory");
}

__device__ __forceinline__ void wait_lsa_completion(const uint32_t* slots, uint32_t epoch, int ranks) {
    // Collective ordering bounds peer skew to one invocation, well below 2^31.
    // Accept an advanced epoch, including across uint32 wrap.
    for (int peer = 0; peer < ranks; ++peer) {
        uint32_t arrived;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                         : "=r"(arrived) : "l"(slots + peer) : "memory");
        } while (static_cast<int32_t>(arrived - epoch) < 0);
    }
}

} // namespace nccl_ep
