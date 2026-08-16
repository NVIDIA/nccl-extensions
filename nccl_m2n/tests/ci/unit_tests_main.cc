/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Shared main for build/bin/unit_tests.
 *
 * The binary includes host-only unit suites plus single-GPU API-contract
 * suites that use EXPECT_EXIT. Threadsafe death tests force each child to
 * re-exec before touching CUDA/NCCL state.
 ************************************************************************/

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
