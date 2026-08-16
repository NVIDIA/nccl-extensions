# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Private CI unit tests for the quantized-tensor prototype (issue #7).
# Only the scale-plane and quantize suites from the nccl-rl prototype are
# wired here; other leftover unit sources under unit/ are not part of this
# build until they are re-ported against nccl-extensions.

include ../../Makefile.common
include ../gtest.mk

CI_UNIT_TEST_SRCS := $(TESTSDIR)/ci/unit/scale_plane_test.cc \
                     $(TESTSDIR)/ci/unit/quantize_test.cc
CI_UNIT_TEST_MAIN_SRC := $(TESTSDIR)/ci/unit_tests_main.cc
CI_UNIT_HOST_GTEST_FILTER := ScalePlane*
CI_UNIT_GPU_GTEST_FILTER := Quantize*

.PHONY: all unit_tests check check-gpu clean help

all: unit_tests

unit_tests: $(BINDIR)/unit_tests ; @:

$(BINDIR)/unit_tests: $(CI_UNIT_TEST_MAIN_SRC) $(CI_UNIT_TEST_SRCS) \
                      $(LIB_SRCS) $(GTEST_A) $(LIB_HDRS)
	@printf "Building private CI unit_tests (scale-plane + quantize)...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -DNCCL_M2N_TESTING $(MPI_CFLAGS) -I$(TESTSDIR) -I$(GTEST_INC) \
	    $(CI_UNIT_TEST_MAIN_SRC) $(CI_UNIT_TEST_SRCS) $(LIB_SRCS) $(GTEST_A) \
	    $(LIB_LDFLAGS) -lpthread -ldl
	@printf "Done! Test binary at: $@\n"

check: $(BINDIR)/unit_tests
	$(BINDIR)/unit_tests --gtest_filter='$(CI_UNIT_HOST_GTEST_FILTER)'

check-gpu: $(BINDIR)/unit_tests
	$(BINDIR)/unit_tests --gtest_filter='$(CI_UNIT_GPU_GTEST_FILTER)'

clean:
	rm -f $(BINDIR)/unit_tests

help:
	@echo "tests/ci/Makefile — private CI unit tests for issue #7"
	@echo "  unit_tests / check / check-gpu / clean / help"
