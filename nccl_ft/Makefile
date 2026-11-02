#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# Top-level build for the fault-tolerant collectives contribution.
#
# The fault-tolerant AllGather/AllReduce library lives in ft_handle/ (see
# README.md). This top-level Makefile builds it first, then the C++ example that
# measures the ft_handle C API (ft_handle_bench, which links -lft_collective).
# Command-line variables (NCCL_HOME, CUDA_HOME) are passed to the sub-makes.

EXAMPLES = ft_handle_bench

all: ft_handle $(EXAMPLES)

# Entry point used by the repository root Makefile (make nccl_ft.build).
build: all

ft_handle:
	$(MAKE) -C ft_handle

$(EXAMPLES): ft_handle
	$(MAKE) -C $@

clean:
	$(MAKE) -C ft_handle clean
	for example in $(EXAMPLES); do \
		$(MAKE) -C $$example clean; \
	done

test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example test; \
	done

help:
	@echo "NCCL FT: Fault-Tolerant Collectives"
	@echo "========================================"
	@echo ""
	@echo "Targets:"
	@echo "  all                - Build the ft_handle library and ft_handle_bench"
	@echo "  ft_handle          - Build only libft_collective (.so/.a)"
	@echo "  clean              - Clean build artifacts"
	@echo "  test               - Build and run ft_handle_bench"
	@echo "  help               - Show this help"
	@echo ""
	@echo "Build options (pass to make):"
	@echo "  NCCL_HOME=<path>   Path to NCCL build (default: vendored third_party/nccl/build)"
	@echo "  CUDA_HOME=<path>   Path to CUDA installation"

.PHONY: all build clean test help ft_handle $(EXAMPLES)
