#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build against the vendored NCCL (../../third_party/nccl) unless NCCL_HOME is
# set; the root Makefile passes NCCL_HOME explicitly.
NCCL_EXTENSIONS_ROOT := $(abspath ../..)
CXXSTD := -std=c++17
include ../../makefiles/common.mk

NCCL_HOME ?= $(NCCL_EXTENSIONS_ROOT)/third_party/nccl/build
INCLUDES  = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LIBRARIES = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDFLAGS   = -lcudart -lnccl -Wl,-rpath,$(NCCL_HOME)/lib

TARGET = ft_handle_bench

FT_HANDLE_DIR = ../ft_handle

INCLUDES += -I$(FT_HANDLE_DIR)

# Link against the pre-built libft_collective.so in the ft_handle directory.
# Set LD_LIBRARY_PATH=../ft_handle (or FT_LIB_DIR) at runtime, or pass
# FT_LIB_DIR=/path/to/dir to override at build time.
FT_LIB_DIR ?= $(FT_HANDLE_DIR)
LIBRARIES  += -L$(FT_LIB_DIR)
LDFLAGS    += -lft_collective -Wl,-rpath,$(abspath $(FT_LIB_DIR))

SOURCES = main.cu
OBJECTS = main.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -lpthread -o $@
	@echo "Built target $@"

%.o: %.cu
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGET)
	NTHREADS=2 ./$(TARGET) --count 1024 --warmup 10 --iters 50

clean:
	rm -f $(OBJECTS) $(TARGET)

help:
	@echo "FT AllGather Latency Benchmark"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build the benchmark (default)"
	@echo "  test   - Build and run a quick 2-rank test"
	@echo "  clean  - Remove build artifacts"
	@echo "  help   - Show this help"
	@echo ""
	@echo "Benchmark options:"
	@echo "  --count N        Float32 elements per rank (default: 1024)"
	@echo "  --warmup N       Warm-up iterations (default: 20)"
	@echo "  --iters N        Timed iterations (default: 100)"
	@echo "  --timeout-us T   FT kernel timeout µs (default: 10000000)"
	@echo ""
	@echo "Environment:"
	@echo "  NTHREADS         Number of ranks/GPUs (default: all visible GPUs)"
	@echo "  FT_LIB_DIR       Path to libft_collective.so (default: ../ft_handle)"
	@echo ""
	@echo "Examples:"
	@echo "  NTHREADS=4 ./ft_handle_bench --count 16 --iters 200"
	@echo "  NTHREADS=4 ./ft_handle_bench --count 1048576 --timeout-us 5000000"

.PHONY: all test clean help
