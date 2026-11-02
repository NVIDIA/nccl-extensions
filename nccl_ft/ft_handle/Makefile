#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../../makefiles/common.mk
include ../../../makefiles/examples.mk

TARGET_SO = libft_collective.so
TARGET_A  = libft_collective.a

INCLUDES += -I.
SOFLAGS   = -Xcompiler -fPIC --shared
# nvcc does not accept -Wl,... directly; convert rpath for the linker via -Xlinker.
NVCC_LDFLAGS = -lcudart -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

all: $(TARGET_SO) $(TARGET_A)

# Shared library — linked directly from the source so nvcc handles device
# linking automatically (no separate dlink step needed for a single .cu).
$(TARGET_SO): ft_collective.cu
	$(NVCC) $(NVCUFLAGS) $(SOFLAGS) $(INCLUDES) $< -o $@ $(LIBRARIES) $(NVCC_LDFLAGS)
	@echo "Built $@"

# Static archive — for direct linkage into executables.
ft_collective.o: ft_collective.cu
	$(NVCC) $(NVCUFLAGS) -Xcompiler -fPIC $(INCLUDES) -c $< -o $@

$(TARGET_A): ft_collective.o
	ar rcs $@ $<
	@echo "Built $@"

clean:
	rm -f ft_collective.o $(TARGET_SO) $(TARGET_A)

help:
	@echo "FTCollectiveHandle library"
	@echo "=========================="
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build libft_collective.so and libft_collective.a (default)"
	@echo "  clean  - Remove build artifacts"
	@echo ""
	@echo "Set NCCL_HOME and CUDA_HOME before building."

.PHONY: all clean help
