# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

.DEFAULT_GOAL := build
NCCL_ND_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
NCCL_EXTENSIONS_ROOT := $(abspath $(NCCL_ND_ROOT)/..)
NCCL_HOME ?= $(NCCL_EXTENSIONS_ROOT)/third_party/nccl/build
NCCL_ND_BUILDDIR ?= $(NCCL_EXTENSIONS_ROOT)/build
BUILDDIR ?= $(NCCL_ND_BUILDDIR)
CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local

OBJDIR := $(BUILDDIR)/obj/nccl_nd
LIBDIR := $(BUILDDIR)/lib
INCDIR := $(BUILDDIR)/include
HEADER := $(NCCL_ND_ROOT)/include/nccl_nd.h
NCCL_ND_MAJOR := $(shell awk '/^.define NCCL_ND_MAJOR / {print $$3}' $(HEADER))
NCCL_ND_MINOR := $(shell awk '/^.define NCCL_ND_MINOR / {print $$3}' $(HEADER))
NCCL_ND_PATCH := $(shell awk '/^.define NCCL_ND_PATCH / {print $$3}' $(HEADER))
SONAME := libnccl_nd.so.$(NCCL_ND_MAJOR)
LIBVERSION := $(SONAME).$(NCCL_ND_MINOR).$(NCCL_ND_PATCH)
LIBTARGET := $(LIBDIR)/$(LIBVERSION)
OBJECT := $(OBJDIR)/nccl_nd.o

# Cleaning and help do not require CUDA or NCCL.
NO_DEPS_GOALS := clean help
BUILD_GOALS := $(if $(MAKECMDGOALS),$(filter-out $(NO_DEPS_GOALS),$(MAKECMDGOALS)),build)
ifneq ($(strip $(BUILD_GOALS)),)
NVCC ?= $(CUDA_HOME)/bin/nvcc
ifeq ($(wildcard $(NVCC)),)
$(error NVCC not found at $(NVCC). Set CUDA_HOME to CUDA 12.5 or newer)
endif
CXXSTD := -std=c++17
# Keep the default valid on CUDA 12.5; add Blackwell only when nvcc supports it.
ifeq ($(origin NVCC_GENCODE),undefined)
NVCC_GENCODE := -gencode=arch=compute_90,code=sm_90
ifneq ($(filter sm_100,$(shell $(NVCC) --list-gpu-code)),)
NVCC_GENCODE += -gencode=arch=compute_100,code=sm_100
endif
endif
include $(NCCL_EXTENSIONS_ROOT)/makefiles/common.mk
ifneq ($(shell test "0$(CUDA_MAJOR)" -gt 12 -o "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 5; echo $$?),0)
$(error NCCL ND requires CUDA 12.5 or newer; found $(CUDA_VERSION))
endif
ifeq ($(wildcard $(NCCL_HOME)/include/nccl_device.h),)
$(error NCCL_HOME must contain include/nccl_device.h; build the vendored NCCL first)
endif
NCCL_LIB ?= $(firstword $(wildcard $(NCCL_HOME)/lib/libnccl.so) \
                       $(wildcard $(NCCL_HOME)/lib64/libnccl.so) \
                       $(wildcard $(NCCL_HOME)/lib/libnccl.so.*) \
                       $(wildcard $(NCCL_HOME)/lib64/libnccl.so.*))
ifeq ($(wildcard $(NCCL_LIB)),)
$(error NCCL_HOME must contain libnccl.so or a versioned SONAME under lib/ or lib64/)
endif
NVCUFLAGS += --expt-relaxed-constexpr -Xcompiler=-fPIC,-fvisibility=hidden,-Wall,-Wextra \
             -I$(NCCL_ND_ROOT)/include -I$(NCCL_ND_ROOT) \
             -isystem $(NCCL_HOME)/include -isystem $(CUDA_HOME)/include/cccl
NCCL_LDFLAGS := -Xlinker $(abspath $(NCCL_LIB))
endif

.PHONY: all build lib install clean help
all build: lib
lib: $(LIBTARGET) $(LIBDIR)/$(SONAME) $(LIBDIR)/libnccl_nd.so $(INCDIR)/nccl_nd.h

$(OBJECT): $(NCCL_ND_ROOT)/nccl_nd.cu $(NCCL_ND_ROOT)/nccl_nd_layout.hpp $(HEADER)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) -MMD -MP -c $< -o $@

$(LIBTARGET): $(OBJECT) $(NCCL_LIB)
	@mkdir -p $(dir $@)
	$(NVCC) -shared -cudart=shared $(OBJECT) $(NCCL_LDFLAGS) \
	  -Xlinker -z -Xlinker defs -Xlinker -soname -Xlinker $(SONAME) -o $@

$(LIBDIR)/$(SONAME): $(LIBTARGET)
	ln -sfn $(LIBVERSION) $@

$(LIBDIR)/libnccl_nd.so: $(LIBDIR)/$(SONAME)
	ln -sfn $(SONAME) $@

$(INCDIR)/nccl_nd.h: $(HEADER)
	@mkdir -p $(dir $@)
	cp $< $@

install: lib
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 755 $(LIBTARGET) $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(HEADER) $(DESTDIR)$(PREFIX)/include/
	ln -sfn $(LIBVERSION) $(DESTDIR)$(PREFIX)/lib/$(SONAME)
	ln -sfn $(SONAME) $(DESTDIR)$(PREFIX)/lib/libnccl_nd.so

clean:
	rm -f $(OBJECT) $(OBJECT:.o=.d) $(LIBTARGET) $(LIBDIR)/$(SONAME) \
	  $(LIBDIR)/libnccl_nd.so $(INCDIR)/nccl_nd.h

help:
	@echo 'Targets: build (default), lib, install, clean, help'
	@echo 'GPU build: CUDA_HOME (CUDA >=12.5), NCCL_HOME (Device API build), NVCC_GENCODE'
	@echo 'Output: BUILDDIR or NCCL_ND_BUILDDIR (default: repository build/)'

-include $(OBJECT:.o=.d)
