#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

.DEFAULT_GOAL := all

EXTENSIONS := nccl_ep nccl_m2n nccl_ft

.PHONY: all clean nccl-submodule

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
NCCL_SUBMODULE_HOME ?= $(REPO_ROOT)/third_party/nccl
BUILDDIR ?= $(REPO_ROOT)/build
NCCL_BUILDDIR ?= $(NCCL_SUBMODULE_HOME)/build

ifeq ($(origin NCCL_HOME),undefined)
NCCL_BUILD_PREREQUISITE := nccl-submodule
endif

NCCL_HOME ?= $(NCCL_BUILDDIR)
NCCL_EP_BUILDDIR ?= $(BUILDDIR)
NCCL_M2N_BUILDDIR ?= $(BUILDDIR)

ABS_NCCL_BUILDDIR := $(abspath $(NCCL_BUILDDIR))
ABS_NCCL_EP_BUILDDIR := $(abspath $(NCCL_EP_BUILDDIR))
ABS_NCCL_M2N_BUILDDIR := $(abspath $(NCCL_M2N_BUILDDIR))
ABS_NCCL_HOME := $(abspath $(NCCL_HOME))

all: $(EXTENSIONS:%=%.build)
clean: $(EXTENSIONS:%=%.clean)

$(EXTENSIONS:%=%.build): $(NCCL_BUILD_PREREQUISITE)

nccl_ep.%:
	$(MAKE) -C $(REPO_ROOT)/nccl_ep $* \
		NCCL_HOME=$(ABS_NCCL_HOME) \
		BUILDDIR=$(ABS_NCCL_EP_BUILDDIR)

nccl_m2n.%:
	$(MAKE) -C $(REPO_ROOT)/nccl_m2n $* \
		NCCL_HOME=$(ABS_NCCL_HOME) \
		BUILDDIR=$(ABS_NCCL_M2N_BUILDDIR)

# nccl_ft keeps its artifacts in-tree (ft_handle/libft_collective.{so,a}).
nccl_ft.%:
	$(MAKE) -C $(REPO_ROOT)/nccl_ft $* \
		NCCL_HOME=$(ABS_NCCL_HOME)

nccl-submodule:
	@if [ ! -e "$(NCCL_SUBMODULE_HOME)/.git" ]; then \
		git -C "$(REPO_ROOT)" submodule update --init third_party/nccl; \
	fi
	$(MAKE) -C $(NCCL_SUBMODULE_HOME) -j src.build \
		BUILDDIR=$(ABS_NCCL_BUILDDIR)
