# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Process-wide runtime state shared by extension facades and bindings."""

from __future__ import annotations

import threading


NATIVE_CALL_LOCK = threading.RLock()

__all__ = ["NATIVE_CALL_LOCK"]
