# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation of the M2N Python buffer boundary."""

import unittest

from nccl.m2n.tensor import _resolve_buffer


class _TensorLike:
    shape = (4,)
    dtype = "float32"

    def __init__(self, is_cuda):
        self.is_cuda = is_cuda

    def data_ptr(self):
        return 1234

    def is_contiguous(self):
        return True


class M2NTensorDeviceTest(unittest.TestCase):
    def test_cpu_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            _resolve_buffer(_TensorLike(False))

    def test_cuda_tensor_and_raw_pointers_are_accepted(self):
        self.assertEqual(_resolve_buffer(_TensorLike(True))[0], 1234)
        self.assertEqual(_resolve_buffer(1234), (1234, None, None))
        self.assertEqual(_resolve_buffer(None), (0, None, None))

    def test_cuda_array_interface_is_accepted(self):
        class CudaArray:
            __cuda_array_interface__ = {
                "shape": (4,), "typestr": "<f4", "data": (5678, False), "version": 3
            }

        self.assertEqual(_resolve_buffer(CudaArray())[0], 5678)

    def test_real_cpu_torch_tensor_is_rejected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            _resolve_buffer(torch.zeros(4))
