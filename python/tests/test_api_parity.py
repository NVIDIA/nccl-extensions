# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks that generated bindings and public facades stay in sync."""

from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import nccl.ep as ep
from nccl._extensions.bindings import nccl_ep as ep_bindings
from nccl.core.typing import NcclDataType


class EpApiParityTest(unittest.TestCase):
    def test_public_enums_match_generated_bindings(self) -> None:
        enum_pairs = (
            (ep.Algorithm, ep_bindings.Algorithm, {}),
            (ep.Layout, ep_bindings.Layout, {}),
            (
                ep.PassDir,
                ep_bindings.PassDir,
                {"FWD_PASS": "FWD", "BWD_PASS": "BWD"},
            ),
            (
                ep.OverflowPolicy,
                ep_bindings.OverflowPolicy,
                {
                    "OVERFLOW_AUTO": "AUTO",
                    "OVERFLOW_TRAP": "TRAP",
                    "OVERFLOW_DROP": "DROP",
                },
            ),
            (
                ep.ZeroCopyMode,
                ep_bindings.ZeroCopyMode,
                {
                    "ZERO_COPY_AUTO": "AUTO",
                    "ZERO_COPY_OFF": "OFF",
                    "ZERO_COPY_ON": "ON",
                },
            ),
            (ep.HtFlow, ep_bindings.HtFlow, {}),
            (
                ep.MetaExchangePolicy,
                ep_bindings.MetaExchangePolicy,
                {
                    "META_EXCHANGE_AUTO": "AUTO",
                    "META_EXCHANGE_REQUIRE_FUSED": "REQUIRE_FUSED",
                    "META_EXCHANGE_FORCE_SCAN": "FORCE_SCAN",
                },
            ),
            (
                ep.ExpertIdKind,
                ep_bindings.ExpertIdKind,
                {
                    "EXPERT_ID_AUTO": "AUTO",
                    "EXPERT_ID_LOCAL": "LOCAL",
                    "EXPERT_ID_GLOBAL": "GLOBAL",
                },
            ),
            (
                ep.DispatchQuantizationRecipe,
                ep_bindings.DispQuant,
                {},
            ),
            (
                ep.CombineQuantizationRecipe,
                ep_bindings.CombQuant,
                {},
            ),
        )

        for public, generated, aliases in enum_pairs:
            with self.subTest(public=public.__name__):
                generated_to_public = {
                    name: aliases.get(name, name) for name in generated.__members__
                }
                self.assertEqual(
                    set(public.__members__), set(generated_to_public.values())
                )
                for generated_name, public_name in generated_to_public.items():
                    self.assertEqual(
                        int(public[public_name]), int(generated[generated_name])
                    )

    def test_facade_dataclass_fields_exist_in_generated_structs(self) -> None:
        facade_pairs = (
            (ep.AllocConfig, ep_bindings.AllocConfig),
            (ep.GroupConfig, ep_bindings.GroupConfig),
            (ep.HandleConfig, ep_bindings.HandleConfig),
            (ep.HandleState, ep_bindings.HandleState),
            (ep.DispatchConfig, ep_bindings.DispatchConfig),
            (ep.CombineConfig, ep_bindings.CombineConfig),
            (ep.LayoutInfo, ep_bindings.LayoutInfo),
            (ep.DispatchInputs, ep_bindings.DispatchInputs),
            (ep.DispatchOutputs, ep_bindings.DispatchOutputs),
            (ep.CombineInputs, ep_bindings.CombineInputs),
            (ep.CombineOutputs, ep_bindings.CombineOutputs),
        )

        for facade, generated in facade_pairs:
            with self.subTest(facade=facade.__name__):
                missing = [
                    field.metadata.get("lowpp", field.name)
                    for field in fields(facade)
                    if not hasattr(
                        generated, field.metadata.get("lowpp", field.name)
                    )
                ]
                self.assertEqual(missing, [])

        state = ep.HandleState(
            num_tokens=8,
            num_topk=2,
            topk_idx_datatype=NcclDataType.INT32,
        )
        self.assertEqual(state._lowpp.num_tokens, 8)
        self.assertEqual(state._lowpp.num_topk, 2)
        self.assertEqual(
            int(state._lowpp.topk_idx_datatype), int(NcclDataType.INT32)
        )

    def test_current_ep_entry_points_are_generated(self) -> None:
        required = {
            "get_version",
            "tensor_alloc",
            "tensor_destroy",
            "create_group",
            "group_destroy",
            "handle_mem_size",
            "init_handle",
            "import_handle",
            "update_handle",
            "create_handle",
            "handle_destroy",
            "dispatch",
            "combine",
            "complete",
            "mask_query",
            "mask_update",
            "mask_clean",
            "get_async_error",
            "error_clear",
        }
        self.assertEqual(
            {name for name in required if not hasattr(ep_bindings, name)}, set()
        )

    def test_split_handle_facade_forwards_and_anchors_memory(self) -> None:
        group = ep.Group(0x1000)
        config = ep.HandleConfig(dispatch_output_per_expert_alignment=8)
        handle_mem = SimpleNamespace(ptr=0x2000)
        state = ep.HandleState(
            num_tokens=8,
            num_topk=2,
            topk_idx_datatype=NcclDataType.INT32,
        )

        with patch.object(ep_bindings, "handle_mem_size", return_value=4096) as call:
            self.assertEqual(
                group.handle_mem_size(
                    ep.Layout.EXPERT_MAJOR, config=config, num_topk=2
                ),
                4096,
            )
            call.assert_called_once_with(
                group.ptr,
                int(ep.Layout.EXPERT_MAJOR),
                config._lowpp.ptr,
                2,
            )

        with patch.object(ep_bindings, "init_handle", return_value=0x3000) as call:
            handle = group.init_handle(
                ep.Layout.EXPERT_MAJOR,
                config=config,
                num_topk=2,
                handle_mem=handle_mem,
            )
            self.assertIs(handle._handle_mem, handle_mem)
            call.assert_called_once_with(
                group.ptr,
                int(ep.Layout.EXPERT_MAJOR),
                config._lowpp.ptr,
                2,
                handle_mem.ptr,
            )

        with patch.object(ep_bindings, "import_handle", return_value=0x4000) as call:
            imported = group.import_handle(
                ep.Layout.EXPERT_MAJOR,
                state,
                handle_mem,
                config=config,
            )
            self.assertIs(imported._handle_mem, handle_mem)
            call.assert_called_once_with(
                group.ptr,
                int(ep.Layout.EXPERT_MAJOR),
                config._lowpp.ptr,
                state._lowpp.ptr,
                handle_mem.ptr,
            )

    def test_from_buffer_preserves_exporter_mutability(self) -> None:
        lowpp_type = ep_bindings.HandleConfig
        dtype = ep_bindings.handle_config_dtype
        field_name = "dispatch_output_per_expert_alignment"
        value = 8

        writable = lowpp_type.from_buffer(bytearray(dtype.itemsize))
        setattr(writable, field_name, value)
        self.assertEqual(getattr(writable, field_name), value)

        readonly = lowpp_type.from_buffer(bytes(dtype.itemsize))
        with self.assertRaisesRegex(ValueError, "read-only"):
            setattr(readonly, field_name, value)


if __name__ == "__main__":
    unittest.main()
