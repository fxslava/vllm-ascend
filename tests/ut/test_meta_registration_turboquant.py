#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""The meta kernels that let a TurboQuant operator be traced rather than run.

Every TurboQuant operator writes through ``Tensor!`` arguments and returns
``()``, so its meta kernel has nothing to allocate -- but it has to exist, or a
FakeTensor reaching one fails with "could not run with Meta backend" instead of
tracing. The served path does not depend on this: both attention entry points
are graph splitting ops, so under piecewise compilation these operators run
eagerly. ``torch.export`` and a full graph do.

These tests drive the registration against a schema defined here rather than the
compiled extension's, because the extension is not importable on a host with no
NPU -- and because the two Cube operators exist only in an Ascend 950 build, so
"which operators are present" is exactly what the registration must not assume.
"""

from unittest.mock import patch

import torch
from torch.library import Library

from tests.ut.base import TestBase

# ``vllm_ascend.meta_registration`` is only ever imported immediately after the
# compiled extension, and its module-level block registers meta kernels for two LoRA
# operators that only exist there -- so importing it on a host with no NPU raises
# "operator _C_ascend::bgmv_expand does not exist". The 310P guard around that block
# is the seam: taking it makes the module importable with nothing registered, which
# is exactly the state these tests want to drive by hand.
with patch("vllm_ascend.utils.is_310p", return_value=True):
    from vllm_ascend import meta_registration

# Named so it cannot collide with a real operator if one is ever loaded beside it.
PROBE_OP = "npu_turboquant_probe_for_meta_registration"
PROBE_SCHEMA = f"{PROBE_OP}(Tensor query, Tensor! out) -> ()"


def _only(*names):
    """Drive the registration against these operator names alone."""
    return patch.object(meta_registration, "TURBOQUANT_MUTATING_OPS", names)


class TestTurboQuantMetaRegistration(TestBase):
    def setUp(self):
        super().setUp()
        self.lib = Library("_C_ascend", "FRAGMENT")
        self.lib.define(PROBE_SCHEMA)
        self.addCleanup(self.lib._destroy)

    @staticmethod
    def _has_meta(op_name: str) -> bool:
        registrations = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
        return f"_C_ascend::{op_name}" in registrations

    def test_an_operator_this_build_registered_gets_a_meta_kernel(self):
        self.assertFalse(self._has_meta(PROBE_OP))
        with _only(PROBE_OP):
            meta_registration.register_turboquant_meta()
        self.assertTrue(self._has_meta(PROBE_OP))

    def test_the_meta_kernel_allocates_nothing_and_returns_nothing(self):
        """The operator's outputs are its own arguments; a meta kernel that
        returned a tensor would be describing an output it does not have."""
        with _only(PROBE_OP):
            meta_registration.register_turboquant_meta()
        query = torch.zeros(2, 4, device="meta")
        out = torch.zeros(2, 4, device="meta")
        self.assertIsNone(getattr(torch.ops._C_ascend, PROBE_OP)(query, out))

    def test_an_operator_this_build_did_not_register_is_skipped(self):
        """The kv4fp8 Cube operators exist only in an Ascend 950 build, and a
        meta kernel for a schema that was never defined raises."""
        with _only("npu_turboquant_no_such_operator"):
            meta_registration.register_turboquant_meta()

    def test_every_mutating_turboquant_operator_is_listed(self):
        """The list is the promise; a new operator that writes through its
        arguments has to be added to it or it will not be traceable."""
        self.assertEqual(
            set(meta_registration.TURBOQUANT_MUTATING_OPS),
            {
                "npu_turboquant_reshape_and_cache",
                "npu_turboquant_rotate_q",
                "npu_turboquant_paged_attention",
                "npu_turboquant_cube_reshape_and_cache",
                "npu_turboquant_cube_decode",
            },
        )
