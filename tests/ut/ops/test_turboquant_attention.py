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
"""The attention call that hands a gated layer's gate to a TurboQuant Cube decode."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops import turboquant_attention as op_module
from vllm_ascend.utils import TURBOQUANT_GATED_ATTENTION_OP

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 16
LAYER_NAME = "model.layers.3.self_attn.attn"


class TestTurboQuantGatedAttention(TestBase):
    def _attn(self, fuses: bool):
        return SimpleNamespace(
            impl=SimpleNamespace(fuses_output_gate=fuses),
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            layer_name=LAYER_NAME,
        )

    def test_the_gate_goes_to_the_backend_only_when_it_fuses_it(self):
        self.assertTrue(op_module.turboquant_fuses_output_gate(self._attn(True)))
        self.assertFalse(op_module.turboquant_fuses_output_gate(self._attn(False)))
        # A backend without the attribute (every non-TurboQuant impl) keeps the model's own gate.
        self.assertFalse(op_module.turboquant_fuses_output_gate(SimpleNamespace(impl=object())))
        self.assertFalse(op_module.turboquant_fuses_output_gate(SimpleNamespace()))

    def test_the_forward_reshapes_like_attention_and_names_the_layer(self):
        num_tokens = 3
        query = torch.randn(num_tokens, NUM_HEADS * HEAD_SIZE)
        key = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE)
        value = torch.randn_like(key)
        gate = torch.randn(num_tokens, NUM_HEADS * HEAD_SIZE)
        op = MagicMock()

        def fill(q, k, v, g, out, name):
            out.copy_(g)

        op.side_effect = fill
        with patch.object(torch.ops.vllm, "turboquant_gated_attention", op, create=True):
            output = op_module.turboquant_gated_attention_forward(self._attn(True), query, key, value, gate)

        args = op.call_args.args
        self.assertEqual(args[0].shape, (num_tokens, NUM_HEADS, HEAD_SIZE))
        self.assertEqual(args[1].shape, (num_tokens, NUM_KV_HEADS, HEAD_SIZE))
        self.assertEqual(args[2].shape, (num_tokens, NUM_KV_HEADS, HEAD_SIZE))
        self.assertEqual(args[3].shape, (num_tokens, NUM_HEADS, HEAD_SIZE))
        self.assertEqual(args[5], LAYER_NAME)
        # The op writes the flat output o_proj reads through its 3-D view.
        self.assertEqual(output.shape, (num_tokens, NUM_HEADS * HEAD_SIZE))
        torch.testing.assert_close(output, gate)

    def test_the_op_calls_the_impl_with_the_gate(self):
        impl = MagicMock()
        layer = SimpleNamespace(impl=impl)
        metadata, kv_cache = object(), object()
        query = torch.randn(2, NUM_HEADS, HEAD_SIZE)
        key = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn_like(key)
        gate = torch.randn_like(query)
        output = torch.empty_like(query)
        with patch(
            "vllm.model_executor.layers.attention.attention.get_attention_context",
            return_value=(metadata, layer, kv_cache, None),
        ) as context:
            op_module.turboquant_gated_attention(query, key, value, gate, output, LAYER_NAME)

        context.assert_called_once_with(LAYER_NAME)
        impl.forward.assert_called_once()
        call = impl.forward.call_args
        self.assertIs(call.args[0], layer)
        self.assertIs(call.args[4], kv_cache)
        self.assertIs(call.args[5], metadata)
        self.assertIs(call.kwargs["output"], output)
        self.assertIs(call.kwargs["output_gate"], gate)

    def test_the_op_name_is_the_registered_splitting_op(self):
        namespace, name = TURBOQUANT_GATED_ATTENTION_OP.split("::")
        self.assertEqual(namespace, "vllm")
        self.assertTrue(hasattr(getattr(torch.ops, namespace), name))
