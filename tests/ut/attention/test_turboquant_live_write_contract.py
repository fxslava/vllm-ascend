# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the compact live NPU suite's reference and launch preconditions."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_turboquant_live_write_npu as live


@pytest.mark.parametrize("context", [1, 17, 65])
def test_vectorized_gqa_reference_matches_independent_head_loop(context):
    generator = torch.Generator().manual_seed(context)
    shape = (1, context + live.TILE_ROWS, live.NUM_KV_HEADS, live.HEAD_SIZE)
    key = torch.randn(shape, generator=generator)
    value = torch.randn(shape, generator=generator)
    # Dead rows must not enter the reference's softmax, either.
    key[:, context:] = float("nan")
    value[:, context:] = float("nan")
    query = torch.randn(1, live.NUM_HEADS, live.HEAD_SIZE, generator=generator)
    adapter = SimpleNamespace(
        gather=lambda *args: (key, None, value, None),
        dequantize_cache=lambda packed, *args: packed,
    )
    actual = live._reference_attention_fp32(adapter, None, (), None, query, context, live.ATTENTION_SCALE)
    expected = []
    for head in range(live.NUM_HEADS):
        kv = head // live.GROUP_HEADS
        scores = (key[0, :context, kv] @ query[0, head]) * live.ATTENTION_SCALE
        expected.append(torch.softmax(scores, dim=-1) @ value[0, :context, kv])
    torch.testing.assert_close(actual, torch.stack(expected).unsqueeze(0), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("threshold", [0, live.FUSED_CONTEXT_LIMIT, 4096])
def test_launch_repoisons_scratch_and_shares_builder_threshold(monkeypatch, threshold):
    monkeypatch.setattr(live, "DEVICE", torch.device("cpu"))
    calls = []

    def build_tiling(contexts, tiling, *geometry):
        calls.append(geometry[-1])
        tiling[0], tiling[10] = 1, 1

    monkeypatch.setattr(
        torch, "ops", SimpleNamespace(turboquant_ops=SimpleNamespace(npu_turboquant_build_decode_tiling=build_tiling))
    )
    contexts = (136, 383)
    buffers = live._build_decode_tiling(contexts, 128, 4, 3, threshold)
    query = torch.ones(len(contexts), live.NUM_HEADS, live.HEAD_SIZE, dtype=torch.bfloat16)

    def decode(**kwargs):
        assert kwargs["fused_context_limit"] == calls[-1] == threshold
        for name in ("output", "workspace", "query_rot"):
            assert torch.isnan(kwargs[name]).all(), f"{name} was not poisoned before launch"
            kwargs[name].zero_()
        return kwargs["output"]

    adapter = SimpleNamespace(fused_decode=decode)
    # The second call verifies stale finite scratch is poisoned again, not only allocated as NaN.
    for _ in range(2):
        output = live._launch_serving_decode(adapter, None, (None,) * 3, None, contexts, query, buffers, threshold)
        assert (output == 0).all()
