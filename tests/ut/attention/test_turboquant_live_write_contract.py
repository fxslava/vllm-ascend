# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the compact live NPU suite's reference and launch preconditions."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.turboquant import register_ops
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
        output = live._launch_serving_decode(
            adapter, None, (None,) * 3, None, contexts, query, buffers, threshold, splits=3
        )
        assert (output == 0).all()


@pytest.mark.parametrize("batch", [1, 2, 3, 9])
@pytest.mark.parametrize("splits", [1, 2, 3, 8])
def test_live_tiling_and_adapter_launch_use_the_same_split_geometry(monkeypatch, batch, splits):
    """Explicit split counts must reach the real adapter and the operator."""
    monkeypatch.setattr(live, "DEVICE", torch.device("cpu"))
    backend = live._backend()
    # Independent statement of decode_launch_split and its descriptor ABI for
    # this fixture: one head chunk, four KV heads, at most 32 resident blocks.
    grid = min(32, batch * live.NUM_KV_HEADS * splits)
    expected_words = batch * splits * 16 + grid * 8
    contexts = tuple(65 + request for request in range(batch))
    calls = []

    def build_tiling(
        context_lens,
        tiling,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_blocks,
        num_splits,
        heads_per_task,
        fused_context_limit,
    ):
        assert context_lens.tolist() == list(contexts)
        assert num_splits == splits
        assert (num_heads, num_kv_heads, head_size) == (live.NUM_HEADS, live.NUM_KV_HEADS, live.HEAD_SIZE)
        assert (block_size, max_blocks, heads_per_task) == (128, 1, live.GROUP_HEADS)
        assert fused_context_limit == live.FUSED_CONTEXT_LIMIT
        assert tiling.numel() == expected_words
        tiling[0], tiling[10] = 1, 1
        calls.append("build")

    def fused_decode(*args):
        names = tuple(inspect.signature(backend.TurboQuantOpsAdapter._kernel_fused_decode).parameters)[1:]
        passed = dict(zip(names, args, strict=True))
        assert passed["num_splits"] == splits
        assert passed["tiling"].numel() == expected_words
        assert passed["workspace"].numel() == batch * live.NUM_HEADS * splits * (live.HEAD_SIZE + 16)
        assert passed["query_rot"].numel() == batch * live.NUM_HEADS * live.HEAD_SIZE
        assert passed["context_lens"].tolist() == list(contexts)
        assert passed["fused_context_limit"] == live.FUSED_CONTEXT_LIMIT
        passed["output"].zero_()
        calls.append("decode")

    register_ops(monkeypatch, **{live.BUILD_TILING_OP: build_tiling, live.FUSED_DECODE_OP: fused_decode})
    live._require_ops(live.BUILD_TILING_OP, live.FUSED_DECODE_OP)
    buffers = live._build_decode_tiling(contexts, 128, 1, splits)
    adapter = live._adapter(backend.TurboQuantKernelStage.DECODE)
    emulator = backend.TurboQuantEmulator(head_size=live.HEAD_SIZE, device=live.DEVICE)
    cache = live._caches(128, num_blocks=1)
    query = torch.ones(batch, live.NUM_HEADS, live.HEAD_SIZE, dtype=torch.bfloat16)
    tables = torch.zeros(batch, 1, dtype=torch.int32)
    output = live._launch_serving_decode(adapter, emulator, cache, tables, contexts, query, buffers, splits=splits)
    assert (output == 0).all()
    assert calls == ["build", "decode"]


@pytest.mark.parametrize("missing", [False, True])
def test_live_require_ops_uses_the_current_registry(monkeypatch, missing):
    backend = live._backend()
    monkeypatch.setattr(backend, "load_turboquant_extension", lambda: True)
    namespace = SimpleNamespace() if missing else SimpleNamespace(**{live.SCATTER_OP: lambda: None})
    monkeypatch.setattr(torch, "ops", SimpleNamespace(**{live.NAMESPACE: namespace}))
    if missing:
        with pytest.raises(pytest.skip.Exception, match="requires an ascend950 TurboQuant build"):
            live._require_ops(live.SCATTER_OP)
    else:
        live._require_ops(live.SCATTER_OP)


def test_live_backend_references_are_current_exports():
    backend = live._backend()
    tree = ast.parse(inspect.getsource(live))
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "backend"
    }
    assert not {name for name in referenced if not hasattr(backend, name)}


def test_live_adapter_calls_match_current_python_signatures():
    adapter_cls = live._backend().TurboQuantOpsAdapter
    methods = {"reshape_and_cache_prefill", "reshape_scatter_decode", "gather", "dequantize_cache", "fused_decode"}
    for node in ast.walk(ast.parse(inspect.getsource(live))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr not in methods:
            continue
        # Every starred adapter argument in this suite is a three-plane cache.
        arguments = [None for argument in node.args for _ in range(3 if isinstance(argument, ast.Starred) else 1)]
        keywords = {keyword.arg: None for keyword in node.keywords}
        inspect.signature(getattr(adapter_cls, node.func.attr)).bind(None, *arguments, **keywords)


@pytest.mark.parametrize("context", [2047, 2048, 2049, 2177])
def test_third_physical_page_matches_expanded_kernel_gather(monkeypatch, context):
    monkeypatch.setattr(live, "DEVICE", torch.device("cpu"))
    cache = live._caches(1024, num_blocks=3)
    for page in range(3):
        for index, plane in enumerate(cache[:2]):
            plane[page].fill_(page + index + 1)
        cache[2].view(3, -1)[page].fill_(page + 1.25)
    tables = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    kernel_cache, kernel_tables = live._kernel_decode_inputs(cache, tables, 128)
    assert kernel_tables.tolist() == [[*range(16, 24), *range(8), *range(8, 16)]]
    assert kernel_cache[0].shape[:2] == (24, 128)
    for original, view in zip(cache, kernel_cache):
        assert original.data_ptr() == view.data_ptr()
    adapter = live._adapter()
    physical = adapter.gather(*cache, tables)
    expanded = adapter.gather(*kernel_cache, kernel_tables)
    for actual, expected in zip(expanded, physical):
        torch.testing.assert_close(actual[:, :context], expected[:, :context], rtol=0, atol=0)
    # The third logical scheduler page is physical page 1, at kernel page 8.
    position = context - 1
    expected_slot = int(tables[0, position // 1024]) * 1024 + position % 1024
    actual_slot = int(kernel_tables[0, position // 128]) * 128 + position % 128
    assert actual_slot == expected_slot
