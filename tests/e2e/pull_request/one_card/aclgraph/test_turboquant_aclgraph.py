# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture both TurboQuant attention launches and replay changing decode batches.

Requires an ascend950 build with TurboQuant kernels. No model weights are needed.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.attention.backend import AttentionType

from tests.helpers.turboquant import make_adapter, make_cache, make_metadata_builder
from vllm_ascend.attention import turboquant_v1 as tq

BATCH = 2
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_SIZE = 256
BLOCK_SIZE = 128
MAX_BLOCKS = 34  # Cover the fused/split transition at context length 4096.
NUM_BLOCKS = BATCH * MAX_BLOCKS


@pytest.fixture
def decode_fixture():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    tq.load_turboquant_extension()
    if not all(
        hasattr(getattr(torch.ops, tq.TURBOQUANT_DEFAULT_NAMESPACE, None), name)
        for name in (
            tq.TURBOQUANT_OP_RESHAPE_SCATTER,
            tq.TURBOQUANT_OP_FUSED_DECODE,
            tq.TURBOQUANT_OP_BUILD_DECODE_TILING,
        )
    ):
        pytest.skip("requires an ascend950 build with TurboQuant kernels")
    device = torch.device("npu")
    adapter = make_adapter(tq.TurboQuantKernelStage.SCATTER, tq.TurboQuantKernelStage.DECODE)
    impl = object.__new__(tq.AscendTurboQuantBackendImpl)
    impl.adapter = adapter
    impl.num_heads, impl.num_kv_heads, impl.head_size = NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE
    impl.scale = HEAD_SIZE**-0.5
    impl.attn_type = AttentionType.DECODER
    impl.kv_sharing_target_layer_name = None
    impl.pcp_enabled = False
    impl.telemetry = tq.InferenceTelemetry(enabled=False)
    impl.emulator = None
    layer = SimpleNamespace(layer_name="turboquant_capture_test", _k_scale_float=1.0, _v_scale_float=1.0)
    emulator = impl._bind_codec_constants(layer, device)
    caches = make_cache(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=device)
    impl.key_cache, impl.value_cache, impl.scale_cache = caches
    # Populate all history with the real scatter writer, before capture.
    generator = torch.Generator().manual_seed(42)
    history = [
        torch.randn(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    adapter.reshape_scatter_decode(
        *history, *caches, torch.arange(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.int32, device=device), emulator
    )

    builder = make_metadata_builder(BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE, MAX_BLOCKS, device=device)
    return impl, builder, layer, caches, generator


def test_capture_and_replay_ingress_and_attention(decode_fixture):
    impl, builder, layer, caches, generator = decode_fixture
    device = caches[0].device
    query = torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device)
    key = torch.zeros(BATCH, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device)
    value = torch.zeros_like(key)
    output = torch.zeros_like(query)
    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device).view(BATCH, MAX_BLOCKS)
    host_block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).view(BATCH, MAX_BLOCKS)
    slots = torch.full((BATCH,), -1, dtype=torch.int64, device=device)
    metadata = SimpleNamespace()

    def prepare(width, lengths):
        live = sum(length > 0 for length in lengths)
        common = SimpleNamespace(
            num_reqs=live,
            num_actual_tokens=live,
            slot_mapping=slots[:width],
            attn_state=tq.AscendAttentionState.DecodeOnly,
        )
        block_table.copy_(host_block_table.to(device))
        host_slots = [
            int(host_block_table[row, (length - 1) // BLOCK_SIZE]) * BLOCK_SIZE + (length - 1) % BLOCK_SIZE
            if length
            else -1
            for row, length in enumerate(lengths)
        ]
        slots[:width].copy_(torch.tensor(host_slots, dtype=torch.int64, device=device))
        extra = builder._build_backend_metadata(
            common,
            block_table=block_table[:width],
            query_lens=torch.ones(width, dtype=torch.int32),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            num_decodes=live,
            num_prefills=0,
        )
        metadata.__dict__.update(
            extra,
            num_actual_tokens=live,
            attn_state=common.attn_state,
            block_tables=block_table[:width],
            model_runner_type="generate",
            causal=True,
            slot_mapping=slots[:width],
        )

    def forward(width):
        impl.forward(layer, query[:width], key[:width], value[:width], caches, metadata, output=output[:width])

    buffer_names = ("_context_lens", "_tiling", "_workspace", "_query_rot", "_decode_slot_mapping")
    addresses = [getattr(builder, name).data_ptr() for name in buffer_names]
    # Two graphs share builder-owned storage. Replaying the first graph after
    # capturing the second also checks that descriptors retain their ownership.
    graphs = {}
    for width in (1, BATCH):
        prepare(width, [64] * width)
        forward(width)  # Warm Python and C++ constants on an eager stream.
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            forward(width)
        graphs[width] = graph

    for width, lengths in (
        (BATCH, [65, 129]),
        (1, [257]),
        (BATCH, [130, 0]),
        (1, [2047]),
        (1, [2048]),
        (1, [2049]),
        (BATCH, [2050, 2177]),
        (BATCH, [4097, 258]),
        (BATCH, [66, 129]),
    ):
        # Change page IDs without changing captured table storage. Replay must
        # consume both the refreshed descriptors and the current page mapping.
        host_block_table = host_block_table.roll(1, dims=1)
        for tensor in (query, key, value):
            tensor.copy_(torch.randn(tensor.shape, generator=generator).to(device=device, dtype=tensor.dtype))
        prepare(width, lengths)
        cache_before = [tensor.clone() for tensor in caches]
        forward(width)
        expected = output[:width].clone()
        expected_cache = [tensor.clone() for tensor in caches]
        # Undo eager ingestion: replay must perform the writes itself.
        for tensor, previous in zip(caches, cache_before):
            tensor.copy_(previous)
        output.fill_(float("nan"))
        graphs[width].replay()
        torch.npu.synchronize()
        live = sum(length > 0 for length in lengths)
        torch.testing.assert_close(output[:live], expected[:live], rtol=0, atol=0)
        for actual, reference in zip(caches, expected_cache):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert addresses == [getattr(builder, name).data_ptr() for name in buffer_names]
