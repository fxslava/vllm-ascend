# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for prepared TurboQuant cache writes and Dynamo graph capture."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from vllm.v1.attention.backend import AttentionType

from tests.helpers.turboquant import make_cache, make_metadata_builder
from tests.ut.attention.test_turboquant_op_schemas import _schema_strings
from vllm_ascend.attention import turboquant_metadata, turboquant_ops, turboquant_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_KERNEL_BLOCK_SIZE as BLOCK_SIZE,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_KERNEL_HEAD_SIZE as HEAD_SIZE,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_KERNEL_KV_HEADS as KV_HEADS,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_NAMESPACE_ENV,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_STAGES_ENV,
    TURBOQUANT_STRIP_TOKENS,
)
from vllm_ascend.attention.turboquant_metadata import AscendTurboQuantMetadata
from vllm_ascend.attention.turboquant_telemetry import NullDispatchObserver

NUM_BLOCKS = 3
NUM_HEADS = 2 * KV_HEADS
CHAMPION_TOKENS = 6 * TURBOQUANT_STRIP_TOKENS


@pytest.fixture
def kernels(monkeypatch):
    # Use a private namespace, including on CI hosts where the real extension
    # is already registered. Copy the raw mutating schemas from the binding.
    namespace = f"turboquant_host_{uuid4().hex}"
    monkeypatch.setenv(TURBOQUANT_NAMESPACE_ENV, namespace)
    monkeypatch.setenv(TURBOQUANT_STAGES_ENV, "reshape_and_cache,scatter,decode")
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS", "0")
    monkeypatch.setattr(turboquant_metadata, "load_turboquant_extension", lambda: False)
    monkeypatch.setattr(turboquant_ops, "load_turboquant_extension", lambda: False)
    monkeypatch.setattr(turboquant_metadata, "_stream_is_capturing", lambda: False)
    monkeypatch.setattr(turboquant_v1, "notify_kv_cache_written", lambda: None)
    calls = []
    schemas = _schema_strings()

    def run(op_name):
        names = tuple(arg.name for arg in torch._C.parse_schema(schemas[op_name]).arguments)

        def cpu(*args):
            # The dispatcher can omit arguments equal to their schema defaults.
            call = dict(zip(names, args))
            if op_name == TURBOQUANT_OP_BUILD_DECODE_TILING:
                call["tiling"][: call["context_lens"].numel()].copy_(call["context_lens"])
                return
            if op_name == TURBOQUANT_OP_FUSED_DECODE:
                query, output = call["query"], call["output"]
                if query.data_ptr() == output.data_ptr():
                    raise ValueError("query and output must not alias")
                lengths = call["context_lens"]
                assert torch.equal(call["tiling"][: lengths.numel()], lengths)
                last = (lengths.long() - 1).clamp_min(0)
                page = call["block_tables"][torch.arange(last.numel()), last // BLOCK_SIZE]
                slots = page * BLOCK_SIZE + last % BLOCK_SIZE
                cache_planes = ("key_cache", "value_cache", "scale_cache")
                marker = sum(call[plane].flatten()[slots].float() for plane in cache_planes)
                output.copy_(torch.where(lengths[:, None, None] > 0, query + marker[:, None, None], 0))
                call["workspace"].fill_(19)
                call["query_rot"].fill_(23)
                calls.append((op_name, lengths.tolist(), call["tiling"].data_ptr()))
                return

            # Store row markers instead of emulating the numerical kernel.
            # Each fake writer mutates the three planes declared in its schema.
            slots = call["slot_mapping"]
            calls.append((op_name, slots.numpy().tolist(), slots.data_ptr()))
            keep = slots >= 0
            dst = slots[keep].long()
            key, value = call["key"][keep, 0, 0], call["value"][keep, 0, 0]
            call["key_cache"].view(-1)[dst] = key.to(torch.int8)
            call["value_cache"].view(-1)[dst] = value.to(torch.int8)
            call["scale_cache"].view(-1)[dst] = (key + value).float()

        return cpu

    lib = torch.library.Library(namespace, "FRAGMENT")
    for name, schema in schemas.items():
        lib.define(schema)
        lib.impl(name, run(name), "CPU")
        lib.impl(name, lambda *args: None, "Meta")
    yield SimpleNamespace(namespace=namespace, calls=calls)


class _Backend(turboquant_v1.AscendTurboQuantBackendImpl):
    def __init__(self):
        self.attn_type = AttentionType.DECODER
        self.kv_sharing_target_layer_name = None
        self.num_heads, self.num_kv_heads, self.head_size = NUM_HEADS, KV_HEADS, HEAD_SIZE
        self.scale = HEAD_SIZE**-0.5
        self.adapter = turboquant_ops.TurboQuantOpsAdapter(dispatch_observer=NullDispatchObserver())
        self.emulator = None
        self._bind_codec_constants(SimpleNamespace(), torch.device("cpu"))
        self.key_cache, self.value_cache, self.scale_cache = make_cache(
            NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, poison=True
        )

    @property
    def caches(self):
        return self.key_cache, self.value_cache, self.scale_cache

    def _run_dense_attention(self, query, key, value, *args, **kwargs):
        # The real prefill attention call is a vendor operator.
        return query + value.mean(dim=1, keepdim=True)


def _metadata(builder, slots, context_lens, query_lens, *, decode=False):
    state = AscendAttentionState.DecodeOnly if decode else AscendAttentionState.PrefillNoCache
    num_tokens = sum(query_lens)
    num_reqs = len(query_lens)
    block_table = torch.zeros((len(context_lens), NUM_BLOCKS), dtype=torch.int32)
    block_table[:, 0] = torch.arange(len(context_lens))
    common = SimpleNamespace(slot_mapping=slots, num_reqs=num_reqs, attn_state=state, num_actual_tokens=num_tokens)
    fields = builder._build_backend_metadata(
        common,
        block_table=block_table,
        query_lens=torch.tensor(query_lens),
        seq_lens=torch.tensor(context_lens),
        num_decodes=num_reqs if decode else 0,
        num_prefills=0 if decode else num_reqs,
    )
    metadata = AscendTurboQuantMetadata(**fields)
    metadata.attn_state = state
    metadata.slot_mapping = slots
    metadata.seq_lens_list = context_lens
    metadata.num_actual_tokens = num_tokens
    metadata.block_tables = block_table
    metadata.actual_seq_lengths_q = torch.tensor(query_lens).cumsum(0).tolist()
    metadata.attn_mask = None
    metadata.causal = True
    return metadata


def _builder():
    return make_metadata_builder(4, NUM_HEADS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE, NUM_BLOCKS)


def _inputs(tokens):
    query = torch.ones((tokens, NUM_HEADS, HEAD_SIZE), dtype=torch.bfloat16)
    row = torch.arange(1, tokens + 1, dtype=torch.bfloat16).view(-1, 1, 1)
    key = row.expand(-1, KV_HEADS, HEAD_SIZE).contiguous()
    return query, key, -key, torch.empty_like(query)


def test_aligned_96_token_prefill_plans_once_for_all_layers_and_rebuilds_next_batch(monkeypatch, kernels):
    calls = []
    tolist = torch.Tensor.tolist
    plan_ingestion = turboquant_metadata.turboquant_plan_strip_ingestion

    def track_slot_reads(tensor):
        if tensor.dtype == torch.int32 and tensor.shape == (CHAMPION_TOKENS,):
            calls.append("tolist")
        return tolist(tensor)

    def plan(slots, page_size):
        calls.append(("plan", tuple(slots)))
        return plan_ingestion(slots, page_size)

    def cold_plan(*args):
        pytest.fail("a layer planned on demand despite builder-prepared metadata")

    monkeypatch.setattr(torch.Tensor, "tolist", track_slot_reads)
    monkeypatch.setattr(turboquant_metadata, "turboquant_plan_strip_ingestion", plan)
    monkeypatch.setattr(turboquant_ops, "turboquant_plan_strip_ingestion", cold_plan)
    builder = _builder()
    source = torch.arange(CHAMPION_TOKENS + 2, dtype=torch.int32)
    source[-2:] = 777  # Poison padding beyond the live extent.
    layers = [_Backend(), _Backend()]
    query, key, value, output = _inputs(CHAMPION_TOKENS)
    for offset in (0, BLOCK_SIZE):
        live = list(range(offset, offset + CHAMPION_TOKENS))
        source[:CHAMPION_TOKENS].copy_(torch.tensor(live))
        before = len(calls)
        metadata = _metadata(builder, source, [CHAMPION_TOKENS], [CHAMPION_TOKENS])
        assert calls[before:] == ["tolist", ("plan", tuple(live))]
        assert metadata.turboquant_strip_plan.is_whole_batch
        prepared = metadata.turboquant_ingestion_slot_mapping
        assert prepared.data_ptr() != source.data_ptr()
        # A new write to the runner's int32 slot buffer must not change the
        # prepared targets or invalidate their whole-batch plan.
        source[0] = offset + CHAMPION_TOKENS + 5
        for layer in layers:
            layer.reshape_and_cache(query, key, value, layer.caches, metadata, output)
        assert calls[before:] == ["tolist", ("plan", tuple(live))]
        writer_call = (TURBOQUANT_OP_RESHAPE_AND_CACHE, live, prepared.data_ptr())
        assert kernels.calls[-2:] == [writer_call] * 2
        for layer in layers:
            torch.testing.assert_close(layer.key_cache.flatten()[live], key[:, 0, 0].to(torch.int8))


def test_fragmented_prefill_preserves_wave_order_duplicate_slots_and_poison(kernels):
    slots = [1, -1, 2, 1, 33, *range(64, 80), 96, 97, 101]
    builder = _builder()
    metadata = _metadata(builder, torch.tensor(slots), [len(slots)], [len(slots)])
    query, key, value, output = _inputs(len(slots))
    layer = _Backend()
    before = [cache.clone() for cache in layer.caches]
    layer.reshape_and_cache(query, key, value, layer.caches, metadata, output)
    launches = [(name, rows) for name, rows, _ in kernels.calls]
    assert launches == [
        (TURBOQUANT_OP_RESHAPE_AND_CACHE, list(range(64, 80))),
        (TURBOQUANT_OP_RESHAPE_SCATTER, [1, -1, 33, 96]),
        (TURBOQUANT_OP_RESHAPE_SCATTER, [2, 97]),
        (TURBOQUANT_OP_RESHAPE_SCATTER, [1, 101]),
    ]
    for row, slot in enumerate(slots):
        if slot >= 0:
            before[0].view(-1)[slot] = key[row, 0, 0].to(torch.int8)
            before[1].view(-1)[slot] = value[row, 0, 0].to(torch.int8)
            before[2].view(-1)[slot] = 0
    for observed, expected in zip(layer.caches, before):
        if observed.is_floating_point():
            observed, expected = observed.view(torch.int32), expected.view(torch.int32)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_default_stages_skip_strip_planning(monkeypatch, kernels):
    monkeypatch.delenv(TURBOQUANT_STAGES_ENV)
    slots = torch.arange(TURBOQUANT_STRIP_TOKENS, dtype=torch.int64)
    metadata = _metadata(_builder(), slots, [slots.numel()], [slots.numel()])
    assert metadata.turboquant_ingestion_slot_mapping.dtype == torch.int32
    assert metadata.turboquant_strip_plan is None
    assert not kernels.calls


def test_prefill_slots_do_not_satisfy_the_decode_capture_guard(monkeypatch, kernels):
    tokens = TURBOQUANT_STRIP_TOKENS
    metadata = _metadata(_builder(), torch.arange(tokens), [tokens], [tokens])
    assert metadata.turboquant_ingestion_slot_mapping is not None
    assert metadata.turboquant_slot_mapping is None
    # The base builder can apply the requested capture state after building.
    metadata.attn_state = AscendAttentionState.DecodeOnly
    query, key, value, output = _inputs(tokens)
    layer = _Backend()
    monkeypatch.setattr(turboquant_v1, "_stream_is_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="decode slot mapping must be prepared"):
        layer.forward(None, query, key, value, layer.caches, metadata, output)
    assert not kernels.calls


def _record_graphs():
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    return graphs, backend


def _run_layer(layer, metadata, query, key, value, output):
    # These are the two TurboQuant hooks called by the shared attention
    # forward. The device-free runner stubs that base class.
    _, _, _, output = layer.reshape_and_cache(query, key, value, layer.caches, metadata, output)
    return layer.forward_impl(query, key, value, layer.caches, metadata, output)


def _check_graph(graphs, kernels, *names):
    assert len(graphs) == 1
    targets = [str(node.target) for node in graphs[0].graph.nodes if node.op in ("call_function", "call_method")]
    assert [target for target in targets if kernels.namespace in target] == [
        f"{kernels.namespace}.{name}" for name in names
    ]
    assert not any("tolist" in target or "data_ptr" in target or "_local_scalar_dense" in target for target in targets)


def test_prepared_prefill_traces_as_one_graph(kernels):
    metadata = _metadata(_builder(), torch.arange(CHAMPION_TOKENS), [CHAMPION_TOKENS], [CHAMPION_TOKENS])
    layer = _Backend()
    query, key, value, output = _inputs(CHAMPION_TOKENS)
    graphs, backend = _record_graphs()
    compiled = torch.compile(_run_layer, backend=backend, fullgraph=True, dynamic=False)
    assert compiled(layer, metadata, query, key, value, output) is output
    _check_graph(graphs, kernels, TURBOQUANT_OP_RESHAPE_AND_CACHE)
    torch.testing.assert_close(output, query + value[:, :1])
    assert [name for name, _, _ in kernels.calls] == [TURBOQUANT_OP_RESHAPE_AND_CACHE]


def test_standalone_adapter_plans_eagerly_and_needs_preparation_for_a_full_graph(kernels):
    layer = _Backend()
    _, key, value, _ = _inputs(CHAMPION_TOKENS)
    slots = torch.arange(CHAMPION_TOKENS, dtype=torch.int32)

    def ingest(key, value, slots):
        layer.adapter.reshape_and_cache_prefill(key, value, *layer.caches, slots, layer.emulator)

    graphs, backend = _record_graphs()
    compiled = torch.compile(ingest, backend=backend, fullgraph=True, dynamic=False)
    with pytest.raises(
        (torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError), match="(?i)tolist|data-dependent|could not guard"
    ):
        compiled(key, value, slots)
    assert not graphs
    assert not kernels.calls

    ingest(key, value, slots)
    assert kernels.calls == [(TURBOQUANT_OP_RESHAPE_AND_CACHE, slots.tolist(), slots.data_ptr())]


def test_fixed_decode_traces_once_and_uses_refreshed_padded_metadata(kernels):
    builder, layer = _builder(), _Backend()
    query, key, value, output = _inputs(2)
    value.copy_(key)
    graphs, backend = _record_graphs()
    compiled = torch.compile(_run_layer, backend=backend, fullgraph=True, dynamic=False)
    # The builder refreshes the backing buffers before the second call and
    # masks the dirty padded slot, while the launch extent stays fixed.
    metadata = _metadata(builder, torch.tensor([0, BLOCK_SIZE + 1]), [1, 2], [1, 1], decode=True)
    assert compiled(layer, metadata, query, key, value, output) is output
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([5, 9], dtype=output.dtype))
    _, workspace, query_rot = layer._get_decode_launch_buffers(metadata, 2)
    assert torch.all(workspace == 19)
    assert torch.all(query_rot == 23)
    refreshed = _metadata(builder, torch.tensor([2, 127]), [3, 0], [1], decode=True)
    assert refreshed.turboquant_slot_mapping.data_ptr() == metadata.turboquant_slot_mapping.data_ptr()
    assert refreshed.turboquant_context_lens.data_ptr() == metadata.turboquant_context_lens.data_ptr()
    assert compiled(layer, refreshed, query, key, value, output) is output
    _check_graph(graphs, kernels, TURBOQUANT_OP_RESHAPE_SCATTER, TURBOQUANT_OP_FUSED_DECODE)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([5, 0], dtype=output.dtype))
    assert kernels.calls[-2][1] == [2, -1]
    assert kernels.calls[-1][1] == [3, 0]
    assert layer.key_cache.view(-1)[127] == 127
