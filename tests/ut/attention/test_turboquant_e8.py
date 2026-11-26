# SPDX-License-Identifier: Apache-2.0
import inspect
import re
import struct
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.helpers.turboquant import make_metadata_builder
from tests.ut.attention.test_turboquant_op_schemas import _schema_strings, _schemas, _source
from vllm_ascend.attention import turboquant_e8 as e8
from vllm_ascend.attention import turboquant_metadata, turboquant_reference_cache, turboquant_v1
from vllm_ascend.attention.turboquant_config import TurboQuantOpsConfig
from vllm_ascend.attention.turboquant_layout import (
    e8_cache_shape,
    e8_packed_plane_bytes,
    turboquant_carve_planes,
    turboquant_plane_views,
)
from vllm_ascend.attention.turboquant_ops import TurboQuantOpsAdapter


def test_e8_packet_bytes_and_strided_views():
    labels = torch.arange(3 * 64 * 32).reshape(3, 64, 32).to(torch.uint16)
    scales = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64) * 0.125
    packed = e8.pack_packets(labels, scales)
    assert packed.shape == (3, 4352)
    assert packed.stride() == (4352, 1)
    got_labels, got_scales = e8.unpack_packets(packed)
    assert got_labels.stride() == (2176, 32, 1)
    assert got_scales.stride() == (1088, 1)
    torch.testing.assert_close(got_labels, labels)
    torch.testing.assert_close(got_scales, scales)
    # Byte checks independently pin the UINT16 little-endian alphabet and scale offset.
    assert packed[0, :4].tolist() == [0, 0, 1, 0]
    assert packed[1, 4096:4100].tolist() == list(struct.pack("<f", 8.0))
    assert got_labels.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert got_scales.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()


@pytest.mark.parametrize("count", [4, 6, 8, 254, 256])
def test_e8_offset_table_has_full_dma_burst(count):
    offsets = e8.packet_byte_offsets([1, 0] * (count // 2), 2)
    assert offsets.dtype == torch.int64
    assert offsets.numel() * 8 % 32 == 0
    assert offsets[:count].tolist() == [4352, 0] * (count // 2)
    assert offsets[count:].tolist() == [0] * (-count % 4)


@pytest.mark.parametrize("ids,total", [([0] * 3, 1), ([0] * 258, 1), ([0, 1, 2, 3], 3), ([-1] * 4, 2), ([0.5] * 4, 2)])
def test_e8_offset_table_refuses_invalid_metadata(ids, total):
    with pytest.raises(ValueError):
        e8.packet_byte_offsets(ids, total)


@pytest.mark.parametrize(
    "fn,args",
    [
        (e8.pack_packets, (torch.zeros(1, 64, 32, dtype=torch.int8), torch.ones(1, 64))),
        (e8.unpack_packets, (torch.zeros(1, 4352, dtype=torch.int8),)),
    ],
)
def test_e8_refuses_native_nibble_buffers(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_e8_schema_call_order_mutation_and_meta_registration():
    text = _source("turboquant_e8_ops")
    schemas = _schemas("turboquant_e8_ops")
    assert set(schemas) == {"reader", "dequant", "dequant_packet"}
    for name, schema in schemas.items():
        fn = getattr(e8, name)
        assert [a.name for a in schema.arguments] == list(inspect.signature(fn).parameters)
        assert not schema.returns
        assert f'ops.impl("{name}", c10::kMeta,' in text
        assert f'ops.impl("{name}", c10::kPrivateUse1,' in text
        args = [object() for _ in schema.arguments]
        op = Mock()
        with patch.object(torch.ops.turboquant_e8_ops, name, op, create=True):
            fn(*args)
        assert list(op.call_args.args) == args
    writes = {a.name for a in schemas["reader"].arguments if a.alias_info and a.alias_info.is_write}
    assert writes == {"output", "key_witness", "value_witness", "score_witness"}
    for name in ("dequant", "dequant_packet"):
        writes = {a.name for a in schemas[name].arguments if a.alias_info and a.alias_info.is_write}
        assert writes == {"bf16", "int8"}


def test_e8_runtime_wrappers_do_not_allocate_or_copy():
    for fn in (e8.reader, e8.dequant, e8.dequant_packet):
        source = inspect.getsource(fn)
        assert not re.search(r"\.(empty|zeros|tensor|to|cpu|clone|contiguous)\(", source)


@pytest.mark.parametrize("block_size,plane_bytes", [(64, 4352), (128, 8704)])
def test_e8_mode_allocates_packet_planes(monkeypatch, block_size, plane_bytes):
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_QUANT_MODE", "2")
    assert TurboQuantOpsConfig.from_env().uses_e8
    backend = turboquant_v1.AscendTurboQuantBackend
    assert backend.get_supported_kernel_block_sizes()[0].base == 64
    shape = e8_cache_shape(3, block_size, 1, 256)
    assert backend.get_kv_cache_shape(3, block_size, 1, 256) == (2, *shape)
    assert e8_packed_plane_bytes(block_size, 1, 256) == plane_bytes
    raw = torch.zeros(3 * 2 * plane_bytes + 64, dtype=torch.int8)
    planes = turboquant_carve_planes(raw, 3, block_size, 1, 256, uses_e8=True)
    key, value = turboquant_plane_views(planes, block_size, 1, 256, uses_e8=True)
    assert key.size() == value.size() == shape
    assert key.dtype == value.dtype == torch.uint8
    assert value.data_ptr() - key.data_ptr() == 3 * plane_bytes
    key.fill_(255)
    assert torch.all(raw[: 3 * plane_bytes] == -1)
    assert torch.count_nonzero(raw[3 * plane_bytes :]) == 0


def test_e8_host_packet_load_validates_before_mutation():
    adapter = TurboQuantOpsAdapter(TurboQuantOpsConfig(quant_mode=2))
    planes = tuple(torch.zeros((2, 2, 3, 4352), dtype=torch.uint8) for _ in range(2))
    ids = torch.tensor([7, 4, 11], dtype=torch.int64)
    labels = torch.arange(3 * 64 * 32, dtype=torch.int32).view(3, 64, 32).to(torch.uint16)
    key = e8.pack_packets(labels, torch.ones(3, 64))
    value = e8.pack_packets(labels, torch.full((3, 64), 2.0))
    with pytest.raises(ValueError, match="unique"):
        adapter.load_e8_packets(planes, torch.tensor([4, 4, 7]), key, value)
    assert not any(torch.count_nonzero(p) for p in planes)
    adapter.load_e8_packets(planes, ids, key, value)
    for plane, expected in zip(planes, (key, value)):
        actual = plane.view(-1, 4352)
        torch.testing.assert_close(actual[ids], expected)
        assert torch.count_nonzero(actual[0]) == 0


def test_e8_cpu_encoder_matches_revision2_oracle_bytes():
    channels = torch.arange(256)
    inputs = torch.stack(
        (
            torch.zeros(256),
            ((channels * 29 + 3) % 227 - 113).float() / 8,
            torch.where(channels % 2 == 0, 16.0, -16.0),
        )
    )
    labels, scales = turboquant_reference_cache.encode_e8_rotated(inputs)
    assert scales.shape == (3,) and scales.dtype == torch.float32
    assert labels.shape == (3, 32) and labels.dtype == torch.uint16
    payload = torch.cat((labels.view(torch.uint8), scales[:, None].view(torch.uint8)), dim=1)
    assert sha256(payload.numpy().tobytes()).hexdigest() == (
        "a6c90c4689d451fed1ff28845dec7b5ece5e0f4acdd54baccc7fb5d1f5b0c991"
    )
    decoded = turboquant_reference_cache.decode_e8_labels(labels, scales)
    assert decoded.dtype == torch.bfloat16
    assert torch.equal(decoded[0], torch.zeros(256, dtype=torch.bfloat16))
    assert torch.equal(decoded[2], torch.tensor([1, -1, 0, -1, 0, -1, 1, -1], dtype=torch.bfloat16).repeat(32))


def test_e8_device_activations_refused_without_touching_packets():
    planes = tuple(torch.full((1, 1, 1, 4352), 95, dtype=torch.uint8) for _ in range(2))
    raw = torch.empty((1, 1, 256), dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="device ingress is gated"):
        turboquant_reference_cache.write_e8_slots(raw, raw, *planes, torch.tensor([0]), None)
    assert all(torch.all(plane == 95) for plane in planes)


def test_e8_metadata_reuses_host_plan_and_int64_offsets(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_QUANT_MODE", "2")
    monkeypatch.setenv("TURBOQUANT_KERNEL_STAGES", "decode")
    monkeypatch.setattr(turboquant_metadata, "load_turboquant_extension", lambda: True)
    monkeypatch.setattr(turboquant_metadata, "_is_op_registered", lambda *args: True)
    builder = make_metadata_builder(3, 48, 8, 256, 128, 3)
    assert not hasattr(builder, "_workspace")
    host = torch.tensor([[400_000, 2, -1], [3, 0, -1], [-1, -1, -1]], dtype=torch.int32)
    common = SimpleNamespace(turboquant_block_table_cpu=host, turboquant_num_blocks=400_001, num_actual_tokens=2)
    with patch.object(turboquant_metadata, "prepare_batch", wraps=e8.prepare_batch) as prepare:
        fields = builder._build_backend_metadata(
            common,
            block_table=torch.empty(host.shape, dtype=host.dtype, device="meta"),
            query_lens=torch.tensor([1, 1, 1]),
            seq_lens=torch.tensor([193, 129, 0]),
            num_decodes=2,
            num_prefills=0,
        )
        assert prepare.call_count == 1
    batch = fields["turboquant_e8_batch"]
    first, second = batch.requests
    assert first.packet_ids[:, 0].tolist() == [6_400_000, 6_400_008, 32, 40]
    assert first.reader_offsets[7].tolist() == [p * 4352 for p in (6_400_007, 6_400_015, 39, 47)]
    assert first.reader_offsets.dtype == torch.int64
    assert first.reader_offsets.data_ptr() % 32 == 0
    assert second.reader_offsets is None
    assert batch.reader_buffers.key_witness.size() == (4, 64, 256)
    with pytest.raises(ValueError, match="outside the cache"):
        e8.prepare_batch(
            host,
            torch.tensor([257]),
            torch.tensor([1]),
            num_tokens=1,
            num_blocks=400_001,
            block_size=128,
            num_kv_heads=8,
            device=torch.device("cpu"),
            use_reader=False,
        )


def test_e8_metadata_skips_sequence_parallel_dummy_row(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_QUANT_MODE", "2")
    monkeypatch.setenv("TURBOQUANT_KERNEL_STAGES", "none")
    builder = make_metadata_builder(2, 1, 1, 256, 64, 1)
    host_table = torch.tensor([[0], [1]], dtype=torch.int32)
    padded_context_lens = torch.tensor([1, 1, 1])

    def prepare(cpu_table, context_lens):
        common = SimpleNamespace(turboquant_block_table_cpu=cpu_table, turboquant_num_blocks=2, num_actual_tokens=2)
        return builder._build_backend_metadata(
            common,
            block_table=torch.empty((3, 1), dtype=torch.int32, device="meta"),
            query_lens=torch.tensor([1, 1, 2]),
            seq_lens=context_lens,
            num_decodes=2,
            num_prefills=0,
        )["turboquant_e8_batch"]

    plan = prepare(host_table, padded_context_lens)
    assert [(req.query_start, req.query_end) for req in plan.requests] == [(0, 1), (1, 2)]
    assert [req.packet_ids.tolist() for req in plan.requests] == [[[0]], [[1]]]
    for tables, lengths in ((host_table[:1], padded_context_lens), (host_table, padded_context_lens[:1])):
        with pytest.raises(ValueError, match="E8 live request 1 requires a block table row and context length"):
            prepare(tables, lengths)


@pytest.mark.parametrize("gqa,heads", [(1, 1), (4, 4), (6, 4), (8, 8), (16, 1)])
def test_e8_host_packed_forward_fused_and_ragged_reference(monkeypatch, gqa, heads):
    adapter = TurboQuantOpsAdapter(TurboQuantOpsConfig(quant_mode=2))
    backend = object.__new__(turboquant_v1.AscendTurboQuantBackendImpl)
    backend.adapter = adapter
    backend._uses_e8 = True
    backend.emulator = None
    backend.kv_sharing_target_layer_name = None
    backend.head_size = 256
    backend.num_heads = gqa * heads
    backend.num_kv_heads = heads
    backend.scale = 1 / 16
    block_size = 64
    num_blocks = 4
    host_tables = torch.tensor([[3, 0, 1, 2], [1, 2, 0, -1]], dtype=torch.int32)
    lengths = torch.tensor([193, 129])
    planes = tuple(torch.zeros(e8_cache_shape(num_blocks, block_size, heads, 256), dtype=torch.uint8) for _ in range(2))
    generator = torch.Generator().manual_seed(72)
    labels = torch.randint(0, 65536, (num_blocks * heads, 64, 32), generator=generator).to(torch.uint16)
    scales = torch.rand((num_blocks * heads, 64), generator=generator)
    key = e8.pack_packets(labels, scales)
    value = e8.pack_packets(labels, scales * 0.5)
    adapter.load_e8_packets(planes, torch.arange(num_blocks * heads), key, value)
    metadata = SimpleNamespace(causal=True, num_actual_tokens=2)
    query = torch.randn((2, backend.num_heads, 256), generator=generator)

    def prepare(use_reader):
        return e8.prepare_batch(
            host_tables,
            lengths,
            torch.tensor([1, 1]),
            num_tokens=2,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=heads,
            device=query.device,
            use_reader=use_reader,
        )

    def mock_reader(query, key, value, offsets, out, *args):
        tiles, last_rows, scale = args[-3:]
        ids = offsets[:tiles] // 4352
        args = [turboquant_reference_cache.decode_e8_labels(*e8.unpack_packets(cache[ids])) for cache in (key, value)]
        size = (tiles - 1) * 64 + last_rows
        k, v = [tensor.flatten(0, 1)[:size].float() for tensor in args]
        weights = (query[:6] @ k.T * scale).softmax(dim=-1)
        out.copy_(weights @ v)

    calls = Mock(side_effect=mock_reader)
    monkeypatch.setattr(e8, "reader", calls)
    layer = SimpleNamespace()
    metadata.turboquant_e8_batch = prepare(False)
    expected = backend.forward(layer, query, None, None, planes, metadata, output=torch.empty_like(query))
    metadata.turboquant_e8_batch = prepare(True)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected CPU copy")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected scalar read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("unexpected list read")),
    ):
        actual = backend.forward(layer, query, None, None, planes, metadata, output=torch.empty_like(query))
    assert calls.call_count == heads * ((gqa + 5) // 6)
    assert torch.equal(expected[1], actual[1])
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.001)


def test_e8_ops_fake_tensor_and_dynamo_fullgraph(monkeypatch):
    namespace = f"tq_e8_{uuid4().hex}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    for schema in _schema_strings("turboquant_e8_ops").values():
        lib.define(schema)

    def reader_cpu(*args):
        args[4].fill_(5)

    def dequant_cpu(*args):
        args[2].fill_(7)

    def dequant_packet_cpu(*args):
        args[1].fill_(9)
        args[2].fill_(11)

    lib.impl("reader", reader_cpu, "CPU")
    lib.impl("dequant", dequant_cpu, "CPU")
    lib.impl("dequant_packet", dequant_packet_cpu, "CPU")
    for name in ("reader", "dequant", "dequant_packet"):
        lib.impl(name, lambda *args: None, "Meta")
    monkeypatch.setattr(torch.ops, "turboquant_e8_ops", getattr(torch.ops, namespace))

    def decode(
        query, key, value, ids, out, key_witness, value_witness, score, labels, scales, bf16, i8, packet_bf16, packet_i8
    ):
        e8.reader(query, key, value, ids, out, key_witness, value_witness, score, 4, 41, 1 / 16)
        e8.dequant(labels, scales, bf16, i8)
        e8.dequant_packet(key, packet_bf16, packet_i8)
        return out, bf16, packet_bf16, packet_i8

    args = (
        torch.zeros((16, 256)),
        torch.empty((4, 4352), dtype=torch.uint8),
        torch.empty((4, 4352), dtype=torch.uint8),
        torch.arange(4, dtype=torch.int64) * 4352,
        torch.zeros((6, 256), dtype=torch.bfloat16),
        torch.empty((4, 64, 256), dtype=torch.bfloat16),
        torch.empty((4, 64, 256), dtype=torch.bfloat16),
        torch.empty((4, 6, 64)),
        torch.empty((16, 32), dtype=torch.uint16),
        torch.empty(16),
        torch.zeros((16, 256), dtype=torch.bfloat16),
        torch.empty((16, 256), dtype=torch.int8),
        torch.zeros((4, 64, 256), dtype=torch.bfloat16),
        torch.empty((4, 64, 256), dtype=torch.int8),
    )
    with FakeTensorMode() as fake:
        fake_args = [fake.from_tensor(tensor) for tensor in args]
        out, bf16, packet_bf16, packet_i8 = decode(*fake_args)
        assert out is fake_args[4] and bf16 is fake_args[10]
        assert packet_bf16 is fake_args[12] and packet_i8 is fake_args[13]
    out, bf16, packet_bf16, packet_i8 = torch.compile(decode, backend="eager", fullgraph=True)(*args)
    assert torch.all(out == 5) and torch.all(bf16 == 7)
    assert torch.all(packet_bf16 == 9) and torch.all(packet_i8 == 11)
