# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant kernel/golden dispatch seam.

Three properties are pinned here:

* every operation picks its branch from ``_has_kernel(op_name)`` alone, and the
  kernel branch passes the staged C launcher's parameters in the launcher's own
  order. A stage switched on whose entry point is unregistered falls back.
* ingestion is bifurcated by phase. More tokens than requests is prefill and
  goes to the batch writer; one token per request is decode and goes to the
  scatter writer. Metadata without ``seq_lens_list`` reads as prefill rather
  than crashing.
* ``npu_turboquant_fused_decode`` returns ``None``, which is what routes a
  decode to the golden gather, dequantise and attend path.
"""

import logging
import types
from typing import Any

import pytest
import torch

from vllm_ascend.attention import turboquant_dry_run
from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_GATHER_PAGES,
    TURBOQUANT_OP_QUANTIZE_PACK,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_ROTATE,
    TURBOQUANT_OP_UNPACK_DEQUANT,
    TURBOQUANT_REGISTERED_OPS,
    TURBOQUANT_REUSE_MODE4_FLAGS,
    TURBOQUANT_STAGE_ROTATED_BASIS,
    TURBOQUANT_STAGE_UNROTATED,
    AscendTurboQuantDryRunBackendImpl,
    TurboQuantEmulator,
    TurboQuantIngestionPhase,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    _is_op_registered,
    load_turboquant_extension,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_VALUE_PLANE,
    turboquant_packed_bytes,
    turboquant_scale_cache_shape,
)

HEAD_SIZE = 64
NUM_KV_HEADS = 2
BLOCK_SIZE = 8
NUM_BLOCKS = 4
NAMESPACE = "turboquant_ops"
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Recorder:
    """A registered entry point that records its call and returns ``result``."""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result

    @property
    def args(self) -> tuple[Any, ...]:
        assert len(self.calls) == 1, f"expected exactly one call, got {len(self.calls)}"
        return self.calls[0][0]


class _OpNamespace:
    """``torch.ops.<namespace>``: an attribute per registered entry point."""

    def __init__(self, entries: dict[str, _Recorder]) -> None:
        self._entries = entries

    def __getattr__(self, name: str) -> _Recorder:
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(name) from None


def _register(monkeypatch: pytest.MonkeyPatch, **entries: _Recorder) -> None:
    """Stand ``torch.ops`` up with ``entries`` registered under ``NAMESPACE``."""
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(**{NAMESPACE: _OpNamespace(dict(entries))}))


def _adapter(*stages: TurboQuantKernelStage) -> TurboQuantOpsAdapter:
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(stages),
            value_layout=TurboQuantValueLayout.TOKEN_MAJOR,
        )
    )


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


def _caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An empty three-plane cache: two INT8 packed planes and the fp32 scales."""
    packed = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(turboquant_scale_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS), dtype=torch.float32),
    )


def _activations(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261107)
    shape = (num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(shape, generator=generator, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# _has_kernel
# ---------------------------------------------------------------------------

ALL_OPS = (
    TURBOQUANT_OP_ROTATE,
    TURBOQUANT_OP_QUANTIZE_PACK,
    TURBOQUANT_OP_UNPACK_DEQUANT,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_GATHER_PAGES,
    TURBOQUANT_OP_FUSED_DECODE,
)


@pytest.mark.parametrize("op_name", ALL_OPS)
def test_no_stage_switched_on_means_no_kernel(monkeypatch: pytest.MonkeyPatch, op_name: str) -> None:
    _register(monkeypatch, **{name: _Recorder() for name in ALL_OPS})
    assert _adapter()._has_kernel(op_name) is False


@pytest.mark.parametrize("op_name", ALL_OPS)
def test_stage_on_and_registered_means_kernel(monkeypatch: pytest.MonkeyPatch, op_name: str) -> None:
    _register(monkeypatch, **{op_name: _Recorder()})
    assert _adapter(*TurboQuantKernelStage)._has_kernel(op_name) is True


@pytest.mark.parametrize("op_name", ALL_OPS)
def test_stage_on_but_unregistered_falls_back(monkeypatch: pytest.MonkeyPatch, op_name: str) -> None:
    _register(monkeypatch)
    adapter = _adapter(*TurboQuantKernelStage)
    assert adapter._has_kernel(op_name) is False
    assert op_name in adapter._missing_kernels


def test_a_missing_namespace_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace())
    assert _adapter(*TurboQuantKernelStage)._has_kernel(TURBOQUANT_OP_ROTATE) is False


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotate_golden_runs_when_the_stage_is_off(emulator: TurboQuantEmulator) -> None:
    x = torch.randn(3, HEAD_SIZE)
    assert torch.equal(_adapter().rotate(x, emulator), emulator.rotate(x))
    assert torch.equal(_adapter().inverse_rotate(x, emulator), emulator.inverse_rotate(x))


def test_rotate_dispatches_both_directions_to_the_kernel(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    ingress, egress = torch.zeros(1), torch.ones(1)
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_rotate=entry)
    adapter = _adapter(TurboQuantKernelStage.ROTATE)
    x = torch.randn(3, HEAD_SIZE)

    entry.result = ingress
    assert adapter.rotate(x, emulator) is ingress
    entry.result = egress
    assert adapter.inverse_rotate(x, emulator) is egress

    # (x, pi_signs, rot_tables, h16, output_stage, inv_sqrt_len).
    first, second = (call[0] for call in entry.calls)
    assert first[0] is x
    assert torch.equal(first[1], emulator.pi_signs)
    assert first[2].dtype is torch.int8
    assert first[3].shape == (16, 16)
    assert first[4] == TURBOQUANT_STAGE_ROTATED_BASIS
    assert first[5] == 1.0
    assert second[4] == TURBOQUANT_STAGE_UNROTATED
    assert second[5] == pytest.approx(emulator.inv_sqrt_len)


# ---------------------------------------------------------------------------
# Quantisation and packing
# ---------------------------------------------------------------------------


def test_quantize_pack_golden_runs_when_the_stage_is_off(emulator: TurboQuantEmulator) -> None:
    rotated = emulator.rotate(torch.randn(3, HEAD_SIZE))
    packed, scales = _adapter().quantize_and_pack(rotated, emulator)
    codes, expected_scales = emulator.quantize(rotated)
    assert torch.equal(packed, emulator.pack(codes))
    assert torch.equal(scales, expected_scales)
    assert torch.equal(
        _adapter().unpack_and_dequantize(packed, scales, emulator),
        emulator.dequantize(emulator.unpack(packed), scales),
    )


def test_quantize_pack_dispatches_to_the_kernel(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> None:
    result = (torch.zeros(1, dtype=torch.int8), torch.ones(1))
    entry = _Recorder(result)
    _register(monkeypatch, npu_turboquant_quantize_pack=entry)
    rotated = torch.randn(3, HEAD_SIZE)

    assert _adapter(TurboQuantKernelStage.QUANTIZE).quantize_and_pack(rotated, emulator) is result
    assert entry.args[0] is rotated
    assert entry.args[1] == pytest.approx(emulator.inv_sqrt_len)


def test_unpack_dequant_dispatches_to_the_kernel(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> None:
    result = torch.ones(1)
    entry = _Recorder(result)
    _register(monkeypatch, npu_turboquant_unpack_dequant=entry)
    packed = torch.zeros(3, turboquant_packed_bytes(HEAD_SIZE), dtype=torch.int8)
    scales = torch.ones(3)

    assert _adapter(TurboQuantKernelStage.DEQUANTIZE).unpack_and_dequantize(packed, scales, emulator) is result
    assert entry.args == (packed, scales)


# ---------------------------------------------------------------------------
# Ingestion: the two golden writers and the two entry points
# ---------------------------------------------------------------------------


def _assert_cache_holds(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    emulator: TurboQuantEmulator,
) -> None:
    """Every written slot carries exactly ``emulator.encode``'s bytes and scales."""
    for token, slot in enumerate(slot_mapping.tolist()):
        page, offset = divmod(slot, BLOCK_SIZE)
        for plane, activation, cache in (
            (TURBOQUANT_KEY_PLANE, key, key_cache),
            (TURBOQUANT_VALUE_PLANE, value, value_cache),
        ):
            packed, scales = emulator.encode(activation[token])
            assert torch.equal(cache[page, offset], packed)
            assert torch.equal(scale_cache[plane, page, :, offset], scales)


def test_both_golden_writers_land_the_codec_bytes(emulator: TurboQuantEmulator) -> None:
    key, value = _activations(BLOCK_SIZE)
    slot_mapping = torch.arange(BLOCK_SIZE, dtype=torch.int32)

    prefill = _caches()
    _adapter().reshape_and_cache_prefill(key, value, *prefill, slot_mapping, emulator)
    _assert_cache_holds(*prefill, key, value, slot_mapping, emulator)

    decode = _caches()
    _adapter().reshape_scatter_decode(key, value, *decode, slot_mapping, emulator)
    for written, expected in zip(decode, prefill):
        assert torch.equal(written, expected)


def test_the_decode_writer_skips_padding_slots(emulator: TurboQuantEmulator) -> None:
    key, value = _activations(3)
    slot_mapping = torch.tensor([5, -1, 11], dtype=torch.int32)
    caches = _caches()
    _adapter().reshape_scatter_decode(key, value, *caches, slot_mapping, emulator)

    key_cache, _, scale_cache = caches
    assert torch.equal(key_cache[0, 6], torch.zeros_like(key_cache[0, 6]))
    assert torch.equal(scale_cache[TURBOQUANT_KEY_PLANE, 0, :, 6], torch.zeros(NUM_KV_HEADS))
    for token, slot in ((0, 5), (2, 11)):
        page, offset = divmod(slot, BLOCK_SIZE)
        assert torch.equal(key_cache[page, offset], emulator.encode(key[token])[0])


def test_a_strip_layout_refuses_a_block_size_it_cannot_hold(emulator: TurboQuantEmulator) -> None:
    """The strip cache is written 16 tokens at a time, so the page must hold whole strips.

    The golden writer used to refuse the channel-major layout outright, because
    the value plane's read-modify-write was the kernel's to do. It does the
    read-modify-write itself now, so the only thing left to refuse is a page
    geometry the layout cannot express -- and this suite's BLOCK_SIZE is 8.
    """
    adapter = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(),
            value_layout=TurboQuantValueLayout.CHANNEL_MAJOR,
        )
    )
    key, value = _activations(1)
    with pytest.raises(ValueError, match="multiple of 16"):
        adapter.reshape_and_cache_prefill(key, value, *_caches(), torch.zeros(1, dtype=torch.int32), emulator)


def test_the_prefill_entry_point_takes_the_batch_writer_abi(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_reshape_and_cache_a8=entry)
    key, value = _activations(BLOCK_SIZE)
    caches = _caches()
    slot_mapping = torch.arange(BLOCK_SIZE, dtype=torch.int32)

    _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE).reshape_and_cache_prefill(
        key, value, *caches, slot_mapping, emulator
    )

    # The schema's order: key, value, the three planes, slot_mapping, pi_signs,
    # tables, inv_sqrt_len, reuse_mode4_flags. Every scalar the C launcher also
    # takes is derived in torch_bindings.cpp from these tensors.
    args = entry.args
    assert len(args) == 10
    assert (args[0], args[1]) == (key, value)
    assert (args[2], args[3], args[4], args[5]) == (*caches, slot_mapping)
    assert torch.equal(args[6], emulator.pi_signs)
    assert args[7].dtype is torch.int8
    assert args[7].numel() == 2 * (HEAD_SIZE // 32) * 32 * 32
    assert args[8] == pytest.approx(emulator.inv_sqrt_len)
    assert args[9] is TURBOQUANT_REUSE_MODE4_FLAGS
    # The kernel owns the write, so the golden model must not have run.
    assert torch.equal(caches[0], torch.zeros_like(caches[0]))


def test_the_decode_entry_point_takes_the_scatter_writer_abi(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_reshape_scatter_aiv=entry)
    key, value = _activations(2)
    caches = _caches()
    slot_mapping = torch.tensor([0, 9], dtype=torch.int32)

    _adapter(TurboQuantKernelStage.SCATTER).reshape_scatter_decode(key, value, *caches, slot_mapping, emulator)

    # The work split (block_dim, item_begin, item_count), the canary probe and
    # the cu_seqlens prefix sum are all the binding's, so the schema is tensors
    # plus the scale fold.
    args = entry.args
    assert len(args) == 8
    assert (args[0], args[1]) == (key, value)
    assert (args[2], args[3], args[4], args[5]) == (*caches, slot_mapping)
    assert torch.equal(args[6], emulator.pi_signs)
    assert args[7] == pytest.approx(emulator.inv_sqrt_len)
    assert torch.equal(caches[0], torch.zeros_like(caches[0]))


# ---------------------------------------------------------------------------
# The cache read
# ---------------------------------------------------------------------------


def test_gather_round_trips_through_the_golden_path(emulator: TurboQuantEmulator) -> None:
    key, value = _activations(BLOCK_SIZE)
    caches = _caches()
    _adapter().reshape_and_cache_prefill(key, value, *caches, torch.arange(BLOCK_SIZE, dtype=torch.int32), emulator)

    adapter = _adapter()
    packed_key, key_scales, _, _ = adapter.gather(*caches, torch.zeros(1, 1, dtype=torch.int32))
    dense = adapter.dequantize_cache(packed_key, key_scales, emulator, torch.float32)
    assert dense.shape == (1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert torch.allclose(dense[0], emulator.decode(*emulator.encode(key)), atol=0.0)


def test_gather_dispatches_to_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    result = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    entry = _Recorder(result)
    _register(monkeypatch, npu_turboquant_gather_pages=entry)
    caches = _caches()
    block_tables = torch.zeros(1, 1, dtype=torch.int32)

    assert _adapter(TurboQuantKernelStage.GATHER).gather(*caches, block_tables) is result
    assert entry.args == (*caches, block_tables)


# ---------------------------------------------------------------------------
# The fused decode
# ---------------------------------------------------------------------------


def _fused_decode(adapter: TurboQuantOpsAdapter, emulator: TurboQuantEmulator) -> torch.Tensor | None:
    caches = _caches()
    return adapter.fused_decode(
        query=torch.randn(1, 2, HEAD_SIZE),
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        context_lens=torch.ones(1, dtype=torch.int32),
        emulator=emulator,
        num_heads=2,
        num_kv_heads=NUM_KV_HEADS,
        scale=0.125,
        output=torch.zeros(1, 2, HEAD_SIZE),
    )


def test_fused_decode_returns_none_without_a_kernel(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    _register(monkeypatch, npu_turboquant_fused_decode=_Recorder())
    assert _fused_decode(_adapter(), emulator) is None


def test_fused_decode_launches_and_returns_its_output(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """With the op registered the seam is taken and `output` comes back, not None.

    `None` is reserved for "the golden gather, dequantise and attend path has to
    run", so a launched decode must be distinguishable from it.
    """
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    adapter = _adapter(TurboQuantKernelStage.DECODE)
    caches = _caches()
    output = torch.zeros(1, 2, HEAD_SIZE)

    assert adapter._has_kernel(TURBOQUANT_OP_FUSED_DECODE) is True
    returned = adapter.fused_decode(
        query=torch.randn(1, 2, HEAD_SIZE),
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=torch.zeros(1, 3, dtype=torch.int32),
        context_lens=torch.ones(1, dtype=torch.int32),
        emulator=emulator,
        num_heads=2,
        num_kv_heads=NUM_KV_HEADS,
        scale=0.125,
        output=output,
    )
    assert returned is output
    assert len(entry.calls) == 1


def test_fused_decode_passes_the_schema_the_binding_declares(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The 18 arguments of npu_turboquant_fused_decode, in the schema's order.

    The mode, the element type, the host tiling descriptors, the Split-K
    workspace and the fp32 rotated query image are all the binding's: the schema
    does not take them, so none of them appears here.
    """
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    caches = _caches()
    query = torch.randn(1, 12, HEAD_SIZE)
    output = torch.zeros_like(query)
    block_tables = torch.zeros(1, 3, dtype=torch.int32)
    context_lens = torch.ones(1, dtype=torch.int32)

    _adapter(TurboQuantKernelStage.DECODE).fused_decode(
        query=query,
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=block_tables,
        context_lens=context_lens,
        emulator=emulator,
        num_heads=12,
        num_kv_heads=NUM_KV_HEADS,
        scale=0.125,
        output=output,
    )

    args = entry.args
    assert len(args) == 18
    assert args[0] is query
    assert torch.equal(args[1], emulator.pi_signs)
    assert args[2].dtype is torch.int8  # rot_tables
    assert args[3].shape == (16, 16) and args[3].dtype is torch.float16  # h16
    assert args[4] is None  # gate, read only at output_stage 2
    assert (args[5], args[6], args[7]) == caches
    assert (args[8], args[9], args[10]) == (block_tables, context_lens, output)
    # num_heads, num_kv_heads, num_splits, heads_per_task, output_stage
    assert args[11:16] == (12, NUM_KV_HEADS, TURBOQUANT_DECODE_SPLITS, 6, TURBOQUANT_STAGE_UNROTATED)
    assert args[16] == pytest.approx(0.125)
    assert args[17] == pytest.approx(emulator.inv_sqrt_len)


# ---------------------------------------------------------------------------
# The phase split in the backend
# ---------------------------------------------------------------------------


class _PhaseRecordingAdapter(TurboQuantOpsAdapter):
    """Records which ingestion entry point the backend chose."""

    def __init__(self) -> None:
        super().__init__(TurboQuantOpsConfig(ops_namespace=NAMESPACE, kernel_stages=frozenset()))
        self.phases: list[TurboQuantIngestionPhase] = []

    def reshape_and_cache_prefill(self, **kwargs: Any) -> None:
        self.phases.append(TurboQuantIngestionPhase.PREFILL)

    def reshape_scatter_decode(self, **kwargs: Any) -> None:
        self.phases.append(TurboQuantIngestionPhase.DECODE)


class _Impl(AscendTurboQuantDryRunBackendImpl):
    """The backend's ``reshape_and_cache`` over only the state it reads."""

    def __init__(self, emulator: TurboQuantEmulator) -> None:
        self.attn_type = "decoder"
        self.kv_sharing_target_layer_name = None
        self.key_cache, self.value_cache, self.scale_cache = _caches()
        self.emulator = emulator
        self.adapter = _PhaseRecordingAdapter()


def _metadata(num_actual_tokens: int, seq_lens_list: Any, with_field: bool = True) -> Any:
    fields: dict[str, Any] = {
        "num_actual_tokens": num_actual_tokens,
        "slot_mapping": torch.arange(max(num_actual_tokens, 1), dtype=torch.int32),
    }
    if with_field:
        fields["seq_lens_list"] = seq_lens_list
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("num_actual_tokens", "seq_lens_list", "expected"),
    [
        (8, [8], TurboQuantIngestionPhase.PREFILL),
        (4, [2, 2], TurboQuantIngestionPhase.PREFILL),
        (2, [4, 9], TurboQuantIngestionPhase.DECODE),
        (1, [7], TurboQuantIngestionPhase.DECODE),
    ],
)
def test_the_phase_comes_from_the_token_count(
    num_actual_tokens: int, seq_lens_list: list[int], expected: TurboQuantIngestionPhase
) -> None:
    phase = AscendTurboQuantDryRunBackendImpl._ingestion_phase(
        _metadata(num_actual_tokens, seq_lens_list), num_actual_tokens
    )
    assert phase is expected


@pytest.mark.parametrize("metadata", [_metadata(2, None), _metadata(2, None, with_field=False)])
def test_absent_or_none_seq_lens_list_reads_as_prefill(metadata: Any) -> None:
    assert AscendTurboQuantDryRunBackendImpl._ingestion_phase(metadata, 2) is TurboQuantIngestionPhase.PREFILL


@pytest.mark.parametrize(
    ("num_actual_tokens", "seq_lens_list", "expected"),
    [
        (8, [8], TurboQuantIngestionPhase.PREFILL),
        (2, [4, 9], TurboQuantIngestionPhase.DECODE),
        (2, None, TurboQuantIngestionPhase.PREFILL),
    ],
)
def test_reshape_and_cache_routes_by_phase(
    emulator: TurboQuantEmulator,
    num_actual_tokens: int,
    seq_lens_list: Any,
    expected: TurboQuantIngestionPhase,
) -> None:
    impl = _Impl(emulator)
    key, value = _activations(num_actual_tokens)
    kv_cache = (impl.key_cache, impl.value_cache, impl.scale_cache)
    query = torch.zeros(num_actual_tokens, 2, HEAD_SIZE)
    output = torch.zeros_like(query)

    returned = impl.reshape_and_cache(query, key, value, kv_cache, _metadata(num_actual_tokens, seq_lens_list), output)

    assert impl.adapter.phases == [expected]
    assert returned == (query, key, value, output)


def test_no_tokens_writes_nothing(emulator: TurboQuantEmulator) -> None:
    impl = _Impl(emulator)
    kv_cache = (impl.key_cache, impl.value_cache, impl.scale_cache)
    empty = torch.zeros(0, NUM_KV_HEADS, HEAD_SIZE)
    impl.reshape_and_cache(empty, empty, empty, kv_cache, _metadata(0, []), empty)
    assert impl.adapter.phases == []


# ---------------------------------------------------------------------------
# Loading the extension that carries the registrations
# ---------------------------------------------------------------------------


@pytest.fixture
def unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the one-shot loader back to "not tried yet" for this test."""
    monkeypatch.setattr(turboquant_dry_run, "_TURBOQUANT_EXTENSION_LOADED", None)


def test_a_phantom_namespace_is_not_mistaken_for_a_registration() -> None:
    """``getattr(torch.ops, ns)`` is not evidence, and must not be used as it.

    ``torch.ops.__getattr__`` manufactures an ``_OpNamespace`` for any name and
    caches it, so it never raises and the name then appears in
    ``dir(torch.ops)``. A presence check written either way would report these
    kernels as registered on a host that has none of them -- which is the whole
    failure this seam exists to report -- so the trap is pinned here.
    """
    phantom = "turboquant_ops_nothing_registers_this"
    assert getattr(torch.ops, phantom, None) is not None
    assert phantom in dir(torch.ops)
    assert _is_op_registered(phantom, "npu_turboquant_fused_decode") is False


def test_a_real_operator_is_reported_as_registered() -> None:
    """The same probe answers yes for something the dispatcher really has."""
    assert _is_op_registered("aten", "mm") is True


def test_no_stage_switched_on_does_not_touch_the_extension(unloaded: None) -> None:
    """A golden-only run must not pay for the import, nor warn about it."""
    _adapter()
    assert turboquant_dry_run._TURBOQUANT_EXTENSION_LOADED is None


def test_a_stage_switched_on_attempts_the_import_once(unloaded: None) -> None:
    _adapter(*TurboQuantKernelStage)
    first = turboquant_dry_run._TURBOQUANT_EXTENSION_LOADED
    assert first is not None, "a stage-on adapter must have tried the import"

    calls: list[int] = []
    monkey = pytest.MonkeyPatch()
    try:
        # Anything the loader would do on a second attempt lands here instead.
        monkey.setattr(turboquant_dry_run.torch.compiler, "is_compiling", lambda: calls.append(1) or False)
        assert load_turboquant_extension() is first
    finally:
        monkey.undo()
    assert calls == [], "the loader must be one-shot per process"


def test_missing_operators_are_reported_with_the_stages_that_wanted_them(
    unloaded: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _adapter(TurboQuantKernelStage.SCATTER)
    assert any(
        "are switched on but torch.ops" in record.message and "ascend950" in record.message for record in caplog.records
    ), f"expected one actionable warning, got {[r.message for r in caplog.records]}"


def test_a_fully_registered_namespace_is_quiet(
    unloaded: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With every registered op present, construction says nothing at WARNING."""
    monkeypatch.setattr(turboquant_dry_run, "_is_op_registered", lambda namespace, op_name: True)
    with caplog.at_level(logging.WARNING):
        _adapter(*TurboQuantKernelStage)
    assert caplog.records == []


def test_every_reported_operator_is_one_the_bindings_declare() -> None:
    """``TURBOQUANT_REGISTERED_OPS`` must not drift from the three real schemas."""
    assert set(TURBOQUANT_REGISTERED_OPS) == {
        TURBOQUANT_OP_RESHAPE_AND_CACHE,
        TURBOQUANT_OP_RESHAPE_SCATTER,
        TURBOQUANT_OP_FUSED_DECODE,
    }
    assert set(TURBOQUANT_REGISTERED_OPS) <= set(ALL_OPS)
