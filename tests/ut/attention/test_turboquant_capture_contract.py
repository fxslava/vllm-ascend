# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TurboQuant decode's ACL-graph capture contract, checked without a device.

Four gates live here, named after the RFC
(``TURBOQUANT_GRAPH_CAPTURE_RFC.md`` §6):

* **G10** -- under a capture, the Python half of the decode launch path issues no
  host synchronisation and allocates nothing. This is the gate the whole change
  set exists to turn green, and it is written as an *allowlist* of the torch
  functions the clean path is permitted to call: anything else fails the test by
  name, so a regression says which operation reappeared rather than that a
  counter moved.
* **G5** -- the op is handed a *device* ``context_lens``, not the CPU mirror
  ``AscendMetadata.seq_lens`` points at.
* **G4** -- no ``int``/``float`` argument of ``npu_turboquant_fused_decode``
  depends on the attention metadata. Those arguments are baked into a captured
  launch, so one that varied per step would replay a stale value.
* **G6** -- the tiling buffer's Python-side upper bound covers every
  ``(tokens, splits, grid)`` the op can derive.
* **C1-C5** -- the capture contract for the *static buffer* itself, which is
  the half G10 cannot see: that the extent the host filled is the extent the
  launch reads, that every buffer keeps one address across steps, that the
  fill never migrates into the launch, that a step this builder cannot tile is
  refused instead of mis-tiled, and that the host staging is safe to overwrite
  on the next step.

**What G10 does not cover.** Everything the C++ binding does between the op call
and the launch is invisible from here. That half of the launcher is covered by
the schema instead: ``tiling``, ``workspace`` and ``query_rot`` are *inputs* of
the op (see ``test_turboquant_op_schemas.py``), so the binding cannot allocate
them, and the remaining host builds -- the basis codec pool, ``mode_tables`` and
the vector-core count -- are function-local statics keyed by
``(head_size, device_index)``. There is no device-free way to assert a static
local from Python; the evidence for those is the schema plus the source.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch
from torch.overrides import TorchFunctionMode

from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_STAGING_SLOTS,
    AscendTurboQuantDryRunBackendImpl,
    TurboQuantEmulator,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    turboquant_decode_tiling_words,
    turboquant_tiling_upper_bound_words,
)
from vllm_ascend.attention.turboquant_layout import turboquant_packed_bytes, turboquant_scale_cache_shape

NAMESPACE = "turboquant_ops"
CPU = torch.device("cpu")

HEAD_SIZE = 256
NUM_KV_HEADS = 2
NUM_HEADS = 8
BLOCK_SIZE = 128
NUM_BLOCKS = 4
BATCH = 2
MAX_BLOCKS = 2


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
    def __init__(self, entries: dict[str, _Recorder]) -> None:
        self._entries = entries

    def __getattr__(self, name: str) -> _Recorder:
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(name) from None


def _register(monkeypatch: pytest.MonkeyPatch, **entries: _Recorder) -> None:
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(**{NAMESPACE: _OpNamespace(dict(entries))}))


# ---------------------------------------------------------------------------
# The torch-function recorder
# ---------------------------------------------------------------------------

# Every torch function the clean decode launch path may call. All of them are
# views, metadata reads or pure Python-visible shape arithmetic: none allocates
# device memory and none reads device memory on the host.
_ALLOWED = frozenset(
    {
        "__get__",  # Tensor.device / .dtype / .shape property access
        "__getitem__",
        "__len__",
        "contiguous",  # a no-op on an already-contiguous tensor
        "detach",
        "dim",
        "is_contiguous",
        "narrow",
        "numel",
        "reshape",
        "size",
        "stride",
        "unsqueeze",
        "view",
    }
)

# Host synchronisations: a device-to-host read. Illegal on a capturing stream
# (107030) and, where it feeds a Python branch, baked into the capture.
_SYNC = frozenset({"cpu", "item", "tolist", "numpy", "__bool__", "__float__", "__int__", "__index__"})


class _LaunchWatcher(TorchFunctionMode):
    """Record every torch function the wrapped region calls."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        name = getattr(func, "__name__", repr(func))
        self.names.append(name)
        if name == "to":
            # `.to()` is only a sync when it crosses to the host; `.to(dtype)`
            # on a device tensor is an ordinary (allocating) device op, which
            # the allowlist already refuses.
            target = kwargs.get("device", args[1] if len(args) > 1 else None)
            if isinstance(target, (str, torch.device)) and torch.device(target).type == "cpu":
                self.names.append("to(cpu)")
        return func(*args, **kwargs)

    @property
    def syncs(self) -> list[str]:
        return [n for n in self.names if n in _SYNC or n == "to(cpu)"]

    @property
    def disallowed(self) -> list[str]:
        return sorted({n for n in self.names if n not in _ALLOWED})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(turboquant_scale_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS), dtype=torch.float32),
    )


def _tier1(tokens: int) -> dict[str, torch.Tensor]:
    """The three buffers the metadata builder now owns, at this geometry."""
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_PARTIAL_TAIL

    return {
        "turboquant_tiling": torch.zeros(turboquant_tiling_upper_bound_words(tokens, 1), dtype=torch.int32),
        "turboquant_workspace": torch.zeros(
            tokens * NUM_HEADS * TURBOQUANT_DECODE_SPLITS * (HEAD_SIZE + TURBOQUANT_PARTIAL_TAIL),
            dtype=torch.float32,
        ),
        "turboquant_query_rot": torch.zeros(tokens * NUM_HEADS * HEAD_SIZE, dtype=torch.float32),
    }


def _metadata(tokens: int = BATCH) -> Any:
    """Decode metadata as the TurboQuant builder produces it.

    ``seq_lens`` is the CPU mirror, as the shared builder leaves it;
    ``seq_lens_gpu`` is the persistent device buffer; ``turboquant_context_lens``
    is the TurboQuant branch's own device view, padded with zeros rather than
    with the ones FIA needs.
    """
    seq_lens_cpu = torch.full((tokens,), 64, dtype=torch.int32)
    fields: dict[str, Any] = {
        "num_actual_tokens": tokens,
        "seq_lens": seq_lens_cpu,
        "seq_lens_cpu": seq_lens_cpu,
        "seq_lens_gpu": seq_lens_cpu.clone(),
        "seq_lens_list": [64] * tokens,
        "block_tables": torch.zeros(tokens, MAX_BLOCKS, dtype=torch.int32),
        "turboquant_context_lens": seq_lens_cpu.clone(),
        "turboquant_num_tokens": tokens,
        "slot_mapping": torch.arange(tokens, dtype=torch.int32),
    }
    fields.update(_tier1(tokens))
    return types.SimpleNamespace(**fields)


def _impl(adapter: TurboQuantOpsAdapter, emulator: TurboQuantEmulator, layer: Any) -> Any:
    impl = object.__new__(AscendTurboQuantDryRunBackendImpl)
    impl.key_cache, impl.value_cache, impl.scale_cache = _caches()
    impl.emulator = emulator
    impl.adapter = adapter
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.head_size = HEAD_SIZE
    impl.scale = 0.125
    impl.layer = layer
    return impl


def _layer(emulator: TurboQuantEmulator) -> Any:
    """A layer carrying the Tier-0 constants ``_ensure_statics`` binds."""
    from vllm_ascend.attention.turboquant_dry_run import _codec_tables, _h16_operand

    return types.SimpleNamespace(
        _turboquant_pi_signs=emulator.pi_signs,
        _turboquant_rotation=emulator.rotation,
        _turboquant_codec_tables=_codec_tables(emulator),
        _turboquant_h16=_h16_operand(emulator.device),
        _turboquant_inv_sqrt_len=emulator.inv_sqrt_len,
    )


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


@pytest.fixture
def decode(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> dict[str, Any]:
    """A decode launch with the fused op registered, ready to run under capture."""
    entry = _Recorder(result=None)
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    adapter = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset({TurboQuantKernelStage.DECODE}),
            value_layout=TurboQuantValueLayout.CHANNEL_MAJOR,
        )
    )
    layer = _layer(emulator)
    impl = _impl(adapter, emulator, layer)
    metadata = _metadata()
    output = torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    query = torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    return {"impl": impl, "entry": entry, "metadata": metadata, "query": query, "output": output}


def _run_decode(decode: dict[str, Any]) -> _LaunchWatcher:
    watcher = _LaunchWatcher()
    with watcher:
        decode["impl"]._forward_decode(decode["query"], decode["metadata"], decode["output"])
    return watcher


# ---------------------------------------------------------------------------
# G10
# ---------------------------------------------------------------------------


def test_g10_the_decode_launch_performs_no_host_sync(decode: dict[str, Any]) -> None:
    """No device-to-host read anywhere in the Python half of the launch."""
    watcher = _run_decode(decode)
    assert watcher.syncs == [], (
        f"the decode launch synchronised with the host: {watcher.syncs}. "
        "Every one of these is illegal on a capturing stream (107030)."
    )


def test_g10_the_decode_launch_allocates_nothing(decode: dict[str, Any]) -> None:
    """Only views and metadata reads between entering the decode and the op call.

    The allowlist is the point: a new entry in the failure message is the name of
    the operation that reappeared in the launch path.
    """
    watcher = _run_decode(decode)
    assert watcher.disallowed == [], (
        f"the decode launch called {watcher.disallowed}, none of which is a view "
        "or a metadata read. Every tensor the launch needs is either a Tier-0 "
        "constant on the layer or a Tier-1 buffer on the metadata."
    )


def test_g10_the_second_step_rebuilds_no_rotation_constant(
    monkeypatch: pytest.MonkeyPatch, decode: dict[str, Any]
) -> None:
    """``_sylvester_hadamard`` is not called by a decode step, ever.

    It is how both ``_codec_tables`` and ``_h16_operand`` build their operands,
    and those are pure functions of the head size. A call here is B7 of the RFC.
    """
    from vllm_ascend.attention import turboquant_dry_run as module

    calls: list[int] = []
    real = module._sylvester_hadamard

    def counting(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        calls.append(order)
        return real(order, device, dtype)

    monkeypatch.setattr(module, "_sylvester_hadamard", counting)
    _run_decode(decode)
    _run_decode(decode)
    assert calls == [], f"the decode rebuilt a rotation constant {len(calls)} times (orders {calls})"


# ---------------------------------------------------------------------------
# G5
# ---------------------------------------------------------------------------

# The schema's argument order, as `ops.def` declares it. Pinned against the
# bindings by test_turboquant_op_schemas.py; named here so a positional
# assertion reads as the argument rather than as an index.
_ARG = {
    name: index
    for index, name in enumerate(
        (
            "query",
            "pi_signs",
            "rot_tables",
            "h16",
            "gate",
            "key_cache",
            "value_cache",
            "scale_cache",
            "block_tables",
            "context_lens",
            "tiling",
            "workspace",
            "query_rot",
            "output",
            "num_heads",
            "num_kv_heads",
            "num_splits",
            "heads_per_task",
            "output_stage",
            "scale",
            "inv_sqrt_len",
        )
    )
}


def test_g5_the_op_is_handed_a_device_context_lens(decode: dict[str, Any]) -> None:
    """The op must not receive ``AscendMetadata.seq_lens``, the CPU mirror.

    On this host every tensor is on the CPU, so the assertion is on *identity*:
    the argument has to be the metadata's device-side context lengths and not
    the host mirror the FIA path reads.
    """
    _run_decode(decode)
    metadata = decode["metadata"]
    passed = decode["entry"].args[_ARG["context_lens"]]
    assert passed is not metadata.seq_lens, (
        "the fused decode was handed AscendMetadata.seq_lens, which is the CPU "
        "mirror (_select_seq_lens prefers _seq_lens_cpu); the binding would copy "
        "it to the device on every launch."
    )
    assert passed.data_ptr() == metadata.turboquant_context_lens.data_ptr()


def test_g5_the_op_is_handed_the_builder_owned_tier1_buffers(decode: dict[str, Any]) -> None:
    """``tiling``, ``workspace`` and ``query_rot`` come off the metadata.

    A buffer allocated inside the launch has a fresh address every step, which a
    captured graph cannot follow.
    """
    _run_decode(decode)
    metadata = decode["metadata"]
    args = decode["entry"].args
    for name, field in (
        ("tiling", "turboquant_tiling"),
        ("workspace", "turboquant_workspace"),
        ("query_rot", "turboquant_query_rot"),
    ):
        assert args[_ARG[name]].data_ptr() == getattr(metadata, field).data_ptr(), (
            f"{name} is not the builder-owned buffer on the metadata"
        )


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------


def test_g4_no_scalar_argument_depends_on_the_metadata(decode: dict[str, Any]) -> None:
    """Every ``int``/``float`` argument is config or tensor geometry.

    A captured launch replays its scalar arguments verbatim, so one taken from
    ``attn_metadata`` would be the first captured step's value forever. The
    assertion is that each scalar equals the value derived from the impl's
    configuration, and that running a *different* batch leaves all of them
    unchanged.
    """
    _run_decode(decode)
    first = decode["entry"].args
    impl = decode["impl"]
    group_heads = NUM_HEADS // NUM_KV_HEADS

    expected = {
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "num_splits": TURBOQUANT_DECODE_SPLITS,
        "heads_per_task": group_heads,
        "output_stage": 1,
        "scale": impl.scale,
        "inv_sqrt_len": impl.emulator.inv_sqrt_len,
    }
    for name, value in expected.items():
        assert first[_ARG[name]] == value, f"{name} is {first[_ARG[name]]!r}, expected {value!r}"

    # A second step with different context lengths and a different request
    # count must not move any of them.
    decode["entry"].calls.clear()
    other = _metadata(tokens=1)
    other.seq_lens_gpu = torch.tensor([4096], dtype=torch.int32)
    other.turboquant_context_lens = torch.tensor([4096], dtype=torch.int32)
    impl._forward_decode(
        torch.zeros(1, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        other,
        torch.zeros(1, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
    )
    second = decode["entry"].args
    for name in expected:
        assert second[_ARG[name]] == first[_ARG[name]], f"{name} changed between steps"


# ---------------------------------------------------------------------------
# G6
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 2, 7, 16, 31, 64, 128, 256])
@pytest.mark.parametrize("splits", [1, 2, 4, 8])
def test_g6_the_upper_bound_covers_every_derivable_geometry(tokens: int, splits: int) -> None:
    """``turboquant_tiling_upper_bound_words`` is never smaller than the need.

    The bound is sized from ``max_num_reqs`` and the maximum split count, so a
    step with fewer requests or fewer splits must fit inside it. ``grid`` is the
    op's own derivation and is capped at ``kDecodeMaxGrid``.
    """
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_DECODE_MAX_GRID, TURBOQUANT_DECODE_MAX_SPLITS

    bound = turboquant_tiling_upper_bound_words(tokens, TURBOQUANT_DECODE_MAX_SPLITS)
    for grid in range(1, TURBOQUANT_DECODE_MAX_GRID + 1):
        need = turboquant_decode_tiling_words(tokens, splits, grid)
        assert bound >= need, f"bound {bound} < need {need} at tokens={tokens} splits={splits} grid={grid}"


def test_g6_the_bound_is_monotone_in_the_request_count() -> None:
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_DECODE_MAX_SPLITS

    bounds = [turboquant_tiling_upper_bound_words(n, TURBOQUANT_DECODE_MAX_SPLITS) for n in range(1, 33)]
    assert bounds == sorted(bounds)
    assert len(set(bounds)) > 1


# ---------------------------------------------------------------------------
# The builder: the padded-row contract and the once-per-step fill
# ---------------------------------------------------------------------------

MAX_NUM_REQS = 8


def _builder() -> Any:
    """The TurboQuant builder over only the state its two per-step methods read.

    ``object.__new__`` because the real ``__init__`` runs the shared builder's,
    which needs a ``VllmConfig``; everything below the allocation is what is
    under test.
    """
    from vllm_ascend.attention.turboquant_dry_run import (
        AscendTurboQuantMetadataBuilder,
        turboquant_decode_query_rot_floats,
        turboquant_decode_workspace_floats,
    )

    builder = object.__new__(AscendTurboQuantMetadataBuilder)
    builder._max_num_reqs = MAX_NUM_REQS
    builder._num_kv_heads = NUM_KV_HEADS
    builder._head_size = HEAD_SIZE
    builder._block_size = BLOCK_SIZE
    builder._num_heads = NUM_HEADS
    builder._num_splits = TURBOQUANT_DECODE_SPLITS
    builder._heads_per_task = NUM_HEADS // NUM_KV_HEADS
    builder.max_num_blocks_per_req = MAX_BLOCKS
    builder._context_lens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    # The production depth, not a literal: the ring gate below has to go red
    # when the real builder drops to a single slot.
    builder._staging_slots = TURBOQUANT_STAGING_SLOTS
    builder._staging_step = 0
    builder._staging_pinned = False
    builder._context_lens_host_ring = [
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32) for _ in range(builder._staging_slots)
    ]
    builder._tiling = torch.zeros(turboquant_tiling_upper_bound_words(MAX_NUM_REQS, 8), dtype=torch.int32)
    builder._workspace = torch.zeros(
        turboquant_decode_workspace_floats(MAX_NUM_REQS, NUM_HEADS, HEAD_SIZE, TURBOQUANT_DECODE_SPLITS),
        dtype=torch.float32,
    )
    builder._query_rot = torch.zeros(
        turboquant_decode_query_rot_floats(MAX_NUM_REQS, NUM_HEADS, HEAD_SIZE), dtype=torch.float32
    )
    return builder


def test_the_padded_rows_get_a_zero_context_not_a_one() -> None:
    """The contract that makes a padded extent safe for this decode.

    The shared builder pads ``seq_lens`` with **1** and a zero block-table row
    when cudagraph or sequence-parallel padding adds a dummy request
    (``attention_v1.py:387-390``), because FIA validates ``actualSeqLengthsKv``
    and refuses a zero. ``build_decode_tiling`` reads a 1 as a *live* worker --
    one tile over page 0 -- so that row would attend over whatever block 0 holds
    and write a nonzero result. Zero is the all-zero descriptor, which is what
    ``num_tiles == 0`` means and the single predicate both kernel roles branch
    on to skip a worker.
    """
    builder = _builder()
    # Two real requests, two FIA dummies: seq_lens arrives length 4 with ones
    # in the tail, exactly as the shared builder leaves it.
    seq_lens = torch.tensor([64, 128, 1, 1], dtype=torch.int32)
    common = types.SimpleNamespace(num_reqs=2)

    context_lens, host = builder._refresh_context_lens(common, seq_lens)

    assert context_lens.tolist() == [64, 128, 0, 0]
    assert context_lens.numel() == 4, "the view covers the padded extent, not just the live requests"
    assert host[:4].tolist() == [64, 128, 0, 0], "the host staging is what build_decode_tiling reads"


def test_the_context_lens_view_is_always_the_same_buffer() -> None:
    """One address for the engine's life, which is what a captured graph needs."""
    builder = _builder()
    common = types.SimpleNamespace(num_reqs=2)
    first, _ = builder._refresh_context_lens(common, torch.tensor([64, 128], dtype=torch.int32))
    second, _ = builder._refresh_context_lens(common, torch.tensor([65, 129], dtype=torch.int32))
    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() == builder._context_lens.data_ptr()
    assert second.tolist() == [65, 129], "the refresh is in place, so the view reads the new step"


def test_a_step_wider_than_the_builder_was_sized_for_is_refused() -> None:
    builder = _builder()
    common = types.SimpleNamespace(num_reqs=MAX_NUM_REQS + 1)
    with pytest.raises(RuntimeError, match="sized for"):
        builder._refresh_context_lens(common, torch.ones(MAX_NUM_REQS + 1, dtype=torch.int32))


def test_the_tiling_fill_is_one_call_for_every_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once per step, with the schema's nine arguments, into the owned buffer.

    The point of the change: this used to happen inside the decode launch, so a
    sixty-layer model did it sixty times per step.
    """
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_OP_BUILD_DECODE_TILING

    entry = _Recorder()
    _register(monkeypatch, **{TURBOQUANT_OP_BUILD_DECODE_TILING: entry})
    builder = _builder()
    host = builder._context_lens_host_ring[0]
    builder._refresh_tiling(host, 4)

    assert len(entry.calls) == 1
    args = entry.args
    assert len(args) == 9
    assert args[0].data_ptr() == host.data_ptr()
    assert args[0].numel() == 4, "the host view is the step's extent"
    assert args[1] is builder._tiling
    assert args[2:] == (
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        MAX_BLOCKS,
        TURBOQUANT_DECODE_SPLITS,
        NUM_HEADS // NUM_KV_HEADS,
    )


def test_the_tiling_fill_is_skipped_when_the_operator_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build that is not an ascend950 one. The decode runs golden there.

    Filling a buffer nothing reads would only be a way to fail on the hosts the
    kernels do not exist on.
    """
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace())
    builder = _builder()
    builder._refresh_tiling(builder._context_lens_host_ring[0], 4)  # must not raise


def test_the_builder_publishes_every_field_the_launch_reads() -> None:
    """``_build_backend_metadata`` is the shared builder's extension point.

    Using it is what keeps ``AscendAttentionMetadataBuilder`` untouched, so no
    other backend can be perturbed by any of this (G7).
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
    from vllm_ascend.attention.turboquant_dry_run import AscendTurboQuantMetadataBuilder

    builder = _builder()
    extra = builder._build_backend_metadata(
        types.SimpleNamespace(num_reqs=2),
        block_table=torch.zeros(2, MAX_BLOCKS, dtype=torch.int32),
        query_lens=torch.ones(2, dtype=torch.int32),
        seq_lens=torch.tensor([64, 128], dtype=torch.int32),
        num_decodes=2,
        num_prefills=0,
    )
    assert set(extra) == {
        "turboquant_context_lens",
        "turboquant_tiling",
        "turboquant_workspace",
        "turboquant_query_rot",
        "turboquant_num_tokens",
    }
    assert extra["turboquant_num_tokens"] == 2, "the extent the descriptors were filled for"
    # All of the new work is in the subclass, reached only through the hook, so
    # a non-TurboQuant backend cannot execute any of it.
    assert issubclass(AscendTurboQuantMetadataBuilder, AscendAttentionMetadataBuilder)
    assert "_build_backend_metadata" in vars(AscendTurboQuantMetadataBuilder)
    assert "_refresh_context_lens" in vars(AscendTurboQuantMetadataBuilder)
    assert "_refresh_tiling" in vars(AscendTurboQuantMetadataBuilder)


# ---------------------------------------------------------------------------
# C1-C5 -- the capture contract for the static buffer
#
# G10 proves the launch path is clean. These prove the thing it launches over is
# correct: that the buffer the host fills before a replay and the buffer the
# replayed kernel reads are the same buffer, at the same address, filled for the
# same extent.
# ---------------------------------------------------------------------------


class _CopyWatcher(TorchFunctionMode):
    """Records every ``copy_`` and whether it was asked to be asynchronous."""

    def __init__(self) -> None:
        self.copies: list[bool] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        if getattr(func, "__name__", "") == "copy_":
            non_blocking = kwargs.get("non_blocking")
            if non_blocking is None and len(args) > 2:
                non_blocking = args[2]
            self.copies.append(bool(non_blocking))
        return func(*args, **kwargs)


def test_c1_the_launch_extent_is_the_one_the_host_filled(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The op's extent follows ``turboquant_num_tokens``, not ``seq_lens_list``.

    This is the gate for the hazard the field exists to close. The descriptor
    array is indexed by token and the dispatch records sit at
    ``decode_dispatch_word_base(tokens, splits)`` -- an offset that moves with
    the token count -- so a launch whose ``tokens`` exceeds the extent the host
    filled reads descriptors out of the dispatch array and dispatch records out
    of words nobody wrote, and the buffer is large enough that no size check
    fires. Here the two quantities are deliberately pulled apart: the builder
    published 2 and a stale list says 5.
    """
    entry = _Recorder(result=None)
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    adapter = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset({TurboQuantKernelStage.DECODE}),
            value_layout=TurboQuantValueLayout.CHANNEL_MAJOR,
        )
    )
    impl = _impl(adapter, emulator, _layer(emulator))
    metadata = _metadata(5)
    metadata.turboquant_num_tokens = 2

    impl._forward_decode(
        torch.zeros(5, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        metadata,
        torch.zeros(5, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
    )

    args = entry.args
    assert len(metadata.seq_lens_list) == 5, "the stale quantity is still on the metadata"
    for name in ("query", "block_tables", "context_lens", "output"):
        rows = args[_ARG[name]].shape[0]
        assert rows == 2, (
            f"{name} was launched over {rows} rows, but the host filled descriptors for 2. "
            "The two have to be the same number."
        )


def test_c1_the_extent_falls_back_only_when_it_was_not_published(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """A metadata object from the base builder still launches, over the list."""
    entry = _Recorder(result=None)
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    adapter = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset({TurboQuantKernelStage.DECODE}),
            value_layout=TurboQuantValueLayout.CHANNEL_MAJOR,
        )
    )
    impl = _impl(adapter, emulator, _layer(emulator))
    metadata = _metadata(BATCH)
    del metadata.turboquant_num_tokens

    impl._forward_decode(
        torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        metadata,
        torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
    )
    assert entry.args[_ARG["query"]].shape[0] == BATCH


def test_c2_every_builder_owned_buffer_keeps_one_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three steps of different widths and lengths, one address each.

    This is the property that makes a captured graph replayable at all: the
    replay re-executes device work against the pointers it recorded, so a buffer
    that moved between steps would be read at its old address. The prefix views
    must also start at word zero, because the op's extent checks are ``>=`` or
    exact against ``numel`` and would not notice a shifted base.
    """
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_OP_BUILD_DECODE_TILING

    _register(monkeypatch, **{TURBOQUANT_OP_BUILD_DECODE_TILING: _Recorder()})
    builder = _builder()
    base = {
        "turboquant_context_lens": builder._context_lens.data_ptr(),
        "turboquant_tiling": builder._tiling.data_ptr(),
        "turboquant_workspace": builder._workspace.data_ptr(),
        "turboquant_query_rot": builder._query_rot.data_ptr(),
    }

    for width, length in ((2, 64), (5, 4096), (1, 128)):
        extra = builder._build_backend_metadata(
            types.SimpleNamespace(num_reqs=width),
            block_table=torch.zeros(width, MAX_BLOCKS, dtype=torch.int32),
            query_lens=torch.ones(width, dtype=torch.int32),
            seq_lens=torch.full((width,), length, dtype=torch.int32),
            num_decodes=width,
            num_prefills=0,
        )
        assert extra["turboquant_num_tokens"] == width
        for name, address in base.items():
            view = extra[name]
            assert view.data_ptr() == address, (
                f"{name} moved on the step of width {width}. A captured replay reads the "
                "address it recorded, so a buffer that moves is read at its old one."
            )
            assert view.storage_offset() == 0, f"{name} is not a prefix of its buffer"


def test_c3_the_launch_path_never_fills_the_tiling_buffer(
    decode: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fill stays in the builder, which runs outside any captured region.

    The builder op takes a **CPU** ``context_lens``, does host arithmetic and
    ends in a host-to-device copy. All three are legal where it runs -- the
    metadata build, ahead of the graph -- and none is legal inside a capture. So
    the one thing that must never happen is a launch that "helpfully" fills its
    own descriptors when it finds them stale, which is why both ops are
    registered here and only one of them may be called.
    """
    from vllm_ascend.attention.turboquant_dry_run import (
        TURBOQUANT_KERNEL_OPS,
        TURBOQUANT_OP_BUILD_DECODE_TILING,
        TURBOQUANT_OP_FUSED_DECODE,
    )

    builder_entry = _Recorder()
    _register(
        monkeypatch,
        **{
            TURBOQUANT_OP_FUSED_DECODE: decode["entry"],
            TURBOQUANT_OP_BUILD_DECODE_TILING: builder_entry,
        },
    )
    decode["impl"]._forward_decode(decode["query"], decode["metadata"], decode["output"])

    assert decode["entry"].calls, "the decode did not launch at all"
    assert builder_entry.calls == [], (
        "the decode launch called the tiling builder. That op takes a CPU tensor, does host "
        "arithmetic and ends in an H2D copy -- it belongs in the metadata build, not in a "
        "launch that has to be legal on a capturing stream."
    )
    assert TURBOQUANT_OP_BUILD_DECODE_TILING not in TURBOQUANT_KERNEL_OPS, (
        "the tiling builder is not a kernel stage; listing it as one would put it behind the "
        "same gate the launch is behind and invite exactly that call."
    )


def test_c4_a_multi_token_decode_step_is_refused() -> None:
    """Speculative and MTP steps, which this builder cannot tile.

    The runner pads tokens to ``num_reqs_padded * uniform_decode_query_len``
    (``model_runner_v1.py:1095``), so such a step carries several query rows per
    request while ``seq_lens`` still carries one length per request. Tiling it
    needs the lengths expanded per accepted token. Refusing is the point: the
    alternative is a launch whose extra rows read descriptors nobody wrote.
    """
    builder = _builder()
    with pytest.raises(RuntimeError, match="multi-token decode step"):
        builder._build_backend_metadata(
            types.SimpleNamespace(num_reqs=2, decode_token_per_req=2),
            block_table=torch.zeros(2, MAX_BLOCKS, dtype=torch.int32),
            query_lens=torch.full((2,), 2, dtype=torch.int32),
            seq_lens=torch.tensor([64, 128], dtype=torch.int32),
            num_decodes=2,
            num_prefills=0,
        )


@pytest.mark.parametrize("per_req", [None, 1])
def test_c4_an_ordinary_decode_step_is_not_refused(per_req: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """One token per request, however the field is spelled."""
    from vllm_ascend.attention.turboquant_dry_run import TURBOQUANT_OP_BUILD_DECODE_TILING

    _register(monkeypatch, **{TURBOQUANT_OP_BUILD_DECODE_TILING: _Recorder()})
    common = types.SimpleNamespace(num_reqs=2)
    if per_req is not None:
        common.decode_token_per_req = per_req
    extra = _builder()._build_backend_metadata(
        common,
        block_table=torch.zeros(2, MAX_BLOCKS, dtype=torch.int32),
        query_lens=torch.ones(2, dtype=torch.int32),
        seq_lens=torch.tensor([64, 128], dtype=torch.int32),
        num_decodes=2,
        num_prefills=0,
    )
    assert extra["turboquant_num_tokens"] == 2


def test_c5_the_staging_is_a_ring_so_a_step_cannot_tear_the_previous_upload() -> None:
    """Consecutive steps stage into different buffers; the third reuses the first.

    The upload is asynchronous, so it completes after the builder has returned
    and the host has moved on to the next step -- which, with async scheduling,
    overlaps this step's execution by design. A single staging buffer would be
    rewritten under its own in-flight copy. Depth two is sufficient because the
    whole of step N+1 is enqueued behind step N's copy on an in-order stream.
    """
    builder = _builder()
    common = types.SimpleNamespace(num_reqs=2)
    assert TURBOQUANT_STAGING_SLOTS >= 2, (
        "a single staging buffer is rewritten by the next step while its own asynchronous "
        "upload is still in flight, which is a torn upload rather than a stale one."
    )
    slots = [
        builder._refresh_context_lens(common, torch.tensor([64, 128], dtype=torch.int32))[1]
        for _ in range(TURBOQUANT_STAGING_SLOTS + 1)
    ]
    assert len({slot.data_ptr() for slot in slots[:-1]}) == TURBOQUANT_STAGING_SLOTS, (
        "two consecutive steps shared one staging buffer"
    )
    assert slots[0].data_ptr() == slots[-1].data_ptr(), "the ring is deeper than it claims to be"


def test_c5_the_upload_is_asynchronous_only_out_of_pinned_staging() -> None:
    """A non-blocking copy out of pageable memory is neither fast nor safe.

    The runtime bounces it through a staging buffer of its own while the caller
    believes the source is free to reuse, which is the one combination that
    gives up both properties. So the flag follows the pinning, and a host with
    no pinned allocator gets a blocking copy -- correct, just slower.
    """
    common = types.SimpleNamespace(num_reqs=2)
    seq_lens = torch.tensor([64, 128], dtype=torch.int32)

    for pinned in (True, False):
        builder = _builder()
        builder._staging_pinned = pinned
        watcher = _CopyWatcher()
        with watcher:
            builder._refresh_context_lens(common, seq_lens)
        assert watcher.copies, "no host-to-device copy was issued at all"
        assert all(flag is pinned for flag in watcher.copies), (
            f"staging pinned={pinned} but the copies were issued non_blocking={watcher.copies}"
        )


def test_c5_the_real_staging_is_pinned_when_the_host_can_pin() -> None:
    """The allocation itself, which no device-free fixture can run.

    ``__init__`` needs a ``VllmConfig`` and a device, so this reads the source
    instead: it must pin, and it must record whether pinning succeeded, because
    the copy's blocking-ness is derived from that rather than assumed.
    """
    import inspect

    from vllm_ascend.attention.turboquant_dry_run import AscendTurboQuantMetadataBuilder

    source = inspect.getsource(AscendTurboQuantMetadataBuilder.__init__)
    assert "pin_memory()" in source
    assert "_staging_pinned = True" in source
    assert "_staging_pinned = False" in source, "a host with no pinned allocator must still work"


# ---------------------------------------------------------------------------
# C6 -- the TorchFunctionMode blind spot
#
# G10 is a ``TorchFunctionMode`` allowlist, so it sees torch functions and
# Tensor methods and nothing else. A Python builtin on a non-tensor container is
# invisible to it *by construction*: ``len(attn_metadata.seq_lens_list)`` reads
# host-side state the engine builds every step, and G10 recorded nothing,
# because a list is not a tensor. C1 found that one by reasoning about the
# descriptor layout rather than by any gate going red.
#
# So this is the gate for the whole class rather than for that instance. Every
# host-side field of ``AscendMetadata`` is replaced with an object that records
# any Python-level access to itself -- ``len``, subscript, iteration, truth,
# attribute -- and the fused launch is run. The field names in the failure
# message are the leak.
#
# Why these four fields. ``seq_lens`` and ``seq_lens_cpu`` are the CPU mirror
# (``_select_seq_lens`` prefers ``_seq_lens_cpu``), ``seq_lens_list`` and
# ``actual_seq_lengths_q`` are Python lists built per step with ``.tolist()``.
# Reading any of them inside a launch means the launch depends on host state a
# captured graph will not re-read. The device fields (``seq_lens_gpu``,
# ``block_tables``, ``turboquant_*``) are deliberately NOT poisoned: reading
# those is the whole point.
# ---------------------------------------------------------------------------


class _HostFieldPoison:
    """An ``AscendMetadata`` field that records every Python-level access."""

    def __init__(self, name: str, log: list[str]) -> None:
        # Bypass __setattr__-free design: plain attributes are fine, but they
        # must not be routed through __getattr__, so they are set directly.
        object.__setattr__(self, "_poison_name", name)
        object.__setattr__(self, "_poison_log", log)

    def _hit(self, how: str) -> None:
        object.__getattribute__(self, "_poison_log").append(f"{object.__getattribute__(self, '_poison_name')}.{how}")

    def __len__(self) -> int:
        self._hit("__len__")
        return 2

    def __getitem__(self, item: Any) -> int:
        self._hit("__getitem__")
        return 64

    def __iter__(self) -> Any:
        self._hit("__iter__")
        return iter([64, 64])

    def __bool__(self) -> bool:
        self._hit("__bool__")
        return True

    def __getattr__(self, attr: str) -> Any:
        # Everything real is reached through object.__getattribute__ above, so
        # anything arriving here is the launch asking this field for something.
        self._hit(attr)
        raise AttributeError(attr)


# The host-side fields of AscendMetadata, by name
# (``attention_v1.py:189-192``). None of these may be read inside a launch.
_HOST_FIELDS = ("seq_lens", "seq_lens_cpu", "seq_lens_list", "actual_seq_lengths_q")


def test_c6_the_launch_reads_no_host_side_metadata_field(decode: dict[str, Any]) -> None:
    """The class of leak ``TorchFunctionMode`` cannot see.

    A revert of C1 -- ``batch_size = len(attn_metadata.seq_lens_list)`` -- turns
    this red with ``seq_lens_list.__len__``, which is the name of the field and
    the operation. G10 stays green through that same revert, which is why this
    gate exists separately.
    """
    log: list[str] = []
    metadata = decode["metadata"]
    for field in _HOST_FIELDS:
        setattr(metadata, field, _HostFieldPoison(field, log))

    decode["impl"]._forward_decode(decode["query"], metadata, decode["output"])

    assert decode["entry"].calls, "the decode did not launch, so this gate proved nothing"
    assert log == [], (
        f"the decode launch read host-side metadata: {log}. Every one of these is state the "
        "engine rebuilds per step and a captured graph never re-reads, and none of them is "
        "visible to the G10 allowlist because none is a tensor."
    )


def test_c6_the_poison_is_live(decode: dict[str, Any]) -> None:
    """The gate's own instrument, so a silent no-op cannot pass for a clean run.

    Without this, a poison that stopped recording -- or a fixture that stopped
    installing it -- would read exactly like a clean launch path.
    """
    log: list[str] = []
    poison = _HostFieldPoison("seq_lens_list", log)
    assert len(poison) == 2
    assert poison[0] == 64
    # ``iter`` rather than ``list``, which also probes ``__len__`` as a size hint
    # and would make the expected log depend on that implementation detail.
    assert next(iter(poison)) == 64
    probed = "tolist"  # via a name, so ruff does not rewrite it to attribute access
    with pytest.raises(AttributeError):
        getattr(poison, probed)
    assert log == [
        "seq_lens_list.__len__",
        "seq_lens_list.__getitem__",
        "seq_lens_list.__iter__",
        "seq_lens_list.tolist",
    ]


def test_c6_the_launch_path_uses_no_stream_api() -> None:
    """The other blind spot: ``torch.npu`` module functions are not dispatched.

    ``torch.npu.synchronize()``, ``current_stream()`` and
    ``is_current_stream_capturing()`` do not go through ``__torch_function__``,
    so G10 would not see one either. The launch path must name none of them;
    the profiler and the FIA fallback may, and do.
    """
    import inspect

    from vllm_ascend.attention.turboquant_dry_run import (
        AscendTurboQuantDryRunBackendImpl,
        TurboQuantOpsAdapter,
    )

    launch_path = (
        AscendTurboQuantDryRunBackendImpl._forward_decode,
        AscendTurboQuantDryRunBackendImpl._decode_num_tokens,
        AscendTurboQuantDryRunBackendImpl._decode_context_lens,
        AscendTurboQuantDryRunBackendImpl._decode_buffers,
        TurboQuantOpsAdapter.fused_decode,
        TurboQuantOpsAdapter._kernel_fused_decode,
        TurboQuantOpsAdapter._resolve_kernel,
        TurboQuantOpsAdapter._has_kernel,
    )
    forbidden = ("torch.npu.", "torch_npu.", "synchronize(", "current_stream(", "record_stream(")
    for function in launch_path:
        source = inspect.getsource(function)
        # The docstrings explain why these are absent, so only code lines count.
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"', "'")))
        for name in forbidden:
            assert name not in code, (
                f"{function.__qualname__} names {name!r}. Stream and device-module calls are "
                "not dispatched through __torch_function__, so G10 cannot see them."
            )
