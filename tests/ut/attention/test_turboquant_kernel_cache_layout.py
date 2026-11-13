# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the cache layout the TurboQuant NPU writers leave behind.

The two audited writers (``csrc/attention/turboquant/op_kernel/turboquant_reshape_batch.cpp``
and ``..._scatter.cpp``) do not write the golden model's cache. Both are built
with ``TURBOQUANT_INGRESS_NZ=1`` and produce the NZ-tiled layout the fused
decode reads; the strip cache -- page axis
``(num_blocks, num_kv_heads, block_size, head_size / 2)``, head before token,
value nibble ``channel * block_size + token`` inside a 16-token strip -- is the
pre-ingress format, kept readable for caches written before the writers crossed
over.

So three things have to hold, and all three are checked here without a device:

* the layout is a property of whichever writer owns the page, not a global, so
  switching an ingestion stage on is enough -- the reader follows. Either NPU
  writer's stage means NZ_TILED.
* a mixed configuration (NPU writer, golden gather) reads back correctly. The
  reference strip cache below is written in PyTorch to the pre-ingress writers'
  documented nibble arithmetic, and gathering it must produce the same four
  panels as gathering the golden model's own cache.
* ``TURBOQUANT_KERNEL_STAGES=none`` falls all the way back: no registered op is
  called and the golden model writes and reads its own layout.
"""

import types
from typing import Any

import pytest
import torch

from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_VALUE_PLANE,
    turboquant_packed_bytes,
    turboquant_scale_cache_shape,
)
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_NAMESPACE_ENV,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_STAGES_ENV,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantEmulator,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
)

# The strip cache's block_size is 16: it is kTokens in the batch writer, a
# compile-time constant of its four-stage schedule, and a prefill strip fills
# one whole physical page.
BLOCK_SIZE = 16
HEAD_SIZE = 64
NUM_KV_HEADS = 2
NUM_BLOCKS = 4
NAMESPACE = "turboquant_ops"
CPU = torch.device("cpu")

INGESTION_OPS = (TURBOQUANT_OP_RESHAPE_AND_CACHE, TURBOQUANT_OP_RESHAPE_SCATTER)
ALL_OPS = (*INGESTION_OPS, TURBOQUANT_OP_FUSED_DECODE)


class _Recorder:
    """A registered entry point that records its call and writes nothing."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        return None


class _OpNamespace:
    def __init__(self, entries: dict[str, _Recorder]) -> None:
        self._entries = entries

    def __getattr__(self, name: str) -> _Recorder:
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(name) from None


def _register(monkeypatch: pytest.MonkeyPatch, *op_names: str) -> dict[str, _Recorder]:
    entries = {name: _Recorder() for name in op_names}
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(**{NAMESPACE: _OpNamespace(entries)}))
    return entries


def _adapter(
    *stages: TurboQuantKernelStage,
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.TOKEN_MAJOR,
) -> TurboQuantOpsAdapter:
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(ops_namespace=NAMESPACE, kernel_stages=frozenset(stages), value_layout=value_layout)
    )


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


def _caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(turboquant_scale_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS), dtype=torch.float32),
    )


def _activations(num_tokens: int, seed: int = 20261107) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(shape, generator=generator, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# The reference strip writer: what the NPU writers leave in the cache
# ---------------------------------------------------------------------------


def _write_strip_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    emulator: TurboQuantEmulator,
) -> None:
    """Write the audited writers' strip layout, in PyTorch, for a test oracle.

    This is deliberately a transcription of the two kernels' documented nibble
    arithmetic rather than a call into the adapter: it is the independent side
    of the differential, so it must not share code with the reader it checks.

    Per ``(block, head)`` page, with ``S`` the block size and ``D`` the head
    size: the key plane holds nibble ``token * D + channel`` and the value plane
    nibble ``channel * S + token``, two nibbles per byte, low nibble first. The
    page axis order is ``(num_blocks, num_kv_heads, ...)``, head before token.
    The scale plane is ``(2, num_blocks, num_kv_heads, S)`` in both layouts.
    """
    num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * TURBOQUANT_PACK_FACTOR

    key_codes, key_scales = emulator.quantize(emulator.rotate(key))
    value_codes, value_scales = emulator.quantize(emulator.rotate(value))

    # The key plane: token-major rows, so one token's packed bytes go straight in.
    key_pages = key_cache.view(num_blocks, num_kv_heads, block_size, packed_bytes)
    packed_key = emulator.pack(key_codes)

    # The value plane is channel-major INSIDE a 16-token strip, so a page's
    # nibbles are [strip, channel, token % 16]. Building that array and packing
    # along its last axis is the writers' formula, term for term.
    strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS
    value_nibbles = torch.zeros(
        (num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS), dtype=torch.int8
    )

    for token, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, block_size)
        strip, lane = divmod(offset, TURBOQUANT_STRIP_TOKENS)
        for head in range(num_kv_heads):
            key_pages[page, head, offset] = packed_key[token, head]
            value_nibbles[page, head, strip, :, lane] = value_codes[token, head]
            scale_cache[TURBOQUANT_KEY_PLANE, page, head, offset] = key_scales[token, head]
            scale_cache[TURBOQUANT_VALUE_PLANE, page, head, offset] = value_scales[token, head]

    value_cache.view(
        num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
    )[:] = emulator.pack(value_nibbles)


# ---------------------------------------------------------------------------
# The layout is a property of the writer
# ---------------------------------------------------------------------------


def test_golden_writers_keep_the_configured_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    _register(monkeypatch, *ALL_OPS)
    assert _adapter().value_layout() is TurboQuantValueLayout.TOKEN_MAJOR
    assert (
        _adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR).value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR
    )


@pytest.mark.parametrize(
    ("stage", "op_name", "expected"),
    [
        (TurboQuantKernelStage.RESHAPE_AND_CACHE, TURBOQUANT_OP_RESHAPE_AND_CACHE, TurboQuantValueLayout.NZ_TILED),
        (TurboQuantKernelStage.SCATTER, TURBOQUANT_OP_RESHAPE_SCATTER, TurboQuantValueLayout.NZ_TILED),
    ],
)
def test_an_engaged_writer_owns_the_cache_layout(
    monkeypatch: pytest.MonkeyPatch, stage: TurboQuantKernelStage, op_name: str, expected: TurboQuantValueLayout
) -> None:
    """One ingestion stage is enough: the cache is that writer's layout from then on.

    Both writers are the NZ ingress now, so either stage puts the cache in
    NZ_TILED; the strip layout is reachable only as an explicit golden config,
    for caches written before the ingress existed.
    """
    _register(monkeypatch, op_name)
    assert _adapter(stage).value_layout() is expected


def test_the_nz_layout_survives_both_writers_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both writers staged: one cache, one layout -- NZ on both sides.

    The two kernels write the same NZ-tiled cache the fused decode reads, so
    there is no writer conflict left to arbitrate; this pins that the adapter
    reports NZ_TILED and neither stage pulls it anywhere else.
    """
    _register(monkeypatch, TURBOQUANT_OP_RESHAPE_AND_CACHE, TURBOQUANT_OP_RESHAPE_SCATTER)
    adapter = _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.SCATTER)
    assert adapter.value_layout() is TurboQuantValueLayout.NZ_TILED


def test_the_decode_stage_alone_does_not_change_the_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused decode reads; it never writes a page, so it owns no layout."""
    _register(monkeypatch, TURBOQUANT_OP_FUSED_DECODE)
    assert _adapter(TurboQuantKernelStage.DECODE).value_layout() is TurboQuantValueLayout.TOKEN_MAJOR


def test_an_unregistered_writer_leaves_the_golden_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage switched on with no op behind it falls back, layout included.

    This is the case a build without the extension takes, and getting it wrong
    would have the golden reader de-interleave a cache the golden writer wrote.
    """
    _register(monkeypatch)
    adapter = _adapter(*TurboQuantKernelStage)
    assert adapter.value_layout() is TurboQuantValueLayout.TOKEN_MAJOR


# ---------------------------------------------------------------------------
# The refusal fires exactly when the golden model owns the page
# ---------------------------------------------------------------------------


def test_the_golden_writer_now_does_the_channel_major_rmw(emulator: TurboQuantEmulator) -> None:
    """It used to refuse this layout; it writes it now, one token at a time.

    The read-modify-write is what makes a single-token decode write safe in a
    plane where token t shares a byte with t ^ 1. tests/ut/attention/
    test_turboquant_strip_cache.py pins it against the kernels' nibble formula;
    this only checks that the refusal is gone and that one token lands.
    """
    adapter = _adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
    key, value = _activations(1)
    caches = _caches()
    adapter.reshape_scatter_decode(key, value, *caches, torch.zeros(1, dtype=torch.int32), emulator)
    assert not torch.equal(caches[TURBOQUANT_VALUE_PLANE], torch.zeros_like(caches[TURBOQUANT_VALUE_PLANE]))


@pytest.mark.parametrize(
    ("stage", "op_name", "entry", "expected"),
    [
        (
            TurboQuantKernelStage.RESHAPE_AND_CACHE,
            TURBOQUANT_OP_RESHAPE_AND_CACHE,
            "reshape_and_cache_prefill",
            TurboQuantValueLayout.NZ_TILED,
        ),
        (
            TurboQuantKernelStage.SCATTER,
            TURBOQUANT_OP_RESHAPE_SCATTER,
            "reshape_scatter_decode",
            TurboQuantValueLayout.NZ_TILED,
        ),
    ],
)
def test_an_engaged_writer_never_reaches_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
    emulator: TurboQuantEmulator,
    stage: TurboQuantKernelStage,
    op_name: str,
    entry: str,
    expected: TurboQuantValueLayout,
) -> None:
    """The kernel owns the cache layout, so the layout must not refuse the call.

    `value_layout()` reports the staged writer's layout -- NZ_TILED for both
    writers now -- and that must not make the entry point raise on its way to
    the kernel it is about to call.
    """
    entries = _register(monkeypatch, op_name)
    adapter = _adapter(stage)
    assert adapter.value_layout() is expected

    key, value = _activations(BLOCK_SIZE)
    getattr(adapter, entry)(key, value, *_caches(), torch.arange(BLOCK_SIZE, dtype=torch.int32), emulator)
    assert entries[op_name].calls == 1


# ---------------------------------------------------------------------------
# A mixed configuration: NPU writer, golden gather
# ---------------------------------------------------------------------------


def test_the_golden_gather_reads_a_strip_cache_back(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """Gathering a kernel-written cache must give the same panels as a golden one.

    Both sides ingest the same activations. One writes the golden token-major
    cache and gathers it; the other writes the strip cache the pre-ingress
    writers produced and gathers it with the strip layout pinned in the config,
    which is how a cache written before the NZ ingress existed is read. The four
    panels must agree bit for bit -- the codec is the same, only the byte order
    the reader undoes differs.
    """
    key, value = _activations(BLOCK_SIZE * 2)
    slot_mapping = torch.arange(BLOCK_SIZE * 2, dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)

    golden_caches = _caches()
    _adapter().reshape_and_cache_prefill(key, value, *golden_caches, slot_mapping, emulator)
    expected = _adapter().gather(*golden_caches, block_tables)

    strip_caches = _caches()
    _write_strip_cache(key, value, *strip_caches, slot_mapping, emulator)
    entries = _register(monkeypatch, TURBOQUANT_OP_RESHAPE_SCATTER)
    # No stage: both NPU writers are NZ now, so staging one would report
    # NZ_TILED. The strip layout is the config's to pin.
    reader = _adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
    assert reader.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR
    actual = reader.gather(*strip_caches, block_tables)

    names = ("packed_key", "key_scales", "packed_value", "value_scales")
    for name, want, got in zip(names, expected, actual):
        assert got.shape == want.shape, name
        assert torch.equal(got, want), name
    # The gather is a read: it must not have launched the writer.
    assert entries[TURBOQUANT_OP_RESHAPE_SCATTER].calls == 0


def test_the_strip_reader_round_trips_to_the_activations(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The whole mixed read path: strip cache -> dequantise -> the model's basis."""
    key, value = _activations(BLOCK_SIZE)
    strip_caches = _caches()
    _write_strip_cache(key, value, *strip_caches, torch.arange(BLOCK_SIZE, dtype=torch.int32), emulator)

    _register(monkeypatch, TURBOQUANT_OP_RESHAPE_SCATTER)
    reader = _adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
    packed_key, key_scales, packed_value, value_scales = reader.gather(
        *strip_caches, torch.zeros(1, 1, dtype=torch.int32)
    )

    assert torch.equal(
        reader.dequantize_cache(packed_key, key_scales, emulator, torch.float32)[0],
        emulator.decode(*emulator.encode(key)),
    )
    assert torch.equal(
        reader.dequantize_cache(packed_value, value_scales, emulator, torch.float32)[0],
        emulator.decode(*emulator.encode(value)),
    )


def test_the_strip_reader_is_not_the_identity(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> None:
    """A guard on the test above: the two caches really do hold different bytes.

    Without this, a reader that silently ignored the layout would pass every
    assertion in this file.
    """
    key, value = _activations(BLOCK_SIZE)
    slot_mapping = torch.arange(BLOCK_SIZE, dtype=torch.int32)

    golden_caches = _caches()
    _adapter().reshape_and_cache_prefill(key, value, *golden_caches, slot_mapping, emulator)
    strip_caches = _caches()
    _write_strip_cache(key, value, *strip_caches, slot_mapping, emulator)

    assert not torch.equal(golden_caches[TURBOQUANT_VALUE_PLANE], strip_caches[TURBOQUANT_VALUE_PLANE])
    # The scale plane is the one thing both layouts index identically.
    assert torch.equal(golden_caches[2], strip_caches[2])


# ---------------------------------------------------------------------------
# TURBOQUANT_KERNEL_STAGES still selects, and `none` falls all the way back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("none", ()),
        ("", ()),
        ("reshape_and_cache", (TURBOQUANT_OP_RESHAPE_AND_CACHE,)),
        ("scatter", (TURBOQUANT_OP_RESHAPE_SCATTER,)),
        ("decode", (TURBOQUANT_OP_FUSED_DECODE,)),
        ("reshape_and_cache,scatter", INGESTION_OPS),
        ("all", ALL_OPS),
    ],
)
def test_the_stage_env_selects_exactly_those_ops(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setenv(TURBOQUANT_STAGES_ENV, raw)
    monkeypatch.setenv(TURBOQUANT_NAMESPACE_ENV, NAMESPACE)
    _register(monkeypatch, *ALL_OPS)
    adapter = TurboQuantOpsAdapter()
    for op_name in ALL_OPS:
        assert adapter._has_kernel(op_name) is (op_name in expected), op_name


def test_an_unknown_stage_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TURBOQUANT_STAGES_ENV, "reshape_and_cache,nonsense")
    with pytest.raises(ValueError, match="unknown stage"):
        TurboQuantOpsConfig.from_env()


def test_stages_none_runs_the_whole_path_golden(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> None:
    """Every op registered, every stage off: nothing launches and the numbers hold.

    This is the no-recompilation fallback guarantee. The ops exist -- as they do
    in an ascend950 wheel -- and TURBOQUANT_KERNEL_STAGES=none still routes
    ingestion, the gather, the dequantisation and the decode through PyTorch.
    """
    monkeypatch.setenv(TURBOQUANT_STAGES_ENV, "none")
    monkeypatch.setenv(TURBOQUANT_NAMESPACE_ENV, NAMESPACE)
    entries = _register(monkeypatch, *ALL_OPS)
    adapter = TurboQuantOpsAdapter()

    key, value = _activations(BLOCK_SIZE)
    caches = _caches()
    slot_mapping = torch.arange(BLOCK_SIZE, dtype=torch.int32)

    adapter.reshape_and_cache_prefill(key, value, *caches, slot_mapping, emulator)
    adapter.reshape_scatter_decode(key, value, *caches, slot_mapping, emulator)
    packed_key, key_scales, _, _ = adapter.gather(*caches, torch.zeros(1, 1, dtype=torch.int32))
    dense = adapter.dequantize_cache(packed_key, key_scales, emulator, torch.float32)
    fused = adapter.fused_decode(
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
        # Empty: the stage gate declines before the launch buffers are read.
        tiling=torch.empty(0, dtype=torch.int32),
        workspace=torch.empty(0, dtype=torch.float32),
        query_rot=torch.empty(0, dtype=torch.float32),
    )

    assert fused is None, "the decode must fall through to the golden attention"
    assert adapter.value_layout() is TurboQuantValueLayout.TOKEN_MAJOR
    assert all(entry.calls == 0 for entry in entries.values()), "no op may launch with stages=none"
    assert torch.equal(dense[0], emulator.decode(*emulator.encode(key)))
