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

import pytest
import torch

from tests.helpers.turboquant import (
    CACHE_POISON_SCALE,
    OpRecorder,
    assert_cache_equal,
    assert_poison_scales_preserved,
    make_activations,
    make_adapter,
    make_cache,
    make_finite_spike_cache,
    make_recycled_page_table,
    register_ops,
    slots_from_page_table,
)
from vllm_ascend.attention import turboquant_codec
from vllm_ascend.attention.turboquant_codec import TurboQuantEmulator
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_NAMESPACE_ENV,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_STAGES_ENV,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantKernelStage,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
)
from vllm_ascend.attention.turboquant_ingestion import turboquant_plan_strip_ingestion
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_VALUE_PLANE,
)
from vllm_ascend.attention.turboquant_ops import TurboQuantOpsAdapter

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


def _register(monkeypatch: pytest.MonkeyPatch, *op_names: str) -> dict[str, OpRecorder]:
    entries = {name: OpRecorder() for name in op_names}
    register_ops(monkeypatch, **entries)
    return entries


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


def _caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return make_cache(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)


def _activations(num_tokens: int, seed: int = 20261107) -> tuple[torch.Tensor, torch.Tensor]:
    return make_activations(num_tokens, NUM_KV_HEADS, HEAD_SIZE, seed=seed)


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
    value_codes, value_scales = emulator.quantize(emulator.rotate(value), fold="value")

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


def test_reference_writers_keep_the_configured_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    _register(monkeypatch, *ALL_OPS)
    assert make_adapter().value_layout() is TurboQuantValueLayout.TOKEN_MAJOR
    assert (
        make_adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR).value_layout()
        is TurboQuantValueLayout.CHANNEL_MAJOR
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
    assert make_adapter(stage).value_layout() is expected


def test_the_nz_layout_survives_both_writers_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both writers staged: one cache, one layout -- NZ on both sides.

    The two kernels write the same NZ-tiled cache the fused decode reads, so
    there is no writer conflict left to arbitrate; this pins that the adapter
    reports NZ_TILED and neither stage pulls it anywhere else.
    """
    _register(monkeypatch, TURBOQUANT_OP_RESHAPE_AND_CACHE, TURBOQUANT_OP_RESHAPE_SCATTER)
    adapter = make_adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.SCATTER)
    assert adapter.value_layout() is TurboQuantValueLayout.NZ_TILED


def test_the_decode_stage_alone_does_not_change_the_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused decode reads; it never writes a page, so it owns no layout."""
    _register(monkeypatch, TURBOQUANT_OP_FUSED_DECODE)
    assert make_adapter(TurboQuantKernelStage.DECODE).value_layout() is TurboQuantValueLayout.TOKEN_MAJOR


def test_an_unregistered_writer_leaves_the_reference_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage switched on with no op behind it falls back, layout included.

    This is the case a build without the extension takes, and getting it wrong
    would have the golden reader de-interleave a cache the golden writer wrote.
    """
    _register(monkeypatch)
    adapter = make_adapter(*TurboQuantKernelStage)
    assert adapter.value_layout() is TurboQuantValueLayout.TOKEN_MAJOR


# ---------------------------------------------------------------------------
# The refusal fires exactly when the golden model owns the page
# ---------------------------------------------------------------------------


def test_the_reference_writer_now_does_the_channel_major_rmw(emulator: TurboQuantEmulator) -> None:
    """It used to refuse this layout; it writes it now, one token at a time.

    The read-modify-write is what makes a single-token decode write safe in a
    plane where token t shares a byte with t ^ 1. tests/ut/attention/
    test_turboquant_strip_cache.py pins it against the kernels' nibble formula;
    this only checks that the refusal is gone and that one token lands.
    """
    adapter = make_adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
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
    adapter = make_adapter(stage)
    assert adapter.value_layout() is expected

    key, value = _activations(BLOCK_SIZE)
    getattr(adapter, entry)(key, value, *_caches(), torch.arange(BLOCK_SIZE, dtype=torch.int32), emulator)
    assert len(entries[op_name].calls) == 1


# ---------------------------------------------------------------------------
# A mixed configuration: NPU writer, golden gather
# ---------------------------------------------------------------------------


def test_the_reference_gather_reads_a_strip_cache_back(
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
    make_adapter().reshape_and_cache_prefill(key, value, *golden_caches, slot_mapping, emulator)
    expected = make_adapter().gather(*golden_caches, block_tables)

    strip_caches = _caches()
    _write_strip_cache(key, value, *strip_caches, slot_mapping, emulator)
    entries = _register(monkeypatch, TURBOQUANT_OP_RESHAPE_SCATTER)
    # No stage: both NPU writers are NZ now, so staging one would report
    # NZ_TILED. The strip layout is the config's to pin.
    reader = make_adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
    assert reader.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR
    actual = reader.gather(*strip_caches, block_tables)

    names = ("packed_key", "key_scales", "packed_value", "value_scales")
    for name, want, got in zip(names, expected, actual):
        assert got.shape == want.shape, name
        assert torch.equal(got, want), name
    # The gather is a read: it must not have launched the writer.
    assert not entries[TURBOQUANT_OP_RESHAPE_SCATTER].calls


def test_the_strip_reader_round_trips_to_the_activations(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The whole mixed read path: strip cache -> dequantise -> the model's basis."""
    key, value = _activations(BLOCK_SIZE)
    strip_caches = _caches()
    _write_strip_cache(key, value, *strip_caches, torch.arange(BLOCK_SIZE, dtype=torch.int32), emulator)

    _register(monkeypatch, TURBOQUANT_OP_RESHAPE_SCATTER)
    reader = make_adapter(value_layout=TurboQuantValueLayout.CHANNEL_MAJOR)
    packed_key, key_scales, packed_value, value_scales = reader.gather(
        *strip_caches, torch.zeros(1, 1, dtype=torch.int32)
    )

    assert torch.equal(
        reader.dequantize_cache(packed_key, key_scales, emulator, torch.float32)[0],
        emulator.decode(*emulator.encode(key)),
    )
    assert torch.equal(
        reader.dequantize_cache(packed_value, value_scales, emulator, torch.float32)[0],
        emulator.decode(*emulator.encode(value, fold="value")),
    )


def test_the_strip_reader_is_not_the_identity(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> None:
    """A guard on the test above: the two caches really do hold different bytes.

    Without this, a reader that silently ignored the layout would pass every
    assertion in this file.
    """
    key, value = _activations(BLOCK_SIZE)
    slot_mapping = torch.arange(BLOCK_SIZE, dtype=torch.int32)

    golden_caches = _caches()
    make_adapter().reshape_and_cache_prefill(key, value, *golden_caches, slot_mapping, emulator)
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
        assert adapter.has_kernel(op_name) is (op_name in expected), op_name


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
    assert all(not entry.calls for entry in entries.values()), "no op may launch with stages=none"
    assert torch.equal(dense[0], emulator.decode(*emulator.encode(key)))


@pytest.mark.parametrize(
    "plane,before,after,budget,passes",
    [
        pytest.param(2, 0x3F791A87, 0x3F791A88, 1, True, id="reported-scale-plus-one-ulp"),
        pytest.param(2, 0x3F826D29, 0x3F826D28, 1, True, id="reported-scale-minus-one-ulp"),
        pytest.param(2, 0x71800000, 0x71800001, 1, False, id="large-finite-poison-stays-exact"),
        pytest.param(2, 0x3F7FFFFF, 0x3F800000, 1, True, id="scale-exponent-boundary"),
        pytest.param(2, 0x3F791A87, 0x3F791A89, 1, False, id="two-ulps-rejected"),
        pytest.param(2, 0x3F791A87, 0x3F791A88, 0, False, id="snapshot-stays-exact"),
        pytest.param(0, 0, 1, 1, False, id="key-byte-stays-exact"),
        pytest.param(1, 0, 1, 1, False, id="value-byte-stays-exact"),
        pytest.param(2, 0x7FC00000, 0x7FC00000, 0, True, id="identical-nan-canary"),
        pytest.param(2, 0x7FC00000, 0x7FC00001, 1, False, id="nan-payload-stays-exact"),
        pytest.param(2, 0x7FC00000, 0x3F800000, 1, False, id="poison-overwritten"),
        pytest.param(2, 0x3F800000, 0x7FC00000, 1, False, id="live-scale-underfilled"),
        pytest.param(2, 0x00000000, 0x00000001, 1, False, id="zero-canary-overwritten"),
        pytest.param(2, 0x00000000, 0x80000000, 1, False, id="zero-sign-stays-exact"),
        pytest.param(2, 0x7F7FFFFF, 0x7F800000, 1, False, id="overflow-not-rounding"),
        pytest.param(2, 0xBF800000, 0xBF800001, 1, False, id="negative-scale-not-rounding"),
        pytest.param(2, 0x3F800000, 0xBF800000, 1, False, id="sign-flip-no-integer-overflow"),
    ],
)
def test_cache_comparison_limits_rounding_to_live_scales(plane, before, after, budget, passes):
    # Identical poison in another lane must neither fail comparison nor hide
    # the finite diagnostic when a live scale exceeds its budget.
    scale_bits = torch.tensor([0x7FC00000, 0, 0, before if plane == 2 else 0], dtype=torch.int64)
    expected = (
        torch.zeros(4, dtype=torch.int8),
        torch.zeros(4, dtype=torch.int8),
        scale_bits.int().view(torch.float32),
    )
    actual = tuple(tensor.clone() for tensor in expected)
    if plane == 2:
        actual[plane].view(torch.int32)[-1] = torch.tensor(after, dtype=torch.int64).int()
    else:
        actual[plane][-1] = after
    if passes:
        assert_cache_equal(actual, expected, scale_max_ulps=budget)
    else:
        with pytest.raises(AssertionError, match=("key_cache", "value_cache", "scale_cache")[plane]) as error:
            assert_cache_equal(actual, expected, scale_max_ulps=budget)
        if plane == 2:
            assert "max finite absolute error nan" not in str(error.value)


@pytest.mark.parametrize("block_size", [128, 256, 1024])
@pytest.mark.parametrize("tail", [1, 8, 31, 127])
def test_recycled_batch_prefill_fixture_preserves_poison_and_reserves_decode_slots(monkeypatch, block_size, tail):
    """Exercise the real Python batch-padding route; stand in only for C++."""
    contexts = (block_size + 8, 2 * block_size + tail)
    tables, num_blocks = make_recycled_page_table(contexts, block_size)
    slots = slots_from_page_table(tables, contexts, block_size)
    assert (tables[:, 0] > num_blocks // 2).all()
    mapped = tables[tables != 0]
    assert mapped.unique().numel() == mapped.numel()
    assert (mapped % 2 == 1).all()  # a physical gap between every mapped page
    assert ((tables >= 0) & (tables < num_blocks)).all()
    actual = make_cache(num_blocks, block_size, 4, 256, poison=True)
    expected = tuple(plane.clone() for plane in actual)
    original = tuple(plane.clone() for plane in actual)
    assert (actual[0] != 0).all() and (actual[1] != 0).all()
    assert torch.isnan(actual[2]).any() and torch.isinf(actual[2]).any()
    assert (actual[2] == CACHE_POISON_SCALE).any()
    emulator = TurboQuantEmulator(head_size=256, device=CPU)
    golden = make_adapter(value_layout=TurboQuantValueLayout.NZ_TILED)
    launches = []

    def batch_kernel(key, value, key_cache, value_cache, scales, mapping, *_):
        launches.append(mapping.clone())
        assert key.shape[0] % TURBOQUANT_STRIP_TOKENS == 0
        golden.reshape_and_cache_prefill(key, value, key_cache, value_cache, scales, mapping, emulator)

    register_ops(monkeypatch, **{TURBOQUANT_OP_RESHAPE_AND_CACHE: batch_kernel})
    kernel = make_adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE)
    key, value = make_activations(sum(contexts), 4, 256, seed=17, dtype=torch.bfloat16)
    plan = turboquant_plan_strip_ingestion(slots.tolist(), block_size)
    assert not plan.waves, "the batch fixture must not silently fall back to scatter"
    kernel.reshape_and_cache_prefill(key, value, *actual, slots, emulator, plan=plan)
    assert launches
    start = 0
    for request, context in enumerate(contexts):
        written = -(-context // TURBOQUANT_STRIP_TOKENS) * TURBOQUANT_STRIP_TOKENS
        pad = written - context
        padded_key = torch.cat(
            (key[start : start + context], key[start + context - 1 : start + context].expand(pad, -1, -1))
        )
        padded_value = torch.cat(
            (value[start : start + context], value[start + context - 1 : start + context].expand(pad, -1, -1))
        )
        padded_slots = slots_from_page_table(tables[request : request + 1], [written], block_size)
        golden.reshape_and_cache_prefill(padded_key, padded_value, *expected, padded_slots, emulator)
        page_index = (context - 1) // block_size
        last_page = tables[request : request + 1, page_index : page_index + 1]
        got = golden.gather(*actual, last_page)
        poison = golden.gather(*original, last_page)
        offset = written - page_index * block_size
        for plane in (0, 2):
            assert torch.equal(got[plane][:, offset:], poison[plane][:, offset:])
        for plane in (1, 3):
            assert torch.isfinite(got[plane][:, :offset]).all()
            assert torch.equal(got[plane][:, offset:].view(torch.int32), poison[plane][:, offset:].view(torch.int32))
        # The two generated-token slots are valid, distinct, and either replace
        # intentional strip padding or occupy still-poisoned rows on a new page.
        extended = slots_from_page_table(tables[request : request + 1], [context + 2], block_size)
        assert extended[-2:].unique().numel() == 2
        assert (extended[-2:] >= 0).all() and (extended[-2:] < num_blocks * block_size).all()
        if tail == 127 and request == 1 and block_size == 128:
            assert extended[-2] // block_size != extended[-1] // block_size
        start += context
    assert_cache_equal(actual, expected)


@pytest.mark.parametrize(
    "before_bits,after_bits,error",
    [
        pytest.param(0x7FC0BEEF, 0, "neutralized", id="nan-to-zero"),
        pytest.param(0x7FC0BEEF, 0x00800000, "neutralized", id="nan-to-small-positive"),
        pytest.param(0x7FC0BEEF, 0x3F800000, "neutralized", id="nan-to-finite"),
        pytest.param(0x7FC0BEEF, 0x7FC0BEF0, "changed bits", id="nan-payload-mutated"),
        pytest.param(0x3F800000, 0x3F800001, "changed bits", id="positive-poison-one-ulp"),
    ],
)
def test_poison_mask_rejects_neutralization_before_live_scale_tolerance(before_bits, after_bits, error):
    initial = (
        torch.zeros(1, dtype=torch.int8),
        torch.zeros(1, dtype=torch.int8),
        torch.tensor([before_bits], dtype=torch.int32).view(torch.float32),
    )
    actual = (*initial[:2], torch.tensor([after_bits], dtype=torch.int32).view(torch.float32))
    mask = torch.ones_like(initial[2], dtype=torch.bool)
    with pytest.raises(AssertionError, match=error):
        assert_poison_scales_preserved(actual, initial, mask)
    with pytest.raises(AssertionError, match="scale_cache"):
        assert_cache_equal(actual, initial, scale_max_ulps=1, poison_mask=mask)
    if before_bits == 0x3F800000:
        # This would be admissible for a live scale, proving the mask rather
        # than the finite/positive predicate is what rejects it as poison.
        assert_cache_equal(actual, initial, scale_max_ulps=1)


def test_reported_scale_index_60254_is_a_batch_owned_value_scale():
    """Reconstruct the ragged-1024 fixture at commit 942755966 from its report."""
    block_size, index = 1024, 60254
    contexts = (block_size + 8, 8 * block_size + 31)
    tables = torch.tensor([[21, 19, 0, 0, 0, 0, 0, 0, 0], [17, 15, 13, 11, 9, 7, 5, 3, 1]], dtype=torch.int32)
    slot, lane = divmod(index, 2 * 4)
    page, offset = divmod(slot, block_size)
    assert (page, offset, lane) == (7, 363, 6)
    request = 1
    logical_page = tables[request].tolist().index(page)
    logical_token = logical_page * block_size + offset
    prefill_count = ((contexts[request] - TURBOQUANT_STRIP_TOKENS) // TURBOQUANT_STRIP_TOKENS) * TURBOQUANT_STRIP_TOKENS
    assert logical_token == 5483 < prefill_count == 8192 < contexts[request]
    assert lane // 4 == 1 and lane % 4 == 2  # V plane, KV head 2
    mapped_slot = slots_from_page_table(tables[request : request + 1], [contexts[request]], block_size)[logical_token]
    assert int(mapped_slot) * 8 + lane == index


@pytest.mark.parametrize("mode", [0, 1])
def test_finite_spike_fixture_is_query_aligned_and_zero_live_is_exact(monkeypatch, mode):
    monkeypatch.setattr(turboquant_codec, "TURBOQUANT_QUANT_MODE", mode)
    emulator = TurboQuantEmulator(head_size=256, device=CPU)
    golden = make_adapter(value_layout=TurboQuantValueLayout.NZ_TILED)
    cache = make_finite_spike_cache(2, 128, 4, 256, emulator=emulator)
    packed_k, scales_k, packed_v, scales_v = golden.gather(*cache, torch.tensor([[1]], dtype=torch.int32))
    key = golden.dequantize_cache(packed_k, scales_k, emulator, torch.float32)[0, 0]
    value = golden.dequantize_cache(packed_v, scales_v, emulator, torch.float32)[0, 0]
    # With 8192 live zero keys, a single finite aligned spike must dominate
    # the softmax; poisoning all nibbles with +7 would not establish alignment
    # with the query in the rotated cache basis.
    score = key.sum(dim=-1) / 16
    probability = torch.sigmoid(score - torch.log(torch.tensor(8192.0)))
    assert torch.isfinite(key).all() and torch.isfinite(value).all()
    assert (probability > 0.99).all()
    assert ((probability[:, None] * value).abs().amax(dim=-1) > 1).all()
    zeros = torch.zeros_like(scales_v)
    live = golden.dequantize_cache(packed_v, zeros, emulator, torch.float32)
    assert (live == 0).all(), "zero live scales must give exact zero in both codebooks"
