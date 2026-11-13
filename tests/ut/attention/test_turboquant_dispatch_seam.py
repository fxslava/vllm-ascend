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
* a prefill step of an arbitrary token count is planned rather than refused:
  whole strips go to the batch writer, and the leftovers go to the golden NZ
  writer now that the batch writer's NZ ingress owns the cache layout (the
  strip scatter kernel is kept off an NZ cache). Every token lands exactly
  once at the slot it was given, and a step that already met the batch
  writer's strip contract is dispatched exactly as it was before the planner
  existed.
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
    TURBOQUANT_DECODE_MAX_SPLITS,
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_KERNEL_OPS,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_GATHER_PAGES,
    TURBOQUANT_OP_QUANTIZE_PACK,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_ROTATE,
    TURBOQUANT_OP_UNPACK_DEQUANT,
    TURBOQUANT_REGISTERED_OPS,
    TURBOQUANT_STAGE_ROTATED_BASIS,
    TURBOQUANT_STAGE_UNROTATED,
    TURBOQUANT_STRIP_TOKENS,
    TURBOQUANT_TABLE_DTYPE,
    AscendTurboQuantDryRunBackendImpl,
    TurboQuantEmulator,
    TurboQuantIngestionPhase,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    _is_op_registered,
    load_turboquant_extension,
    turboquant_decode_query_rot_floats,
    turboquant_decode_workspace_floats,
    turboquant_plan_strip_ingestion,
    turboquant_tiling_upper_bound_words,
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


# The suite's BLOCK_SIZE is 8, which no strip fits in: it is there to pin the
# ABI and the layout arithmetic, not a page either writer would accept. The
# strip planner needs a page that holds whole 16-token strips, so the
# ingestion-split tests carve their own.
STRIP_BLOCK_SIZE = 2 * TURBOQUANT_STRIP_TOKENS
STRIP_NUM_BLOCKS = 8
STRIP_SLOTS = STRIP_BLOCK_SIZE * STRIP_NUM_BLOCKS


def _strip_caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (STRIP_NUM_BLOCKS, STRIP_BLOCK_SIZE, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(
            turboquant_scale_cache_shape(STRIP_NUM_BLOCKS, STRIP_BLOCK_SIZE, NUM_KV_HEADS), dtype=torch.float32
        ),
    )


# The ingestion-split tests run in the NZ-tiled cache, because that is what the
# staged batch writer writes now: TURBOQUANT_INGRESS_NZ=1 puts the page in
# 64-row tiles, so the page is 64 slots, and the slot-major scale record is
# padded to one fp32 block, which the (2, num_blocks, H, block_size) plane only
# fits exactly when 2 * H is a multiple of 8 -- hence 4 KV heads here, the
# geometry the kernel itself is compiled for.
NZ_BLOCK_SIZE = 64
NZ_NUM_KV_HEADS = 4
NZ_NUM_BLOCKS = 8


def _nz_caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (NZ_NUM_BLOCKS, NZ_BLOCK_SIZE, NZ_NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(
            turboquant_scale_cache_shape(NZ_NUM_BLOCKS, NZ_BLOCK_SIZE, NZ_NUM_KV_HEADS), dtype=torch.float32
        ),
    )


def _nz_activations(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261107)
    shape = (num_tokens, NZ_NUM_KV_HEADS, HEAD_SIZE)
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
    # The same BF16 rotation operand the two writers take; see _codec_tables.
    assert first[2].dtype is TURBOQUANT_TABLE_DTYPE
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
    # One whole strip at a strip-aligned slot: the plan is a single unpadded run,
    # so the adapter hands its own tensors over rather than a slice of them and
    # the recorded arguments are the caller's own objects.
    key, value = _activations(TURBOQUANT_STRIP_TOKENS)
    caches = _caches()
    slot_mapping = torch.arange(TURBOQUANT_STRIP_TOKENS, dtype=torch.int32)

    _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE).reshape_and_cache_prefill(
        key, value, *caches, slot_mapping, emulator
    )

    # The schema's order: key, value, the three planes, slot_mapping, pi_signs,
    # tables, inv_sqrt_len. Every scalar the C launcher also takes is derived in
    # torch_bindings.cpp from these tensors. There is no mode-4 channel-policy
    # argument: the audited writer is mode-2 only.
    args = entry.args
    assert len(args) == 9
    assert (args[0], args[1]) == (key, value)
    assert (args[2], args[3], args[4], args[5]) == (*caches, slot_mapping)
    assert torch.equal(args[6], emulator.pi_signs)
    # BF16, not int8: the batch writer stages this blob as the Cube's BF16 L1
    # operand (stage_blocks_nz reads `Blocks * Dim * Dim` bfloat16_t elements),
    # so an int8 tensor of the same element count is half the bytes the kernel
    # reads AND the wrong arithmetic. This assertion said int8 and passed while
    # the operand was wrong, so the byte count is asserted too -- that is the
    # quantity the kernel's L1 reservation is sized in.
    assert args[7].dtype is TURBOQUANT_TABLE_DTYPE
    assert args[7].dtype is torch.bfloat16
    assert args[7].numel() == 2 * (HEAD_SIZE // 32) * 32 * 32
    assert args[7].numel() * args[7].element_size() == 2 * 2 * (HEAD_SIZE // 32) * 32 * 32
    # +-1 is exact in BF16, so the conversion is a representation change and not
    # a rounding: every entry is still a unit sign.
    assert torch.equal(args[7].abs().to(torch.float32), torch.ones_like(args[7], dtype=torch.float32))
    assert args[8] == pytest.approx(emulator.inv_sqrt_len)
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
# Arbitrary token counts: the strip plan
# ---------------------------------------------------------------------------


def _plan(slots: list[int], block_size: int = STRIP_BLOCK_SIZE) -> list[tuple[int, int, int]]:
    plan = turboquant_plan_strip_ingestion(slots, block_size)
    return [(run.start, run.stop, run.pad) for run in plan.runs]


def _dispatched(plan: Any) -> list[int]:
    """Every token index the plan hands to a writer, runs then waves."""
    tokens = [token for run in plan.runs for token in range(run.start, run.stop)]
    tokens += [token for wave in plan.waves for token in wave]
    return tokens


def test_a_conforming_step_is_one_unpadded_run() -> None:
    plan = turboquant_plan_strip_ingestion(list(range(STRIP_BLOCK_SIZE)), STRIP_BLOCK_SIZE)
    assert plan.is_whole_batch
    assert plan.waves == ()
    assert plan.num_launches == 1


def test_an_empty_step_plans_nothing() -> None:
    """No tokens is no launch, rather than a grid of zero blocks."""
    plan = turboquant_plan_strip_ingestion([], STRIP_BLOCK_SIZE)
    assert (plan.runs, plan.waves, plan.num_launches) == ((), (), 0)
    assert not plan.is_whole_batch


def test_a_220_token_prompt_is_whole_strips_plus_a_padded_tail() -> None:
    """220 == 13 * 16 + 12: the crash this planner exists for.

    The 12 trailing tokens start at slot 208, which is strip-aligned, so they
    are completed to a whole strip by four replicas of token 219 rather than
    being refused or scattered one at a time.
    """
    plan = turboquant_plan_strip_ingestion(list(range(220)), STRIP_BLOCK_SIZE)
    assert [(run.start, run.stop, run.pad) for run in plan.runs] == [(0, 208, 0), (208, 220, 4)]
    assert plan.waves == ()
    assert plan.num_launches == 2


@pytest.mark.parametrize("num_tokens", list(range(1, 2 * TURBOQUANT_STRIP_TOKENS + 1)))
def test_no_token_count_is_refused_and_every_token_is_dispatched_once(num_tokens: int) -> None:
    """Whatever the count, the plan covers ``[0, T)`` exactly once.

    The runs are disjoint and ordered and the waves hold what is left, which is
    what makes the split bit-exact for the tokens the step was given.
    """
    plan = turboquant_plan_strip_ingestion(list(range(num_tokens)), STRIP_BLOCK_SIZE)
    assert sorted(_dispatched(plan)) == list(range(num_tokens))
    assert len(_dispatched(plan)) == num_tokens


def test_one_run_spans_a_page_boundary_when_every_strip_is_aligned() -> None:
    """The strip contract is per strip, not per step.

    Each strip's page comes off its own first slot, so two non-adjacent pages
    coalesce into one launch as long as no strip straddles them.
    """
    page_0 = list(range(0, STRIP_BLOCK_SIZE))
    page_5 = list(range(5 * STRIP_BLOCK_SIZE, 6 * STRIP_BLOCK_SIZE))
    assert _plan(page_0 + page_5) == [(0, 2 * STRIP_BLOCK_SIZE, 0)]


def test_a_request_boundary_inside_a_strip_is_padded_not_scattered() -> None:
    """Two requests in one batch, the first ending mid-strip.

    20 tokens from slot 0 and 6 from slot 64. Strip 0 is whole; tokens 16..19
    are a strip-aligned partial strip of the first request and tokens 20..25 one
    of the second, so both are padded out and the batch writer takes all three.
    """
    slots = list(range(0, 20)) + list(range(64, 70))
    assert _plan(slots) == [(0, 16, 0), (16, 20, 12), (20, 26, 10)]


def test_a_continuation_that_does_not_start_on_a_strip_scatters_its_head() -> None:
    """A chunk resuming at slot 20 cannot pad backwards.

    Slots 16..31 are one strip, and the four of them below 20 hold the previous
    chunk's tokens, so every token of this chunk that lands in that strip has to
    be read-modify-written. All twelve share it, so they are twelve waves -- one
    launch each, which is the price of not corrupting the four already there.
    """
    plan = turboquant_plan_strip_ingestion(list(range(20, 52)), STRIP_BLOCK_SIZE)
    assert [(run.start, run.stop, run.pad) for run in plan.runs] == [(12, 28, 0), (28, 32, 12)]
    assert plan.waves == tuple((token,) for token in range(12))
    assert not plan.is_whole_batch


def test_two_tokens_of_one_strip_never_share_a_wave() -> None:
    """The scatter writer read-modify-writes the whole strip.

    Two of its items on one strip in one launch would read the same 2048 bytes,
    splice their own nibbles in and write both copies back, so one update would
    be lost. Tokens of distinct strips go in one wave.
    """
    same_strip = turboquant_plan_strip_ingestion([5, 6, 7], STRIP_BLOCK_SIZE)
    assert same_strip.runs == ()
    assert same_strip.waves == ((0,), (1,), (2,))

    distinct_strips = turboquant_plan_strip_ingestion([5, 21, 37], STRIP_BLOCK_SIZE)
    assert distinct_strips.runs == ()
    assert distinct_strips.waves == ((0, 1, 2),)


def test_padding_slots_all_share_one_wave() -> None:
    """A dummy capture run is all -1, which the scatter writer skips.

    It used to reach the batch writer, which refuses a negative slot outright.
    """
    plan = turboquant_plan_strip_ingestion([-1, -1, -1], STRIP_BLOCK_SIZE)
    assert plan.runs == ()
    assert plan.waves == ((0, 1, 2),)


def test_the_pad_is_refused_when_it_would_land_on_a_token_of_this_step() -> None:
    """Replicas may never beat a real token to its slot.

    Twelve consecutive tokens from slot 0, then the four slots the pad wants --
    out of order, so they are not part of the run. The tail is scattered instead
    of padded.
    """
    slots = list(range(12)) + [13, 12, 15, 14]
    plan = turboquant_plan_strip_ingestion(slots, STRIP_BLOCK_SIZE)
    assert all(run.pad == 0 for run in plan.runs)
    assert sorted(_dispatched(plan)) == list(range(len(slots)))


def test_the_pad_is_refused_when_it_would_cross_into_the_next_page() -> None:
    """Unreachable with a legal page, and checked rather than assumed.

    A page of 24 tokens cannot hold whole strips, so the strip at slot 16 runs
    off the end of page 0. Four tokens there would want eight replicas in page
    1, which belongs to someone else; the planner scatters them instead.
    """
    plan = turboquant_plan_strip_ingestion([16, 17, 18, 19], 24)
    assert plan.runs == ()
    assert sorted(_dispatched(plan)) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Arbitrary token counts: the dispatch, and what lands
# ---------------------------------------------------------------------------


def _strip_layout_adapter() -> TurboQuantOpsAdapter:
    """A golden adapter in the strip layout, which only the scatter writer writes now.

    ``value_layout()`` is derived from which ingestion stage is staged, so a
    reference write has to be configured into the channel-major strip cache
    explicitly or it would land the golden model's own token-major one and
    compare against the wrong bytes.
    """
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(),
            value_layout=TurboQuantValueLayout.CHANNEL_MAJOR,
        )
    )


def _nz_layout_adapter() -> TurboQuantOpsAdapter:
    """A golden adapter in the NZ-tiled layout the staged batch writer writes.

    Same derivation caveat as above, the other way around: with no stage
    staged, ``value_layout()`` would report the golden model's own token-major
    cache, so the NZ layout is pinned explicitly.
    """
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(),
            value_layout=TurboQuantValueLayout.NZ_TILED,
        )
    )


def _staged_adapter() -> TurboQuantOpsAdapter:
    return _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.SCATTER)


class _GoldenWriter:
    """A registered writer that performs the golden NZ write and records its slots.

    The batch-writer kernel and the golden model land the same nibbles and the
    same fp32 scales -- that is the premise of this whole file, and the NZ
    ingress keeps it: TURBOQUANT_INGRESS_NZ changes WHERE the bytes go, not
    what they are. Standing the op up as the golden NZ write is what lets a
    host with no NPU check that the split lands exactly what one unsplit call
    would have.
    """

    def __init__(self, emulator: TurboQuantEmulator) -> None:
        self._emulator = emulator
        self.slots: list[list[int]] = []

    def __call__(self, key: torch.Tensor, value: torch.Tensor, *planes_and_rest: Any) -> None:
        key_cache, value_cache, scale_cache, slot_mapping = planes_and_rest[:4]
        self.slots.append(slot_mapping.tolist())
        _nz_layout_adapter()._write_slots(
            key, value, key_cache, value_cache, scale_cache, slot_mapping, self._emulator
        )


def _golden_writer(monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator) -> _GoldenWriter:
    """Register the batch writer as the golden NZ write, and the scatter as never-called.

    The strip scatter kernel is kept OFF an NZ cache by the adapter, so in the
    configuration both stages staged it must not run at all: the split's waves
    go to the golden NZ writer instead. The recorder standing in for it is what
    proves that.
    """
    writer = _GoldenWriter(emulator)
    scatter = _Recorder()
    _register(
        monkeypatch,
        npu_turboquant_reshape_and_cache_a8=writer,
        npu_turboquant_reshape_scatter_aiv=scatter,
    )
    return writer


def _replicas(slot_mapping: torch.Tensor, plan: Any) -> tuple[list[int], list[int]]:
    """The rows a plan replicates and the slots they land on."""
    rows: list[int] = []
    slots: list[int] = []
    for run in plan.runs:
        rows += [run.stop - 1] * run.pad
        slots += [int(slot_mapping[run.stop - 1]) + offset for offset in range(1, run.pad + 1)]
    return rows, slots


@pytest.mark.parametrize("first_slot", [0, 20])
@pytest.mark.parametrize("num_tokens", [1, 12, 16, 17, 31, 32, 64, 220])
def test_the_split_lands_what_one_unsplit_call_would_have(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator, num_tokens: int, first_slot: int
) -> None:
    """Bit-exact for every token the step was given, at any token count.

    The reference is one unsplit golden NZ write of the step, and then the
    replicas the plan asks for -- which land on slots the step does not write,
    so the bytes of every real token in it came from that single unsplit call.
    The two caches then have to agree everywhere, replicas included. The waves
    go to the golden NZ writer inside the adapter, not to the strip scatter
    kernel, which is exactly the interim state the NZ ingress leaves behind.
    """
    key, value = _nz_activations(num_tokens)
    slot_mapping = torch.arange(first_slot, first_slot + num_tokens, dtype=torch.int32)
    plan = turboquant_plan_strip_ingestion(slot_mapping.tolist(), NZ_BLOCK_SIZE)
    rows, pad_slots = _replicas(slot_mapping, plan)
    assert set(pad_slots).isdisjoint(slot_mapping.tolist())

    reference = _nz_layout_adapter()
    expected = _nz_caches()
    reference._write_slots(key, value, *expected, slot_mapping, emulator)
    if rows:
        index = torch.tensor(rows, dtype=torch.long)
        reference._write_slots(
            key.index_select(0, index),
            value.index_select(0, index),
            *expected,
            torch.tensor(pad_slots, dtype=torch.int32),
            emulator,
        )

    _golden_writer(monkeypatch, emulator)
    written = _nz_caches()
    _staged_adapter().reshape_and_cache_prefill(key, value, *written, slot_mapping, emulator)

    for got, want in zip(written, expected):
        assert torch.equal(got, want)


def test_the_only_slots_touched_beyond_the_step_are_the_planned_pad(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """A 220-token step writes slots 0..219 and nothing but its four replicas.

    The replicas are token 219 again, at slots 220..223, inside the page the
    request already owns -- so the whole side effect of the padding is four
    copies of the last token past the sequence length, and the pages past it
    never move.
    """
    num_tokens = 220
    key, value = _nz_activations(num_tokens)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

    writer = _golden_writer(monkeypatch, emulator)
    written = _nz_caches()
    _staged_adapter().reshape_and_cache_prefill(key, value, *written, slot_mapping, emulator)

    # Every slot the step was given, plus the four replicas, and not one more.
    assert sorted(slot for slots in writer.slots for slot in slots) == list(range(num_tokens + 4))

    # Slots 0..223 are pages 0..3 of this geometry, so page 4 is untouched. A
    # page IS one 64-row tile of the NZ fractal here, so the nominal page axis
    # and the physical one coincide.
    key_plane = written[0]
    assert torch.equal(key_plane[4], torch.zeros_like(key_plane[4]))
    assert int(key_plane[3].abs().sum()) > 0


def test_a_conforming_step_still_reaches_the_batch_writer_in_one_launch(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    batch, scatter = _Recorder(), _Recorder()
    _register(
        monkeypatch,
        npu_turboquant_reshape_and_cache_a8=batch,
        npu_turboquant_reshape_scatter_aiv=scatter,
    )
    key, value = _nz_activations(NZ_BLOCK_SIZE)
    slot_mapping = torch.arange(NZ_BLOCK_SIZE, dtype=torch.int32)

    _staged_adapter().reshape_and_cache_prefill(key, value, *_nz_caches(), slot_mapping, emulator)

    assert len(batch.calls) == 1
    assert scatter.calls == []
    assert batch.args[0] is key
    assert batch.args[5] is slot_mapping


def test_a_head_misaligned_step_reaches_the_batch_writer_and_golden_waves(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The head tokens go to the golden NZ writer, NOT to the strip scatter kernel.

    The batch launches are the plan's, unchanged by the ingress. What changed is
    who owns a wave: with the batch writer staged the cache is NZ-tiled, and the
    strip scatter kernel must not touch it, so the staged adapter falls back to
    its golden NZ writer even though the scatter op is registered. Proven twice
    over: the scatter recorder stays silent, and the head tokens' bytes are in
    the cache anyway.
    """
    batch, scatter = _Recorder(), _Recorder()
    _register(
        monkeypatch,
        npu_turboquant_reshape_and_cache_a8=batch,
        npu_turboquant_reshape_scatter_aiv=scatter,
    )
    key, value = _nz_activations(32)
    slot_mapping = torch.arange(20, 52, dtype=torch.int32)

    written = _nz_caches()
    _staged_adapter().reshape_and_cache_prefill(key, value, *written, slot_mapping, emulator)

    # Two batch launches -- one whole strip, one padded -- and no scatter call.
    assert len(batch.calls) == 2
    assert batch.calls[0][0][5].tolist() == list(range(32, 48))
    assert batch.calls[1][0][5].tolist() == list(range(48, 64))
    assert scatter.calls == []

    # The 12 head tokens landed regardless, through the golden NZ writer.
    head = slice(0, 12)
    reference_slots = torch.arange(20, 32, dtype=torch.int32)
    head_only = _nz_caches()
    _nz_layout_adapter()._write_slots(key[head], value[head], *head_only, reference_slots, emulator)
    for got, want in zip(written, head_only):
        mask = want != 0
        assert torch.equal(got[mask], want[mask])


def test_the_scatter_waves_fall_back_to_golden_without_the_scatter_kernel(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """Staging the batch writer on alone must still land every token.

    The leftover tokens go through the public decode entry, which picks the
    golden writer when the scatter kernel is not registered -- the same fallback
    every other operation in this file takes, in the NZ layout the staged batch
    writer puts the cache in.
    """
    batch = _Recorder()
    _register(monkeypatch, npu_turboquant_reshape_and_cache_a8=batch)
    key, value = _nz_activations(4)
    slot_mapping = torch.arange(20, 24, dtype=torch.int32)
    caches = _nz_caches()

    _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE).reshape_and_cache_prefill(
        key, value, *caches, slot_mapping, emulator
    )

    assert batch.calls == []
    expected = _nz_caches()
    _nz_layout_adapter()._write_slots(key, value, *expected, slot_mapping, emulator)
    for got, want in zip(caches, expected):
        assert torch.equal(got, want)


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


def _launch_buffers(tokens: int, num_heads: int, head_size: int = HEAD_SIZE) -> dict[str, torch.Tensor]:
    """The three buffers the fused decode takes as inputs.

    They belong to AscendTurboQuantMetadataBuilder in service; a test supplies
    its own. None of them has a default on the adapter, deliberately: an
    unfilled tiling buffer is all zeros, which the kernel reads as "every worker
    has no work" and returns from having written nothing.
    """
    splits = TURBOQUANT_DECODE_SPLITS
    return {
        "tiling": torch.zeros(
            turboquant_tiling_upper_bound_words(tokens, TURBOQUANT_DECODE_MAX_SPLITS), dtype=torch.int32
        ),
        "workspace": torch.zeros(
            turboquant_decode_workspace_floats(tokens, num_heads, head_size, splits), dtype=torch.float32
        ),
        "query_rot": torch.zeros(
            turboquant_decode_query_rot_floats(tokens, num_heads, head_size), dtype=torch.float32
        ),
    }


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
        **_launch_buffers(1, 2),
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
        **_launch_buffers(1, 2),
    )
    assert returned is output
    assert len(entry.calls) == 1


def test_fused_decode_passes_the_schema_the_binding_declares(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """The 21 arguments of npu_turboquant_fused_decode, in the schema's order.

    The mode, the element type and the two codec constant pools are the
    binding's, so none of them appears here. The tiling descriptors, the Split-K
    workspace and the fp32 rotated query image used to be as well and are
    arguments now: a buffer the op allocates has a fresh address every step, and
    a captured ACL graph replays the address it recorded.
    """
    entry = _Recorder()
    _register(monkeypatch, npu_turboquant_fused_decode=entry)
    caches = _caches()
    query = torch.randn(1, 12, HEAD_SIZE)
    output = torch.zeros_like(query)
    block_tables = torch.zeros(1, 3, dtype=torch.int32)
    context_lens = torch.ones(1, dtype=torch.int32)
    buffers = _launch_buffers(1, 12)

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
        **buffers,
    )

    args = entry.args
    assert len(args) == 21
    assert args[0] is query
    assert torch.equal(args[1], emulator.pi_signs)
    assert args[2].dtype is TURBOQUANT_TABLE_DTYPE  # rot_tables: the same BF16 blob
    # Unread under KV4_FP8 (ConstTableWords is 0 for an affine mode with the
    # gather LUT off), so the decode would have accepted either dtype. Asserted
    # anyway: a decode that runs is no evidence that this operand is right.
    assert args[2].numel() == 2 * (HEAD_SIZE // 32) * 32 * 32
    assert args[3].shape == (16, 16) and args[3].dtype is torch.float16  # h16
    assert args[4] is None  # gate, read only at output_stage 2
    assert (args[5], args[6], args[7]) == caches
    assert (args[8], args[9]) == (block_tables, context_lens)
    assert (args[10], args[11], args[12]) == (buffers["tiling"], buffers["workspace"], buffers["query_rot"])
    assert args[13] is output
    # num_heads, num_kv_heads, num_splits, heads_per_task, output_stage
    assert args[14:19] == (12, NUM_KV_HEADS, TURBOQUANT_DECODE_SPLITS, 6, TURBOQUANT_STAGE_UNROTATED)
    assert args[19] == pytest.approx(0.125)
    assert args[20] == pytest.approx(emulator.inv_sqrt_len)


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
    """``TURBOQUANT_REGISTERED_OPS`` must not drift from the real schemas.

    Four, and only three of them are stage-gated.
    ``npu_turboquant_build_decode_tiling`` is not an alternative to a golden
    model -- it is the fused decode's own descriptor builder, registered with
    the decode and absent with it -- so it has no entry in
    ``TURBOQUANT_KERNEL_OPS`` and is not in ``ALL_OPS``.
    """
    assert set(TURBOQUANT_REGISTERED_OPS) == {
        TURBOQUANT_OP_RESHAPE_AND_CACHE,
        TURBOQUANT_OP_RESHAPE_SCATTER,
        TURBOQUANT_OP_FUSED_DECODE,
        TURBOQUANT_OP_BUILD_DECODE_TILING,
    }
    assert set(ALL_OPS) == set(TURBOQUANT_KERNEL_OPS)
    assert set(TURBOQUANT_REGISTERED_OPS) - set(ALL_OPS) == {TURBOQUANT_OP_BUILD_DECODE_TILING}


# ---------------------------------------------------------------------------
# What the staged kernels can actually run
#
# The binding TORCH_CHECKs every one of these, and that is the check that
# matters -- a direct caller cannot skip it. These cover the earlier refusal:
# the binding's arrives from inside an attention forward, one decode step into a
# server that has already loaded weights and sized a KV cache, and names a
# tensor shape; this one names the model and the stage at layer construction.
#
# The guard is a REFUSAL rather than a warn-and-fall-back, unlike a missing
# registration. A kernel that is absent has a golden model behind it and the run
# stays correct. A kernel that is present and mis-specialised writes one head's
# rows into another head's page with every canary intact, and neither writer
# checks its runtime head count against the count it was unrolled over.
# ---------------------------------------------------------------------------


def _geometry(
    *stages: TurboQuantKernelStage,
    head_size: int = turboquant_dry_run.TURBOQUANT_KERNEL_HEAD_SIZE,
    num_kv_heads: int = turboquant_dry_run.TURBOQUANT_KERNEL_KV_HEADS,
    num_heads: int | None = None,
    model_dtype: torch.dtype = turboquant_dry_run.TURBOQUANT_KERNEL_DTYPE,
) -> Any:
    """The state :meth:`_check_kernel_geometry` reads, and nothing else.

    Called unbound, like the ``_ingestion_phase`` test above: the real
    ``__init__`` wants a vLLM attention layer and the method wants five
    attributes.
    """
    return types.SimpleNamespace(
        adapter=_adapter(*stages),
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads if num_heads is not None else num_kv_heads * 4,
        model_dtype=model_dtype,
    )


def _check(state: Any) -> None:
    AscendTurboQuantDryRunBackendImpl._check_kernel_geometry(state)


@pytest.mark.parametrize(
    "stages",
    [
        (),
        (TurboQuantKernelStage.ROTATE,),
        (TurboQuantKernelStage.QUANTIZE, TurboQuantKernelStage.GATHER),
    ],
)
def test_a_golden_run_is_not_constrained_by_the_kernels(stages: tuple[TurboQuantKernelStage, ...]) -> None:
    """No writer and no decode staged: every shape the golden model admits is fine.

    The four stages with no kernel behind them must not constrain anything
    either -- they always fall back, so a run that names them is a golden run.
    """
    _check(_geometry(*stages, head_size=64, num_kv_heads=7, num_heads=7, model_dtype=torch.float16))


@pytest.mark.parametrize(
    "stage",
    [
        TurboQuantKernelStage.RESHAPE_AND_CACHE,
        TurboQuantKernelStage.SCATTER,
        TurboQuantKernelStage.DECODE,
    ],
)
def test_every_kernel_pins_the_head_size(stage: TurboQuantKernelStage) -> None:
    """``kHeadSize`` in both writers, ``dim != kHeadSize`` in ``valid_launch``."""
    _check(_geometry(stage))  # the pinned size is admitted
    with pytest.raises(ValueError, match=r"head_size 256"):
        _check(_geometry(stage, head_size=128))


@pytest.mark.parametrize("stage", [TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.SCATTER])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_both_writers_need_exactly_four_kv_heads(stage: TurboQuantKernelStage, num_kv_heads: int) -> None:
    """``kChunks = 2 * kHeads`` chunk phases are unrolled by template argument.

    Fewer is refused as well as more: the schedule is not "up to four".
    """
    with pytest.raises(ValueError, match=r"specialised to exactly 4 KV heads"):
        _check(_geometry(stage, num_kv_heads=num_kv_heads))


@pytest.mark.parametrize("num_kv_heads", [1, 2, 3, 4])
def test_the_decode_tolerates_fewer_kv_heads_than_the_writers_need(num_kv_heads: int) -> None:
    """``valid_launch`` refuses more than four, not fewer."""
    _check(_geometry(TurboQuantKernelStage.DECODE, num_kv_heads=num_kv_heads))


@pytest.mark.parametrize("num_kv_heads", [5, 8, 16])
def test_the_decode_refuses_the_kv_head_count_a_gqa_model_brings(num_kv_heads: int) -> None:
    """The constraint a deployed model is most likely to hit: 8 KV heads is common."""
    with pytest.raises(ValueError, match=r"num_kv_heads outside 1\.\.4"):
        _check(_geometry(TurboQuantKernelStage.DECODE, num_kv_heads=num_kv_heads))


def test_the_decode_needs_a_whole_gqa_group() -> None:
    with pytest.raises(ValueError, match=r"multiple of num_kv_heads"):
        _check(_geometry(TurboQuantKernelStage.DECODE, num_kv_heads=3, num_heads=16))


@pytest.mark.parametrize(
    "stage",
    [
        TurboQuantKernelStage.RESHAPE_AND_CACHE,
        TurboQuantKernelStage.SCATTER,
        TurboQuantKernelStage.DECODE,
    ],
)
def test_fp16_is_refused_rather_than_dispatched(stage: TurboQuantKernelStage) -> None:
    """There is no fp16 instance, and the decode's dispatch does not say so.

    ``turboquant_mm_fused_decode_opt_raw_query_impl`` guards its launch with
    ``if (type == AscendType::BF16)`` and returns without launching for anything
    else, which would leave the attention output unwritten and report success.
    That is the failure this refuses, so it is refused at construction and
    again in the binding.
    """
    with pytest.raises(ValueError, match=r"torch.bfloat16 only"):
        _check(_geometry(stage, model_dtype=torch.float16))


def test_the_refusal_names_the_stage_that_cannot_run_the_model() -> None:
    """A message that named only the shape would not say which switch to flip."""
    with pytest.raises(ValueError, match=r"reshape_and_cache") as both:
        _check(_geometry(TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.DECODE, head_size=64))
    assert "decode" in str(both.value)
    assert turboquant_dry_run.TURBOQUANT_STAGES_ENV in str(both.value)


def test_the_rotation_operand_is_the_bf16_blob_the_cube_stages() -> None:
    """``_codec_tables`` sizing, in the quantity the kernel's L1 reservation uses.

    ``stage_blocks_nz<Blocks, Dim>`` reads ``Blocks * Dim * Dim`` bfloat16_t
    elements, so the BYTE count is what has to match -- an int8 blob of the same
    element count is half of it, which is the defect this pins down. Every entry
    is a unit sign, exact in BF16, so the conversion is a representation change
    and the operand stays bit-identical to the oracle's.
    """
    emulator = TurboQuantEmulator(head_size=turboquant_dry_run.TURBOQUANT_KERNEL_HEAD_SIZE, device=CPU)
    tables = turboquant_dry_run._codec_tables(emulator)
    groups = turboquant_dry_run.TURBOQUANT_KERNEL_HEAD_SIZE // 32
    assert tables.dtype is TURBOQUANT_TABLE_DTYPE
    assert tables.shape == (2 * groups, 32, 32)
    assert tables.element_size() == 2
    assert tables.numel() * tables.element_size() == 2 * 2 * groups * 32 * 32 == 32768
    assert torch.equal(tables.abs().to(torch.float32), torch.ones(tables.shape, dtype=torch.float32))
