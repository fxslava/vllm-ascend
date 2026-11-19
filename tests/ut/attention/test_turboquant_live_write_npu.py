# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live TurboQuant cache contracts: writers, then prefill -> first decode steps.

Qwen3.5-27B A8W8 full attention uses 24 query heads, 4 KV heads and D=256.
These are kernel tests, not model/logit acceptance tests. They need an Ascend
NPU and the registered TurboQuant extension; CPU collection skips them.

Recycled-page cases start in the shuffled upper half of a cache filled with
nonzero packed bytes and NaN/Inf/large scales. Batch prefill deliberately pads
the final 16-token strip. The rest of its last page must retain the original
poison, including dead rows in the final 64-token decode tile. Attention must
ignore both the strip padding and that poison immediately after prefill and
on the first two decode steps. Short contexts use a shared builder/launch
threshold of 192 to exercise real Split-K without long Python references.
Representative grids cover page crossings, splits 1/2/3/8, and block sizes
128/256/1024. Cross-request cases poison one request and check clean siblings.
Finite-spike profiles use zero live keys, an all-ones query, and either exact
zero live values (contribution leaks) or constant live values (denominator
leaks). A single exposed spike must produce finite, nontrivial attention.

Run: pytest -sv tests/ut/attention/test_turboquant_live_write_npu.py
"""

from __future__ import annotations

import sys

import pytest
import torch

HEAD_SIZE = 256
NUM_KV_HEADS = 4
NUM_HEADS = 24
GROUP_HEADS = NUM_HEADS // NUM_KV_HEADS
STRIP = 16
TILE_ROWS = 64
FUSED_CONTEXT_LIMIT = 192
PREFILL_SPLITS = (1, 2, 3, 8)
APPEND_SPLITS = (1, 3, 8)
WRITER_BLOCK_SIZE = 64
NUM_BLOCKS = 8
ATTENTION_SCALE = HEAD_SIZE**-0.5
DECODE_ROW_MAX_BUDGET = 0.10
DECODE_COSINE_FLOOR = 0.999
NAMESPACE = "turboquant_ops"
PREFILL_OP = "npu_turboquant_reshape_and_cache_a8"
SCATTER_OP = "npu_turboquant_reshape_scatter_aiv"
FUSED_DECODE_OP = "npu_turboquant_fused_decode"
BUILD_TILING_OP = "npu_turboquant_build_decode_tiling"


def _npu_device():
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available() and torch.npu.device_count():
            return torch.device("npu:0")
    except (ImportError, AttributeError, RuntimeError):
        pass
    return None


DEVICE = _npu_device()
requires_npu = pytest.mark.skipif(DEVICE is None, reason="requires an Ascend NPU")


def _backend():
    # Lazy imports keep standalone CPU collection independent of vLLM.
    from vllm_ascend.attention import turboquant_v1

    return turboquant_v1


def _helpers():
    from tests.helpers import turboquant

    return turboquant


def _set_decode_splits(monkeypatch, splits):
    # The adapter reads its defining module's constant, not the entry point's
    # re-export. Patching turboquant_v1 leaves the launch at eight splits.
    from vllm_ascend.attention import turboquant_ops

    monkeypatch.setattr(turboquant_ops, "TURBOQUANT_DECODE_SPLITS", splits)


def _require_ops(*names):
    backend = _backend()
    backend.load_turboquant_extension()
    missing = [name for name in names if not hasattr(getattr(torch.ops, NAMESPACE, None), name)]
    if missing:
        pytest.skip(f"requires an ascend950 TurboQuant build with {missing}")


def _adapter(*stages):
    backend = _backend()
    return backend.TurboQuantOpsAdapter(
        backend.TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(stages),
            value_layout=backend.TurboQuantValueLayout.NZ_TILED,
        )
    )


def _caches(block_size, num_blocks=NUM_BLOCKS):
    return _helpers().make_cache(num_blocks, block_size, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, poison=True)


def _activations(tokens, seed):
    return _helpers().make_activations(tokens, NUM_KV_HEADS, HEAD_SIZE, seed=seed, device=DEVICE, dtype=torch.bfloat16)


def _compare(actual, expected, *, writer_reference=False, poison_mask=None):
    # The supplied NPU run showed one-ULP Lloyd-Max scale differences even
    # though the source reduction trees agree. Codes, poison and snapshots
    # remain exact; the precise device arithmetic difference is unconfirmed.
    budget = int(writer_reference and _backend().TURBOQUANT_QUANT_MODE == 1)
    _helpers().assert_cache_equal(actual, expected, scale_max_ulps=budget, poison_mask=poison_mask)


def _clone_cache(cache):
    return tuple(plane.clone() for plane in cache)


@requires_npu
class TestCacheWriters:
    @pytest.mark.parametrize(
        "writer,steps,placement",
        [
            pytest.param("batch", (16,), "scattered", id="batch-one-strip"),
            pytest.param("batch", (64,), "scattered", id="batch-scattered-strips"),
            pytest.param("batch", (64,), "one-page", id="batch-every-strip-of-page"),
            pytest.param("scatter", (1,), "scattered", id="scatter-one-token"),
            pytest.param("scatter", (3,), "scattered", id="scatter-three-tokens"),
            pytest.param("scatter", (17,), "scattered", id="scatter-seventeen-tokens"),
            pytest.param("scatter", (1,) * STRIP, "one-page", id="scatter-repeated-steps"),
        ],
    )
    def test_write_preserves_unowned_slots_and_gather_round_trips(self, writer, steps, placement):
        """Consolidate writer byte checks, strip ownership and golden read-back."""
        _require_ops(PREFILL_OP if writer == "batch" else SCATTER_OP)
        backend = _backend()
        emulator = backend.TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
        stage = (
            backend.TurboQuantKernelStage.RESHAPE_AND_CACHE
            if writer == "batch"
            else backend.TurboQuantKernelStage.SCATTER
        )
        kernel, golden = _adapter(stage), _adapter()
        actual, expected = _caches(WRITER_BLOCK_SIZE), _caches(WRITER_BLOCK_SIZE)
        poison = _clone_cache(actual)
        poison_mask = torch.ones_like(actual[2], dtype=torch.bool)
        tokens = sum(steps)
        generator = torch.Generator().manual_seed(7)
        if placement == "one-page":
            slots = 3 * WRITER_BLOCK_SIZE + torch.arange(tokens)
        elif writer == "batch":
            strips = torch.randperm(NUM_BLOCKS * WRITER_BLOCK_SIZE // STRIP, generator=generator)[: tokens // STRIP]
            slots = (strips[:, None] * STRIP + torch.arange(STRIP)).reshape(-1)
        else:
            slots = torch.randperm(NUM_BLOCKS * WRITER_BLOCK_SIZE, generator=generator)[:tokens]
        slots = slots.to(device=DEVICE, dtype=torch.int32)
        key, value = _activations(tokens, seed=1500 + tokens)
        method = "reshape_and_cache_prefill" if writer == "batch" else "reshape_scatter_decode"
        start = 0
        for count in steps:
            rows = slice(start, start + count)
            for adapter, cache in ((kernel, actual), (golden, expected)):
                getattr(adapter, method)(key[rows], value[rows], *cache, slots[rows], emulator)
            start += count
            poison_mask.view(-1, 2 * NUM_KV_HEADS)[slots[rows].long()] = False
            torch.npu.synchronize()
            _helpers().assert_poison_scales_preserved(actual, poison, poison_mask)
            _compare(actual, expected, writer_reference=True, poison_mask=poison_mask)
        tables = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=DEVICE).unsqueeze(0)
        gathered = [golden.gather(*cache, tables) for cache in (actual, expected)]
        _compare(
            (gathered[0][0], gathered[0][2], torch.stack((gathered[0][1], gathered[0][3]))),
            (gathered[1][0], gathered[1][2], torch.stack((gathered[1][1], gathered[1][3]))),
            writer_reference=True,
        )
        for original, packed, scales in (
            (key, gathered[0][0], gathered[0][1]),
            (value, gathered[0][2], gathered[0][3]),
        ):
            dense = golden.dequantize_cache(packed[0, slots.long()], scales[0, slots.long()], emulator, torch.float32)
            cosine = torch.nn.functional.cosine_similarity(
                dense.flatten().cpu(), original.float().flatten().cpu(), dim=0
            )
            assert float(cosine) > 0.98, "cache round trip lost activation information"

    def test_scatter_preserves_nan_payload_in_adjacent_and_remote_slots(self):
        """Minimal reproducer: one written row, two neighbors and a remote NaN."""
        _require_ops(SCATTER_OP)
        backend = _backend()
        block_size, physical_page = 128, 5
        cache = _caches(block_size)
        # Use a noncanonical NaN payload so canonicalization also fails.
        cache[2].view(torch.int32).fill_(0x7FC0BEEF)
        initial = _clone_cache(cache)
        written_slot = physical_page * block_size + TILE_ROWS - 1
        poison_slots = torch.tensor([written_slot - 1, written_slot + 1, 0, 7 * block_size + 127])
        before = cache[2].view(-1, 2 * NUM_KV_HEADS)[poison_slots.to(DEVICE)].cpu()
        assert torch.isnan(before).all(), "fixture did not plant NaN in the known poison slots"
        key, value = _activations(1, seed=60254)
        emulator = backend.TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
        _adapter(backend.TurboQuantKernelStage.SCATTER).reshape_scatter_decode(
            key, value, *cache, torch.tensor([written_slot], dtype=torch.int32, device=DEVICE), emulator
        )
        torch.npu.synchronize()
        mask = torch.ones_like(cache[2], dtype=torch.bool)
        mask.view(-1, 2 * NUM_KV_HEADS)[written_slot] = False
        _helpers().assert_poison_scales_preserved(cache, initial, mask)
        after = cache[2].view(-1, 2 * NUM_KV_HEADS)[poison_slots.to(DEVICE)].cpu()
        assert torch.isnan(after).all(), "scatter neutralized an adjacent/remote NaN canary"
        assert torch.equal(after.view(torch.int32), before.view(torch.int32)), "scatter changed a NaN payload"
        # Read through the layout-aware gather as well as direct GM readback.
        pages = (poison_slots // block_size).to(device=DEVICE, dtype=torch.int32).unsqueeze(1)
        gathered = _adapter().gather(*cache, pages)
        for plane, scales in enumerate((gathered[1], gathered[3])):
            rows = scales[torch.arange(len(poison_slots), device=DEVICE), (poison_slots % block_size).to(DEVICE)].cpu()
            assert torch.isnan(rows).all(), "gather neutralized a NaN scale"
            expected = before[:, plane * NUM_KV_HEADS : (plane + 1) * NUM_KV_HEADS]
            assert torch.equal(rows.view(torch.int32), expected.view(torch.int32)), "gather changed a NaN payload"

    def test_scatter_configuration_uses_reference_decode(self, monkeypatch):
        """The scatter-only configuration declines fused decode and reads NZ."""
        _require_ops(SCATTER_OP)
        backend = _backend()
        monkeypatch.setenv(backend.TURBOQUANT_STAGES_ENV, "scatter")
        monkeypatch.setenv(backend.TURBOQUANT_NAMESPACE_ENV, NAMESPACE)
        adapter = backend.TurboQuantOpsAdapter()
        assert adapter.value_layout() is backend.TurboQuantValueLayout.NZ_TILED
        emulator = backend.TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
        caches = _caches(WRITER_BLOCK_SIZE)
        key, value = _activations(STRIP, seed=5000)
        slots = torch.arange(STRIP, dtype=torch.int32, device=DEVICE)
        adapter.reshape_scatter_decode(key, value, *caches, slots, emulator)
        query = torch.zeros(1, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
        empty = torch.empty(0, device=DEVICE)
        assert (
            adapter.fused_decode(
                query=query,
                key_cache=caches[0],
                value_cache=caches[1],
                scale_cache=caches[2],
                block_tables=torch.zeros((1, 1), dtype=torch.int32, device=DEVICE),
                context_lens=torch.tensor([STRIP], dtype=torch.int32, device=DEVICE),
                emulator=emulator,
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                scale=ATTENTION_SCALE,
                output=torch.empty_like(query),
                tiling=empty,
                workspace=empty,
                query_rot=empty,
            )
            is None
        )
        packed, scales, _, _ = adapter.gather(*caches, torch.zeros((1, 1), dtype=torch.int32, device=DEVICE))
        dense = adapter.dequantize_cache(packed[0, :STRIP], scales[0, :STRIP], emulator, torch.float32)
        cosine = torch.nn.functional.cosine_similarity(dense.flatten().cpu(), key.float().flatten().cpu(), dim=0)
        assert float(cosine) > 0.98


def _build_decode_tiling(contexts, block_size, max_pages, splits, fused_context_limit=FUSED_CONTEXT_LIMIT):
    backend = _backend()
    batch = len(contexts)
    heads_per_task = backend.turboquant_decode_heads_per_task(NUM_HEADS, NUM_KV_HEADS)
    chunks_per_group = -(-GROUP_HEADS // heads_per_task)
    # Match decode_launch_split in op_host/turboquant_tiling.h: descriptors
    # cover every (token, split), followed by one dispatch record per block.
    grid = min(backend.TURBOQUANT_DECODE_MAX_GRID, batch * NUM_KV_HEADS * splits * chunks_per_group)
    tiling_words = backend.turboquant_decode_tiling_words(batch, splits, grid)
    buffers = (
        torch.zeros(tiling_words, dtype=torch.int32),
        torch.empty(
            backend.turboquant_decode_workspace_floats(batch, NUM_HEADS, HEAD_SIZE, splits),
            dtype=torch.float32,
            device=DEVICE,
        ),
        torch.empty(
            backend.turboquant_decode_query_rot_floats(batch, NUM_HEADS, HEAD_SIZE), dtype=torch.float32, device=DEVICE
        ),
    )
    getattr(getattr(torch.ops, NAMESPACE), BUILD_TILING_OP)(
        torch.tensor(contexts, dtype=torch.int32),
        buffers[0],
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        block_size,
        max_pages,
        splits,
        heads_per_task,
        fused_context_limit,
    )
    descriptor = buffers[0][: backend.TURBOQUANT_DECODE_TILING_WORDS]
    assert int(descriptor[0]) > 0 and int(descriptor[10]) & 1, "builder left a dead decode descriptor"
    return (buffers[0].to(DEVICE), *buffers[1:])


def _reference_attention_fp32(
    adapter,
    emulator,
    caches,
    block_tables: torch.Tensor,
    query: torch.Tensor,
    context: int,
    scale: float,
) -> torch.Tensor:
    """The reference: the cache read by the golden gather, attended in fp32.

    This is the backend's fallback path with the maths inlined: gather the
    packed pages, dequantise into the model's basis, softmax(q k^T scale) v per
    query head over its KV group, all in fp32. ``[B, N, D]`` like the kernel's
    output.
    """
    packed_key, key_scales, packed_value, value_scales = adapter.gather(*caches, block_tables)
    key_dense = adapter.dequantize_cache(packed_key, key_scales, emulator, torch.float32)
    value_dense = adapter.dequantize_cache(packed_value, value_scales, emulator, torch.float32)
    key_dense = key_dense[0, :context].float()  # [S, kv, D]
    value_dense = value_dense[0, :context].float()

    q = query[0].float().reshape(NUM_KV_HEADS, GROUP_HEADS, HEAD_SIZE)
    scores = torch.matmul(q, key_dense.permute(1, 2, 0)) * scale
    output = torch.matmul(torch.softmax(scores, dim=-1), value_dense.permute(1, 0, 2))
    return output.reshape(1, NUM_HEADS, HEAD_SIZE).cpu()


def _assert_attention_close(tag, got, expected, context):
    got32, expected32 = got.float().cpu(), expected.float().cpu()
    assert torch.isfinite(expected32).all(), f"{tag}: non-finite reference attention"
    assert torch.isfinite(got32).all(), f"{tag}: non-finite attention; context={context}"
    if bool((expected32 == 0).all()):
        assert bool((got32 == 0).all()), (
            f"{tag}: finite tail spike leaked into zero-live attention; max_abs={float(got32.abs().max())}"
        )
        return
    assert (got32.abs().amax(dim=-1) > 0).all(), f"{tag}: zero attention row; context={context}"
    row_max = expected32.abs().amax(dim=-1).clamp(min=1e-30)
    error = (got32 - expected32).abs().amax(dim=-1) / row_max
    assert float(error.max()) <= DECODE_ROW_MAX_BUDGET, (
        f"{tag}: attention max error {float(error.max()):.6g} of row maximum; context={context}, "
        f"budget={DECODE_ROW_MAX_BUDGET} (FP8 query/probability rounding)"
    )
    cosine = torch.nn.functional.cosine_similarity(got32.flatten(), expected32.flatten(), dim=0)
    assert float(cosine) >= DECODE_COSINE_FLOOR, f"{tag}: attention cosine={float(cosine):.6g}"


def _launch_serving_decode(
    adapter, emulator, caches, block_tables, contexts, query, buffers, fused_context_limit=FUSED_CONTEXT_LIMIT
):
    output = torch.full_like(query, float("nan"))
    for buffer in buffers[1:]:
        buffer.fill_(float("nan"))
    fused = adapter.fused_decode(
        query=query,
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=block_tables,
        context_lens=torch.tensor(contexts, dtype=torch.int32, device=query.device),
        emulator=emulator,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        scale=ATTENTION_SCALE,
        output=output,
        tiling=buffers[0],
        workspace=buffers[1],
        query_rot=buffers[2],
        fused_context_limit=fused_context_limit,
    )
    assert fused is not None, "the registered decode kernel must run, not fall back"
    return output.cpu()


def _scale_rows(cache, slots, value):
    # Four KV heads use exactly eight FP32 lanes per physical NZ slot.
    rows = cache[2].view(-1, 2 * NUM_KV_HEADS)
    rows[slots.to(device=rows.device, dtype=torch.long)] = value


def _copy_cache_rows(target, source, slots):
    slots = slots.to(device=DEVICE, dtype=torch.long)
    offsets = _backend().TurboQuantOpsAdapter._nz_byte_offsets(slots, NUM_KV_HEADS, HEAD_SIZE // 2)
    for plane in (0, 1):
        target[plane].view(-1)[offsets] = source[plane].view(-1)[offsets]
    target[2].view(-1, 2 * NUM_KV_HEADS)[slots] = source[2].view(-1, 2 * NUM_KV_HEADS)[slots]


def _assert_prefill_tail(cache, poison, golden, tables_host, contexts, written_ends, block_size):
    """Assert strip padding and untouched poison separately, before decoding."""
    tail_slots, exposed_contexts = [], []
    for request, (context, written) in enumerate(zip(contexts, written_ends)):
        page_index, last = divmod(context - 1, block_size)
        live_end = last + 1
        written_end = written - page_index * block_size
        table = tables_host[request : request + 1, page_index : page_index + 1].to(DEVICE)
        actual = golden.gather(*cache, table)
        initial = golden.gather(*poison, table)
        for scales in (actual[1], actual[3]):
            assert torch.isfinite(scales[:, :written_end]).all(), "prefill left poison in a live/padded row"
        _compare(
            (
                actual[0][:, written_end:],
                actual[2][:, written_end:],
                torch.stack((actual[1][:, written_end:], actual[3][:, written_end:])),
            ),
            (
                initial[0][:, written_end:],
                initial[2][:, written_end:],
                torch.stack((initial[1][:, written_end:], initial[3][:, written_end:])),
            ),
        )
        poison_in_tile = -(-context // TILE_ROWS) * TILE_ROWS - written
        print(
            f"  request={request} context={context} page={int(table[0, 0])}: "
            f"strip padding={written_end - live_end}, poisoned page tail={block_size - written_end}, "
            f"poisoned rows in loaded tile={max(0, poison_in_tile)}"
        )
        if 0 < context % TILE_ROWS <= TILE_ROWS - STRIP:
            assert poison_in_tile > 0, "fixture lost the poison inside the final decode tile"
        physical = int(tables_host[request, page_index]) * block_size
        tail_slots.append(physical + torch.arange(live_end, block_size, dtype=torch.int32))
        exposed_contexts.append(-(-context // TILE_ROWS) * TILE_ROWS if poison_in_tile > 0 else context)
    return torch.cat(tail_slots), tuple(exposed_contexts)


def _write_prefill(writer, key, value, cache, golden_cache, tables_host, contexts, block_size, emulator):
    helpers, backend = _helpers(), _backend()
    slots = helpers.slots_from_page_table(tables_host, contexts, block_size).to(DEVICE)
    if writer == "batch":
        adapter = _adapter(backend.TurboQuantKernelStage.RESHAPE_AND_CACHE)
        plan = backend.turboquant_plan_strip_ingestion(slots.cpu().tolist(), block_size)
        assert not plan.waves, "batch-prefill regression must not route its ragged end through scatter"
        adapter.reshape_and_cache_prefill(key, value, *cache, slots, emulator, plan=plan)
    else:
        _adapter(backend.TurboQuantKernelStage.SCATTER).reshape_scatter_decode(key, value, *cache, slots, emulator)
    golden = _adapter()
    start, written_ends = 0, []
    for request, context in enumerate(contexts):
        written = -(-context // STRIP) * STRIP if writer == "batch" else context
        pad = written - context
        rows = slice(start, start + context)
        # Independently model the public batch adapter's documented last-row
        # padding, without borrowing its strip-run slicing or slot helpers.
        padded_key = torch.cat((key[rows], key[start + context - 1 : start + context].expand(pad, -1, -1)))
        padded_value = torch.cat((value[rows], value[start + context - 1 : start + context].expand(pad, -1, -1)))
        padded_slots = helpers.slots_from_page_table(tables_host[request : request + 1], [written], block_size).to(
            DEVICE
        )
        method = golden.reshape_and_cache_prefill if writer == "batch" else golden.reshape_scatter_decode
        method(padded_key, padded_value, *golden_cache, padded_slots, emulator)
        written_ends.append(written)
        start += context
    return written_ends


@requires_npu
class TestPrefillDecode:
    @pytest.mark.parametrize(
        "writer,payload,block_size,contexts",
        [
            pytest.param("batch", "random", 128, (64, 128), id="batch-random-aligned"),
            pytest.param("batch", "random", 128, (136, 383), id="batch-random-mixed"),
            pytest.param("batch", "random", 256, (257, 511), id="batch-random-page-tail"),
            pytest.param("batch", "random", 1024, (257,), id="batch-random-large-page"),
            pytest.param("scatter", "random", 128, (136, 383), id="scatter-random-mixed"),
            pytest.param("batch", "zero-live-spike", 128, (63,), id="zero-live-spike-fused"),
            pytest.param("batch", "zero-live-spike", 128, (136, 257), id="zero-live-spike-split"),
            pytest.param("batch", "constant-live-spike", 128, (63,), id="constant-live-spike-fused"),
            pytest.param("batch", "constant-live-spike", 128, (136, 257), id="constant-live-spike-split"),
        ],
    )
    def test_recycled_pages_mask_prefill_tail_through_first_decode_steps(
        self, writer, payload, block_size, contexts, monkeypatch
    ):
        """Ragged prefill -> immediate attention -> two KV appends and decode.

        Separate writer/tail assertions precede the mask check. A sanitized
        copy isolates dead-tail poison from live cache content; extending the
        context into that same poison proves the canary can affect attention.
        The zero-live-spike profile uses finite poison aligned with an all-ones
        query and requires bit-exact zero; exposing one spike must exceed 1.0.
        Constant live values additionally detect key-only mask leaks that steal
        softmax denominator weight even if the value mask suppresses the tail.
        """
        _require_ops(PREFILL_OP if writer == "batch" else SCATTER_OP, SCATTER_OP, FUSED_DECODE_OP, BUILD_TILING_OP)
        backend, helpers = _backend(), _helpers()
        finite_spikes = payload.endswith("-spike")
        tables_host, num_blocks = helpers.make_recycled_page_table(contexts, block_size)
        tables = tables_host.to(DEVICE)
        emulator = backend.TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
        if finite_spikes:
            cache = helpers.make_finite_spike_cache(
                num_blocks, block_size, NUM_KV_HEADS, HEAD_SIZE, emulator=emulator, device=DEVICE
            )
        else:
            cache = _caches(block_size, num_blocks)
        golden_cache, poison = _clone_cache(cache), _clone_cache(cache)
        golden, scatter, decode = (
            _adapter(),
            _adapter(backend.TurboQuantKernelStage.SCATTER),
            _adapter(backend.TurboQuantKernelStage.DECODE),
        )
        key, value = _activations(sum(contexts), seed=6000 + block_size)
        if finite_spikes:
            key.zero_()
            value.fill_(0.0 if payload == "zero-live-spike" else 1.0)
        written_ends = _write_prefill(
            writer, key, value, cache, golden_cache, tables_host, contexts, block_size, emulator
        )
        poison_mask = torch.ones_like(cache[2], dtype=torch.bool)
        written_slots = helpers.slots_from_page_table(tables_host, written_ends, block_size).to(DEVICE)
        poison_mask.view(-1, 2 * NUM_KV_HEADS)[written_slots.long()] = False
        generator = torch.Generator().manual_seed(7000 + block_size)
        writer_scale_errors = []
        for step in range(3):
            phase = "after-prefill" if step == 0 else f"decode-{step}"
            if step:
                new_key, new_value = _activations(len(contexts), seed=8000 + step)
                if finite_spikes:
                    new_key.zero_()
                    new_value.fill_(0.0 if payload == "zero-live-spike" else 1.0)
                slots = torch.tensor(
                    [
                        int(row[context // block_size]) * block_size + context % block_size
                        for row, context in zip(tables_host, contexts)
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                )
                for adapter, target in ((scatter, cache), (golden, golden_cache)):
                    adapter.reshape_scatter_decode(new_key, new_value, *target, slots, emulator)
                poison_mask.view(-1, 2 * NUM_KV_HEADS)[slots.long()] = False
                contexts = tuple(context + 1 for context in contexts)
                written_ends = [max(written, context) for written, context in zip(written_ends, contexts)]
            torch.npu.synchronize()
            helpers.assert_poison_scales_preserved(cache, poison, poison_mask)
            try:
                _compare(cache, golden_cache, writer_reference=True, poison_mask=poison_mask)
            except AssertionError as error:
                if not str(error).startswith("scale_cache:"):
                    raise
                # Poison survival has already passed. Keep live-scale failures
                # fatal, but let the independent tail-mask checks run first.
                writer_scale_errors.append(f"{phase}: {error}")
            tail_slots, exposed_contexts = _assert_prefill_tail(
                cache, poison, golden, tables_host, contexts, written_ends, block_size
            )
            if finite_spikes:
                live_slots = helpers.slots_from_page_table(tables_host, contexts, block_size).to(DEVICE)
                # Lloyd-Max's epsilon and zero-free codebook do not encode an
                # exact zero row. Zero the live scale records explicitly, after
                # validating the batch writer, to isolate the decode mask.
                # Restore all INVALID rows from the finite spike cache too,
                # including intentional strip padding, so any leaked row is loud.
                for target in (cache, golden_cache):
                    if payload == "zero-live-spike":
                        _scale_rows(target, live_slots, 0.0)
                    else:
                        target[2].view(-1, 2 * NUM_KV_HEADS)[live_slots.long(), :NUM_KV_HEADS] = 0.0
                    _copy_cache_rows(target, poison, tail_slots)
                # Expose just ONE previously masked spike, rather than an
                # entire tile, to prove a single-row leak is observable.
                exposed_contexts = tuple(context + 1 for context in contexts)
            sanitized = _clone_cache(cache) if step == 0 else None
            if sanitized is not None:
                _scale_rows(sanitized, tail_slots, 0.0)
            if finite_spikes and step == 0:
                underfilled = _clone_cache(cache)
                last_slots = helpers.slots_from_page_table(tables_host, contexts, block_size)
                last_indices = torch.tensor(contexts).cumsum(0) - 1
                _copy_cache_rows(underfilled, poison, last_slots[last_indices])
                wrong_tables = tables.clone()
                for request, context in enumerate(contexts):
                    wrong_tables[request, (context - 1) // block_size] = 0
            query = torch.randn(len(contexts), NUM_HEADS, HEAD_SIZE, generator=generator).to(DEVICE, torch.bfloat16)
            if finite_spikes:
                query.fill_(1.0)
            expected = [
                _reference_attention_fp32(
                    golden, emulator, golden_cache, tables[r : r + 1], query[r : r + 1], context, ATTENTION_SCALE
                )
                for r, context in enumerate(contexts)
            ]
            if payload == "zero-live-spike":
                assert all(bool((reference == 0).all()) for reference in expected), (
                    "zero-live reference must be exactly zero"
                )
            snapshots = (_clone_cache(cache), _clone_cache(golden_cache))
            for splits in PREFILL_SPLITS if step == 0 else APPEND_SPLITS:
                _set_decode_splits(monkeypatch, splits)
                buffers = _build_decode_tiling(contexts, block_size, tables.shape[1], splits)
                variants = [("poisoned-tail", cache, tables, contexts, False)]
                if step == 0 and splits in (1, 3):
                    variants.insert(0, ("sanitized-tail", sanitized, tables, contexts, False))
                # Sensitivity controls need one representative split count, not
                # another Cartesian sweep of the identical fault mechanisms.
                if finite_spikes and step == 0 and splits == 3:
                    variants.extend(
                        [
                            ("underfilled-live-row", underfilled, tables, contexts, True),
                            ("wrong-page", cache, wrong_tables, contexts, True),
                            ("tail-exposed-as-live", cache, tables, exposed_contexts, True),
                        ]
                    )
                for name, source, table, lengths, fault in variants:
                    launch_buffers = (
                        buffers
                        if lengths == contexts
                        else _build_decode_tiling(lengths, block_size, tables.shape[1], splits)
                    )
                    output = _launch_serving_decode(decode, emulator, source, table, lengths, query, launch_buffers)
                    for request, context in enumerate(contexts):
                        tag = f"{writer}/{payload}/{phase}/{name}: block={block_size} S={splits} request={request}"
                        if fault:
                            if finite_spikes and name in (
                                "wrong-page",
                                "tail-exposed-as-live",
                                "underfilled-live-row",
                            ):
                                assert torch.isfinite(output[request]).all(), f"{tag}: spike control must stay finite"
                                assert float(output[request].abs().max()) > 1.0, (
                                    f"{tag}: finite spike control is too weak"
                                )
                            with pytest.raises(AssertionError, match="attention"):
                                _assert_attention_close(tag, output[request : request + 1], expected[request], context)
                        else:
                            _assert_attention_close(tag, output[request : request + 1], expected[request], context)
            _compare(cache, snapshots[0])
            _compare(golden_cache, snapshots[1])
        assert not writer_scale_errors, "Writer scale contract failed after decode checks:\n" + "\n".join(
            writer_scale_errors
        )


@requires_npu
@pytest.mark.parametrize(
    "contexts,poison_request,profile",
    [
        pytest.param((129, 385), 0, "query-nan", id="B2-fused-query-nan"),
        pytest.param((65, 257, 513), 2, "query-inf", id="B3-split-query-inf"),
        pytest.param((65, 257, 513), 1, "kv-nan", id="B3-middle-kv-nan"),
        pytest.param((257, 385), 1, "kv-inf", id="B2-all-split-kv-inf"),
        pytest.param((129, 385), 1, "tail-spike", id="B2-masked-tail-spike"),
        pytest.param((65, 257, 513), 0, "live-spike", id="B3-fused-live-spike"),
    ],
)
def test_cross_request_poison_cannot_change_clean_siblings(contexts, poison_request, profile, monkeypatch):
    """One hostile request, clean random/zero siblings, and NaN launch scratch.

    Poisoned requests have no defined non-finite output contract: FP8 casts can
    neutralize their NaNs/Infs. Only clean siblings are required to match the
    independent reference. Finite live spikes additionally stay observable.
    """
    _require_ops(PREFILL_OP, FUSED_DECODE_OP, BUILD_TILING_OP)
    backend, helpers = _backend(), _helpers()
    block_size = 128
    tables_host, num_blocks = helpers.make_recycled_page_table(contexts, block_size)
    tables = tables_host.to(DEVICE)
    emulator = backend.TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    cache = _caches(block_size, num_blocks)
    golden_cache = _clone_cache(cache)
    key, value = _activations(sum(contexts), seed=9200 + len(contexts))
    _write_prefill("batch", key, value, cache, golden_cache, tables_host, contexts, block_size, emulator)
    golden = _adapter()
    query = torch.randn(len(contexts), NUM_HEADS, HEAD_SIZE, generator=torch.Generator().manual_seed(9400)).to(
        DEVICE, torch.bfloat16
    )
    clean_requests = [request for request in range(len(contexts)) if request != poison_request]
    if len(contexts) == 3:
        # Keep a bit-exact zero sibling as well as a random one in every B=3 case.
        request = clean_requests[-1]
        zero_slots = helpers.slots_from_page_table(tables_host[request : request + 1], [contexts[request]], block_size)
        for target in (cache, golden_cache):
            _scale_rows(target, zero_slots, 0.0)
    expected = {
        request: _reference_attention_fp32(
            golden,
            emulator,
            golden_cache,
            tables[request : request + 1],
            query[request : request + 1],
            contexts[request],
            ATTENTION_SCALE,
        )
        for request in clean_requests
    }
    row = tables_host[poison_request : poison_request + 1]
    live_slots = helpers.slots_from_page_table(row, [contexts[poison_request]], block_size)
    if profile.startswith("query-"):
        query[poison_request].fill_(float(profile.split("-")[1]))
        assert not torch.isfinite(query[poison_request]).all(), "query poison was not planted"
    elif profile.startswith("kv-"):
        _scale_rows(cache, live_slots, float(profile.split("-")[1]))
        assert not torch.isfinite(cache[2].view(-1, 2 * NUM_KV_HEADS)[live_slots.to(DEVICE).long()]).all()
    else:
        spikes = helpers.make_finite_spike_cache(
            num_blocks, block_size, NUM_KV_HEADS, HEAD_SIZE, emulator=emulator, device=DEVICE
        )
        if profile == "live-spike":
            _copy_cache_rows(cache, spikes, live_slots)
            query[poison_request].fill_(1.0)
        else:
            context = contexts[poison_request]
            logical = torch.arange(context, -(-context // TILE_ROWS) * TILE_ROWS)
            tail_slots = row[0, logical // block_size] * block_size + logical % block_size
            assert tail_slots.numel(), "masked finite-spike control needs a partial tile"
            _copy_cache_rows(cache, spikes, tail_slots)
            # The poisoned request must also ignore its OWN masked finite tail.
            clean_requests.append(poison_request)
            expected[poison_request] = _reference_attention_fp32(
                golden,
                emulator,
                golden_cache,
                tables[poison_request : poison_request + 1],
                query[poison_request : poison_request + 1],
                contexts[poison_request],
                ATTENTION_SCALE,
            )
    snapshot = _clone_cache(cache)
    decode = _adapter(backend.TurboQuantKernelStage.DECODE)
    for splits in PREFILL_SPLITS:
        _set_decode_splits(monkeypatch, splits)
        buffers = _build_decode_tiling(contexts, block_size, tables.shape[1], splits)
        output = _launch_serving_decode(decode, emulator, cache, tables, contexts, query, buffers)
        for request in clean_requests:
            tag = f"cross-request/{profile}: B={len(contexts)} S={splits} poison={poison_request} request={request}"
            _assert_attention_close(tag, output[request : request + 1], expected[request], contexts[request])
        if profile == "live-spike":
            assert torch.isfinite(output[poison_request]).all() and output[poison_request].abs().max() > 1.0
    _compare(cache, snapshot)


def main():
    if DEVICE is None:
        print("requires an Ascend NPU: nothing to run")
        return 0
    return pytest.main([__file__, "-sv"])


if __name__ == "__main__":
    sys.exit(main())
