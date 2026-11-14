# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live NPU gate: the two TurboQuant cache writers against the golden model.

**This needs an Ascend NPU and a build that registers the ops.** Every test
below skips otherwise, so collecting it on a host without one is harmless. It
was written on a host with no device and has therefore never executed; it is
the gate to run on the device box, not evidence that anything passed.

Section 5 is the fused decode's half of the gate: the kernel's one-launch
decode against the golden gather attended in fp32, at the serving geometry
(batch 1, 24 query heads over 4 KV heads -- group_heads 6, which no other
decode gate exercises -- head_size 256, block_size 128) with ragged tails at
8 and 31 rows. Its budget is a fraction of each row's max, not bit-exactness:
the decode rounds two operands to fp8 e4m3 by design.

What it establishes that no host-side test can: that the compiled kernels and
the PyTorch golden model put the *same bytes* in the cache. The codec is
designed so that this is bit-exact rather than approximate -- every scale is a
power of two taken from the fp32 exponent field and the single rounding is
round-half-to-even -- so a mismatch here is a real defect, not tolerance.

Run it standalone::

    python tests/ut/attention/test_turboquant_live_write_npu.py

or under pytest::

    pytest -sv tests/ut/attention/test_turboquant_live_write_npu.py

The geometry is only partly negotiable. Both writers are specialised at compile
time to ``head_size == 256`` and to a 16-token strip, and the batch writer
additionally to 4 KV heads. Both are built with ``TURBOQUANT_INGRESS_NZ=1``
now, so they need ``block_size`` to be a multiple of 64 -- exercised here at
``block_size == 64``, one NZ tile to a page -- and write the NZ-tiled cache the
fused decode reads. The scatter writer's NZ mode additionally drops the strip
read-modify-write: the NZ pairing is token-major, so a per-token write is a
few aligned bursts and nothing it does not own moves. The golden side of each
differential below is pinned to the layout the staged writer owns -- NZ_TILED
for both writers.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Geometry the kernels are specialised to; see the module docstring.
HEAD_SIZE = 256
NUM_KV_HEADS = 4
STRIP = 16
PREFILL_BLOCK_SIZE = 64
DECODE_BLOCK_SIZE = 64
NUM_BLOCKS = 8
NAMESPACE = "turboquant_ops"

PREFILL_OP = "npu_turboquant_reshape_and_cache_a8"
SCATTER_OP = "npu_turboquant_reshape_scatter_aiv"
FUSED_DECODE_OP = "npu_turboquant_fused_decode"
BUILD_TILING_OP = "npu_turboquant_build_decode_tiling"

# The serving geometry the fused decode differential below reproduces: batch 1,
# 24 query heads = 4 KV heads x 6 (group_heads 6), head_size 256, block_size 128.
# This is the shape the deployed engine runs, and group_heads 6 sits outside the
# geometry every other decode gate exercises (they are all group_heads 16).
SERVING_NUM_HEADS = 24
SERVING_NUM_KV_HEADS = 4
SERVING_GROUP_HEADS = SERVING_NUM_HEADS // SERVING_NUM_KV_HEADS
SERVING_BLOCK_SIZE = 128
SERVING_NUM_BLOCKS = 16
SERVING_SCALE = 1.0 / (HEAD_SIZE**0.5)


def _npu_device() -> torch.device | None:
    """The NPU, or ``None`` when this host has no usable one."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return None
    npu = getattr(torch, "npu", None)
    try:
        if npu is None or not npu.is_available() or npu.device_count() < 1:
            return None
    except Exception:
        return None
    return torch.device("npu:0")


def _op_registered(name: str) -> bool:
    """Whether the dispatcher has ``turboquant_ops::name``, loading the extension first.

    The registrations are static initialisers of ``vllm_ascend.vllm_ascend_C``,
    and nothing imports that module until ``load_turboquant_extension()`` runs
    -- the usual loader withholds it on this part (see the loader's docstring).
    A skipif probe that only reads ``torch.ops`` at collection time therefore
    reports every op as missing on exactly the ascend950 build this gate exists
    for, and the whole file skips. The probe is ``_is_op_registered`` rather
    than a ``getattr`` on ``torch.ops``: ``torch.ops.__getattr__`` manufactures
    a namespace object for any name, registered or not.

    Absence is still the normal answer on any SoC that is not ascend950: the
    whole translation unit is left out of the build, which is also what makes
    the Python side fall back to its golden model.
    """
    try:
        from vllm_ascend.attention.turboquant_v1 import _is_op_registered, load_turboquant_extension

        load_turboquant_extension()
        return _is_op_registered(NAMESPACE, name)
    except Exception:
        # A skip decision must never be a collection error: any failure here
        # means the kernels cannot be reached from this host, which is a skip.
        return False


DEVICE = _npu_device()
requires_npu = pytest.mark.skipif(DEVICE is None, reason="no Ascend NPU on this host")


def _requires_op(name: str):
    return pytest.mark.skipif(
        not _op_registered(name),
        reason=f"torch.ops.{NAMESPACE}.{name} is not registered; this needs an ascend950 build",
    )


# ---------------------------------------------------------------------------
# Fixtures: two identical caches, and the adapters that write them
# ---------------------------------------------------------------------------


def _import_backend():
    from vllm_ascend.attention.turboquant_v1 import (
        TurboQuantEmulator,
        TurboQuantKernelStage,
        TurboQuantOpsAdapter,
        TurboQuantOpsConfig,
        TurboQuantValueLayout,
    )

    return (
        TurboQuantEmulator,
        TurboQuantKernelStage,
        TurboQuantOpsAdapter,
        TurboQuantOpsConfig,
        TurboQuantValueLayout,
    )


def _caches(block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One three-plane cache, zeroed.

    Zeroed rather than poisoned on purpose: both writers leave the lanes they do
    not own alone, so an untouched lane reading zero on both sides is part of
    what the comparison checks.
    """
    packed = (NUM_BLOCKS, block_size, NUM_KV_HEADS, HEAD_SIZE // 2)
    scales = (2, NUM_BLOCKS, NUM_KV_HEADS, block_size)
    return (
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(scales, dtype=torch.float32, device=device),
    )


def _activations(num_tokens: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, H, D]`` bf16 K and V, which is the only dtype either writer takes."""
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    return key.to(device=device, dtype=torch.bfloat16), value.to(device=device, dtype=torch.bfloat16)


def _adapters(stage):
    """``(kernel_adapter, golden_adapter)`` over the same configuration otherwise.

    The golden one is pinned to the staged writer's layout explicitly, because
    with no ingestion kernel engaged ``value_layout()`` would otherwise report
    the golden model's own TOKEN_MAJOR cache -- and the point here is to write
    the same layout both ways: NZ_TILED, which either staged ingestion writer
    reports (``TURBOQUANT_INGRESS_NZ=1`` builds both of them as NZ ingresses).
    """
    _, _, TurboQuantOpsAdapter, TurboQuantOpsConfig, TurboQuantValueLayout = _import_backend()
    kernel = TurboQuantOpsAdapter(TurboQuantOpsConfig(ops_namespace=NAMESPACE, kernel_stages=frozenset({stage})))
    golden = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(),
            value_layout=kernel.value_layout(),
        )
    )
    assert golden.value_layout() is kernel.value_layout()
    return kernel, golden


def _slot_mapping_prefill(num_strips: int, generator: torch.Generator) -> torch.Tensor:
    """Whole 16-token strips at random distinct STRIP slots, consecutive within a strip.

    The batch writer reads one slot per strip on the scalar unit and derives both
    the page and the strip inside it from that slot, so within a strip the slots
    must be ``slot_mapping[0] + i`` with ``slot_mapping[0]`` a multiple of 16.
    Which strip of which page it lands on is free.

    Randomising over strips rather than over pages is the point at
    ``block_size 64``: several strips of one page get picked, which is exactly
    what the old writer got wrong -- every strip of a page computed the same
    base and overwrote the others.
    """
    total_strips = NUM_BLOCKS * PREFILL_BLOCK_SIZE // STRIP
    chosen = torch.randperm(total_strips, generator=generator)[:num_strips]
    within = torch.arange(STRIP, dtype=torch.int32)
    return (chosen.to(torch.int32).unsqueeze(1) * STRIP + within).reshape(-1)


def _slot_mapping_decode(num_tokens: int, generator: torch.Generator) -> torch.Tensor:
    """Arbitrary distinct slots: the scatter writer assumes nothing about them."""
    total = NUM_BLOCKS * DECODE_BLOCK_SIZE
    return torch.randperm(total, generator=generator)[:num_tokens].to(torch.int32)


def _compare(kernel_caches, golden_caches) -> None:
    names = ("key_cache", "value_cache", "scale_cache")
    for name, written, expected in zip(names, kernel_caches, golden_caches):
        a = written.cpu()
        b = expected.cpu()
        if torch.equal(a, b):
            continue
        differing = (a != b).sum().item()
        first = int((a != b).reshape(-1).nonzero()[0].item())
        raise AssertionError(
            f"{name}: the kernel and the golden model disagree in {differing} of {a.numel()} "
            f"elements, first at flat index {first} "
            f"(kernel {a.reshape(-1)[first]!r} vs golden {b.reshape(-1)[first]!r}). "
            "The codec is bit-exact by construction, so this is a defect rather than tolerance."
        )


# ---------------------------------------------------------------------------
# 1. The prefill batch writer, block_size 64, NZ-tiled ingress
# ---------------------------------------------------------------------------


@requires_npu
@_requires_op(PREFILL_OP)
def test_the_batch_writer_fills_every_strip_of_one_page() -> None:
    """All four strips of a single page, which is the in-page offset's regression.

    Before `run_vector_core` derived `strip_in_block`, every one of these four
    blocks computed the same page base and wrote over the previous one: the cache
    came out holding only the last strip, with every other byte of the page
    still zero and no check anywhere failing. Under the NZ ingress each strip
    owns 16 tile rows of every (kv head, column group) of the page's tile, so
    "written" is asserted through the NZ address function, not a row slice.
    """
    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.RESHAPE_AND_CACHE)

    strips_per_page = PREFILL_BLOCK_SIZE // STRIP
    key, value = _activations(PREFILL_BLOCK_SIZE, DEVICE, seed=1500)
    # Page 3, every strip of it, in order.
    slot_mapping = (3 * PREFILL_BLOCK_SIZE + torch.arange(PREFILL_BLOCK_SIZE, dtype=torch.int32)).to(DEVICE)

    on_npu = _caches(PREFILL_BLOCK_SIZE, DEVICE)
    on_golden = _caches(PREFILL_BLOCK_SIZE, DEVICE)
    kernel_adapter.reshape_and_cache_prefill(key, value, *on_npu, slot_mapping, emulator)
    golden_adapter.reshape_and_cache_prefill(key, value, *on_golden, slot_mapping, emulator)
    torch.npu.synchronize()
    _compare(on_npu, on_golden)

    # And the whole page really was written, not just one strip of it: gather
    # each strip's bytes through the NZ address function and check they moved.
    flat = on_npu[0].view(-1).cpu()
    packed = HEAD_SIZE // 2
    for strip in range(strips_per_page):
        slots = 3 * PREFILL_BLOCK_SIZE + strip * STRIP + torch.arange(STRIP, dtype=torch.int64)
        offsets = kernel_adapter._nz_byte_offsets(slots, NUM_KV_HEADS, packed)
        rows = flat[offsets.reshape(-1)]
        assert int(rows.abs().sum()) > 0, (
            f"strip {strip} of the page is still zero: the in-page offset did not take effect"
        )


@requires_npu
@_requires_op(PREFILL_OP)
@pytest.mark.parametrize("num_strips", [1, 4])
def test_the_batch_writer_matches_the_golden_model(num_strips: int) -> None:
    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.RESHAPE_AND_CACHE)

    key, value = _activations(num_strips * STRIP, DEVICE, seed=1000 + num_strips)
    slot_mapping = _slot_mapping_prefill(num_strips, torch.Generator().manual_seed(7)).to(DEVICE)

    on_npu = _caches(PREFILL_BLOCK_SIZE, DEVICE)
    on_golden = _caches(PREFILL_BLOCK_SIZE, DEVICE)
    kernel_adapter.reshape_and_cache_prefill(key, value, *on_npu, slot_mapping, emulator)
    golden_adapter.reshape_and_cache_prefill(key, value, *on_golden, slot_mapping, emulator)
    torch.npu.synchronize()
    _compare(on_npu, on_golden)


# ---------------------------------------------------------------------------
# 2. The decode scatter writer, block_size 64
# ---------------------------------------------------------------------------


@requires_npu
@_requires_op(SCATTER_OP)
@pytest.mark.parametrize("num_tokens", [1, 3, 17])
def test_the_scatter_writer_matches_the_golden_model(num_tokens: int) -> None:
    """Arbitrary slots at block_size 64, which is what 5984a47fa allowed.

    This writer used to be the one whose value plane was a read-modify-write;
    under the NZ ingress both planes are pure writes, so the interesting part
    is unchanged in kind: the bytes it must NOT disturb. Both caches start
    zeroed and every untouched byte has to still be zero on both sides.
    """
    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.SCATTER)

    key, value = _activations(num_tokens, DEVICE, seed=2000 + num_tokens)
    slot_mapping = _slot_mapping_decode(num_tokens, torch.Generator().manual_seed(11)).to(DEVICE)

    on_npu = _caches(DECODE_BLOCK_SIZE, DEVICE)
    on_golden = _caches(DECODE_BLOCK_SIZE, DEVICE)
    kernel_adapter.reshape_scatter_decode(key, value, *on_npu, slot_mapping, emulator)
    golden_adapter.reshape_scatter_decode(key, value, *on_golden, slot_mapping, emulator)
    torch.npu.synchronize()
    _compare(on_npu, on_golden)


@requires_npu
@_requires_op(SCATTER_OP)
def test_the_scatter_writer_accumulates_over_steps() -> None:
    """Token by token into one tile, which is what a real decode does.

    The NZ pairing is token-major, so a token's bytes are wholly its own and a
    later step's write cannot clobber an earlier one's -- the property the
    strip cache needed the read-modify-write for. A per-token write that
    addressed the wrong bytes would still pass the single-step test above.
    """
    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.SCATTER)

    key, value = _activations(STRIP, DEVICE, seed=3000)
    slots = torch.arange(STRIP, dtype=torch.int32, device=DEVICE)

    on_npu = _caches(DECODE_BLOCK_SIZE, DEVICE)
    on_golden = _caches(DECODE_BLOCK_SIZE, DEVICE)
    for token in range(STRIP):
        one = slice(token, token + 1)
        kernel_adapter.reshape_scatter_decode(key[one], value[one], *on_npu, slots[one], emulator)
        golden_adapter.reshape_scatter_decode(key[one], value[one], *on_golden, slots[one], emulator)
    torch.npu.synchronize()
    _compare(on_npu, on_golden)


# ---------------------------------------------------------------------------
# 3. Read-back of an NPU-written cache
# ---------------------------------------------------------------------------


@requires_npu
@_requires_op(SCATTER_OP)
def test_the_golden_gather_reads_the_npu_written_cache() -> None:
    """Write with the kernel, read with ``_golden_gather_pages_nz``.

    Two assertions, and they are different in kind:

    * against the emulator's own round trip, EXACTLY. Both sides run the same
      codec, so the write-then-read chain must be lossless relative to it. This
      is the one that catches a layout error.
    * against the original activations, within an INT4 tolerance. That residual
      is the codec's quantisation error and nothing to do with the cache.
    """
    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    kernel_adapter, _ = _adapters(TurboQuantKernelStage.SCATTER)

    num_tokens = DECODE_BLOCK_SIZE
    key, value = _activations(num_tokens, DEVICE, seed=4000)
    slots = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    caches = _caches(DECODE_BLOCK_SIZE, DEVICE)
    kernel_adapter.reshape_scatter_decode(key, value, *caches, slots, emulator)
    torch.npu.synchronize()

    block_tables = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    packed_key, key_scales, packed_value, value_scales = kernel_adapter.gather(*caches, block_tables)

    for name, activations, packed, scales in (
        ("key", key, packed_key, key_scales),
        ("value", value, packed_value, value_scales),
    ):
        dense = kernel_adapter.dequantize_cache(packed, scales, emulator, torch.float32)[0]
        reference = emulator.decode(*emulator.encode(activations))
        assert torch.equal(dense.cpu(), reference.cpu()), (
            f"{name}: the NPU-written cache did not read back as the codec's own round trip; "
            "the layout the reader assumes and the one the kernel wrote disagree"
        )

        original = activations.to(torch.float32)
        cosine = torch.nn.functional.cosine_similarity(
            dense.reshape(-1).cpu(), original.reshape(-1).cpu(), dim=0
        ).item()
        print(f"  {name}: cosine against the original activations = {cosine:.6f}")
        assert cosine > 0.98, f"{name}: INT4 round trip lost too much, cosine {cosine}"


# ---------------------------------------------------------------------------
# 4. TURBOQUANT_KERNEL_STAGES=scatter end to end
# ---------------------------------------------------------------------------


@requires_npu
@_requires_op(SCATTER_OP)
def test_the_scatter_stage_reads_back_without_the_decode_kernel() -> None:
    """The configuration the integration actually ships: writes on the NPU, decode golden.

    ``TURBOQUANT_KERNEL_STAGES=scatter`` is read from the environment here rather
    than constructed, so this also covers the env parsing and the derived
    layout. The fused decode stage is off, so ``fused_decode`` returns ``None``
    and the backend falls through to the golden gather -- which must read the
    cache the kernel wrote. No exception, and the numbers hold.
    """
    TurboQuantEmulator, _, TurboQuantOpsAdapter, _, TurboQuantValueLayout = _import_backend()
    previous = os.environ.get("TURBOQUANT_KERNEL_STAGES")
    os.environ["TURBOQUANT_KERNEL_STAGES"] = "scatter"
    os.environ["TURBOQUANT_OPS_NAMESPACE"] = NAMESPACE
    try:
        adapter = TurboQuantOpsAdapter()
        assert adapter.value_layout() is TurboQuantValueLayout.NZ_TILED
        emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)

        num_tokens = STRIP
        key, value = _activations(num_tokens, DEVICE, seed=5000)
        slots = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        caches = _caches(DECODE_BLOCK_SIZE, DEVICE)
        adapter.reshape_scatter_decode(key, value, *caches, slots, emulator)
        torch.npu.synchronize()

        fused = adapter.fused_decode(
            query=torch.randn(1, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.float16),
            key_cache=caches[0],
            value_cache=caches[1],
            scale_cache=caches[2],
            block_tables=torch.zeros((1, 1), dtype=torch.int32, device=DEVICE),
            context_lens=torch.full((1,), num_tokens, dtype=torch.int32, device=DEVICE),
            emulator=emulator,
            num_heads=NUM_KV_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            scale=1.0 / (HEAD_SIZE**0.5),
            output=torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.float16),
            # The decode stage is off, so the call must decline before touching
            # these: empty dummies, because the buffers are required positional
            # arguments since the metadata builder took ownership of them.
            tiling=torch.empty(0, dtype=torch.int32, device=DEVICE),
            workspace=torch.empty(0, dtype=torch.float32, device=DEVICE),
            query_rot=torch.empty(0, dtype=torch.float32, device=DEVICE),
        )
        assert fused is None, "the decode stage is off, so the fused decode must decline"

        packed_key, key_scales, _, _ = adapter.gather(*caches, torch.zeros((1, 1), dtype=torch.int32, device=DEVICE))
        dense = adapter.dequantize_cache(packed_key, key_scales, emulator, torch.float32)[0]
        assert torch.equal(dense[:num_tokens].cpu(), emulator.decode(*emulator.encode(key)).cpu())
    finally:
        if previous is None:
            os.environ.pop("TURBOQUANT_KERNEL_STAGES", None)
        else:
            os.environ["TURBOQUANT_KERNEL_STAGES"] = previous


# ---------------------------------------------------------------------------
# 5. The fused decode against the golden attention, at the serving geometry
# ---------------------------------------------------------------------------
#
# group_heads 6 (24 query heads over 4 KV heads) is the geometry the deployed
# engine runs; every other decode gate exercises group_heads 16. The reference
# is the same cache read back through the golden gather, dequantised and
# attended in fp32 -- the fallback path the backend takes with the decode stage
# off -- so a disagreement is the decode kernel, not the cache.
#
# The budget is not bit-exactness: the decode rounds two operands to fp8 e4m3
# (the query and the softmax probabilities), which an fp32 reference does not
# reproduce. Calibrated on CAModel against the fp64 golden: 0.3-0.4% of the row
# max on aligned contexts, up to 3.6% on a ragged 8-row tail. The gate is
# therefore a fraction of each row's max magnitude, with the worst channels
# printed either way so a failure names its coordinates.
DECODE_ROW_MAX_BUDGET = 0.10
DECODE_COSINE_FLOOR = 0.999


def _decode_and_golden_adapters():
    """The decode-stage adapter, and the golden adapter pinned to the NZ cache.

    Not ``_adapters(DECODE)``: with no ingestion stage engaged,
    ``value_layout()`` falls back to the config default ``TOKEN_MAJOR``, which
    would pin the golden side to a cache layout the fused decode does not read.
    The serving cache is NZ (both writers are ``TURBOQUANT_INGRESS_NZ=1``
    builds), so the golden side is pinned to NZ_TILED explicitly.
    """
    _, TurboQuantKernelStage, TurboQuantOpsAdapter, TurboQuantOpsConfig, TurboQuantValueLayout = _import_backend()
    decode = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(ops_namespace=NAMESPACE, kernel_stages=frozenset({TurboQuantKernelStage.DECODE}))
    )
    golden = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace=NAMESPACE,
            kernel_stages=frozenset(),
            value_layout=TurboQuantValueLayout.NZ_TILED,
        )
    )
    assert golden.value_layout() is TurboQuantValueLayout.NZ_TILED
    return decode, golden


def _serving_caches(num_blocks: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One three-plane cache at the serving geometry, zeroed (see ``_caches``)."""
    packed = (num_blocks, SERVING_BLOCK_SIZE, SERVING_NUM_KV_HEADS, HEAD_SIZE // 2)
    scales = (2, num_blocks, SERVING_NUM_KV_HEADS, SERVING_BLOCK_SIZE)
    return (
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(scales, dtype=torch.float32, device=device),
    )


def _serving_activations(context: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[context, 4, 256]`` bf16 K and V at the serving KV head count."""
    generator = torch.Generator().manual_seed(seed)
    shape = (context, SERVING_NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    return key.to(device=device, dtype=torch.bfloat16), value.to(device=device, dtype=torch.bfloat16)


def _build_decode_tiling(
    context_lens_host: torch.Tensor,
    block_size: int,
    max_blocks_per_req: int,
    num_splits: int,
    heads_per_task: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three launch buffers, with ``tiling`` filled by the builder op.

    Exactly what ``AscendTurboQuantMetadataBuilder`` does per step: the op owns
    the descriptor arithmetic, Python only sizes the buffers.
    ``context_lens_host`` must stay on the host -- ``build_decode_tiling`` is
    host C++ and the binding refuses a device tensor.
    """
    from vllm_ascend.attention.turboquant_v1 import (
        TURBOQUANT_DEFAULT_NAMESPACE,
        TURBOQUANT_OP_BUILD_DECODE_TILING,
        turboquant_decode_query_rot_floats,
        turboquant_decode_workspace_floats,
        turboquant_tiling_upper_bound_words,
    )

    namespace = getattr(torch.ops, TURBOQUANT_DEFAULT_NAMESPACE)
    entry = getattr(namespace, TURBOQUANT_OP_BUILD_DECODE_TILING)
    tiling = torch.zeros(turboquant_tiling_upper_bound_words(batch, num_splits), dtype=torch.int32, device=DEVICE)
    workspace = torch.zeros(
        turboquant_decode_workspace_floats(batch, SERVING_NUM_HEADS, HEAD_SIZE, num_splits),
        dtype=torch.float32,
        device=DEVICE,
    )
    query_rot = torch.zeros(
        turboquant_decode_query_rot_floats(batch, SERVING_NUM_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device=DEVICE,
    )
    entry(
        context_lens_host,
        tiling,
        SERVING_NUM_HEADS,
        SERVING_NUM_KV_HEADS,
        HEAD_SIZE,
        block_size,
        max_blocks_per_req,
        num_splits,
        heads_per_task,
    )
    return tiling, workspace, query_rot


def _golden_attention_fp32(
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

    q = query[0].float()  # [N, D]
    outs = []
    for head in range(SERVING_NUM_HEADS):
        kv = head // SERVING_GROUP_HEADS
        s = (key_dense[:, kv, :] @ q[head]) * scale  # [S]
        p = torch.softmax(s, dim=-1)
        outs.append(p @ value_dense[:, kv, :])  # [D]
    return torch.stack(outs).unsqueeze(0)  # [1, N, D]


def _assert_attention_close(tag: str, got: torch.Tensor, expected: torch.Tensor, context: int) -> None:
    """Row-relative budget with the worst channels printed either way.

    The decode kernel rounds its query and probability operands to fp8 e4m3; the
    gate is the fraction of each row's max magnitude that rounding can explain
    (CAModel-calibrated, see the section header). A failure prints the worst
    eight channels by absolute residual, with the got/expected values and each
    row's scale ratio, so the coordinates are in the log before the assert.
    """
    got32 = got[0].float().cpu()
    exp32 = expected[0].float().cpu()
    diff = (got32 - exp32).abs()
    row_max = exp32.abs().amax(dim=-1)  # [N]
    worst_row = int((diff.amax(dim=-1) / row_max.clamp(min=1e-30)).argmax().item())
    worst_abs = float(diff.amax().item())
    worst_ratio = float((diff.amax(dim=-1) / row_max.clamp(min=1e-30)).max().item())
    scale_ratio = float((row_max / got32.abs().amax(dim=-1).clamp(min=1e-30))[worst_row].item())
    flat = (diff / row_max.clamp(min=1e-30).unsqueeze(-1)).reshape(-1)
    top = torch.topk(flat, min(8, flat.numel()))
    print(
        f"  {tag}: context {context}, worst abs {worst_abs:.4g}, worst {worst_ratio:.4f} of row max "
        f"(budget {DECODE_ROW_MAX_BUDGET}), worst row {worst_row}, scale ratio {scale_ratio:.6f}"
    )
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        head, channel = divmod(index, HEAD_SIZE)
        print(
            f"    channel ({head}, {channel}): got {got32[head, channel]:.6g} "
            f"expected {exp32[head, channel]:.6g}, {value:.4f} of row max"
        )
    cosine = torch.nn.functional.cosine_similarity(got32.reshape(-1), exp32.reshape(-1), dim=0).item()
    print(f"  {tag}: cosine {cosine:.6f} (floor {DECODE_COSINE_FLOOR})")
    assert worst_ratio <= DECODE_ROW_MAX_BUDGET, (
        f"{tag}: the fused decode is off by {worst_ratio:.4f} of a row max at context {context} "
        f"(worst row {worst_row}, scale ratio {scale_ratio:.6f}); the fp8 operand rounding "
        f"explains at most ~0.04. The worst channels are listed above."
    )
    assert cosine >= DECODE_COSINE_FLOOR, f"{tag}: cosine {cosine:.6f} against the golden attention"


@requires_npu
@_requires_op(FUSED_DECODE_OP)
@pytest.mark.parametrize(
    "context",
    [
        64,  # one full NZ tile, nothing ragged
        SERVING_BLOCK_SIZE,  # one whole page, aligned
        200,  # two pages, an 8-row tail -- the CAModel worst case
        1055,  # nine pages, a 31-row tail
    ],
)
def test_the_fused_decode_matches_the_golden_attention_serving_geometry(context: int) -> None:
    """Batch 1, 24 query heads over 4 KV heads, on a golden-written cache.

    The cache is written by the golden model pinned to the NZ layout, so its
    content is canonical by construction and only the decode kernel is under
    test. The query is the serving dtype (bf16); the reference is the golden
    gather attended in fp32.
    """
    from vllm_ascend.attention.turboquant_v1 import TURBOQUANT_DECODE_SPLITS, turboquant_decode_heads_per_task

    TurboQuantEmulator, _, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    decode_adapter, golden_adapter = _decode_and_golden_adapters()

    pages = -(-context // SERVING_BLOCK_SIZE)
    key, value = _serving_activations(context, DEVICE, seed=6000 + context)
    slots = torch.arange(context, dtype=torch.int32, device=DEVICE)
    caches = _serving_caches(SERVING_NUM_BLOCKS, DEVICE)
    golden_adapter.reshape_scatter_decode(key, value, *caches, slots, emulator)
    torch.npu.synchronize()

    block_tables = torch.arange(pages, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    context_lens = torch.full((1,), context, dtype=torch.int32, device=DEVICE)
    generator = torch.Generator().manual_seed(7000 + context)
    query = torch.randn(1, SERVING_NUM_HEADS, HEAD_SIZE, generator=generator, dtype=torch.float32).to(
        device=DEVICE, dtype=torch.bfloat16
    )
    output = torch.zeros(1, SERVING_NUM_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)

    splits = TURBOQUANT_DECODE_SPLITS
    heads_per_task = turboquant_decode_heads_per_task(SERVING_NUM_HEADS, SERVING_NUM_KV_HEADS)
    tiling, workspace, query_rot = _build_decode_tiling(
        context_lens.cpu(), SERVING_BLOCK_SIZE, pages, splits, heads_per_task, batch=1
    )
    fused = decode_adapter.fused_decode(
        query=query,
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=block_tables,
        context_lens=context_lens,
        emulator=emulator,
        num_heads=SERVING_NUM_HEADS,
        num_kv_heads=SERVING_NUM_KV_HEADS,
        scale=SERVING_SCALE,
        output=output,
        tiling=tiling,
        workspace=workspace,
        query_rot=query_rot,
    )
    torch.npu.synchronize()
    assert fused is not None, "the decode stage is on and the op is registered, so it must run"

    expected = _golden_attention_fp32(golden_adapter, emulator, caches, block_tables, query, context, SERVING_SCALE)
    _assert_attention_close("fused decode", output, expected, context)


@requires_npu
@_requires_op(FUSED_DECODE_OP)
@_requires_op(SCATTER_OP)
@pytest.mark.parametrize("context", [SERVING_BLOCK_SIZE, 200, 1055])
def test_the_fused_decode_reads_the_kernel_written_cache_serving_geometry(context: int) -> None:
    """The whole serving chain: the NPU scatter writer's cache, decoded by the kernel.

    The writer differential above already pins the kernel writer bit for bit to
    the golden model, so the reference is still the golden gather over the same
    bytes. What this adds over the golden-written variant is the write path a
    live decode actually produces, token by token through the scatter writer.
    """
    from vllm_ascend.attention.turboquant_v1 import TURBOQUANT_DECODE_SPLITS, turboquant_decode_heads_per_task

    TurboQuantEmulator, TurboQuantKernelStage, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    decode_adapter, golden_adapter = _decode_and_golden_adapters()
    scatter_adapter, _ = _adapters(TurboQuantKernelStage.SCATTER)

    pages = -(-context // SERVING_BLOCK_SIZE)
    key, value = _serving_activations(context, DEVICE, seed=8000 + context)
    caches = _serving_caches(SERVING_NUM_BLOCKS, DEVICE)
    for token in range(context):
        one = slice(token, token + 1)
        slot = torch.tensor([token], dtype=torch.int32, device=DEVICE)
        scatter_adapter.reshape_scatter_decode(key[one], value[one], *caches, slot, emulator)
    torch.npu.synchronize()

    block_tables = torch.arange(pages, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    context_lens = torch.full((1,), context, dtype=torch.int32, device=DEVICE)
    generator = torch.Generator().manual_seed(9000 + context)
    query = torch.randn(1, SERVING_NUM_HEADS, HEAD_SIZE, generator=generator, dtype=torch.float32).to(
        device=DEVICE, dtype=torch.bfloat16
    )
    output = torch.zeros(1, SERVING_NUM_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)

    splits = TURBOQUANT_DECODE_SPLITS
    heads_per_task = turboquant_decode_heads_per_task(SERVING_NUM_HEADS, SERVING_NUM_KV_HEADS)
    tiling, workspace, query_rot = _build_decode_tiling(
        context_lens.cpu(), SERVING_BLOCK_SIZE, pages, splits, heads_per_task, batch=1
    )
    fused = decode_adapter.fused_decode(
        query=query,
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=block_tables,
        context_lens=context_lens,
        emulator=emulator,
        num_heads=SERVING_NUM_HEADS,
        num_kv_heads=SERVING_NUM_KV_HEADS,
        scale=SERVING_SCALE,
        output=output,
        tiling=tiling,
        workspace=workspace,
        query_rot=query_rot,
    )
    torch.npu.synchronize()
    assert fused is not None, "the decode stage is on and the op is registered, so it must run"

    expected = _golden_attention_fp32(golden_adapter, emulator, caches, block_tables, query, context, SERVING_SCALE)
    _assert_attention_close("kernel-written fused decode", output, expected, context)


@requires_npu
@_requires_op(FUSED_DECODE_OP)
def test_the_fused_decode_handles_two_requests_serving_geometry() -> None:
    """Batch 2 with unequal contexts, [128, 200]: the descriptor's token dimension.

    The per-token descriptors are independent, so this is the cheapest rung that
    proves a second request neither shifts the first nor inherits its tail.
    """
    from vllm_ascend.attention.turboquant_v1 import TURBOQUANT_DECODE_SPLITS, turboquant_decode_heads_per_task

    TurboQuantEmulator, _, _, _, _ = _import_backend()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=DEVICE)
    decode_adapter, golden_adapter = _decode_and_golden_adapters()

    contexts = [SERVING_BLOCK_SIZE, 200]
    batch = len(contexts)
    caches = _serving_caches(SERVING_NUM_BLOCKS, DEVICE)
    block_rows = []
    next_block = 0
    for request, context in enumerate(contexts):
        pages = -(-context // SERVING_BLOCK_SIZE)
        key, value = _serving_activations(context, DEVICE, seed=10000 + context)
        slots = next_block * SERVING_BLOCK_SIZE + torch.arange(context, dtype=torch.int32, device=DEVICE)
        golden_adapter.reshape_scatter_decode(key, value, *caches, slots, emulator)
        block_rows.append(torch.arange(next_block, next_block + pages, dtype=torch.int32))
        next_block += pages
    max_pages = max(row.numel() for row in block_rows)
    block_tables = torch.full((batch, max_pages), -1, dtype=torch.int32, device=DEVICE)
    for request, row in enumerate(block_rows):
        block_tables[request, : row.numel()] = row.to(DEVICE)
    context_lens = torch.tensor(contexts, dtype=torch.int32, device=DEVICE)
    torch.npu.synchronize()

    generator = torch.Generator().manual_seed(11000)
    query = torch.randn(batch, SERVING_NUM_HEADS, HEAD_SIZE, generator=generator, dtype=torch.float32).to(
        device=DEVICE, dtype=torch.bfloat16
    )
    output = torch.zeros(batch, SERVING_NUM_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)

    splits = TURBOQUANT_DECODE_SPLITS
    heads_per_task = turboquant_decode_heads_per_task(SERVING_NUM_HEADS, SERVING_NUM_KV_HEADS)
    tiling, workspace, query_rot = _build_decode_tiling(
        context_lens.cpu(), SERVING_BLOCK_SIZE, max_pages, splits, heads_per_task, batch=batch
    )
    fused = decode_adapter.fused_decode(
        query=query,
        key_cache=caches[0],
        value_cache=caches[1],
        scale_cache=caches[2],
        block_tables=block_tables,
        context_lens=context_lens,
        emulator=emulator,
        num_heads=SERVING_NUM_HEADS,
        num_kv_heads=SERVING_NUM_KV_HEADS,
        scale=SERVING_SCALE,
        output=output,
        tiling=tiling,
        workspace=workspace,
        query_rot=query_rot,
    )
    torch.npu.synchronize()
    assert fused is not None, "the decode stage is on and the op is registered, so it must run"

    for request, context in enumerate(contexts):
        expected = _golden_attention_fp32(
            golden_adapter,
            emulator,
            caches,
            block_tables[request : request + 1],
            query[request : request + 1],
            context,
            SERVING_SCALE,
        )
        _assert_attention_close(f"fused decode request {request}", output[request : request + 1], expected, context)


# ---------------------------------------------------------------------------


def main() -> int:
    """Standalone entry point, so this can be run on the device box directly."""
    if DEVICE is None:
        print("no Ascend NPU on this host: nothing to run")
        return 0
    missing = [name for name in (PREFILL_OP, SCATTER_OP) if not _op_registered(name)]
    if missing:
        print(f"these ops are not registered, so this build cannot be gated: {missing}")
        return 0
    return pytest.main([__file__, "-sv"])


if __name__ == "__main__":
    sys.exit(main())
