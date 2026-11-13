# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live NPU gate: the two TurboQuant cache writers against the golden model.

**This needs an Ascend NPU and a build that registers the ops.** Every test
below skips otherwise, so collecting it on a host without one is harmless. It
was written on a host with no device and has therefore never executed; it is
the gate to run on the device box, not evidence that anything passed.

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


def _op(name: str):
    """The registered entry point, or ``None``.

    Absence is the normal case on any SoC that is not ascend950: the whole
    translation unit is left out of the build, which is also what makes the
    Python side fall back to its golden model.
    """
    namespace = getattr(torch.ops, NAMESPACE, None)
    return getattr(namespace, name, None) if namespace is not None else None


DEVICE = _npu_device()
requires_npu = pytest.mark.skipif(DEVICE is None, reason="no Ascend NPU on this host")


def _requires_op(name: str):
    return pytest.mark.skipif(
        _op(name) is None,
        reason=f"torch.ops.{NAMESPACE}.{name} is not registered; this needs an ascend950 build",
    )


# ---------------------------------------------------------------------------
# Fixtures: two identical caches, and the adapters that write them
# ---------------------------------------------------------------------------


def _import_backend():
    from vllm_ascend.attention.turboquant_dry_run import (
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


def _adapters(stage, layout):
    """``(kernel_adapter, golden_adapter)`` over the same configuration otherwise.

    The golden one is pinned to the staged writer's layout explicitly, because
    with no ingestion kernel engaged ``value_layout()`` would otherwise report
    the golden model's own TOKEN_MAJOR cache -- and the point here is to write
    the same layout both ways: NZ_TILED for the batch writer (the NZ ingress),
    CHANNEL_MAJOR for the scatter writer (still a strip kernel).
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
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.RESHAPE_AND_CACHE, None)

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
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.RESHAPE_AND_CACHE, None)

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
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.SCATTER, None)

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
    kernel_adapter, golden_adapter = _adapters(TurboQuantKernelStage.SCATTER, None)

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
    kernel_adapter, _ = _adapters(TurboQuantKernelStage.SCATTER, None)

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
        assert cosine > 0.99, f"{name}: INT4 round trip lost too much, cosine {cosine}"


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


def main() -> int:
    """Standalone entry point, so this can be run on the device box directly."""
    if DEVICE is None:
        print("no Ascend NPU on this host: nothing to run")
        return 0
    missing = [name for name in (PREFILL_OP, SCATTER_OP) if _op(name) is None]
    if missing:
        print(f"these ops are not registered, so this build cannot be gated: {missing}")
        return 0
    return pytest.main([__file__, "-sv"])


if __name__ == "__main__":
    sys.exit(main())
