# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared TurboQuant test setup; numerical and layout oracles stay in the tests."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_DECODE_MAX_SPLITS,
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_DEFAULT_NAMESPACE,
    TURBOQUANT_NZ_TILE_ROWS,
    TurboQuantKernelStage,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    turboquant_decode_query_rot_floats,
    turboquant_decode_workspace_floats,
    turboquant_tiling_upper_bound_words,
)
from vllm_ascend.attention.turboquant_layout import turboquant_packed_bytes, turboquant_scale_cache_shape
from vllm_ascend.attention.turboquant_metadata import AscendTurboQuantMetadataBuilder
from vllm_ascend.attention.turboquant_ops import TurboQuantOpsAdapter

CACHE_POISON_SCALE = 2.0**100


class OpRecorder:
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


def register_ops(monkeypatch, *, namespace: str = TURBOQUANT_DEFAULT_NAMESPACE, **entries: OpRecorder) -> None:
    monkeypatch.setattr(torch, "ops", SimpleNamespace(**{namespace: SimpleNamespace(**entries)}))


def make_adapter(
    *stages: TurboQuantKernelStage,
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.TOKEN_MAJOR,
) -> TurboQuantOpsAdapter:
    return TurboQuantOpsAdapter(TurboQuantOpsConfig(kernel_stages=frozenset(stages), value_layout=value_layout))


def make_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    *,
    device="cpu",
    decode: bool = False,
    poison: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (num_blocks, block_size, num_kv_heads, turboquant_packed_bytes(head_size))
    scales = (
        (num_blocks * block_size * TurboQuantOpsAdapter._nz_scale_slot_floats(num_kv_heads),)
        if decode
        else turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)
    )
    cache = (
        torch.full(packed, 127 if poison else 0, dtype=torch.int8, device=device),
        torch.full(packed, -128 if poison else 0, dtype=torch.int8, device=device),
        torch.zeros(scales, dtype=torch.float32, device=device),
    )
    if poison:
        pattern = torch.tensor([float("nan"), float("inf"), CACHE_POISON_SCALE], dtype=torch.float32, device=device)
        cache[2].view(-1).copy_(pattern.repeat(-(-cache[2].numel() // len(pattern)))[: cache[2].numel()])
    return cache


def make_activations(
    num_tokens: int, num_kv_heads: int, head_size: int, *, seed: int, device="cpu", dtype=torch.float32
):
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, num_kv_heads, head_size)
    return tuple(torch.randn(shape, generator=generator).to(device=device, dtype=dtype) for _ in range(2))


def make_decode_buffers(tokens: int, num_heads: int, head_size: int, *, device="cpu") -> dict[str, torch.Tensor]:
    return {
        "tiling": torch.zeros(
            turboquant_tiling_upper_bound_words(tokens, TURBOQUANT_DECODE_MAX_SPLITS), dtype=torch.int32, device=device
        ),
        "workspace": torch.zeros(
            turboquant_decode_workspace_floats(tokens, num_heads, head_size, TURBOQUANT_DECODE_SPLITS),
            dtype=torch.float32,
            device=device,
        ),
        "query_rot": torch.zeros(
            turboquant_decode_query_rot_floats(tokens, num_heads, head_size), dtype=torch.float32, device=device
        ),
    }


def make_metadata_builder(
    max_num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    max_blocks: int,
    *,
    device="cpu",
) -> AscendTurboQuantMetadataBuilder:
    """Run the real buffer allocation, replacing only engine-specific base setup."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
        model_config=SimpleNamespace(get_num_attention_heads=lambda _: num_heads),
        parallel_config=SimpleNamespace(),
    )
    spec = SimpleNamespace(num_kv_heads=num_kv_heads, head_size=head_size, block_size=block_size)

    def init_base(builder, *args, **kwargs):
        builder.max_num_blocks_per_req = max_blocks

    with patch.object(AscendAttentionMetadataBuilder, "__init__", init_base):
        builder = AscendTurboQuantMetadataBuilder(spec, ["attention"], config, torch.device(device))
    builder._block_size = block_size
    return builder


def assert_cache_equal(kernel_caches, golden_caches, *, scale_max_ulps: int = 0, poison_mask=None) -> None:
    """Compare packed bytes and scale bits, with an explicit live-scale budget.

    Only positive, finite FP32 scales can use the ULP allowance. Zero lanes,
    infinities, NaN payloads and the large finite poison retain exact bits.
    The optional ownership-based poison_mask excludes unwritten scales from
    all tolerance, even if both their values have become positive and finite.
    Keep the default strict for read-only cache snapshots and uniform scales.
    """
    names = ("key_cache", "value_cache", "scale_cache")
    for name, written, expected in zip(names, kernel_caches, golden_caches):
        a = written.cpu()
        b = expected.cpu()
        assert a.shape == b.shape and a.dtype == b.dtype, f"{name}: cache shape or dtype differs"
        if a.dtype == torch.float32:
            bits_a = a.view(torch.int32).to(torch.int64)
            bits_b = b.view(torch.int32).to(torch.int64)
            ulps = (bits_a - bits_b).abs()
            live = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0) & (b != CACHE_POISON_SCALE)
            if poison_mask is not None:
                assert poison_mask.shape == b.shape and poison_mask.dtype == torch.bool
                live &= ~poison_mask.cpu()
            equal = (bits_a == bits_b) | (live & (ulps <= scale_max_ulps))
        else:
            equal = a == b
        if bool(equal.all()):
            continue
        differing = (~equal).sum().item()
        first = int((~equal).reshape(-1).nonzero()[0].item())
        precision = ""
        if a.dtype == b.dtype == torch.float32:
            first_a = a.reshape(-1)[first]
            first_b = b.reshape(-1)[first]
            bits_a = int(first_a.view(torch.int32)) & 0xFFFFFFFF
            bits_b = int(first_b.view(torch.int32)) & 0xFFFFFFFF
            finite = torch.isfinite(a) & torch.isfinite(b)
            max_error = float((a[finite] - b[finite]).abs().max()) if bool(finite.any()) else float("nan")
            max_ulps = int(ulps[live].max()) if bool(live.any()) else 0
            precision = (
                f" First FP32 bits: 0x{bits_a:08x} vs 0x{bits_b:08x}; "
                f"max finite absolute error {max_error:.9g}; "
                f"max live scale error {max_ulps} ULP (budget {scale_max_ulps})."
            )
        raise AssertionError(
            f"{name}: the kernel and the golden model disagree in {differing} of {a.numel()} "
            f"elements, first at flat index {first} "
            f"(kernel {a.reshape(-1)[first]!r} vs golden {b.reshape(-1)[first]!r}). "
            f"{precision} Packed bytes and non-live scale bits must match exactly; "
            "check codec arithmetic and cache addressing."
        )


def make_recycled_page_table(contexts, block_size, *, reserve_tokens=2, seed=7):
    """Shuffled, nonadjacent pages in the upper half of a larger cache.

    Reserve slots for the first decode steps; table padding names poisoned page
    zero. Returning the allocation extent keeps page IDs independent of block 0.
    """
    page_counts = [-(-(context + reserve_tokens) // block_size) for context in contexts]
    total_pages = sum(page_counts)
    num_blocks = 4 * total_pages + 1
    pages = 2 * torch.randperm(total_pages, generator=torch.Generator().manual_seed(seed)) + 2 * total_pages + 1
    tables = torch.zeros((len(contexts), max(page_counts)), dtype=torch.int32)
    start = 0
    for row, count in zip(tables, page_counts):
        row[:count] = pages[start : start + count]
        start += count
    return tables, num_blocks


def slots_from_page_table(tables, contexts, block_size):
    """Physical slots for each request's logical token order, on the host."""
    slots = []
    for row, context in zip(tables, contexts):
        logical = torch.arange(context, dtype=torch.int64)
        slots.append(row[logical // block_size].long() * block_size + logical % block_size)
    return torch.cat(slots).to(torch.int32)


def assert_poison_scales_preserved(cache, initial, poison_mask):
    """Check surviving NaNs first, then every unwritten scale's exact bits."""
    actual, expected = cache[2].cpu(), initial[2].cpu()
    mask = poison_mask.cpu()
    nan_slots = mask & torch.isnan(expected)
    assert torch.isnan(actual[nan_slots]).all(), "NaN poison was neutralized in an unowned scale slot"
    unchanged = actual.view(torch.int32)[mask] == expected.view(torch.int32)[mask]
    assert unchanged.all(), "unowned scale poison changed bits (including NaN payloads)"


def make_finite_spike_cache(num_blocks, block_size, num_kv_heads, head_size, *, emulator, device="cpu"):
    """Valid finite codebook rows aligned with a model-basis all-ones query."""
    cache = make_cache(num_blocks, block_size, num_kv_heads, head_size, device=device)
    golden = make_adapter(value_layout=TurboQuantValueLayout.NZ_TILED)
    ones = torch.ones(TURBOQUANT_NZ_TILE_ROWS, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)
    slots = torch.arange(TURBOQUANT_NZ_TILE_ROWS, dtype=torch.int32, device=device)
    golden.reshape_scatter_decode(ones * 10, ones * 100, *cache, slots, emulator)
    tile_bytes = TURBOQUANT_NZ_TILE_ROWS * num_kv_heads * turboquant_packed_bytes(head_size)
    for plane in cache[:2]:
        tile = plane.view(-1)[:tile_bytes].clone()
        plane.view(-1, tile_bytes).copy_(tile.unsqueeze(0).expand(plane.numel() // tile_bytes, -1))
    scale_floats = TURBOQUANT_NZ_TILE_ROWS * TurboQuantOpsAdapter._nz_scale_slot_floats(num_kv_heads)
    tile_scales = cache[2].view(-1)[:scale_floats].clone()
    cache[2].view(-1, scale_floats).copy_(tile_scales.unsqueeze(0).expand(cache[2].numel() // scale_floats, -1))
    assert torch.isfinite(cache[2]).all() and (cache[2] > 0).all(), "spike fixture must contain finite positive scales"
    return cache
