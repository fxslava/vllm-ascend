# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the golden NZ-tiled cache writer and reader.

The batch-writer kernel is built with ``TURBOQUANT_INGRESS_NZ=1`` and writes
the NZ-tiled cache the fused decode reads, and so is the scatter kernel: the
golden NZ writer below is the fallback for a stage that is switched off or
unregistered, and the dispatch rule these tests pin is that a staged scatter
kernel takes the NZ write itself.

* the scalar address functions of
  ``vllm_ascend/attention/turboquant_nz_ingress.py`` -- byte for byte, scale
  for scale. Those functions are themselves pinned against
  ``op_kernel/common/turboquant_layout.h`` by
  tests/ut/attention/test_turboquant_nz_ingress.py, so this closes the loop
  between the golden writer and the normative contract the kernel's
  ``static_assert``s are checked against.
* the whole-cache conversions ``turboquant_codes_to_nz_tiled`` and
  ``turboquant_scales_to_nz_tiled`` of the same module, which the golden
  per-token writes must reproduce exactly when every slot of a cache is
  covered.

Plus the dispatch rule the interim state depends on: the strip scatter kernel
is never allowed to write an NZ cache, even when its stage is on.
"""

import types

import pytest
import torch

from vllm_ascend.attention.turboquant_nz_ingress import (
    turboquant_codes_to_nz_tiled,
    turboquant_nz_packed_byte,
    turboquant_nz_scale_index,
    turboquant_scales_to_nz_tiled,
)
from vllm_ascend.attention.turboquant_v1 import (
    TurboQuantEmulator,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 128
NUM_KV_HEADS = 4
HEAD_SIZE = 256
PACKED = HEAD_SIZE // 2
NAMESPACE = "turboquant_ops"
CPU = torch.device("cpu")


def _adapter(
    *stages: TurboQuantKernelStage,
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.NZ_TILED,
) -> TurboQuantOpsAdapter:
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(ops_namespace=NAMESPACE, kernel_stages=frozenset(stages), value_layout=value_layout)
    )


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


def _caches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, PACKED)
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros((2, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE), dtype=torch.float32),
    )


def _activations(num_tokens: int, seed: int = 20261113) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(shape, generator=generator, dtype=torch.float32),
    )


@pytest.fixture
def written() -> tuple:
    """A cache 40 scattered tokens were golden-NZ-written into, plus the inputs."""
    torch.manual_seed(0)
    slots = sorted(torch.randperm(NUM_BLOCKS * BLOCK_SIZE)[:40].tolist())
    key, value = _activations(40)
    caches = _caches()
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)
    _adapter().reshape_scatter_decode(key, value, *caches, torch.tensor(slots, dtype=torch.int32), emulator)
    return caches, slots, key, value, emulator


def test_the_golden_nz_write_matches_the_scalar_oracle(written) -> None:
    """Every byte and every scale lane, against NzTiledPackedByte/NzTiledScaleIndex."""
    (key_cache, value_cache, scale_cache), slots, key, value, emulator = written
    codes_k, scales_k = emulator.quantize(emulator.rotate(key))
    codes_v, scales_v = emulator.quantize(emulator.rotate(value))
    key_flat = key_cache.view(-1).view(torch.uint8)
    value_flat = value_cache.view(-1).view(torch.uint8)
    scale_flat = scale_cache.view(-1)

    for t, slot in enumerate(slots):
        for head in range(NUM_KV_HEADS):
            for col in (0, 1, 31, 32, 64, 96, 127):
                want_k = (int(codes_k[t, head, col]) & 0xF) | ((int(codes_k[t, head, col + PACKED]) & 0xF) << 4)
                want_v = (int(codes_v[t, head, col]) & 0xF) | ((int(codes_v[t, head, col + PACKED]) & 0xF) << 4)
                assert int(key_flat[turboquant_nz_packed_byte(slot, head, col, NUM_KV_HEADS, PACKED)]) == want_k
                assert int(value_flat[turboquant_nz_packed_byte(slot, head, col, NUM_KV_HEADS, PACKED)]) == want_v
            assert float(scale_flat[turboquant_nz_scale_index(slot, head, 0, NUM_KV_HEADS)]) == float(scales_k[t, head])
            assert float(scale_flat[turboquant_nz_scale_index(slot, head, 1, NUM_KV_HEADS)]) == float(scales_v[t, head])


def test_the_golden_nz_write_matches_the_whole_cache_conversion(written) -> None:
    """The per-token writes reproduce turboquant_codes_to_nz_tiled over the cache."""
    (key_cache, value_cache, scale_cache), slots, key, value, emulator = written
    codes_k, scales_k = emulator.quantize(emulator.rotate(key))
    codes_v, scales_v = emulator.quantize(emulator.rotate(value))

    all_codes_k = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=torch.int8)
    all_codes_v = torch.zeros_like(all_codes_k)
    all_scales = torch.zeros(2, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, dtype=torch.float32)
    for t, slot in enumerate(slots):
        page, token = divmod(slot, BLOCK_SIZE)
        all_codes_k[page, :, token] = codes_k[t]
        all_codes_v[page, :, token] = codes_v[t]
        all_scales[0, page, :, token] = scales_k[t]
        all_scales[1, page, :, token] = scales_v[t]

    assert torch.equal(
        key_cache.view(-1).view(torch.uint8), turboquant_codes_to_nz_tiled(all_codes_k).view(torch.uint8)
    )
    assert torch.equal(
        value_cache.view(-1).view(torch.uint8), turboquant_codes_to_nz_tiled(all_codes_v).view(torch.uint8)
    )
    want_scales = turboquant_scales_to_nz_tiled(all_scales).reshape(-1)
    assert torch.equal(scale_cache.view(-1)[: want_scales.numel()], want_scales)


def test_the_golden_nz_gather_round_trips(written) -> None:
    """Gathering the NZ cache returns the codec's own codes and scales, token-major."""
    (key_cache, value_cache, scale_cache), slots, key, value, emulator = written
    codes_k, scales_k = emulator.quantize(emulator.rotate(key))
    codes_v, scales_v = emulator.quantize(emulator.rotate(value))

    block_tables = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    packed_k, got_scales_k, packed_v, got_scales_v = _adapter().gather(
        key_cache, value_cache, scale_cache, block_tables
    )
    got_codes_k = TurboQuantEmulator.unpack(packed_k[0])
    got_codes_v = TurboQuantEmulator.unpack(packed_v[0])
    for t, slot in enumerate(slots):
        assert torch.equal(got_codes_k[slot], codes_k[t])
        assert torch.equal(got_codes_v[slot], codes_v[t])
        assert torch.equal(got_scales_k[0, slot], scales_k[t])
        assert torch.equal(got_scales_v[0, slot], scales_v[t])


def test_repeated_decode_writes_accumulate_without_a_read_modify_write(emulator: TurboQuantEmulator) -> None:
    """Sixteen decode steps into one tile row each, which is what a real decode does.

    The NZ pairing is token-major, so a later token's write cannot disturb an
    earlier one's bytes -- the property the strip cache needed a read-modify-
    write for. Checked by writing token by token and reading the tile back.
    """
    adapter = _adapter()
    caches = _caches()
    slots = torch.arange(16, dtype=torch.int32)
    for token in range(16):
        key, value = _activations(1, seed=100 + token)
        adapter.reshape_scatter_decode(key, value, *caches, slots[token : token + 1], emulator)

    codes_k = []
    for token in range(16):
        key, _ = _activations(1, seed=100 + token)
        codes_k.append(emulator.quantize(emulator.rotate(key))[0][0])
    packed_k, _, _, _ = adapter.gather(*caches, torch.tensor([[0]], dtype=torch.int32))
    got = TurboQuantEmulator.unpack(packed_k[0])
    for token in range(16):
        assert torch.equal(got[token], codes_k[token])


def test_the_scatter_kernel_writes_the_nz_cache_it_is_staged_for(
    monkeypatch: pytest.MonkeyPatch, emulator: TurboQuantEmulator
) -> None:
    """Both stages staged: a decode step reaches the scatter kernel directly.

    The scatter writer is the NZ ingress too, so there is no layout left to
    arbitrate: the staged op is called, and the golden writer is only the
    fallback for when the stage is off.
    """
    calls: list[tuple] = []

    class _Entry:
        def __call__(self, *args, **kwargs):
            calls.append(args)
            return None

    monkeypatch.setattr(
        torch,
        "ops",
        types.SimpleNamespace(
            turboquant_ops=types.SimpleNamespace(
                npu_turboquant_reshape_and_cache_a8=_Entry(),
                npu_turboquant_reshape_scatter_aiv=_Entry(),
            )
        ),
    )
    adapter = _adapter(TurboQuantKernelStage.RESHAPE_AND_CACHE, TurboQuantKernelStage.SCATTER)
    assert adapter.value_layout() is TurboQuantValueLayout.NZ_TILED

    key, value = _activations(3)
    caches = _caches()
    adapter.reshape_scatter_decode(key, value, *caches, torch.tensor([5, 70, 129], dtype=torch.int32), emulator)

    assert len(calls) == 1, "the staged NZ scatter kernel did not take the write"


def test_the_nz_writer_refuses_a_page_that_is_not_whole_tiles(emulator: TurboQuantEmulator) -> None:
    """block_size 32 is writable in the strip cache and unreadable in the NZ one."""
    adapter = _adapter()
    packed = (NUM_BLOCKS, 32, NUM_KV_HEADS, PACKED)
    caches = (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros((2, NUM_BLOCKS, NUM_KV_HEADS, 32), dtype=torch.float32),
    )
    key, value = _activations(1)
    with pytest.raises(ValueError, match="64-row tiles"):
        adapter.reshape_scatter_decode(key, value, *caches, torch.tensor([0], dtype=torch.int32), emulator)
