# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The golden model's strip cache, against the two NPU writers' own formula.

The audited writers do not produce the golden model's cache. Their strip layout
puts the page axis in a different order and makes the value plane channel-major
*inside a 16-token strip*, which the two kernels state as

    key   nibble = ((block * H + head) * S + token) * D + channel
    value nibble = ((block * H + head) * S) * D
                 + (token // 16) * (16 * D) + channel * 16 + (token % 16)

(``turboquant_reshape_scatter.cpp``: the ``key_base`` burst, and
``StripWordOffset`` plus the within-strip index). The oracle below computes that
index element by element with no tensor tricks, so it shares nothing with the
vectorised implementation it checks.

What needs a device and is therefore NOT what these tests establish: that the
kernels agree with their own documented formula. These pin the golden model to
the formula; a byte-for-byte differential against the compiled kernels needs an
NPU.
"""

import types

import pytest
import torch

from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantEmulator,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_VALUE_PLANE,
    turboquant_packed_bytes,
    turboquant_scale_cache_shape,
)

HEAD_SIZE = 64
NUM_KV_HEADS = 2
NUM_BLOCKS = 4
R = TURBOQUANT_STRIP_TOKENS


def _adapter(layout):
    return TurboQuantOpsAdapter(
        TurboQuantOpsConfig(ops_namespace="turboquant_ops", kernel_stages=frozenset(), value_layout=layout)
    )


def _caches(block_size):
    packed = (NUM_BLOCKS, block_size, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    return (
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(packed, dtype=torch.int8),
        torch.zeros(turboquant_scale_cache_shape(NUM_BLOCKS, block_size, NUM_KV_HEADS), dtype=torch.float32),
    )


def _acts(n, seed):
    g = torch.Generator().manual_seed(seed)
    shape = (n, NUM_KV_HEADS, HEAD_SIZE)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def _nibble(buf_bytes, idx):
    """Signed INT4 at nibble index `idx` of a flat int8 byte tensor."""
    byte = int(buf_bytes[idx // 2]) & 0xFF
    code = (byte >> (4 * (idx % 2))) & 0xF
    return code - 16 if code >= 8 else code


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_strip_write_matches_the_kernel_nibble_formula(block_size):
    """Scalar oracle: every nibble lands where the two writers say it does.

    key   nibble = ((block*H + head)*S + token)*D + channel
    value nibble = ((block*H + head)*S)*D
                 + (token//16)*(16*D) + channel*16 + (token%16)
    """
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    adapter = _adapter(TurboQuantValueLayout.CHANNEL_MAJOR)
    num_tokens = 2 * block_size
    key, value = _acts(num_tokens, 11)
    caches = _caches(block_size)
    slots = torch.arange(num_tokens, dtype=torch.int32)
    adapter.reshape_and_cache_prefill(key, value, *caches, slots, emulator)

    key_bytes = caches[0].reshape(-1)
    value_bytes = caches[1].reshape(-1)
    codes_key, _ = emulator.quantize(emulator.rotate(key))
    codes_value, _ = emulator.quantize(emulator.rotate(value))

    D = HEAD_SIZE
    H = NUM_KV_HEADS
    S = block_size
    for t in range(num_tokens):
        block, token = divmod(t, S)
        for head in range(H):
            page = (block * H + head) * S
            for channel in range(D):
                assert _nibble(key_bytes, page * D + token * D + channel) == int(codes_key[t, head, channel]), (
                    f"key t={t} head={head} ch={channel}"
                )
                vidx = page * D + (token // R) * (R * D) + channel * R + (token % R)
                assert _nibble(value_bytes, vidx) == int(codes_value[t, head, channel]), (
                    f"value t={t} head={head} ch={channel}"
                )


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_the_strip_reader_inverts_the_strip_writer(block_size):
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    adapter = _adapter(TurboQuantValueLayout.CHANNEL_MAJOR)
    num_tokens = 2 * block_size
    key, value = _acts(num_tokens, 12)
    caches = _caches(block_size)
    adapter.reshape_and_cache_prefill(key, value, *caches, torch.arange(num_tokens, dtype=torch.int32), emulator)

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    pk, ks, pv, vs = adapter.gather(*caches, block_tables)
    assert pk.shape == (1, num_tokens, NUM_KV_HEADS, turboquant_packed_bytes(HEAD_SIZE))
    assert torch.equal(
        adapter.dequantize_cache(pk, ks, emulator, torch.float32)[0], emulator.decode(*emulator.encode(key))
    )
    assert torch.equal(
        adapter.dequantize_cache(pv, vs, emulator, torch.float32)[0], emulator.decode(*emulator.encode(value))
    )


@pytest.mark.parametrize("block_size", [16, 64])
def test_a_per_token_write_preserves_the_rest_of_the_strip(block_size):
    """The read-modify-write contract: one token must not disturb its neighbours."""
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    adapter = _adapter(TurboQuantValueLayout.CHANNEL_MAJOR)
    num_tokens = block_size
    key, value = _acts(num_tokens, 13)

    whole = _caches(block_size)
    adapter.reshape_and_cache_prefill(key, value, *whole, torch.arange(num_tokens, dtype=torch.int32), emulator)

    # The same content, written one token at a time in a scrambled order.
    piecemeal = _caches(block_size)
    order = torch.randperm(num_tokens, generator=torch.Generator().manual_seed(5))
    for t in order.tolist():
        adapter.reshape_scatter_decode(
            key[t : t + 1], value[t : t + 1], *piecemeal, torch.tensor([t], dtype=torch.int32), emulator
        )
    for name, a, b in zip(("key", "value", "scale"), whole, piecemeal):
        assert torch.equal(a, b), name


def test_padding_slots_touch_nothing():
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    adapter = _adapter(TurboQuantValueLayout.CHANNEL_MAJOR)
    key, value = _acts(3, 14)
    caches = _caches(16)
    adapter.reshape_scatter_decode(key, value, *caches, torch.tensor([-1, -1, -1], dtype=torch.int32), emulator)
    for c in caches:
        assert torch.equal(c, torch.zeros_like(c))


def test_token_major_is_untouched_by_all_this():
    """The golden layout must still be the golden layout."""
    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    adapter = _adapter(TurboQuantValueLayout.TOKEN_MAJOR)
    key, value = _acts(16, 15)
    caches = _caches(16)
    adapter.reshape_and_cache_prefill(key, value, *caches, torch.arange(16, dtype=torch.int32), emulator)
    for plane, acts, cache in (
        (TURBOQUANT_KEY_PLANE, key, caches[0]),
        (TURBOQUANT_VALUE_PLANE, value, caches[1]),
    ):
        packed, scales = emulator.encode(acts)
        for t in range(16):
            assert torch.equal(cache[0, t], packed[t]), (plane, t)
            assert torch.equal(caches[2][plane, 0, :, t], scales[t])


# ---------------------------------------------------------------------------
# A kernel-written cache, read back by the golden gather
# ---------------------------------------------------------------------------


def _oracle_strip_cache(key, value, caches, slot_mapping, emulator):
    """Fill the three planes from the writers' nibble formula, scalar by scalar.

    Shares no code with the adapter: this is what an NPU writer would leave
    behind, transcribed from `turboquant_reshape_scatter.cpp`. It is the stand-in
    for the real kernels, which cannot run on a host with no NPU.
    """
    key_cache, value_cache, scale_cache = caches
    _, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * 2
    key_flat = key_cache.reshape(-1)
    value_flat = value_cache.reshape(-1)
    codes_key, key_scales = emulator.quantize(emulator.rotate(key))
    codes_value, value_scales = emulator.quantize(emulator.rotate(value))

    def put(flat, idx, code):
        byte, half = idx // 2, idx % 2
        raw = int(flat[byte]) & 0xFF
        nib = int(code) & 0xF
        raw = (raw & ~(0xF << (4 * half))) | (nib << (4 * half))
        flat[byte] = torch.tensor(raw, dtype=torch.uint8).view(torch.int8)

    for t, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        block, token = divmod(slot, block_size)
        for head in range(num_kv_heads):
            page = (block * num_kv_heads + head) * block_size
            for channel in range(head_size):
                put(key_flat, page * head_size + token * head_size + channel, codes_key[t, head, channel])
                put(
                    value_flat,
                    page * head_size + (token // R) * (R * head_size) + channel * R + (token % R),
                    codes_value[t, head, channel],
                )
            scale_cache[TURBOQUANT_KEY_PLANE, block, head, token] = key_scales[t, head]
            scale_cache[TURBOQUANT_VALUE_PLANE, block, head, token] = value_scales[t, head]


@pytest.mark.parametrize("block_size", [16, 64])
def test_the_golden_gather_reads_a_kernel_written_cache(monkeypatch, block_size):
    """Item 3: an NPU-written strip cache, read back with no numerical drift.

    The cache is filled by the scalar oracle above -- the writers' formula, not
    the adapter's code -- and then read with the scatter stage engaged, which is
    what makes `value_layout()` report CHANNEL_MAJOR and select the strip
    reader. The configuration is TURBOQUANT_KERNEL_STAGES=scatter with the
    decode still in golden fallback.

    NOT a differential against the compiled kernels: that needs an NPU. What it
    establishes is that the golden reader inverts the writers' documented
    formula exactly.
    """

    class _Entry:
        def __call__(self, *a, **k):
            return None

    monkeypatch.setattr(
        torch,
        "ops",
        types.SimpleNamespace(turboquant_ops=types.SimpleNamespace(npu_turboquant_reshape_scatter_aiv=_Entry())),
    )
    adapter = TurboQuantOpsAdapter(
        TurboQuantOpsConfig(
            ops_namespace="turboquant_ops",
            kernel_stages=frozenset({TurboQuantKernelStage.SCATTER}),
        )
    )
    assert adapter.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR

    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    num_tokens = 2 * block_size
    key, value = _acts(num_tokens, 21)
    caches = _caches(block_size)
    _oracle_strip_cache(key, value, caches, torch.arange(num_tokens, dtype=torch.int32), emulator)

    pk, ks, pv, vs = adapter.gather(*caches, torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.equal(
        adapter.dequantize_cache(pk, ks, emulator, torch.float32)[0], emulator.decode(*emulator.encode(key))
    )
    assert torch.equal(
        adapter.dequantize_cache(pv, vs, emulator, torch.float32)[0], emulator.decode(*emulator.encode(value))
    )


def test_stages_scatter_writes_through_the_kernel_and_reads_back_golden(monkeypatch):
    """TURBOQUANT_KERNEL_STAGES=scatter: NPU writes, golden decode, no exception.

    The host-side half of the configuration the integration ships. It drives the
    stage set from the environment, so the env parsing and the derived layout are
    covered too, and it checks the four things that have to hold together:

    * the scatter op is really called -- the write is not silently golden;
    * nothing raises on the way, which is what the old CHANNEL_MAJOR refusal
      used to do;
    * `fused_decode` declines, because the decode stage is off;
    * the gather still reads the strip layout, so a cache the kernel wrote comes
      back correctly.

    The device half -- that the kernel's bytes equal the golden model's -- is
    tests/ut/attention/test_turboquant_live_write_npu.py and needs an NPU.
    """
    calls = []

    class _Entry:
        def __call__(self, *args, **kwargs):
            calls.append(args)
            return None

    monkeypatch.setenv("TURBOQUANT_KERNEL_STAGES", "scatter")
    monkeypatch.setenv("TURBOQUANT_OPS_NAMESPACE", "turboquant_ops")
    monkeypatch.setattr(
        torch,
        "ops",
        types.SimpleNamespace(turboquant_ops=types.SimpleNamespace(npu_turboquant_reshape_scatter_aiv=_Entry())),
    )

    adapter = TurboQuantOpsAdapter()
    assert adapter.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR

    emulator = TurboQuantEmulator(head_size=HEAD_SIZE, device=torch.device("cpu"))
    block_size = 64
    key, value = _acts(block_size, 31)
    slots = torch.arange(block_size, dtype=torch.int32)

    # The kernel owns the write, so the planes stay as the (stubbed) op left
    # them -- untouched. That the op was reached is the assertion.
    kernel_caches = _caches(block_size)
    adapter.reshape_scatter_decode(key, value, *kernel_caches, slots, emulator)
    assert len(calls) == 1, "the scatter stage did not reach the kernel"
    assert all(torch.equal(c, torch.zeros_like(c)) for c in kernel_caches)

    # The decode stage is off, so the fused decode has to decline and let the
    # backend fall through to the golden gather plus FIA.
    assert (
        adapter.fused_decode(
            query=torch.randn(1, 2, HEAD_SIZE),
            key_cache=kernel_caches[0],
            value_cache=kernel_caches[1],
            scale_cache=kernel_caches[2],
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            context_lens=torch.ones(1, dtype=torch.int32),
            emulator=emulator,
            num_heads=2,
            num_kv_heads=NUM_KV_HEADS,
            scale=0.125,
            output=torch.zeros(1, 2, HEAD_SIZE),
        )
        is None
    )

    # And the reader this configuration leans on still inverts the strip layout.
    written = _caches(block_size)
    _oracle_strip_cache(key, value, written, slots, emulator)
    packed_key, key_scales, _, _ = adapter.gather(*written, torch.zeros(1, 1, dtype=torch.int32))
    dense = adapter.dequantize_cache(packed_key, key_scales, emulator, torch.float32)
    assert torch.equal(dense[0], emulator.decode(*emulator.encode(key)))
