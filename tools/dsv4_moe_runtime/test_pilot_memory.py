"""Self-contained pilot verification for the DeepSeek-V4 static expert slot pool.

Run from anywhere::

    python -m pytest tools/dsv4_moe_runtime/test_pilot_memory.py -v

CPU-only, synthetic tensors -- nothing here touches the NPU. The suite checks
the four pilot invariants:

1. ``test_expert_byte_size`` -- exact byte accounting against the real
   DeepSeek-V4 metadata (13,369,344 bytes = 12.75 MiB per expert slot).
2. ``test_zero_runtime_allocations`` -- 50 emulated decode steps with the
   ``torch`` allocators booby-trapped; arena ``data_ptr()``s never move.
3. ``test_step_lock_prevents_eviction_collision`` -- a step's top-k slots are
   locked atomically; exhausted free lists raise instead of evicting locked
   slots, and the pool stays transactional.
4. ``test_hash_layer_routing`` -- layers 0-2 resolve through their per-layer
   ``tid2eid`` tables; layers 3+ resolve through bias-shifted score top-k.

When ``C:\\DeepSeekV4`` is present the byte test additionally cross-checks the
real ``config.json`` / ``model.safetensors.index.json``; without it the suite
still passes on embedded constants (self-contained requirement).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from slot_pool import (
    EXPERT_PARAM_NAMES,
    DeepSeekV4MoEConfig,
    ExpertTensorLayout,
    HashRouteResolver,
    RouteResolverProtocol,
    ScoreRouteResolver,
    SlotExhaustionError,
    StaticExpertSlotPool,
    WeightProviderProtocol,
)

# --------------------------------------------------------------------------- #
# Constants derived from the real DeepSeek-V4 Flash metadata (C:\DeepSeekV4).  #
# --------------------------------------------------------------------------- #

DEEPSEEK_V4_MODEL_DIR = Path(r"C:\DeepSeekV4")
HF_CONFIG_FILENAME = "config.json"
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"

EXPECTED_SLOT_NUM_BYTES = 13_369_344  # 12.75 MiB = 3 x (17 * 2**18)
EXPECTED_PACKED_PROJECTION_BYTES = 4_194_304  # [2048,2048] or [4096,1024] packed fp4
EXPECTED_SCALE_PROJECTION_BYTES = 262_144  # [2048,128] or [4096,64] e8m0
EXPECTED_INSTANCE_COUNT = 11_264  # (43 decoder blocks + 1 MTP block) x 256 experts
EXPECTED_EXPERT_TENSOR_COUNT = 67_584  # instances x {w1,w2,w3} x {weight,scale}
EXPECTED_TOTAL_EXPERT_BYTES = 150_592_290_816  # instances x slot bytes
EXPECTED_NON_EXPERT_BYTES = 9_017_195_080  # index total minus routed experts
EXPECTED_INDEX_TOTAL_SIZE = 159_609_485_896
EXPECTED_HASH_LAYERS = (0, 1, 2)
EXPERT_INDEX_TENSOR_PATTERN = re.compile(
    r"(?:layers\.(\d+)|mtp\.(\d+))\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)"
)

FULL_MODEL_CONFIG = DeepSeekV4MoEConfig()  # real DeepSeek-V4 Flash geometry

# Scaled-down geometry for behaviour tests: identical structure (3 packed fp4
# projections with block-32 e8m0 scales), ~13 KB slots instead of 12.75 MiB.
TEST_CONFIG = DeepSeekV4MoEConfig(
    hidden_size=128,
    moe_intermediate_size=64,
    num_routed_experts=256,
    top_k=6,
    num_layers=43,
    num_hash_layers=3,
    vocab_size=4096,
)

NUM_DECODE_STEPS = 50
NUM_SLOTS_FOR_CHURN_TEST = 16


# --------------------------------------------------------------------------- #
# WeightProvider implementations (DIP: production-shaped + synthetic mock).    #
# --------------------------------------------------------------------------- #


class PinnedMemoryWeightProvider:
    """Production skeleton: serves expert weights from staged host buffers.

    The real deployment stages safetensors bytes into pinned DDR once at
    startup; this class models that contract with a plain mapping. On a
    CUDA/pinned-capable build the staging tensor would be ``.pin_memory()``-ed;
    the CPU-only pilot keeps page buffers and documents the difference.
    """

    def __init__(self, buffers: Mapping[tuple[int, int, str], torch.Tensor]):
        self._buffers: dict[tuple[int, int, str], torch.Tensor] = {}
        for key, tensor in buffers.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"host provider buffer {key} must live on CPU, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"host provider buffer {key} must be contiguous")
            self._buffers[key] = tensor

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        try:
            return self._buffers[(layer_idx, expert_id, param_key)]
        except KeyError:
            raise KeyError(
                f"no host buffer staged for (layer={layer_idx}, expert={expert_id}, param={param_key})"
            ) from None


class MockWeightProvider(PinnedMemoryWeightProvider):
    """Pre-allocates deterministic synthetic buffers for every expert parameter.

    Each buffer starts with a 4-byte big-endian tag uniquely encoding
    (layer, expert, param), followed by a position-dependent pattern -- so
    ``torch.equal`` against a slot view proves the *right* expert's bytes
    landed in the *right* slot. All allocation happens at construction; the
    serve path is a dict lookup, i.e. allocation-free.
    """

    def __init__(self, layout: ExpertTensorLayout, layer_ids: Sequence[int], num_experts: int):
        buffers: dict[tuple[int, int, str], torch.Tensor] = {}
        num_params = len(layout.specs)
        for layer_idx in layer_ids:
            for expert_id in range(num_experts):
                for param_index, spec in enumerate(layout.specs):
                    tag = (layer_idx * 256 + expert_id) * num_params + param_index
                    buffers[(layer_idx, expert_id, spec.param_key)] = _synthetic_bytes(
                        tag, spec.view_shape[0], spec.view_shape[1]
                    )
        super().__init__(buffers)


def _synthetic_bytes(tag: int, rows: int, cols: int) -> torch.Tensor:
    row_index = torch.arange(rows).unsqueeze(1)
    col_index = torch.arange(cols).unsqueeze(0)
    buffer = ((tag + 31 * row_index + col_index) % 256).to(torch.uint8)
    flat = buffer.view(-1)
    for byte_pos in range(4):
        flat[byte_pos] = (tag >> (8 * (3 - byte_pos))) & 0xFF
    return buffer


# --------------------------------------------------------------------------- #
# Zero-allocation guard + arena fingerprinting helpers.                        #
# --------------------------------------------------------------------------- #

FORBIDDEN_ALLOCATOR_NAMES = (
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    "tensor",
    "as_tensor",
    "cat",
    "stack",
    "rand",
    "randn",
    "randint",
)


@contextmanager
def forbid_torch_allocations() -> Iterator[None]:
    """Booby-trap every torch allocator for the duration of the decode loop."""
    originals = {name: getattr(torch, name) for name in FORBIDDEN_ALLOCATOR_NAMES}
    original_clone = torch.Tensor.clone

    def make_guard(name: str):
        def guard(*_args, **_kwargs):
            raise AssertionError(f"decode-step path attempted allocation via torch.{name}")

        return guard

    def guard_clone(*_args, **_kwargs):
        raise AssertionError("decode-step path attempted allocation via Tensor.clone")

    try:
        for name in originals:
            setattr(torch, name, make_guard(name))
        torch.Tensor.clone = guard_clone  # type: ignore[assignment]
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch, name, fn)
        torch.Tensor.clone = original_clone  # type: ignore[assignment]


def pool_fingerprints(pool: StaticExpertSlotPool) -> list[int]:
    """data_ptr of every base tensor and pre-sliced view the pool owns."""
    tensors: list[torch.Tensor] = [pool.slot_arena, pool.expert_slot_table, pool.step_slot_ids_buffer]
    for slot_id in range(pool.num_slots):
        tensors.extend(pool.weight_views(slot_id).values())
        tensors.extend(pool.scale_views(slot_id).values())
    return [tensor.data_ptr() for tensor in tensors]


def churn_expert_ids(step: int, top_k: int, num_experts: int) -> list[int]:
    """Deterministic decode-step request mixing loads, LRU hits and evictions.

    Steps come in groups of five that request the same six experts (four pure
    LRU-hit steps after the initial loads); every fifth step shifts the window
    by eight experts, forcing the pool to evict the LRU-oldest residents. The
    16-slot churn pool keeps the active window resident, so hit steps also
    prove that eviction never touches live slots.
    """
    window_base = (step // 5) * 8
    return [(window_base + 3 * expert_pos) % num_experts for expert_pos in range(top_k)]


# --------------------------------------------------------------------------- #
# 1. Byte accounting                                                           #
# --------------------------------------------------------------------------- #


def test_expert_byte_size() -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(FULL_MODEL_CONFIG)

    assert layout.slot_num_bytes == EXPECTED_SLOT_NUM_BYTES
    assert layout.slot_num_bytes == 12.75 * 1024 * 1024  # 12.75 MiB, exact

    for name in EXPERT_PARAM_NAMES:
        packed = layout.spec_for(name)
        scale = layout.spec_for(name + "_scale")
        assert packed.num_bytes == EXPECTED_PACKED_PROJECTION_BYTES, name
        assert scale.num_bytes == EXPECTED_SCALE_PROJECTION_BYTES, name
        assert packed.num_bytes + scale.num_bytes == 4_456_448, name
    per_projection_total = EXPECTED_PACKED_PROJECTION_BYTES + EXPECTED_SCALE_PROJECTION_BYTES
    assert per_projection_total * 3 == EXPECTED_SLOT_NUM_BYTES

    # Regions are 128-byte aligned with no hidden padding beyond alignment.
    for spec in layout.specs:
        assert spec.offset_bytes % 128 == 0
        assert spec.offset_bytes + spec.num_bytes <= layout.slot_num_bytes
    assert layout.specs[-1].offset_bytes + layout.specs[-1].num_bytes == layout.slot_num_bytes

    # --- Cross-check against the real checkpoint metadata, when available. ---
    config_path = DEEPSEEK_V4_MODEL_DIR / HF_CONFIG_FILENAME
    if config_path.exists():
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert hf_config["hidden_size"] == FULL_MODEL_CONFIG.hidden_size
        assert hf_config["moe_intermediate_size"] == FULL_MODEL_CONFIG.moe_intermediate_size
        assert hf_config["n_routed_experts"] == FULL_MODEL_CONFIG.num_routed_experts
        assert hf_config["num_experts_per_tok"] == FULL_MODEL_CONFIG.top_k
        assert hf_config["num_hidden_layers"] == FULL_MODEL_CONFIG.num_layers
        assert hf_config["num_hash_layers"] == FULL_MODEL_CONFIG.num_hash_layers
        assert hf_config["expert_dtype"] == "fp4"

    index_path = DEEPSEEK_V4_MODEL_DIR / SAFETENSORS_INDEX_FILENAME
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Mapping[str, str] = index["weight_map"]
        expert_tensors = [name for name in weight_map if EXPERT_INDEX_TENSOR_PATTERN.fullmatch(name)]
        assert len(expert_tensors) == EXPECTED_EXPERT_TENSOR_COUNT
        instances = {name[: -len(".w1.weight")] for name in expert_tensors if name.endswith(".w1.weight")}
        assert len(instances) == EXPECTED_INSTANCE_COUNT
        total_expert_bytes = len(instances) * layout.slot_num_bytes
        assert total_expert_bytes == EXPECTED_TOTAL_EXPERT_BYTES
        total_size = index["metadata"]["total_size"]
        assert total_size == EXPECTED_INDEX_TOTAL_SIZE
        assert total_size - total_expert_bytes == EXPECTED_NON_EXPERT_BYTES
        hash_layers = {
            int(match.group(1))
            for name in weight_map
            if (match := re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.tid2eid", name))
        }
        assert tuple(sorted(hash_layers)) == EXPECTED_HASH_LAYERS
        bias_layers = {
            int(match.group(1))
            for name in weight_map
            if (match := re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias", name))
        }
        assert tuple(sorted(bias_layers)) == tuple(range(max(EXPECTED_HASH_LAYERS) + 1, FULL_MODEL_CONFIG.num_layers))


# --------------------------------------------------------------------------- #
# 2. Zero runtime allocations over 50 decode steps                             #
# --------------------------------------------------------------------------- #


def test_zero_runtime_allocations() -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(TEST_CONFIG)
    pool = StaticExpertSlotPool(TEST_CONFIG, num_slots=NUM_SLOTS_FOR_CHURN_TEST, layout=layout)
    pilot_layer = 3
    provider = MockWeightProvider(layout, layer_ids=(pilot_layer,), num_experts=TEST_CONFIG.num_routed_experts)
    assert isinstance(provider, WeightProviderProtocol)

    baseline = pool_fingerprints(pool)

    with forbid_torch_allocations():
        for step in range(NUM_DECODE_STEPS):
            expert_ids = churn_expert_ids(step, TEST_CONFIG.top_k, TEST_CONFIG.num_routed_experts)
            reservation = pool.acquire_for_step(pilot_layer, expert_ids, provider)
            try:
                assert len(set(reservation.slot_ids)) == TEST_CONFIG.top_k
                assert all(0 <= slot < pool.num_slots for slot in reservation.slot_ids)
                assert [int(value) for value in reservation.device_slot_ids] == list(reservation.slot_ids)
                for expert_id, slot_id in zip(expert_ids, reservation.slot_ids):
                    assert pool.slot_of(pilot_layer, expert_id) == slot_id
                    for name in EXPERT_PARAM_NAMES:
                        assert torch.equal(
                            pool.weight_views(slot_id)[name],
                            provider.pinned_cpu_weight(pilot_layer, expert_id, name),
                        ), f"step {step}: wrong bytes in slot {slot_id} for {name}"
                        assert torch.equal(
                            pool.scale_views(slot_id)[name],
                            provider.pinned_cpu_weight(pilot_layer, expert_id, name + "_scale"),
                        ), f"step {step}: wrong scale bytes in slot {slot_id} for {name}"
                assert pool_fingerprints(pool) == baseline, f"step {step}: arena moved"
            finally:
                pool.release_step(reservation)

    assert pool.resident_expert_count == NUM_SLOTS_FOR_CHURN_TEST
    assert pool.stats.hits >= 1, "expected cross-step LRU hits in the sliding window"
    assert pool.stats.hits + pool.stats.loads == NUM_DECODE_STEPS * TEST_CONFIG.top_k
    assert pool.stats.evictions == pool.stats.loads - NUM_SLOTS_FOR_CHURN_TEST
    assert pool_fingerprints(pool) == baseline


# --------------------------------------------------------------------------- #
# 3. Transactional step locking                                                #
# --------------------------------------------------------------------------- #


def test_step_lock_prevents_eviction_collision() -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(TEST_CONFIG)
    pool = StaticExpertSlotPool(TEST_CONFIG, num_slots=TEST_CONFIG.top_k, layout=layout)
    provider = MockWeightProvider(layout, layer_ids=(0,), num_experts=TEST_CONFIG.num_routed_experts)

    first_ids = [10, 11, 12, 13, 14, 15]
    reservation = pool.acquire_for_step(0, first_ids, provider)
    first_slots = {expert_id: pool.slot_of(0, expert_id) for expert_id in first_ids}
    assert len(set(first_slots.values())) == TEST_CONFIG.top_k
    baseline = pool_fingerprints(pool)

    # Same step, all 6 slots locked, 6 fresh experts requested: must refuse
    # atomically instead of evicting any locked slot.
    with pytest.raises(SlotExhaustionError, match="evictable"):
        pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], provider)

    # State is untouched: residency, bytes, counters.
    assert {e: pool.slot_of(0, e) for e in first_ids} == first_slots
    assert pool.stats.evictions == 0
    assert pool.resident_expert_count == TEST_CONFIG.top_k
    for expert_id, slot_id in first_slots.items():
        assert torch.equal(pool.weight_views(slot_id)["w1"], provider.pinned_cpu_weight(0, expert_id, "w1"))
    assert pool_fingerprints(pool) == baseline

    pool.release_step(reservation)

    # Locks released: the same request now succeeds by evicting exactly the
    # 6 (now-unlocked) LRU-oldest slots, reusing all of them.
    second_ids = [100, 101, 102, 103, 104, 105]
    second = pool.acquire_for_step(0, second_ids, provider)
    assert set(second.slot_ids) == set(first_slots.values())
    assert all(pool.slot_of(0, expert_id) == -1 for expert_id in first_ids)
    assert pool.stats.evictions == TEST_CONFIG.top_k
    for expert_id, slot_id in zip(second_ids, second.slot_ids):
        assert torch.equal(pool.weight_views(slot_id)["w2"], provider.pinned_cpu_weight(0, expert_id, "w2"))
    pool.release_step(second)

    # --- Free slots do not weaken the lock: 8 slots, 6 locked, 2 free. ---
    wide_pool = StaticExpertSlotPool(TEST_CONFIG, num_slots=8, layout=layout)
    held = wide_pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    assert wide_pool.resident_expert_count == 6
    with pytest.raises(SlotExhaustionError):
        wide_pool.acquire_for_step(0, [6, 7, 8, 9, 10, 11], provider)  # 2 free < 6 needed, rest locked
    wide_pool.release_step(held)

    evicting = wide_pool.acquire_for_step(0, [6, 7, 8, 9, 10, 11], provider)
    assert all(wide_pool.slot_of(0, expert_id) == -1 for expert_id in (0, 1, 2, 3))  # LRU-oldest evicted
    survivors = [wide_pool.slot_of(0, expert_id) for expert_id in (4, 5)]
    assert -1 not in survivors
    assert sorted(list(evicting.slot_ids) + survivors) == list(range(8))
    wide_pool.release_step(evicting)

    # --- Resident hits are never evicted to make room for co-requested misses. ---
    hit_pool = StaticExpertSlotPool(TEST_CONFIG, num_slots=TEST_CONFIG.top_k, layout=layout)
    warmup = hit_pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    hit_pool.release_step(warmup)
    mixed = hit_pool.acquire_for_step(0, [2, 3, 4, 5, 6, 7], provider)
    assert [hit_pool.slot_of(0, expert_id) for expert_id in (2, 3, 4, 5)] == [
        warmup.slot_ids[2],
        warmup.slot_ids[3],
        warmup.slot_ids[4],
        warmup.slot_ids[5],
    ]
    assert all(hit_pool.slot_of(0, expert_id) == -1 for expert_id in (0, 1))  # only true LRU victims
    assert hit_pool.stats.evictions == 2
    hit_pool.release_step(mixed)


# --------------------------------------------------------------------------- #
# 4. Hash layers 0-2 vs score-gated layers 3+                                  #
# --------------------------------------------------------------------------- #


def test_hash_layer_routing() -> None:
    generator = torch.Generator().manual_seed(1234)
    # The checkpoint ships one tid2eid per hash layer: [num_hash_layers, vocab, top_k].
    tid2eid = torch.randint(
        0,
        TEST_CONFIG.num_routed_experts,
        (TEST_CONFIG.num_hash_layers, TEST_CONFIG.vocab_size, TEST_CONFIG.top_k),
        dtype=torch.int32,
        generator=generator,
    )
    assert not torch.equal(tid2eid[0], tid2eid[1])  # per-layer tables are independent

    resolver = HashRouteResolver(TEST_CONFIG, tid2eid)
    assert isinstance(resolver, RouteResolverProtocol)
    token_ids = torch.tensor([5, 5, 4095, 0, 123], dtype=torch.int64)

    for layer_idx in range(TEST_CONFIG.num_hash_layers):
        out = torch.full((len(token_ids), TEST_CONFIG.top_k), -1, dtype=torch.int32)
        resolver.resolve(layer_idx, token_ids, out)
        assert torch.equal(out, tid2eid[layer_idx][token_ids]), f"layer {layer_idx} hash mismatch"
        again = torch.full_like(out, -2)
        resolver.resolve(layer_idx, token_ids, again)
        assert torch.equal(out, again)  # deterministic by construction

    with pytest.raises(ValueError, match="score-gated"):
        resolver.resolve(TEST_CONFIG.num_hash_layers, token_ids, torch.empty(1, TEST_CONFIG.top_k, dtype=torch.int32))

    # --- Score-gated layers: bias shifts selection, reference-checked. ---
    gate_bias = torch.zeros(TEST_CONFIG.num_layers, TEST_CONFIG.num_routed_experts, dtype=torch.float32)
    score_resolver = ScoreRouteResolver(TEST_CONFIG, gate_bias, max_batch_tokens=8)
    scores = torch.zeros(1, TEST_CONFIG.num_routed_experts, dtype=torch.float32)
    scores[0, :7] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.9])

    routing_out = torch.empty(1, TEST_CONFIG.top_k, dtype=torch.int64)
    score_resolver.resolve(3, scores, routing_out)
    assert torch.equal(routing_out, torch.topk(scores + gate_bias[3], TEST_CONFIG.top_k, dim=-1).indices)
    assert sorted(routing_out[0].tolist()) == [0, 1, 2, 3, 4, 5]

    # Bias demotes the strongest raw expert -- selection changes, scores do not.
    gate_bias[3, 0] = -100.0
    score_resolver.resolve(3, scores, routing_out)
    assert sorted(routing_out[0].tolist()) == [1, 2, 3, 4, 5, 6]
    assert torch.equal(routing_out, torch.topk(scores + gate_bias[3], TEST_CONFIG.top_k, dim=-1).indices)

    with pytest.raises(ValueError, match="hash-routed"):
        score_resolver.resolve(0, scores, routing_out)

    # --- End-to-end: hash-resolved ids drive the slot pool on a hash layer. ---
    layout = ExpertTensorLayout.for_deepseek_v4_flash(TEST_CONFIG)
    pool = StaticExpertSlotPool(TEST_CONFIG, num_slots=8, layout=layout)
    provider = MockWeightProvider(layout, layer_ids=(1,), num_experts=TEST_CONFIG.num_routed_experts)
    routed = torch.full((4, TEST_CONFIG.top_k), -1, dtype=torch.int32)
    resolver.resolve(1, torch.tensor([7, 9, 11, 13], dtype=torch.int64), routed)
    reservation = pool.acquire_for_step(1, routed[0].tolist(), provider)
    assert len(reservation.slot_ids) == TEST_CONFIG.top_k
    pool.release_step(reservation)
