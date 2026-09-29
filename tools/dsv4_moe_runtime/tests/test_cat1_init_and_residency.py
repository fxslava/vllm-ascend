"""CAT 1 -- init & residency: AOT allocation, memory residency, alignment.

Covers the construction-time invariants of the static runtime: exact slot byte
accounting against the real checkpoint metadata, 128-byte region alignment,
single-allocation arena geometry with pre-sliced fixed-address expert views,
transactional step locking over resident slots, and the routing tables that
decide which experts become resident.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import regex as re
import torch

from ..core.config import EXPERT_PARAM_NAMES, DeepSeekV4MoEConfig
from ..core.layout import SLOT_REGION_ALIGN_BYTES, ExpertTensorLayout
from ..core.ledger import SlotExhaustionError
from ..core.slot_pool import UNRESIDENT_SLOT_ID, StaticExpertSlotPool
from ..protocols.router import RouteResolverProtocol
from ..routing.hash_router import HashRouteResolver
from ..routing.score_router import ScoreRouteResolver

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


def test_expert_byte_size(full_layout: ExpertTensorLayout) -> None:
    assert full_layout.slot_num_bytes == EXPECTED_SLOT_NUM_BYTES
    assert full_layout.slot_num_bytes == 12.75 * 1024 * 1024  # 12.75 MiB, exact

    for name in EXPERT_PARAM_NAMES:
        packed = full_layout.spec_for(name)
        scale = full_layout.spec_for(name + "_scale")
        assert packed.num_bytes == EXPECTED_PACKED_PROJECTION_BYTES, name
        assert scale.num_bytes == EXPECTED_SCALE_PROJECTION_BYTES, name
        assert packed.num_bytes + scale.num_bytes == 4_456_448, name
    per_projection_total = EXPECTED_PACKED_PROJECTION_BYTES + EXPECTED_SCALE_PROJECTION_BYTES
    assert per_projection_total * 3 == EXPECTED_SLOT_NUM_BYTES

    # Regions are 128-byte aligned with no hidden padding beyond alignment.
    for spec in full_layout.specs:
        assert spec.offset_bytes % 128 == 0
        assert spec.offset_bytes + spec.num_bytes <= full_layout.slot_num_bytes
    assert full_layout.specs[-1].offset_bytes + full_layout.specs[-1].num_bytes == full_layout.slot_num_bytes

    # --- Cross-check against the real checkpoint metadata, when available. ---
    config_path = DEEPSEEK_V4_MODEL_DIR / HF_CONFIG_FILENAME
    if config_path.exists():
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert hf_config["hidden_size"] == DeepSeekV4MoEConfig().hidden_size
        assert hf_config["moe_intermediate_size"] == DeepSeekV4MoEConfig().moe_intermediate_size
        assert hf_config["n_routed_experts"] == DeepSeekV4MoEConfig().num_routed_experts
        assert hf_config["num_experts_per_tok"] == DeepSeekV4MoEConfig().top_k
        assert hf_config["num_hidden_layers"] == DeepSeekV4MoEConfig().num_layers
        assert hf_config["num_hash_layers"] == DeepSeekV4MoEConfig().num_hash_layers
        assert hf_config["expert_dtype"] == "fp4"

    index_path = DEEPSEEK_V4_MODEL_DIR / SAFETENSORS_INDEX_FILENAME
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        expert_tensors = [name for name in weight_map if EXPERT_INDEX_TENSOR_PATTERN.fullmatch(name)]
        assert len(expert_tensors) == EXPECTED_EXPERT_TENSOR_COUNT
        instances = {name[: -len(".w1.weight")] for name in expert_tensors if name.endswith(".w1.weight")}
        assert len(instances) == EXPECTED_INSTANCE_COUNT
        total_expert_bytes = len(instances) * full_layout.slot_num_bytes
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
        assert tuple(sorted(bias_layers)) == tuple(
            range(max(EXPECTED_HASH_LAYERS) + 1, DeepSeekV4MoEConfig().num_layers)
        )


def test_slot_pool_aot_allocation_and_view_alignment(
    sanity_config: DeepSeekV4MoEConfig, sanity_layout: ExpertTensorLayout
) -> None:
    """One arena allocation; every expert view pre-sliced at an aligned offset."""
    num_slots = 8
    pool = StaticExpertSlotPool(sanity_config, num_slots=num_slots, layout=sanity_layout)

    arena = pool.slot_arena
    assert arena.dtype == torch.uint8
    assert arena.numel() == num_slots * sanity_layout.slot_num_bytes
    arena_base = arena.data_ptr()

    # Nothing resident before the first step; the device table starts all-miss.
    assert torch.equal(pool.expert_slot_table, torch.full_like(pool.expert_slot_table, UNRESIDENT_SLOT_ID))
    assert pool.step_slot_ids_buffer.shape == (sanity_config.top_k,)
    assert pool.step_slot_ids_buffer.dtype == torch.int32

    for slot_id in range(num_slots):
        for spec in sanity_layout.specs:
            views = pool.weight_views(slot_id) if spec.kind == "packed_fp4" else pool.scale_views(slot_id)
            view = views[spec.name]
            expected_offset = slot_id * sanity_layout.slot_num_bytes + spec.offset_bytes
            assert view.data_ptr() - arena_base == expected_offset, (slot_id, spec.name)
            assert expected_offset % SLOT_REGION_ALIGN_BYTES == 0, (slot_id, spec.name)
            assert tuple(view.shape) == spec.view_shape
            assert view.dtype == torch.uint8
            assert view.is_contiguous()
        # The provider-keyed views alias the very same storage regions.
        for spec in sanity_layout.specs:
            param_view = pool.param_views(slot_id)[spec.param_key]
            assert param_view.data_ptr() - arena_base == slot_id * sanity_layout.slot_num_bytes + spec.offset_bytes

    assert pool.resident_expert_count == 0


def test_step_lock_prevents_eviction_collision(
    sanity_config: DeepSeekV4MoEConfig,
    sanity_layout: ExpertTensorLayout,
    mock_provider_factory,
) -> None:
    pool = StaticExpertSlotPool(sanity_config, num_slots=sanity_config.top_k, layout=sanity_layout)
    provider = mock_provider_factory(sanity_layout, layer_ids=(0,), num_experts=sanity_config.num_routed_experts)

    first_ids = [10, 11, 12, 13, 14, 15]
    reservation = pool.acquire_for_step(0, first_ids, provider)
    first_slots = {expert_id: pool.slot_of(0, expert_id) for expert_id in first_ids}
    assert len(set(first_slots.values())) == sanity_config.top_k
    baseline = [pool.slot_arena.data_ptr(), pool.expert_slot_table.data_ptr(), pool.step_slot_ids_buffer.data_ptr()]

    # Same step, all 6 slots locked, 6 fresh experts requested: must refuse
    # atomically instead of evicting any locked slot.
    with pytest.raises(SlotExhaustionError, match="evictable"):
        pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], provider)

    # State is untouched: residency, bytes, counters.
    assert {e: pool.slot_of(0, e) for e in first_ids} == first_slots
    assert pool.stats.evictions == 0
    assert pool.resident_expert_count == sanity_config.top_k
    for expert_id, slot_id in first_slots.items():
        assert torch.equal(pool.weight_views(slot_id)["w1"], provider.pinned_cpu_weight(0, expert_id, "w1"))
    assert [
        pool.slot_arena.data_ptr(),
        pool.expert_slot_table.data_ptr(),
        pool.step_slot_ids_buffer.data_ptr(),
    ] == baseline

    pool.release_step(reservation)

    # Locks released: the same request now succeeds by evicting exactly the
    # 6 (now-unlocked) LRU-oldest slots, reusing all of them.
    second_ids = [100, 101, 102, 103, 104, 105]
    second = pool.acquire_for_step(0, second_ids, provider)
    assert set(second.slot_ids) == set(first_slots.values())
    assert all(pool.slot_of(0, expert_id) == -1 for expert_id in first_ids)
    assert pool.stats.evictions == sanity_config.top_k
    for expert_id, slot_id in zip(second_ids, second.slot_ids):
        assert torch.equal(pool.weight_views(slot_id)["w2"], provider.pinned_cpu_weight(0, expert_id, "w2"))
    pool.release_step(second)

    # --- Free slots do not weaken the lock: 8 slots, 6 locked, 2 free. ---
    wide_pool = StaticExpertSlotPool(sanity_config, num_slots=8, layout=sanity_layout)
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
    hit_pool = StaticExpertSlotPool(sanity_config, num_slots=sanity_config.top_k, layout=sanity_layout)
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


def test_hash_layer_routing(sanity_config: DeepSeekV4MoEConfig, sanity_layout: ExpertTensorLayout) -> None:
    generator = torch.Generator().manual_seed(1234)
    # The checkpoint ships one tid2eid per hash layer: [num_hash_layers, vocab, top_k].
    tid2eid = torch.randint(
        0,
        sanity_config.num_routed_experts,
        (sanity_config.num_hash_layers, sanity_config.vocab_size, sanity_config.top_k),
        dtype=torch.int32,
        generator=generator,
    )
    assert not torch.equal(tid2eid[0], tid2eid[1])  # per-layer tables are independent

    resolver = HashRouteResolver(sanity_config, tid2eid)
    assert isinstance(resolver, RouteResolverProtocol)
    token_ids = torch.tensor([5, 5, 4095, 0, 123], dtype=torch.int64)

    for layer_idx in range(sanity_config.num_hash_layers):
        out = torch.full((len(token_ids), sanity_config.top_k), -1, dtype=torch.int32)
        resolver.resolve(layer_idx, token_ids, out)
        assert torch.equal(out, tid2eid[layer_idx][token_ids]), f"layer {layer_idx} hash mismatch"
        again = torch.full_like(out, -2)
        resolver.resolve(layer_idx, token_ids, again)
        assert torch.equal(out, again)  # deterministic by construction

    with pytest.raises(ValueError, match="score-gated"):
        resolver.resolve(
            sanity_config.num_hash_layers, token_ids, torch.empty(1, sanity_config.top_k, dtype=torch.int32)
        )

    # --- Score-gated layers: bias shifts selection, reference-checked. ---
    gate_bias = torch.zeros(sanity_config.num_layers, sanity_config.num_routed_experts, dtype=torch.float32)
    score_resolver = ScoreRouteResolver(sanity_config, gate_bias, max_batch_tokens=8)
    scores = torch.zeros(1, sanity_config.num_routed_experts, dtype=torch.float32)
    scores[0, :7] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.9])

    routing_out = torch.empty(1, sanity_config.top_k, dtype=torch.int64)
    score_resolver.resolve(3, scores, routing_out)
    assert torch.equal(routing_out, torch.topk(scores + gate_bias[3], sanity_config.top_k, dim=-1).indices)
    assert sorted(routing_out[0].tolist()) == [0, 1, 2, 3, 4, 5]

    # Bias demotes the strongest raw expert -- selection changes, scores do not.
    gate_bias[3, 0] = -100.0
    score_resolver.resolve(3, scores, routing_out)
    assert sorted(routing_out[0].tolist()) == [1, 2, 3, 4, 5, 6]
    assert torch.equal(routing_out, torch.topk(scores + gate_bias[3], sanity_config.top_k, dim=-1).indices)

    with pytest.raises(ValueError, match="hash-routed"):
        score_resolver.resolve(0, scores, routing_out)

    # --- End-to-end: hash-resolved ids drive the slot pool on a hash layer. ---
    pool = StaticExpertSlotPool(sanity_config, num_slots=8, layout=sanity_layout)
    provider = _table_backed_provider(sanity_layout)
    routed = torch.full((4, sanity_config.top_k), -1, dtype=torch.int32)
    resolver.resolve(1, torch.tensor([7, 9, 11, 13], dtype=torch.int64), routed)
    reservation = pool.acquire_for_step(1, routed[0].tolist(), provider)
    assert len(reservation.slot_ids) == sanity_config.top_k
    pool.release_step(reservation)


def _table_backed_provider(layout: ExpertTensorLayout):
    from ..tests.conftest import MockWeightProvider

    return MockWeightProvider(layout, layer_ids=(1,), num_experts=256)
