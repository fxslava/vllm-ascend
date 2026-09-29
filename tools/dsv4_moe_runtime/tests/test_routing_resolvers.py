"""Hash (tid2eid, layers 0-2) vs score-gated (bias top-k, layers 3+) routing."""

from __future__ import annotations

import pytest
import torch

from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..protocols.router import RouteResolverProtocol
from ..routing.hash_router import HashRouteResolver
from ..routing.score_router import ScoreRouteResolver


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
