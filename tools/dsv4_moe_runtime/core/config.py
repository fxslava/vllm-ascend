"""Geometry configuration for the DeepSeek-V4 Flash MoE static runtime.

Pure data with no torch import and no hardware binding (SRP): everything in
this module is unit-testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPERT_PARAM_NAMES: tuple[str, ...] = ("w1", "w2", "w3")
SCALE_PARAM_SUFFIX = "_scale"


@dataclass(frozen=True)
class DeepSeekV4MoEConfig:
    """Routed-expert geometry, mirroring ``C:\\DeepSeekV4\\config.json``.

    ``num_layers`` counts the decoder blocks that own routed experts (the MTP
    block, ``mtp.0``, owns another identical MoE and is served by giving this
    pool ``num_layers=44`` or by a second pool instance).
    """

    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_routed_experts: int = 256
    top_k: int = 6
    num_layers: int = 43
    num_hash_layers: int = 3
    vocab_size: int = 129280


SANITY_GEOMETRY = DeepSeekV4MoEConfig(
    hidden_size=128,
    moe_intermediate_size=64,
    num_routed_experts=256,
    top_k=6,
    num_layers=43,
    num_hash_layers=3,
    vocab_size=4096,
)
