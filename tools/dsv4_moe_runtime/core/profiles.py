"""Architecture presets: routed-expert geometry + slot layout + budget defaults.

One :class:`MoELayoutProfile` bundles everything a harness needs to target a
concrete checkpoint family, so no benchmark, test or provider ever hard-codes a
geometry again (OCP: a new target is a new registry entry, not a new branch):

* ``geometry`` -- the routed-expert dimensions the slot pool and router trace
  run on (``MoEGeometry``);
* ``expert_kinds`` -- the storage family of one expert slot, which fixes
  ``slot_num_bytes`` (``FP4_BLOCK32_KINDS`` for DeepSeek-V4 Flash,
  ``DENSE_BF16_KINDS`` for DeepSeek-V2-Lite);
* ``backbone`` -- the *always-resident* parameters (embeddings, MLA
  projections, the dense prefix layers and the shared experts). Shared experts
  live here by design: they are activated by every token, so they stay in
  device memory permanently and are never candidates for eviction;
* pool/transit defaults and an advisory pool-slot range for the target device.

Layer indices: ``geometry.num_layers`` counts *all* decoder blocks, while only
``moe_layer_ids`` own routed experts -- DeepSeek-V2-Lite replaces layer 0 with
a dense MLP, so its trace runs over layers 1..26.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .config import DSV2_LITE_GEOMETRY, SANITY_GEOMETRY, DeepSeekV4MoEConfig, MoEGeometry
from .layout import DENSE_BF16_KINDS, FP4_BLOCK32_KINDS, ExpertTensorLayout

DSV4_FLASH_PROFILE = "dsv4-flash"
DSV2_LITE_PROFILE = "dsv2-lite"
SANITY_PROFILE = "sanity"

#: Checkpoint tensor-naming schemes (see ``hardware.safetensors_provider``).
NAMING_DSV4_FLAT = "dsv4-flat"
NAMING_HF_DEEPSEEK = "hf-deepseek"

BYTES_PER_BF16 = 2


@dataclass(frozen=True)
class BackboneProfile:
    """Always-resident parameter geometry (never staged, never evicted).

    ``num_bytes`` is the device-memory reservation the routed-expert pool has to
    coexist with: token embeddings and the LM head, per-layer MLA projections,
    the dense prefix MLPs and the shared experts of every MoE layer.
    """

    hidden_size: int
    vocab_size: int
    num_layers: int
    num_attention_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    q_lora_rank: int | None
    dense_intermediate_size: int
    first_k_dense_layers: int
    num_shared_experts: int
    moe_intermediate_size: int
    bytes_per_element: int = BYTES_PER_BF16
    tied_word_embeddings: bool = False

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers - self.first_k_dense_layers

    @property
    def embedding_num_bytes(self) -> int:
        matrices = 1 if self.tied_word_embeddings else 2  # embed_tokens (+ lm_head)
        return matrices * self.vocab_size * self.hidden_size * self.bytes_per_element

    @property
    def attention_num_bytes(self) -> int:
        """MLA projections of every layer (q, kv down/up, output)."""
        heads = self.num_attention_heads
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            q_elements = self.hidden_size * heads * q_head_dim
        else:
            q_elements = self.hidden_size * self.q_lora_rank + self.q_lora_rank * heads * q_head_dim
        kv_a_elements = self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_b_elements = self.kv_lora_rank * heads * (self.qk_nope_head_dim + self.v_head_dim)
        out_elements = heads * self.v_head_dim * self.hidden_size
        per_layer = q_elements + kv_a_elements + kv_b_elements + out_elements
        return self.num_layers * per_layer * self.bytes_per_element

    @property
    def dense_mlp_num_bytes(self) -> int:
        """The dense prefix layers' gate/up/down MLPs."""
        per_layer = 3 * self.hidden_size * self.dense_intermediate_size
        return self.first_k_dense_layers * per_layer * self.bytes_per_element

    @property
    def shared_expert_num_bytes(self) -> int:
        """Shared experts of every MoE layer: resident forever, never evicted."""
        shared_intermediate = self.moe_intermediate_size * self.num_shared_experts
        per_layer = 3 * self.hidden_size * shared_intermediate
        return self.num_moe_layers * per_layer * self.bytes_per_element

    @property
    def num_bytes(self) -> int:
        return (
            self.embedding_num_bytes
            + self.attention_num_bytes
            + self.dense_mlp_num_bytes
            + self.shared_expert_num_bytes
        )

    def breakdown(self) -> dict[str, int]:
        return {
            "embeddings": self.embedding_num_bytes,
            "attention": self.attention_num_bytes,
            "dense_mlp": self.dense_mlp_num_bytes,
            "shared_experts": self.shared_expert_num_bytes,
            "total": self.num_bytes,
        }


@dataclass(frozen=True)
class MoELayoutProfile:
    """One named architecture target of the static MoE runtime."""

    name: str
    description: str
    geometry: MoEGeometry
    expert_kinds: tuple[str, ...]
    default_pool_slots: int
    default_transit_slots: int
    default_hot_experts: int
    checkpoint_naming: str = NAMING_DSV4_FLAT
    backbone: BackboneProfile | None = None
    pool_slot_advisory: tuple[int, int] | None = field(default=None)

    @property
    def expert_layout(self) -> ExpertTensorLayout:
        return ExpertTensorLayout.for_expert_geometry(
            self.geometry.hidden_size, self.geometry.moe_intermediate_size, self.expert_kinds
        )

    @property
    def first_moe_layer(self) -> int:
        return 0 if self.backbone is None else self.backbone.first_k_dense_layers

    @property
    def num_moe_layers(self) -> int:
        return self.geometry.num_layers - self.first_moe_layer

    @property
    def moe_layer_ids(self) -> range:
        return range(self.first_moe_layer, self.geometry.num_layers)

    @property
    def num_shared_experts(self) -> int:
        return 0 if self.backbone is None else self.backbone.num_shared_experts

    @property
    def backbone_num_bytes(self) -> int:
        return 0 if self.backbone is None else self.backbone.num_bytes

    @property
    def routed_expert_num_bytes(self) -> int:
        """Every routed expert of every MoE layer (the offloadable working set)."""
        return self.num_moe_layers * self.geometry.num_routed_experts * self.expert_layout.slot_num_bytes


DSV2_LITE_BACKBONE = BackboneProfile(
    hidden_size=2048,
    vocab_size=102400,
    num_layers=27,
    num_attention_heads=16,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
    q_lora_rank=None,  # V2-Lite projects q directly from hidden
    dense_intermediate_size=10944,
    first_k_dense_layers=1,  # first_k_dense_replace: layer 0 is a plain MLP
    num_shared_experts=2,
    moe_intermediate_size=1408,
    bytes_per_element=BYTES_PER_BF16,
)

LAYOUT_PROFILES: dict[str, MoELayoutProfile] = {
    DSV4_FLASH_PROFILE: MoELayoutProfile(
        name=DSV4_FLASH_PROFILE,
        description="DeepSeek-V4 Flash, block-32 FP4 + E8M0 scales (12.75 MiB/slot)",
        geometry=DeepSeekV4MoEConfig(),
        expert_kinds=FP4_BLOCK32_KINDS,
        default_pool_slots=3200,
        default_transit_slots=2048,
        default_hot_experts=64,  # a quarter of the 256 routed experts
        checkpoint_naming=NAMING_DSV4_FLAT,
        # The Flash backbone (FP8 MLA + shared expert) is not modeled here:
        # its config is not public, and a guessed reservation would corrupt the
        # memory budget. The guard therefore counts only the slot pool on this
        # target and says so in the plan.
        backbone=None,
    ),
    DSV2_LITE_PROFILE: MoELayoutProfile(
        name=DSV2_LITE_PROFILE,
        description="DeepSeek-V2-Lite 16B, dense BF16 experts (16.50 MiB/slot)",
        geometry=DSV2_LITE_GEOMETRY,
        expert_kinds=DENSE_BF16_KINDS,
        # RTX 5070 (12 GiB) budget: 384 slots = 6.19 GiB VRAM pool,
        # 256 transit slots = 4.13 GiB pinned host DDR.
        default_pool_slots=384,
        default_transit_slots=256,
        default_hot_experts=16,  # a quarter of the 64 routed experts
        checkpoint_naming=NAMING_HF_DEEPSEEK,
        backbone=DSV2_LITE_BACKBONE,
        pool_slot_advisory=(384, 512),
    ),
    SANITY_PROFILE: MoELayoutProfile(
        name=SANITY_PROFILE,
        description="13 KiB/slot sanity geometry for CPU dry runs",
        geometry=SANITY_GEOMETRY,
        expert_kinds=FP4_BLOCK32_KINDS,
        # Keeps the dsv4-flash slot counts so `--small-geometry` alone
        # reproduces the historical plan; `--dry-run` narrows them to 16/64.
        default_pool_slots=3200,
        default_transit_slots=2048,
        default_hot_experts=64,
        checkpoint_naming=NAMING_DSV4_FLAT,
        backbone=None,
    ),
}

PROFILE_NAMES: tuple[str, ...] = tuple(LAYOUT_PROFILES)


def profile_for(name: str) -> MoELayoutProfile:
    try:
        return LAYOUT_PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown layout profile {name!r}; known: {list(PROFILE_NAMES)}") from None


def describe_profiles(names: Sequence[str] = PROFILE_NAMES) -> str:
    return "; ".join(f"{name}: {profile_for(name).description}" for name in names)
