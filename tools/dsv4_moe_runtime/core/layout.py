"""Byte-level layout schema for one isomorphic expert slot.

Memory invariant: every tensor region inside a slot starts at a 128-byte
aligned offset and ``ExpertTensorLayout.slot_num_bytes`` is the aligned total.
Three storage families are supported, one ``kind`` per region:

* **block-32 FP4 + E8M0 scales** -- the production DeepSeek-V4 Flash geometry
  (13,369,344 bytes = 12.75 MiB per slot at hidden 4096 / intermediate 2048);
* **dense BF16** -- unquantized checkpoints such as DeepSeek-V2-Lite
  (17,301,504 bytes = 16.50 MiB per slot at hidden 2048 / intermediate 1408),
  where an expert owns three projections and no scale regions at all.
* **FP8 E4M3FN + FP32 tensor scales** -- routed V2-Lite projections
  (8,651,264 bytes per slot, including aligned scale regions and padding).

Slot views are always ``uint8`` byte windows; ``ExpertTensorSpec.view_shape``
is that byte shape and ``logical_shape`` is the element shape a compute layer
recovers with ``Tensor.view(dtype)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..core.config import EXPERT_PARAM_NAMES, DeepSeekV4MoEConfig

FP4_ELEMS_PER_BYTE = 2
FP4_BLOCK_SIZE = 32
E8M0_SCALE_NUM_BYTES = 1
BF16_NUM_BYTES = 2
SLOT_REGION_ALIGN_BYTES = 128

PACKED_FP4_KIND = "packed_fp4"
E8M0_SCALE_KIND = "e8m0_scale"
DENSE_BF16_KIND = "dense_bf16"
FP8_KIND = "fp8_e4m3fn"
FP32_SCALE_KIND = "fp32_scale"
FP4_BLOCK32_KINDS: tuple[str, ...] = (PACKED_FP4_KIND, E8M0_SCALE_KIND)
DENSE_BF16_KINDS: tuple[str, ...] = (DENSE_BF16_KIND,)
KNOWN_TENSOR_KINDS: tuple[str, ...] = (PACKED_FP4_KIND, E8M0_SCALE_KIND, DENSE_BF16_KIND, FP8_KIND, FP32_SCALE_KIND)


@dataclass(frozen=True)
class ExpertTensorSpec:
    """Byte-level geometry of one expert parameter region inside a slot.

    ``cols`` is always the *logical* element count along the reduction dim;
    storage compaction (2 FP4 nibbles/byte), block scaling (1 E8M0 byte per 32
    elements) or widening (2 bytes per BF16 element) is derived from ``kind``,
    never hand-written per tensor.
    """

    name: str
    kind: str  # one of KNOWN_TENSOR_KINDS
    rows: int
    cols: int
    offset_bytes: int

    @property
    def stored_cols(self) -> int:
        """Bytes per row of the region (the second dim of ``view_shape``)."""
        if self.kind == PACKED_FP4_KIND:
            if self.cols % FP4_ELEMS_PER_BYTE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by {FP4_ELEMS_PER_BYTE}")
            return self.cols // FP4_ELEMS_PER_BYTE
        if self.kind == E8M0_SCALE_KIND:
            if self.cols % FP4_BLOCK_SIZE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by block {FP4_BLOCK_SIZE}")
            return self.cols // FP4_BLOCK_SIZE * E8M0_SCALE_NUM_BYTES
        if self.kind == DENSE_BF16_KIND:
            return self.cols * BF16_NUM_BYTES
        if self.kind == FP8_KIND:
            return self.cols
        if self.kind == FP32_SCALE_KIND:
            return self.cols * 4
        raise ValueError(f"unknown tensor kind: {self.kind}")

    @property
    def num_bytes(self) -> int:
        return self.rows * self.stored_cols

    @property
    def view_shape(self) -> tuple[int, int]:
        """Byte shape of the region's ``uint8`` slot view."""
        return (self.rows, self.stored_cols)

    @property
    def logical_shape(self) -> tuple[int, int]:
        """Element shape after ``Tensor.view(dtype)`` at compute bind time."""
        if self.kind in (DENSE_BF16_KIND, FP8_KIND, FP32_SCALE_KIND):
            return (self.rows, self.cols)
        return self.view_shape  # FP4 nibble pairs and E8M0 bytes stay byte-shaped

    @property
    def is_scale(self) -> bool:
        """Whether the region holds block scales rather than weight storage."""
        return self.kind in (E8M0_SCALE_KIND, FP32_SCALE_KIND)

    @property
    def param_key(self) -> str:
        """Key used with :class:`dsv4_moe_runtime.protocols.provider.WeightProviderProtocol`."""
        return self.name + "_scale" if self.is_scale else self.name


@dataclass(frozen=True)
class ExpertTensorLayout:
    """Declarative schema for one isomorphic expert slot.

    Drives arena sizing, per-slot view slicing and the schema-driven DMA loop,
    so w1/w2/w3 never get hand-copied anywhere (DRY).
    """

    specs: tuple[ExpertTensorSpec, ...]
    slot_num_bytes: int

    @classmethod
    def for_expert_geometry(
        cls,
        hidden_size: int,
        moe_intermediate_size: int,
        kinds: Sequence[str] = FP4_BLOCK32_KINDS,
    ) -> ExpertTensorLayout:
        """Lay out w1/w2/w3 with one region per ``kind``, 128-byte aligned.

        The projection shapes are the architecture-independent part (gate/up
        are ``[intermediate, hidden]``, down is ``[hidden, intermediate]``);
        ``kinds`` selects the storage family -- ``FP4_BLOCK32_KINDS`` for
        packed FP4 plus E8M0 scales, ``DENSE_BF16_KINDS`` for a BF16
        checkpoint with no scale regions.
        """
        unknown = [kind for kind in kinds if kind not in KNOWN_TENSOR_KINDS]
        if unknown:
            raise ValueError(f"unknown tensor kinds {unknown}; known: {list(KNOWN_TENSOR_KINDS)}")
        inter = moe_intermediate_size
        hidden = hidden_size
        projection_shapes: dict[str, tuple[int, int]] = {
            "w1": (inter, hidden),
            "w2": (hidden, inter),
            "w3": (inter, hidden),
        }
        specs: list[ExpertTensorSpec] = []
        cursor = 0
        for name in EXPERT_PARAM_NAMES:
            rows, cols = projection_shapes[name]
            for kind in kinds:
                spec = ExpertTensorSpec(name=name, kind=kind, rows=rows, cols=cols, offset_bytes=_align_up(cursor))
                specs.append(spec)
                cursor = spec.offset_bytes + spec.num_bytes
        return cls(specs=tuple(specs), slot_num_bytes=_align_up(cursor))

    @classmethod
    def for_deepseek_v4_flash(cls, config: DeepSeekV4MoEConfig) -> ExpertTensorLayout:
        return cls.for_expert_geometry(config.hidden_size, config.moe_intermediate_size, FP4_BLOCK32_KINDS)

    @classmethod
    def for_dense_bf16(cls, config: DeepSeekV4MoEConfig) -> ExpertTensorLayout:
        """Unquantized BF16 expert slot (DeepSeek-V2-Lite and friends)."""
        return cls.for_expert_geometry(config.hidden_size, config.moe_intermediate_size, DENSE_BF16_KINDS)

    @classmethod
    def for_fp8(cls, config: DeepSeekV4MoEConfig) -> ExpertTensorLayout:
        """E4M3FN projections and scalar FP32 decoding scales, 512-byte slots."""
        dense = cls.for_dense_bf16(config)
        specs = []
        cursor = 0
        for weight in dense.specs:
            spec = ExpertTensorSpec(weight.name, FP8_KIND, weight.rows, weight.cols, _align_up(cursor))
            specs.append(spec)
            scale = ExpertTensorSpec(weight.name, FP32_SCALE_KIND, 1, 1, _align_up(spec.offset_bytes + spec.num_bytes))
            specs.append(scale)
            cursor = scale.offset_bytes + scale.num_bytes
        return cls(tuple(specs), _align_up(cursor, 512))

    @classmethod
    def for_shared_expert(
        cls,
        config: DeepSeekV4MoEConfig,
        num_shared_experts: int,
        kinds: Sequence[str] = DENSE_BF16_KINDS,
    ) -> ExpertTensorLayout:
        """The shared-expert module: one expert of ``n x moe_intermediate_size`` width.

        DeepSeek exports ``n_shared_experts`` as a *single* fused module rather
        than n separate ones, so DeepSeek-V2-Lite's two shared experts ship as
        one set of ``(2816, 2048)`` / ``(2048, 2816)`` projections -- 2816 being
        ``2 x 1408``. The layout is therefore the routed geometry with a wider
        intermediate, which is what makes :meth:`slots_per_region` come out to a
        whole number and lets the shared expert live in paired routed slots
        instead of a second, differently-sized pool.
        """
        if num_shared_experts <= 0:
            raise ValueError(f"num_shared_experts must be positive, got {num_shared_experts}")
        return cls.for_expert_geometry(config.hidden_size, config.moe_intermediate_size * num_shared_experts, kinds)

    def slots_per_region(self, slot_layout: ExpertTensorLayout) -> int:
        """How many ``slot_layout`` slots this layout occupies, exactly.

        Refuses a non-integral ratio rather than rounding up. A shared expert
        that did not tile the routed slot size would leave a partial slot whose
        tail belongs to neither region -- reachable from both the routed free
        list and the shared views, which is silent corruption rather than a
        wasted page.
        """
        if self.slot_num_bytes % slot_layout.slot_num_bytes:
            raise ValueError(
                f"a {self.slot_num_bytes}-byte region does not tile the {slot_layout.slot_num_bytes}-byte slot "
                f"({self.slot_num_bytes / slot_layout.slot_num_bytes:.4f} slots); paired reservation needs a "
                "whole-number ratio"
            )
        return self.slot_num_bytes // slot_layout.slot_num_bytes

    def slice_region_views(self, arena: torch.Tensor, byte_offset: int) -> dict[str, torch.Tensor]:
        """Pre-slice this layout's parameter views at an arbitrary arena offset.

        ``slice_slot_views`` indexes by slot id, which assumes the region *is* a
        slot. A shared expert spans several, so it is placed by byte offset and
        keyed by ``param_key`` -- the same keys the provider fills.
        """
        if byte_offset % SLOT_REGION_ALIGN_BYTES:
            raise ValueError(f"region offset {byte_offset} is not {SLOT_REGION_ALIGN_BYTES}-byte aligned")
        base = arena.narrow(0, byte_offset, self.slot_num_bytes)
        return {
            spec.param_key: base.narrow(0, spec.offset_bytes, spec.num_bytes).view(spec.view_shape)
            for spec in self.specs
        }

    def spec_for(self, param_key: str) -> ExpertTensorSpec:
        for spec in self.specs:
            if spec.param_key == param_key:
                return spec
        raise KeyError(f"layout has no parameter {param_key!r}; known: {[s.param_key for s in self.specs]}")

    def slice_slot_views(
        self, arena: torch.Tensor, slot_id: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pre-slice one slot into (weight storage, block scale) views.

        Called once per slot at init; the decode loop only *reads* the dicts.
        Views are ``uint8`` byte windows (byte storage of ``float4_e2m1fn_x2``,
        ``float8_e8m0fnu`` or ``bfloat16``); compute layers reinterpret them
        with ``Tensor.view(dtype)`` at bind time. A dense BF16 layout carries no
        scale regions, so the second dict comes back empty.
        """
        slot_base = arena.narrow(0, slot_id * self.slot_num_bytes, self.slot_num_bytes)
        weights: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        for spec in self.specs:
            region = slot_base.narrow(0, spec.offset_bytes, spec.num_bytes).view(spec.view_shape)
            (scales if spec.is_scale else weights)[spec.name] = region
        return weights, scales


def _align_up(value: int, alignment: int = SLOT_REGION_ALIGN_BYTES) -> int:
    if value % alignment == 0:
        return value
    return (value // alignment + 1) * alignment
