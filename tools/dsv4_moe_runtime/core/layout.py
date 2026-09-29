"""Byte-level layout schema for one isomorphic expert slot.

Memory invariant: every tensor region inside a slot starts at a 128-byte
aligned offset and ``ExpertTensorLayout.slot_num_bytes`` is the aligned total
(13,369,344 bytes = 12.75 MiB for the production DeepSeek-V4 Flash geometry:
three block-32 FP4 packed projections with E8M0 scales).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.config import EXPERT_PARAM_NAMES, DeepSeekV4MoEConfig

FP4_ELEMS_PER_BYTE = 2
FP4_BLOCK_SIZE = 32
E8M0_SCALE_NUM_BYTES = 1
SLOT_REGION_ALIGN_BYTES = 128


@dataclass(frozen=True)
class ExpertTensorSpec:
    """Byte-level geometry of one expert parameter region inside a slot.

    ``cols`` is always the *logical* FP4 element count along the reduction dim;
    storage compaction (2 nibbles/byte) or block scaling (1 E8M0 byte per 32
    elements) is derived, never hand-written per tensor.
    """

    name: str
    kind: str  # "packed_fp4" or "e8m0_scale"
    rows: int
    cols: int
    offset_bytes: int

    @property
    def stored_cols(self) -> int:
        if self.kind == "packed_fp4":
            if self.cols % FP4_ELEMS_PER_BYTE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by {FP4_ELEMS_PER_BYTE}")
            return self.cols // FP4_ELEMS_PER_BYTE
        if self.kind == "e8m0_scale":
            if self.cols % FP4_BLOCK_SIZE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by block {FP4_BLOCK_SIZE}")
            return self.cols // FP4_BLOCK_SIZE
        raise ValueError(f"unknown tensor kind: {self.kind}")

    @property
    def num_bytes(self) -> int:
        return self.rows * self.stored_cols * E8M0_SCALE_NUM_BYTES

    @property
    def view_shape(self) -> tuple[int, int]:
        return (self.rows, self.stored_cols)

    @property
    def param_key(self) -> str:
        """Key used with :class:`dsv4_moe_runtime.protocols.provider.WeightProviderProtocol`."""
        return self.name if self.kind == "packed_fp4" else self.name + "_scale"


@dataclass(frozen=True)
class ExpertTensorLayout:
    """Declarative schema for one isomorphic expert slot.

    Drives arena sizing, per-slot view slicing and the schema-driven DMA loop,
    so w1/w2/w3 never get hand-copied anywhere (DRY).
    """

    specs: tuple[ExpertTensorSpec, ...]
    slot_num_bytes: int

    @classmethod
    def for_deepseek_v4_flash(cls, config: DeepSeekV4MoEConfig) -> ExpertTensorLayout:
        inter = config.moe_intermediate_size
        hidden = config.hidden_size
        projection_shapes: dict[str, tuple[int, int]] = {
            "w1": (inter, hidden),
            "w2": (hidden, inter),
            "w3": (inter, hidden),
        }
        specs: list[ExpertTensorSpec] = []
        cursor = 0
        for name in EXPERT_PARAM_NAMES:
            rows, cols = projection_shapes[name]
            for kind in ("packed_fp4", "e8m0_scale"):
                spec = ExpertTensorSpec(name=name, kind=kind, rows=rows, cols=cols, offset_bytes=_align_up(cursor))
                specs.append(spec)
                cursor = spec.offset_bytes + spec.num_bytes
        return cls(specs=tuple(specs), slot_num_bytes=_align_up(cursor))

    def spec_for(self, param_key: str) -> ExpertTensorSpec:
        for spec in self.specs:
            if spec.param_key == param_key:
                return spec
        raise KeyError(f"layout has no parameter {param_key!r}; known: {[s.param_key for s in self.specs]}")

    def slice_slot_views(
        self, arena: torch.Tensor, slot_id: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pre-slice one slot into (packed weights, e8m0 scales) views.

        Called once per slot at init; the decode loop only *reads* the dicts.
        Views are ``uint8`` byte windows (itemsize-1 storage of both
        ``float4_e2m1fn_x2`` and ``float8_e8m0fnu``); compute layers reinterpret
        them with ``Tensor.view(dtype)`` at bind time.
        """
        slot_base = arena.narrow(0, slot_id * self.slot_num_bytes, self.slot_num_bytes)
        weights: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        for spec in self.specs:
            region = slot_base.narrow(0, spec.offset_bytes, spec.num_bytes).view(spec.view_shape)
            (weights if spec.kind == "packed_fp4" else scales)[spec.name] = region
        return weights, scales


def _align_up(value: int, alignment: int = SLOT_REGION_ALIGN_BYTES) -> int:
    if value % alignment == 0:
        return value
    return (value // alignment + 1) * alignment
