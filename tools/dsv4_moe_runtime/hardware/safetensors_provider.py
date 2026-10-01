"""Safetensors-backed expert weight provider with streaming ingestion.

Parses the safetensors container format directly (8-byte little-endian header
length, JSON header, contiguous byte payload) -- no third-party dependency --
and binds every routed-expert tensor to the exact slot layout:

* per-parameter byte spans must match ``ExpertTensorLayout`` exactly
  (DeepSeek-V4 Flash w1/w3: 4,194,304 packed-FP4 bytes + 262,144 E8M0 scale
  bytes; w2: 4,194,304 + 262,144, summing to the 13,369,344-byte expert slot --
  DeepSeek-V2-Lite instead binds three 5,767,168-byte BF16 projections into a
  17,301,504-byte slot and has no scale regions);
* every span offset must be 128-byte aligned (strict mode, the packing
  contract of this runtime). Upstream Hugging Face checkpoints only guarantee
  8-byte alignment, so ``strict_alignment=False`` is the right setting for them;
* tensor names come from a :class:`ExpertNamingScheme`, which is what makes the
  same binder serve this runtime's flat ``layers.N.ffn.experts.E.w1.weight``
  keys and Hugging Face's ``model.layers.N.mlp.experts.E.gate_proj.weight``.

Serving paths (``SlotFillProviderProtocol``): ``fill_slot_params`` streams
the file spans through the :class:`StreamingWeightLoader` chunk pool straight
into the pool's pre-sliced HBM views (disk -> pinned DDR -> non-blocking
DMA). ``pinned_cpu_weight`` hands out zero-copy mmap views for the legacy
blocking path and for byte-level verification (pageable host memory by
design on the CPU dry run). Dense-backbone slices stream through
:meth:`stream_dense_into` into the monolithic backbone tensor.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..core.layout import SLOT_REGION_ALIGN_BYTES, ExpertTensorLayout, ExpertTensorSpec
from ..core.profiles import NAMING_DSV4_FLAT, NAMING_HF_DEEPSEEK
from .runtime import DeviceRuntime
from .weight_loader import (
    DEFAULT_EXPERT_CHUNK_BYTES,
    DEFAULT_EXPERT_CHUNKS,
    StreamingWeightLoader,
)

SAFETENSORS_HEADER_LENGTH_BYTES = 8
DENSE_SECTION = "__dense__"


class WeightLayoutMismatchError(RuntimeError):
    """Raised when checkpoint byte spans do not bind to the expert slot layout."""


@dataclass(frozen=True)
class ExpertNamingScheme:
    """How one checkpoint family names an expert's projection tensors."""

    name: str
    template: str  # formatted with layer, expert, projection, suffix
    projections: Mapping[str, str]  # layout spec name -> checkpoint projection name
    weight_suffix: str
    scale_suffix: str
    #: Shared experts are addressed by layer alone -- there is one fused module
    #: per layer, not one per expert id -- so they need their own template.
    #: ``None`` means this checkpoint family has no shared experts to name.
    shared_template: str | None = None

    def _projection(self, spec: ExpertTensorSpec) -> str:
        try:
            return self.projections[spec.name]
        except KeyError:
            raise WeightLayoutMismatchError(
                f"naming scheme {self.name!r} has no projection for layout tensor {spec.name!r}"
            ) from None

    def tensor_name(self, layer_idx: int, expert_id: int, spec: ExpertTensorSpec) -> str:
        return self.template.format(
            layer=layer_idx,
            expert=expert_id,
            projection=self._projection(spec),
            suffix=self.scale_suffix if spec.is_scale else self.weight_suffix,
        )

    def shared_tensor_name(self, layer_idx: int, spec: ExpertTensorSpec) -> str:
        """Name of one projection of ``layer_idx``'s shared-expert module."""
        if self.shared_template is None:
            raise WeightLayoutMismatchError(f"naming scheme {self.name!r} does not name shared experts")
        return self.shared_template.format(
            layer=layer_idx,
            projection=self._projection(spec),
            suffix=self.scale_suffix if spec.is_scale else self.weight_suffix,
        )


DSV4_FLAT_NAMING = ExpertNamingScheme(
    name=NAMING_DSV4_FLAT,
    template="layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}",
    projections={"w1": "w1", "w2": "w2", "w3": "w3"},
    weight_suffix="weight",
    scale_suffix="scale",
)
#: Hugging Face DeepSeek MoE naming (V2/V2-Lite/V3): gate/up/down projections
#: under ``model.layers.N.mlp.experts.E``, block scales as ``weight_scale_inv``.
HF_DEEPSEEK_NAMING = ExpertNamingScheme(
    name=NAMING_HF_DEEPSEEK,
    template="model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}",
    projections={"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"},
    weight_suffix="weight",
    scale_suffix="weight_scale_inv",
    shared_template="model.layers.{layer}.mlp.shared_experts.{projection}.{suffix}",
)
NAMING_SCHEMES: dict[str, ExpertNamingScheme] = {
    DSV4_FLAT_NAMING.name: DSV4_FLAT_NAMING,
    HF_DEEPSEEK_NAMING.name: HF_DEEPSEEK_NAMING,
}


def naming_scheme(naming: str | ExpertNamingScheme) -> ExpertNamingScheme:
    if isinstance(naming, ExpertNamingScheme):
        return naming
    try:
        return NAMING_SCHEMES[naming]
    except KeyError:
        raise KeyError(f"unknown naming scheme {naming!r}; known: {list(NAMING_SCHEMES)}") from None


def parse_safetensors_header(path: str) -> tuple[dict, int, int]:
    """Return ``(header, data_start, file_size)`` for a safetensors container."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(SAFETENSORS_HEADER_LENGTH_BYTES)
        if len(prefix) != SAFETENSORS_HEADER_LENGTH_BYTES:
            raise WeightLayoutMismatchError(f"{path}: truncated safetensors prefix")
        (header_len,) = struct.unpack("<Q", prefix)
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise WeightLayoutMismatchError(f"{path}: truncated safetensors header")
    header = json.loads(header_bytes.decode("utf-8"))
    data_start = SAFETENSORS_HEADER_LENGTH_BYTES + header_len
    if file_size < data_start:
        raise WeightLayoutMismatchError(f"{path}: payload smaller than the header claims")
    header.pop("__metadata__", None)
    return header, data_start, file_size


def _span(header: Mapping[str, Mapping[str, object]], name: str, data_start: int) -> tuple[int, int]:
    try:
        entry = header[name]
        begin, end = entry["data_offsets"]  # type: ignore[index]
        return data_start + int(begin), data_start + int(end)
    except KeyError as exc:
        raise WeightLayoutMismatchError(f"checkpoint is missing tensor {name!r}") from exc


def validate_expert_span(
    spec: ExpertTensorSpec,
    name: str,
    begin: int,
    size: int,
    strict_alignment: bool = True,
    file_size: int | None = None,
) -> None:
    """Byte-level admissibility of one checkpoint span against the slot layout.

    Shared by the single-file and sharded binders so both enforce the identical
    contract: exact span size, in-file span end and (strict mode) 128-byte
    aligned start.
    """
    if size != spec.num_bytes:
        raise WeightLayoutMismatchError(f"{name}: span is {size} bytes, layout requires {spec.num_bytes}")
    if file_size is not None and begin + size > file_size:
        raise WeightLayoutMismatchError(
            f"{name}: span ends at {begin + size}, beyond the {file_size}-byte file (truncated payload?)"
        )
    if strict_alignment and begin % SLOT_REGION_ALIGN_BYTES:
        raise WeightLayoutMismatchError(f"{name}: span offset {begin} is not {SLOT_REGION_ALIGN_BYTES}-byte aligned")


def bind_expert_spans(
    header: Mapping[str, Mapping[str, object]],
    data_start: int,
    layout: ExpertTensorLayout,
    layer_ids: Sequence[int],
    num_experts: int,
    strict_alignment: bool = True,
    file_size: int | None = None,
    naming: str | ExpertNamingScheme = NAMING_DSV4_FLAT,
    expert_ids: Sequence[int] | None = None,
) -> dict[tuple[int, int], dict[str, tuple[int, int]]]:
    """Bind every routed expert's tensor spans to the slot layout.

    Enforces the byte-exact contract: each parameter span matches its
    ``ExpertTensorSpec`` size, every expert's spans sum to exactly one slot, and
    (strict mode) every span starts 128-byte aligned. ``expert_ids`` overrides
    the default ``range(num_experts)`` when a shard holds a subset.
    """
    scheme = naming_scheme(naming)
    ids = range(num_experts) if expert_ids is None else expert_ids
    spans: dict[tuple[int, int], dict[str, tuple[int, int]]] = {}
    for layer_idx in layer_ids:
        for expert_id in ids:
            params: dict[str, tuple[int, int]] = {}
            total = 0
            for spec in layout.specs:
                name = scheme.tensor_name(layer_idx, expert_id, spec)
                begin, end = _span(header, name, data_start)
                size = end - begin
                validate_expert_span(spec, name, begin, size, strict_alignment, file_size)
                params[spec.param_key] = (begin, size)
                total += size
            if total != layout.slot_num_bytes:
                raise WeightLayoutMismatchError(
                    f"layer {layer_idx} expert {expert_id} ({scheme.name}): spans total {total} bytes, "
                    f"slot layout is {layout.slot_num_bytes}"
                )
            spans[(layer_idx, expert_id)] = params
    return spans


def bind_dense_spans(
    header: Mapping[str, Mapping[str, object]],
    data_start: int,
    names: Sequence[str],
    strict_alignment: bool = True,
) -> dict[str, tuple[int, int]]:
    """Bind non-expert (dense backbone) tensor spans by exact declared size."""
    spans: dict[str, tuple[int, int]] = {}
    for name in names:
        begin, end = _span(header, name, data_start)
        size = end - begin
        if strict_alignment and begin % SLOT_REGION_ALIGN_BYTES:
            raise WeightLayoutMismatchError(f"{name}: dense span offset {begin} is not 128-byte aligned")
        if size <= 0:
            raise WeightLayoutMismatchError(f"{name}: dense span is empty")
        spans[name] = (begin, size)
    return spans


class SafetensorsExpertProvider:
    """``WeightProviderProtocol`` over a real safetensors expert checkpoint.

    Construct with a ``StreamingWeightLoader`` bound to the same file for the
    DirectStorage path (disk -> pinned chunks -> non-blocking HBM copies);
    without one, a small expert-sized pool is built automatically. Residency
    is *not* tracked here -- the :class:`StaticExpertSlotPool` decides who is
    resident; this provider is a stateless byte-source with a validated TOC.
    """

    def __init__(
        self,
        runtime: DeviceRuntime,
        file_path: str,
        layout: ExpertTensorLayout,
        layer_ids: Sequence[int],
        num_experts: int,
        loader: StreamingWeightLoader | None = None,
        dense_names: Sequence[str] = (),
        strict_alignment: bool = True,
        naming: str | ExpertNamingScheme = NAMING_DSV4_FLAT,
    ):
        self._runtime = runtime
        self._layout = layout
        self._naming = naming_scheme(naming)
        header, data_start, file_size = parse_safetensors_header(file_path)
        self.expert_spans = bind_expert_spans(
            header,
            data_start,
            layout,
            layer_ids=layer_ids,
            num_experts=num_experts,
            strict_alignment=strict_alignment,
            file_size=file_size,
            naming=self._naming,
        )
        self.dense_spans = bind_dense_spans(header, data_start, dense_names, strict_alignment)
        self._loader = loader or StreamingWeightLoader(
            runtime,
            file_path,
            chunk_bytes=DEFAULT_EXPERT_CHUNK_BYTES,
            num_chunks=DEFAULT_EXPERT_CHUNKS,
        )
        self._file_handle = open(  # noqa: SIM115 - lives for the provider lifetime
            file_path, "rb"
        )  # keeps the mmap-backed views valid
        self.bytes_streamed = 0
        self.experts_served = 0

    @property
    def staged_expert_count(self) -> int:
        return len(self.expert_spans)

    @property
    def loader(self) -> StreamingWeightLoader:
        return self._loader

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        if (layer_idx, expert_id) not in self.expert_spans:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) not present in the checkpoint TOC")

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        """Zero-copy mmap view over the file bytes (pageable host memory)."""
        self.ensure_staged(layer_idx, expert_id)
        begin, size = self.expert_spans[(layer_idx, expert_id)][param_key]
        return self._loader.mmap_view(begin, size)

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int:
        """DirectStorage fill: file -> pinned chunks -> non-blocking HBM copies."""
        self.ensure_staged(layer_idx, expert_id)
        spans = self.expert_spans[(layer_idx, expert_id)]
        moved = 0
        for param_key, destination in views.items():
            begin, size = spans[param_key]
            flat = destination.view(-1)
            self._loader.stream_into(flat, begin, size)
            moved += size
        self.bytes_streamed += moved
        self.experts_served += 1
        return moved

    def dense_names(self) -> list[str]:
        return list(self.dense_spans)

    def stream_dense_into(self, name: str, destination: torch.Tensor) -> int:
        """Stream one dense-backbone tensor into its monolithic HBM target."""
        if name not in self.dense_spans:
            raise KeyError(f"dense tensor {name!r} not present in the checkpoint TOC")
        begin, size = self.dense_spans[name]
        if destination.numel() < size:
            raise ValueError(f"backbone target for {name!r} holds {destination.numel()} bytes < {size}")
        moved = self._loader.stream_into(destination.view(-1), begin, size)
        self.bytes_streamed += moved
        return moved

    def synchronize(self) -> None:
        self._loader.synchronize()

    def close(self) -> None:
        self._file_handle.close()
        self._loader.close()
