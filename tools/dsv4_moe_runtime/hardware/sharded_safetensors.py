"""Sharded safetensors ingestion: an HF checkpoint directory as a byte source.

Upstream checkpoints (DeepSeek-V2-Lite and friends) ship as N safetensors
shards plus a ``model.safetensors.index.json`` weight map. This module turns
such a directory into the two interfaces the static runtime consumes:

* :class:`~tools.dsv4_moe_runtime.protocols.provider.WeightByteSource` --
  ``fill_slot`` materializes one expert's full slot region inside the pinned
  transit window, which is what makes the *exclusive staging* benchmark run on
  real weights: NVMe -> pinned host DDR -> non-blocking H2D DMA into a VRAM
  slot, with no permanent host copy of any expert;
* ``SlotFillProviderProtocol`` -- ``fill_slot_params`` streams the file spans
  straight into the pool's pre-sliced device views (the DirectStorage path,
  bypassing the transit window entirely).

Byte path per span: unbuffered positional read (``O_DIRECT`` / Windows
``FILE_FLAG_NO_BUFFERING``) into a *fixed* pool of pinned host chunks, then an
in-place ``copy_`` into the destination. The chunk hop is not incidental:
unbuffered reads must start on a 4096-byte file sector, while safetensors span
offsets are only 8-byte aligned, so the chunk absorbs the alignment head.

Allocation contract: the chunk pool and every span table are built in
``__init__``; serving performs positional reads and in-place copies only. One
pinned chunk pool is shared by all shard loaders -- safe because fills are
strictly sequential and every ``stream_into`` synchronizes its copy stream
before returning.
"""

from __future__ import annotations

import ctypes
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..core.layout import DENSE_BF16_KIND, FP8_KIND, FP32_SCALE_KIND, ExpertTensorLayout, ExpertTensorSpec
from ..core.profiles import NAMING_HF_DEEPSEEK
from .runtime import DeviceRuntime
from .safetensors_provider import (
    ExpertNamingScheme,
    WeightLayoutMismatchError,
    naming_scheme,
    parse_safetensors_header,
    validate_expert_span,
)
from .weight_loader import (
    DEFAULT_EXPERT_CHUNK_BYTES,
    DEFAULT_EXPERT_CHUNKS,
    IO_ALIGNMENT,
    MODE_AUTO,
    StreamingWeightLoader,
)

SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_SHARD_FILENAME = "model.safetensors"
#: Header dtype strings accepted for a dense BF16 expert region.
BF16_HEADER_DTYPES = ("BF16",)


@dataclass(frozen=True)
class TensorSpan:
    """One tensor's absolute byte span inside one shard file."""

    shard: Path
    begin: int
    num_bytes: int
    dtype: str
    shape: tuple[int, ...]


class SafetensorsShardIndex:
    """Tensor name -> (shard, absolute offset, size) over a checkpoint directory."""

    def __init__(self, spans: Mapping[str, TensorSpan]):
        self._spans = dict(spans)
        self._shards = tuple(dict.fromkeys(span.shard for span in self._spans.values()))

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str]) -> SafetensorsShardIndex:
        """Parse every shard listed by the HF index (or a single-file checkpoint)."""
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"checkpoint directory {root} does not exist")
        index_path = root / SAFETENSORS_INDEX_FILENAME
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            if not weight_map:
                raise WeightLayoutMismatchError(f"{index_path}: index has an empty weight_map")
            shard_names = dict.fromkeys(weight_map.values())
        else:
            shard_names = dict.fromkeys(
                sorted(path.name for path in root.glob("*.safetensors")) or [SINGLE_SHARD_FILENAME]
            )
        spans: dict[str, TensorSpan] = {}
        for shard_name in shard_names:
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"shard {shard_path} referenced by the checkpoint is missing")
            header, data_start, file_size = parse_safetensors_header(str(shard_path))
            for name, entry in header.items():
                begin, end = entry["data_offsets"]
                absolute_begin = data_start + int(begin)
                num_bytes = int(end) - int(begin)
                if absolute_begin + num_bytes > file_size:
                    raise WeightLayoutMismatchError(
                        f"{shard_path}: span of {name!r} ends beyond the {file_size}-byte file"
                    )
                spans[name] = TensorSpan(
                    shard=shard_path,
                    begin=absolute_begin,
                    num_bytes=num_bytes,
                    dtype=str(entry.get("dtype", "")),
                    shape=tuple(int(dim) for dim in entry.get("shape", ())),
                )
        return cls(spans)

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return self._shards

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._spans)

    @property
    def total_bytes(self) -> int:
        return sum(span.num_bytes for span in self._spans.values())

    def has(self, name: str) -> bool:
        return name in self._spans

    def span(self, name: str) -> TensorSpan:
        try:
            return self._spans[name]
        except KeyError:
            raise WeightLayoutMismatchError(f"checkpoint is missing tensor {name!r}") from None


def bind_sharded_expert_spans(
    index: SafetensorsShardIndex,
    layout: ExpertTensorLayout,
    layer_ids: Sequence[int],
    expert_ids: Sequence[int],
    naming: str | ExpertNamingScheme = NAMING_HF_DEEPSEEK,
    strict_alignment: bool = False,
    check_dtypes: bool = True,
) -> dict[tuple[int, int], dict[str, TensorSpan]]:
    """Bind every (layer, expert) to its per-parameter spans across shards.

    Same byte-exact contract as the single-file binder -- per-spec span sizes and
    a per-expert total of exactly one slot -- plus a dtype/shape check, which is
    what catches an FP8 or reshaped checkpoint being bound as dense BF16 before
    any byte moves.
    """
    scheme = naming_scheme(naming)
    bound: dict[tuple[int, int], dict[str, TensorSpan]] = {}
    for layer_idx in layer_ids:
        for expert_id in expert_ids:
            params: dict[str, TensorSpan] = {}
            total = 0
            for spec in layout.specs:
                name = scheme.tensor_name(layer_idx, expert_id, spec)
                span = index.span(name)
                validate_expert_span(spec, name, span.begin, span.num_bytes, strict_alignment)
                if check_dtypes:
                    _validate_dtype_and_shape(spec, name, span)
                params[spec.param_key] = span
                total += span.num_bytes
            if total != sum(spec.num_bytes for spec in layout.specs):
                raise WeightLayoutMismatchError(
                    f"layer {layer_idx} expert {expert_id} ({scheme.name}): spans total {total} bytes, "
                    f"slot layout is {layout.slot_num_bytes}"
                )
            bound[(layer_idx, expert_id)] = params
    return bound


def _validate_dtype_and_shape(spec: ExpertTensorSpec, name: str, span: TensorSpan) -> None:
    required = {FP8_KIND: "F8_E4M3", FP32_SCALE_KIND: "F32"}
    if spec.kind in required and span.dtype != required[spec.kind]:
        raise WeightLayoutMismatchError(f"{name}: expected {required[spec.kind]}, got {span.dtype}")
    if spec.kind == DENSE_BF16_KIND and span.dtype not in BF16_HEADER_DTYPES:
        raise WeightLayoutMismatchError(
            f"{name}: checkpoint dtype {span.dtype!r} is not BF16, but the slot layout is dense BF16"
        )
    if span.shape and tuple(span.shape) != spec.logical_shape:
        raise WeightLayoutMismatchError(
            f"{name}: checkpoint shape {tuple(span.shape)} does not match the layout's {spec.logical_shape}"
        )


class ShardedSafetensorsExpertSource:
    """Real sharded weights as a ``WeightByteSource`` *and* a slot-fill provider."""

    def __init__(
        self,
        runtime: DeviceRuntime,
        index: SafetensorsShardIndex,
        layout: ExpertTensorLayout,
        layer_ids: Sequence[int],
        expert_ids: Sequence[int],
        naming: str | ExpertNamingScheme = NAMING_HF_DEEPSEEK,
        strict_alignment: bool = False,
        chunk_bytes: int = DEFAULT_EXPERT_CHUNK_BYTES,
        num_chunks: int = DEFAULT_EXPERT_CHUNKS,
        io_mode: str = MODE_AUTO,
        check_dtypes: bool = True,
        shared_layout: ExpertTensorLayout | None = None,
    ):
        self._runtime = runtime
        self._layout = layout
        self._index = index
        self._naming = naming_scheme(naming)
        # Shared experts are bound lazily: a run may reserve them for a subset
        # of layers, and binding all 26 up front would check spans nothing asks
        # for. The layout is held because the fill needs it and it must not be
        # the routed one -- see bind_shared_expert.
        self._shared_layout = shared_layout
        self._shared_spans: dict[int, dict[str, TensorSpan]] = {}
        self.shared_fills = 0
        self.expert_spans = bind_sharded_expert_spans(
            index,
            layout,
            layer_ids=layer_ids,
            expert_ids=expert_ids,
            naming=self._naming,
            strict_alignment=strict_alignment,
            check_dtypes=check_dtypes,
        )
        # One pinned chunk pool shared by every shard loader: fills are
        # sequential and each stream_into syncs its copy stream before
        # returning, so a chunk is never read by two transfers at once.
        pin = runtime.supports_pinned_host_memory
        flags: dict[str, object] = {"pin_memory": True} if pin else {}
        self._chunks = [torch.empty(chunk_bytes, dtype=torch.uint8, **flags) for _ in range(num_chunks)]
        self.pinned_chunks = pin
        self._loaders: dict[Path, StreamingWeightLoader] = {
            shard: StreamingWeightLoader(
                runtime,
                str(shard),
                chunk_bytes=chunk_bytes,
                num_chunks=num_chunks,
                io_mode=io_mode,
                chunk_pool=self._chunks,
            )
            for shard in index.shard_paths
        }
        self.bytes_streamed = 0
        self.experts_served = 0
        self.slot_fills = 0
        self._closed = False
        self.closed_read_attempts = 0

    # ------------------------------------------------------------- inspection

    @property
    def backends(self) -> dict[str, str]:
        """IO backend actually negotiated per shard ("unbuffered" or "mmap")."""
        return {shard.name: loader.backend for shard, loader in self._loaders.items()}

    @property
    def staged_expert_count(self) -> int:
        return len(self.expert_spans)

    @property
    def chunk_pool_bytes(self) -> int:
        return sum(chunk.numel() for chunk in self._chunks)

    def chunks_are_io_aligned(self) -> bool:
        """Unbuffered reads require 4096-aligned chunk base addresses."""
        return all(chunk.data_ptr() % IO_ALIGNMENT == 0 for chunk in self._chunks)

    # --------------------------------------------------------- WeightByteSource

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self.expert_spans

    def fill_slot(self, destination: torch.Tensor, layer_idx: int, expert_id: int) -> None:
        """Materialize one expert's whole slot region (disk -> pinned window)."""
        self._require_open()
        spans = self._spans_of(layer_idx, expert_id)
        if destination.numel() != self._layout.slot_num_bytes:
            raise ValueError(
                f"destination holds {destination.numel()} bytes, slot layout is {self._layout.slot_num_bytes}"
            )
        flat = destination.view(-1)
        for spec in self._layout.specs:
            span = spans[spec.param_key]
            region = flat.narrow(0, spec.offset_bytes, spec.num_bytes)
            self.bytes_streamed += self._loader_for(span).stream_into(region, span.begin, span.num_bytes)
        self.slot_fills += 1

    def read_param(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        """Zero-copy host view of one parameter's authoritative bytes."""
        self._require_open()
        span = self._spans_of(layer_idx, expert_id)[param_key]
        spec = self._layout.spec_for(param_key)
        return self._loader_for(span).mmap_view(span.begin, span.num_bytes).view(spec.view_shape)

    # ------------------------------------------------------------- SlotFill

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        if not self.contains(layer_idx, expert_id):
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) not present in the checkpoint index")

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int:
        """DirectStorage fill: disk -> pinned chunks -> non-blocking device copies."""
        self._require_open()
        spans = self._spans_of(layer_idx, expert_id)
        moved = 0
        for param_key, destination in views.items():
            span = spans[param_key]
            moved += self._loader_for(span).stream_into(destination.view(-1), span.begin, span.num_bytes)
        self.bytes_streamed += moved
        self.experts_served += 1
        return moved

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        return self.read_param(layer_idx, expert_id, param_key)

    # -------------------------------------------------------- shared experts

    def bind_shared_expert(self, layer_idx: int, shared_layout: ExpertTensorLayout) -> dict[str, TensorSpan]:
        """Bind one layer's shared-expert projections, checked like a routed slot.

        Memoized, because the dispatcher asks per layer and the answer is a
        property of the checkpoint. The dtype and shape checks are the routed
        ones: a shared expert bound against the *routed* layout would ask for
        5,767,168-byte spans out of 11,534,336-byte tensors and quietly load a
        third of each projection.
        """
        cached = self._shared_spans.get(layer_idx)
        if cached is not None:
            return cached
        spans: dict[str, TensorSpan] = {}
        total = 0
        for spec in shared_layout.specs:
            name = self._naming.shared_tensor_name(layer_idx, spec)
            span = self._index.span(name)
            validate_expert_span(spec, name, span.begin, span.num_bytes, False)
            _validate_dtype_and_shape(spec, name, span)
            spans[spec.param_key] = span
            total += span.num_bytes
        if total != shared_layout.slot_num_bytes:
            raise WeightLayoutMismatchError(
                f"layer {layer_idx} shared expert: spans total {total} bytes, shared layout is "
                f"{shared_layout.slot_num_bytes}"
            )
        self._shared_spans[layer_idx] = spans
        return spans

    def fill_shared_expert(self, layer_idx: int, views: Mapping[str, torch.Tensor]) -> int:
        """Stream one layer's shared expert into its reserved views; returns bytes."""
        self._require_open()
        if self._shared_layout is None:
            raise WeightLayoutMismatchError(
                "this source was built without a shared_layout, so it cannot bind shared experts"
            )
        spans = self.bind_shared_expert(layer_idx, self._shared_layout)
        moved = 0
        for param_key, destination in views.items():
            span = spans[param_key]
            if destination.numel() != span.num_bytes:
                raise ValueError(
                    f"layer {layer_idx} shared {param_key}: view holds {destination.numel()} bytes, "
                    f"span is {span.num_bytes}"
                )
            moved += self._loader_for(span).stream_into(destination.view(-1), span.begin, span.num_bytes)
        self.bytes_streamed += moved
        self.shared_fills += 1
        return moved

    def synchronize(self) -> None:
        for loader in self._loaders.values():
            loader.synchronize()

    def close(self) -> None:
        self._closed = True
        self.synchronize()
        for chunk in self._chunks:
            ctypes.memset(chunk.data_ptr(), 0, chunk.numel())
        for loader in self._loaders.values():
            loader.close()
            loader._chunks.clear()
        self._chunks.clear()

    def _require_open(self) -> None:
        if self._closed:
            self.closed_read_attempts += 1
            raise RuntimeError("runtime disk reads are forbidden after checkpoint source shutdown")

    # ------------------------------------------------------------- internals

    def _spans_of(self, layer_idx: int, expert_id: int) -> dict[str, TensorSpan]:
        try:
            return self.expert_spans[(layer_idx, expert_id)]
        except KeyError:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) not present in the checkpoint index") from None

    def _loader_for(self, span: TensorSpan) -> StreamingWeightLoader:
        return self._loaders[span.shard]
