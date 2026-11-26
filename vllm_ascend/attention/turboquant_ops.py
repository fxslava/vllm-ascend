# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch TurboQuant operations to custom kernels or the PyTorch reference."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.attention import turboquant_e8, turboquant_reference_cache
from vllm_ascend.attention.turboquant_codec import (
    TurboQuantEmulator,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_DECODE_UNPACK,
    TURBOQUANT_FUSED_CONTEXT_LIMIT,
    TURBOQUANT_KERNEL_DTYPE,
    TURBOQUANT_KERNEL_OPS,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_GATHER_PAGES,
    TURBOQUANT_OP_QUANTIZE_PACK,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_ROTATE,
    TURBOQUANT_OP_UNPACK_DEQUANT,
    TURBOQUANT_QUANT_MODE,
    TURBOQUANT_REGISTERED_OPS,
    TURBOQUANT_STAGE_ROTATED_BASIS,
    TURBOQUANT_STAGE_UNROTATED,
    TurboQuantKernelStage,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    turboquant_decode_heads_per_task,
)
from vllm_ascend.attention.turboquant_ingestion import (
    TurboQuantStripPlan,
    _strip_run_rows,
    _strip_run_slots,
    turboquant_plan_strip_ingestion,
)
from vllm_ascend.attention.turboquant_layout import (
    E8_HEAD_SIZE,
    E8_PACKET_BYTES,
    E8_PACKET_ROWS,
    turboquant_scale_block_views,
)
from vllm_ascend.attention.turboquant_telemetry import (
    DispatchImplementation,
    DispatchObserver,
    create_dispatch_observer,
)

_TURBOQUANT_EXTENSION_LOADED: bool | None = None


def _is_op_registered(namespace: str, op_name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel(f"{namespace}::{op_name}"))
    except (AttributeError, RuntimeError):
        try:
            getattr(getattr(torch.ops, namespace), op_name)
        except (AttributeError, RuntimeError):
            return False
        return True


def load_turboquant_extension() -> bool:
    global _TURBOQUANT_EXTENSION_LOADED
    if _TURBOQUANT_EXTENSION_LOADED is not None:
        return _TURBOQUANT_EXTENSION_LOADED

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        if not torch.compiler.is_compiling():
            bootstrap_custom_op_env()
    except ImportError as exc:
        logger.debug("TurboQuant: bootstrap_custom_op_env is unavailable (%s); importing anyway.", exc)

    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")

        _TURBOQUANT_EXTENSION_LOADED = True
    except (ImportError, OSError) as exc:
        _TURBOQUANT_EXTENSION_LOADED = False
        logger.debug("TurboQuant: vllm_ascend.vllm_ascend_C did not import (%s).", exc)
    return _TURBOQUANT_EXTENSION_LOADED


def turboquant_decode_graph_supported() -> bool:
    """Full decode capture requires kernel ingress, decode and host tiling.

    Prefill and mixed batches still use host metadata and must remain outside
    the full graph. Check registrations as well as stage selection: a missing
    kernel otherwise silently selects the uncapturable emulator.
    """
    config = TurboQuantOpsConfig.from_env()
    if config.uses_e8:
        return False
    if not all(config.enables(stage) for stage in (TurboQuantKernelStage.SCATTER, TurboQuantKernelStage.DECODE)):
        return False
    load_turboquant_extension()
    return all(
        _is_op_registered(config.ops_namespace, name)
        for name in (
            TURBOQUANT_OP_RESHAPE_SCATTER,
            TURBOQUANT_OP_FUSED_DECODE,
            TURBOQUANT_OP_BUILD_DECODE_TILING,
        )
    )


class TurboQuantOpsAdapter:
    def __init__(
        self,
        config: TurboQuantOpsConfig | None = None,
        *,
        dispatch_observer: DispatchObserver | None = None,
    ) -> None:
        self.config: TurboQuantOpsConfig = config if config is not None else TurboQuantOpsConfig.from_env()
        self.dispatch_observer = dispatch_observer if dispatch_observer is not None else create_dispatch_observer()
        self._emulators: dict[tuple[int, str], TurboQuantEmulator] = {}
        self._missing_kernels: set[str] = set()
        if self.config.kernel_stages:
            load_turboquant_extension()
            if not self.config.uses_e8:
                self._report_namespace()

    def _report_namespace(self) -> None:
        stages = sorted(stage.value for stage in self.config.kernel_stages)
        namespace = self.config.ops_namespace
        missing = [op for op in TURBOQUANT_REGISTERED_OPS if not _is_op_registered(namespace, op)]
        if not missing:
            logger.debug(
                "TurboQuant: torch.ops.%s carries %s for stages %s.",
                namespace,
                list(TURBOQUANT_REGISTERED_OPS),
                stages,
            )
            return
        logger.warning(
            "TurboQuant stages %s are switched on but torch.ops.%s is missing %s, which will "
            "run the PyTorch golden model instead. These registrations are static initialisers "
            "of vllm_ascend.vllm_ascend_C (import of it %s here), and CMakeLists.txt compiles "
            "csrc/attention/turboquant/bindings/torch_bindings.cpp only when SOC_VERSION matches "
            "ascend950 -- so on any other part they are absent by construction. Operators the "
            "dispatcher does have under this namespace: %s",
            stages,
            namespace,
            missing,
            "succeeded" if _TURBOQUANT_EXTENSION_LOADED else "failed",
            [op for op in TURBOQUANT_REGISTERED_OPS if op not in missing] or "none",
        )

    def get_or_create_emulator(self, head_size: int, device: torch.device) -> TurboQuantEmulator:
        key = (head_size, str(device))
        emulator = self._emulators.get(key)
        if emulator is None:
            emulator = TurboQuantEmulator(head_size, device)
            self._emulators[key] = emulator
        return emulator

    def _resolve_kernel(self, op_name: str) -> Callable[..., Any] | None:
        if self.config.uses_e8 or not self.config.enables(TURBOQUANT_KERNEL_OPS[op_name]):
            return None
        namespace = getattr(torch.ops, self.config.ops_namespace, None)
        kernel = getattr(namespace, op_name, None) if namespace is not None else None
        if kernel is None:
            if op_name not in self._missing_kernels:
                self._missing_kernels.add(op_name)
                logger.warning(
                    "TurboQuant stage %s is switched on but torch.ops.%s.%s is not registered; "
                    "falling back to the PyTorch golden model.",
                    TURBOQUANT_KERNEL_OPS[op_name].value,
                    self.config.ops_namespace,
                    op_name,
                )
            return None
        return kernel

    def has_kernel(self, op_name: str) -> bool:
        return self._resolve_kernel(op_name) is not None

    def _require_kernel(self, op_name: str) -> Callable[..., Any]:
        kernel = self._resolve_kernel(op_name)
        if kernel is None:
            raise RuntimeError(f"torch.ops.{self.config.ops_namespace}.{op_name} is not registered")
        return kernel

    def value_layout(self) -> TurboQuantValueLayout:
        if self.has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE) or self.has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            return TurboQuantValueLayout.NZ_TILED
        return self.config.value_layout

    def block_major_cache_views(self, kv_cache: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(kv_cache) == 2:
            return tuple(kv_cache)
        key_cache, value_cache, scale_cache = kv_cache
        scales = turboquant_scale_block_views(
            scale_cache, slot_major=self.value_layout() is TurboQuantValueLayout.NZ_TILED
        )
        return key_cache, value_cache, *scales

    @staticmethod
    def load_e8_packets(
        kv_cache: Sequence[torch.Tensor], packet_ids: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Install whole host-packed packets using physical block/tile/head packet IDs."""
        key_cache, value_cache = kv_cache
        if (
            key_cache.shape != value_cache.shape
            or key_cache.device != value_cache.device
            or key_cache.dtype != torch.uint8
            or value_cache.dtype != torch.uint8
            or key_cache.ndim != 4
            or key_cache.size(-1) != E8_PACKET_BYTES
        ):
            raise ValueError("E8 cache requires two matching UINT8 packet planes")
        planes = (key_cache.view(-1, E8_PACKET_BYTES), value_cache.view(-1, E8_PACKET_BYTES))
        if packet_ids.device.type != "cpu" or packet_ids.dtype != torch.int64 or packet_ids.ndim != 1:
            raise ValueError("E8 packet IDs must be flat CPU INT64")
        for packets in (key, value):
            if packets.device.type != "cpu" or packets.shape != (packet_ids.numel(), E8_PACKET_BYTES):
                raise ValueError("E8 source packets must be CPU [count, 4352]")
            turboquant_e8.unpack_packets(packets)
        if bool(((packet_ids < 0) | (packet_ids >= planes[0].size(0))).any()) or (
            packet_ids.unique().numel() != packet_ids.numel()
        ):
            raise ValueError("E8 packet IDs must be unique and within the cache")
        device = key_cache.device
        ids = packet_ids.to(device, non_blocking=packet_ids.is_pinned())
        sources = tuple(source.to(device, non_blocking=source.is_pinned()) for source in (key, value))
        for plane, source in zip(planes, sources):
            plane.index_copy_(0, ids, source)

    def read_e8_packets(self, cache: torch.Tensor, plan: turboquant_e8.E8RequestPlan) -> torch.Tensor:
        """Gather and decode one request, with a context bound independent of batch size."""
        packet_ids = plan.packet_ids
        tiles, heads = packet_ids.shape
        flat_cache = cache.view(-1, E8_PACKET_BYTES)
        # Bound temporary tensors for the portable integer decoder by splitting packet batches.
        tiles_per_chunk = max(1, 32 // heads)
        dense = torch.empty((tiles, E8_PACKET_ROWS, heads, E8_HEAD_SIZE), dtype=torch.bfloat16, device=cache.device)
        use_kernel = (
            cache.device.type != "cpu"
            and self.config.enables(TurboQuantKernelStage.DEQUANTIZE)
            and _is_op_registered(turboquant_e8.E8_OPS_NAMESPACE, "dequant_packet")
        )
        for start in range(0, tiles, tiles_per_chunk):
            batch_ids = packet_ids[start : start + tiles_per_chunk]
            packets = flat_cache.index_select(0, batch_ids.flatten())
            if use_kernel:
                shape = (packets.size(0), E8_PACKET_ROWS, E8_HEAD_SIZE)
                decoded = torch.empty(shape, dtype=torch.bfloat16, device=cache.device)
                int8 = torch.empty(shape, dtype=torch.int8, device=cache.device)
                turboquant_e8.dequant_packet(packets, decoded, int8)
            else:
                labels, scales = turboquant_e8.unpack_packets(packets)
                decoded = turboquant_reference_cache.decode_e8_labels(labels, scales)
            decoded = decoded.view(-1, heads, E8_PACKET_ROWS, E8_HEAD_SIZE).transpose(1, 2)
            dense[start : start + batch_ids.size(0)].copy_(decoded)
        return dense.flatten(0, 1)[: plan.context_length]

    def rotate(self, activations: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        if self.has_kernel(TURBOQUANT_OP_ROTATE):
            self.dispatch_observer.record(
                TURBOQUANT_OP_ROTATE, DispatchImplementation.CUSTOM_OP, activations=activations
            )
            return self._kernel_rotate(
                activations=activations,
                pi_signs=emulator.pi_signs,
                rot_tables=emulator.codec_tables,
                h16=emulator.h16,
                output_stage=TURBOQUANT_STAGE_ROTATED_BASIS,
                inv_sqrt_len=1.0,
            )
        self.dispatch_observer.record(TURBOQUANT_OP_ROTATE, DispatchImplementation.PYTORCH, activations=activations)
        return self._reference_rotate(activations, emulator)

    def inverse_rotate(self, rotated: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        if self.has_kernel(TURBOQUANT_OP_ROTATE):
            self.dispatch_observer.record(TURBOQUANT_OP_ROTATE, DispatchImplementation.CUSTOM_OP, rotated=rotated)
            return self._kernel_rotate(
                activations=rotated,
                pi_signs=emulator.pi_signs,
                rot_tables=emulator.codec_tables,
                h16=emulator.h16,
                output_stage=TURBOQUANT_STAGE_UNROTATED,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
        self.dispatch_observer.record(TURBOQUANT_OP_ROTATE, DispatchImplementation.PYTORCH, rotated=rotated)
        return self._reference_inverse_rotate(rotated, emulator)

    def _kernel_rotate(
        self,
        activations: torch.Tensor,
        pi_signs: torch.Tensor,
        rot_tables: torch.Tensor,
        h16: torch.Tensor,
        output_stage: int,
        inv_sqrt_len: float,
    ) -> torch.Tensor:
        return self._require_kernel(TURBOQUANT_OP_ROTATE)(
            activations, pi_signs, rot_tables, h16, output_stage, inv_sqrt_len
        )

    def _reference_rotate(self, activations: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        return emulator.rotate(activations)

    def _reference_inverse_rotate(self, rotated: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        return emulator.inverse_rotate(rotated)

    def quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.has_kernel(TURBOQUANT_OP_QUANTIZE_PACK):
            self.dispatch_observer.record(
                TURBOQUANT_OP_QUANTIZE_PACK, DispatchImplementation.CUSTOM_OP, rotated=rotated
            )
            return self._kernel_quantize_and_pack(rotated=rotated, inv_sqrt_len=emulator.inv_sqrt_len)
        self.dispatch_observer.record(TURBOQUANT_OP_QUANTIZE_PACK, DispatchImplementation.PYTORCH, rotated=rotated)
        return self._reference_quantize_and_pack(rotated, emulator)

    def unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        if self.has_kernel(TURBOQUANT_OP_UNPACK_DEQUANT):
            self.dispatch_observer.record(
                TURBOQUANT_OP_UNPACK_DEQUANT, DispatchImplementation.CUSTOM_OP, packed=packed, scales=scales
            )
            return self._kernel_unpack_and_dequantize(packed=packed, scales=scales)
        self.dispatch_observer.record(
            TURBOQUANT_OP_UNPACK_DEQUANT, DispatchImplementation.PYTORCH, packed=packed, scales=scales
        )
        return self._reference_unpack_and_dequantize(packed, scales, emulator)

    def _kernel_quantize_and_pack(
        self, rotated: torch.Tensor, inv_sqrt_len: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._require_kernel(TURBOQUANT_OP_QUANTIZE_PACK)(rotated, inv_sqrt_len)

    def _kernel_unpack_and_dequantize(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self._require_kernel(TURBOQUANT_OP_UNPACK_DEQUANT)(packed, scales)

    def _reference_quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, scales = emulator.quantize(rotated)
        return emulator.pack(codes), scales

    def _reference_unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        return emulator.dequantize(emulator.unpack(packed), scales)

    def reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        plan: TurboQuantStripPlan | None = None,
    ) -> None:
        if self.has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE):
            self._ingest_prefill_with_kernels(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
                plan=plan,
            )
            return
        self.dispatch_observer.record(
            TURBOQUANT_OP_RESHAPE_AND_CACHE,
            DispatchImplementation.PYTORCH,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
        )
        self._reference_reshape_and_cache_prefill(
            key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator
        )

    def _ingest_prefill_with_kernels(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        plan: TurboQuantStripPlan | None = None,
    ) -> None:
        if plan is None:
            plan = turboquant_plan_strip_ingestion(slot_mapping.tolist(), int(key_cache.shape[1]))
        tables = emulator.codec_tables

        if plan.is_whole_batch:
            self._kernel_cache_prefill(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
                tables=tables,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
            return

        for run in plan.runs:
            self._kernel_cache_prefill(
                key=_strip_run_rows(key, run),
                value=_strip_run_rows(value, run),
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=_strip_run_slots(slot_mapping, run),
                pi_signs=emulator.pi_signs,
                tables=tables,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )

        for wave in plan.waves:
            rows = torch.as_tensor(wave, dtype=torch.long, device=slot_mapping.device)
            self.reshape_scatter_decode(
                key=key.index_select(0, rows).contiguous(),
                value=value.index_select(0, rows).contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping.index_select(0, rows).contiguous(),
                emulator=emulator,
            )

    def _kernel_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        pi_signs: torch.Tensor,
        tables: torch.Tensor,
        inv_sqrt_len: float,
        quant_mode: int = TURBOQUANT_QUANT_MODE,
    ) -> None:
        self.dispatch_observer.record(
            TURBOQUANT_OP_RESHAPE_AND_CACHE,
            DispatchImplementation.CUSTOM_OP,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
            pi_signs=pi_signs,
            tables=tables,
        )
        self._require_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE)(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            pi_signs,
            tables,
            inv_sqrt_len,
            quant_mode,
        )

    def _reference_reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        self._write_slots(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            emulator,
            lloyd_max_folds=("key", "value"),
        )

    def reshape_scatter_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        if self.has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            self.dispatch_observer.record(
                TURBOQUANT_OP_RESHAPE_SCATTER,
                DispatchImplementation.CUSTOM_OP,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
            )
            self._kernel_cache_decode(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
            return
        self.dispatch_observer.record(
            TURBOQUANT_OP_RESHAPE_SCATTER,
            DispatchImplementation.PYTORCH,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
        )
        self._reference_reshape_scatter_decode(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    def _kernel_cache_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        pi_signs: torch.Tensor,
        inv_sqrt_len: float,
        quant_mode: int = TURBOQUANT_QUANT_MODE,
    ) -> None:
        self._require_kernel(TURBOQUANT_OP_RESHAPE_SCATTER)(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            pi_signs,
            inv_sqrt_len,
            quant_mode,
        )

    def _reference_reshape_scatter_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        self._write_slots(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            emulator,
            lloyd_max_folds=("key", "key"),
        )

    def _write_slots(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        lloyd_max_folds: tuple[str, str] = ("key", "value"),
    ) -> None:
        turboquant_reference_cache.write_slots(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            emulator,
            rotate=self.rotate,
            layout=self.value_layout(),
            lloyd_max_folds=lloyd_max_folds,
        )

    def _write_strip_planes(
        self,
        codes_key: torch.Tensor,
        codes_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        turboquant_reference_cache.write_strip_planes(codes_key, codes_value, key_cache, value_cache, pages, offsets)

    _nz_byte_offsets = staticmethod(turboquant_reference_cache.nz_byte_offsets)
    _nz_scale_slot_floats = staticmethod(turboquant_reference_cache.nz_scale_slot_floats)

    def gather(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_block_tables(block_tables, int(key_cache.shape[0]))
        if self.has_kernel(TURBOQUANT_OP_GATHER_PAGES):
            self.dispatch_observer.record(
                TURBOQUANT_OP_GATHER_PAGES,
                DispatchImplementation.CUSTOM_OP,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
            )
            return self._kernel_gather_pages(
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
            )
        self.dispatch_observer.record(
            TURBOQUANT_OP_GATHER_PAGES,
            DispatchImplementation.PYTORCH,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
        )
        layout = self.value_layout()
        if layout is TurboQuantValueLayout.CHANNEL_MAJOR:
            return self._reference_gather_pages_strip(key_cache, value_cache, scale_cache, block_tables)
        if layout is TurboQuantValueLayout.NZ_TILED:
            return self._reference_gather_pages_nz(key_cache, value_cache, scale_cache, block_tables)
        return self._reference_gather_pages(key_cache, value_cache, scale_cache, block_tables)

    @staticmethod
    def _validate_block_tables(block_tables: torch.Tensor, num_blocks: int) -> None:
        if block_tables.numel() == 0:
            return
        low = block_tables.min()
        high = block_tables.max()
        if bool(low < 0) or bool(high >= num_blocks):
            raise ValueError(
                f"block_tables must lie in [0, {num_blocks}) -- num_blocks is key_cache.shape[0], "
                f"the cache's own page count -- but block_tables has min={int(low)} and "
                f"max={int(high)} (shape {tuple(block_tables.shape)}). A negative page wraps onto "
                "the last page in the golden reader, and a page past the end reads outside the "
                "cache; both are refused before any gather happens."
            )

    def _kernel_gather_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._require_kernel(TURBOQUANT_OP_GATHER_PAGES)(key_cache, value_cache, scale_cache, block_tables)

    _reference_gather_pages = staticmethod(turboquant_reference_cache.gather_pages)
    _reference_gather_pages_strip = staticmethod(turboquant_reference_cache.gather_pages_strip)
    _reference_gather_pages_nz = staticmethod(turboquant_reference_cache.gather_pages_nz)

    def dequantize_cache(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        emulator: TurboQuantEmulator,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rotated = self.unpack_and_dequantize(packed, scales, emulator)
        return self.inverse_rotate(rotated, emulator).to(dtype)

    def fused_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        emulator: TurboQuantEmulator,
        num_heads: int,
        num_kv_heads: int,
        scale: float,
        output: torch.Tensor,
        tiling: torch.Tensor,
        workspace: torch.Tensor,
        query_rot: torch.Tensor,
        fused_context_limit: int = TURBOQUANT_FUSED_CONTEXT_LIMIT,
        num_splits: int | None = None,
    ) -> torch.Tensor | None:
        """Launch decode; split/threshold overrides must match the tiling builder."""
        if not self.has_kernel(TURBOQUANT_OP_FUSED_DECODE):
            self.dispatch_observer.record(
                TURBOQUANT_OP_FUSED_DECODE,
                DispatchImplementation.PYTORCH,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                output=output,
                tiling=tiling,
                workspace=workspace,
                query_rot=query_rot,
            )
            return None
        self._validate_fused_decode_geometry(query, key_cache, scale_cache, output, num_kv_heads)
        self.dispatch_observer.record(
            TURBOQUANT_OP_FUSED_DECODE,
            DispatchImplementation.CUSTOM_OP,
            query=query,
            pi_signs=emulator.pi_signs,
            rot_tables=emulator.codec_tables,
            h16=emulator.h16,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
            output=output,
        )
        return self._kernel_fused_decode(
            query=query,
            pi_signs=emulator.pi_signs,
            rot_tables=emulator.codec_tables,
            h16=emulator.h16,
            gate=None,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
            output=output,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_splits=TURBOQUANT_DECODE_SPLITS if num_splits is None else num_splits,
            heads_per_task=turboquant_decode_heads_per_task(num_heads, num_kv_heads),
            output_stage=TURBOQUANT_STAGE_UNROTATED,
            scale=scale,
            inv_sqrt_len=emulator.inv_sqrt_len,
            fused_context_limit=fused_context_limit,
        )

    def _validate_fused_decode_geometry(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        output: torch.Tensor,
        num_kv_heads: int,
    ) -> None:
        if query.dtype != TURBOQUANT_KERNEL_DTYPE:
            raise TypeError(
                f"the fused TurboQuant decode kernel only has {TURBOQUANT_KERNEL_DTYPE} "
                f"instances, but query.dtype is {query.dtype} (query shape "
                f"{tuple(query.shape)}). The golden fallback serves any floating dtype; "
                "this is the kernel launch path."
            )
        if output.dtype != TURBOQUANT_KERNEL_DTYPE:
            raise TypeError(
                f"the fused TurboQuant decode kernel writes {TURBOQUANT_KERNEL_DTYPE} only, but "
                f"output.dtype is {output.dtype} (output shape {tuple(output.shape)}). The "
                "golden fallback serves any floating dtype; this is the kernel launch path."
            )
        # The binding also rejects this alias at runtime. Dynamo's fake
        # tensors have no physical addresses to compare during the trace.
        if not torch.compiler.is_compiling() and query.data_ptr() == output.data_ptr():
            raise ValueError(
                f"query and output must not alias: both are at data_ptr={query.data_ptr()}. "
                "The fused decode reads the query through its whole prologue while writing "
                "output, so an in-place call would overwrite the query it is still reading."
            )

        num_blocks = int(key_cache.shape[0])
        block_size = int(key_cache.shape[1])
        slot_floats = self._nz_scale_slot_floats(num_kv_heads)
        need = num_blocks * block_size * slot_floats
        if scale_cache.numel() < need:
            raise ValueError(
                f"scale_cache is too small for the fused TurboQuant decode: it holds "
                f"{scale_cache.numel()} floats, but the kernel reads {slot_floats} floats per "
                f"slot (NzScaleSlotFloats({num_kv_heads})) over {num_blocks} blocks activations "
                f"{block_size} tokens = {need} (key_cache.shape={tuple(key_cache.shape)}, "
                f"scale_cache.shape={tuple(scale_cache.shape)})."
            )

    def _kernel_fused_decode(
        self,
        query: torch.Tensor,
        pi_signs: torch.Tensor,
        rot_tables: torch.Tensor,
        h16: torch.Tensor,
        gate: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        tiling: torch.Tensor,
        workspace: torch.Tensor,
        query_rot: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        num_splits: int,
        heads_per_task: int,
        output_stage: int,
        scale: float,
        inv_sqrt_len: float,
        decode_unpack: int = TURBOQUANT_DECODE_UNPACK,
        fused_context_limit: int = TURBOQUANT_FUSED_CONTEXT_LIMIT,
    ) -> torch.Tensor:
        self._require_kernel(TURBOQUANT_OP_FUSED_DECODE)(
            query,
            pi_signs,
            rot_tables,
            h16,
            gate,
            key_cache,
            value_cache,
            scale_cache,
            block_tables,
            context_lens,
            tiling,
            workspace,
            query_rot,
            output,
            num_heads,
            num_kv_heads,
            num_splits,
            heads_per_task,
            output_stage,
            scale,
            inv_sqrt_len,
            decode_unpack,
            fused_context_limit,
        )
        return output


def _stream_is_capturing() -> bool:
    with suppress(AttributeError, RuntimeError):
        return bool(torch.npu.is_current_stream_capturing())
    return False
