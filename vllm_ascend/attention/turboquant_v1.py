# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention integration and public backend entry point.

Codec, dispatch, planning, metadata and telemetry live in dedicated modules.
Existing public imports from this entry point remain supported."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.turboquant_codec import (
    TurboQuantEmulator,
    turboquant_pi_signs,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_DECODE_DISPATCH_WORDS,
    TURBOQUANT_DECODE_MAX_GRID,
    TURBOQUANT_DECODE_MAX_HEADS_PER_TASK,
    TURBOQUANT_DECODE_MAX_KV_HEADS,
    TURBOQUANT_DECODE_MAX_SPLITS,
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_DECODE_TILING_WORDS,
    TURBOQUANT_DECODE_UNPACK,
    TURBOQUANT_DEFAULT_NAMESPACE,
    TURBOQUANT_FIA_DUMMY_KV_LEN,
    TURBOQUANT_FIA_VERSION_ENV,
    TURBOQUANT_H16_ORDER,
    TURBOQUANT_HEADROOM_BITS,
    TURBOQUANT_KERNEL_BLOCK_SIZE,
    TURBOQUANT_KERNEL_DTYPE,
    TURBOQUANT_KERNEL_HEAD_SIZE,
    TURBOQUANT_KERNEL_KV_HEADS,
    TURBOQUANT_KERNEL_OPS,
    TURBOQUANT_LM_CENTROIDS,
    TURBOQUANT_LM_EPS,
    TURBOQUANT_LM_REDUCTION_LANES,
    TURBOQUANT_LM_THRESHOLDS,
    TURBOQUANT_NAMESPACE_ENV,
    TURBOQUANT_NIBBLE_LEVELS,
    TURBOQUANT_NIBBLE_MASK,
    TURBOQUANT_NIBBLE_MAX,
    TURBOQUANT_NIBBLE_MIN,
    TURBOQUANT_NZ_FP32_PER_BLOCK,
    TURBOQUANT_NZ_OPERAND_C0,
    TURBOQUANT_NZ_TILE_ROWS,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_GATHER_PAGES,
    TURBOQUANT_OP_QUANTIZE_PACK,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_ROTATE,
    TURBOQUANT_OP_UNPACK_DEQUANT,
    TURBOQUANT_PARTIAL_TAIL,
    TURBOQUANT_PI_SEED,
    TURBOQUANT_QUANT_MODE,
    TURBOQUANT_REGISTERED_OPS,
    TURBOQUANT_STAGE_GATED,
    TURBOQUANT_STAGE_ROTATED_BASIS,
    TURBOQUANT_STAGE_UNROTATED,
    TURBOQUANT_STAGES_ENV,
    TURBOQUANT_STAGING_SLOTS,
    TURBOQUANT_STRIP_TOKENS,
    TURBOQUANT_TABLE_DTYPE,
    TurboQuantIngestionPhase,
    TurboQuantKernelStage,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    turboquant_decode_heads_per_task,
    turboquant_decode_query_rot_floats,
    turboquant_decode_tiling_words,
    turboquant_decode_workspace_floats,
    turboquant_tiling_upper_bound_words,
)
from vllm_ascend.attention.turboquant_ingestion import (
    TurboQuantStripPlan,
    TurboQuantStripRun,
    turboquant_plan_strip_ingestion,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_GROUP,
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_PLANES,
    TURBOQUANT_VALUE_PLANE,
    turboquant_packed_cache_shape,
    turboquant_scale_cache_shape,
)
from vllm_ascend.attention.turboquant_metadata import (
    AscendTurboQuantMetadata,
    AscendTurboQuantMetadataBuilder,
    TurboQuantQueryLayout,
)
from vllm_ascend.attention.turboquant_ops import (
    TurboQuantOpsAdapter,
    _stream_is_capturing,
    load_turboquant_extension,
    turboquant_decode_graph_supported,
)
from vllm_ascend.attention.turboquant_telemetry import (
    TURBOQUANT_PROFILE_ENV,
    TURBOQUANT_PROFILE_NOTICE,
    TURBOQUANT_PROFILE_TRACE_DIR,
    InferenceTelemetry,
    turboquant_profile_enabled,
)
from vllm_ascend.attention.utils import notify_kv_cache_written

__all__ = [
    "AscendTurboQuantBackend",
    "AscendTurboQuantBackendImpl",
    "AscendTurboQuantMetadata",
    "AscendTurboQuantMetadataBuilder",
    "TURBOQUANT_DECODE_DISPATCH_WORDS",
    "TURBOQUANT_DECODE_MAX_GRID",
    "TURBOQUANT_DECODE_MAX_HEADS_PER_TASK",
    "TURBOQUANT_DECODE_MAX_KV_HEADS",
    "TURBOQUANT_DECODE_MAX_SPLITS",
    "TURBOQUANT_DECODE_SPLITS",
    "TURBOQUANT_DECODE_TILING_WORDS",
    "TURBOQUANT_DECODE_UNPACK",
    "TURBOQUANT_DEFAULT_NAMESPACE",
    "TURBOQUANT_FIA_DUMMY_KV_LEN",
    "TURBOQUANT_FIA_VERSION_ENV",
    "TURBOQUANT_H16_ORDER",
    "TURBOQUANT_HEADROOM_BITS",
    "TURBOQUANT_KERNEL_BLOCK_SIZE",
    "TURBOQUANT_KERNEL_DTYPE",
    "TURBOQUANT_KERNEL_HEAD_SIZE",
    "TURBOQUANT_KERNEL_KV_HEADS",
    "TURBOQUANT_KERNEL_OPS",
    "TURBOQUANT_LM_CENTROIDS",
    "TURBOQUANT_LM_EPS",
    "TURBOQUANT_LM_REDUCTION_LANES",
    "TURBOQUANT_LM_THRESHOLDS",
    "TURBOQUANT_NAMESPACE_ENV",
    "TURBOQUANT_NIBBLE_LEVELS",
    "TURBOQUANT_NIBBLE_MASK",
    "TURBOQUANT_NIBBLE_MAX",
    "TURBOQUANT_NIBBLE_MIN",
    "TURBOQUANT_NZ_FP32_PER_BLOCK",
    "TURBOQUANT_NZ_OPERAND_C0",
    "TURBOQUANT_NZ_TILE_ROWS",
    "TURBOQUANT_OP_BUILD_DECODE_TILING",
    "TURBOQUANT_OP_FUSED_DECODE",
    "TURBOQUANT_OP_GATHER_PAGES",
    "TURBOQUANT_OP_QUANTIZE_PACK",
    "TURBOQUANT_OP_RESHAPE_AND_CACHE",
    "TURBOQUANT_OP_RESHAPE_SCATTER",
    "TURBOQUANT_OP_ROTATE",
    "TURBOQUANT_OP_UNPACK_DEQUANT",
    "TURBOQUANT_PARTIAL_TAIL",
    "TURBOQUANT_PACK_FACTOR",
    "TURBOQUANT_KEY_PLANE",
    "TURBOQUANT_VALUE_PLANE",
    "TURBOQUANT_PI_SEED",
    "TURBOQUANT_PROFILE_ENV",
    "TURBOQUANT_PROFILE_TRACE_DIR",
    "TURBOQUANT_PROFILE_NOTICE",
    "turboquant_profile_enabled",
    "TURBOQUANT_QUANT_MODE",
    "TURBOQUANT_REGISTERED_OPS",
    "TURBOQUANT_STAGES_ENV",
    "TURBOQUANT_STAGE_GATED",
    "TURBOQUANT_STAGE_ROTATED_BASIS",
    "TURBOQUANT_STAGE_UNROTATED",
    "TURBOQUANT_STAGING_SLOTS",
    "TURBOQUANT_STRIP_TOKENS",
    "TURBOQUANT_TABLE_DTYPE",
    "TurboQuantEmulator",
    "TurboQuantIngestionPhase",
    "TurboQuantKernelStage",
    "TurboQuantOpsAdapter",
    "TurboQuantOpsConfig",
    "TurboQuantQueryLayout",
    "TurboQuantStripPlan",
    "TurboQuantStripRun",
    "TurboQuantValueLayout",
    "load_turboquant_extension",
    "turboquant_decode_graph_supported",
    "turboquant_decode_heads_per_task",
    "turboquant_decode_query_rot_floats",
    "turboquant_decode_tiling_words",
    "turboquant_decode_workspace_floats",
    "turboquant_pi_signs",
    "turboquant_plan_strip_ingestion",
    "turboquant_tiling_upper_bound_words",
]


@register_backend(AttentionBackendEnum.TURBOQUANT)
class AscendTurboQuantBackend(AscendAttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type[AscendTurboQuantBackendImpl]:
        return AscendTurboQuantBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendTurboQuantMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [TURBOQUANT_KERNEL_BLOCK_SIZE]

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return turboquant_packed_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_scale_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int) -> tuple[int, ...]:
        return turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)

    @classmethod
    def swap_blocks(
        cls,
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        adapter = TurboQuantOpsAdapter()
        src_views = adapter.block_major_cache_views(src_kv_cache)
        dst_views = adapter.block_major_cache_views(dst_kv_cache)
        for src_cache, dst_cache in zip(src_views, dst_views):
            dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

    @classmethod
    def copy_blocks(
        cls,
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]
        adapter = TurboQuantOpsAdapter()
        for kv_cache in kv_caches:
            for cache in adapter.block_major_cache_views(kv_cache):
                cache[dst_indices] = cache[src_indices]


class AscendTurboQuantBackendImpl(AscendAttentionBackendImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        adapter: TurboQuantOpsAdapter | None = None,
        telemetry: InferenceTelemetry | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        self.requested_kv_cache_dtype: str = kv_cache_dtype
        self.model_dtype: torch.dtype = self.kv_cache_dtype
        self.kv_cache_dtype = torch.int8
        self.enable_c8_quant = False

        if head_size % TURBOQUANT_GROUP:
            raise ValueError(f"TurboQuant needs head_size to be a multiple of {TURBOQUANT_GROUP}, got {head_size}")
        if alibi_slopes is not None:
            raise NotImplementedError("TurboQuant has no ALiBi path: the decode kernel takes no slope operand.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TurboQuant has no sliding-window path: the fused decode reads whole pages from the block table."
            )
        if sinks is not None:
            raise NotImplementedError(
                "learnable attention sinks need the kernel's softmax denominator as an out-tensor; "
                "the dequantising golden path does not expose one."
            )
        if self.use_bnsd_kv_cache:
            raise NotImplementedError(
                "the TurboQuant packed cache is token-major ND; VLLM_KV_CACHE_LAYOUT must not be HND or LBHNC."
            )

        self.adapter: TurboQuantOpsAdapter = adapter if adapter is not None else TurboQuantOpsAdapter()
        self._validate_kernel_geometry()
        self.emulator: TurboQuantEmulator | None = None
        self.scale_cache: torch.Tensor | None = None
        self._use_fia_v2: bool | None = None
        self._warned_scale_carve = False
        self.telemetry = (
            telemetry if telemetry is not None else InferenceTelemetry(enabled=turboquant_profile_enabled())
        )
        staged = ", ".join(sorted(stage.value for stage in self.adapter.config.kernel_stages))
        logger.info_once(
            "TurboQuant backend active: packed INT4 KV cache, kernel stages=%s, ops namespace=%s",
            staged or "none (all golden)",
            self.adapter.config.ops_namespace,
        )

    def _validate_kernel_geometry(self) -> None:
        config = self.adapter.config
        writers = (
            TurboQuantKernelStage.RESHAPE_AND_CACHE,
            TurboQuantKernelStage.SCATTER,
        )
        enabled_writers = [stage for stage in writers if config.enables(stage)]
        decode_enabled = config.enables(TurboQuantKernelStage.DECODE)
        if not enabled_writers and not decode_enabled:
            return

        enabled_stages = list(enabled_writers)
        if decode_enabled:
            enabled_stages.append(TurboQuantKernelStage.DECODE)
        stage_names = ", ".join(sorted(stage.value for stage in enabled_stages))

        if self.head_size != TURBOQUANT_KERNEL_HEAD_SIZE:
            raise ValueError(
                f"TurboQuant kernel stages {stage_names} are switched on, but all three kernels are "
                f"specialised to head_size {TURBOQUANT_KERNEL_HEAD_SIZE} and this model's is "
                f"{self.head_size}. The golden model has no such constraint: unset "
                f"{TURBOQUANT_STAGES_ENV} to run this shape."
            )

        if enabled_writers and self.num_kv_heads != TURBOQUANT_KERNEL_KV_HEADS:
            raise ValueError(
                f"TurboQuant ingestion stages {stage_names} are switched on, but both cache writers are "
                f"specialised to exactly {TURBOQUANT_KERNEL_KV_HEADS} KV heads and this model has "
                f"{self.num_kv_heads}. The kernel does not check this: outside it the writer reads "
                f"one head's rows and writes another head's page, with every canary intact."
            )

        if decode_enabled and not 0 < self.num_kv_heads <= TURBOQUANT_DECODE_MAX_KV_HEADS:
            raise ValueError(
                f"the TurboQuant fused decode refuses num_kv_heads outside 1.."
                f"{TURBOQUANT_DECODE_MAX_KV_HEADS} (valid_launch), and this model has "
                f"{self.num_kv_heads}. This is the constraint a GQA model is most likely to hit; "
                f"unset {TURBOQUANT_STAGES_ENV} or drop the 'decode' stage to run golden."
            )
        if decode_enabled and self.num_heads % max(self.num_kv_heads, 1):
            raise ValueError(
                f"the TurboQuant fused decode needs num_heads ({self.num_heads}) to be a multiple "
                f"of num_kv_heads ({self.num_kv_heads})."
            )

        if self.model_dtype != TURBOQUANT_KERNEL_DTYPE:
            raise ValueError(
                f"TurboQuant kernel stages {stage_names} are switched on, but every kernel is "
                f"{TURBOQUANT_KERNEL_DTYPE} only and this model runs in {self.model_dtype}. The "
                f"fused decode's dispatch returns WITHOUT LAUNCHING for any other element type, "
                f"which would leave the attention output unwritten and report success, so this is "
                f"refused rather than attempted."
            )

    def _bind_codec_constants(self, layer: AttentionLayer, device: torch.device) -> TurboQuantEmulator:
        emulator = self.adapter.get_or_create_emulator(self.head_size, device)
        if self.emulator is not emulator:
            self.emulator = emulator
            layer._turboquant_pi_signs = emulator.pi_signs
            layer._turboquant_rotation = emulator.rotation
            layer._turboquant_codec_tables = emulator.codec_tables
            layer._turboquant_h16 = emulator.h16
            layer._turboquant_inv_sqrt_len = emulator.inv_sqrt_len
        return emulator

    def _ensure_scale_cache(self, kv_cache: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        if self.scale_cache is not None:
            return self.scale_cache
        if len(kv_cache) >= 3 and isinstance(kv_cache[2], torch.Tensor):
            scale_cache = kv_cache[2]
            expected = turboquant_scale_cache_shape(
                self.key_cache.shape[0], self.key_cache.shape[1], self.key_cache.shape[2]
            )
            if tuple(scale_cache.shape) != expected:
                raise ValueError(
                    f"the allocator's scale plane has shape {tuple(scale_cache.shape)}, expected {expected}"
                )
            if scale_cache.dtype is not torch.float32:
                raise ValueError(f"the scale plane must be fp32, got {scale_cache.dtype}")
            self.scale_cache = scale_cache
            return scale_cache

        num_blocks, block_size, num_kv_heads = self.key_cache.shape[:3]
        if not self._warned_scale_carve:
            self._warned_scale_carve = True
            logger.warning(
                "TurboQuant backend is allocating its own fp32 scale plane %s (%.1f MiB per layer): the "
                "KV cache spec carved only two planes. The production path carves three. "
                "TODO: extend the Ascend KV cache spec so the allocator owns this plane.",
                turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads),
                TURBOQUANT_PLANES * num_blocks * num_kv_heads * block_size * 4 / 2**20,
            )
        self.scale_cache = torch.zeros(
            turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads),
            dtype=torch.float32,
            device=self.key_cache.device,
        )
        return self.scale_cache

    def get_kv_cache_block_views(self, kv_cache: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Physical page views for the runner's cache copy and zeroing paths."""
        return self.adapter.block_major_cache_views(kv_cache)

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return query, key, value, output
        if kv_cache is None or len(kv_cache) < 2:
            return query, key, value, output

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            if self.is_kv_producer and attn_metadata.reshape_cache_event is not None:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        prepared_slots = getattr(attn_metadata, "turboquant_slot_mapping", None)
        if getattr(attn_metadata, "attn_state", None) == AscendAttentionState.DecodeOnly and prepared_slots is not None:
            # The captured writer, like attention, must cover the padded extent.
            # Live request counts change between replays; -1 slots mask padding.
            num_actual_tokens = self._decode_num_tokens(attn_metadata)
        else:
            num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return query, key, value, output

        emulator = self.emulator
        if emulator is None:
            raise RuntimeError("TurboQuant statics were never bound; forward() must run _bind_codec_constants first.")
        scale_cache = self._ensure_scale_cache(kv_cache)

        phase = self._ingestion_phase(attn_metadata, num_actual_tokens)
        slot_mapping, strip_plan = self._ingestion_inputs(attn_metadata, num_actual_tokens, phase)
        if phase is TurboQuantIngestionPhase.PREFILL:
            self.adapter.reshape_and_cache_prefill(
                key=key[:num_actual_tokens].contiguous(),
                value=value[:num_actual_tokens].contiguous(),
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
                plan=strip_plan,
            )
        else:
            self.adapter.reshape_scatter_decode(
                key=key[:num_actual_tokens].contiguous(),
                value=value[:num_actual_tokens].contiguous(),
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
            )
        notify_kv_cache_written()
        return query, key, value, output

    def _ingestion_inputs(
        self,
        attn_metadata: AscendMetadata,
        num_actual_tokens: int,
        phase: TurboQuantIngestionPhase,
    ) -> tuple[torch.Tensor, TurboQuantStripPlan | None]:
        if phase is TurboQuantIngestionPhase.DECODE:
            decode_slots = getattr(attn_metadata, "turboquant_slot_mapping", None)
            if decode_slots is not None:
                return decode_slots[:num_actual_tokens], None
        slot_mapping = getattr(attn_metadata, "turboquant_ingestion_slot_mapping", None)
        if slot_mapping is None:
            # Standalone callers may supply metadata without the builder's
            # fields. In that case the prefill adapter plans on demand.
            return attn_metadata.slot_mapping[:num_actual_tokens].to(torch.int32), None
        return slot_mapping, getattr(attn_metadata, "turboquant_strip_plan", None)

    @staticmethod
    def _ingestion_phase(attn_metadata: AscendMetadata, num_actual_tokens: int) -> TurboQuantIngestionPhase:
        if getattr(attn_metadata, "attn_state", None) == AscendAttentionState.DecodeOnly:
            return TurboQuantIngestionPhase.DECODE
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None)
        num_requests = len(seq_lens_list) if seq_lens_list is not None else 0
        if num_actual_tokens > num_requests:
            return TurboQuantIngestionPhase.PREFILL
        return TurboQuantIngestionPhase.DECODE

    def _gather_dense_kv(
        self, attn_metadata: AscendMetadata, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )
        emulator = self.emulator
        assert emulator is not None

        block_tables = attn_metadata.block_tables[:batch_size]
        packed_key, key_scales, packed_value, value_scales = self.adapter.gather(
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=block_tables,
        )
        dense_key = self.adapter.dequantize_cache(packed_key, key_scales, emulator, dtype)
        dense_value = self.adapter.dequantize_cache(packed_value, value_scales, emulator, dtype)

        padded_context_length = dense_key.shape[1]
        positions = torch.arange(padded_context_length, device=dense_key.device, dtype=torch.int32)
        seq_lens = getattr(attn_metadata, "turboquant_context_lens", None)
        if seq_lens is None:
            seq_lens = attn_metadata.seq_lens[:batch_size].to(device=dense_key.device, dtype=torch.int32)
        else:
            seq_lens = seq_lens[:batch_size]
        live_token_mask = (positions.unsqueeze(0) < seq_lens.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        zero = torch.zeros((), dtype=dtype, device=dense_key.device)
        return torch.where(live_token_mask, dense_key, zero), torch.where(live_token_mask, dense_value, zero)

    def _run_dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_layout: str,
        actual_seq_lengths: Any,
        actual_seq_lengths_kv: Any,
        atten_mask: torch.Tensor | None,
        sparse_mode: int,
    ) -> torch.Tensor:
        if self._use_fia_v2 is None:
            requested = envs.TURBOQUANT_FIA_VERSION.strip().lower()
            if requested == "v2":
                self._use_fia_v2 = True
            elif requested == "v1":
                self._use_fia_v2 = False
            else:
                self._use_fia_v2 = not hasattr(torch_npu, "npu_fused_infer_attention_score")

        if not self._use_fia_v2:
            try:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=atten_mask,
                    input_layout=input_layout,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=sparse_mode,
                )
                return attn_output
            except RuntimeError as refusal:
                logger.warning(
                    "npu_fused_infer_attention_score was refused (%s); retrying on _v2 for the rest of this run.",
                    refusal,
                )
                self._use_fia_v2 = True

        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            atten_mask=atten_mask,
            input_layout=input_layout,
            actual_seq_qlen=actual_seq_lengths,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            sparse_mode=sparse_mode,
        )
        return attn_output

    def _forward_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        emulator = self.emulator
        assert emulator is not None
        batch_size = self._decode_num_tokens(attn_metadata)
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )

        tiling, workspace, query_rot = self._get_decode_launch_buffers(attn_metadata, batch_size)
        fused_output = self.adapter.fused_decode(
            query=query[:batch_size],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=attn_metadata.block_tables[:batch_size],
            context_lens=self._decode_context_lens(attn_metadata, batch_size),
            emulator=emulator,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            output=output[:batch_size],
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
        )
        if fused_output is not None:
            return output

        dense_key, dense_value = self._gather_dense_kv(attn_metadata, batch_size, query.dtype)
        attn_output = self._run_dense_attention(
            query=query[:batch_size].unsqueeze(1),
            key=dense_key,
            value=dense_value,
            input_layout="BSND",
            actual_seq_lengths=None,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            atten_mask=None,
            sparse_mode=0,
        )
        output[:batch_size] = attn_output.view(batch_size, self.num_heads, self.head_size)
        return output

    @staticmethod
    def _decode_num_tokens(attn_metadata: AscendMetadata) -> int:
        tokens = int(getattr(attn_metadata, "turboquant_num_tokens", 0) or 0)
        if tokens:
            return tokens
        return len(attn_metadata.seq_lens_list)

    @staticmethod
    def _decode_context_lens(attn_metadata: AscendMetadata, batch_size: int) -> torch.Tensor:
        context_lens = getattr(attn_metadata, "turboquant_context_lens", None)
        if context_lens is None:
            context_lens = attn_metadata.seq_lens_gpu
        if context_lens is None:
            raise RuntimeError(
                "the TurboQuant decode has no device-side context lengths. "
                "AscendTurboQuantMetadataBuilder publishes turboquant_context_lens; "
                "a metadata object without it and without seq_lens_gpu cannot be decoded "
                "without a host-to-device copy inside the launch."
            )
        return context_lens[:batch_size]

    def _get_decode_launch_buffers(
        self, attn_metadata: AscendMetadata, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        splits = TURBOQUANT_DECODE_SPLITS
        tiling = getattr(attn_metadata, "turboquant_tiling", None)
        workspace = getattr(attn_metadata, "turboquant_workspace", None)
        query_rot = getattr(attn_metadata, "turboquant_query_rot", None)
        if tiling is None or workspace is None or query_rot is None:
            raise RuntimeError(
                "the TurboQuant decode found no builder-owned launch buffers on its metadata "
                "(turboquant_tiling / turboquant_workspace / turboquant_query_rot). They are "
                "published by AscendTurboQuantMetadataBuilder; the operator cannot allocate them "
                "itself without giving every step a fresh address, which a captured graph cannot "
                "follow."
            )
        required_workspace_floats = turboquant_decode_workspace_floats(
            batch_size, self.num_heads, self.head_size, splits
        )
        required_rotated_query_floats = turboquant_decode_query_rot_floats(batch_size, self.num_heads, self.head_size)
        if workspace.numel() < required_workspace_floats or query_rot.numel() < required_rotated_query_floats:
            raise RuntimeError(
                f"the TurboQuant launch buffers are too small for {batch_size} tokens: workspace holds "
                f"{workspace.numel()} floats (needs {required_workspace_floats}) and query_rot {query_rot.numel()} "
                f"(needs {required_rotated_query_floats}). The builder sizes them from max_num_reqs."
            )
        return (
            tiling,
            workspace.narrow(0, 0, required_workspace_floats),
            query_rot.narrow(0, 0, required_rotated_query_floats),
        )

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        attn_output = self._run_dense_attention(
            query=query[:num_tokens],
            key=key[:num_tokens].contiguous(),
            value=value[:num_tokens].contiguous(),
            input_layout="TND",
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
            atten_mask=attn_metadata.attn_mask,
            sparse_mode=3 if attn_metadata.causal else 0,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output

    def _forward_mixed(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        emulator = self.emulator
        assert emulator is not None

        layout = self._mixed_query_layout(attn_metadata, query.device)
        batch_size = len(layout.query_lengths_list)
        total_tokens = layout.num_tokens
        max_query_len = layout.max_query_len
        request_indices = layout.request_indices
        positions_in_request = layout.positions_in_request

        padded_query = torch.zeros(
            (batch_size, max_query_len, self.num_heads, self.head_size),
            dtype=query.dtype,
            device=query.device,
        )
        padded_query[request_indices, positions_in_request] = query[:total_tokens]

        dense_key, dense_value = self._gather_dense_kv(attn_metadata, batch_size, query.dtype)
        padded_kv_length = dense_key.shape[1]

        seq_lens_device = getattr(attn_metadata, "turboquant_context_lens", None)
        if seq_lens_device is None:
            seq_lens = attn_metadata.seq_lens[:batch_size].to(device=query.device, dtype=torch.int64)
        else:
            seq_lens = seq_lens_device[:batch_size]
        query_lengths = layout.query_lengths
        key_positions = torch.arange(padded_kv_length, device=query.device, dtype=torch.int64)
        query_positions = torch.arange(max_query_len, device=query.device, dtype=torch.int64)
        horizon = (seq_lens - query_lengths).unsqueeze(1) + query_positions.unsqueeze(0)
        atten_mask = (key_positions.view(1, 1, -1) > horizon.unsqueeze(-1)).unsqueeze(1)

        attn_output = self._run_dense_attention(
            query=padded_query,
            key=dense_key,
            value=dense_value,
            input_layout="BSND",
            actual_seq_lengths=layout.query_lengths_list,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            atten_mask=atten_mask,
            sparse_mode=0,
        )
        attn_output = attn_output.view(batch_size, max_query_len, self.num_heads, self.head_size)
        output[:total_tokens] = attn_output[request_indices, positions_in_request]
        return output

    @staticmethod
    def _mixed_query_layout(attn_metadata: AscendMetadata, device: torch.device) -> TurboQuantQueryLayout:
        cumulative_lengths = tuple(attn_metadata.actual_seq_lengths_q)
        cached = getattr(attn_metadata, "_turboquant_query_layout_cache", None)
        if cached is not None and cached[:2] == (cumulative_lengths, device):
            return cached[2]

        query_lengths = torch.diff(torch.tensor((0, *cumulative_lengths), dtype=torch.int64))
        query_lengths_list = query_lengths.tolist()
        num_tokens = cumulative_lengths[-1]
        request_indices = torch.repeat_interleave(torch.arange(len(query_lengths_list)), query_lengths)
        row_starts = torch.cumsum(query_lengths, dim=0) - query_lengths
        positions_in_request = torch.arange(num_tokens) - row_starts[request_indices]
        host_layout = torch.cat((request_indices, positions_in_request, query_lengths))
        if device.type != "cpu":
            host_layout = host_layout.pin_memory()
        device_layout = host_layout.to(device, non_blocking=host_layout.is_pinned())
        layout = TurboQuantQueryLayout(
            request_indices=device_layout[:num_tokens],
            positions_in_request=device_layout[num_tokens : 2 * num_tokens],
            query_lengths=device_layout[2 * num_tokens :],
            query_lengths_list=query_lengths_list,
            max_query_len=max(query_lengths_list),
            num_tokens=num_tokens,
        )
        attn_metadata._turboquant_query_layout_cache = (cumulative_lengths, device, layout)
        return layout

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            return self._forward_prefill_no_cache(query, key, value, attn_metadata, output)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_decode(query, attn_metadata, output)
        return self._forward_mixed(query, attn_metadata, output)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None:
            if _stream_is_capturing():
                if (
                    attn_metadata.attn_state != AscendAttentionState.DecodeOnly
                    or not self.adapter.has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER)
                    or not self.adapter.has_kernel(TURBOQUANT_OP_FUSED_DECODE)
                ):
                    raise RuntimeError("TurboQuant full graph capture requires kernel-only single-token decode")
                if self.emulator is None:
                    raise RuntimeError("TurboQuant constants must be warmed up before graph capture")
                if getattr(attn_metadata, "turboquant_slot_mapping", None) is None:
                    raise RuntimeError("TurboQuant decode slot mapping must be prepared before graph capture")
            self._bind_codec_constants(layer, query.device)
        forward_arguments: dict[str, Any] = {
            "layer": layer,
            "query": query,
            "key": key,
            "value": value,
            "kv_cache": kv_cache,
            "attn_metadata": attn_metadata,
            "output": output,
            "output_scale": output_scale,
            "output_block_scale": output_block_scale,
        }
        if attn_metadata is None or _stream_is_capturing():
            return super().forward(**forward_arguments)
        return self.telemetry.run(super().forward, query.device.type, **forward_arguments)
