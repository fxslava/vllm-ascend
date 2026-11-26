# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build persistent TurboQuant launch buffers and scheduler metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_DECODE_MAX_SPLITS,
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_FIA_DUMMY_KV_LEN,
    TURBOQUANT_KERNEL_BLOCK_SIZE,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_STAGING_SLOTS,
    TurboQuantKernelStage,
    TurboQuantOpsConfig,
    turboquant_decode_heads_per_task,
    turboquant_decode_query_rot_floats,
    turboquant_decode_workspace_floats,
    turboquant_tiling_upper_bound_words,
)
from vllm_ascend.attention.turboquant_e8 import E8_OPS_NAMESPACE, E8BatchPlan, prepare_batch
from vllm_ascend.attention.turboquant_ingestion import TurboQuantStripPlan, turboquant_plan_strip_ingestion
from vllm_ascend.attention.turboquant_ops import (
    _is_op_registered,
    _stream_is_capturing,
    load_turboquant_extension,
    turboquant_decode_graph_supported,
)
from vllm_ascend.attention.turboquant_telemetry import (
    DispatchImplementation,
    DispatchObserver,
    create_dispatch_observer,
)


@dataclass
class AscendTurboQuantMetadata(AscendMetadata):
    turboquant_context_lens: torch.Tensor = None
    turboquant_tiling: torch.Tensor = None
    turboquant_workspace: torch.Tensor = None
    turboquant_query_rot: torch.Tensor = None
    turboquant_num_tokens: int = 0
    turboquant_slot_mapping: torch.Tensor | None = None
    turboquant_ingestion_slot_mapping: torch.Tensor | None = None
    turboquant_strip_plan: TurboQuantStripPlan | None = None
    turboquant_e8_batch: E8BatchPlan | None = None


@dataclass(frozen=True)
class TurboQuantQueryLayout:
    request_indices: torch.Tensor
    positions_in_request: torch.Tensor
    query_lengths: torch.Tensor
    query_lengths_list: list[int]
    max_query_len: int
    num_tokens: int


class AscendTurboQuantMetadataBuilder(AscendAttentionMetadataBuilder):
    metadata_cls: type[AscendMetadata] = AscendTurboQuantMetadata

    @classmethod
    def get_cudagraph_support(cls, vllm_config: Any, kv_cache_spec: Any) -> AttentionCGSupport:
        if not getattr(kv_cache_spec, "uses_e8", False) and turboquant_decode_graph_supported():
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
        *,
        dispatch_observer: DispatchObserver | None = None,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.dispatch_observer = dispatch_observer if dispatch_observer is not None else create_dispatch_observer()
        self._max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._num_kv_heads = int(kv_cache_spec.num_kv_heads)
        self._head_size = int(kv_cache_spec.head_size)

        self._block_size = TURBOQUANT_KERNEL_BLOCK_SIZE
        self._num_heads = int(vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config))
        self._num_splits = TURBOQUANT_DECODE_SPLITS
        self._heads_per_task = turboquant_decode_heads_per_task(self._num_heads, self._num_kv_heads)
        ops_config = TurboQuantOpsConfig.from_env()
        self._ops_namespace = ops_config.ops_namespace
        self._prefill_kernel_enabled = ops_config.enables(TurboQuantKernelStage.RESHAPE_AND_CACHE)
        self._uses_e8 = getattr(kv_cache_spec, "uses_e8", ops_config.uses_e8)
        if self._uses_e8:
            self._block_size = kv_cache_spec.block_size
            self._e8_device = device
            self._e8_reader_enabled = ops_config.enables(TurboQuantKernelStage.DECODE)
            return

        self._context_lens = torch.zeros(self._max_num_reqs, dtype=torch.int32, device=device)
        self._decode_slot_mapping = torch.full((self._max_num_reqs,), -1, dtype=torch.int32, device=device)
        self._tiling = torch.zeros(
            turboquant_tiling_upper_bound_words(self._max_num_reqs, TURBOQUANT_DECODE_MAX_SPLITS),
            dtype=torch.int32,
            device=device,
        )
        self._workspace = torch.zeros(
            turboquant_decode_workspace_floats(self._max_num_reqs, self._num_heads, self._head_size, self._num_splits),
            dtype=torch.float32,
            device=device,
        )
        self._query_rot = torch.zeros(
            turboquant_decode_query_rot_floats(self._max_num_reqs, self._num_heads, self._head_size),
            dtype=torch.float32,
            device=device,
        )
        self._staging_slots = TURBOQUANT_STAGING_SLOTS
        self._staging_step = 0
        staging = [torch.zeros(self._max_num_reqs, dtype=torch.int32, device="cpu") for _ in range(self._staging_slots)]
        try:
            staging = [slot.pin_memory() for slot in staging]
            self._staging_pinned = True
        except (RuntimeError, NotImplementedError):
            self._staging_pinned = False
        self._context_lens_host_ring = staging
        self._context_lens_upload_events: list[Any | None] = [None] * self._staging_slots

    def build(self, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False) -> AscendMetadata:
        if self._uses_e8:
            host_lengths = common_attn_metadata._seq_lens_cpu
            if host_lengths is None:
                host_lengths = common_attn_metadata.seq_lens_cpu
            if (
                host_lengths is None
                or host_lengths.device.type != "cpu"
                or getattr(self.speculative_config, "parallel_drafting", False)
            ):
                raise RuntimeError("TurboQuant E8 requires authoritative CPU sequence lengths")
        return super().build(common_prefix_len, common_attn_metadata, fast_build)

    def _build_backend_metadata(
        self,
        common_attn_metadata: Any,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, Any]:
        if getattr(self, "_uses_e8", False):
            return {"turboquant_e8_batch": self._build_e8_batch(common_attn_metadata, seq_lens, query_lens)}
        if getattr(common_attn_metadata, "offload_dummy", False):
            # Dummy/capture runs bypass scheduler allocation. Invalidate the
            # shared source before deriving a cast or strip plan: a stale live
            # slot here would overwrite a previous request's cache during capture.
            common_attn_metadata.slot_mapping.fill_(-1)
        elif envs.VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS:
            self._validate_slot_mapping(common_attn_metadata, block_table, query_lens, seq_lens)
        context_lens, host_context_lens = self._refresh_context_lens(common_attn_metadata, seq_lens)
        tokens = int(context_lens.numel())
        self._reject_multi_token_steps(common_attn_metadata, tokens)
        self._refresh_tiling(host_context_lens, tokens)
        slot_mapping = None
        ingestion_slot_mapping = None
        strip_plan = None
        if getattr(common_attn_metadata, "attn_state", None) == AscendAttentionState.DecodeOnly:
            source = common_attn_metadata.slot_mapping
            if source.numel() < tokens:
                raise RuntimeError("TurboQuant decode slot mapping does not cover the padded capture extent")
            live_tokens = int(common_attn_metadata.num_actual_tokens)
            if not 0 <= live_tokens <= tokens:
                raise RuntimeError("TurboQuant live decode token count exceeds the padded capture extent")
            # Copy/cast before replay into storage shared by every layer and
            # capture size. Padded rows must never write to the cache.
            slot_mapping = self._decode_slot_mapping[:tokens]
            slot_mapping.copy_(source[:tokens])
            slot_mapping[live_tokens:].fill_(-1)
        elif (live_tokens := getattr(common_attn_metadata, "num_actual_tokens", 0)) > 0:
            # Match the backend's phase selection, which counts the padded
            # seq_lens_list entries when it distinguishes mixed from decode.
            prepare_strip_plan = False
            if live_tokens > tokens and self._prefill_kernel_enabled:
                load_turboquant_extension()
                prepare_strip_plan = _is_op_registered(self._ops_namespace, TURBOQUANT_OP_RESHAPE_AND_CACHE)
            if prepare_strip_plan and _stream_is_capturing():
                raise RuntimeError("TurboQuant prefill ingestion planning requires metadata outside graph capture")
            source = common_attn_metadata.slot_mapping[:live_tokens]
            # Snapshot planned slots even for an int32 source. The runner can
            # reuse its slot buffer and the plan must describe the launch's slots.
            ingestion_slot_mapping = source.to(torch.int32, copy=prepare_strip_plan)
            if prepare_strip_plan:
                strip_plan = turboquant_plan_strip_ingestion(ingestion_slot_mapping.tolist(), self._block_size)
        return {
            "turboquant_context_lens": context_lens,
            "turboquant_tiling": self._tiling,
            "turboquant_workspace": self._workspace,
            "turboquant_query_rot": self._query_rot,
            "turboquant_num_tokens": tokens,
            "turboquant_slot_mapping": slot_mapping,
            "turboquant_ingestion_slot_mapping": ingestion_slot_mapping,
            "turboquant_strip_plan": strip_plan,
        }

    def _build_e8_batch(self, common: Any, seq_lens: torch.Tensor, query_lens: torch.Tensor) -> E8BatchPlan:
        if getattr(common, "offload_dummy", False):
            return E8BatchPlan((), 0, self._block_size, None)
        host_blocks = common.turboquant_block_table_cpu
        if host_blocks is None:
            raise RuntimeError("TurboQuant E8 requires the scheduler's CPU block-table mirror")
        use_reader = (
            self._e8_reader_enabled and load_turboquant_extension() and _is_op_registered(E8_OPS_NAMESPACE, "reader")
        )
        return prepare_batch(
            host_blocks,
            seq_lens,
            query_lens,
            num_tokens=common.num_actual_tokens,
            num_blocks=common.turboquant_num_blocks,
            block_size=self._block_size,
            num_kv_heads=self._num_kv_heads,
            device=self._e8_device,
            use_reader=use_reader,
        )

    def _validate_slot_mapping(
        self,
        common_attn_metadata: Any,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Diagnostic only: check live writes before uploading lengths or tiling."""
        if _stream_is_capturing():
            raise RuntimeError("TurboQuant slot validation requires metadata construction outside graph capture")
        num_reqs = int(common_attn_metadata.num_reqs)
        num_tokens = int(common_attn_metadata.num_actual_tokens)
        lengths = []
        remaining = num_tokens
        for length in query_lens[:num_reqs].tolist():
            if remaining == 0:
                break  # FIA/graph padding requests do not own live write slots.
            if length < 0 or length > remaining:
                raise ValueError("TurboQuant live query lengths do not cover the current request/token count")
            lengths.append(length)
            remaining -= length
        num_live_reqs = len(lengths)
        contexts = seq_lens[:num_live_reqs].tolist()
        if remaining or len(contexts) != num_live_reqs:
            raise ValueError("TurboQuant live query lengths do not cover the current request/token count")
        if block_table.shape[0] < num_live_reqs or common_attn_metadata.slot_mapping.numel() < num_tokens:
            raise ValueError("TurboQuant block table or slot mapping is shorter than the current batch")
        slots = common_attn_metadata.slot_mapping[:num_tokens].to(device="cpu", dtype=torch.int64)
        pages = block_table[:num_live_reqs].to(device="cpu", dtype=torch.int64)
        start = 0
        for request, (length, context) in enumerate(zip(lengths, contexts)):
            if length < 0 or context < length:
                raise ValueError(f"TurboQuant request {request} has query_len={length}, context_len={context}")
            positions = torch.arange(context - length, context, dtype=torch.int64)
            logical_pages = positions // self._block_size
            if length and int(logical_pages[-1]) >= pages.shape[1]:
                raise ValueError(f"TurboQuant request {request} exceeds its block-table row")
            physical_pages = pages[request, logical_pages]
            if bool((physical_pages < 0).any()):
                raise ValueError(f"TurboQuant request {request} has an unmapped live block-table entry")
            expected = physical_pages * self._block_size + positions % self._block_size
            actual = slots[start : start + length]
            mismatch = torch.nonzero(actual != expected, as_tuple=False).flatten()
            if mismatch.numel():
                row = int(mismatch[0])
                raise ValueError(
                    f"TurboQuant slot_mapping mismatch: request={request}, token_row={start + row}, "
                    f"position={int(positions[row])}, actual_slot={int(actual[row])}, "
                    f"expected_slot={int(expected[row])}, physical_page={int(physical_pages[row])}, "
                    f"kernel_block_size={self._block_size}"
                )
            start += length
        unique_slots, counts = torch.unique(slots, return_counts=True)
        duplicate_slots = unique_slots[counts > 1]
        if duplicate_slots.numel():
            slot = int(duplicate_slots[0])
            rows = torch.nonzero(slots == slot, as_tuple=False).flatten().tolist()
            raise ValueError(f"TurboQuant live writes alias physical slot {slot} at token rows {rows}")

    def _reject_multi_token_steps(self, common_attn_metadata: Any, tokens: int) -> None:
        per_req = int(getattr(common_attn_metadata, "decode_token_per_req", 1) or 1)
        if per_req > 1:
            raise RuntimeError(
                "the TurboQuant fused decode cannot tile a multi-token decode step: this step "
                f"carries {per_req} query rows per request over {tokens} descriptor rows, and the "
                "tiling descriptors are indexed by token while this builder fills one row per "
                "request. Speculative decoding and MTP need the per-request context lengths "
                "expanded per accepted token first."
            )

    def _refresh_context_lens(
        self, common_attn_metadata: Any, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert seq_lens.device.type == "cpu", (
            f"_refresh_context_lens stages `seq_lens` into a host ring, but seq_lens is on "
            f"device {seq_lens.device!r} (device type {seq_lens.device.type!r}). A device "
            "tensor would force a per-step device-to-host synchronisation here and break "
            "ACL graph capture; the CPU mirror is what this method exists to upload."
        )
        num_reqs = int(common_attn_metadata.num_reqs)
        padded_num_requests = int(seq_lens.shape[0])
        if padded_num_requests > self._max_num_reqs:
            raise RuntimeError(
                f"the TurboQuant builder was sized for {self._max_num_reqs} requests but this step "
                f"carries {padded_num_requests}. max_num_seqs is what sizes the launch buffers."
            )
        num_live_requests = min(num_reqs, padded_num_requests)
        host_context_lens = self._staging_slot()
        self._staging_step += 1
        host_context_lens[:num_live_requests] = seq_lens[:num_live_requests].to(torch.int32)
        if padded_num_requests > num_live_requests:
            tail = seq_lens[num_reqs:]
            if tail.numel():
                invalid_padding = (tail != 0) & (tail != TURBOQUANT_FIA_DUMMY_KV_LEN)
                if bool(invalid_padding.any()):
                    invalid_request_indices = (
                        num_reqs + torch.nonzero(invalid_padding, as_tuple=False).flatten()
                    ).tolist()
                    raise RuntimeError(
                        f"seq_lens[{num_reqs}:] carries a dirty padded tail: expected all 0 "
                        f"(cudagraph padding) or all {TURBOQUANT_FIA_DUMMY_KV_LEN} (the FIA dummy "
                        f"build() appends), but seq_lens has max={int(tail.max())}, "
                        f"min={int(tail.min())} at indices {invalid_request_indices} (shape "
                        f"{tuple(seq_lens.shape)}, num_reqs={num_reqs}). Refusing rather than "
                        "masking a stale length."
                    )
            host_context_lens[num_live_requests:padded_num_requests] = 0
        self._context_lens[:padded_num_requests].copy_(
            host_context_lens[:padded_num_requests], non_blocking=self._staging_pinned
        )
        if self._staging_pinned and self._context_lens.device.type != "cpu":
            index = (self._staging_step - 1) % self._staging_slots
            event = self._context_lens_upload_events[index]
            if event is None:
                event = torch.npu.Event()
                self._context_lens_upload_events[index] = event
            event.record()
        return self._context_lens[:padded_num_requests], host_context_lens

    def _staging_slot(self) -> torch.Tensor:
        slots = self._staging_slots
        ring = self._context_lens_host_ring
        assert slots == TURBOQUANT_STAGING_SLOTS, (
            f"the TurboQuant host staging ring has {slots} slots, but the asynchronous upload "
            f"contract needs exactly {TURBOQUANT_STAGING_SLOTS} slots."
        )
        assert len(ring) == slots, (
            f"`_context_lens_host_ring` has {len(ring)} buffers but `_staging_slots` says "
            f"{slots}; the ring and the modulo that indexes it must agree."
        )
        assert isinstance(self._staging_step, int) and self._staging_step >= 0, (
            f"`_staging_step` must be a non-negative int, got {self._staging_step!r}; a negative "
            "step would wrap the modulo onto an arbitrary slot."
        )
        index = self._staging_step % slots
        assert 0 <= index < slots, f"staging index {index} is outside the {slots}-slot ring"
        for position, slot in enumerate(ring):
            assert slot.shape == (self._max_num_reqs,), (
                f"staging slot {position} has shape {tuple(slot.shape)}, expected "
                f"({self._max_num_reqs},): every slot must cover the whole device buffer."
            )
            assert slot.dtype == torch.int32, (
                f"staging slot {position} has dtype {slot.dtype}, expected torch.int32: it is "
                "uploaded into the int32 context-lens buffer."
            )
            assert slot.device.type == "cpu", (
                f"staging slot {position} is on device {slot.device!r}, expected a CPU tensor: "
                "a device staging slot defeats the single host-to-device upload."
            )
        events = getattr(self, "_context_lens_upload_events", None)
        event = events[index] if events is not None else None
        if event is not None and not event.query():
            ring[index] = torch.empty_like(ring[index], pin_memory=self._staging_pinned)
        return ring[index]

    def _refresh_tiling(self, host_context_lens: torch.Tensor, tokens: int) -> None:
        namespace = getattr(torch.ops, self._ops_namespace, None)
        build_tiling = getattr(namespace, TURBOQUANT_OP_BUILD_DECODE_TILING, None) if namespace else None
        if build_tiling is None:
            return
        if self._block_size % 64 or self._block_size & (self._block_size - 1):
            raise RuntimeError(
                f"the TurboQuant fused decode's closed-form tile cursor needs block_size to be a "
                f"power-of-two multiple of 64, but this engine's KV cache spec carries "
                f"block_size={self._block_size}. On a hybrid model that value comes from the "
                f"Mamba/kpool page alignment in vllm_ascend.patch.platform.patch_mamba_config; "
                f"otherwise it is the --block-size the engine was started with."
            )
        self.dispatch_observer.record(
            TURBOQUANT_OP_BUILD_DECODE_TILING,
            DispatchImplementation.CUSTOM_OP,
            context_lens=host_context_lens[:tokens],
            tiling=self._tiling,
        )
        build_tiling(
            host_context_lens[:tokens],
            self._tiling,
            self._num_heads,
            self._num_kv_heads,
            self._head_size,
            self._block_size,
            self.max_num_blocks_per_req,
            self._num_splits,
            self._heads_per_task,
        )
