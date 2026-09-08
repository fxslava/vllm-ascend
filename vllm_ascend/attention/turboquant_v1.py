#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""TurboQuant 4-bit KV cache attention backend for Ascend 950PR and 910B.

The cache is plain contiguous ND -- ``(2, num_blocks, block_size, num_kv_heads,
head_size // 2)`` int8, two 4-bit codes per byte.

Rotation is a *pure runtime activation transform*.  Model weights are never
touched: ``q``, ``k`` and ``v`` reach the operators exactly as the projections
and RoPE produced them, and the kernels apply

    Pi x = D (H (D x)),   D = diag(+-1),   H = normalised Walsh-Hadamard

in UB.  Pi is symmetric (``Pi^T = D^T H^T D^T = D H D = Pi``) and an involution
(``Pi^2 = D H D D H D = I``), so a single transform both rotates K, V and Q on
the way in and un-rotates the attention accumulator on the way out.  Being
orthogonal, it also leaves scores alone: ``(Pi q) . (Pi k) == q . k``.

Keeping the rotation out of the weights is what makes RoPE correct: RoPE is
applied to q and k *before* anything is rotated, so the position encoding never
sees the rotated basis.
"""

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import notify_kv_cache_written

# Seed of the +-1 diagonal of Pi.  It is fixed so that every rank, every layer
# and every restart agree on the rotation; the cache is only ever read back by
# the same transform that wrote it.
TURBOQUANT_PI_SEED = 0x5F3759DF

# Two 4-bit codes per byte.
TURBOQUANT_PACK_FACTOR = 2

_PI_SIGN_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def turboquant_pi_signs(head_size: int, device: torch.device) -> torch.Tensor:
    """Return the deterministic +-1 diagonal of Pi for ``head_size`` channels.

    Generated from an explicit LCG rather than ``torch.Generator`` so that the
    host C++ reference (``csrc/tests/reference/turbo_quant_cpu.h``) produces the
    identical vector without either side depending on the other's RNG.
    """
    key = (head_size, str(device))
    cached = _PI_SIGN_CACHE.get(key)
    if cached is not None:
        return cached
    state = (TURBOQUANT_PI_SEED + head_size) & 0xFFFFFFFF
    bits = []
    for _ in range(head_size):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        bits.append(1.0 if (state >> 16) & 1 else -1.0)
    signs = torch.tensor(bits, dtype=torch.float32, device=device)
    _PI_SIGN_CACHE[key] = signs
    return signs


def walsh_hadamard(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Normalised fast Walsh-Hadamard transform along ``dim``.

    Host reference for the vectorised Ascend C transform; used by the tests and
    by anyone reproducing the cache contents offline.
    """
    length = x.shape[dim]
    if length & (length - 1):
        raise ValueError(f"Walsh-Hadamard needs a power-of-two length, got {length}")
    out = x.movedim(dim, -1).contiguous()
    lead = out.shape[:-1]
    out = out.reshape(-1, length)
    stride = 1
    while stride < length:
        out = out.view(-1, length // (2 * stride), 2, stride)
        top = out[:, :, 0, :]
        bottom = out[:, :, 1, :]
        out = torch.stack((top + bottom, top - bottom), dim=2)
        out = out.reshape(-1, length)
        stride *= 2
    out = out.reshape(*lead, length) / (length**0.5)
    return out.movedim(-1, dim)


def apply_pi(x: torch.Tensor, pi_signs: torch.Tensor) -> torch.Tensor:
    """Host reference for ``Pi x = D (H (D x))`` over the last axis.

    Pi is its own inverse, so this is both the rotation and the un-rotation.
    """
    return walsh_hadamard(x * pi_signs, dim=-1) * pi_signs


class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    """Backend descriptor for the 4-bit TurboQuant KV cache."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["AscendTurboQuantAttentionBackendImpl"]:
        return AscendTurboQuantAttentionBackendImpl

    @classmethod
    def supports_pcp(cls) -> bool:
        # Prefill context parallelism is owned by the dense GQA implementation.
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # Contiguous ND, two 4-bit codes per byte.  Deliberately not a 5D NZ
        # view: the kernels address the cache as flat bytes.
        return (2, num_blocks, block_size, num_kv_heads, head_size // TURBOQUANT_PACK_FACTOR)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention over a 4-bit TurboQuant KV cache.

    Activate it on an existing layer with :func:`activate_turboquant_backend`,
    which swaps the impl class the same way the C8 KV cache scheme does.

    Scope: ``PrefillNoCache`` runs at full precision over the K/V of the current
    batch (nothing is read back from the cache), and ``DecodeOnly`` runs the
    TurboQuant paged kernel.  Chunked prefill and prefill-with-cache-hit would
    need a paged prefill kernel and are rejected explicitly rather than silently
    falling back to a path that cannot read this cache layout.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key_scale_cache: torch.Tensor | None = None
        self.value_scale_cache: torch.Tensor | None = None
        self._pi_signs: torch.Tensor | None = None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # Nothing is folded into the weights: q_proj, k_proj, v_proj and o_proj
        # stay exactly as loaded, which is what keeps RoPE correct.
        super().process_weights_after_loading(act_dtype)
        logger.info_once("[vllm-ascend/turboquant] 4-bit KV cache active; Pi is applied at runtime in the kernel")

    def pi_signs(self, device: torch.device) -> torch.Tensor:
        if self._pi_signs is None:
            self._pi_signs = turboquant_pi_signs(self.head_size, device)
        return self._pi_signs

    def _ensure_scale_caches(self, kv_cache: tuple[torch.Tensor, ...]) -> None:
        """Allocate the fp32 scale planes that accompany the packed cache.

        The packed cache shape is fixed by ``get_kv_cache_shape`` and has no room
        for a per-vector scale, so the scales live in a side allocation of
        ``(num_blocks, num_kv_heads, block_size)`` -- 6% on top of the payload at
        head_size 128.
        """
        if self.key_scale_cache is not None:
            return
        num_blocks, block_size, num_kv_heads, _ = kv_cache[0].shape
        shape = (num_blocks, num_kv_heads, block_size)
        options = {"dtype": torch.float32, "device": kv_cache[0].device}
        self.key_scale_cache = torch.zeros(shape, **options)
        self.value_scale_cache = torch.zeros(shape, **options)

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) <= 1:
            return query, key, value, output

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            # KV-sharing layers consume another layer's cache; writing here
            # would overwrite shared slots.
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        self._ensure_scale_caches(kv_cache)
        num_actual_tokens = attn_metadata.num_actual_tokens
        encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
        slots = attn_metadata.slot_mapping if encoder_decoder else attn_metadata.slot_mapping[:num_actual_tokens]

        # Post-RoPE key and raw projection value, straight through: the kernel
        # applies Pi and quantises, nothing is pre-rotated on the host.
        torch.ops._C_ascend.npu_turboquant_reshape_and_cache(
            key if encoder_decoder else key[:num_actual_tokens],
            value if encoder_decoder else value[:num_actual_tokens],
            self.key_cache,
            self.value_cache,
            self.key_scale_cache,
            self.value_scale_cache,
            slots.to(torch.int32),
            self.pi_signs(key.device),
        )
        notify_kv_cache_written()
        return query, key, value, output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        num_tokens = query.shape[0]
        # Post-RoPE query.  The kernel rotates it, runs softmax and the value
        # accumulation in the rotated basis, and applies Pi once more on output.
        torch.ops._C_ascend.npu_turboquant_paged_attention(
            query[:num_tokens],
            self.key_cache,
            self.value_cache,
            self.key_scale_cache,
            self.value_scale_cache,
            attn_metadata.block_tables.to(torch.int32),
            attn_metadata.seq_lens.to(torch.int32),
            self.pi_signs(query.device),
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            output[:num_tokens].view(num_tokens, self.num_heads, self.head_size),
        )
        return output

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Full-precision prefill over the current batch's K/V.

        Nothing is read back from the quantised cache here, so this is exact.
        """
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:num_tokens],
            key=key[:num_tokens],
            value=value[:num_tokens],
            atten_mask=attn_metadata.attn_mask,
            input_layout="TND",
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_qlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        state = attn_metadata.attn_state
        if state == AscendAttentionState.DecodeOnly:
            return self.forward_paged_attention(query, attn_metadata, output)
        if state == AscendAttentionState.PrefillNoCache:
            return self._forward_prefill_no_cache(query, key, value, attn_metadata, output)
        raise NotImplementedError(
            f"TurboQuant KV cache supports PrefillNoCache and DecodeOnly, got {state.name}. "
            "Chunked prefill needs a paged prefill kernel over the 4-bit cache."
        )

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
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendTurboQuantAttentionBackendImpl"
            )
        if self.sliding_window is not None:
            raise NotImplementedError("TurboQuant KV cache does not implement sliding window attention")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        if self.key_cache is None and kv_cache is not None and len(kv_cache) >= 2:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        if key is not None and value is not None:
            query, key, value, output = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


def activate_turboquant_backend(layer: torch.nn.Module) -> None:
    """Swap an already-built attention layer onto the TurboQuant impl.

    Mirrors how ``AscendC8KVCacheAttentionMethod`` upgrades a layer, so a
    quantization scheme can opt a model in without forking the model runner.
    """
    layer.kv_cache_torch_dtype = torch.int8
    impl = getattr(layer, "impl", None)
    if impl is not None:
        impl.__class__ = AscendTurboQuantAttentionBackendImpl
        impl.key_scale_cache = None
        impl.value_scale_cache = None
        impl._pi_signs = None
