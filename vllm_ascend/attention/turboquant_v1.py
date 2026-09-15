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

``q``, ``k`` and ``v`` reach the operators exactly as the projections and RoPE
produced them, and the kernels apply

    Pi x = D (H (D x)),   D = diag(+-1),   H = normalised Walsh-Hadamard

in UB.  Pi is symmetric (``Pi^T = D^T H^T D^T = D H D = Pi``) and an involution
(``Pi^2 = D H D D H D = I``).  Being orthogonal, it leaves scores alone:
``(Pi q) . (Pi k) == q . k``.  Keeping Q, K and V rotation out of the weights is
what makes RoPE correct: RoPE is applied *before* anything is rotated.

The decode's output is left in the rotated basis, ``O~ = softmax(q k^T) V~``:
the combine kernel no longer un-rotates it.  Where it goes back out depends on
the layer, and :attr:`AscendTurboQuantAttentionBackendImpl.output_rotation_folded`
records which:

* folded -- the checkpoint's ``o_proj`` was rewritten offline to
  ``W_o (I_H (x) Pi)`` (``scripts/tq_fold_output_rotation.py``), so the
  projection un-rotates for free.  Prefill then has to produce ``O~`` as well,
  and does by rotating V before the dense attention.
* not folded -- the default, and the only option for a layer with an
  elementwise output gate between attention and ``o_proj``.  The decode
  un-rotates its own output with ``npu_turboquant_rotate_q``, and prefill is
  unchanged.

See :mod:`vllm_ascend.attention.turboquant_rotation` for the fold itself.
"""

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.turboquant_rotation import output_rotation_is_folded, turboquant_pi_signs
from vllm_ascend.attention.utils import notify_kv_cache_written

TURBOQUANT_PACK_FACTOR = 2

TURBOQUANT_BURST_FLOATS = 8

TURBOQUANT_TILE_ROWS = 16

TURBOQUANT_LEVELS = 16

TURBOQUANT_LLOYD_MAX_CENTROIDS = (
    -2.7325895709951710, -2.0690172265313920, -1.6180463860218863, -1.2562311973471796,
    -0.9423404564869651, -0.6567591185324659, -0.3880482994902919, -0.1283950298511473,
    0.1283950298511473, 0.3880482994902919, 0.6567591185324659, 0.9423404564869651,
    1.2562311973471796, 1.6180463860218863, 2.0690172265313920, 2.7325895709951710,
)

TURBOQUANT_LLOYD_MAX_THRESHOLDS = (
    -2.4008033987632817, -1.8435318062766393, -1.4371387916845330, -1.0992858269170722,
    -0.7995497875097155, -0.5224037090113789, -0.2582216646707196, 0.0,
    0.2582216646707196, 0.5224037090113789, 0.7995497875097155, 1.0992858269170722,
    1.4371387916845330, 1.8435318062766393, 2.4008033987632817,
)

_CODEC_TABLE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_HADAMARD16_CACHE: dict[str, torch.Tensor] = {}

TURBOQUANT_ROTATE_TILE = 16

_WORKSPACE_MEMO_LIMIT = 1024


def _is_capturing() -> bool:
    """Whether an ACL graph capture is in progress.

    Reading the flag needs a live forward context, and the answer is only ever
    used to refuse an allocation that would be unsafe *inside* a capture. Where
    there is no forward context at all there is no capture either, so a missing
    one is answered rather than raised -- that keeps a diagnostic from turning
    into the failure it exists to describe.
    """
    try:
        return bool(_EXTRA_CTX.capturing)
    except (AssertionError, AttributeError, RuntimeError):
        return False


def turboquant_hadamard16(device: torch.device) -> torch.Tensor:
    """Return the 16x16 fp16 Hadamard constant the Cube rotation path uses.

    ``H[i][j] = (-1) ** popcount(i & j)``: Sylvester's construction, symmetric,
    holding only +-1 -- both exactly representable in fp16, so the Cube stage of
    the rotation contributes no arithmetic error of its own.

    Sylvester also factorises the transform. For ``D = 16 R`` the butterfly
    stages of stride 1, 2, 4 and 8 are exactly a right multiply by this matrix,
    whatever ``D`` is, so four stages of the query's rotation collapse into a
    single Mmad and only the ``log2(R)`` whole-row stages above stride 8 stay on
    the vector unit.  Mirrors ``FillHadamard16Half`` in
    ``csrc/attention/turboquant/turboquant_rotate_q.h``.
    """
    key = str(device)
    cached = _HADAMARD16_CACHE.get(key)
    if cached is not None:
        return cached
    rows = []
    for i in range(TURBOQUANT_ROTATE_TILE):
        rows.append([
            -1.0 if bin(i & j).count("1") & 1 else 1.0
            for j in range(TURBOQUANT_ROTATE_TILE)
        ])
    matrix = torch.tensor(rows, dtype=torch.float16, device=device)
    _HADAMARD16_CACHE[key] = matrix
    return matrix


def turboquant_scale_slot(num_kv_heads: int) -> int:
    """fp32 lanes one token occupies in the scale plane.

    A token's K scales for every kv head are followed by its V scales, padded to
    a whole 32-byte burst.  That padding is what lets the kernel scatter a
    token's scales with one aligned ``DataCopy`` instead of ``2 * num_kv_heads``
    four-byte writes, which were both a DMA-transaction disaster and a hazard --
    two cores could land in the same 32-byte line.  It costs 25% on top of the
    payload at ``num_kv_heads == 2`` and 12.5% from 4 upward.
    """
    lanes = 2 * num_kv_heads
    return -(-lanes // TURBOQUANT_BURST_FLOATS) * TURBOQUANT_BURST_FLOATS


def turboquant_codec_tables(head_size: int, batch_rows: int, device: torch.device) -> torch.Tensor:
    """Build the codec's constant-table image.

    The sign patterns, XOR shuffle offsets and nibble split tables are a pure
    function of ``(head_size, batch_rows)``.  They used to be regenerated inside
    the kernel on every launch -- roughly forty dependent vector instructions and
    their pipe barriers on the critical path of each decode step.  The host
    builds them once here and the kernel pulls the finished image into UB with a
    single aligned ``DataCopy``.

    Layout, in 4-byte words, mirroring ``TurboQuantCodec<4>::ConstTableWords``:

        [0, 6D)         sign_[s] then xorOffset_[s], for strides 1, 2, 4
        [6D, 7D)        evenOffset_ then oddOffset_, D/2 words each
        [7D, 7D + B)    expandOffset_
        [7D + B, +B)    oddSelect_                      (B = D * batch_rows)
        [7D + 2B, +16) centroid_, the Lloyd-Max reconstruction levels

    ``sign_``, ``oddSelect_`` and ``centroid_`` are fp32 bit patterns, the rest
    uint32 byte offsets for Gather; everything is four bytes wide so one int32
    copy moves the lot and the device needs no cast.  The centroid table is
    appended rather than prepended so that adding it left every pre-existing
    offset in the image unchanged; ``head_size`` is a power of two at least 64,
    so every section boundary is still a whole 32-byte burst.
    """
    key = (head_size, batch_rows, str(device))
    cached = _CODEC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    word = 4
    channels = torch.arange(head_size, dtype=torch.int64)
    parts: list[torch.Tensor] = []
    for stage in range(3):
        stride = 1 << stage
        sign = (1 - 2 * ((channels // stride) & 1)).to(torch.float32)
        parts.append(sign.view(torch.int32))
        parts.append((word * (channels ^ stride)).to(torch.int32))

    pairs = torch.arange(head_size // TURBOQUANT_PACK_FACTOR, dtype=torch.int64)
    parts.append((word * TURBOQUANT_PACK_FACTOR * pairs).to(torch.int32))
    parts.append((word * TURBOQUANT_PACK_FACTOR * pairs + word).to(torch.int32))

    batch = torch.arange(head_size * batch_rows, dtype=torch.int64)
    parts.append((word * (batch // TURBOQUANT_PACK_FACTOR)).to(torch.int32))
    parts.append((batch % TURBOQUANT_PACK_FACTOR).to(torch.float32).view(torch.int32))

    centroids = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
    parts.append(centroids.view(torch.int32))

    tables = torch.cat(parts).contiguous().to(device)
    expected = 7 * head_size + 2 * head_size * batch_rows + TURBOQUANT_LEVELS
    if tables.numel() != expected:
        raise RuntimeError(f"codec table image is {tables.numel()} words, expected {expected}")
    _CODEC_TABLE_CACHE[key] = tables
    return tables


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
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
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
        self.scale_cache: torch.Tensor | None = None
        self._pi_signs: torch.Tensor | None = None
        self.decode_workspace: torch.Tensor | None = None
        self._workspace_floats: dict[tuple[int, int], int] = {}
        self.rotated_query: torch.Tensor | None = None
        self._hadamard16: torch.Tensor | None = None
        self.output_rotation_folded = False

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        torch.ops._C_ascend.npu_turboquant_vector_core_num()
        logger.info_once(
            "[vllm-ascend/turboquant] 4-bit KV cache active; Pi is applied at runtime to K/V on the write path and "
            "to Q before each decode. The decode output stays rotated for a folded o_proj and is un-rotated on "
            "the device otherwise."
        )

    def pi_signs(self, device: torch.device) -> torch.Tensor:
        if self._pi_signs is None:
            self._pi_signs = turboquant_pi_signs(self.head_size, device)
        return self._pi_signs

    def codec_tables(self, device: torch.device, batch_rows: int) -> torch.Tensor:
        return turboquant_codec_tables(self.head_size, batch_rows, device)

    def hadamard16(self, device: torch.device) -> torch.Tensor:
        if self._hadamard16 is None:
            self._hadamard16 = turboquant_hadamard16(device)
        return self._hadamard16

    def _ensure_scale_cache(self, kv_cache: tuple[torch.Tensor, ...]) -> None:
        """Allocate the fp32 scale plane that accompanies the packed cache.

        The packed cache shape is fixed by ``get_kv_cache_shape`` and has no room
        for a per-vector scale, so the scales live in a side allocation shaped
        ``(num_blocks, block_size, scale_slot)``.  Indexing by token rather than
        by head is what makes the kernel's scatter a single aligned burst; see
        :func:`turboquant_scale_slot` for what the padding costs.
        """
        if self.scale_cache is not None:
            return
        num_blocks, block_size, num_kv_heads, _ = kv_cache[0].shape
        shape = (num_blocks, block_size, turboquant_scale_slot(num_kv_heads))
        self.scale_cache = torch.zeros(shape, dtype=torch.float32, device=kv_cache[0].device)

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
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        self._ensure_scale_cache(kv_cache)
        num_actual_tokens = attn_metadata.num_actual_tokens
        encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
        slots = attn_metadata.slot_mapping if encoder_decoder else attn_metadata.slot_mapping[:num_actual_tokens]

        torch.ops._C_ascend.npu_turboquant_reshape_and_cache(
            key if encoder_decoder else key[:num_actual_tokens],
            value if encoder_decoder else value[:num_actual_tokens],
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            slots.to(torch.int32),
            self.pi_signs(key.device),
            self.codec_tables(key.device, 1),
        )
        notify_kv_cache_written()
        return query, key, value, output

    def _decode_workspace(self, num_tokens: int, max_blocks_per_seq: int, device: torch.device) -> torch.Tensor:
        """Return scratch for the decode split/combine stages.

        The split stage writes one partial per (token, head, sequence split) and
        the combine stage reduces them.  That buffer used to be an ``at::empty``
        inside the operator, which put an allocation on every decode step and
        left the operator uncapturable: a graph replays the addresses it was
        captured with, and a fresh allocation each step is a different address.

        So the buffer lives here instead, is grown only when a decode needs more
        than any seen before, and is otherwise handed to the kernel unchanged.
        Steady-state decode allocates nothing.  "More" is deliberately not "a
        bigger batch": a batch small enough to leave cores idle is split further
        along the sequence, so the peak requirement can sit at a *small* decode.

        The size comes from the operator's own arithmetic rather than a copy of
        it -- the split count depends on the device's vector core count, which
        only the C++ side knows -- and is memoised per shape so the hot path
        costs a dict lookup rather than an operator dispatch.
        """
        key = (num_tokens, max_blocks_per_seq)
        needed = self._workspace_floats.get(key)
        if needed is None:
            if len(self._workspace_floats) >= _WORKSPACE_MEMO_LIMIT:
                self._workspace_floats.clear()
            needed = int(
                torch.ops._C_ascend.npu_turboquant_workspace_size(
                    num_tokens, self.num_heads, self.head_size, max_blocks_per_seq
                )
            )
            self._workspace_floats[key] = needed

        workspace = self.decode_workspace
        if workspace is not None and workspace.numel() >= needed:
            return workspace

        if _is_capturing():
            raise RuntimeError(
                "[vllm-ascend/turboquant] the decode workspace would have to grow to "
                f"{needed} float32 words during a graph capture "
                f"(num_tokens={num_tokens}, max_blocks_per_seq={max_blocks_per_seq}). "
                "Run this shape once outside capture so the buffer is sized first."
            )

        self.decode_workspace = torch.empty(needed, dtype=torch.float32, device=device)
        return self.decode_workspace

    def _rotated_query(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return the fp32 buffer ``npu_turboquant_rotate_q`` writes into.

        The rotation used to happen inside the decode split, once per
        ``(token, kv_head, sequence split)`` and again per query head of the
        group.  It now happens once per ``(token, head)`` in its own operator,
        which needs somewhere to put the result.

        Persistent for exactly the reason ``decode_workspace`` is: a graph
        replays the addresses it was captured with, so a fresh allocation on
        every decode step would both cost an allocation and make the operator
        uncapturable.  The requirement is monotonic in ``num_tokens`` -- unlike
        the workspace's, which is not -- so growing it can only ever happen on
        the way up.
        """
        needed = num_tokens * self.num_heads * self.head_size
        buffer = self.rotated_query
        if buffer is not None and buffer.numel() >= needed:
            return buffer[:needed].view(num_tokens, self.num_heads, self.head_size)

        if _is_capturing():
            raise RuntimeError(
                "[vllm-ascend/turboquant] the rotated-query buffer would have to grow to "
                f"{needed} float32 words during a graph capture (num_tokens={num_tokens}). "
                "Run this shape once outside capture so the buffer is sized first."
            )

        self.rotated_query = torch.empty(needed, dtype=torch.float32, device=device)
        return self.rotated_query.view(num_tokens, self.num_heads, self.head_size)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        num_tokens = query.shape[0]
        rotated_query = self._rotated_query(num_tokens, query.device)
        torch.ops._C_ascend.npu_turboquant_rotate_q(
            query[:num_tokens],
            self.pi_signs(query.device),
            self.codec_tables(query.device, 1),
            self.hadamard16(query.device),
            rotated_query,
        )
        block_tables = attn_metadata.block_tables.to(torch.int32)
        attention_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        torch.ops._C_ascend.npu_turboquant_paged_attention(
            rotated_query,
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            attn_metadata.seq_lens.to(torch.int32),
            self.codec_tables(query.device, TURBOQUANT_TILE_ROWS),
            self._decode_workspace(num_tokens, block_tables.shape[1], query.device),
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            attention_output,
        )
        if not self.output_rotation_folded:
            torch.ops._C_ascend.npu_turboquant_rotate_q(
                attention_output,
                self.pi_signs(query.device),
                self.codec_tables(query.device, 1),
                self.hadamard16(query.device),
                rotated_query,
            )
            attention_output.copy_(rotated_query)
        return output

    def _rotate_value_for_prefill(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``V~ = Pi V`` in ``value``'s own dtype.

        For a folded o_proj the prefill has to hand the projection the rotated
        basis the decode does. Rotating V is sufficient -- ``softmax(q k^T)(V Pi)
        = O Pi`` -- and it is ``H_KV`` vectors per token rather than the ``H_Q`` a
        rotation of the output would be. Q and K stay unrotated, so RoPE and the
        scores are untouched.

        Allocates: prefill is not replayed from a captured graph, and a buffer
        sized for the longest prompt would sit in HBM for the whole run.
        """
        value = value.contiguous()
        rotated = torch.empty(value.shape, dtype=torch.float32, device=value.device)
        torch.ops._C_ascend.npu_turboquant_rotate_q(
            value,
            self.pi_signs(value.device),
            self.codec_tables(value.device, 1),
            self.hadamard16(value.device),
            rotated,
        )
        return rotated.to(value.dtype)

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Full-precision prefill over the current batch's K/V.

        Nothing is read back from the quantised cache here, so this is exact up
        to the rotation of V, which a folded o_proj undoes.
        """
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        prefill_value = value[:num_tokens]
        if self.output_rotation_folded:
            prefill_value = self._rotate_value_for_prefill(prefill_value)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:num_tokens],
            key=key[:num_tokens],
            value=prefill_value,
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
        impl.scale_cache = None
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        vllm_config = getattr(impl, "vllm_config", None)
        hf_config = None if vllm_config is None else vllm_config.model_config.hf_config
        layer_name = getattr(layer, "layer_name", "")
        impl.output_rotation_folded = output_rotation_is_folded(hf_config, layer_name, impl.head_size)
        logger.debug(
            "[vllm-ascend/turboquant] %s: o_proj %s",
            layer_name,
            "is Pi-folded; the decode output stays rotated"
            if impl.output_rotation_folded
            else "is not folded; the decode un-rotates its output on the device",
        )
