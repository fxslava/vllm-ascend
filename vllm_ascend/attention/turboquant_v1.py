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

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
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

# fp32 lanes in one 32-byte burst.
TURBOQUANT_BURST_FLOATS = 8

# Rows of a paged block the decode kernel handles per tile.  Must match
# kTileRows in csrc/attention/turboquant/turboquant_kernels.cpp: the codec's
# shuffle tables are sized for that batch, and the kernel asserts the length.
TURBOQUANT_TILE_ROWS = 16

# Reconstruction levels of the codec, and so the length of the centroid table
# the constant-table image carries.  Mirrors TurboQuantCodec<4>::kLevels.
TURBOQUANT_LEVELS = 16

# The 16-level Lloyd-Max quantiser for N(0, 1): the fixed point of
#
#     t_i = (c_{i-1} + c_i) / 2,     c_i = E[X | t_i < X < t_{i+1}],
#
# solved with the closed-form truncated-Gaussian moments and symmetrised
# exactly.  Its distortion E[(X - Q(X))^2] is 0.0095010080, i.e. 20.222 dB,
# against 0.01388 for the uniform mid-rise grid it replaced.
#
# The rotation is what earns the table: Pi drives the coordinates of a KV vector
# to very nearly i.i.d. N(0, 1) -- measured |excess kurtosis| < 0.12 at every
# full-attention layer of Qwen3.5-2B -- which is precisely the distribution
# these levels are optimal for.  The matching scale is therefore the RMS
# ||v||_2 / sqrt(d) rather than an absmax, which sizes its step off a single
# outlier and leaves the bulk of the coordinates crushed.
#
# Byte-identical to TurboQuantCodec<4>::Threshold() and kLloydMaxCentroids in
# csrc/tests/reference/turbo_quant_cpu.h.  Re-derivable with
# scripts/tq_kv_quant_reference.py::lloyd_max_gaussian_table().
TURBOQUANT_LLOYD_MAX_CENTROIDS = (
    -2.7325895709951710, -2.0690172265313920, -1.6180463860218863, -1.2562311973471796,
    -0.9423404564869651, -0.6567591185324659, -0.3880482994902919, -0.1283950298511473,
    0.1283950298511473, 0.3880482994902919, 0.6567591185324659, 0.9423404564869651,
    1.2562311973471796, 1.6180463860218863, 2.0690172265313920, 2.7325895709951710,
)

# The 15 interior decision boundaries, t_i = (c_{i-1} + c_i) / 2 exactly.  The
# device holds these as Adds immediates rather than in UB -- they are the same
# on every launch, so a UB copy would buy nothing and cost fifteen scalar loads.
TURBOQUANT_LLOYD_MAX_THRESHOLDS = (
    -2.4008033987632817, -1.8435318062766393, -1.4371387916845330, -1.0992858269170722,
    -0.7995497875097155, -0.5224037090113789, -0.2582216646707196, 0.0,
    0.2582216646707196, 0.5224037090113789, 0.7995497875097155, 1.0992858269170722,
    1.4371387916845330, 1.8435318062766393, 2.4008033987632817,
)

_PI_SIGN_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_CODEC_TABLE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}

# Distinct decode shapes whose workspace size is remembered before the memo is
# dropped; see AscendTurboQuantAttentionBackendImpl._decode_workspace.
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

    word = 4  # bytes per Gather offset unit
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
        self.scale_cache: torch.Tensor | None = None
        self._pi_signs: torch.Tensor | None = None
        # Scratch for the decode split/combine reduction, grown to a high-water
        # mark and then reused forever; see _decode_workspace.
        self.decode_workspace: torch.Tensor | None = None
        self._workspace_floats: dict[tuple[int, int], int] = {}

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # Nothing is folded into the weights: q_proj, k_proj, v_proj and o_proj
        # stay exactly as loaded, which is what keeps RoPE correct.
        super().process_weights_after_loading(act_dtype)
        # Prime the C++ device registry while we are still loading. The vector
        # core count is what sizes both kernels' grids, and the driver query
        # behind it is only ever paid on this first call -- here, rather than on
        # a decode step or, worse, inside a graph capture.
        torch.ops._C_ascend.npu_turboquant_vector_core_num()
        logger.info_once("[vllm-ascend/turboquant] 4-bit KV cache active; Pi is applied at runtime in the kernel")

    def pi_signs(self, device: torch.device) -> torch.Tensor:
        if self._pi_signs is None:
            self._pi_signs = turboquant_pi_signs(self.head_size, device)
        return self._pi_signs

    def codec_tables(self, device: torch.device, batch_rows: int) -> torch.Tensor:
        return turboquant_codec_tables(self.head_size, batch_rows, device)

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
            # KV-sharing layers consume another layer's cache; writing here
            # would overwrite shared slots.
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        self._ensure_scale_cache(kv_cache)
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
                # A memo, not a table. Captured runs reuse a handful of shapes
                # forever, but an uncaptured one drifts through new
                # max_blocks_per_seq as sequences grow, and a dict that only
                # ever grows is a slow leak. Dropping it costs one operator
                # dispatch per shape afterwards.
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
            # Replacing the buffer mid-capture would bake a pointer into the
            # graph that nothing owns by the time it replays.  What keeps this
            # from firing is that vLLM warms every capture size up with
            # capturing unset before it captures that size, so the buffer has
            # already grown by the time the capture runs.  Note it is the warmup
            # and not the capture order that saves us: the requirement is not
            # monotonic in num_tokens -- a small batch leaves cores idle, which
            # raises num_splits and can need *more* scratch than a large one --
            # so capturing widest-first would not on its own be enough.
            raise RuntimeError(
                "[vllm-ascend/turboquant] the decode workspace would have to grow to "
                f"{needed} float32 words during a graph capture "
                f"(num_tokens={num_tokens}, max_blocks_per_seq={max_blocks_per_seq}). "
                "Run this shape once outside capture so the buffer is sized first."
            )

        self.decode_workspace = torch.empty(needed, dtype=torch.float32, device=device)
        return self.decode_workspace

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
        # Two kernels behind one op: the split stage writes per-sequence-split
        # partials into the persistent workspace, the combine stage reduces
        # them. They are separate launches ordered by the stream because no
        # in-kernel barrier orders an arbitrary grid; see
        # npu_turboquant_paged_attention in the adapter header.
        block_tables = attn_metadata.block_tables.to(torch.int32)
        torch.ops._C_ascend.npu_turboquant_paged_attention(
            query[:num_tokens],
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            attn_metadata.seq_lens.to(torch.int32),
            self.pi_signs(query.device),
            self.codec_tables(query.device, TURBOQUANT_TILE_ROWS),
            self._decode_workspace(num_tokens, block_tables.shape[1], query.device),
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
        impl.scale_cache = None
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
