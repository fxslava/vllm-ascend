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
"""What the TurboQuant operators expect in memory: cache geometry and constant tables.

Everything here is a pure function of the layer's shape, so it belongs to the
cache layout rather than to the attention backend that happens to drive it.
:mod:`vllm_ascend.attention.turboquant_v1` re-exports the lot, and the module
carries **no vLLM import** so that a caller outside the plugin --
``tools/tq_longbench``, which drives the same operators with no engine around
them -- can load it by file path and be certain it builds byte-identical
tables. A second, drifting copy of :func:`turboquant_codec_tables` would not
raise; it would decode the cache against the wrong centroids and quietly return
plausible numbers.

Like :mod:`vllm_ascend.attention.turboquant_rotation`, this module imports only
``torch``. Keep it that way.
"""

import enum

import torch

TURBOQUANT_PACK_FACTOR = 2

TURBOQUANT_BURST_FLOATS = 8

TURBOQUANT_TILE_ROWS = 16

TURBOQUANT_LEVELS = 16

TURBOQUANT_LLOYD_MAX_CENTROIDS = (
    -2.7325895709951710,
    -2.0690172265313920,
    -1.6180463860218863,
    -1.2562311973471796,
    -0.9423404564869651,
    -0.6567591185324659,
    -0.3880482994902919,
    -0.1283950298511473,
    0.1283950298511473,
    0.3880482994902919,
    0.6567591185324659,
    0.9423404564869651,
    1.2562311973471796,
    1.6180463860218863,
    2.0690172265313920,
    2.7325895709951710,
)

TURBOQUANT_LLOYD_MAX_THRESHOLDS = (
    -2.4008033987632817,
    -1.8435318062766393,
    -1.4371387916845330,
    -1.0992858269170722,
    -0.7995497875097155,
    -0.5224037090113789,
    -0.2582216646707196,
    0.0,
    0.2582216646707196,
    0.5224037090113789,
    0.7995497875097155,
    1.0992858269170722,
    1.4371387916845330,
    1.8435318062766393,
    2.4008033987632817,
)

# The writer stores each nibble pair as ``low + 16 * high - 128``; see turboquant_dequantize.
_INT8_CODE_BIAS = 128

_CODEC_TABLE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_CENTROID_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_HADAMARD16_CACHE: dict[str, torch.Tensor] = {}

TURBOQUANT_ROTATE_TILE = 16

# The Cube decode reads a block one 64-row tile at a time.
TURBOQUANT_CUBE_TILE_ROWS = 64

# The decodes' optional softmax-statistics out-tensor, mirroring ``kLseStride``,
# ``kPartialMaxLane`` and ``kPartialSumLane`` in
# ``csrc/attention/turboquant/op_kernel/common/turboquant_layout.h``. One whole
# 64-byte group per ``(token, head)``, of which two lanes carry anything: the
# running max over the whole context and the mass ``sum_i exp(s_i - max)``.
#
# The padding is the point. Every global-memory copy the kernels make moves whole
# 32-byte bursts, and one float per head is not a burst; at this stride the offset
# of any ``(token, head)`` is a whole 64 bytes whatever ``num_heads`` is, so the
# write needs no unaligned-copy path and cannot start mid-burst.
TURBOQUANT_LSE_STRIDE = 16
TURBOQUANT_LSE_MAX_LANE = 0
TURBOQUANT_LSE_SUM_LANE = 8


def turboquant_lse_buffer(num_tokens: int, num_heads: int, device: torch.device) -> torch.Tensor:
    """Allocate the ``[num_tokens, num_heads, TURBOQUANT_LSE_STRIDE]`` fp32 out-tensor.

    Zeroed rather than empty: a decode writes every ``(token, head)`` it is given, but
    a caller that hands one operator a buffer sized for a larger capture shape leaves
    the tail unwritten, and a zeroed tail reads as ``max = 0, mass = 0`` -- the same
    thing an empty context writes, which every reader of this already has to handle.
    Uninitialised device memory would instead read as a plausible mass.
    """
    return torch.zeros((num_tokens, num_heads, TURBOQUANT_LSE_STRIDE), dtype=torch.float32, device=device)


def turboquant_split_lse(lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(running_max, mass)`` pair a decode wrote, each ``[num_tokens, num_heads]``.

    Views, not copies. ``mass == 0`` is how a token whose context was empty is spelled,
    and the accompanying max is then whatever the kernel initialised its accumulator to,
    so a reader must branch on the mass and never on the max.
    """
    if lse.shape[-1] != TURBOQUANT_LSE_STRIDE:
        raise ValueError(
            f"[vllm-ascend/turboquant] the softmax-statistics tensor is {tuple(lse.shape)}; its last "
            f"dimension must be TURBOQUANT_LSE_STRIDE = {TURBOQUANT_LSE_STRIDE}"
        )
    return lse[..., TURBOQUANT_LSE_MAX_LANE], lse[..., TURBOQUANT_LSE_SUM_LANE]


class TurboQuantOutputStage(enum.IntEnum):
    """What the Cube decode writes; mirrors ``TurboQuantOutputStage`` in turboquant_layout.h."""

    # softmax(q k^T) Pi v, for an o_proj with Pi folded in.
    ROTATED_BASIS = 0
    # Pi applied once more to the fp32 accumulator before the fp16 cast.
    UNROTATED = 1
    # UNROTATED, then multiplied by sigmoid(gate): an attn_output_gate layer.
    GATED = 2


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
    ``csrc/attention/turboquant/op_host/turboquant_tiling.cpp``.
    """
    key = str(device)
    cached = _HADAMARD16_CACHE.get(key)
    if cached is not None:
        return cached
    rows = []
    for i in range(TURBOQUANT_ROTATE_TILE):
        rows.append([-1.0 if bin(i & j).count("1") & 1 else 1.0 for j in range(TURBOQUANT_ROTATE_TILE)])
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


def turboquant_dequantize(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Expand packed 4-bit codes back to their Lloyd-Max levels, low nibble first.

    ``packed`` is ``[..., head_size // 2]`` int8 straight out of a cache plane, and the
    result is ``[..., head_size]`` bare reconstruction levels -- the per-vector scale in
    the scale plane still has to be applied on top.  Mirrors ``Dequantize4Bit`` in
    ``csrc/attention/turboquant/op_kernel/common/turboquant_codec_950.h``: the writer
    stores ``low + 16 * high - 128``, so the bias comes off before the nibbles split.

    This is the host's only reader of the packed format, and it lives here rather than
    beside its caller for the reason the codec tables do: a second copy of the grid would
    not raise, it would decode the cache against the wrong levels and return plausible
    numbers.
    """
    if packed.dtype != torch.int8:
        raise ValueError(f"the packed 4-bit cache is int8, got {packed.dtype}")
    byte = packed.to(torch.int32) + _INT8_CODE_BIAS
    codes = torch.stack((byte % TURBOQUANT_LEVELS, byte // TURBOQUANT_LEVELS), dim=-1).flatten(-2)
    centroids = _centroid_table(packed.device, dtype)
    return centroids[codes.to(torch.long)]


def _centroid_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(dtype))
    cached = _CENTROID_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=dtype, device=device)
        _CENTROID_CACHE[key] = cached
    return cached
