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
"""Host reference for the QJL "3 + 1" reading of the packed 4-bit KV cache.

The twin of ``csrc/attention/turboquant/op_kernel/common/qjl_codec.h``: same bit
arithmetic, same two sign polarities, same magnitude grids, in torch so the layout and
the estimator can be tested without a device.

Two things live here, and they are independent of each other:

**The layout.** A 4-bit code word read as 1 sign bit plus a 3-bit magnitude index. This
is a *relabelling* of bits the cache already holds, not a new quantizer.
``tests/research/QJL_VS_INT4_FINDINGS.md`` (2026-09-22) measured the sign-magnitude
layout against the shipped codecs and found the reconstruction bit-identical -- at 4 bits
a sign bit plus a 3-bit magnitude spans the same symmetric 16-point grid a signed int4
word does. Nothing in this module changes a stored byte: the packed plane stays
``(num_blocks, block_size, num_kv_heads, head_size // 2)`` int8, two codes per byte.

**The estimator.** The unbiasedness the QJL construction is actually about,
``E[<q, k_hat>] = <q, k>``, is a property of the *encoder*, not of the packing. No
deterministic quantizer can have it: ``E[k_hat | k] = Q(k) != k`` for every bit layout.
:func:`encode_stochastic` gets it by drawing the level index with stochastic rounding, and
on an absmax scale rounded up through fp16 nothing clips, so the estimator is unbiased
*exactly* rather than asymptotically -- see :func:`expected_reconstruction`, which
evaluates ``E[k_hat | k]`` in closed form and returns ``k``.

Stochastic rounding is not free: it trades bias for variance (per-coordinate error power
rises from about ``step^2 / 12`` to ``step^2 / 6``). Whether that trade is worth making on
attention output is a Phase-2 measurement, not a claim this module makes.
"""

import enum
import math
from typing import Optional

import torch

# ---------------------------------------------------------------- the word --
# The 4-bit footprint the cache allocator depends on. These are shared with
# ``qjl_codec.h``; changing one without the other decodes the cache against the wrong grid.
QJL_SIGN_BITS = 1
QJL_MAGNITUDE_BITS = 3
QJL_CODE_BITS = QJL_SIGN_BITS + QJL_MAGNITUDE_BITS
QJL_MAGNITUDE_LEVELS = 1 << QJL_MAGNITUDE_BITS
QJL_CODE_LEVELS = 1 << QJL_CODE_BITS
QJL_NIBBLES_PER_BYTE = 8 // QJL_CODE_BITS

QJL_MAGNITUDE_MASK = QJL_MAGNITUDE_LEVELS - 1
QJL_NIBBLE_MASK = QJL_CODE_LEVELS - 1
QJL_SIGN_SHIFT = QJL_MAGNITUDE_BITS

# The mid-rise grid KV4_FP8 ships: levels +-{0.5, 1.5, ..., 7.5}, bin b reconstructing to
# ``b - 7.5``. It has no exact zero, which is what makes the stored sign bit meaningful --
# a magnitude grid *with* a zero level annihilates 16.8% of its own sign bits.
QJL_MID_RISE_OFFSET = 0.5
QJL_AFFINE_BIAS = QJL_CODE_LEVELS / 2 - QJL_MID_RISE_OFFSET  # 7.5
QJL_MAX_ABS_LEVEL = QJL_AFFINE_BIAS

# The AIV default (``TurboQuantCodec<4>``): the positive half of the 16-level Lloyd-Max
# table for N(0, 1). Kept as the magnitude half so the symmetry is visible; the full table
# lives in ``turboquant_layout.TURBOQUANT_LLOYD_MAX_CENTROIDS``.
QJL_LLOYD_MAX_MAGNITUDES = (
    0.1283950298511473,
    0.3880482994902919,
    0.6567591185324659,
    0.9423404564869651,
    1.2562311973471796,
    1.6180463860218863,
    2.0690172265313920,
    2.7325895709951710,
)

# The int8 bias the AIV writer applies to each packed byte (``low + 16 * high - 128``).
QJL_INT8_BYTE_BIAS = 128

_EPS = 1e-20


class SignPolarity(enum.Enum):
    """Which way a stored nibble's bit 3 reads.

    Both shipped writers store a bin index in ``[0, 16)``, but they do not store the same
    bits. The Cube plane (``TurboQuantModeCodec<KV4_FP8>``) stages ``n = bin - 8`` through
    ``int4b_t``, so its nibble is two's complement and bit 3 set means *negative*. The AIV
    plane (``TurboQuantCodec<4>``) stores the raw bin in its low nibble, where bit 3 set
    means *positive*; its high nibble picks up the byte's ``-128`` bias, which flips bit 7
    and therefore reads like the Cube's.
    """

    SET_MEANS_NEGATIVE = 0
    SET_MEANS_POSITIVE = 1


class CachePlane(enum.Enum):
    """Which writer produced the packed plane being read."""

    CUBE_KV4_FP8 = 0
    AIV_LLOYD_MAX = 1


class ScaleMode(enum.Enum):
    """How the per-vector scale is chosen.

    ``ABSMAX`` sizes the step so the peak coordinate lands exactly on the outermost level,
    which means nothing clips and stochastic rounding is *exactly* unbiased. ``RMS`` is the
    shipped default; it is better on reconstruction SNR for a Gaussian, but its tail clips,
    and a clipped coordinate is the one place the unbiased estimator loses its guarantee.
    """

    ABSMAX = 0
    RMS = 1


# ----------------------------------------------------------- bit primitives --
def sign_bit(nibble: torch.Tensor) -> torch.Tensor:
    """Bit 3 of each stored nibble. One shift and one mask; no compare, no select."""
    return (nibble >> QJL_SIGN_SHIFT) & 1


def is_negative(nibble: torch.Tensor, polarity: SignPolarity) -> torch.Tensor:
    """Whether each nibble reconstructs to a negative value, on this plane's polarity."""
    bit = sign_bit(nibble)
    return bit.bool() if polarity is SignPolarity.SET_MEANS_NEGATIVE else ~bit.bool()


def magnitude_index(nibble: torch.Tensor, polarity: SignPolarity) -> torch.Tensor:
    """The 3-bit magnitude index, branch-free.

    Half of the code space counts outward from the grid centre and half counts inward, so
    one half has to be complemented to read as a magnitude. Replicating the sign bit into a
    ``0x0``/``0x7`` mask and XOR-ing does that without a select -- the ones'-complement
    absolute value, three bits wide::

        mask = (0 - sign_bit) & 0x7
        mag = (nibble ^ mask) & 0x7
    """
    bit = sign_bit(nibble)
    complement = bit if polarity is SignPolarity.SET_MEANS_NEGATIVE else bit ^ 1
    mask = (-complement) & QJL_MAGNITUDE_MASK
    return (nibble ^ mask) & QJL_MAGNITUDE_MASK


def join_sign_magnitude(negative: torch.Tensor, magnitude: torch.Tensor, polarity: SignPolarity) -> torch.Tensor:
    """The inverse of :func:`sign_bit` + :func:`magnitude_index`: fold them back to a nibble."""
    negative_int = negative.to(torch.int32)
    bit = negative_int if polarity is SignPolarity.SET_MEANS_NEGATIVE else negative_int ^ 1
    complement = bit if polarity is SignPolarity.SET_MEANS_NEGATIVE else bit ^ 1
    mask = (-complement) & QJL_MAGNITUDE_MASK
    body = (magnitude.to(torch.int32) & QJL_MAGNITUDE_MASK) ^ mask
    return body | (bit << QJL_SIGN_SHIFT)


def magnitude_table(plane: CachePlane, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The 8-entry magnitude codebook of a plane, indexed by :func:`magnitude_index`."""
    if plane is CachePlane.CUBE_KV4_FP8:
        values = [index + QJL_MID_RISE_OFFSET for index in range(QJL_MAGNITUDE_LEVELS)]
    else:
        values = list(QJL_LLOYD_MAX_MAGNITUDES)
    return torch.tensor(values, dtype=dtype, device=device)


def _plane_polarity(plane: CachePlane, head_size: int, device: torch.device) -> torch.Tensor:
    """Per-coordinate polarity for a plane, as a bool mask that is True for SET_MEANS_NEGATIVE.

    Constant for the Cube plane. For the AIV plane it alternates: the byte's ``-128`` bias
    lives in bit 7, so only the odd (high-nibble) coordinates are complemented.
    """
    if plane is CachePlane.CUBE_KV4_FP8:
        return torch.ones(head_size, dtype=torch.bool, device=device)
    coordinate = torch.arange(head_size, device=device)
    return (coordinate % QJL_NIBBLES_PER_BYTE) == 1


def stored_nibbles(bins: torch.Tensor, plane: CachePlane) -> torch.Tensor:
    """The nibble field each bin index actually occupies in the byte, per coordinate.

    Cube: ``bin ^ 8`` everywhere, because ``int4b_t`` two's complement of ``bin - 8`` has
    those bits. AIV: the raw bin in the even (low) slots and ``bin ^ 8`` in the odd (high)
    ones, because ``-128 == +128 (mod 256)`` and that carry flips bit 7 alone.
    """
    complemented = _plane_polarity(plane, bins.shape[-1], bins.device)
    return torch.where(complemented, bins ^ QJL_MAGNITUDE_LEVELS, bins) & QJL_NIBBLE_MASK


def bins_from_nibbles(nibbles: torch.Tensor, plane: CachePlane) -> torch.Tensor:
    """The inverse of :func:`stored_nibbles`; the XOR is its own inverse."""
    return stored_nibbles(nibbles, plane)


# ------------------------------------------------------------ plane packing --
def pack_plane(bins: torch.Tensor, plane: CachePlane) -> torch.Tensor:
    """Pack ``[..., head_size]`` bin indices into ``[..., head_size // 2]`` int8.

    Coordinate ``2 * i`` takes the low nibble and ``2 * i + 1`` the high one, matching the
    ``evenOffset_``/``oddOffset_`` gathers the shipped encoders issue. The returned buffer
    is bit-compatible with the existing pool allocation; no stride, no side-car and no shape
    changes.
    """
    if bins.shape[-1] % QJL_NIBBLES_PER_BYTE != 0:
        raise ValueError(f"head_size must be even to pack two codes per byte, got {bins.shape[-1]}")
    nibbles = stored_nibbles(bins.to(torch.int32), plane)
    low = nibbles[..., 0::QJL_NIBBLES_PER_BYTE]
    high = nibbles[..., 1::QJL_NIBBLES_PER_BYTE]
    byte = low | (high << QJL_CODE_BITS)
    # int8 wraps the top half of the byte to negative, which is exactly what the writer stores.
    return torch.where(byte >= QJL_INT8_BYTE_BIAS, byte - 2 * QJL_INT8_BYTE_BIAS, byte).to(torch.int8)


def unpack_plane(packed: torch.Tensor, plane: CachePlane) -> torch.Tensor:
    """Expand ``[..., head_size // 2]`` int8 back to ``[..., head_size]`` bin indices."""
    if packed.dtype != torch.int8:
        raise ValueError(f"the packed 4-bit cache is int8, got {packed.dtype}")
    byte = packed.to(torch.int32) & 0xFF
    low = byte & QJL_NIBBLE_MASK
    high = (byte >> QJL_CODE_BITS) & QJL_NIBBLE_MASK
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    return bins_from_nibbles(nibbles, plane)


def decode_levels(bins: torch.Tensor, plane: CachePlane, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Bin indices to bare reconstruction levels, through the sign/magnitude split.

    Deliberately routed through :func:`magnitude_index` rather than a 16-entry lookup: this
    is the identity Phase 2 has to preserve, so the reference exercises it.
    """
    nibbles = stored_nibbles(bins.to(torch.int32), plane)
    polarity_mask = _plane_polarity(plane, bins.shape[-1], bins.device)
    negative = torch.where(
        polarity_mask,
        is_negative(nibbles, SignPolarity.SET_MEANS_NEGATIVE),
        is_negative(nibbles, SignPolarity.SET_MEANS_POSITIVE),
    )
    magnitude = torch.where(
        polarity_mask,
        magnitude_index(nibbles, SignPolarity.SET_MEANS_NEGATIVE),
        magnitude_index(nibbles, SignPolarity.SET_MEANS_POSITIVE),
    )
    table = magnitude_table(plane, bins.device, dtype)
    levels = table[magnitude.to(torch.long)]
    return torch.where(negative, -levels, levels)


# ------------------------------------------------------------------ scales --
def _fp16_round_up(value: torch.Tensor) -> torch.Tensor:
    """Round a scale through float16, upward.

    The cache stores the scale in reduced precision, and round-to-nearest can land *below*
    ``absmax / 7.5``, which would push the peak coordinate past the outermost level and
    clip it. One clipped coordinate is enough to break exact unbiasedness, so the reference
    takes the next representable fp16 up whenever the rounding went down.
    """
    rounded = value.to(torch.float16).to(value.dtype)
    bumped = torch.nextafter(rounded.to(torch.float16), torch.tensor(math.inf, dtype=torch.float16)).to(value.dtype)
    return torch.where(rounded < value, bumped, rounded)


def compute_scale(vectors: torch.Tensor, mode: ScaleMode) -> torch.Tensor:
    """The per-vector scale, ``[..., 1]``, rounded up through fp16 as the cache stores it."""
    if mode is ScaleMode.ABSMAX:
        raw = vectors.abs().amax(dim=-1, keepdim=True) / QJL_MAX_ABS_LEVEL
    else:
        raw = vectors.norm(dim=-1, keepdim=True) / math.sqrt(vectors.shape[-1])
    return _fp16_round_up(raw + _EPS)


# ---------------------------------------------------------------- encoders --
def encode_deterministic(
    vectors: torch.Tensor, scale_mode: ScaleMode = ScaleMode.ABSMAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest on the mid-rise grid: the shipped uniform codec, relabelled.

    Returns ``(bins, scale)``. Its reconstruction is bit-identical to the shipped KV4_FP8
    decode -- that equality is the headline of ``QJL_VS_INT4_FINDINGS.md`` and is pinned as
    a test, not assumed.
    """
    scale = compute_scale(vectors, scale_mode)
    level = vectors / scale + QJL_AFFINE_BIAS
    bins = torch.clamp(torch.round(level), 0, QJL_CODE_LEVELS - 1).to(torch.int32)
    return bins, scale


def encode_stochastic(
    vectors: torch.Tensor,
    scale_mode: ScaleMode = ScaleMode.ABSMAX,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dithered rounding on the same grid: the unbiased encoder.

    With ``t = x / s + 7.5`` in ``[0, 15]``, take ``floor(t)`` and step up with probability
    ``t - floor(t)``. Then ``E[bin] = t`` exactly, so ``E[x_hat] = x`` exactly, so
    ``E[<q, k_hat>] = <q, k>`` exactly -- provided nothing clips, which is why the default
    scale is ``ABSMAX`` with an upward fp16 round.

    On device this is one uniform draw and one Add per coordinate in the encoder. The
    decoder is untouched, and so is every stored byte.
    """
    scale = compute_scale(vectors, scale_mode)
    level = vectors / scale + QJL_AFFINE_BIAS
    floor = torch.floor(level)
    noise = torch.rand(level.shape, generator=generator, device=level.device, dtype=level.dtype)
    bins = torch.clamp(floor + (noise < (level - floor)).to(level.dtype), 0, QJL_CODE_LEVELS - 1)
    return bins.to(torch.int32), scale


def expected_reconstruction(vectors: torch.Tensor, scale_mode: ScaleMode = ScaleMode.ABSMAX) -> torch.Tensor:
    """``E[x_hat | x]`` under :func:`encode_stochastic`, in closed form.

    ``E[bin] = floor(t) + (t - floor(t)) = t`` whenever ``t`` is in range, so this returns
    ``x`` exactly on an unclipped scale. Evaluating it directly turns "the estimator is
    unbiased" into a deterministic assertion rather than a Monte-Carlo threshold.
    """
    scale = compute_scale(vectors, scale_mode)
    level = vectors / scale + QJL_AFFINE_BIAS
    expected_bin = torch.clamp(level, 0, QJL_CODE_LEVELS - 1)
    return (expected_bin - QJL_AFFINE_BIAS) * scale


def encode_asymmetric_uniform(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard asymmetric (affine min/max) 4-bit quantization, the comparison baseline.

    ``s = (max - min) / 15``, ``z = round(-min / s)``, ``q = clamp(round(x / s) + z, 0, 15)``,
    reconstruction ``(q - z) * s``. Returns ``(codes, scale, zero_point)``.

    Two things to note before treating it as a like-for-like baseline. It needs a *second*
    per-vector side-car (the zero point) that the symmetric grid does not, so at equal code
    width it costs more memory, not the same. And ``z`` is rounded to an integer, which
    offsets every coordinate of the vector by up to half a step in the same direction -- a
    rank-1 error term that does not average out across the head and is the dominant bias
    this baseline carries into the dot product.
    """
    lo = vectors.amin(dim=-1, keepdim=True)
    hi = vectors.amax(dim=-1, keepdim=True)
    scale = _fp16_round_up((hi - lo) / (QJL_CODE_LEVELS - 1) + _EPS)
    zero_point = torch.clamp(torch.round(-lo / scale), 0, QJL_CODE_LEVELS - 1)
    codes = torch.clamp(torch.round(vectors / scale) + zero_point, 0, QJL_CODE_LEVELS - 1)
    return codes.to(torch.int32), scale, zero_point


def decode_asymmetric_uniform(codes: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Reconstruction for :func:`encode_asymmetric_uniform`."""
    return (codes.to(scale.dtype) - zero_point) * scale


# ------------------------------------------------------------ reconstruction --
def reconstruct(bins: torch.Tensor, scale: torch.Tensor, plane: CachePlane = CachePlane.CUBE_KV4_FP8) -> torch.Tensor:
    """Bin indices plus per-vector scale back to the approximated vectors."""
    return decode_levels(bins, plane, dtype=scale.dtype) * scale


def roundtrip_through_cache(
    vectors: torch.Tensor,
    encode_fn,
    plane: CachePlane = CachePlane.CUBE_KV4_FP8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode, pack to int8, read the packed bytes back, decode. Returns ``(recon, packed)``.

    The packed tensor is the byte-for-byte cache payload, so a caller can assert the
    footprint is unchanged rather than take it on trust.
    """
    bins, scale = encode_fn(vectors)
    packed = pack_plane(bins, plane)
    return reconstruct(unpack_plane(packed, plane), scale, plane), packed


def inner_product_bias(
    query: torch.Tensor,
    keys: torch.Tensor,
    recon: torch.Tensor,
) -> torch.Tensor:
    """``<q, k_hat> - <q, k>`` per key. Averaging this over trials estimates the bias."""
    return recon.matmul(query) - keys.matmul(query)
