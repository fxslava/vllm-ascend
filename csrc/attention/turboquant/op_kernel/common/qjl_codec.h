/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The QJL "3 + 1" reading of the shipped 4-bit cache word: 1 sign bit and a 3-bit magnitude
// index, with no change to a single stored byte.
//
// WHAT THIS HEADER IS, AND IS NOT
//
// It is *not* a new quantizer. tests/research/QJL_VS_INT4_FINDINGS.md (2026-09-22) measured a
// sign + 3-bit-magnitude layout against the codecs this tree ships and found the reconstruction
// bit-identical (max abs diff 0.00e+00) in two of three variants: at 4 bits a sign bit plus a
// 3-bit magnitude spans the same symmetric 16-point grid a signed int4 word does. What this
// header adds is the *decoder identity* -- the exact, branch-free bit arithmetic that reads the
// already-stored word as (sign, magnitude) -- so a kernel that wants the sign separately (for a
// masked accumulation, a sign-only sketch, or a magnitude-only table lookup) can take it without
// re-deriving the packing and without a second on-disk format. Every function here is a
// relabelling of bits that are already in the cache.
//
// The one thing the layout does NOT give you is an unbiased inner-product estimator. Any
// deterministic quantizer Q has E[k_hat | k] = Q(k) != k, whatever its bit layout. Unbiasedness
// comes from *stochastic rounding* of the level index, which is a change to the encoder only and
// is orthogonal to this file; see kStochasticRoundingIsExact below and
// vllm_ascend/attention/qjl_reference.py for the reference implementation and its proof test.
//
// Dependency-free on purpose, like turboquant_layout.h: the Torch-free host tests and
// op_host/turboquant_tiling.h include it without a kernel toolchain.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_QJL_CODEC_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_QJL_CODEC_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {
namespace qjl {

// ------------------------------------------------------------------ shape --
// The physical layout is untouched: (2, num_blocks, block_size, num_kv_heads, head_size / 2)
// int8, two 4-bit codes per byte, plus the separate fp32 scale plane. These are the only
// numbers this file asserts about it.
constexpr uint32_t kSignBits = 1;
constexpr uint32_t kMagnitudeBits = 3;
constexpr uint32_t kCodeBits = kSignBits + kMagnitudeBits;
constexpr uint32_t kMagnitudeLevels = 1u << kMagnitudeBits;  // 8
constexpr uint32_t kCodeLevels = 1u << kCodeBits;            // 16
constexpr uint32_t kNibblesPerByte = 8u / kCodeBits;         // 2
constexpr uint8_t kMagnitudeMask = 0x7;
constexpr uint8_t kNibbleMask = 0xF;
constexpr uint32_t kSignShift = 3;

// ------------------------------------------------------ the two sign words --
// Both shipped writers store a bin index `b` in [0, 16), but they do not store the same bits,
// and a Phase-2 unpack that assumes one convention on the other plane decodes negative
// coordinates to the wrong magnitude without raising. The difference is a single XOR by 8:
//
//   plane                                   low nibble bits   high nibble bits   level of bin b
//   -------------------------------------   ---------------   ----------------   --------------
//   AIV  TurboQuantCodec<4>, int8 byte       b_even            b_odd ^ 8          Lloyd-Max C[b]
//        stored as `lo + 16 * hi - 128`
//   Cube TurboQuantModeCodec<KV4_FP8>,       b_even ^ 8        b_odd ^ 8          b - 7.5
//        int4b_t pairs (n = b - 8)
//
// The AIV byte is biased by -128, and -128 == +128 (mod 256), so the bias flips bit 7 -- the
// MSB of the *high* nibble only. The Cube plane casts through int4b_t, whose two's-complement
// nibble n = b - 8 has the same bit pattern as b ^ 8 in both halves.
//
// After the XOR the nibble is two's complement: bit 3 set means the reconstruction is negative.
// Without it, bit 3 set means positive. Those are the only two polarities in the tree.
enum class SignPolarity : uint32_t {
  kSetMeansNegative = 0,  // two's-complement nibble: Cube plane, and the AIV high nibble
  kSetMeansPositive = 1,  // raw bin index:           the AIV low nibble
};

// ------------------------------------------------------- branch-free unpack --
// The sign bit, as stored. No compare, no select: one shift and one mask.
constexpr uint8_t SignBit(uint8_t nibble) {
  return static_cast<uint8_t>((static_cast<uint32_t>(nibble) >> kSignShift) & 0x1u);
}

// True when this nibble, on this plane, reconstructs to a negative value.
constexpr bool IsNegative(uint8_t nibble, SignPolarity polarity) {
  return (polarity == SignPolarity::kSetMeansNegative) ? (SignBit(nibble) != 0u) : (SignBit(nibble) == 0u);
}

// The 3-bit magnitude index, branch-free. `b ^ 8` counts *outward* from the grid centre on one
// side of zero and *inward* on the other, so one half of the code space has to be complemented
// to read as a magnitude. Replicating the sign bit into a 0x0 / 0x7 mask and XOR-ing does that
// without a select:
//
//     mask = (0 - sign_bit) & 0x7         all zeros, or all ones
//     mag  = (nibble ^ mask) & 0x7        ones'-complement abs, 3 bits wide
//
// For kSetMeansPositive the mask is inverted, which is the same instruction sequence with the
// sign bit complemented first. On the AIV this is ShiftRight + And + Xor on a uint8 view of the
// plane -- but see kNativeUnpackIsCheaper before writing it.
constexpr uint8_t MagnitudeIndex(uint8_t nibble, SignPolarity polarity) {
  const uint32_t complement = (polarity == SignPolarity::kSetMeansNegative)
                                  ? static_cast<uint32_t>(SignBit(nibble))
                                  : (static_cast<uint32_t>(SignBit(nibble)) ^ 0x1u);
  // 0x00 when the magnitude already counts outward, 0x07 when it has to be complemented.
  const uint32_t mask = (0u - complement) & static_cast<uint32_t>(kMagnitudeMask);
  return static_cast<uint8_t>((static_cast<uint32_t>(nibble) ^ mask) & static_cast<uint32_t>(kMagnitudeMask));
}

// +1.0f or -1.0f, with no branch on device: 1 - 2 * sign_bit.
constexpr float SignValue(uint8_t nibble, SignPolarity polarity) {
  return IsNegative(nibble, polarity) ? -1.0f : 1.0f;
}

// The inverse: fold a sign and a magnitude index back into the stored nibble.
constexpr uint8_t PackNibble(bool negative, uint8_t magnitude, SignPolarity polarity) {
  const uint32_t signBit =
      (polarity == SignPolarity::kSetMeansNegative) ? (negative ? 1u : 0u) : (negative ? 0u : 1u);
  const uint32_t complement = (polarity == SignPolarity::kSetMeansNegative) ? signBit : (signBit ^ 0x1u);
  const uint32_t mask = (0u - complement) & static_cast<uint32_t>(kMagnitudeMask);
  const uint32_t body = (static_cast<uint32_t>(magnitude) & static_cast<uint32_t>(kMagnitudeMask)) ^ mask;
  return static_cast<uint8_t>(body | (signBit << kSignShift));
}

// -------------------------------------------------------- magnitude grids --
// Two 3-bit magnitude codebooks, one per shipped level table. Both are the positive half of a
// symmetric 16-level grid, which is why the split is lossless: the optimal scalar quantizer of a
// symmetric density is itself symmetric, so its levels already pair up as +-c_i.

// KV4_FP8 / the shipped uniform mid-rise grid: |level| = magnitude + 0.5, so no table is needed
// at all and the magnitude decode is a single Adds. The absence of an exact zero level is what
// makes the sign bit meaningful -- a magnitude grid *with* a zero level annihilates 16.8% of its
// own sign bits (QJL_VS_INT4_FINDINGS.md, finding 1).
constexpr float kMidRiseOffset = 0.5f;
constexpr float UniformMagnitude(uint8_t index) {
  return static_cast<float>(static_cast<uint32_t>(index) & static_cast<uint32_t>(kMagnitudeMask)) + kMidRiseOffset;
}

// TurboQuantCodec<4>, the AIV default: the positive half of the 16-level Lloyd-Max table for
// N(0, 1). Sliced from TURBOQUANT_LLOYD_MAX_CENTROIDS[8:], not re-derived.
constexpr float LloydMaxMagnitude(uint8_t index) {
  constexpr float kMagnitudes[kMagnitudeLevels] = {
      0.1283950298511473f, 0.3880482994902919f, 0.6567591185324659f, 0.9423404564869651f,
      1.2562311973471796f, 1.6180463860218863f, 2.0690172265313920f, 2.7325895709951710f};
  return kMagnitudes[static_cast<uint32_t>(index) & static_cast<uint32_t>(kMagnitudeMask)];
}

// The full signed level of a stored nibble, on each plane. These are the identities Phase 2 has
// to preserve; the static_asserts at the bottom of this file prove them over all 16 codes.
constexpr float DecodeCubeLevel(uint8_t nibble) {
  return SignValue(nibble, SignPolarity::kSetMeansNegative) *
         UniformMagnitude(MagnitudeIndex(nibble, SignPolarity::kSetMeansNegative));
}

constexpr float DecodeAivLowLevel(uint8_t nibble) {
  return SignValue(nibble, SignPolarity::kSetMeansPositive) *
         LloydMaxMagnitude(MagnitudeIndex(nibble, SignPolarity::kSetMeansPositive));
}

// ------------------------------------------------------------- byte views --
// The nibble fields of one packed byte, per writer. `AivHighNibble` takes the int8 the AIV plane
// stores; the Cube accessors take the raw int4b_t pair.
constexpr uint8_t CubeLowNibble(int8_t packed) {
  return static_cast<uint8_t>(static_cast<uint32_t>(static_cast<uint8_t>(packed)) & static_cast<uint32_t>(kNibbleMask));
}

constexpr uint8_t CubeHighNibble(int8_t packed) {
  return static_cast<uint8_t>((static_cast<uint32_t>(static_cast<uint8_t>(packed)) >> 4) &
                              static_cast<uint32_t>(kNibbleMask));
}

constexpr uint8_t AivLowNibble(int8_t packed) { return CubeLowNibble(packed); }

// The -128 bias lives entirely in bit 7, so the high nibble comes back by undoing that one bit.
constexpr uint8_t AivHighNibble(int8_t packed) {
  return static_cast<uint8_t>(static_cast<uint32_t>(CubeHighNibble(packed)) ^ 0x8u);
}

// --------------------------------------------- notes for the Phase 2 kernel --
// 1. kNativeUnpackIsCheaper. TurboQuantModeCodec<KV4_FP8>::UnpackAffine already widens the plane
//    with `Cast(half, srcPacked.ReinterpretCast<int4b_t>(), CAST_NONE)` followed by one
//    `DeInterleave` -- the AIV's native signed-nibble widen, two instructions for both planes.
//    The shift/mask/XOR sequence above is three vector instructions *per plane* on a uint8 view
//    and then still needs the widen. Use the bit arithmetic only where the sign is wanted as a
//    separate tensor (a mask, a sketch); never as a replacement for the native unpack.
// 2. The sign is already exact. On the mid-rise grid the stored sign bit equals sign(x) for
//    100.0000% of coordinates (QJL_VS_INT4_FINDINGS.md), because the grid has no zero level.
//    No kernel change is needed to "deliver" the 1 exact sign bit; it is already delivered.
// 3. Unbiasedness is an encoder property. The estimator E[<q, k_hat>] = <q, k> holds exactly
//    when the level index is drawn by stochastic rounding and nothing clips, which on this grid
//    means pairing it with an absmax/7.5 scale rounded *up* through fp16. It costs the encoder
//    one uniform random draw and one Add per coordinate, costs the decoder nothing, and does not
//    touch a single byte of the layout.
constexpr bool kNativeUnpackIsCheaper = true;
constexpr bool kStochasticRoundingIsExact = true;

// ------------------------------------------------------- compile-time proof --
// The whole point of this header is that it decodes the bytes already in the cache. That is
// checked here over the entire 16-code space rather than trusted, so a Phase-2 edit to the bit
// arithmetic cannot compile.
namespace detail {

// The Cube writer stores n = b - 8 as a two's-complement nibble, i.e. the bits of b ^ 8.
constexpr uint8_t CubeNibbleOfBin(uint32_t bin) {
  return static_cast<uint8_t>((bin ^ 0x8u) & static_cast<uint32_t>(kNibbleMask));
}

// TurboQuantModeTraits<KV4_FP8>::kCentroids[b].
constexpr float CubeCentroid(uint32_t bin) { return static_cast<float>(bin) - 7.5f; }

// TURBOQUANT_LLOYD_MAX_CENTROIDS[b], rebuilt from the magnitude half so the two cannot drift.
constexpr float AivCentroid(uint32_t bin) {
  const uint32_t magnitude = (bin >= kMagnitudeLevels) ? (bin - kMagnitudeLevels) : ((kMagnitudeLevels - 1u) - bin);
  return ((bin >= kMagnitudeLevels) ? 1.0f : -1.0f) * LloydMaxMagnitude(static_cast<uint8_t>(magnitude));
}

constexpr bool CubeDecodeMatchesCentroids() {
  for (uint32_t bin = 0; bin < kCodeLevels; ++bin) {
    if (DecodeCubeLevel(CubeNibbleOfBin(bin)) != CubeCentroid(bin)) {
      return false;
    }
  }
  return true;
}

constexpr bool AivDecodeMatchesCentroids() {
  for (uint32_t bin = 0; bin < kCodeLevels; ++bin) {
    if (DecodeAivLowLevel(static_cast<uint8_t>(bin)) != AivCentroid(bin)) {
      return false;
    }
  }
  return true;
}

// (sign, magnitude) -> nibble -> (sign, magnitude) is the identity on every one of the 16 words,
// on both polarities. If this fails, a decode is silently reading another grid.
constexpr bool RoundTripsEveryCode(SignPolarity polarity) {
  for (uint32_t code = 0; code < kCodeLevels; ++code) {
    const uint8_t nibble = static_cast<uint8_t>(code);
    const uint8_t magnitude = MagnitudeIndex(nibble, polarity);
    if (static_cast<uint32_t>(magnitude) >= kMagnitudeLevels) {
      return false;
    }
    if (PackNibble(IsNegative(nibble, polarity), magnitude, polarity) != nibble) {
      return false;
    }
  }
  return true;
}

// The AIV byte's -128 bias touches bit 7 and nothing else.
constexpr bool AivByteSplitsBack() {
  for (uint32_t low = 0; low < kCodeLevels; ++low) {
    for (uint32_t high = 0; high < kCodeLevels; ++high) {
      const int32_t stored = static_cast<int32_t>(low) + 16 * static_cast<int32_t>(high) - 128;
      const int8_t packed = static_cast<int8_t>(stored);
      if (static_cast<uint32_t>(AivLowNibble(packed)) != low ||
          static_cast<uint32_t>(AivHighNibble(packed)) != high) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail

static_assert(detail::CubeDecodeMatchesCentroids(),
              "the 3+1 read of the KV4_FP8 word disagrees with TurboQuantModeTraits<KV4_FP8>::kCentroids");
static_assert(detail::AivDecodeMatchesCentroids(),
              "the 3+1 read of the AIV word disagrees with TURBOQUANT_LLOYD_MAX_CENTROIDS");
static_assert(detail::RoundTripsEveryCode(SignPolarity::kSetMeansNegative), "two's-complement nibble round-trip");
static_assert(detail::RoundTripsEveryCode(SignPolarity::kSetMeansPositive), "raw-bin nibble round-trip");
static_assert(detail::AivByteSplitsBack(), "the -128 byte bias is not confined to bit 7");
static_assert(kCodeLevels == 16u && kMagnitudeLevels == 8u && kNibblesPerByte == 2u, "the 4-bit footprint moved");

}  // namespace qjl
}  // namespace turboquant
}  // namespace vllm_ascend

#endif
