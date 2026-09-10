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

/*
 * TurboQuantMode: the three storage rates of the Cube-native KV cache, and the
 * layout, codebook and Cube operand type each of them implies.
 *
 *   mode     bits  levels  bytes/vec(d=256)  operand grid  Cube instruction
 *   kv3fp4     3       8          96          fp4 e2m1      mad_mx
 *   kv4fp8     4      16         128          fp8 e4m3fn    mad
 *   kv5fp8     5      32         160          fp8 e4m3fn    mad
 *
 * All three share one pipeline: the AIV rotates with Pi = D H D, bins the
 * coordinates against a Lloyd-Max table for N(0,1) at kBits, packs, and later
 * expands through a LUT straight onto the Cube's operand grid; the Cube runs
 * two GEMMs with fp32 accumulate; the AIV un-rotates the accumulator.
 *
 * On arch35 MmadCal dispatches to mad_mx for
 * <float, fp4x2_e2m1_t, fp4x2_e2m1_t> and to mad for
 * <float, fp8_e4m3fn_t, fp8_e4m3fn_t>.  Mixed operand types are rejected by a
 * static_assert, so kv3fp4 must put the query and the softmax row on the fp4
 * grid as well; kv3fp4 is scaffolded, not recommended.
 *
 * The stored table is cast(g * c) for a per-mode gain g, and the decoder
 * reconstructs (s / g) * lut[q], folding 1/g into the per-vector scale it
 * already multiplies by.  Without the gain a Lloyd-Max codebook's centroids are
 * not grid points of e2m1 or e4m3fn.
 *
 * The tables below are generated:
 *
 *     python scripts/tq_multimode_calibration.py --emit-header
 *
 * This header is deliberately free of AscendC types in its layout half, so the
 * host side can include it and size buffers from the same arithmetic the device
 * indexes with.
 */

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_MODE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_MODE_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

// The storage rate.  The name is <storage rate><Cube operand type>: kv5fp8
// stores 5 bits and computes in fp8.  The values cross the launch boundary, so
// do not renumber them.
enum class TurboQuantMode : int32_t {
    // 3-bit storage, 8 centroids, expanded to fp4 e2m1 for Cube mad_mx.
    KV3_FP4 = 3,
    // 4-bit storage, 16 centroids, expanded to fp8 e4m3fn for Cube mad.
    KV4_FP8 = 4,
    // 5-bit storage, 32 centroids, expanded to fp8 e4m3fn for Cube mad.
    // The primary target: the cheapest rate that clears cos > 0.995.
    KV5_FP8 = 5,
};

// The Cube operand grid a mode expands its codes onto.  Distinct from the mode
// because two modes share fp8_e4m3fn, and because the dispatch in
// turboquant_cube_mm.h keys off the operand type rather than off the rate.
enum class TurboQuantOperand : int32_t {
    kFp4E2m1 = 0,   // Cube mad_mx
    kFp8E4m3fn = 1, // Cube mad
};

/*
 * Everything about a mode that is a pure function of the mode itself.
 *
 * Sizes are per vector *slot*: one KV vector of `head_size` coordinates.  The
 * scale plane is shared across modes and indexed by token, so it is not counted
 * here; turboquant_torch_adpt.h owns its geometry.
 *
 * kElemsPerGroup / kBytesPerGroup is the packing unit, chosen so the group
 * boundary is also a byte boundary:
 *
 *   3-bit  8 codes -> 3 bytes   (three bit-planes; 24 bits, no remainder)
 *   4-bit  2 codes -> 1 byte    (two nibbles; the shipping layout)
 *   5-bit  8 codes -> 5 bytes   (five bit-planes; 40 bits, no remainder)
 */
struct TurboQuantModeConfig {
    TurboQuantMode mode;
    TurboQuantOperand operand;
    // Bits per stored coordinate.
    int32_t bits;
    // Reconstruction levels; always 1 << bits.
    int32_t levels;
    // The packing unit.
    int32_t elems_per_group;
    int32_t bytes_per_group;
    // cast(g * c) is the stored codebook and (s / g) the effective scale.
    float codebook_gain;
    // What the codebook delivers on N(0,1) after the cast, E[(X - Q(X))^2].
    float distortion;

    // Packed bytes for one vector of `head_size` coordinates.
    constexpr int64_t PackedBytes(int64_t head_size) const
    {
        return head_size * bytes_per_group / elems_per_group;
    }

    // Whether a vector slot is a whole number of 32-byte bursts, which is what
    // keeps every DMA on the path a DataCopy rather than a DataCopyPad.
    constexpr bool IsBurstAligned(int64_t head_size) const
    {
        return PackedBytes(head_size) % kBurstBytes == 0;
    }

    // Whether a slot is also a whole number of 64-byte bursts.  Not required --
    // 96 and 160 are not -- because the cache is indexed by token and a token
    // carries num_kv_heads slots contiguously.
    constexpr bool IsWideBurstAligned(int64_t head_size) const
    {
        return PackedBytes(head_size) % kWideBurstBytes == 0;
    }

    // The whole of one token's packed bytes across every kv head, which is the
    // unit the scatter writes and the tile loader reads.
    constexpr int64_t PackedPlaneBytes(int64_t head_size, int64_t num_kv_heads) const
    {
        return PackedBytes(head_size) * num_kv_heads;
    }

    // 32-byte DMA burst; the alignment every DataCopy on this path needs.
    static constexpr int64_t kBurstBytes = 32;
    // 64-byte burst; the width the Cube's L1 fill prefers.
    static constexpr int64_t kWideBurstBytes = 64;
};

/*
 * Compile-time traits.  The tables are `static constexpr` because the encoder
 * consumes the thresholds as Adds immediates; only the centroids reach UB.
 */
template <TurboQuantMode MODE>
struct TurboQuantModeTraits;

// --- kv3fp4 -------------------------------------------------------------
template <>
struct TurboQuantModeTraits<TurboQuantMode::KV3_FP4> {
    static constexpr TurboQuantMode kMode = TurboQuantMode::KV3_FP4;
    static constexpr TurboQuantOperand kOperand = TurboQuantOperand::kFp4E2m1;
    static constexpr int32_t kBits = 3;
    static constexpr int32_t kLevels = 8;
    static constexpr int32_t kElemsPerGroup = 8;
    static constexpr int32_t kBytesPerGroup = 3;
    static constexpr float kGain = 2.7881744355f;
    static constexpr float kDistortion = 0.0384442586f;

    // Every entry is an exact fp4 e2m1 grid point: {0, .5, 1, 1.5, 2, 3, 4, 6}
    // and negatives.  The gain search collapsed the innermost Lloyd-Max pair
    // onto +-0.5 rather than onto zero, which is the whole point of the gain.
    static constexpr float kCentroids[kLevels] = {
        -6.0000000000f, -4.0000000000f, -2.0000000000f, -0.5000000000f,
        +0.5000000000f, +2.0000000000f, +4.0000000000f, +6.0000000000f,
    };
    // Lloyd-Max boundaries of the *fp32* codebook, not re-derived after the
    // cast: the encoder and the host reference have to agree on the bin, and
    // the cast perturbs the reconstruction levels only.
    static constexpr float kThresholds[kLevels - 1] = {
        -1.7479274915f, -1.0499572799f, -0.5005497301f, +0.0000000000f,
        +0.5005497301f, +1.0499572799f, +1.7479274915f,
    };
};

// --- kv4fp8 -------------------------------------------------------------
template <>
struct TurboQuantModeTraits<TurboQuantMode::KV4_FP8> {
    static constexpr TurboQuantMode kMode = TurboQuantMode::KV4_FP8;
    static constexpr TurboQuantOperand kOperand = TurboQuantOperand::kFp8E4m3fn;
    static constexpr int32_t kBits = 4;
    static constexpr int32_t kLevels = 16;
    static constexpr int32_t kElemsPerGroup = 2;
    static constexpr int32_t kBytesPerGroup = 1;
    static constexpr float kGain = 2.1609589041f;
    static constexpr float kDistortion = 0.0097203519f;

    static constexpr float kCentroids[kLevels] = {
        -6.0000000000f, -4.5000000000f, -3.5000000000f, -2.7500000000f,
        -2.0000000000f, -1.3750000000f, -0.8125000000f, -0.2812500000f,
        +0.2812500000f, +0.8125000000f, +1.3750000000f, +2.0000000000f,
        +2.7500000000f, +3.5000000000f, +4.5000000000f, +6.0000000000f,
    };
    // Identical to TurboQuantCodec<4>::Threshold, which is the contract that
    // lets a cache written by the shipping 4-bit encoder be read by this path.
    static constexpr float kThresholds[kLevels - 1] = {
        -2.4008033988f, -1.8435318063f, -1.4371387917f, -1.0992858269f,
        -0.7995497875f, -0.5224037090f, -0.2582216647f, +0.0000000000f,
        +0.2582216647f, +0.5224037090f, +0.7995497875f, +1.0992858269f,
        +1.4371387917f, +1.8435318063f, +2.4008033988f,
    };
};

// --- kv5fp8 (primary target) --------------------------------------------
template <>
struct TurboQuantModeTraits<TurboQuantMode::KV5_FP8> {
    static constexpr TurboQuantMode kMode = TurboQuantMode::KV5_FP8;
    static constexpr TurboQuantOperand kOperand = TurboQuantOperand::kFp8E4m3fn;
    static constexpr int32_t kBits = 5;
    static constexpr int32_t kLevels = 32;
    static constexpr int32_t kElemsPerGroup = 8;
    static constexpr int32_t kBytesPerGroup = 5;
    static constexpr float kGain = 2.2064579256f;
    static constexpr float kDistortion = 0.0028686787f;

    // Every entry is an exact fp8_e4m3fn value.  The cast costs 0.589 dB of
    // codebook SNR here against 0.099 dB at 16 levels -- a 32-level codebook
    // asks more of the grid than e4m3fn's 3 mantissa bits give near +-0.14 --
    // and the rate still wins by 5.3 dB overall.
    static constexpr float kCentroids[kLevels] = {
        -7.0000000000f, -6.0000000000f, -5.0000000000f, -4.5000000000f,
        -4.0000000000f, -3.5000000000f, -3.0000000000f, -2.7500000000f,
        -2.2500000000f, -2.0000000000f, -1.6250000000f, -1.3750000000f,
        -1.0000000000f, -0.7500000000f, -0.4375000000f, -0.1406250000f,
        +0.1406250000f, +0.4375000000f, +0.7500000000f, +1.0000000000f,
        +1.3750000000f, +1.6250000000f, +2.0000000000f, +2.2500000000f,
        +2.7500000000f, +3.0000000000f, +3.5000000000f, +4.0000000000f,
        +4.5000000000f, +5.0000000000f, +6.0000000000f, +7.0000000000f,
    };
    static constexpr float kThresholds[kLevels - 1] = {
        -2.9759260354f, -2.5044294908f, -2.1732339018f, -1.9079808085f,
        -1.6817306482f, -1.4812842091f, -1.2990723601f, -1.1302938503f,
        -0.9716742187f, -0.8208504105f, -0.6760346638f, -0.5358165735f,
        -0.3990389144f, -0.2647150677f, -0.1319707447f, +0.0000000000f,
        +0.1319707447f, +0.2647150677f, +0.3990389144f, +0.5358165735f,
        +0.6760346638f, +0.8208504105f, +0.9716742187f, +1.1302938503f,
        +1.2990723601f, +1.4812842091f, +1.6817306482f, +1.9079808085f,
        +2.1732339018f, +2.5044294908f, +2.9759260354f,
    };
};

/*
 * The runtime view of a mode, for the host and for the launch boundary.
 *
 * Returns kv5fp8's configuration for any unrecognised value: this sizes buffers
 * on the launch path and the enum crosses an ABI, so a device-side assert would
 * abort a decode step rather than produce a diagnosable answer.  The host
 * validates with TurboQuantModeIsValid before it gets here.
 */
constexpr TurboQuantModeConfig TurboQuantModeConfigOf(TurboQuantMode mode)
{
    return mode == TurboQuantMode::KV3_FP4
               ? TurboQuantModeConfig{TurboQuantMode::KV3_FP4, TurboQuantOperand::kFp4E2m1, 3, 8, 8, 3,
                                      2.7881744355f, 0.0384442586f}
           : mode == TurboQuantMode::KV4_FP8
               ? TurboQuantModeConfig{TurboQuantMode::KV4_FP8, TurboQuantOperand::kFp8E4m3fn, 4, 16, 2, 1,
                                      2.1609589041f, 0.0097203519f}
               : TurboQuantModeConfig{TurboQuantMode::KV5_FP8, TurboQuantOperand::kFp8E4m3fn, 5, 32, 8, 5,
                                      2.2064579256f, 0.0028686787f};
}

constexpr bool TurboQuantModeIsValid(int32_t raw)
{
    return raw == static_cast<int32_t>(TurboQuantMode::KV3_FP4) ||
           raw == static_cast<int32_t>(TurboQuantMode::KV4_FP8) ||
           raw == static_cast<int32_t>(TurboQuantMode::KV5_FP8);
}

inline const char *TurboQuantModeName(TurboQuantMode mode)
{
    switch (mode) {
        case TurboQuantMode::KV3_FP4:
            return "kv3fp4";
        case TurboQuantMode::KV4_FP8:
            return "kv4fp8";
        case TurboQuantMode::KV5_FP8:
            return "kv5fp8";
    }
    return "kv5fp8";
}

// Bytes one Cube operand element occupies, which is what the unpack writes and
// the L1 stage moves.  fp4 is half a byte, so the fp4 count is in *pairs*:
// a d=256 fp4 row is 128 bytes of fp4x2, and the Cube's operand tensor is
// LocalTensor<fp4x2_e2m1_t> of length 128.
constexpr int64_t TurboQuantOperandBytes(TurboQuantOperand operand, int64_t elems)
{
    return operand == TurboQuantOperand::kFp4E2m1 ? elems / 2 : elems;
}

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_MODE_H
