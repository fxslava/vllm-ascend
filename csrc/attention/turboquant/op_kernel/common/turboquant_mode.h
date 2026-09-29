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

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_MODE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_MODE_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

enum class TurboQuantMode : int32_t {
    KV3_FP4 = 3,
    KV4_FP8 = 4,
    KV5_FP8 = 5,
};

enum class TurboQuantOperand : int32_t {
    kFp4E2m1 = 0,
    kFp8E4m3fn = 1,
};

struct TurboQuantModeConfig {
    TurboQuantMode mode;
    TurboQuantOperand operand;
    int32_t bits;
    int32_t levels;
    int32_t elems_per_group;
    int32_t bytes_per_group;
    float codebook_gain;
    float distortion;
    bool is_affine;
    float affine_bias;

    constexpr int64_t PackedBytes(int64_t head_size) const
    {
        return head_size * bytes_per_group / elems_per_group;
    }

    constexpr bool IsBurstAligned(int64_t head_size) const
    {
        return PackedBytes(head_size) % kBurstBytes == 0;
    }

    constexpr bool IsWideBurstAligned(int64_t head_size) const
    {
        return PackedBytes(head_size) % kWideBurstBytes == 0;
    }

    constexpr int64_t PackedPlaneBytes(int64_t head_size, int64_t num_kv_heads) const
    {
        return PackedBytes(head_size) * num_kv_heads;
    }

    static constexpr int64_t kBurstBytes = 32;
    static constexpr int64_t kWideBurstBytes = 64;
};

template <TurboQuantMode MODE>
struct TurboQuantModeTraits;

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
    static constexpr bool kIsAffine = false;
    static constexpr float kAffineBias = 0.0f;

    static constexpr float kCentroids[kLevels] = {
        -6.0000000000f, -4.0000000000f, -2.0000000000f, -0.5000000000f,
        +0.5000000000f, +2.0000000000f, +4.0000000000f, +6.0000000000f,
    };
    static constexpr float kThresholds[kLevels - 1] = {
        -1.7479274915f, -1.0499572799f, -0.5005497301f, +0.0000000000f,
        +0.5005497301f, +1.0499572799f, +1.7479274915f,
    };
};

template <>
struct TurboQuantModeTraits<TurboQuantMode::KV4_FP8> {
    static constexpr TurboQuantMode kMode = TurboQuantMode::KV4_FP8;
    static constexpr TurboQuantOperand kOperand = TurboQuantOperand::kFp8E4m3fn;
    static constexpr int32_t kBits = 4;
    static constexpr int32_t kLevels = 16;
    static constexpr int32_t kElemsPerGroup = 2;
    static constexpr int32_t kBytesPerGroup = 1;
    static constexpr float kGain = 2.9832882881f;
    static constexpr float kDistortion = 0.0115428844f;
    static constexpr bool kIsAffine = true;
    static constexpr float kAffineBias = 7.5f;

    static constexpr float kCentroids[kLevels] = {
        -7.5000000000f, -6.5000000000f, -5.5000000000f, -4.5000000000f,
        -3.5000000000f, -2.5000000000f, -1.5000000000f, -0.5000000000f,
        +0.5000000000f, +1.5000000000f, +2.5000000000f, +3.5000000000f,
        +4.5000000000f, +5.5000000000f, +6.5000000000f, +7.5000000000f,
    };
    static constexpr float kThresholds[kLevels - 1] = {
        -2.3464041433f, -2.0112035514f, -1.6760029595f, -1.3408023676f,
        -1.0056017757f, -0.6704011838f, -0.3352005919f, +0.0000000000f,
        +0.3352005919f, +0.6704011838f, +1.0056017757f, +1.3408023676f,
        +1.6760029595f, +2.0112035514f, +2.3464041433f,
    };
};

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
    static constexpr bool kIsAffine = false;
    static constexpr float kAffineBias = 0.0f;

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

// The non-uniform codebook a mode can be quantised on instead of its own grid, declared here and defined
// for kv4fp8 alone. It is what the LLOYD_MAX_LUT toggle substitutes: the 16 Lloyd-Max levels of a unit
// Gaussian in place of kv4fp8's uniform mid-rise grid, at the same 4 bits, the same two nibbles per byte
// and the same NZ tiling, so nothing about the cache layout moves.
//
// The two differences from TurboQuantModeTraits are deliberate and paired:
//
//   * the centroids are rounded into fp8 e4m3, because here they ARE the Cube operand. The uniform grid
//     can carry the exact code b - 7.5 (every one of those sixteen values is an e4m3 number) and leave
//     the step to kGain, which the decode divides into the per-vector scale. A non-uniform codebook has
//     no single step to factor out, so the reconstruction has to survive the operand cast intact -- and
//     e4m3 keeps only 3 mantissa bits, so the table is the rounded grid or the grid is not what decodes.
//   * the thresholds are the exact midpoints of those ROUNDED centroids, not of the unrounded ones. The
//     encoder's job is nearest-neighbour assignment on the codebook that actually reconstructs; using
//     Lloyd-Max's own thresholds here would mis-assign every coordinate near a boundary the rounding moved.
//
// Consequently kGain is 1: the centroid is the reconstruction and there is no step left to divide out.
template <TurboQuantMode MODE>
struct TurboQuantLloydMaxTraits;

template <>
struct TurboQuantLloydMaxTraits<TurboQuantMode::KV4_FP8> {
    static constexpr int32_t kLevels = 16;
    static constexpr int32_t kThresholdCount = kLevels - 1;
    // The levels are the reconstruction, so nothing is factored out of the operand.
    static constexpr float kGain = 1.0f;
    // E[(x - q(x))^2] over N(0, 1) on the rounded grid below: 0.0098195, i.e. 20.079 dB, against the
    // unrounded Lloyd-Max optimum's 0.0095010 (20.222 dB) and the uniform grid's 0.0115429 (19.377 dB).
    // The e4m3 rounding costs 0.14 dB of the 0.85 dB the codebook is worth.
    static constexpr float kDistortion = 0.0098195019f;

    // Max's 16-level table for N(0, 1), each entry rounded to nearest fp8 e4m3.
    static constexpr float kCentroids[kLevels] = {
        -2.7500000000f, -2.0000000000f, -1.6250000000f, -1.2500000000f,
        -0.9375000000f, -0.6875000000f, -0.3750000000f, -0.1250000000f,
        +0.1250000000f, +0.3750000000f, +0.6875000000f, +0.9375000000f,
        +1.2500000000f, +1.6250000000f, +2.0000000000f, +2.7500000000f,
    };
    // Exact midpoints of the rounded centroids above; every one is representable in fp32 without error.
    static constexpr float kThresholds[kThresholdCount] = {
        -2.3750000000f, -1.8125000000f, -1.4375000000f, -1.0937500000f,
        -0.8125000000f, -0.5312500000f, -0.2500000000f, +0.0000000000f,
        +0.2500000000f, +0.5312500000f, +0.8125000000f, +1.0937500000f,
        +1.4375000000f, +1.8125000000f, +2.3750000000f,
    };
};

// The gain the decode divides into the per-vector scale, for a codec instantiated with or without the
// non-uniform codebook. A free template so the arm that is not taken is never instantiated, which is what
// lets a mode with no TurboQuantLloydMaxTraits specialisation still name this.
template <TurboQuantMode MODE, bool LLOYD_MAX>
constexpr float TurboQuantCodebookGain()
{
    if constexpr (LLOYD_MAX) {
        return TurboQuantLloydMaxTraits<MODE>::kGain;
    } else {
        return TurboQuantModeTraits<MODE>::kGain;
    }
}

// Whether the codebook covers exactly the mode's levels, which it must: the packed nibble is an index
// into it. Written as a function rather than inline in a static_assert because `||` does not stop a
// static_assert's operand from being instantiated, and TurboQuantLloydMaxTraits has no definition for a
// mode that has no codebook.
template <TurboQuantMode MODE, bool LLOYD_MAX>
constexpr bool TurboQuantCodebookCoversMode()
{
    if constexpr (LLOYD_MAX) {
        return TurboQuantModeTraits<MODE>::kLevels == TurboQuantLloydMaxTraits<MODE>::kLevels;
    } else {
        return true;
    }
}

// kv4fp8 writes its packed planes NZ-tiled (turboquant_layout.h, NzTiledPackedByte); the codebook modes
// keep row-major slots. The cache writer and the Cube decode both key on this.
template <TurboQuantMode MODE>
constexpr bool kStoresNzTiles = MODE == TurboQuantMode::KV4_FP8;

// How the Cube decode's vector half expands a packed 4-bit plane into Cube operands. This selects an
// instruction sequence, never a stored format: every variant reads the same bytes at the same addresses
// and moves the same GM traffic, which is what makes the benchmark legs comparable.
//
//   kNative     the shipping affine expand -- int4b_t Cast, DeInterleave, then one Adds per plane, because
//               the operand carries the code b - 7.5 and the step folds into the score scale (kGain).
//   kBypass     a timing instrument: no expand at all, the zeroed staging buffer goes to L1 instead. Its
//               output is all-zero by construction and numerically meaningless.
//   kGatherLut  Option C: the Adds replaced by a 16-entry UB Gather, the sequence a non-uniform codebook
//               (Lloyd-Max) would need. The table it gathers is the uniform grid itself, so the result is
//               bit-identical to kNative and the leg prices the instruction sequence alone, with the
//               codebook question held out. See tests/research/QJL_3PLUS1_PHASE1.md.
//   kLloydMaxLut  kGatherLut's instruction sequence carrying the codebook it was built for: the same
//               Gather over the same 16-entry table at the same UB cost, filled with
//               TurboQuantLloydMaxTraits' e4m3-rounded Lloyd-Max levels instead of the uniform grid. It
//               is therefore the one variant whose output legitimately differs from kNative's, and the
//               only one whose cache has to be written by an encoder on the same codebook -- the
//               thresholds move with the levels. The pair (kGatherLut, kLloydMaxLut) separates the
//               expand's latency from the codebook's fidelity: identical instructions, different table.
enum class DecodeUnpack : uint32_t {
    kNative = 0,
    kBypass = 1,
    kGatherLut = 2,
    kLloydMaxLut = 3,
};

// Whether an expand resolves its operands through the UB centroid table, and whether that table holds the
// non-uniform codebook. kLloydMaxLut is a kGatherLut with different bytes in the same 64 B.
constexpr bool DecodeUnpackGathers(DecodeUnpack unpack)
{
    return unpack == DecodeUnpack::kGatherLut || unpack == DecodeUnpack::kLloydMaxLut;
}

constexpr bool DecodeUnpackIsLloydMax(DecodeUnpack unpack)
{
    return unpack == DecodeUnpack::kLloydMaxLut;
}

constexpr TurboQuantModeConfig TurboQuantModeConfigOf(TurboQuantMode mode)
{
    return mode == TurboQuantMode::KV3_FP4
               ? TurboQuantModeConfig{TurboQuantMode::KV3_FP4, TurboQuantOperand::kFp4E2m1, 3, 8, 8, 3,
                                      2.7881744355f, 0.0384442586f, false, 0.0f}
           : mode == TurboQuantMode::KV4_FP8
               ? TurboQuantModeConfig{TurboQuantMode::KV4_FP8, TurboQuantOperand::kFp8E4m3fn, 4, 16, 2, 1,
                                      2.9832882881f, 0.0115428844f, true, 7.5f}
               : TurboQuantModeConfig{TurboQuantMode::KV5_FP8, TurboQuantOperand::kFp8E4m3fn, 5, 32, 8, 5,
                                      2.2064579256f, 0.0028686787f, false, 0.0f};
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

constexpr int64_t TurboQuantOperandBytes(TurboQuantOperand operand, int64_t elems)
{
    return operand == TurboQuantOperand::kFp4E2m1 ? elems / 2 : elems;
}

}
}

#endif
