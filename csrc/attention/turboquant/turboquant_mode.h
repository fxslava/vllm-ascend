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
 * TurboQuant mode definitions for the three KV-cache storage formats.
 *
 * The mode enum and traits describe the 3-bit, 4-bit and 5-bit codebook
 * layouts and the corresponding Cube operand path.
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
    // 4-bit storage, 16 uniform levels, expanded to fp8 e4m3fn for Cube mad.
    // The only affine rate: the code IS the level, so there is no codebook.
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
    // cast(g * c) is the stored codebook and (s / g) the effective scale.  For
    // an affine mode this is 1 / step; see the file header.
    float codebook_gain;
    // What the codebook delivers on N(0,1) after the cast, E[(X - Q(X))^2].
    float distortion;
    // Whether the levels are uniform, so the code is the level and the expand
    // needs no centroid table.  True for kv4fp8 only.
    bool is_affine;
    // The offset subtracted from a code to reach its level, (levels - 1) / 2 for
    // a midrise quantizer.  Meaningless when is_affine is false.
    float affine_bias;

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
    static constexpr bool kIsAffine = false;
    static constexpr float kAffineBias = 0.0f;

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
    // 1 / step for the optimal 16-level uniform midrise quantizer of N(0,1):
    // step = 0.3352006, clipping at +-7.5 step = +-2.514 sigma.  Searched by
    // scripts/tq_multimode_calibration.py.
    static constexpr float kGain = 2.9832882881f;
    static constexpr float kDistortion = 0.0115428844f;
    static constexpr bool kIsAffine = true;
    static constexpr float kAffineBias = 7.5f;

    // The levels are the integers q - 7.5, and every one of the sixteen is an
    // exact fp8_e4m3fn value (7.5 = 1.875 * 4, three mantissa bits), so the
    // cast onto the operand grid is lossless and the step lives entirely in
    // kGain.  The table is kept because the host mirror and the distortion
    // figure are quoted against it; the device does NOT read it -- see
    // TurboQuantModeCodec::UnpackAffine, where the whole expand is one Adds.
    static constexpr float kCentroids[kLevels] = {
        -7.5000000000f, -6.5000000000f, -5.5000000000f, -4.5000000000f,
        -3.5000000000f, -2.5000000000f, -1.5000000000f, -0.5000000000f,
        +0.5000000000f, +1.5000000000f, +2.5000000000f, +3.5000000000f,
        +4.5000000000f, +5.5000000000f, +6.5000000000f, +7.5000000000f,
    };
    // Uniform boundaries, (i - 7) * step.  NOT the same table as
    // TurboQuantCodec<4>::Threshold any more: a cache written by the shipping
    // AIV-only 4-bit encoder is a Lloyd-Max *index* cache and cannot be read by
    // this path, and vice versa.  The two are separate caches with separate
    // kernels, so there is nothing to keep compatible -- but the pair
    // (kThresholds, kCentroids, kGain) has to stay internally consistent, which
    // is what --emit-header guarantees.
    static constexpr float kThresholds[kLevels - 1] = {
        -2.3464041433f, -2.0112035514f, -1.6760029595f, -1.3408023676f,
        -1.0056017757f, -0.6704011838f, -0.3352005919f, +0.0000000000f,
        +0.3352005919f, +0.6704011838f, +1.0056017757f, +1.3408023676f,
        +1.6760029595f, +2.0112035514f, +2.3464041433f,
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
    static constexpr bool kIsAffine = false;
    static constexpr float kAffineBias = 0.0f;

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

/*
 * Compile-time cut points for the Cube decode's ablation ladder.
 *
 * Each stage runs everything the one before it does plus one more piece of the
 * split kernel, so the latency difference between neighbours is that piece's
 * cost.  A template parameter on TurboQuantCubeDecodeSplit, never a runtime
 * argument: a stage below STAGE_5_FULL_PIPELINE compiles the later pieces out,
 * and STAGE_5_FULL_PIPELINE is the shipping kernel itself.  The values cross
 * the launch boundary, so do not renumber them.
 */
enum class DecodeAblationStage : int32_t {
    // Packed K/V and scale tiles, GM -> UB through CopyInTile.
    STAGE_0_MTE2_ONLY = 0,
    // + the affine unpack onto the fp8 grid, in UB.
    STAGE_1_UNPACK = 1,
    // + the query's GM read, cast, Pi rotation and operand cast.  Once per
    // task, not per tile: the cache is stored rotated, so the split kernel's
    // only Walsh-Hadamard is the query's.
    STAGE_2_HADAMARD = 2,
    // + every V -> MTE3 edge and UB -> L1 copy, K/V tiles and the query.
    STAGE_3_L1_STAGING = 3,
    // + MTE1 load, the score Mmad and its Fixpipe, with the AIV/AIC handshake.
    STAGE_4_SCORE_GEMM = 4,
    // + online softmax, context GEMM, accumulator and the partial writeback.
    STAGE_5_FULL_PIPELINE = 5,
};

constexpr int32_t kDecodeAblationStageCount = 6;

constexpr bool DecodeAblationStageIsValid(int32_t raw)
{
    return raw >= 0 && raw < kDecodeAblationStageCount;
}

inline const char *DecodeAblationStageName(DecodeAblationStage stage)
{
    switch (stage) {
        case DecodeAblationStage::STAGE_0_MTE2_ONLY:
            return "stage0_mte2";
        case DecodeAblationStage::STAGE_1_UNPACK:
            return "stage1_unpack";
        case DecodeAblationStage::STAGE_2_HADAMARD:
            return "stage2_hadamard";
        case DecodeAblationStage::STAGE_3_L1_STAGING:
            return "stage3_l1_staging";
        case DecodeAblationStage::STAGE_4_SCORE_GEMM:
            return "stage4_score_gemm";
        case DecodeAblationStage::STAGE_5_FULL_PIPELINE:
            return "stage5_full";
    }
    return "stage5_full";
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
