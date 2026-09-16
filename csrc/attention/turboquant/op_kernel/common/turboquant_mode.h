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
