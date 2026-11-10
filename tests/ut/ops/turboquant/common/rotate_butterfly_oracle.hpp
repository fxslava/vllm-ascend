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

#pragma once
// Host model of src/scatter/routine_vector_rotate_butterfly.hpp.
//
// Three things live here, and the distinction between them is the whole point
// of the Tier-1 probe that consumes this header:
//
//   dense_*        the definition. Pi = (H8 (x) I32) . diag(signs) . (I8 (x) H32)
//                  built entry by entry and applied as a plain matrix product.
//                  Nothing about the device implementation enters it.
//   staged_*       the eight-stage butterfly cascade, in each direction's own
//                  stage order with the sign diagonal at that direction's own
//                  position. This is what the kernel claims to compute.
//   algo_*         a lane-accurate simulation of each DEVICE implementation --
//                  including the shuffle recursion's bit-rotation bookkeeping
//                  and its permuted sign frame. This is what the kernel
//                  actually computes.
//
// A gate that only compared `staged_` against `dense_` would check the stage table and nothing
// else; the shuffle recursion's P^6 = I argument would stay unverified until a CAModel run
// disagreed with the oracle and said nothing about which of the two was wrong.
//
// Every value here is FP32 and every operation is an add, a subtract or a
// multiply by +-1, so for inputs whose rotated panel stays inside 2^24 the
// comparisons below are exact and bit equality is demandable.

#include <cstdint>
#include <cstring>
#include <vector>

namespace rot_oracle {

constexpr int kChannels = 256;
constexpr int kGroup = 32;   // the inner Hadamard's width
constexpr int kGroups = kChannels / kGroup;  // 8, the outer Hadamard's width
constexpr int kVecLanes = 64;                // FP32 lanes in VECTOR_REG_WIDTH
constexpr int kRegisters = kChannels / kVecLanes;  // 4
constexpr int kStages = 8;
constexpr int kLaneBits = 6;  // log2(kVecLanes)

enum class Direction { Egress = 0, Ingress = 1 };

// Mirrors kRotStrideOf / kRotSignAfterStage in the routine header.
constexpr int kStrideOf[2][kStages] = {
    {32, 64, 128, 1, 2, 4, 8, 16},
    {1, 2, 4, 8, 16, 32, 64, 128},
};
constexpr int kSignAfterStage[2] = {3, 5};

inline int parity(unsigned value)
{
    int p = 0;
    while (value != 0u) {
        p ^= static_cast<int>(value & 1u);
        value >>= 1;
    }
    return p;
}

// H[i][j] = (-1)^popcount(i & j), the Sylvester Hadamard entry.
inline int hadamard(int i, int j)
{
    return parity(static_cast<unsigned>(i & j)) ? -1 : 1;
}

// The canonical sign diagonal, identical to tq_oracle::pi_sign. Balanced
// enough that a dropped sign multiply cannot be mistaken for a pass, and not a
// simple alternation, which the butterfly's own structure could mask.
inline int pi_sign(int channel)
{
    return ((channel * channel + 3 * channel + 1) % 7 < 3) ? -1 : 1;
}

inline std::vector<float> canonical_signs()
{
    std::vector<float> s(kChannels);
    for (int c = 0; c < kChannels; ++c) s[c] = static_cast<float>(pi_sign(c));
    return s;
}

// --------------------------------------------------------------- the matrix
// Pi[c][b] with c = c1*32 + c0 and b = g*32 + i:
//   Pi[c][b] = H8[c1][g] . signs[g*32 + c0] . H32[c0][i]
inline float dense_entry(const float* signs, int row, int col)
{
    const int c1 = row / kGroup;
    const int c0 = row % kGroup;
    const int g = col / kGroup;
    const int i = col % kGroup;
    return static_cast<float>(hadamard(c1, g)) * signs[g * kGroup + c0] *
           static_cast<float>(hadamard(c0, i));
}

// y = Pi . x
inline std::vector<float> dense_left(const float* x, const float* signs)
{
    std::vector<float> y(kChannels, 0.0f);
    for (int r = 0; r < kChannels; ++r) {
        double acc = 0.0;
        for (int c = 0; c < kChannels; ++c) acc += dense_entry(signs, r, c) * x[c];
        y[r] = static_cast<float>(acc);
    }
    return y;
}

// y = x . Pi
inline std::vector<float> dense_right(const float* x, const float* signs)
{
    std::vector<float> y(kChannels, 0.0f);
    for (int c = 0; c < kChannels; ++c) {
        double acc = 0.0;
        for (int r = 0; r < kChannels; ++r) acc += x[r] * dense_entry(signs, r, c);
        y[c] = static_cast<float>(acc);
    }
    return y;
}

// -------------------------------------------------------- the stage cascade
// out[i] = in[i ^ s] + sgn_s(i) . in[i], sgn_s(i) = (i & s) ? -1 : +1.
inline void butterfly_stage(std::vector<float>& value, int stride)
{
    std::vector<float> next(kChannels, 0.0f);
    for (int i = 0; i < kChannels; ++i) {
        const float sign = (i & stride) ? -1.0f : 1.0f;
        next[i] = value[i ^ stride] + sign * value[i];
    }
    value.swap(next);
}

inline std::vector<float> staged(Direction dir, const float* x, const float* signs,
                                 float scale = 1.0f)
{
    const int d = static_cast<int>(dir);
    std::vector<float> value(x, x + kChannels);
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage == kSignAfterStage[d])
            for (int i = 0; i < kChannels; ++i) value[i] *= signs[i];
        butterfly_stage(value, kStrideOf[d][stage]);
    }
    for (int i = 0; i < kChannels; ++i) value[i] *= scale;
    return value;
}

// ------------------------------------------------- device algorithm: RegGather
// Four 64-lane registers; channel i lives in register i >> 6, lane i & 63.
// Lane-bit stages are a gather of the partner lane plus a signed fused
// multiply-add; register-bit stages pair whole registers.
inline void reg_pair_stage(std::vector<float>& v, int stride)
{
    // stride is 64 or 128: i ^ stride never leaves the lane, only the register.
    std::vector<float> next(kChannels, 0.0f);
    for (int i = 0; i < kChannels; ++i) {
        const float sign = (i & stride) ? -1.0f : 1.0f;
        next[i] = v[i ^ stride] + sign * v[i];
    }
    v.swap(next);
}

inline void lane_gather_stage(std::vector<float>& v, int stride)
{
    for (int r = 0; r < kRegisters; ++r) {
        float partner[kVecLanes];
        for (int lane = 0; lane < kVecLanes; ++lane)
            partner[lane] = v[r * kVecLanes + (lane ^ stride)];
        for (int lane = 0; lane < kVecLanes; ++lane) {
            const float sign = (lane & stride) ? -1.0f : 1.0f;
            v[r * kVecLanes + lane] = partner[lane] + sign * v[r * kVecLanes + lane];
        }
    }
}

inline std::vector<float> algo_reg_gather(Direction dir, const float* x, const float* signs,
                                          float scale = 1.0f)
{
    std::vector<float> v(x, x + kChannels);
    auto apply_signs = [&]() {
        for (int i = 0; i < kChannels; ++i) v[i] *= signs[i];
    };
    if (dir == Direction::Egress) {
        lane_gather_stage(v, 32);
        reg_pair_stage(v, 64);
        reg_pair_stage(v, 128);
        apply_signs();
        for (int s : {1, 2, 4, 8, 16}) lane_gather_stage(v, s);
    } else {
        for (int s : {1, 2, 4, 8, 16}) lane_gather_stage(v, s);
        apply_signs();
        lane_gather_stage(v, 32);
        reg_pair_stage(v, 64);
        reg_pair_stage(v, 128);
    }
    for (int i = 0; i < kChannels; ++i) v[i] *= scale;
    return v;
}

// ------------------------------------------------ device algorithm: RegShuffle
// One register's worth of the recursion. `lanes` is 64 FP32.
//
//   DeInterleave(E, O, V, V):  E[j] = V[2j] and O[j] = V[2j+1] for j < 32, each
//                              duplicated into the upper half of its register.
//   Select(V, S, D, maskH):    lanes 0..31 from S, lanes 32..63 from D.
//
// which composes to (P . B_0) with (P x)[k] = x[rotl6(k)].
inline void shuffle_step(float* lanes)
{
    float even[kVecLanes];
    float odd[kVecLanes];
    for (int j = 0; j < kVecLanes / 2; ++j) {
        even[j] = lanes[2 * j];
        even[j + kVecLanes / 2] = lanes[2 * j];
        odd[j] = lanes[2 * j + 1];
        odd[j + kVecLanes / 2] = lanes[2 * j + 1];
    }
    for (int k = 0; k < kVecLanes; ++k) {
        const float sum = even[k] + odd[k];
        const float diff = even[k] - odd[k];
        lanes[k] = (k < kVecLanes / 2) ? sum : diff;
    }
}

// The same splice with the wings replaced by the halves themselves: P alone.
inline void shuffle_permute(float* lanes)
{
    float even[kVecLanes];
    float odd[kVecLanes];
    for (int j = 0; j < kVecLanes / 2; ++j) {
        even[j] = lanes[2 * j];
        even[j + kVecLanes / 2] = lanes[2 * j];
        odd[j] = lanes[2 * j + 1];
        odd[j + kVecLanes / 2] = lanes[2 * j + 1];
    }
    for (int k = 0; k < kVecLanes; ++k) lanes[k] = (k < kVecLanes / 2) ? even[k] : odd[k];
}

// Ingress only. The sign registers are carried into the P^5 frame once; the
// data runs six uniform steps and comes back to natural order at P^6 = I.
inline std::vector<float> algo_reg_shuffle(const float* x, const float* signs,
                                           float scale = 1.0f)
{
    std::vector<float> perm_signs(signs, signs + kChannels);
    for (int r = 0; r < kRegisters; ++r)
        for (int k = 0; k < 5; ++k) shuffle_permute(perm_signs.data() + r * kVecLanes);

    std::vector<float> v(x, x + kChannels);
    for (int step = 0; step < 5; ++step)
        for (int r = 0; r < kRegisters; ++r) shuffle_step(v.data() + r * kVecLanes);
    for (int i = 0; i < kChannels; ++i) v[i] *= perm_signs[i];
    for (int r = 0; r < kRegisters; ++r) shuffle_step(v.data() + r * kVecLanes);
    reg_pair_stage(v, 64);
    reg_pair_stage(v, 128);
    for (int i = 0; i < kChannels; ++i) v[i] *= scale;
    return v;
}

// ------------------------------------------------- Approach A's UB tables
// Three 256-entry offset vectors and three 256-entry sign vectors, in stride
// order {1, 2, 4}, exactly as NativeRotateBlockedEngine indexes them.
constexpr int kGatherStages = 3;
constexpr int kGatherStrideOf[kGatherStages] = {1, 2, 4};

inline std::vector<std::uint32_t> blocked_gather_offsets()
{
    std::vector<std::uint32_t> table(static_cast<std::size_t>(kGatherStages) * kChannels);
    for (int t = 0; t < kGatherStages; ++t)
        for (int i = 0; i < kChannels; ++i)
            // AscendC::Gather's offsets are BYTE offsets into the source.
            table[static_cast<std::size_t>(t) * kChannels + i] =
                static_cast<std::uint32_t>((i ^ kGatherStrideOf[t]) * 4);
    return table;
}

inline std::vector<float> blocked_gather_signs()
{
    std::vector<float> table(static_cast<std::size_t>(kGatherStages) * kChannels);
    for (int t = 0; t < kGatherStages; ++t)
        for (int i = 0; i < kChannels; ++i)
            table[static_cast<std::size_t>(t) * kChannels + i] =
                (i & kGatherStrideOf[t]) ? -1.0f : 1.0f;
    return table;
}

}  // namespace rot_oracle
