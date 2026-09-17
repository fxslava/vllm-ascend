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

// The TurboQuant layout contract: every constant both the kernels and the host tiling have to agree
// on. Dependency-free on purpose -- op_host/turboquant_tiling.h and the Torch-free tests include it
// without a kernel toolchain.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

// Vector lanes.
constexpr uint32_t kFp32PerBlock = 8;
constexpr uint32_t kFp32PerRepeat = 64;
constexpr uint32_t kBrcbSrcLanes = kFp32PerBlock;
constexpr uint32_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;
constexpr uint32_t kGatherSrcBase = 0;
constexpr uint32_t kMaxRepeatTimes = 255;
constexpr uint32_t kVectorSubcoresPerBlock = 2;

// Codec.
constexpr uint32_t kPackFactor = 2;
constexpr uint32_t kCodecLevels = 16;

// Decode task layout shared by both decodes.
constexpr uint32_t kMaxSequenceSplits = 8;
constexpr uint32_t kFusedContextLimit = 4096;
constexpr uint32_t kPartialTail = 2 * kFp32PerBlock;
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;
constexpr uint32_t kMinBurstBytes = 128;

// AIV decode.
constexpr uint32_t kAivTileRows = 16;

// Cube decode.
constexpr uint32_t kCubeTileRows = 64;
constexpr uint32_t kCubeUnpackRows = 8;
constexpr uint32_t kOperandC0 = 32;
constexpr uint32_t kCubeTileM = 16;
constexpr uint32_t kCubeSlots = 2;

// An NZ-tiled packed cache (kv4fp8) lays a physical block out as [kCubeTileRows-row tile][kv head]
// [kOperandC0-byte column group][tile row][byte], so the packed plane of one (tile, kv head) is a single
// contiguous [groups, kCubeTileRows, kOperandC0] fractal image. The decode reads it in one burst, and
// because the nibble expand is byte-wise it lands in the Cube's NZ order as it is unpacked. The block
// size must be a multiple of kCubeTileRows. This is the byte that holds packed column `column` of
// (slot, kv head); the kernels issue the same arithmetic as strided DMA.
constexpr uint64_t NzTiledPackedByte(uint64_t slot, uint64_t kvHead, uint64_t column, uint64_t numKvHeads,
                                     uint64_t packedBytes)
{
    const uint64_t tileRow = slot % kCubeTileRows;
    return (slot - tileRow) * numKvHeads * packedBytes + kvHead * kCubeTileRows * packedBytes +
           ((column / kOperandC0) * kCubeTileRows + tileRow) * kOperandC0 + column % kOperandC0;
}

constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagProductReady = 1;
constexpr uint16_t kFlagSlotReady = 0;
constexpr uint16_t kFlagSlotFree = 2;
constexpr uint16_t kFlagScoresReady = 4;
constexpr uint16_t kFlagContextReady = 5;
constexpr uint16_t kFlagProbsReady = 6;

// Query rotation.
constexpr uint32_t kRotateQTile = 16;
constexpr uint32_t kRotateQMaxChunkElements = 4096;
constexpr uint32_t kRotateQMinCubeTokens = 2;
constexpr uint32_t kRotateQH16Elements = kRotateQTile * kRotateQTile;
constexpr uint16_t kRotateQHalfOne = 0x3C00;
constexpr uint16_t kRotateQHalfMinusOne = 0xBC00;

enum RotateQVariant : uint32_t {
    kHiLo = 0x1u,
    kDualDst = 0x2u,
};

enum class RotateQPrecision : uint32_t {
    kSinglePassRint = 0,
    kHiLoResidual = 1,
};

// The raw-query Cube decode's query prologue. A block whose share holds kQueryBasisMinCubeVectors or more
// query vectors rotates them kQueryBasisCubeVectors at a time: one single-pass fp16 H16 Mmad per chunk, split
// between the two vector subcores by a dual-destination Fixpipe. A share below that, and the remainder of
// one, rotate on the vector cores alone, so no Mmad carries padding rows that would idle a subcore.
constexpr uint32_t kQueryBasisCubeVectors = 16;
constexpr uint32_t kQueryBasisMinCubeVectors = 16;

// What the raw-query Cube decode writes to its output, chosen per launch:
//   kRotatedBasis  softmax(q k^T) Pi v, for an o_proj with Pi folded into it offline
//   kUnrotated     Pi applied once more to the fp32 accumulator before the output cast (Pi^2 = I)
//   kGated         kUnrotated, then divided by 1 + exp(-gate): an attn_output_gate layer's sigmoid gate
enum class TurboQuantOutputStage : uint32_t {
    kRotatedBasis = 0,
    kUnrotated = 1,
    kGated = 2,
};

// Gate logits are clamped from below before exp(-gate), so the denominator stays finite in fp32.
constexpr float kGateLogitFloor = -80.0f;

}
}

#endif
