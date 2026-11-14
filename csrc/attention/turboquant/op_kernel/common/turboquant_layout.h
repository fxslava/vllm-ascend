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

// The normative TurboQuant layout contract: every constant and address function the kernels and
// the host tiling have to agree on, including the RFC 3.1 strip layout and the NZ-tiled cache.
// Dependency-free on purpose -- the host tiling and the Torch-free tests include it without a
// kernel toolchain.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

// Vector lanes.
constexpr uint32_t kFp32PerBlock = 8;
constexpr uint32_t kFp32PerRepeat = 64;
constexpr uint32_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;
constexpr uint32_t kGatherSrcBase = 0;
constexpr uint32_t kMaxRepeatTimes = 255;
constexpr uint32_t kVectorSubcoresPerBlock = 2;

// Codec.
constexpr uint32_t kPackFactor = 2;
constexpr uint32_t kCodecLevels = 16;

// RFC 3.1 strip layout. A strip is R = kStripTokens tokens of one (block, kv head) page: the
// unit the prefill operator rotates and quantises in one pass, and the row group the value
// plane is channel-major inside.
//
//   key   nibble = ((block*H + head)*S + token)*d + channel
//   value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
//   scale index  = ((kv*P + block)*H + head)*S + token
//
// with H kv heads, S the page size in tokens, d the head size, P the page count, and kv 0 for
// the key plane and 1 for the value plane. Nibbles are two per byte, low nibble first (even
// element index in bits [3:0], odd in [7:4]), signed INT4 in [-8, 7].
constexpr uint32_t kStripTokens = 16;

// Decode task layout shared by both decodes.
constexpr uint32_t kMaxSequenceSplits = 8;
constexpr uint32_t kFusedContextLimit = 4096;
constexpr uint32_t kPartialTail = 2 * kFp32PerBlock;
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;
constexpr uint32_t kMinBurstBytes = 128;

// Cube decode.
constexpr uint32_t kCubeTileRows = 64;
constexpr uint32_t kCubeUnpackRows = 8;
constexpr uint32_t kOperandC0 = 32;
constexpr uint32_t kCubeTileM = 16;
constexpr uint32_t kCubeSlots = 2;

// The modulo-scheduled fused decode keeps a deeper ring than a lock-step one: kCubePipelineSlots
// tiles in flight, so the MTE2 read of tile k + kCubePipelineSlots is issued inside the very
// iteration whose Cube and Vector work it hides behind. Three is what the cross-core flag space
// affords at one id per slot. GetffstMsg masks a flag id to four bits and the toolkit reserves
// 11..14 for its own barriers (asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:
// SYNC_AIC_FLAG, SYNC_AIV_FLAG, SYNC_AIC_AIV_FLAG, SYNC_AIV_ONLY_ALL), which leaves ids 0..10,
// and the schedule spends 3 + 3 + 2 + 1 + 1 of them. Neither buffer is the limit: a slot costs
// about 10 KB of the 256 KB UB partition and 32 KB of the 512 KB L1 at headSize 256.
constexpr uint32_t kCubePipelineSlots = 3;

// Score Fixpipe windows of the modulo schedule. Two, so the Cube's score GEMM for tile k + 1
// fills one window while the vector unit still owns tile k's -- the overlap a single window
// cannot have, because the online softmax rewrites the score rows in place. The context window
// stays single: a tile's context is accumulated in the same iteration that produced it, before
// the next GEMM pair can start.
constexpr uint32_t kCubeScoreWindows = 2;

// AKA3003. Consecutive UB buffers of the tile ingest are offset by this much, so the slot MTE2
// is filling and the slot the vector ALU is reading do not present the same bank to both ports
// in the same cycle.
constexpr uint32_t kBankSkewBytes = 32;

// ------------------------------------------------------- the NZ-tiled packed cache
// An NZ-tiled packed cache (kv4fp8) lays a physical block out as [kCubeTileRows-row tile]
// [kv head][kOperandC0-byte column group][tile row][byte], so the packed plane of one
// (tile, kv head) is a single contiguous [groups, kCubeTileRows, kOperandC0] fractal image. The
// decode reads it in one burst, and because the nibble expand is byte-wise it lands in the
// Cube's NZ order as it is unpacked. The block size must be a multiple of kCubeTileRows.
//
// This is the byte that holds packed column `column` of (slot, kv head); the kernels issue the
// same arithmetic as strided DMA.
constexpr uint64_t NzTiledPackedByte(uint64_t slot, uint64_t kvHead, uint64_t column,
                                     uint64_t numKvHeads, uint64_t packedBytes)
{
    const uint64_t tileRow = slot % kCubeTileRows;
    return (slot - tileRow) * numKvHeads * packedBytes + kvHead * kCubeTileRows * packedBytes +
           ((column / kOperandC0) * kCubeTileRows + tileRow) * kOperandC0 + column % kOperandC0;
}

// NzTiledPackedByte names the byte of a packed COLUMN. A LOGICAL element is
// (slot, kvHead, channel), and getting from one to the other is the kv4fp8 nibble pairing,
// which is NOT the RFC 3.1 strip contract's "two adjacent channels per byte":
//
//   channel c          -> the LOW  nibble of column  c                 , for c <  headSize/2
//   channel c          -> the HIGH nibble of column (c - headSize/2)   , for c >= headSize/2
//
// so one byte carries channels (c, c + headSize/2) of ONE token. That is forced by the decode's
// own expand: UnpackAffine (src/common/turboquant_codec_mx.h) DeInterleaves a chunk into a low
// plane and a high plane of equal length, and StageTileToL1
// (src/decode/vector/turboquant_vector_service.h) stages the low plane at L1 offset 0 and the
// high plane at + tilePlaneBytes_. The L1 operand is the kCubeTileRows x headSize NZ image whose
// groups span kOperandC0 channels each, so the low plane lands on channels [0, headSize/2) and
// the high plane on [headSize/2, headSize).
//
// An ingress writer that feeds this decode must produce exactly this pairing. The inverse of the
// expand is production code -- StageAffinePlane's Interleave + int4b_t Cast in the same codec --
// and tests/cpu/test_layout_bridge.cpp is the gate that the writer's address function and the
// decode's name the same byte and the same nibble.
//
// `slot` is the GLOBAL row, physical * blockSize + token, which makes these self-contained: no
// separate page base is added, because (slot - slot % kCubeTileRows) already carries it.
constexpr uint64_t NzTiledChannelByte(uint64_t slot, uint64_t kvHead, uint64_t channel,
                                      uint64_t numKvHeads, uint64_t headSize)
{
    const uint64_t packedBytes = headSize / kPackFactor;
    return NzTiledPackedByte(slot, kvHead, channel % packedBytes, numKvHeads, packedBytes);
}

// 0 for the low nibble of that byte, 1 for the high one.
constexpr uint32_t NzTiledChannelNibble(uint64_t channel, uint64_t headSize)
{
    return channel < headSize / kPackFactor ? 0u : 1u;
}

// The scale lanes of ONE cache row: 2 * numKvHeads (a key lane and a value lane per head)
// rounded up to a whole fp32 block, so a tile's scales are one contiguous
// kCubeTileRows * NzScaleSlotFloats burst beside the tile. ScaleSlotFloats in
// turboquant_common.h is this function; it lives here because the host tiling and the
// toolkit-free tests need it without a kernel toolchain.
constexpr uint32_t NzScaleSlotFloats(uint32_t numKvHeads)
{
    return ((2u * numKvHeads + kFp32PerBlock - 1u) / kFp32PerBlock) * kFp32PerBlock;
}

// kv 0 for the key plane, 1 for the value plane. `slot` is the global row, as above.
constexpr uint64_t NzTiledScaleIndex(uint64_t slot, uint64_t kvHead, uint64_t kv,
                                     uint64_t numKvHeads)
{
    return slot * NzScaleSlotFloats(static_cast<uint32_t>(numKvHeads)) + kv * numKvHeads + kvHead;
}

// ------------------------------------------------------- cross-core flag ids
// The lock-step decode's flags.
constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagProductReady = 1;
constexpr uint16_t kFlagSlotReady = 0;
constexpr uint16_t kFlagSlotFree = 2;
constexpr uint16_t kFlagScoresReady = 4;
constexpr uint16_t kFlagContextReady = 5;
constexpr uint16_t kFlagProbsReady = 6;

// The modulo-scheduled decode's flags, one id per ring slot and per score window so that neither
// core can ever be released by a token another tile's handshake posted. Both cores derive the
// same index from the same compile-time constant, so every flag is a point-to-point edge between
// one producer step and one consumer step. Ids 0..10; see kCubePipelineSlots for why that is the
// whole budget.
constexpr uint16_t kOptFlagSlotReady = 0;
constexpr uint16_t kOptFlagSlotFree = kOptFlagSlotReady + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagScoresReady =
    kOptFlagSlotFree + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagContextReady =
    kOptFlagScoresReady + static_cast<uint16_t>(kCubeScoreWindows);
constexpr uint16_t kOptFlagProbsReady = kOptFlagContextReady + 1;
// The first id the toolkit keeps for its own barriers; every id above belongs to the schedule.
constexpr uint16_t kOptFlagReservedFirst = 11;

// ------------------------------------------------------- query rotation
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

// The raw-query Cube decode's query prologue. A block whose share holds
// kQueryBasisMinCubeVectors or more query vectors rotates them kQueryBasisCubeVectors at a time:
// one single-pass fp16 H16 Mmad per chunk, split between the two vector subcores by a
// dual-destination Fixpipe. A share below that, and the remainder of one, rotate on the vector
// cores alone, so no Mmad carries padding rows that would idle a subcore.
constexpr uint32_t kQueryBasisCubeVectors = 16;
constexpr uint32_t kQueryBasisMinCubeVectors = 16;

// What the raw-query Cube decode writes to its output, chosen per launch:
//   kRotatedBasis  softmax(q k^T) Pi v, for an o_proj with Pi folded into it offline
//   kUnrotated     Pi applied once more to the fp32 accumulator before the output cast (Pi^2 = I)
//   kGated         kUnrotated, then divided by 1 + exp(-gate): an attn_output_gate layer's gate
enum class TurboQuantOutputStage : uint32_t {
    kRotatedBasis = 0,
    kUnrotated = 1,
    kGated = 2,
};

// Gate logits are clamped from below before exp(-gate), so the denominator stays finite in fp32.
constexpr float kGateLogitFloor = -80.0f;

}  // namespace turboquant
}  // namespace vllm_ascend

#endif
