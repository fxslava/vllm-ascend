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

// The modulo-scheduled fused decode (turboquant_fused_decode_opt.cpp) keeps a deeper ring than the
// lock-step one: kCubePipelineSlots tiles in flight, so the MTE2 read of tile k + kCubePipelineSlots is
// issued inside the very iteration whose Cube and Vector work it hides behind. Three is what the
// cross-core flag space affords at one id per slot. GetffstMsg masks a flag id to four bits and the
// toolkit reserves 11..14 for its own barriers (asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:
// SYNC_AIC_FLAG, SYNC_AIV_FLAG, SYNC_AIC_AIV_FLAG, SYNC_AIV_ONLY_ALL), which leaves ids 0..10, and the
// schedule spends 3 + 3 + 2 + 1 + 1 of them. Neither buffer is the limit: a slot costs about 10 KB of the
// 256 KB UB partition and 32 KB of the 512 KB L1 at headSize 256. The flag space is.
constexpr uint32_t kCubePipelineSlots = 3;

// Score Fixpipe windows of the modulo schedule. Two, so the Cube's score GEMM for tile k + 1 fills one
// window while the vector unit still owns tile k's -- the overlap a single window cannot have, because
// the online softmax rewrites the score rows in place. The context window stays single: a tile's context
// is accumulated in the same iteration that produced it, before the next GEMM pair can start.
constexpr uint32_t kCubeScoreWindows = 2;

// AKA3003. Consecutive UB buffers of the tile ingest are offset by this much, so the slot MTE2 is filling
// and the slot the vector ALU is reading do not present the same bank to both ports in the same cycle.
constexpr uint32_t kBankSkewBytes = 32;

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

// The modulo-scheduled decode's flags, one id per ring slot and per score window so that neither core can
// ever be released by a token another tile's handshake posted. Both cores derive the same index from the
// same compile-time constant, so every flag is a point-to-point edge between one producer step and one
// consumer step. Ids 0..10; see kCubePipelineSlots for why that is the whole budget.
constexpr uint16_t kOptFlagSlotReady = 0;
constexpr uint16_t kOptFlagSlotFree = kOptFlagSlotReady + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagScoresReady = kOptFlagSlotFree + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagContextReady = kOptFlagScoresReady + static_cast<uint16_t>(kCubeScoreWindows);
constexpr uint16_t kOptFlagProbsReady = kOptFlagContextReady + 1;
// The first id the toolkit keeps for its own barriers; every id above belongs to the schedule.
constexpr uint16_t kOptFlagReservedFirst = 11;

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

// The Unified Buffer budget.
//
// kUbPartitionBytes is what one AIV subcore of a dav-3510 MIX block addresses, and TPipe hands out every
// UB allocation from it in kUbBlockBytes units. A launch whose buffers do not fit does not fail at
// InitBuffer: it faults on first access (AIV error 341, "The address for VEC to access UB is out of
// bounds") or, worse, silently aliases, so the host refuses the launch instead -- see
// op_host/turboquant_tiling.h FusedDecodeUbBytesFor and the adapter's check.
//
// FusedDecodeUbBytes below MIRRORS TurboQuantVectorDecodeService::InitBuffers, the codec's Init and
// TurboQuantQueryBasis's UB half, in allocation order. It is not derived from them: a buffer added to
// either without a line here makes this an underestimate. The L1/L0 buffers of TurboQuantCubeMm and of
// the query prologue (A1/B1/A2/B2/CO1) are deliberately absent -- they are not UB.
constexpr int64_t kUbPartitionBytes = 256 * 1024;
constexpr int64_t kUbBlockBytes = 32;

// TurboQuantCodec4::kBinLanes, which the codecs' work buffer scales with. Kept here because the host
// tiling has to size the same buffer; turboquant_codec_950.h static_asserts that the two agree.
constexpr int64_t kCodecBinLanes = 8;
// TurboQuantCodecMx::kPeriodicWords.
constexpr int64_t kCodecPeriodicWords = 3 * static_cast<int64_t>(kFp32PerBlock);

constexpr int64_t AlignUp64(int64_t value, int64_t to) { return (value + to - 1) / to * to; }

// One UB allocation as TPipe actually charges it.
constexpr int64_t UbAlloc(int64_t bytes) { return AlignUp64(bytes, kUbBlockBytes); }

// Words a codec's work and constant buffers take for vec_len elements over batch_rows rows. Mirrors
// TurboQuantCodecMx::WorkBufferWords and ConstTableWords.
constexpr int64_t CodecWorkWords(int64_t vec_len, int64_t batch_rows, bool affine)
{
    const int64_t shared =
        static_cast<int64_t>(kBrcbDstLanes) + static_cast<int64_t>(kFp32PerBlock) + kCodecBinLanes * vec_len;
    return affine ? vec_len * batch_rows + 3 * vec_len + shared : 4 * vec_len * batch_rows + vec_len + shared;
}

constexpr int64_t CodecConstWords(int64_t vec_len, int64_t batch_rows, bool affine, bool gather_lut)
{
    if (affine) {
        return gather_lut ? static_cast<int64_t>(kCodecLevels) : 0;
    }
    return 2 * vec_len * batch_rows + kCodecPeriodicWords + vec_len + static_cast<int64_t>(kCodecLevels);
}

constexpr int64_t CodecUbBytes(int64_t vec_len, int64_t batch_rows, bool affine, bool gather_lut)
{
    int64_t bytes = UbAlloc(CodecWorkWords(vec_len, batch_rows, affine) * static_cast<int64_t>(sizeof(float)));
    const int64_t const_words = CodecConstWords(vec_len, batch_rows, affine, gather_lut);
    if (const_words > 0) {
        bytes += UbAlloc(const_words * static_cast<int64_t>(sizeof(int32_t)));
    }
    return bytes;
}

// What one fused-decode launch asks of the UB partition.
struct FusedDecodeUbShape {
    int64_t head_size = 0;
    int64_t num_kv_heads = 0;
    // TurboQuantModeConfig::PackedBytes(head_size).
    int64_t packed_bytes = 0;
    // TurboQuantCubeMm::OperandElems(head_size) and OperandElems(kCubeTileRows).
    int64_t operand_elems = 0;
    int64_t prob_operand_elems = 0;
    int64_t scalar_bytes = static_cast<int64_t>(sizeof(uint16_t));
    // RING_SLOTS, SCORE_WINDOWS and BANK_SKEW of the service instantiation.
    int64_t ring_slots = static_cast<int64_t>(kCubeSlots);
    int64_t score_windows = 1;
    int64_t bank_skew = 0;
    // kStoresNzTiles<MODE>, kBatchedCubeDecode<MODE>, the codec's kIsAffine and kGatherLut.
    bool nz_tiled = true;
    bool batched = true;
    bool affine_codec = true;
    bool gather_lut = false;
    // Whether the launch carries the raw-query prologue (TurboQuantQueryBasis).
    bool raw_query_prologue = false;
};

constexpr int64_t FusedDecodeUbBytes(const FusedDecodeUbShape &s)
{
    constexpr int64_t kHalfRows = static_cast<int64_t>(kCubeTileM) / 2;
    constexpr int64_t kFloat = static_cast<int64_t>(sizeof(float));
    constexpr int64_t kScoreBytes = kHalfRows * static_cast<int64_t>(kCubeTileRows) * kFloat;
    const int64_t scale_slot = AlignUp64(2 * s.num_kv_heads, static_cast<int64_t>(kFp32PerBlock));
    const int64_t rows = kHalfRows * s.head_size * kFloat;

    // The two Fixpipe destinations, first so their offsets never move.
    int64_t bytes = s.score_windows * UbAlloc(kScoreBytes);
    bytes += UbAlloc(rows); // contextBuf_
    // The tile ingest ring, each slot followed by its bank-skew pad.
    bytes += s.ring_slots * UbAlloc(static_cast<int64_t>(kCubeTileRows) * s.packed_bytes);
    bytes += s.ring_slots * UbAlloc(static_cast<int64_t>(kCubeTileRows) * scale_slot * kFloat);
    if (s.bank_skew > 0) {
        bytes += 2 * s.ring_slots * UbAlloc(s.bank_skew);
    }
    // The Cube operand staging: a whole NZ tile, or one unpack band.
    bytes += UbAlloc((s.nz_tiled ? static_cast<int64_t>(kCubeTileRows) : static_cast<int64_t>(kCubeUnpackRows)) *
                     s.operand_elems);
    bytes += UbAlloc(rows);                                               // accBuf_
    bytes += UbAlloc((s.batched ? kHalfRows : 1) * s.head_size * kFloat); // queryWorkBuf_
    bytes += UbAlloc(rows);                                               // queryInBuf_
    bytes += UbAlloc(kHalfRows * s.operand_elems);                        // queryOperandBuf_
    bytes += UbAlloc(kHalfRows * s.prob_operand_elems);                   // probOperandBuf_
    bytes += UbAlloc((s.batched ? 12 : 8) * kHalfRows * static_cast<int64_t>(kFp32PerBlock) * kFloat);
    bytes += UbAlloc(512 * kFloat); // reduceBuf_
    bytes += UbAlloc(2 * static_cast<int64_t>(kCubeTileRows) * static_cast<int64_t>(sizeof(int32_t)));
    bytes += UbAlloc((s.head_size > static_cast<int64_t>(kCubeTileRows) ? s.head_size
                                                                       : static_cast<int64_t>(kCubeTileRows)) *
                     kFloat);                                         // scratchBuf_
    bytes += UbAlloc(kHalfRows * s.head_size * s.scalar_bytes);       // outBuf_
    bytes += UbAlloc(2 * static_cast<int64_t>(kCubeTileRows) * kFloat); // maskBuf_

    // The ingest codec. An affine mode expands kCubeTileRows / 2 rows at a time, the others kCubeUnpackRows.
    const int64_t unpack_rows =
        s.affine_codec ? static_cast<int64_t>(kCubeTileRows) / 2 : static_cast<int64_t>(kCubeUnpackRows);
    bytes += CodecUbBytes(s.head_size, unpack_rows, s.affine_codec, s.gather_lut);

    // The raw-query prologue's UB half: four kScratchRows scratch planes, the sign vector and its own
    // single-row codec. Its tile and H16 staging live in L1 and L0, which this budget does not cover.
    if (s.raw_query_prologue) {
        constexpr int64_t kScratchRows =
            static_cast<int64_t>(kCubeTileM) / static_cast<int64_t>(kVectorSubcoresPerBlock);
        const int64_t scratch = kScratchRows * s.head_size;
        bytes += UbAlloc(scratch * s.scalar_bytes);                         // inBuf_
        bytes += UbAlloc(scratch * kFloat);                                 // floatBuf_
        bytes += UbAlloc(scratch * static_cast<int64_t>(sizeof(uint16_t))); // halfBuf_
        bytes += UbAlloc(scratch * kFloat);                                 // pairBuf_
        bytes += UbAlloc(s.head_size * kFloat);                             // signBuf_
        // The rotation codec is never affine: it carries the Pi table image.
        bytes += CodecUbBytes(s.head_size, 1, false, false);
    }

    return bytes;
}

constexpr bool FusedDecodeFitsUb(const FusedDecodeUbShape &s)
{
    return FusedDecodeUbBytes(s) <= kUbPartitionBytes;
}

}
}

#endif
