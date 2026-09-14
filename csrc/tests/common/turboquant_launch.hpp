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

// Torch-free binding for the TurboQuant 4-bit KV-cache kernels.
//
// csrc/attention/turboquant/turboquant_torch_adpt.h is the production entry
// point and is unusable from this suite: it includes <ATen/ATen.h>,
// torch/library.h and torch_npu. What it contributes on top of the kernel
// launch is arithmetic -- the scale-plane geometry, the codec table layout, the
// grid and the flash-decoding split count -- and none of it needs a tensor.
// This header restates that arithmetic against plain pointers so a test can
// drive
//
//     turboquant_reshape_and_cache_impl        (one launch)
//     turboquant_paged_attention_impl          (split then combine, two launches)
//
// It is a *mirror*, not a shared source, so the two can drift. Every quantity
// below names the function in turboquant_torch_adpt.h it mirrors, and
// TurboQuantLaunchContract in test_sim_950pr_turboquant_kernels.cpp pins them.
//
// The kernels come from libvllm_ascend_turboquant.so, built by
// ascendc_library() out of the same turboquant_kernels.cpp the wheel builds.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../attention/turboquant/turboquant_mode.h"
#include "../../attention/turboquant/turboquant_rotate_q.h"
#include "../../kernels/types.h"

namespace vllm_ascend {

// Defined in csrc/attention/turboquant/turboquant_kernels.cpp and exported by
// libvllm_ascend_turboquant.so. Declared here rather than included because the
// only header that declares them drags in ATen. A mismatch with the definition
// is a link error, not silent corruption.
void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore,
                                       float invSqrtLen);

// `queryRot` is the fp32 output of turboquant_rotate_q_impl, not the model's
// query: neither split kernel rotates any more. `output` comes back in the
// ROTATED basis -- the combine no longer un-rotates, because production folds
// Pi into the output projection. A test comparing against an unrotated
// reference applies UnrotateHeads below, which is that fold, on the host.
void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                     void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                     void *blockTables, void *contextLens, void *tables, void *workspace,
                                     void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t splitTasksPerCore, uint32_t combineTasksPerCore,
                                     float scale, float invSqrtLen);

/*
 * Pi q for a whole decode step's worth of query vectors, in one launch.
 * Defined in csrc/attention/turboquant/turboquant_rotate_q.cpp.
 *
 * `useCube` selects the Sylvester-factorised path over the vector-only one; the
 * host decides with turboquant::PlanRotateQ rather than the kernel, so the plan
 * that is validated is the plan that runs. `h16` is only read by the Cube path
 * and `rotTables` only by the vector one, but both are always passed -- a null
 * for the unused one would be a second thing to keep in sync.
 */
void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                              void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                              uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                              uint32_t variant, float invSqrtLen);

// The reduction alone, shared by the Cube split. Reads the workspace and nothing
// else, and writes a rotated output exactly as turboquant_paged_attention_impl.
void turboquant_paged_attention_combine_impl(AscendType type, void *stream, uint32_t blockDim, void *workspace,
                                             void *output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,
                                             uint32_t numSplits, uint32_t tasksPerCore);

// Defined in csrc/attention/turboquant/turboquant_mm_kernels.cpp: the
// multi-mode, Cube-native path. `mode` is the TurboQuantMode enumerator value
// (3, 4 or 5); the launcher picks the entry point, because the mode is a
// template parameter on the device and not a runtime branch.
void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *key,
                                          void *value, void *keyCache, void *valueCache, void *scaleCache,
                                          void *slotMapping, void *piSigns, void *rotTables, void *modeTables,
                                          uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                          uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen);

void turboquant_mm_decode_split_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *modeTables,
                                     void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen);

// The kv4fp8 split cut at `stage`, a DecodeAblationStage value. Same arguments
// as turboquant_mm_decode_split_impl less the mode; stage 5 launches the
// shipping kernel itself. Defined in turboquant_mm_kernels.cpp only when the
// library is built with VLLM_ASCEND_TQ_DECODE_ABLATION, which csrc/tests does
// and the wheel does not.
void turboquant_mm_decode_ablation_impl(int32_t stage, AscendType type, void *stream, uint32_t blockDim,
                                        void *queryRot,
                                        void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                        void *contextLens, void *modeTables,
                                        void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                        uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                        uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen);

/*
 * The physical fp16 baseline the benchmark compares every quantised Cube rate
 * against -- kv4fp8 and kv5fp8 both: the same Cube decode over an unquantised
 * fp16 paged cache. Split then combine, two launches on one stream.
 */
void turboquant_fp16_decode_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                 void *query, void *keyCache, void *valueCache, void *blockTables,
                                 void *contextLens, void *workspace, void *output, uint32_t numTokens,
                                 uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                 uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t splitTasksPerCore,
                                 uint32_t combineTasksPerCore, float scale);

// A bare fp8 Cube GEMM whose fractal parameters are arguments, so a test can
// pin them numerically instead of asserting a guess. See
// csrc/tests/sim/test_sim_950pr_cube_gemm.cpp.
void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant);

namespace test {
namespace turboquant_host {

// --- constants mirrored from turboquant_torch_adpt.h ------------------------

// Bounds the flash-decoding workspace. Mirrors turboquant_adpt::kMaxSequenceSplits.
constexpr int64_t kMaxSequenceSplits = 8;
// fp32 words appended to every partial accumulator: one 32B block for the
// running max and a second for the running sum. Mirrors
// turboquant_adpt::kPartialTail.
constexpr int64_t kPartialTail = 16;
// fp32 lanes in one 32B burst. Mirrors turboquant_adpt::kFp32PerBlock.
constexpr int64_t kFp32PerBlock = 8;
// Rows of a paged block the decode kernel processes per tile. Must equal
// kTileRows in turboquant_kernels.cpp, because the codec's shuffle tables are
// built for that batch size. Mirrors turboquant_adpt::kTileRows.
constexpr int64_t kTileRows = 16;
// Two 4-bit codes per byte.
constexpr int64_t kPackFactor = 2;

// Vector core count assumed when the runtime cannot report one. A grid is only
// a work split, so a wrong count changes how many blocks are launched and not
// what the kernels compute. Sized to the smallest 950PR bin.
constexpr int64_t kFallbackVectorCoreNum = 8;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// --- layout, mirrored from turboquant_torch_adpt.h --------------------------

// fp32 words one token occupies in the scale plane: K then V for every kv head,
// padded to a whole 32-byte burst. Mirrors turboquant_adpt::ScaleSlotFloats,
// ScaleSlotFloats() in turboquant_kernels.cpp, turboquant_scale_slot() on the
// Python side and cpu_scale_slot_floats() in the CPU reference.
inline int64_t ScaleSlotFloats(int64_t num_kv_heads) {
  return CeilDiv(2 * num_kv_heads, kFp32PerBlock) * kFp32PerBlock;
}

// Reconstruction levels of the codec, and so the length of the Lloyd-Max
// centroid table the image carries. Mirrors turboquant_adpt::kCodecLevels and
// TurboQuantCodec<4>::kLevels.
constexpr int64_t kCodecLevels = 16;

// Words in the codec's constant-table image. Mirrors
// turboquant_adpt::CodecTableWords and TurboQuantCodec<4>::ConstTableWords.
inline int64_t CodecTableWords(int64_t head_size, int64_t batch_rows) {
  return 7 * head_size + 2 * head_size * batch_rows + kCodecLevels;
}

// int8 bytes in the packed KV cache, [num_blocks, block_size, num_kv_heads, head_size / 2].
inline size_t PackedCacheBytes(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads, int64_t head_size) {
  return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(head_size / kPackFactor);
}

// fp32 words in the scale plane, [num_blocks, block_size, scale_slot].
inline size_t ScalePlaneFloats(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads) {
  return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) *
         static_cast<size_t>(ScaleSlotFloats(num_kv_heads));
}

// --- host-built constant data ----------------------------------------------

// The +-1 diagonal of Pi, as the fp32 vector the kernel reads. The C++ host
// reference (reference/turbo_quant_cpu.h::cpu_pi_sign_vector) and the Python
// builder (turboquant_pi_signs) generate the same sequence from the same
// explicit LCG; this returns it in the float layout piSigns wants.
std::vector<float> PiSigns(int64_t head_size);

// The codec's constant-table image, in 4-byte words.
//
//   [0, 6D)          sign_[s] then xorOffset_[s], for strides 1, 2, 4
//   [6D, 7D)         evenOffset_ then oddOffset_, D/2 words each
//   [7D, 7D + B)     expandOffset_
//   [7D + B, +B)     oddSelect_                       (B = D * batch_rows)
//   [7D + 2B, +16)   centroid_, the Lloyd-Max reconstruction levels
//
// sign_, oddSelect_ and centroid_ are fp32 bit patterns; the offset tables are
// uint32 byte offsets for Gather. The C++ mirror of turboquant_codec_tables()
// in vllm_ascend/attention/turboquant_v1.py.
std::vector<int32_t> CodecTables(int64_t head_size, int64_t batch_rows);

// --- grid arithmetic, mirrored from turboquant_torch_adpt.h -----------------

// Vector core count from aclGetDeviceCapability, or kFallbackVectorCoreNum when
// the runtime declines to answer. `queried` reports which happened so a test can
// print it rather than silently reporting a grid the device never chose.
int64_t VectorCoreNum(bool *queried);

// The grid npu_turboquant_reshape_and_cache would have launched.
struct ReshapeAndCacheGrid {
  uint32_t block_dim = 0;
  uint32_t tokens_per_core = 0;
};

ReshapeAndCacheGrid PlanReshapeAndCache(int64_t num_tokens, int64_t aiv_num);

// The two grids and the workspace npu_turboquant_paged_attention would have
// used. The split stage has num_splits times as many tasks as the combine
// stage. Mirrors turboquant_adpt::PagedAttentionPlan and PlanPagedAttention().
struct PagedAttentionGrid {
  uint32_t split_block_dim = 0;
  uint32_t combine_block_dim = 0;
  uint32_t split_tasks_per_core = 0;
  uint32_t combine_tasks_per_core = 0;
  int64_t num_splits = 1;
  // fp32 words of workspace the split stage writes and the combine stage reads:
  // base_tasks * num_splits * (head_size + kPartialTail).
  size_t workspace_floats = 0;
};

PagedAttentionGrid PlanPagedAttention(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                      int64_t max_blocks_per_seq, int64_t aiv_num);

// --- the output basis ------------------------------------------------------

/*
 * Take a decode output out of the rotated basis: Pi applied to every
 * `head_size`-wide row of `rotated`, which is [tokens * heads * head_size].
 *
 * This is what the folded output projection does in production -- W_o' = W_o
 * (I_H (x) Pi), so W_o' O~ = W_o (Pi O~) = W_o O -- restated on the host so a
 * test can keep comparing attention contexts against an unrotated reference.
 * Uses the CPU reference's own cpu_apply_pi, so the sign vector cannot drift
 * from the one the write path rotated with.
 */
std::vector<float> UnrotateHeads(std::vector<float> rotated, int64_t head_size);

/*
 * UB the combine kernel asks InitBuffer for, per block, in bytes. A mirror of
 * TurboQuantPagedAttentionCombine::Init in turboquant_kernels.cpp, kept with the
 * footprint of the kernel it replaced so the saving is a number with a test
 * behind it rather than a claim in a comment.
 *
 * `unrotation_*` is what the combine held only to apply Pi on the way out, and
 * released when that moved into the output projection's weights:
 *
 *   codec tables   TurboQuantCodec<4>::ConstTableWords(D, kTileRows) words
 *   codec scratch  TurboQuantCodec<4>::WorkBufferWords(D, kTileRows) words
 *   signs          D fp32 words, the Pi diagonal
 *   ping-pong      D fp32 words, ApplyPi's second vector in accBuf_
 */
struct CombineUbFootprint {
  size_t out_queue = 0;
  size_t accumulators = 0;
  size_t state = 0;
  size_t partial = 0;
  size_t broadcast = 0;
  size_t unrotation_codec_tables = 0;
  size_t unrotation_codec_scratch = 0;
  size_t unrotation_signs = 0;
  size_t unrotation_ping_pong = 0;

  size_t Current() const { return out_queue + accumulators + state + partial + broadcast; }
  size_t Released() const {
    return unrotation_codec_tables + unrotation_codec_scratch + unrotation_signs + unrotation_ping_pong;
  }
  size_t Previous() const { return Current() + Released(); }
};

// `scalar_bytes` is sizeof the output element: 2 for fp16 and bf16.
CombineUbFootprint PlanCombineUb(int64_t head_size, int64_t scalar_bytes);

// Words in TurboQuantCodec<4>'s uninitialised scratch buffer. Mirrors
// TurboQuantCodec<4>::WorkBufferWords; no kernel reads it from the host.
inline int64_t CodecWorkBufferWords(int64_t head_size, int64_t batch_rows) {
  constexpr int64_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;
  constexpr int64_t kBinLanes = 8;
  return 2 * head_size * batch_rows + head_size + kBrcbDstLanes + kFp32PerBlock + kBinLanes * head_size;
}

// --- multi-mode, Cube-native path -------------------------------------------
//
// The layout half of csrc/attention/turboquant/turboquant_mode.h is free of
// AscendC types, so this header includes it and sizes buffers from the same
// arithmetic the device indexes with. What still has to be mirrored is the
// table *image*: ModeTables below is that mirror, and
// TurboQuantModeCodec<MODE>::ConstTableWords is the contract it writes to.

// Rows of the packed cache one Cube tile covers. Mirrors kCubeTileRows in
// turboquant_mm_kernels.cpp.
constexpr int64_t kCubeTileRows = 64;
// Rows the codec expands per Unpack call on the codebook path, and therefore
// the batch_rows the mode table image must be built for. Mirrors kUnpackRows
// there. The affine path is byte-major and needs no image; see ModeTables.
constexpr int64_t kUnpackRows = 8;
// Elements of a Cube operand in one C0 block. Mirrors kOperandC0 there and
// TurboQuantCubeMm::kOperandC0.
constexpr int64_t kOperandC0 = 32;
// The GEMM's M dimension. Mirrors kCubeTileM in turboquant_cube_mm.h.
constexpr int64_t kCubeTileM = 16;

// Packed bytes for one vector slot at `mode`.
int64_t ModePackedBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size);

// int8 bytes in a packed KV cache at `mode`.
size_t ModePackedCacheBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t num_blocks, int64_t block_size,
                            int64_t num_kv_heads, int64_t head_size);

// Words in the mode codec's constant-table image. Mirrors
// TurboQuantModeCodec<MODE>::ConstTableWords(head_size, batch_rows) for a
// codebook rate; one 32-byte block for an affine one, which reads no table.
int64_t ModeTableWords(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows);

// The image itself. For an affine rate (kv4fp8) it is a single zero block and
// nothing below applies: that codec expands with shifts and an Adds, so it has
// no byte-offset table and no codebook, and batch_rows / nz_rows are ignored.
//
//   [0, B)        lowOffset_   uint32 byte offsets   B = head_size * batch_rows
//   [B, 2B)       msbOffset_   uint32 byte offsets
//   [2B, 2B+8)    lowRecip_    fp32 block, radix^-(p mod dpb)
//   [2B+8, +8)    msbRecip_    fp32 block, 2^-(p mod 8)
//   [2B+16, +8)   msbWeight_   fp32 block, 2^(p mod 8)
//   [2B+24, +D)   packOffset_  uint32, the encoder's low-plane gathers
//   [.., +levels) centroid_    the mode's stored codebook
//
// The two offset tables carry the NZ permutation when `nz_rows` is non-zero:
// output position p is then an NZ position within a band of `batch_rows` rows
// of an `nz_rows`-row tile. Pass nz_rows = 0 for plain row-major output, which
// is what the encode path and the host reference want.
std::vector<int32_t> ModeTables(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows,
                                int64_t nz_rows);

// The NZ position of logical (r, c) in a `rows` x `cols` tile of 1-byte Cube
// operands. Mirrors TurboQuantCubeMm::NzOffset; a test pins the two.
inline int64_t NzOffset(int64_t r, int64_t c, int64_t rows) {
  return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
}

// The grid the Cube decode's split stage launches with.
//
// The task is (token, kv_head, split) rather than (token, head, split), so the
// task count is num_kv_heads times smaller than the AIV path's. The workspace
// is the same size either way: it is indexed per head, because the combine
// stage shared with the AIV path reads it that way.
struct CubeDecodeGrid {
  uint32_t split_block_dim = 0;
  uint32_t combine_block_dim = 0;
  uint32_t split_tasks_per_core = 0;
  uint32_t combine_tasks_per_core = 0;
  int64_t num_splits = 1;
  size_t workspace_floats = 0;
};

CubeDecodeGrid PlanCubeDecode(int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                              int64_t max_blocks_per_seq, int64_t aiv_num);

// The 16x16 fp16 Hadamard constant the Cube rotation path multiplies by, as a
// host image ready for DeviceBuffer::FromHost. One definition, shared with the
// torch operator through turboquant::FillHadamard16Half.
std::vector<uint16_t> Hadamard16Half();

/*
 * Rotate a decode step's whole query into `query_rot`, the way
 * npu_turboquant_rotate_q does: plan the grid, pick the path, launch once.
 * Every decode call site in this suite goes through it, so a test cannot
 * accidentally feed a split kernel an unrotated query -- which would not fault,
 * and would read as a codec fidelity failure.
 *
 * Does NOT synchronise: the rotation and the decode belong on one stream, and
 * stream order is what sequences them.
 *
 * `query` is scalar_t, `query_rot` is fp32 with the same element count. Returns
 * the plan so a test can assert which path ran.
 */
vllm_ascend::turboquant::RotateQPlan RotateQuery(void *stream, AscendType type, void *query, void *pi_signs,
                                                 void *h16, void *rot_tables, void *query_rot, int64_t num_tokens,
                                                 int64_t num_heads, int64_t head_size, int64_t aiv_num,
                                                 bool input_exact_in_half);

}  // namespace turboquant_host
}  // namespace test
}  // namespace vllm_ascend
