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
// point, and it is unusable from this suite: it includes <ATen/ATen.h>,
// torch/library.h and torch_npu, which is exactly the dependency the bare-metal
// tests exist to avoid. What that header actually contributes on top of the
// kernel launch is arithmetic - the scale-plane geometry, the codec table
// layout, the grid and the flash-decoding split count - and none of it needs a
// tensor. This header restates that arithmetic against plain pointers and
// std::vector so a test can drive
//
//     turboquant_reshape_and_cache_impl        (one launch)
//     turboquant_paged_attention_impl          (split then combine, two launches)
//
// with the same shapes the production path would have produced.
//
// It is a *mirror*, not a copy of a shared source, so the two can drift. Every
// quantity below names the function in turboquant_torch_adpt.h it mirrors, and
// TurboQuantLaunchContract in test_turboquant_kernels_950pr.cpp pins the ones
// that are pure functions of the shapes against the values written here. The
// codec table image is additionally pinned against the layout contract in
// TurboQuantCodec<4>::ConstTableWords and against the Python builder
// vllm_ascend/attention/turboquant_v1.py::turboquant_codec_tables.
//
// The kernels themselves come from libvllm_ascend_turboquant.so, built by
// ascendc_library() out of the same turboquant_kernels.cpp the wheel builds.
// The two _impl symbols below are its only C++ entry points; everything else it
// exports is a generated aclrtlaunch_* wrapper.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../kernels/types.h"

namespace vllm_ascend {

// Defined in csrc/attention/turboquant/turboquant_kernels.cpp and exported by
// libvllm_ascend_turboquant.so. Declared here rather than included because the
// only header that declares them (turboquant_torch_adpt.h) drags in ATen.
//
// A mismatch with the definition is a link error, not silent corruption: these
// are ordinary C++ symbols with mangled names, unlike the dlsym-resolved aclnn
// operators in aclnn_ops.hpp.
void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore,
                                       float invSqrtLen);

void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                     void *query, void *keyCache, void *valueCache, void *scaleCache,
                                     void *blockTables, void *contextLens, void *piSigns, void *tables,
                                     void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
                                     uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                     uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t splitTasksPerCore,
                                     uint32_t combineTasksPerCore, float scale, float invSqrtLen);

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

// Vector core count assumed when the runtime cannot report one. The simulator
// answers aclGetDeviceCapability for some bins and not others, and a grid is
// only a work split - a wrong count changes how many blocks are launched, not
// what the kernels compute - so a test that cannot ask simply says so and uses
// this. Sized to the smallest 950PR bin so the assumption never over-subscribes.
constexpr int64_t kFallbackVectorCoreNum = 8;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// --- layout, mirrored from turboquant_torch_adpt.h --------------------------

// fp32 words one token occupies in the scale plane: K then V for every kv head,
// padded to a whole 32-byte burst so the scatter never touches an unaligned
// global address. Mirrors turboquant_adpt::ScaleSlotFloats, ScaleSlotFloats()
// in turboquant_kernels.cpp, turboquant_scale_slot() on the Python side and
// cpu_scale_slot_floats() in the CPU reference - five copies of one number,
// which is why TurboQuantLaunchContract checks it against the reference.
inline int64_t ScaleSlotFloats(int64_t num_kv_heads) {
  return CeilDiv(2 * num_kv_heads, kFp32PerBlock) * kFp32PerBlock;
}

// Words in the codec's constant-table image. Mirrors
// turboquant_adpt::CodecTableWords and TurboQuantCodec<4>::ConstTableWords.
inline int64_t CodecTableWords(int64_t head_size, int64_t batch_rows) {
  return 7 * head_size + 2 * head_size * batch_rows;
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
//
// sign_ and oddSelect_ are fp32 bit patterns; the offset tables are uint32 byte
// offsets for Gather. Everything is four bytes wide, so one int32 DataCopy
// moves the lot and the device needs no cast and no arithmetic.
//
// This is the C++ mirror of turboquant_codec_tables() in
// vllm_ascend/attention/turboquant_v1.py. The kernel treats the image as
// read-only and never rewrites a word of it, so building it once per test is
// enough.
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
// stage, and sizing them separately keeps the combine launch from spawning
// cores with nothing to do.
//
// Mirrors turboquant_adpt::PagedAttentionPlan and PlanPagedAttention(), which
// is the single place the production path derives all of this - the operator
// and the host that pre-allocates the workspace both read it from there.
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

}  // namespace turboquant_host
}  // namespace test
}  // namespace vllm_ascend
