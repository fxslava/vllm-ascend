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

// Shapes and platform rules for the Ascend 950PR (A5, DaVinci arch35) leg of
// the suite. Kept separate from qwen_shapes.hpp, whose rules are 310P (v200)
// rules that do not hold here.
//
// The Qwen3.5-2B layer-3 block below mirrors scripts/dump_qwen35_layer3.py,
// which produced csrc/tests/data/golden_layer3.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace shapes950 {

// --- part identification -----------------------------------------------------

// aclrtGetSocName() returns the platform_config SoC_version, e.g.
// "Ascend950PR_9599"; every 950PR bin shares Short_SoC_version=Ascend950, so the
// tests match on the "Ascend950PR" prefix rather than on a bin name.
// Ascend950DT is deliberately NOT accepted: different core counts and
// cube/vector split, and nothing here has been checked against it.
inline const char* kSocNamePrefix = "Ascend950PR";

inline bool IsAscend950PrSocName(const std::string& soc_name) {
  return soc_name.rfind(kSocNamePrefix, 0) == 0;
}

// --- Ascend 950PR (DaVinci arch35) platform rules ----------------------------
//
// Sourced from data/platform_config/Ascend950PR_9599.ini in the CANN toolkit
// and from the ascend950 AICore configs in csrc/*/op_host/*_def.cpp.

// MTE bursts are still 32 bytes, so an fp16 row wants a multiple of 16
// elements. Every width in the layer below satisfies this.
constexpr int64_t kFp16ElementsPerBurst = 16;

// InplacePartialRotaryMul's operator prototype states the head-dim ceiling:
// "For Ascend 950 AI Processor, D should be less or equal to 1024"
// (csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_proto.cpp).
constexpr int64_t kMaxRotaryHeadDim = 1024;

// Same prototype: "In half, interleave and interleave-half mode, D must be a
// multiple of 2. In quarter mode, D must be a multiple of 4."
constexpr int64_t kRotaryHalfModeDimMultiple = 2;

// The generic (non-310P) attention backend allocates
//   AscendAttentionBackend.get_kv_cache_shape() ->
//       (2, num_blocks, block_size, num_kv_heads, head_size)
// so each of the key and value caches is a plain ND
//   [num_blocks, block_size, num_kv_heads, head_size]
// with no fractal trailing axis. See vllm_ascend/attention/attention_v1.py.
constexpr int64_t kKvCacheRank = 4;

// AscendAttentionBackendImpl._get_fia_params reshapes that cache to
//   [num_blocks, block_size, num_kv_heads * head_size]
// before handing it to npu_fused_infer_attention_score with input_layout="TND".
inline std::vector<int64_t> FiaKeyCacheView(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
                                            int64_t head_size) {
  return {num_blocks, block_size, num_kv_heads * head_size};
}

// _get_fia_params uses 128 wherever it has to name a block size itself (the
// PrefillNoCache branch), and otherwise reads it off the cache shape.
constexpr int64_t kDefaultBlockSize = 128;

// torch_npu's two FIA window parameters. attention_v1.py passes
// SWA_INT_MAX = 2147483647 for pre_tokens; the non-causal decode call site
// leaves both at the torch_npu default, which is the same value.
constexpr int64_t kFiaUnboundedTokens = 2147483647;

// sparse_mode 0 is "no sparsity", which is what the non-causal decode call in
// attention_v1.py passes. inner_precise 1 is the torch_npu default.
constexpr int64_t kFiaSparseModeNone = 0;
constexpr int64_t kFiaInnerPreciseDefault = 1;

// The two scalars V5 adds over V2 (ops950::kFusedInferAttentionScoreV5).
// queryQuantMode 0 is "the query is not quantised".
constexpr int64_t kFiaQueryQuantModeNone = 0;

// pseType selects how a positional-encoding shift is combined with the scores.
// This suite passes no pseShift, but the operator still range-checks the value;
// 1 is torch_npu's default for npu_fused_infer_attention_score's pse_type.
// UNVERIFIED on hardware, like the whole V5 argument list.
constexpr int64_t kFiaPseTypeDefault = 1;

// --- Qwen3.5-2B, layer 3 (the first full_attention block) --------------------
//
// Mirrors the constants at the top of scripts/dump_qwen35_layer3.py.

constexpr int64_t kTokens = 1;  // one decode step
constexpr int64_t kHidden = 2048;
constexpr int64_t kIntermediate = 6144;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 256;
constexpr int64_t kQDim = kNumHeads * kHeadDim;     // 2048
constexpr int64_t kKvDim = kNumKvHeads * kHeadDim;  // 512

// partial_rotary_factor 0.25: channels [0, 64) of each 256-wide head rotate and
// [64, 256) pass through untouched. This is the single shape fact that decides
// which rotary operator the suite can use.
constexpr int64_t kRotaryDim = 64;

constexpr float kRmsNormEps = 1e-6f;

// 1/sqrt(256) = 0.0625, exact in fp16, so the scale itself contributes no
// rounding difference between the device and the dump.
constexpr float kAttentionScale = 0.0625f;

// Paging for the golden decode. One token needs one block; a non-zero physical
// block is used so a block table that is ignored reads zeros rather than
// passing by luck.
constexpr int64_t kBlockSize = kDefaultBlockSize;
constexpr int64_t kNumBlocks = 4;
constexpr int32_t kPhysicalBlock = 2;
constexpr int64_t kContextLen = 1;
constexpr int64_t kMaxBlocksPerSeq = 1;

// --- shape sweeps for the operator unit tests --------------------------------
//
// The same values the 310P and CUDA suites sweep (see qwen_shapes.hpp), plus
// the head dim the other two parts cannot run.

// Decode is one token at a time; the projections are therefore GEMV-shaped.
constexpr int64_t kDecodeTokenCount = 1;

inline std::vector<int64_t> TokenCounts() { return {1, 7, 32, 128}; }

inline std::vector<int64_t> RmsNormHiddenSizes() { return {1536, 2048, 4096, 8192}; }

inline std::vector<int64_t> IntermediateSizes() { return {4096, 8960, 11008, 14336}; }

inline std::vector<int64_t> LinearInputSizes() { return {2048, 4096}; }

inline std::vector<int64_t> LinearOutputSizes() { return {2048, 4096, 11008}; }

// Head dims for the rotary unit test. 64 and 128 are what the 310P suite is
// limited to and are kept so the three backends can be compared on the same
// cases; 256 is the Qwen3.5 head dim that the 310P part cannot run at all, and
// is the reason this file exists.
inline std::vector<int64_t> RotaryHeadDims() { return {64, 128, 256}; }

constexpr double kRopeThetaDefault = 10000.0;
constexpr double kRopeThetaExtended = 1000000.0;
constexpr int64_t kMaxPositionEmbeddings = 4096;

}  // namespace shapes950
}  // namespace test
}  // namespace vllm_ascend
