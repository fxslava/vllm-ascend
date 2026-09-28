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

#include <cstdint>
#include <string>
#include <vector>

#include "sweep_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace shapes950 {

inline const char* kSocNamePrefix = "Ascend950PR";

inline bool IsAscend950PrSocName(const std::string& soc_name) { return soc_name.rfind(kSocNamePrefix, 0) == 0; }

// Shared with the 310P shape table; see sweep_shapes.hpp.
constexpr int64_t kFp16ElementsPerBurst = sweeps::kFp16ElementsPerBurst;

constexpr int64_t kMaxRotaryHeadDim = 1024;

constexpr int64_t kRotaryHalfModeDimMultiple = 2;

inline std::vector<int64_t> FiaKeyCacheView(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
                                            int64_t head_size) {
  return {num_blocks, block_size, num_kv_heads * head_size};
}

constexpr int64_t kDefaultBlockSize = 128;

constexpr int64_t kFiaUnboundedTokens = 2147483647;

constexpr int64_t kFiaSparseModeNone = 0;
constexpr int64_t kFiaInnerPreciseDefault = 1;

constexpr int64_t kFiaQueryQuantModeNone = 0;

constexpr int64_t kFiaPseTypeDefault = 0;

constexpr int64_t kTokens = 1;
constexpr int64_t kHidden = 2048;
constexpr int64_t kIntermediate = 6144;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 256;
constexpr int64_t kQDim = kNumHeads * kHeadDim;
constexpr int64_t kKvDim = kNumKvHeads * kHeadDim;

constexpr int64_t kRotaryDim = 64;

constexpr float kRmsNormEps = sweeps::kRmsNormEpsilon;

constexpr float kAttentionScale = 0.0625f;

constexpr int64_t kBlockSize = kDefaultBlockSize;
constexpr int64_t kNumBlocks = 4;
constexpr int32_t kPhysicalBlock = 2;
constexpr int64_t kContextLen = 1;
constexpr int64_t kMaxBlocksPerSeq = 1;

constexpr int64_t kDecodeTokenCount = sweeps::kDecodeTokenCount;

inline std::vector<int64_t> TokenCounts() { return sweeps::TokenCounts(); }

inline std::vector<int64_t> RmsNormHiddenSizes() { return sweeps::RmsNormHiddenSizes(); }

inline std::vector<int64_t> IntermediateSizes() { return sweeps::IntermediateSizes(); }

inline std::vector<int64_t> LinearInputSizes() { return sweeps::LinearInputSizes(); }

inline std::vector<int64_t> LinearOutputSizes() { return sweeps::LinearOutputSizes(); }

inline std::vector<int64_t> RotaryHeadDims() { return {64, 128, 256}; }

constexpr double kRopeThetaDefault = sweeps::kRopeThetaDefault;
constexpr double kRopeThetaExtended = sweeps::kRopeThetaExtended;
constexpr int64_t kMaxPositionEmbeddings = sweeps::kMaxPositionEmbeddings;

}  // namespace shapes950
}  // namespace test
}  // namespace vllm_ascend
