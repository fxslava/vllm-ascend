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
#include <vector>

#include "sweep_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace shapes {

// Shared with the 950PR shape table; see sweep_shapes.hpp.
constexpr int64_t kFp16ElementsPerBurst = sweeps::kFp16ElementsPerBurst;

constexpr int64_t kSwiGluLastDimMultiple = 32;

constexpr int64_t kKvCacheFractalWidth = 16;

inline std::vector<int64_t> Supported310PBlockSizes() { return {64, 128}; }

constexpr int64_t kAttentionBlockSizeLimit = 128 * 128;

inline bool IsValid310PBlockSize(int64_t block_size, int64_t head_size) {
  return block_size * head_size <= kAttentionBlockSizeLimit;
}

constexpr int64_t kDecodeTokenCount = sweeps::kDecodeTokenCount;

inline std::vector<int64_t> PrefillTokenCounts() { return {32, 128, 512}; }

inline std::vector<int64_t> LinearInputSizes() { return sweeps::LinearInputSizes(); }

inline std::vector<int64_t> LinearOutputSizes() { return sweeps::LinearOutputSizes(); }

inline std::vector<int64_t> RmsNormHiddenSizes() { return sweeps::RmsNormHiddenSizes(); }

inline std::vector<int64_t> TokenCounts() { return sweeps::TokenCounts(); }

inline std::vector<int64_t> BenchmarkTokenCounts() { return {1, 7, 32, 128, 512}; }

constexpr float kRmsNormEpsilon = sweeps::kRmsNormEpsilon;

inline std::vector<int64_t> IntermediateSizes() { return sweeps::IntermediateSizes(); }

inline std::vector<int64_t> RotaryHeadDims() { return {64, 128}; }

constexpr double kRopeThetaDefault = sweeps::kRopeThetaDefault;
constexpr double kRopeThetaExtended = sweeps::kRopeThetaExtended;

constexpr int64_t kMaxPositionEmbeddings = sweeps::kMaxPositionEmbeddings;

struct AttentionHeads {
  const char* label;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_size;
};

inline std::vector<AttentionHeads> GqaConfigurations() {
  return {
      AttentionHeads{"mha_8h_8kv_d128", 8, 8, 128},
      AttentionHeads{"gqa_28h_4kv_d128", 28, 4, 128},
      AttentionHeads{"gqa_32h_8kv_d128", 32, 8, 128},
      AttentionHeads{"gqa_16h_2kv_d64", 16, 2, 64},
  };
}

}  // namespace shapes
}  // namespace test
}  // namespace vllm_ascend
