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

namespace vllm_ascend {
namespace test {
namespace shapes {

constexpr int64_t kFp16ElementsPerBurst = 16;

constexpr int64_t kSwiGluLastDimMultiple = 32;

constexpr int64_t kKvCacheFractalWidth = 16;

inline std::vector<int64_t> Supported310PBlockSizes() { return {64, 128}; }

constexpr int64_t kAttentionBlockSizeLimit = 128 * 128;

inline bool IsValid310PBlockSize(int64_t block_size, int64_t head_size) {
  return block_size * head_size <= kAttentionBlockSizeLimit;
}

constexpr int64_t kDecodeTokenCount = 1;

inline std::vector<int64_t> PrefillTokenCounts() { return {32, 128, 512}; }

inline std::vector<int64_t> LinearInputSizes() { return {2048, 4096}; }

inline std::vector<int64_t> LinearOutputSizes() { return {2048, 4096, 11008}; }

inline std::vector<int64_t> RmsNormHiddenSizes() { return {1536, 2048, 4096, 8192}; }

inline std::vector<int64_t> TokenCounts() { return {1, 7, 32, 128}; }

inline std::vector<int64_t> BenchmarkTokenCounts() { return {1, 7, 32, 128, 512}; }

constexpr float kRmsNormEpsilon = 1e-6f;

inline std::vector<int64_t> IntermediateSizes() { return {4096, 8960, 11008, 14336}; }

inline std::vector<int64_t> RotaryHeadDims() { return {64, 128}; }

constexpr double kRopeThetaDefault = 10000.0;
constexpr double kRopeThetaExtended = 1000000.0;

constexpr int64_t kMaxPositionEmbeddings = 4096;

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

}
}
}
