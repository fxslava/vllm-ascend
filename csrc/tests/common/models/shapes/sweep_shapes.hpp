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

// Sweep widths, tolerances and rope settings shared verbatim by the Ascend 950PR
// (ascend950_shapes.hpp) and Ascend 310P (qwen_shapes.hpp) shape tables. The
// per-part headers alias these so existing call sites keep their names; values
// that genuinely differ per part stay in the part headers.

#include <cstdint>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace sweeps {

// The MTE burst width in fp16 elements every layer width must align to.
constexpr int64_t kFp16ElementsPerBurst = 16;

constexpr float kRmsNormEpsilon = 1e-6f;

constexpr double kRopeThetaDefault = 10000.0;
constexpr double kRopeThetaExtended = 1000000.0;
constexpr int64_t kMaxPositionEmbeddings = 4096;

constexpr int64_t kDecodeTokenCount = 1;

inline std::vector<int64_t> TokenCounts() { return {1, 7, 32, 128}; }

inline std::vector<int64_t> RmsNormHiddenSizes() { return {1536, 2048, 4096, 8192}; }

inline std::vector<int64_t> IntermediateSizes() { return {4096, 8960, 11008, 14336}; }

inline std::vector<int64_t> LinearInputSizes() { return {2048, 4096}; }

inline std::vector<int64_t> LinearOutputSizes() { return {2048, 4096, 11008}; }

}  // namespace sweeps
}  // namespace test
}  // namespace vllm_ascend
