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

// Loader for the Qwen3.5 layer-3 golden dump in csrc/tests/data/golden_layer3.
//
// Split out of the test that consumes it so the parsing has host-only coverage:
// nothing here touches ACL or allocates device memory.
//
// The files are raw little-endian fp16, C-contiguous, no header, written by
// scripts/dump_qwen35_layer3.py. Linear weights keep the torch
// [out_features, in_features] layout.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vllm_ascend {
namespace test {

// Everything the dump provides, widened to float on the way in.
struct GoldenLayer3 {
  std::vector<float> input_x;
  std::vector<float> input_norm_gamma;
  std::vector<float> post_attn_norm_gamma;
  std::vector<float> w_q, w_k, w_v, w_gate_attn, w_out;
  std::vector<float> w_gate, w_up, w_down;
  std::vector<float> cos_tab, sin_tab;
  std::vector<float> tap_norm1, tap_qkv, tap_rope_q, tap_rope_k;
  std::vector<float> tap_attn_out, tap_norm2, tap_swiglu;
  std::vector<float> golden_output;
};

// QWEN_GOLDEN_LAYER3_DIR if set, otherwise the VLLM_ASCEND_GOLDEN_LAYER3_DIR
// baked in at compile time by csrc/tests/CMakeLists.txt. Empty when neither is
// available, which the loader turns into a skip rather than a crash.
std::string GoldenLayer3Dir();

// Reads `expected_elements` fp16 values and widens them to float. Returns false
// and sets `error` rather than throwing, so a missing or unfetched dump becomes
// a skip with an actionable message.
bool ReadHalfFile(const std::string& path, size_t expected_elements, std::vector<float>* out,
                  std::string* error);

// Loads every file the layer-3 dump contains, checking each length against the
// layer configuration in ascend950_shapes.hpp. A size that does not match is an
// error, not a truncation: a dump regenerated at a different head count would
// otherwise be read as garbage.
bool LoadGoldenLayer3(GoldenLayer3* golden, std::string* error);

}  // namespace test
}  // namespace vllm_ascend
