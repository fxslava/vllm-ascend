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

#include <cstddef>
#include <string>
#include <vector>

namespace vllm_ascend {
namespace test {

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

std::string GoldenLayer3Dir();

bool ReadHalfFile(const std::string& path, size_t expected_elements, std::vector<float>* out,
                  std::string* error);

bool LoadGoldenLayer3(GoldenLayer3* golden, std::string* error);

}
}
