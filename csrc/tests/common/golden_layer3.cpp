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

#include "golden_layer3.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>

#include "ascend950_shapes.hpp"
#include "fp16.hpp"

#ifndef VLLM_ASCEND_GOLDEN_LAYER3_DIR
#define VLLM_ASCEND_GOLDEN_LAYER3_DIR ""
#endif

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

// std::getenv trips MSVC's C4996 and this suite can be built warnings-as-errors,
// so the sanctioned _dupenv_s is used there and plain getenv everywhere else.
std::string EnvOrEmpty(const char* name) {
#ifdef _MSC_VER
  char* value = nullptr;
  size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    return std::string();
  }
  const std::string result(value);
  std::free(value);
  return result;
#else
  const char* value = std::getenv(name);
  return (value != nullptr) ? std::string(value) : std::string();
#endif
}

}  // namespace

std::string GoldenLayer3Dir() {
  const std::string env = EnvOrEmpty("QWEN_GOLDEN_LAYER3_DIR");
  if (!env.empty()) {
    return env;
  }
  return VLLM_ASCEND_GOLDEN_LAYER3_DIR;
}

bool ReadHalfFile(const std::string& path, size_t expected_elements, std::vector<float>* out,
                  std::string* error) {
  std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
  if (!file) {
    *error = "cannot open " + path;
    return false;
  }

  const std::streamsize size_bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  const std::streamsize expected_bytes = static_cast<std::streamsize>(expected_elements * sizeof(uint16_t));
  if (size_bytes != expected_bytes) {
    // An unfetched LFS pointer is a ~130 byte text file whose first line is
    // "version https://git-lfs.github.com/spec/v1". Recognising it turns a
    // clone that never ran `git lfs pull` into a skip that names the fix.
    char head[8] = {0};
    file.read(head, static_cast<std::streamsize>(sizeof(head) - 1));
    if (file.gcount() >= 7 && std::string(head) == "version") {
      *error = path + " is an unfetched Git LFS pointer; run `git lfs pull`";
    } else {
      *error = path + " is " + std::to_string(static_cast<long long>(size_bytes)) + " bytes, expected " +
               std::to_string(static_cast<long long>(expected_bytes));
    }
    return false;
  }

  std::vector<uint16_t> bits(expected_elements);
  file.read(reinterpret_cast<char*>(bits.data()), expected_bytes);
  if (file.gcount() != expected_bytes) {
    *error = "short read from " + path;
    return false;
  }

  out->resize(expected_elements);
  for (size_t i = 0; i < expected_elements; ++i) {
    (*out)[i] = HalfBitsToFloat(bits[i]);
  }
  return true;
}

bool LoadGoldenLayer3(GoldenLayer3* golden, std::string* error) {
  const std::string dir = GoldenLayer3Dir();
  if (dir.empty()) {
    *error =
        "no golden dump directory: VLLM_ASCEND_GOLDEN_LAYER3_DIR was not defined at compile time and "
        "QWEN_GOLDEN_LAYER3_DIR is not set";
    return false;
  }

  struct Entry {
    const char* file;
    size_t elements;
    std::vector<float>* target;
  };

  const Entry entries[] = {
      {"input_x.bin", static_cast<size_t>(s::kTokens * s::kHidden), &golden->input_x},
      {"input_norm_gamma.bin", static_cast<size_t>(s::kHidden), &golden->input_norm_gamma},
      {"post_attn_norm_gamma.bin", static_cast<size_t>(s::kHidden), &golden->post_attn_norm_gamma},
      {"w_q.bin", static_cast<size_t>(s::kQDim * s::kHidden), &golden->w_q},
      {"w_k.bin", static_cast<size_t>(s::kKvDim * s::kHidden), &golden->w_k},
      {"w_v.bin", static_cast<size_t>(s::kKvDim * s::kHidden), &golden->w_v},
      {"w_gate_attn.bin", static_cast<size_t>(s::kQDim * s::kHidden), &golden->w_gate_attn},
      {"w_out.bin", static_cast<size_t>(s::kHidden * s::kQDim), &golden->w_out},
      {"w_gate.bin", static_cast<size_t>(s::kIntermediate * s::kHidden), &golden->w_gate},
      {"w_up.bin", static_cast<size_t>(s::kIntermediate * s::kHidden), &golden->w_up},
      {"w_down.bin", static_cast<size_t>(s::kHidden * s::kIntermediate), &golden->w_down},
      {"cos_tab_d64.bin", static_cast<size_t>(s::kTokens * s::kRotaryDim), &golden->cos_tab},
      {"sin_tab_d64.bin", static_cast<size_t>(s::kTokens * s::kRotaryDim), &golden->sin_tab},
      {"tap_norm1.bin", static_cast<size_t>(s::kTokens * s::kHidden), &golden->tap_norm1},
      {"tap_qkv.bin", static_cast<size_t>(s::kTokens * (s::kQDim + 2 * s::kKvDim)), &golden->tap_qkv},
      {"tap_rope_q.bin", static_cast<size_t>(s::kTokens * s::kQDim), &golden->tap_rope_q},
      {"tap_rope_k.bin", static_cast<size_t>(s::kTokens * s::kKvDim), &golden->tap_rope_k},
      {"tap_attn_out.bin", static_cast<size_t>(s::kTokens * s::kHidden), &golden->tap_attn_out},
      {"tap_norm2.bin", static_cast<size_t>(s::kTokens * s::kHidden), &golden->tap_norm2},
      {"tap_swiglu.bin", static_cast<size_t>(s::kTokens * s::kIntermediate), &golden->tap_swiglu},
      {"golden_output.bin", static_cast<size_t>(s::kTokens * s::kHidden), &golden->golden_output},
  };

  for (const Entry& entry : entries) {
    if (!ReadHalfFile(dir + "/" + entry.file, entry.elements, entry.target, error)) {
      return false;
    }
  }
  return true;
}

}  // namespace test
}  // namespace vllm_ascend
