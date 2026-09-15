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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_audit {

constexpr int64_t kPrefillChunkTokens = 2048;

inline int64_t EnvInt64(const char* name, int64_t fallback, int64_t minimum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  const long long parsed = std::strtoll(raw, nullptr, 10);
  return parsed < minimum ? minimum : static_cast<int64_t>(parsed);
}

struct ModelSpec {
  const char* key;
  const char* label;
  int64_t head_size;
  int64_t num_heads;
  int64_t num_kv_heads;
  bool folds_output;
  const char* note;
};

inline int64_t GlmHeadSize() {
  const int64_t d = EnvInt64("ASCEND_BENCH_TQ_AUDIT_GLM_D", 128, 64);
  if (d != 128 && d != 256) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_AUDIT_GLM_D=%lld is neither 128 nor 256; using 128\n",
                static_cast<long long>(d));
    return 128;
  }
  return d;
}

inline std::vector<ModelSpec> Models() {
  return {
      ModelSpec{"qwen35", "Qwen3.5-9B", 128, 4, 1, false,
                "attn_output_gate: sigmoid(gate) * context sits between attention and o_proj, "
                "so W_o cannot absorb Pi and the O de-rotation is measured"},
      ModelSpec{"dsv4", "DeepSeek-V4-Flash", 256, 16, 1, true,
                "MLA, decoupled latent KV; the 16:1 group exactly fills the Cube's M=16 fractal; "
                "W_o folded offline"},
      ModelSpec{"glm52", "GLM-5.2-744B", GlmHeadSize(), 8, 1, true,
                "ultra-wide GQA; W_o folded offline"},
  };
}

enum class PathMode { kCube, kAiv };

inline const char* PathLabel(PathMode path) { return path == PathMode::kCube ? "Cube" : "AIV"; }

inline PathMode SelectPath(const ModelSpec& model) {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_AUDIT_PATH");
  const std::string forced = raw == nullptr ? std::string() : std::string(raw);
  if (forced == "cube") {
    return PathMode::kCube;
  }
  if (forced == "aiv") {
    return PathMode::kAiv;
  }
  const int64_t group = model.num_heads / model.num_kv_heads;
  return group >= turboquant_host::kCubeTileM ? PathMode::kCube : PathMode::kAiv;
}

inline int64_t PrefillChunk(int64_t seq_len) {
  const int64_t chunk = EnvInt64("ASCEND_BENCH_TQ_AUDIT_CHUNK", kPrefillChunkTokens, 1);
  return std::min(seq_len, chunk);
}

}
}
}
