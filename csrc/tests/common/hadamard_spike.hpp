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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vllm_ascend {

void sim_hadamard_hybrid_impl(void *stream, void *input, void *h16, void *output, uint32_t headDim,
                              uint32_t numVectors, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim);

void sim_hadamard_aiv_impl(void *stream, void *input, void *tables, void *output, uint32_t headDim,
                           uint32_t numVectors, float invSqrtDim);

namespace test {
namespace hadamard_spike {

constexpr int64_t kTile = 16;

constexpr int64_t kEarlyStages = 3;

constexpr int64_t kMaxChunkElements = 4096;

enum HadamardVariant : uint32_t {
  kHybridSingleMmad = 0x0u,
  kHybridHiLo = 0x1u,
  kHybridDualDst = 0x2u,
  kHybridLockstep = 0x4u,
};

constexpr uint16_t kHalfOne = 0x3C00;
constexpr uint16_t kHalfMinusOne = 0xBC00;

inline std::vector<uint16_t> Hadamard16Half() {
  std::vector<uint16_t> h(static_cast<size_t>(kTile * kTile));
  for (int64_t i = 0; i < kTile; ++i) {
    for (int64_t j = 0; j < kTile; ++j) {
      unsigned bits = static_cast<unsigned>(i & j);
      int parity = 0;
      while (bits != 0) {
        parity ^= static_cast<int>(bits & 1u);
        bits >>= 1;
      }
      h[static_cast<size_t>(i * kTile + j)] = parity ? kHalfMinusOne : kHalfOne;
    }
  }
  return h;
}

inline std::vector<int32_t> EarlyStageTables(int64_t dim) {
  constexpr int32_t kWord = 4;
  std::vector<int32_t> tables;
  tables.reserve(static_cast<size_t>(kEarlyStages * 2 * dim));
  for (int64_t stage = 0; stage < kEarlyStages; ++stage) {
    const int64_t stride = static_cast<int64_t>(1) << stage;
    for (int64_t c = 0; c < dim; ++c) {
      const float sign = static_cast<float>(1 - 2 * ((c / stride) & 1));
      int32_t bits = 0;
      std::memcpy(&bits, &sign, sizeof(bits));
      tables.push_back(bits);
    }
    for (int64_t c = 0; c < dim; ++c) {
      tables.push_back(static_cast<int32_t>(kWord * (c ^ stride)));
    }
  }
  return tables;
}

inline int64_t HadamardVectorsPerChunk(int64_t dim, int64_t num_vectors) {
  const int64_t fits = kMaxChunkElements / dim;
  const int64_t chunk = fits < num_vectors ? fits : num_vectors;
  return chunk < 1 ? 1 : chunk;
}

constexpr int64_t kPipelineVectorsPerChunk = 4;

inline bool HadamardDualDstApplies(int64_t dim, int64_t num_vectors) {
  return HadamardVectorsPerChunk(dim, num_vectors) % 2 == 0;
}

inline float InvSqrtDim(int64_t dim) {
  return 1.0f / std::sqrt(static_cast<float>(dim));
}

inline std::vector<float> SyntheticBatch(int64_t dim, int64_t num_vectors, uint32_t seed = 0x9E3779B9u) {
  std::vector<float> x(static_cast<size_t>(dim * num_vectors));
  uint32_t state = seed;
  for (size_t i = 0; i < x.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    x[i] = static_cast<float>(static_cast<double>(state >> 8) / 8388608.0 - 1.0);
  }
  return x;
}

struct Deviation {
  double max_abs = 0.0;
  double rms = 0.0;
  size_t worst_at = 0;
};

inline Deviation Compare(const std::vector<float>& got, const std::vector<float>& want) {
  Deviation d;
  double sumsq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    sumsq += err * err;
    if (err > d.max_abs) {
      d.max_abs = err;
      d.worst_at = i;
    }
  }
  d.rms = std::sqrt(sumsq / static_cast<double>(want.size()));
  return d;
}

inline std::string CaseLabel(int64_t dim, int64_t num_vectors, const char* variant) {
  return "d" + std::to_string(dim) + "_v" + std::to_string(num_vectors) + "_" + variant;
}

}
}
}
