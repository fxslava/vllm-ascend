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

#include "turboquant_launch.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../reference/turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace turboquant_host {
namespace {

namespace tq = turboquant_ref;

constexpr int32_t kWord = 4;

int32_t FloatBits(float value) {
  int32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}

std::vector<float> PiSigns(int64_t head_size) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(head_size));
  std::vector<float> out(signs.size());
  for (size_t i = 0; i < signs.size(); ++i) {
    out[i] = static_cast<float>(signs[i]);
  }
  return out;
}

std::vector<int32_t> CodecTables(int64_t head_size, int64_t batch_rows) {
  const int64_t d = head_size;
  const int64_t batch = d * batch_rows;
  std::vector<int32_t> tables;
  tables.reserve(static_cast<size_t>(CodecTableWords(d, batch_rows)));

  for (int stage = 0; stage < 3; ++stage) {
    const int64_t stride = static_cast<int64_t>(1) << stage;
    for (int64_t c = 0; c < d; ++c) {
      const float sign = static_cast<float>(1 - 2 * ((c / stride) & 1));
      tables.push_back(FloatBits(sign));
    }
    for (int64_t c = 0; c < d; ++c) {
      tables.push_back(static_cast<int32_t>(kWord * (c ^ stride)));
    }
  }

  for (int64_t p = 0; p < d / kPackFactor; ++p) {
    tables.push_back(static_cast<int32_t>(kWord * kPackFactor * p));
  }
  for (int64_t p = 0; p < d / kPackFactor; ++p) {
    tables.push_back(static_cast<int32_t>(kWord * kPackFactor * p + kWord));
  }

  for (int64_t b = 0; b < batch; ++b) {
    tables.push_back(static_cast<int32_t>(kWord * (b / kPackFactor)));
  }
  for (int64_t b = 0; b < batch; ++b) {
    tables.push_back(FloatBits(static_cast<float>(b % kPackFactor)));
  }

  for (int level = 0; level < static_cast<int>(kCodecLevels); ++level) {
    tables.push_back(FloatBits(tq::kLloydMaxCentroids[level]));
  }

  return tables;
}

int64_t VectorCoreNum(bool *queried) {
  int64_t aiv_num = 0;
  const bool ok = aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS && aiv_num > 0;
  if (queried != nullptr) {
    *queried = ok;
  }
  return ok ? aiv_num : kFallbackVectorCoreNum;
}

namespace {

namespace tqm = vllm_ascend::turboquant;

struct ModePlanes {
  int32_t low_bits;
  int32_t msb_bits;
  int32_t low_radix;
  int32_t low_per_byte;
};

ModePlanes PlanesOf(tqm::TurboQuantMode mode) {
  switch (mode) {
    case tqm::TurboQuantMode::KV3_FP4:
      return {2, 1, 4, 4};
    case tqm::TurboQuantMode::KV4_FP8:
      return {4, 0, 16, 2};
    default:
      return {4, 1, 16, 2};
  }
}

const float *CentroidsOf(tqm::TurboQuantMode mode) {
  switch (mode) {
    case tqm::TurboQuantMode::KV3_FP4:
      return tqm::TurboQuantModeTraits<tqm::TurboQuantMode::KV3_FP4>::kCentroids;
    case tqm::TurboQuantMode::KV4_FP8:
      return tqm::TurboQuantModeTraits<tqm::TurboQuantMode::KV4_FP8>::kCentroids;
    default:
      return tqm::TurboQuantModeTraits<tqm::TurboQuantMode::KV5_FP8>::kCentroids;
  }
}

int64_t Shift(int32_t digits_per_byte) {
  int64_t shift = 0;
  while ((static_cast<int64_t>(1) << shift) < digits_per_byte) {
    ++shift;
  }
  return shift;
}

}

int64_t ModePackedBytes(tqm::TurboQuantMode mode, int64_t head_size) {
  return tqm::TurboQuantModeConfigOf(mode).PackedBytes(head_size);
}

size_t ModePackedCacheBytes(tqm::TurboQuantMode mode, int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
                            int64_t head_size) {
  return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(ModePackedBytes(mode, head_size));
}

int64_t ModeTableWords(tqm::TurboQuantMode mode, int64_t head_size, int64_t batch_rows) {
  const tqm::TurboQuantModeConfig cfg = tqm::TurboQuantModeConfigOf(mode);
  if (cfg.is_affine) {
    return kFp32PerBlock;
  }
  return 2 * head_size * batch_rows + 3 * kFp32PerBlock + head_size + cfg.levels;
}

std::vector<int32_t> ModeTables(tqm::TurboQuantMode mode, int64_t head_size, int64_t batch_rows, int64_t nz_rows) {
  const ModePlanes planes = PlanesOf(mode);
  const tqm::TurboQuantModeConfig cfg = tqm::TurboQuantModeConfigOf(mode);
  if (cfg.is_affine) {
    return std::vector<int32_t>(static_cast<size_t>(kFp32PerBlock), 0);
  }
  const int64_t d = head_size;
  const int64_t batch = d * batch_rows;
  const int64_t packed_bytes = cfg.PackedBytes(d);
  const int64_t low_bytes = d / planes.low_per_byte;
  const int64_t low_shift = Shift(planes.low_per_byte);

  std::vector<int32_t> tables;
  tables.reserve(static_cast<size_t>(ModeTableWords(mode, d, batch_rows)));

  for (int plane = 0; plane < 2; ++plane) {
    for (int64_t p = 0; p < batch; ++p) {
      int64_t r = p / d;
      int64_t c = p % d;
      if (nz_rows > 0) {
        const int64_t band = batch_rows * kOperandC0;
        const int64_t b = p / band;
        const int64_t rem = p % band;
        r = rem / kOperandC0;
        c = b * kOperandC0 + (rem % kOperandC0);
      }
      const int64_t byte = plane == 0 ? r * packed_bytes + (c >> low_shift)
                                      : r * packed_bytes + low_bytes + (c >> 3);
      tables.push_back(static_cast<int32_t>(kWord * byte));
    }
  }

  for (int64_t lane = 0; lane < kFp32PerBlock; ++lane) {
    float recip = 1.0f;
    for (int64_t j = 0; j < lane % planes.low_per_byte; ++j) {
      recip /= static_cast<float>(planes.low_radix);
    }
    tables.push_back(FloatBits(recip));
  }
  for (int64_t lane = 0; lane < kFp32PerBlock; ++lane) {
    tables.push_back(FloatBits(1.0f / static_cast<float>(static_cast<int64_t>(1) << lane)));
  }
  for (int64_t lane = 0; lane < kFp32PerBlock; ++lane) {
    tables.push_back(FloatBits(static_cast<float>(static_cast<int64_t>(1) << lane)));
  }

  for (int64_t j = 0; j < planes.low_per_byte; ++j) {
    for (int64_t b = 0; b < low_bytes; ++b) {
      tables.push_back(static_cast<int32_t>(kWord * (b * planes.low_per_byte + j)));
    }
  }

  const float *centroids = CentroidsOf(mode);
  for (int64_t level = 0; level < cfg.levels; ++level) {
    tables.push_back(FloatBits(centroids[level]));
  }

  return tables;
}

std::vector<float> UnrotateHeads(std::vector<float> rotated, int64_t head_size) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(head_size));
  const size_t row = static_cast<size_t>(head_size);
  for (size_t base = 0; base + row <= rotated.size(); base += row) {
    tq::cpu_apply_pi(rotated.data() + base, static_cast<int>(head_size), signs.data());
  }
  return rotated;
}

std::vector<uint16_t> Hadamard16Half() {
  std::vector<uint16_t> h(static_cast<size_t>(vllm_ascend::turboquant::kRotateQH16Elements));
  vllm_ascend::turboquant::FillHadamard16Half(h.data());
  return h;
}

vllm_ascend::turboquant::RotateQPlan RotateQuery(void *stream, AscendType type, void *query, void *pi_signs,
                                                 void *h16, void *rot_tables, void *query_rot, int64_t num_tokens,
                                                 int64_t num_heads, int64_t head_size, int64_t aiv_num,
                                                 vllm_ascend::turboquant::RotateQPrecision precision) {
  const int64_t num_vectors = num_tokens * num_heads;
  const vllm_ascend::turboquant::RotateQPlan plan = vllm_ascend::turboquant::PlanRotateQ(
      num_tokens, num_vectors, head_size, vllm_ascend::turboquant::RotateQCoreNum(aiv_num), precision);
  const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
  turboquant_rotate_q_impl(type, stream, plan.block_dim, plan.use_cube, query, pi_signs, h16, rot_tables, query_rot,
                           static_cast<uint32_t>(num_vectors), static_cast<uint32_t>(head_size),
                           plan.vectors_per_block, plan.vectors_per_chunk, plan.variant, inv_sqrt_len);
  return plan;
}

}
}
}
