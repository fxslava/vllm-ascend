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
#include <cstring>

#include "../reference/turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace turboquant_host {
namespace {

namespace tq = turboquant_ref;

// Bytes per Gather offset unit. The device gathers 4-byte lanes, so every
// offset table below is a byte offset scaled by four.
constexpr int32_t kWord = 4;

// Reinterprets a float as the int32 the table image carries. The kernel copies
// the image in as int32 and reads the sign lanes back as float, so what has to
// match is the bit pattern, not the numeric value.
int32_t FloatBits(float value) {
  int32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace

std::vector<float> PiSigns(int64_t head_size) {
  // One generator for the whole project: the CPU reference's LCG. Regenerating
  // the sequence here would be a second place for the seed to drift.
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

  // Stages 0, 1, 2 (strides 1, 2, 4): the sign pattern of the butterfly and the
  // byte offset of the XOR partner Gather materialises.
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

  // The nibble split: even channel then odd channel of each packed byte.
  for (int64_t p = 0; p < d / kPackFactor; ++p) {
    tables.push_back(static_cast<int32_t>(kWord * kPackFactor * p));
  }
  for (int64_t p = 0; p < d / kPackFactor; ++p) {
    tables.push_back(static_cast<int32_t>(kWord * kPackFactor * p + kWord));
  }

  // Batched expansion: which packed byte each output channel comes from, and
  // whether it is the low or the high nibble.
  for (int64_t b = 0; b < batch; ++b) {
    tables.push_back(static_cast<int32_t>(kWord * (b / kPackFactor)));
  }
  for (int64_t b = 0; b < batch; ++b) {
    tables.push_back(FloatBits(static_cast<float>(b % kPackFactor)));
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

ReshapeAndCacheGrid PlanReshapeAndCache(int64_t num_tokens, int64_t aiv_num) {
  ReshapeAndCacheGrid grid;
  if (num_tokens <= 0) {
    return grid;
  }
  const int64_t tokens_per_core = CeilDiv(num_tokens, aiv_num);
  grid.tokens_per_core = static_cast<uint32_t>(tokens_per_core);
  grid.block_dim = static_cast<uint32_t>(CeilDiv(num_tokens, tokens_per_core));
  return grid;
}

PagedAttentionGrid PlanPagedAttention(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                      int64_t max_blocks_per_seq, int64_t aiv_num) {
  PagedAttentionGrid grid;
  if (num_tokens <= 0 || num_heads <= 0) {
    return grid;
  }

  const int64_t base_tasks = num_tokens * num_heads;

  // Split the sequence only as far as there are idle cores to absorb it, and
  // never past the number of blocks a sequence actually has - a split with no
  // block in it contributes an empty partial the combine stage still has to
  // read.
  int64_t num_splits = CeilDiv(aiv_num, base_tasks);
  num_splits = std::min(num_splits, std::min<int64_t>(kMaxSequenceSplits, std::max<int64_t>(1, max_blocks_per_seq)));
  num_splits = std::max<int64_t>(num_splits, 1);
  grid.num_splits = num_splits;

  const int64_t partial_stride = head_size + kPartialTail;
  grid.workspace_floats = static_cast<size_t>(base_tasks * num_splits * partial_stride);

  const int64_t split_tasks = base_tasks * num_splits;
  const int64_t split_tasks_per_core = CeilDiv(split_tasks, aiv_num);
  const int64_t combine_tasks_per_core = CeilDiv(base_tasks, aiv_num);

  grid.split_tasks_per_core = static_cast<uint32_t>(split_tasks_per_core);
  grid.combine_tasks_per_core = static_cast<uint32_t>(combine_tasks_per_core);
  grid.split_block_dim = static_cast<uint32_t>(CeilDiv(split_tasks, split_tasks_per_core));
  grid.combine_block_dim = static_cast<uint32_t>(CeilDiv(base_tasks, combine_tasks_per_core));
  return grid;
}

}  // namespace turboquant_host
}  // namespace test
}  // namespace vllm_ascend
