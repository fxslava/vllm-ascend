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

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turboquant_launch.hpp"
#include "turboquant_mirrored_cache.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_host {

constexpr vllm_ascend::turboquant::TurboQuantMode kFusedMode = vllm_ascend::turboquant::TurboQuantMode::KV4_FP8;
constexpr uint32_t kFusedSeed = 0xF05Eu;
constexpr float kFusedOutputSentinel = -1234.5f;

struct FusedShape {
  int64_t num_heads = 4;
  int64_t num_kv_heads = 2;
  int64_t head_size = 256;
  int64_t block_size = kCubeTileRows;
  int64_t context_len = kCubeTileRows;
  int64_t pool_factor = 4;
};

struct FusedRun {
  std::vector<float> output;
  std::vector<Half> raw;
  int64_t num_splits = 1;
  uint32_t block_dim = 0;
  uint32_t heads_per_task = 0;
  int64_t launches = 0;
  size_t untouched = 0;
  double host_s = 0.0;
};

struct FusedAgreement {
  size_t differing = 0;
  size_t compared = 0;
  double max_abs = 0.0;
};

inline double FusedCosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (a.size() != b.size() || na <= 0.0 || nb <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

inline FusedAgreement CompareValues(const std::vector<float>& a, const std::vector<float>& b) {
  FusedAgreement agreement;
  agreement.compared = std::min(a.size(), b.size());
  for (size_t i = 0; i < agreement.compared; ++i) {
    if (a[i] != b[i]) {
      ++agreement.differing;
      agreement.max_abs = std::max(agreement.max_abs, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
  }
  if (a.size() != b.size()) {
    agreement.differing += std::max(a.size(), b.size()) - agreement.compared;
  }
  return agreement;
}

class FusedCubeScenario {
 public:
  FusedCubeScenario(const FusedShape& shape, aclrtStream stream, int64_t aiv_num)
      : shape_(shape), stream_(stream), aiv_num_(aiv_num) {
    const int64_t d = shape.head_size;
    blocks_per_seq_ = CeilDiv(shape.context_len, shape.block_size);
    const int64_t num_blocks = blocks_per_seq_ * shape.pool_factor;

    DeterministicRandom rng(kFusedSeed + static_cast<uint32_t>(d + shape.context_len));
    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks));
    block_table_.assign(permutation.begin(), permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq_));
    cache_ = BuildMirroredKvCache(rng, shape.context_len, num_blocks, shape.block_size, shape.num_kv_heads, d,
                                  block_table_);
    const std::vector<float> query = rng.NormalHalfExact(static_cast<size_t>(shape.num_heads * d), 0.0f, 1.0f);

    scale_ = 1.0f / std::sqrt(static_cast<float>(d));
    query_ = DeviceBuffer::FromHost(FloatToHalf(query));
    pi_signs_ = DeviceBuffer::FromHost(PiSigns(d));
    h16_ = DeviceBuffer::FromHost(Hadamard16Half());
    rot_tables_ = DeviceBuffer::FromHost(CodecTables(d, 1));
    mode_tables_ = DeviceBuffer::FromHost(ModeTables(kFusedMode, d, kUnpackRows, kCubeTileRows));
    key_cache_ = DeviceBuffer::FromHost(cache_.key_packed);
    value_cache_ = DeviceBuffer::FromHost(cache_.value_packed);
    scale_plane_ = DeviceBuffer::FromHost(cache_.scales);
    block_tables_ = DeviceBuffer::FromHost(block_table_);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(1, static_cast<int32_t>(shape.context_len)));
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(shape.num_heads * d));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(shape.num_heads * d));

    RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                query_rot_.get(), 1, shape.num_heads, d, aiv_num_);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    query_rot_host_ = query_rot_.ToHost<float>();
  }

  FusedRun RunFused(int64_t fused_context_limit = kFusedContextLimit) { return Run(false, fused_context_limit); }

  // The same grid through the test-only instance that keeps every vector barrier: the bit-exact A/B
  // reference for the barrier-free kernel.
  FusedRun RunBarriered(int64_t fused_context_limit = kFusedContextLimit) { return Run(true, fused_context_limit); }

  FusedRun Run(bool barriered, int64_t fused_context_limit) {
    const int64_t d = shape_.head_size;
    const FusedDecodeGrid grid = PlanFusedDecode(1, shape_.num_heads, shape_.num_kv_heads, d, blocks_per_seq_,
                                                 shape_.block_size, aiv_num_, fused_context_limit);
    DeviceBuffer workspace = DeviceBuffer::Empty<float>(grid.workspace_floats);
    FusedRun run;
    run.num_splits = grid.num_splits;
    run.block_dim = grid.block_dim;
    run.heads_per_task = grid.heads_per_task;
    run.launches = 1;
    PoisonOutput();
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(d));
    const auto start = std::chrono::steady_clock::now();
    if (barriered) {
      turboquant_mm_fused_decode_barriered_impl(
          AscendType::FP16, stream_, grid.block_dim, query_rot_.get(), key_cache_.get(), value_cache_.get(),
          scale_plane_.get(), block_tables_.get(), context_lens_.get(), mode_tables_.get(), workspace.get(),
          out_.get(), 1, static_cast<uint32_t>(shape_.num_heads), static_cast<uint32_t>(shape_.num_kv_heads),
          static_cast<uint32_t>(d), static_cast<uint32_t>(shape_.block_size), static_cast<uint32_t>(blocks_per_seq_),
          static_cast<uint32_t>(grid.num_splits), grid.heads_per_task, grid.tasks_per_block,
          grid.reduce_tasks_per_block, static_cast<uint32_t>(fused_context_limit), scale_, inv_sqrt_len);
    } else {
      turboquant_mm_fused_decode_impl(
          static_cast<int32_t>(kFusedMode), AscendType::FP16, stream_, grid.block_dim, query_rot_.get(),
          key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
          mode_tables_.get(), workspace.get(), out_.get(), 1, static_cast<uint32_t>(shape_.num_heads),
          static_cast<uint32_t>(shape_.num_kv_heads), static_cast<uint32_t>(d),
          static_cast<uint32_t>(shape_.block_size), static_cast<uint32_t>(blocks_per_seq_),
          static_cast<uint32_t>(grid.num_splits), grid.heads_per_task, grid.tasks_per_block,
          grid.reduce_tasks_per_block, static_cast<uint32_t>(fused_context_limit), scale_, inv_sqrt_len);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    run.host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Collect(&run);
    return run;
  }

  std::vector<float> Reference() const {
    const int64_t d = shape_.head_size;
    const int64_t heads_per_kv = shape_.num_heads / shape_.num_kv_heads;
    std::vector<float> out(static_cast<size_t>(shape_.num_heads * d), 0.0f);
    std::vector<double> logits(static_cast<size_t>(shape_.context_len), 0.0);
    for (int64_t h = 0; h < shape_.num_heads; ++h) {
      const int64_t kv = h / heads_per_kv;
      double max_logit = -1e30;
      for (int64_t t = 0; t < shape_.context_len; ++t) {
        double dot = 0.0;
        for (int64_t c = 0; c < d; ++c) {
          dot += static_cast<double>(query_rot_host_[static_cast<size_t>(h * d + c)]) *
                 static_cast<double>(cache_.key[static_cast<size_t>((t * shape_.num_kv_heads + kv) * d + c)]);
        }
        logits[static_cast<size_t>(t)] = dot * static_cast<double>(scale_);
        max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
      }
      double denom = 0.0;
      for (int64_t t = 0; t < shape_.context_len; ++t) {
        logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
        denom += logits[static_cast<size_t>(t)];
      }
      for (int64_t t = 0; t < shape_.context_len; ++t) {
        const double weight = logits[static_cast<size_t>(t)] / denom;
        for (int64_t c = 0; c < d; ++c) {
          out[static_cast<size_t>(h * d + c)] += static_cast<float>(
              weight * static_cast<double>(cache_.value[static_cast<size_t>((t * shape_.num_kv_heads + kv) * d + c)]));
        }
      }
    }
    return out;
  }

  int64_t blocks_per_seq() const { return blocks_per_seq_; }

 private:
  void PoisonOutput() {
    out_ = DeviceBuffer::FromHost(
        FloatToHalf(std::vector<float>(static_cast<size_t>(shape_.num_heads * shape_.head_size), kFusedOutputSentinel)));
  }

  void Collect(FusedRun* run) const {
    run->raw = out_.ToHost<Half>();
    run->output = HalfToFloat(run->raw);
    run->untouched = static_cast<size_t>(std::count(run->output.begin(), run->output.end(), kFusedOutputSentinel));
  }

  FusedShape shape_;
  aclrtStream stream_;
  int64_t aiv_num_ = 0;
  int64_t blocks_per_seq_ = 0;
  float scale_ = 1.0f;
  std::vector<int32_t> block_table_;
  MirroredKvCache cache_;
  std::vector<float> query_rot_host_;
  DeviceBuffer query_, pi_signs_, h16_, rot_tables_, mode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, block_tables_, context_lens_;
  DeviceBuffer query_rot_, out_;
};

}
}
}
