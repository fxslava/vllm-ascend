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
  // Draw independent vector halves and write the cache through the kv4fp8 kernel writer
  // (turboquant_mm_reshape_and_cache_impl) instead of uploading the host-built image.
  bool kernel_writer = false;
};

// What the kernel writer put into GM, against the host encoder of the same levels.
struct WrittenCacheAgreement {
  size_t packed_bytes = 0;
  size_t packed_mismatches = 0;
  size_t swapped_lane_mismatches = 0;
  size_t scale_lanes = 0;
  size_t scale_pad_mismatches = 0;
  double max_scale_rel_err = 0.0;
};

struct FusedRun {
  std::vector<float> output;
  std::vector<Half> raw;
  int64_t num_splits = 1;
  uint32_t block_dim = 0;
  uint32_t heads_per_task = 0;
  uint32_t fused_context_limit = 0;
  int64_t launches = 0;
  size_t untouched = 0;
  uint64_t fnv1a = 0;
  double host_s = 0.0;
  // Raw-query runs only: fp32 words of the in-launch rotation that differ from the rotate_q launch's.
  size_t prologue_mismatches = 0;
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

// FNV-1a over the output's half bit patterns, low byte first: the regression golden of a launch.
inline uint64_t Fnv1a64(const std::vector<Half>& raw) {
  constexpr uint64_t kOffsetBasis = 0xcbf29ce484222325ull;
  constexpr uint64_t kPrime = 0x100000001b3ull;
  constexpr unsigned kByteBits = 8;
  constexpr uint32_t kByteMask = 0xff;
  uint64_t hash = kOffsetBasis;
  for (const Half& value : raw) {
    for (unsigned byte = 0; byte < sizeof(value.bits); ++byte) {
      hash ^= static_cast<uint64_t>((static_cast<uint32_t>(value.bits) >> (byte * kByteBits)) & kByteMask);
      hash *= kPrime;
    }
  }
  return hash;
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
                                  block_table_, !shape.kernel_writer);
    const std::vector<float> query = rng.NormalHalfExact(static_cast<size_t>(shape.num_heads * d), 0.0f, 1.0f);

    scale_ = 1.0f / std::sqrt(static_cast<float>(d));
    query_ = DeviceBuffer::FromHost(FloatToHalf(query));
    pi_signs_ = DeviceBuffer::FromHost(PiSigns(d));
    h16_ = DeviceBuffer::FromHost(Hadamard16Half());
    rot_tables_ = DeviceBuffer::FromHost(CodecTables(d, 1));
    mode_tables_ = DeviceBuffer::FromHost(ModeTables(kFusedMode, d, kUnpackRows, kCubeTileRows));
    if (shape.kernel_writer) {
      WriteThroughKernel();
    } else {
      key_cache_ = DeviceBuffer::FromHost(cache_.key_packed);
      value_cache_ = DeviceBuffer::FromHost(cache_.value_packed);
      scale_plane_ = DeviceBuffer::FromHost(cache_.scales);
    }
    block_tables_ = DeviceBuffer::FromHost(block_table_);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(1, static_cast<int32_t>(shape.context_len)));
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(shape.num_heads * d));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(shape.num_heads * d));

    RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                query_rot_.get(), 1, shape.num_heads, d, aiv_num_);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    query_rot_host_ = query_rot_.ToHost<float>();
  }

  // plan_aiv is the vector-core budget the planner tiles for (0: the device's own); a smaller budget
  // than the device has forces fewer, wider tasks without changing the kernel.
  FusedDecodeGrid Plan(int64_t plan_aiv = 0, FusedSplitPolicy split_policy = FusedSplitPolicy::kFillBlocks) const {
    return PlanFusedDecode(1, shape_.num_heads, shape_.num_kv_heads, shape_.head_size, blocks_per_seq_,
                           shape_.block_size, plan_aiv > 0 ? plan_aiv : aiv_num_, kFusedContextLimit, split_policy);
  }

  FusedRun RunFused(const FusedDecodeGrid& grid) {
    return Launch(grid, [&](void* workspace, float inv_sqrt_len) {
      turboquant_mm_fused_decode_impl(
          static_cast<int32_t>(kFusedMode), AscendType::FP16, stream_, grid.block_dim, query_rot_.get(),
          key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
          mode_tables_.get(), workspace, out_.get(), 1, static_cast<uint32_t>(shape_.num_heads),
          static_cast<uint32_t>(shape_.num_kv_heads), static_cast<uint32_t>(shape_.head_size),
          static_cast<uint32_t>(shape_.block_size), static_cast<uint32_t>(blocks_per_seq_),
          static_cast<uint32_t>(grid.num_splits), grid.heads_per_task, grid.tasks_per_block,
          grid.reduce_tasks_per_block, grid.fused_context_limit, scale_, inv_sqrt_len);
    });
  }

  // The same grid through the raw-query decode: the launch is handed the fp16 query and rotates it itself,
  // into a fresh buffer that is then compared word for word with what rotate_q wrote at setup.
  FusedRun RunFusedRawQuery(const FusedDecodeGrid& grid) {
    DeviceBuffer rotated = DeviceBuffer::FromHost(std::vector<float>(query_rot_host_.size(), kFusedOutputSentinel));
    FusedRun run = Launch(grid, [&](void* workspace, float inv_sqrt_len) {
      turboquant_mm_fused_decode_raw_query_impl(
          static_cast<int32_t>(kFusedMode), AscendType::FP16, stream_, grid.block_dim, query_.get(), pi_signs_.get(),
          rot_tables_.get(), rotated.get(), key_cache_.get(), value_cache_.get(), scale_plane_.get(),
          block_tables_.get(), context_lens_.get(), mode_tables_.get(), workspace, out_.get(), 1,
          static_cast<uint32_t>(shape_.num_heads), static_cast<uint32_t>(shape_.num_kv_heads),
          static_cast<uint32_t>(shape_.head_size), static_cast<uint32_t>(shape_.block_size),
          static_cast<uint32_t>(blocks_per_seq_), static_cast<uint32_t>(grid.num_splits), grid.heads_per_task,
          grid.tasks_per_block, grid.reduce_tasks_per_block, grid.prologue_vectors_per_block,
          grid.fused_context_limit, scale_, inv_sqrt_len);
    });
    const std::vector<float> in_launch = rotated.ToHost<float>();
    for (size_t i = 0; i < query_rot_host_.size(); ++i) {
      run.prologue_mismatches +=
          (i >= in_launch.size() || std::memcmp(&in_launch[i], &query_rot_host_[i], sizeof(float)) != 0) ? 1u : 0u;
    }
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

  // The kernel-written GM cache against the host encoder: the packed planes byte for byte (and, for the
  // diagnosis, with the two nibble lanes of every byte swapped), and the scale plane against the RMS of the
  // vectors the writer was given, with every pad lane and unwritten slot left at zero.
  WrittenCacheAgreement CompareWrittenCache() const {
    constexpr int kNibbleBits = 4;
    constexpr uint32_t kByteMask = 0xFF;
    const auto swap_lanes = [](int8_t byte) {
      const uint32_t bits = static_cast<uint8_t>(byte);
      return static_cast<int8_t>(((bits << kNibbleBits) | (bits >> kNibbleBits)) & kByteMask);
    };
    WrittenCacheAgreement agreement;
    const std::vector<int8_t>* expected[] = {&cache_.key_packed, &cache_.value_packed};
    const std::vector<int8_t>* written[] = {&written_key_, &written_value_};
    for (int plane = 0; plane < 2; ++plane) {
      const std::vector<int8_t>& want = *expected[plane];
      const std::vector<int8_t>& got = *written[plane];
      agreement.packed_bytes += want.size();
      if (got.size() != want.size()) {
        agreement.packed_mismatches += want.size();
        agreement.swapped_lane_mismatches += want.size();
        continue;
      }
      for (size_t i = 0; i < want.size(); ++i) {
        agreement.packed_mismatches += (got[i] != want[i]) ? 1u : 0u;
        agreement.swapped_lane_mismatches += (got[i] != swap_lanes(want[i])) ? 1u : 0u;
      }
    }
    agreement.scale_lanes = expected_scales_.size();
    for (size_t i = 0; i < expected_scales_.size() && i < written_scales_.size(); ++i) {
      if (expected_scales_[i] == 0.0) {
        agreement.scale_pad_mismatches += (written_scales_[i] != 0.0f) ? 1u : 0u;
        continue;
      }
      const double rel =
          std::fabs(static_cast<double>(written_scales_[i]) - expected_scales_[i]) / expected_scales_[i];
      agreement.max_scale_rel_err = std::max(agreement.max_scale_rel_err, rel);
    }
    if (written_scales_.size() != expected_scales_.size()) {
      agreement.scale_pad_mismatches += expected_scales_.size();
    }
    return agreement;
  }

 private:
  // The writer is given the unrotated fp16 of the host's dequantised vectors. It rotates them back and
  // quantises each coordinate onto the level it came from, since a dequantised coordinate sits half a step
  // from both neighbouring thresholds. Its scale is the RMS of those vectors rather than of the draw, so
  // the exact-attention reference is rescaled to the scales the writer stored.
  void WriteThroughKernel() {
    const int64_t d = shape_.head_size;
    const int64_t tokens = shape_.context_len;
    const int64_t block_size = shape_.block_size;
    const size_t head = static_cast<size_t>(d);
    const size_t kv_heads = static_cast<size_t>(shape_.num_kv_heads);
    const size_t slot_floats = static_cast<size_t>(MirroredScaleSlotFloats(shape_.num_kv_heads));

    std::vector<int32_t> slots(static_cast<size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      slots[static_cast<size_t>(t)] = block_table_[static_cast<size_t>(t / block_size)] *
                                          static_cast<int32_t>(block_size) +
                                      static_cast<int32_t>(t % block_size);
    }
    DeviceBuffer key_fp16 = DeviceBuffer::FromHost(FloatToHalf(UnrotateHeads(cache_.key, d)));
    DeviceBuffer value_fp16 = DeviceBuffer::FromHost(FloatToHalf(UnrotateHeads(cache_.value, d)));
    DeviceBuffer slot_mapping = DeviceBuffer::FromHost(slots);
    DeviceBuffer write_tables = DeviceBuffer::FromHost(ModeTables(kFusedMode, d, 1, 0));
    key_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_.key_packed.size(), 0));
    value_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_.value_packed.size(), 0));
    scale_plane_ = DeviceBuffer::FromHost(std::vector<float>(cache_.scales.size(), 0.0f));

    const ReshapeAndCacheGrid grid = PlanReshapeAndCache(tokens, aiv_num_);
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(kFusedMode), AscendType::FP16, stream_, grid.block_dim, key_fp16.get(), value_fp16.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), slot_mapping.get(), pi_signs_.get(),
        rot_tables_.get(), write_tables.get(), static_cast<uint32_t>(tokens), static_cast<uint32_t>(kv_heads),
        static_cast<uint32_t>(d), static_cast<uint32_t>(block_size), grid.tokens_per_core,
        1.0f / std::sqrt(static_cast<float>(d)));
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    written_key_ = key_cache_.ToHost<int8_t>();
    written_value_ = value_cache_.ToHost<int8_t>();
    written_scales_ = scale_plane_.ToHost<float>();

    expected_scales_.assign(cache_.scales.size(), 0.0);
    for (int64_t t = 0; t < tokens; ++t) {
      const size_t slot = static_cast<size_t>(slots[static_cast<size_t>(t)]);
      for (size_t kv = 0; kv < kv_heads; ++kv) {
        for (int plane = 0; plane < 2; ++plane) {
          std::vector<float>& dense = plane == 0 ? cache_.key : cache_.value;
          const size_t row = (static_cast<size_t>(t) * kv_heads + kv) * head;
          double energy = 0.0;
          for (size_t c = 0; c < head; ++c) {
            energy += static_cast<double>(dense[row + c]) * static_cast<double>(dense[row + c]);
          }
          const size_t lane = slot * slot_floats + (plane == 0 ? kv : kv_heads + kv);
          expected_scales_[lane] = std::sqrt(energy / static_cast<double>(head));
          const double factor =
              static_cast<double>(written_scales_[lane]) / static_cast<double>(cache_.scales[lane]);
          for (size_t c = 0; c < head; ++c) {
            dense[row + c] = static_cast<float>(static_cast<double>(dense[row + c]) * factor);
          }
        }
      }
    }
  }

  // One timed launch over `grid`: a fresh workspace, a poisoned output, then `launch(workspace, 1 / sqrt(D))`.
  template <typename LaunchFn>
  FusedRun Launch(const FusedDecodeGrid& grid, LaunchFn&& launch) {
    DeviceBuffer workspace = DeviceBuffer::Empty<float>(grid.workspace_floats);
    FusedRun run;
    run.num_splits = grid.num_splits;
    run.block_dim = grid.block_dim;
    run.heads_per_task = grid.heads_per_task;
    run.fused_context_limit = grid.fused_context_limit;
    run.launches = 1;
    PoisonOutput();
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(shape_.head_size));
    const auto start = std::chrono::steady_clock::now();
    launch(workspace.get(), inv_sqrt_len);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    run.host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Collect(&run);
    return run;
  }

  void PoisonOutput() {
    out_ = DeviceBuffer::FromHost(
        FloatToHalf(std::vector<float>(static_cast<size_t>(shape_.num_heads * shape_.head_size), kFusedOutputSentinel)));
  }

  void Collect(FusedRun* run) const {
    run->raw = out_.ToHost<Half>();
    run->output = HalfToFloat(run->raw);
    run->untouched = static_cast<size_t>(std::count(run->output.begin(), run->output.end(), kFusedOutputSentinel));
    run->fnv1a = Fnv1a64(run->raw);
  }

  FusedShape shape_;
  aclrtStream stream_;
  int64_t aiv_num_ = 0;
  int64_t blocks_per_seq_ = 0;
  float scale_ = 1.0f;
  std::vector<int32_t> block_table_;
  MirroredKvCache cache_;
  std::vector<float> query_rot_host_;
  std::vector<int8_t> written_key_;
  std::vector<int8_t> written_value_;
  std::vector<float> written_scales_;
  std::vector<double> expected_scales_;
  DeviceBuffer query_, pi_signs_, h16_, rot_tables_, mode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, block_tables_, context_lens_;
  DeviceBuffer query_rot_, out_;
};

}
}
}
