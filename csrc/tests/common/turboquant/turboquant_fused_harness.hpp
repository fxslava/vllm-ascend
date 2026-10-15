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
#include <limits>
#include <stdexcept>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp4_e2m1.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turboquant_launch.hpp"
#include "turboquant_mirrored_cache.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_host {

constexpr uint32_t kFusedSeed = 0xF05Eu;
// TurboQuantCodec<BITS>::kEps, duplicated rather than included: turboquant_codec_950.h is an
// Ascend C header and does not compile on the host. It guards a divide, and at double precision
// over these magnitudes it changes nothing -- it is here so the reference divides by the same
// denominator the kernel does rather than by one that happens to be close.
constexpr double kOperandGridEps = 1e-20;
constexpr float kFusedOutputSentinel = -1234.5f;
constexpr int8_t kSparsePackedCanary = static_cast<int8_t>(0x5a);
constexpr float kSparseScaleCanary = 12345.0f;

struct FusedShape {
  vllm_ascend::turboquant::TurboQuantMode mode = vllm_ascend::turboquant::TurboQuantMode::KV4_FP8;
  int64_t num_heads = 4;
  int64_t num_kv_heads = 2;
  int64_t head_size = 256;
  int64_t block_size = kCubeTileRows;
  int64_t context_len = kCubeTileRows;
  int64_t pool_factor = 4;
  // Draw independent vector halves and write the cache through the selected Cube kernel writer
  // (turboquant_mm_reshape_and_cache_impl) instead of uploading the host-built image.
  bool kernel_writer = false;
  // Decode tokens. Every token reads the same sequence through its own block-table row, with its own query.
  int64_t batch = 1;
  // Draw an attn_output_gate logit per query element, for the raw-query decode's kGated output stage.
  bool gated = false;
  // Fill every block-table row with -1, the entry a padding or unmapped block carries. The block range of
  // every task is then non-empty but holds no tile, which is the decode's numTiles == 0 path: both cores
  // have to bypass the cross-core pipeline in lock step, and the launch has to finish rather than strand
  // the AIC on a flag nothing will set. The cache is still built and uploaded, so a decode that read one
  // of these rows anyway would not read zeros.
  bool unmapped_block_table = false;
  // Quantise on the e4m3-rounded Lloyd-Max codebook instead of the uniform mid-rise grid
  // (DecodeUnpack::kLloydMaxLut). The whole scenario moves together: the host cache, the mode-table image,
  // the kernel writer and the decode entry point, because a cache and a decode on different codebooks
  // produce a plausible wrong answer rather than an error. Not available through the raw-query entry.
  bool lloyd_max = false;
  // Skip all but the first, second, middle and last Cube tiles. Requires one tile per block.
  bool sparse_boundary = false;
};

// The raw GM image one cache-writer launch produced, copied out of device memory.
struct WrittenImage {
  std::vector<int8_t> key;
  std::vector<int8_t> value;
  std::vector<float> scales;
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
  // Raw-query runs only: fp32 words of the in-launch rotation that differ from the rotate_q launch's, and the
  // largest difference.
  size_t prologue_mismatches = 0;
  double prologue_max_abs = 0.0;
};

constexpr uint32_t kRotatedBasis = static_cast<uint32_t>(vllm_ascend::turboquant::TurboQuantOutputStage::kRotatedBasis);
constexpr uint32_t kGatedOutput = static_cast<uint32_t>(vllm_ascend::turboquant::TurboQuantOutputStage::kGated);
// The spread of the drawn gate logits: sigmoid(gate) then covers most of (0, 1).
constexpr float kGateLogitStddev = 2.0f;

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
    if (shape.sparse_boundary && shape.block_size != kCubeTileRows) {
      throw std::invalid_argument("sparse boundary verification needs one Cube tile per block");
    }
    const int64_t d = shape.head_size;
    blocks_per_seq_ = CeilDiv(shape.context_len, shape.block_size);
    const int64_t num_blocks = blocks_per_seq_ * shape.pool_factor;

    DeterministicRandom rng(kFusedSeed + static_cast<uint32_t>(d + shape.context_len));
    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks));
    block_table_.assign(permutation.begin(), permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq_));
    cache_ = BuildMirroredKvCache(rng, shape.context_len, num_blocks, shape.block_size, shape.num_kv_heads, d,
                                  block_table_, !shape.kernel_writer, codebook(), shape.mode);
    if (shape.sparse_boundary) {
      if (shape.kernel_writer) {
        throw std::invalid_argument("sparse boundary verification requires a host-uploaded cache");
      }
      sparse_tiles_ = SparseBoundaryTiles(blocks_per_seq_);
      const size_t packed_stride = static_cast<size_t>(
          vllm_ascend::turboquant::TurboQuantModeConfigOf(shape.mode).PackedPlaneBytes(d, shape.num_kv_heads));
      const size_t scale_stride = static_cast<size_t>(MirroredScaleSlotFloats(shape.num_kv_heads));
      for (int64_t tile = 0; tile < blocks_per_seq_; ++tile) {
        if (IsSparseTile(tile)) {
          continue;
        }
        const size_t first_slot = static_cast<size_t>(block_table_[static_cast<size_t>(tile)] * shape.block_size);
        const size_t last_slot = first_slot + static_cast<size_t>(shape.block_size);
        std::fill(cache_.key_packed.begin() + first_slot * packed_stride,
                  cache_.key_packed.begin() + last_slot * packed_stride, kSparsePackedCanary);
        std::fill(cache_.value_packed.begin() + first_slot * packed_stride,
                  cache_.value_packed.begin() + last_slot * packed_stride, kSparsePackedCanary);
        std::fill(cache_.scales.begin() + first_slot * scale_stride, cache_.scales.begin() + last_slot * scale_stride,
                  kSparseScaleCanary);
      }
    }
    const size_t query_elems = static_cast<size_t>(shape.batch * shape.num_heads * d);
    const std::vector<float> query = rng.NormalHalfExact(query_elems, 0.0f, 1.0f);
    if (shape.gated) {
      gate_host_ = rng.NormalHalfExact(query_elems, 0.0f, kGateLogitStddev);
      gate_ = DeviceBuffer::FromHost(FloatToHalf(gate_host_));
    }

    scale_ = 1.0f / std::sqrt(static_cast<float>(d));
    query_ = DeviceBuffer::FromHost(FloatToHalf(query));
    pi_signs_ = DeviceBuffer::FromHost(PiSigns(d));
    h16_ = DeviceBuffer::FromHost(Hadamard16Half());
    rot_tables_ = DeviceBuffer::FromHost(CodecTables(d, 1));
    mode_tables_ = DeviceBuffer::FromHost(ModeTables(shape.mode, d, kUnpackRows, kCubeTileRows, shape.lloyd_max));
    if (shape.kernel_writer) {
      WriteThroughKernel();
    } else {
      key_cache_ = DeviceBuffer::FromHost(cache_.key_packed);
      value_cache_ = DeviceBuffer::FromHost(cache_.value_packed);
      scale_plane_ = DeviceBuffer::FromHost(cache_.scales);
    }
    std::vector<int32_t> block_rows;
    for (int64_t token = 0; token < shape.batch; ++token) {
      if (shape.unmapped_block_table) {
        block_rows.insert(block_rows.end(), static_cast<size_t>(blocks_per_seq_), -1);
      } else {
        for (int64_t tile = 0; tile < blocks_per_seq_; ++tile) {
          block_rows.push_back(shape.sparse_boundary && !IsSparseTile(tile) ? -1
                                                                            : block_table_[static_cast<size_t>(tile)]);
        }
      }
    }
    block_tables_ = DeviceBuffer::FromHost(block_rows);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(shape.batch), static_cast<int32_t>(shape.context_len)));
    query_rot_ = DeviceBuffer::Empty<float>(query_elems);
    out_ = DeviceBuffer::Empty<Half>(query_elems);

    RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                query_rot_.get(), shape.batch, shape.num_heads, d, aiv_num_);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    query_rot_host_ = query_rot_.ToHost<float>();
  }

  // plan_aiv is the vector-core budget the planner tiles for (0: the device's own); a smaller budget
  // than the device has forces fewer, wider tasks without changing the kernel.
  FusedDecodeGrid Plan(int64_t plan_aiv = 0, FusedSplitPolicy split_policy = FusedSplitPolicy::kFillBlocks) const {
    return PlanFusedDecode(shape_.batch, shape_.num_heads, shape_.num_kv_heads, shape_.head_size, blocks_per_seq_,
                           shape_.block_size, plan_aiv > 0 ? plan_aiv : aiv_num_, kFusedContextLimit, split_policy);
  }

  // The shape's codebook picks the entry point: the two take the same arguments and differ only in which
  // 16 levels their expand gathers, which is the same 16 the cache above was written with.
  FusedRun RunFused(const FusedDecodeGrid& grid) {
    const bool lloyd_max = shape_.lloyd_max;
    return Launch(grid, [&](void* workspace, float inv_sqrt_len) {
      const auto launch = lloyd_max ? &turboquant_mm_fused_decode_lloydmax_impl : &turboquant_mm_fused_decode_impl;
      launch(static_cast<int32_t>(shape_.mode), AscendType::FP16, stream_, grid.block_dim, query_rot_.get(),
             key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
             mode_tables_.get(), workspace, out_.get(), static_cast<uint32_t>(shape_.batch),
             static_cast<uint32_t>(shape_.num_heads), static_cast<uint32_t>(shape_.num_kv_heads),
             static_cast<uint32_t>(shape_.head_size), static_cast<uint32_t>(shape_.block_size),
             static_cast<uint32_t>(blocks_per_seq_), static_cast<uint32_t>(grid.num_splits), grid.heads_per_task,
             grid.tasks_per_block, grid.reduce_tasks_per_block, grid.fused_context_limit, scale_, inv_sqrt_len);
    });
  }

  // The same grid through the raw-query decode: the launch is handed the fp16 query and rotates it itself,
  // into a fresh buffer that is then compared word for word with what rotate_q wrote at setup. output_stage is a
  // TurboQuantOutputStage; kGatedOutput needs a gated shape.
  FusedRun RunFusedRawQuery(const FusedDecodeGrid& grid, uint32_t output_stage = kRotatedBasis) {
    // There is no Lloyd-Max raw-query entry point, and running the uniform one over a Lloyd-Max cache
    // would decode a plausible wrong answer instead of failing. Report nothing launched; every caller
    // asserts on launches == 1.
    if (shape_.lloyd_max || shape_.mode != vllm_ascend::turboquant::TurboQuantMode::KV4_FP8) {
      std::printf("[ fused ] the raw-query decode requires uniform KV4_FP8; not launched\n");
      return FusedRun{};
    }
    DeviceBuffer rotated = DeviceBuffer::FromHost(std::vector<float>(query_rot_host_.size(), kFusedOutputSentinel));
    void* gate = output_stage == kGatedOutput ? gate_.get() : nullptr;
    FusedRun run = Launch(grid, [&](void* workspace, float inv_sqrt_len) {
      turboquant_mm_fused_decode_raw_query_impl(
          static_cast<int32_t>(shape_.mode), AscendType::FP16, stream_, grid.block_dim, query_.get(), pi_signs_.get(),
          rot_tables_.get(), h16_.get(), gate, rotated.get(), key_cache_.get(), value_cache_.get(), scale_plane_.get(),
          block_tables_.get(), context_lens_.get(), mode_tables_.get(), workspace, out_.get(),
          static_cast<uint32_t>(shape_.batch), static_cast<uint32_t>(shape_.num_heads),
          static_cast<uint32_t>(shape_.num_kv_heads), static_cast<uint32_t>(shape_.head_size),
          static_cast<uint32_t>(shape_.block_size), static_cast<uint32_t>(blocks_per_seq_),
          static_cast<uint32_t>(grid.num_splits), grid.heads_per_task, grid.tasks_per_block,
          grid.reduce_tasks_per_block, grid.prologue_vectors_per_block, grid.prologue_cube_chunk_vectors, output_stage,
          grid.fused_context_limit, scale_, inv_sqrt_len);
    });
    const std::vector<float> in_launch = rotated.ToHost<float>();
    for (size_t i = 0; i < query_rot_host_.size(); ++i) {
      const bool written = i < in_launch.size();
      run.prologue_mismatches +=
          (!written || std::memcmp(&in_launch[i], &query_rot_host_[i], sizeof(float)) != 0) ? 1u : 0u;
      const double err = written ? std::fabs(static_cast<double>(in_launch[i]) - query_rot_host_[i]) : HUGE_VAL;
      run.prologue_max_abs = std::max(run.prologue_max_abs, err);
    }
    return run;
  }

  // What an o_proj reads: the rotated-basis rows un-rotated head by head, then multiplied by sigmoid(gate).
  std::vector<float> ModelBasis(const std::vector<float>& rotated) const {
    std::vector<float> out = UnrotateHeads(rotated, shape_.head_size);
    for (size_t i = 0; i < out.size() && i < gate_host_.size(); ++i) {
      const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(gate_host_[i])));
      out[i] = static_cast<float>(static_cast<double>(out[i]) * gate);
    }
    return out;
  }

  std::vector<float> Reference() const {
    const int64_t d = shape_.head_size;
    const int64_t heads_per_kv = shape_.num_heads / shape_.num_kv_heads;
    std::vector<float> out(static_cast<size_t>(shape_.batch * shape_.num_heads * d), 0.0f);
    std::vector<int64_t> active_tokens;
    for (int64_t t = 0; t < shape_.context_len; ++t) {
      if (!shape_.sparse_boundary || IsSparseTile(t / shape_.block_size)) {
        active_tokens.push_back(t);
      }
    }
    std::vector<double> logits(active_tokens.size(), 0.0);
    // h is a (token, head) row; every token attends to the same sequence.
    for (int64_t h = 0; h < shape_.batch * shape_.num_heads; ++h) {
      const int64_t kv = (h % shape_.num_heads) / heads_per_kv;
      double max_logit = -1e30;
      for (size_t i = 0; i < active_tokens.size(); ++i) {
        const int64_t t = active_tokens[i];
        double dot = 0.0;
        for (int64_t c = 0; c < d; ++c) {
          dot += static_cast<double>(query_rot_host_[static_cast<size_t>(h * d + c)]) *
                 static_cast<double>(cache_.key[static_cast<size_t>((t * shape_.num_kv_heads + kv) * d + c)]);
        }
        logits[i] = dot * static_cast<double>(scale_);
        max_logit = std::max(max_logit, logits[i]);
      }
      double denom = 0.0;
      for (double& logit : logits) {
        logit = std::exp(logit - max_logit);
        denom += logit;
      }
      for (size_t i = 0; i < active_tokens.size(); ++i) {
        const int64_t t = active_tokens[i];
        const double weight = logits[i] / denom;
        for (int64_t c = 0; c < d; ++c) {
          out[static_cast<size_t>(h * d + c)] += static_cast<float>(
              weight * static_cast<double>(cache_.value[static_cast<size_t>((t * shape_.num_kv_heads + kv) * d + c)]));
        }
      }
    }
    return out;
  }

  // Reference() in fp32 over the levels the cache holds answers "did the quantiser lose anything",
  // which for an fp4 mode is about a hundredth of a cosine -- enough slack to hide a transposed
  // column group, a tail row off by one or a single corrupted lane. This one answers "did the kernel
  // compute what it set out to" instead, by taking every step the KV3 decode takes on the numbers it
  // takes them on: the query and the per-tile probability rows land on the e2m1 grid here too, the
  // online softmax runs tile by tile in the kernel's order over the kernel's mapped tiles, and the
  // per-slot scales are applied where the kernel applies them -- the key's after the score Mmad, the
  // value's folded into the probability row before its cast. What is left over is fp32 accumulation
  // order in the Cube against double here, and the fp16 output cast, so the gate is 0.999, not 0.95.
  //
  // KV3 only: it reads the bare levels, which an fp8 mode does not keep.
  std::vector<float> ReferenceOnOperandGrid() const {
    const int64_t d = shape_.head_size;
    const int64_t heads_per_kv = shape_.num_heads / shape_.num_kv_heads;
    const int64_t scale_stride = MirroredScaleSlotFloats(shape_.num_kv_heads);
    const double gain = static_cast<double>(MirroredKv3Traits::kGain);
    const double eps = kOperandGridEps;
    const double operand_max = static_cast<double>(vllm_ascend::test::fp4::kE2m1Max);
    std::vector<float> out(static_cast<size_t>(shape_.batch * shape_.num_heads * d), 0.0f);

    std::vector<double> query_operand(static_cast<size_t>(d), 0.0);
    std::vector<double> logits(static_cast<size_t>(kCubeTileRows), 0.0);
    std::vector<double> probs(static_cast<size_t>(kCubeTileRows), 0.0);
    std::vector<double> acc(static_cast<size_t>(d), 0.0);
    std::vector<double> context(static_cast<size_t>(d), 0.0);

    for (int64_t h = 0; h < shape_.batch * shape_.num_heads; ++h) {
      const int64_t kv = (h % shape_.num_heads) / heads_per_kv;

      // ComputeQueryOperandHeads: amax over the head, then the whole row onto the operand grid.
      double amax = 0.0;
      for (int64_t c = 0; c < d; ++c) {
        amax = std::max(amax, std::fabs(static_cast<double>(query_rot_host_[static_cast<size_t>(h * d + c)])));
      }
      const double q_scale = operand_max / (amax + eps);
      for (int64_t c = 0; c < d; ++c) {
        query_operand[static_cast<size_t>(c)] = static_cast<double>(vllm_ascend::test::fp4::E2m1Operand(
            static_cast<float>(static_cast<double>(query_rot_host_[static_cast<size_t>(h * d + c)]) * q_scale)));
      }
      const double q_scale_inv = 1.0 / q_scale;

      std::fill(acc.begin(), acc.end(), 0.0);
      double run_max = -std::numeric_limits<double>::infinity();
      double run_sum = 0.0;

      for (int64_t tile = 0; tile < blocks_per_seq_; ++tile) {
        if (shape_.sparse_boundary && !IsSparseTile(tile)) {
          continue;  // NextTile walks past an unmapped block without touching the accumulators
        }
        const int64_t first = tile * kCubeTileRows;
        const int64_t valid = std::min<int64_t>(kCubeTileRows, shape_.context_len - first);
        if (valid <= 0) {
          continue;
        }
        const int64_t slot_base = block_table_[static_cast<size_t>(tile)] * shape_.block_size;

        // The score Mmad, then keyScale = slot scale * (softmax scale / gain) and 1 / qScale.
        double tile_max = -std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < valid; ++i) {
          const size_t row = static_cast<size_t>(((first + i) * shape_.num_kv_heads + kv) * d);
          double dot = 0.0;
          for (int64_t c = 0; c < d; ++c) {
            dot += query_operand[static_cast<size_t>(c)] *
                   static_cast<double>(cache_.key_levels[row + static_cast<size_t>(c)]);
          }
          const double key_scale =
              static_cast<double>(cache_.scales[static_cast<size_t>((slot_base + i) * scale_stride + kv)]);
          logits[static_cast<size_t>(i)] = dot * key_scale * (static_cast<double>(scale_) / gain) * q_scale_inv;
          tile_max = std::max(tile_max, logits[static_cast<size_t>(i)]);
        }

        // One online-softmax step. The masked tail rows exp() to exactly zero, so they are simply absent.
        const double new_max = std::max(run_max, tile_max);
        const double alpha = std::isinf(run_max) ? 0.0 : std::exp(run_max - new_max);
        double tile_sum = 0.0;
        for (int64_t i = 0; i < valid; ++i) {
          probs[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - new_max);
          tile_sum += probs[static_cast<size_t>(i)];
        }
        run_sum = run_sum * alpha + tile_sum;

        // The value's slot scale rides in the probability row, so the row's own max sets probScale.
        double prob_max = 0.0;
        for (int64_t i = 0; i < valid; ++i) {
          const double value_scale =
              static_cast<double>(
                  cache_.scales[static_cast<size_t>((slot_base + i) * scale_stride + shape_.num_kv_heads + kv)]) /
              gain;
          probs[static_cast<size_t>(i)] *= value_scale;
          prob_max = std::max(prob_max, probs[static_cast<size_t>(i)]);
        }
        const double prob_scale = operand_max / (prob_max + eps);
        for (int64_t i = 0; i < valid; ++i) {
          probs[static_cast<size_t>(i)] = static_cast<double>(vllm_ascend::test::fp4::E2m1Operand(
              static_cast<float>(probs[static_cast<size_t>(i)] * prob_scale)));
        }

        // The context Mmad, its prob scale undone, onto the decayed accumulator.
        std::fill(context.begin(), context.end(), 0.0);
        for (int64_t i = 0; i < valid; ++i) {
          const size_t row = static_cast<size_t>(((first + i) * shape_.num_kv_heads + kv) * d);
          const double weight = probs[static_cast<size_t>(i)];
          for (int64_t c = 0; c < d; ++c) {
            context[static_cast<size_t>(c)] +=
                weight * static_cast<double>(cache_.value_levels[row + static_cast<size_t>(c)]);
          }
        }
        for (int64_t c = 0; c < d; ++c) {
          acc[static_cast<size_t>(c)] =
              acc[static_cast<size_t>(c)] * alpha + context[static_cast<size_t>(c)] / prob_scale;
        }
        run_max = new_max;
      }

      const double inv_sum = 1.0 / (run_sum + eps);
      for (int64_t c = 0; c < d; ++c) {
        out[static_cast<size_t>(h * d + c)] = static_cast<float>(acc[static_cast<size_t>(c)] * inv_sum);
      }
    }
    return out;
  }

  int64_t blocks_per_seq() const { return blocks_per_seq_; }
  size_t sparse_tile_count() const { return sparse_tiles_.size(); }

  // Cache planes are read-only inputs to decode. Inspect the skipped physical slots after the launch.
  size_t SparseCanaryMismatches() const {
    if (!shape_.sparse_boundary) {
      return 0;
    }
    const std::vector<int8_t> key = key_cache_.ToHost<int8_t>();
    const std::vector<int8_t> value = value_cache_.ToHost<int8_t>();
    const std::vector<float> scales = scale_plane_.ToHost<float>();
    const size_t packed_stride = static_cast<size_t>(vllm_ascend::turboquant::TurboQuantModeConfigOf(shape_.mode)
                                                         .PackedPlaneBytes(shape_.head_size, shape_.num_kv_heads));
    const size_t scale_stride = static_cast<size_t>(MirroredScaleSlotFloats(shape_.num_kv_heads));
    size_t mismatches = 0;
    for (int64_t tile = 0; tile < blocks_per_seq_; ++tile) {
      if (IsSparseTile(tile)) {
        continue;
      }
      const size_t first_slot = static_cast<size_t>(block_table_[static_cast<size_t>(tile)] * shape_.block_size);
      const size_t last_slot = first_slot + static_cast<size_t>(shape_.block_size);
      for (size_t i = first_slot * packed_stride; i < last_slot * packed_stride; ++i) {
        mismatches += key[i] != kSparsePackedCanary;
        mismatches += value[i] != kSparsePackedCanary;
      }
      for (size_t i = first_slot * scale_stride; i < last_slot * scale_stride; ++i) {
        mismatches += scales[i] != kSparseScaleCanary;
      }
    }
    return mismatches;
  }

  MirroredCodebook codebook() const {
    return shape_.lloyd_max ? MirroredCodebook::kLloydMax : MirroredCodebook::kUniform;
  }

  // The GM image the last cache-writer launch produced. Note that the construction-time launch was fed the
  // original dense cache and every launch after it is fed the rescaled one (see WriteThroughKernel), so a
  // caller comparing images across launches has to take its own control rather than use the constructor's.
  WrittenImage CaptureWrittenImage() const { return {written_key_, written_value_, written_scales_}; }

  size_t ExpectedPackedPlaneBytes() const { return cache_.key_packed.size(); }
  size_t ExpectedScalePlaneFloats() const { return cache_.scales.size(); }
  int64_t PoolSlots() const { return blocks_per_seq_ * shape_.pool_factor * shape_.block_size; }

  // Fill only the inactive suffix of the last mapped block with extreme values.
  // A correctly masked decode must produce the same half bits as before poisoning.
  void PoisonInactiveTail() {
    const int64_t tail = shape_.context_len % shape_.block_size;
    if (tail == 0) {
      return;
    }
    std::vector<int8_t> key = key_cache_.ToHost<int8_t>();
    std::vector<int8_t> value = value_cache_.ToHost<int8_t>();
    std::vector<float> scales = scale_plane_.ToHost<float>();
    const size_t packed_bytes = static_cast<size_t>(ModePackedBytes(shape_.mode, shape_.head_size));
    const size_t token_row_stride = static_cast<size_t>(vllm_ascend::turboquant::TurboQuantModeConfigOf(shape_.mode)
                                                            .PackedPlaneBytes(shape_.head_size, shape_.num_kv_heads));
    const size_t scale_stride = static_cast<size_t>(MirroredScaleSlotFloats(shape_.num_kv_heads));
    const int64_t block = block_table_.back();
    for (int64_t row = tail; row < shape_.block_size; ++row) {
      const size_t slot = static_cast<size_t>(block * shape_.block_size + row);
      for (int64_t kv = 0; kv < shape_.num_kv_heads; ++kv) {
        for (size_t byte = 0; byte < packed_bytes; ++byte) {
          const size_t offset =
              shape_.mode == vllm_ascend::turboquant::TurboQuantMode::KV3_FP4
                  ? slot * token_row_stride + static_cast<size_t>(kv) * packed_bytes + byte
                  : MirroredPackedByte(slot, static_cast<size_t>(kv), byte, static_cast<size_t>(shape_.num_kv_heads),
                                       packed_bytes, shape_.mode);
          const int8_t poison = row % 2 == 0 ? static_cast<int8_t>(0xff) : static_cast<int8_t>(0x00);
          key[offset] = poison;
          value[offset] = poison;
        }
        scales[slot * scale_stride + static_cast<size_t>(kv)] = 1.0e4f;
        scales[slot * scale_stride + static_cast<size_t>(shape_.num_kv_heads + kv)] = 1.0e4f;
      }
    }
    key_cache_ = DeviceBuffer::FromHost(key);
    value_cache_ = DeviceBuffer::FromHost(value);
    scale_plane_ = DeviceBuffer::FromHost(scales);
  }

  // Re-runs the selected Cube cache writer with slot_mapping handed in `offset_elems` int32s into its allocation,
  // which is what a PyTorch sub-slice of a slot-mapping tensor looks like: contiguous, element aligned, and
  // at offset 1, 2 or 3 NOT on a 32-byte burst boundary. The kernel reads slot_mapping only with
  // GlobalTensor::GetValue, one int32 at a time, so nothing about it needs burst alignment -- this is the
  // execution-side half of the adapter's CheckGmScalarAligned contract. Needs a kernel_writer shape.
  //
  // Returns the GM image to compare against CaptureWrittenImage(), and deliberately does NOT touch the
  // scenario's host-side reference: the scale rescale WriteThroughKernel does at construction is a one-shot
  // that rescales cache_.key / cache_.value in place, and running it again would compound it.
  WrittenImage RewriteAtSlotMappingOffset(int64_t offset_elems) {
    LaunchWriter(offset_elems);
    return CaptureWrittenImage();
  }

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
      // Compare GM and CPU images at the same linear byte offset. For KV3 each
      // slot is one row of PackedPlaneBytes, including the 16-byte D128/HKV1 pad.
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
      const double rel = std::fabs(static_cast<double>(written_scales_[i]) - expected_scales_[i]) / expected_scales_[i];
      agreement.max_scale_rel_err = std::max(agreement.max_scale_rel_err, rel);
    }
    if (written_scales_.size() != expected_scales_.size()) {
      agreement.scale_pad_mismatches += expected_scales_.size();
    }
    return agreement;
  }

 private:
  bool IsSparseTile(int64_t tile) const {
    return std::find(sparse_tiles_.begin(), sparse_tiles_.end(), tile) != sparse_tiles_.end();
  }
  // The writer is given the unrotated fp16 of the host's dequantised vectors. It rotates them back and
  // quantises each coordinate onto the level it came from, because a level always sits strictly inside its
  // own decision cell -- centred there on the uniform grid, off-centre but never within 0.125 scale units
  // of a boundary on the Lloyd-Max one, which is orders of magnitude more slack than the fp16 cast and the
  // rotation round trip consume. Its scale is the RMS of those vectors rather than of the draw, so the
  // exact-attention reference is rescaled to the scales the writer stored.
  void WriteThroughKernel() {
    const std::vector<int32_t> slots = LaunchWriter(0);
    const int64_t tokens = shape_.context_len;
    const size_t head = static_cast<size_t>(shape_.head_size);
    const size_t kv_heads = static_cast<size_t>(shape_.num_kv_heads);
    const size_t slot_floats = static_cast<size_t>(MirroredScaleSlotFloats(shape_.num_kv_heads));

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
          const double factor = static_cast<double>(written_scales_[lane]) / static_cast<double>(cache_.scales[lane]);
          for (size_t c = 0; c < head; ++c) {
            dense[row + c] = static_cast<float>(static_cast<double>(dense[row + c]) * factor);
          }
        }
      }
    }
  }

  // One cache-writer launch, with slot_mapping handed in `slot_offset_elems` int32s into its allocation.
  // Fills written_key_ / written_value_ / written_scales_ and returns the slots it wrote. Pure with respect
  // to the host-side reference; WriteThroughKernel is the one that rescales it, once.
  std::vector<int32_t> LaunchWriter(int64_t slot_offset_elems) {
    const int64_t d = shape_.head_size;
    const int64_t tokens = shape_.context_len;
    const int64_t block_size = shape_.block_size;
    const size_t kv_heads = static_cast<size_t>(shape_.num_kv_heads);

    std::vector<int32_t> slots(static_cast<size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      slots[static_cast<size_t>(t)] =
          block_table_[static_cast<size_t>(t / block_size)] * static_cast<int32_t>(block_size) +
          static_cast<int32_t>(t % block_size);
    }
    DeviceBuffer key_fp16 = DeviceBuffer::FromHost(FloatToHalf(UnrotateHeads(cache_.key, d)));
    DeviceBuffer value_fp16 = DeviceBuffer::FromHost(FloatToHalf(UnrotateHeads(cache_.value, d)));
    // The slots land `slot_offset_elems` int32s into the allocation, and the kernel is given that address.
    std::vector<int32_t> slot_storage(static_cast<size_t>(slot_offset_elems), 0);
    slot_storage.insert(slot_storage.end(), slots.begin(), slots.end());
    DeviceBuffer slot_mapping = DeviceBuffer::FromHost(slot_storage);
    void* slot_arg = static_cast<void*>(static_cast<int32_t*>(slot_mapping.get()) + slot_offset_elems);
    DeviceBuffer write_tables = DeviceBuffer::FromHost(ModeTables(shape_.mode, d, 1, 0, shape_.lloyd_max));
    key_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_.key_packed.size(), 0));
    value_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_.value_packed.size(), 0));
    scale_plane_ = DeviceBuffer::FromHost(std::vector<float>(cache_.scales.size(), 0.0f));

    const ReshapeAndCacheGrid grid = PlanReshapeAndCache(tokens, aiv_num_);
    // Same argument list, same layout written; the codebook decides only which thresholds bin a coordinate.
    const auto write =
        shape_.lloyd_max ? &turboquant_mm_reshape_and_cache_lloydmax_impl : &turboquant_mm_reshape_and_cache_impl;
    write(static_cast<int32_t>(shape_.mode), AscendType::FP16, stream_, grid.block_dim, key_fp16.get(),
          value_fp16.get(), key_cache_.get(), value_cache_.get(), scale_plane_.get(), slot_arg, pi_signs_.get(),
          rot_tables_.get(), write_tables.get(), static_cast<uint32_t>(tokens), static_cast<uint32_t>(kv_heads),
          static_cast<uint32_t>(d), static_cast<uint32_t>(block_size),
          static_cast<uint32_t>(blocks_per_seq_ * shape_.pool_factor), grid.tokens_per_core,
          1.0f / std::sqrt(static_cast<float>(d)));
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    written_key_ = key_cache_.ToHost<int8_t>();
    written_value_ = value_cache_.ToHost<int8_t>();
    written_scales_ = scale_plane_.ToHost<float>();
    return slots;
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
    out_ = DeviceBuffer::FromHost(FloatToHalf(std::vector<float>(
        static_cast<size_t>(shape_.batch * shape_.num_heads * shape_.head_size), kFusedOutputSentinel)));
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
  std::vector<int64_t> sparse_tiles_;
  DeviceBuffer query_, pi_signs_, h16_, rot_tables_, mode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, block_tables_, context_lens_;
  DeviceBuffer query_rot_, out_;
  std::vector<float> gate_host_;
  DeviceBuffer gate_;
};

}  // namespace turboquant_host
}  // namespace test
}  // namespace vllm_ascend
