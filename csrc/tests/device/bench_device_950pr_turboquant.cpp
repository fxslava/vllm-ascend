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

// WHAT IS STILL NOT PRICED: the fp32 -> fp16 narrowing between
// npu_turboquant_rotate_q and the native FIA, whose modelled bytes the CSV
// carries as untimed_cast_bytes; and the kUnwrittenSentinel poisoning of
// query_dec_prologue_rot_ between rotation-mode agreement runs, which is what
// lets CompareRotationModes count untouched words.
//
// The sweep, traffic and leg model lives in common/turboquant_audit_scenario,
// the tables and CSV exporters in common/turboquant_audit_report; this file is
// the device side: buffers, planned native operators, launches, timing.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_ops.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"
#include "turboquant_audit_models.hpp"
#include "turboquant_audit_scenario.hpp"
#include "turboquant_audit_report.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName =
    "turboquant_950pr (end-to-end audit: TurboQuant against the native CANN V5 attention operator)";

namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;
namespace tqa = turboquant_audit;

using tqa::Budget;
using tqa::BuildSweep;
using tqa::CaseName;
using tqa::Config;
using tqa::DecodeDispatch;
using tqa::DecodeLegs;
using tqa::HostTiled;
using tqa::kBlockSize;
using tqa::kCausalMaskSide;
using tqa::kLegDecAttnCore;
using tqa::kLegDecE2E;
using tqa::kLegDecFusedQAttnCore;
using tqa::kLegDecFusedQE2E;
using tqa::kLegDecGather;
using tqa::kLegDecNoUnpack;
using tqa::kLegDecRotO;
using tqa::kLegDecRotQ;
using tqa::kLegDecV5;
using tqa::kLegPfAttnCore;
using tqa::kLegPfE2E;
using tqa::kLegPfIngest;
using tqa::kLegPfRotO;
using tqa::kLegPfRotQ;
using tqa::kLegPfV5;
using tqa::kLegPfV5Ingest;
using tqa::kPatternTokens;
using tqa::kRotationModeEnv;
using tqa::kRotationModes;
using tqa::kShortIterations;
using tqa::kShortWarmup;
using tqa::kUnpackEnv;
using tqa::LeadingElements;
using tqa::LegEnabled;
using tqa::ModelTraffic;
using tqa::NativeEnabled;
using tqa::PathLabel;
using tqa::PathMode;
using tqa::PhaseEnabled;
using tqa::PlanDispatch;
using tqa::PrefillLegs;
using tqa::RotationMode;
using tqa::SplitPolicy;
using tqa::TileToDevice;
using tqa::UnpackAblationEnabled;
using tqa::UnpackGatherEnabled;
using tqa::UnpackStandardEnabled;

constexpr int64_t kFiaSparseModeRightDownCausal = 3;
constexpr int64_t kFiaNoPaging = 0;

constexpr double kHbmBudget = 0.85;

constexpr size_t kExactContextElements = 1u << 22;

constexpr double kRotatedBasisMinCosine = 0.999;
constexpr double kDecodeTiePointMinCosine = 0.99;
// The two rotation modes decode the same cache; only rotate_q's Cube rounding (B >= 2) may separate them.
constexpr double kRotationModeMinCosine = 0.9999;
constexpr float kUnwrittenSentinel = -1234.5f;

constexpr int kPipelineBatch = 1;

const AclnnOp& FiaV5() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV5);
  return op;
}
const AclnnOp& FiaV2() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
  return op;
}
const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

class Scenario {
 public:
  Scenario(const Config& config, int64_t aiv_num) : config_(config), aiv_num_(aiv_num) {
    const int64_t d = config.model.head_size;
    const int64_t hq = config.model.num_heads;
    const int64_t hkv = config.model.num_kv_heads;
    const int64_t context = config.context_tokens();
    const int64_t chunk = config.chunk_tokens();
    const int64_t pattern = std::min<int64_t>(context, kPatternTokens);
    const auto elems = [](int64_t a, int64_t b, int64_t c) {
      return static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(c);
    };

    exact_context_ = elems(context, hkv, d) <= kExactContextElements;

    DeterministicRandom rng(0xA53Du +
                            static_cast<uint32_t>(config.seq_len + 7 * config.batch + 31 * d + 127 * hq + 509 * hkv));

    const std::vector<float> key_pattern = rng.NormalHalfExact(elems(pattern, hkv, d), 0.0f, 1.0f);
    const std::vector<float> value_pattern = rng.NormalHalfExact(elems(pattern, hkv, d), 0.0f, 1.0f);
    std::vector<float> key_rot_pattern = key_pattern;
    std::vector<float> value_rot_pattern = value_pattern;
    tqa::RotatePiRowsInPlace(key_rot_pattern, d);
    tqa::RotatePiRowsInPlace(value_rot_pattern, d);

    key_ctx_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    key_ctx_rot_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_rot_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_ctx_, FloatToHalf(key_pattern));
    TileToDevice(value_ctx_, FloatToHalf(value_pattern));
    TileToDevice(key_ctx_rot_, FloatToHalf(key_rot_pattern));
    TileToDevice(value_ctx_rot_, FloatToHalf(value_rot_pattern));

    const int64_t chunk_pattern = std::min<int64_t>(chunk, kPatternTokens);
    const std::vector<float> chunk_key = rng.NormalHalfExact(elems(chunk_pattern, hkv, d), 0.0f, 1.0f);
    const std::vector<float> chunk_value = rng.NormalHalfExact(elems(chunk_pattern, hkv, d), 0.0f, 1.0f);
    key_chunk_ = DeviceBuffer::Empty<Half>(elems(chunk, hkv, d), kBenchmarkAlignBytes);
    value_chunk_ = DeviceBuffer::Empty<Half>(elems(chunk, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_chunk_, FloatToHalf(chunk_key));
    TileToDevice(value_chunk_, FloatToHalf(chunk_value));

    const int64_t query_pattern = std::min<int64_t>(chunk, kPatternTokens);
    std::vector<float> prefill_query = rng.NormalHalfExact(elems(query_pattern, hq, d), 0.0f, 1.0f);
    query_pf_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    TileToDevice(query_pf_, FloatToHalf(prefill_query));
    std::vector<float> prefill_query_rot = prefill_query;
    tqa::RotatePiRowsInPlace(prefill_query_rot, d);
    query_pf_rot_half_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    TileToDevice(query_pf_rot_half_, FloatToHalf(prefill_query_rot));
    query_pf_rot_fp32_ = DeviceBuffer::Empty<float>(elems(chunk, hq, d), kBenchmarkAlignBytes);

    out_pf_tq_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    out_pf_v5_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    out_pf_rot_ = DeviceBuffer::Empty<float>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    lse_pf_tq_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);
    lse_pf_v5_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    const std::vector<float> decode_query = rng.NormalHalfExact(elems(config.batch, hq, d), 0.0f, 1.0f);
    query_dec_ = DeviceBuffer::FromHost(FloatToHalf(decode_query), kBenchmarkAlignBytes);
    query_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    query_dec_prologue_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_tq_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_v5_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    lse_dec_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    const int64_t blocks_per_seq = config.blocks_per_seq();
    const std::vector<int32_t> pool = rng.Permutation(static_cast<int32_t>(config.pool_blocks()));
    std::vector<int32_t> block_table(static_cast<size_t>(config.batch * blocks_per_seq));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t block = 0; block < blocks_per_seq; ++block) {
        block_table[static_cast<size_t>(seq * blocks_per_seq + block)] =
            pool[static_cast<size_t>(seq * blocks_per_seq + block)];
      }
    }
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);

    std::vector<int32_t> slots_full(static_cast<size_t>(context));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t pos = 0; pos < config.seq_len; ++pos) {
        const int32_t block = block_table[static_cast<size_t>(seq * blocks_per_seq + pos / kBlockSize)];
        slots_full[static_cast<size_t>(seq * config.seq_len + pos)] =
            block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(pos % kBlockSize);
      }
    }
    std::vector<int32_t> slots_chunk(static_cast<size_t>(chunk));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t j = 0; j < config.chunk; ++j) {
        const int64_t pos = config.seq_len - config.chunk + j;
        slots_chunk[static_cast<size_t>(seq * config.chunk + j)] =
            slots_full[static_cast<size_t>(seq * config.seq_len + pos)];
      }
    }
    slots_full_ = DeviceBuffer::FromHost(slots_full, kBenchmarkAlignBytes);
    slots_chunk_ = DeviceBuffer::FromHost(slots_chunk, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(config.batch), static_cast<int32_t>(config.seq_len)),
        kBenchmarkAlignBytes);

    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(d), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);

    const int64_t pool_blocks = config.pool_blocks();
    if (config.path == PathMode::kCube) {
      key_cache_ = DeviceBuffer::Empty<int8_t>(tqh::ModePackedCacheBytes(kCubeMode, pool_blocks, kBlockSize, hkv, d),
                                               kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::ModeTables(kCubeMode, d, 1, 0), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(tqh::ModeTables(kCubeMode, d, tqh::kUnpackRows, tqh::kCubeTileRows),
                                              kBenchmarkAlignBytes);
      cube_grid_ = tqh::PlanFusedDecode(config.batch, hq, hkv, d, blocks_per_seq, kBlockSize, aiv_num,
                                        tqh::kFusedContextLimit, SplitPolicy());
      num_splits_ = cube_grid_.num_splits;
      workspace_floats_ = cube_grid_.workspace_floats;
    } else {
      key_cache_ =
          DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, tqh::kTileRows), kBenchmarkAlignBytes);
      aiv_grid_ = tqh::PlanPagedAttention(config.batch, hq, d, blocks_per_seq, kBlockSize, aiv_num);
      num_splits_ = aiv_grid_.num_splits;
      workspace_floats_ = aiv_grid_.workspace_floats;
    }
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ =
        DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(pool_blocks, kBlockSize, hkv), kBenchmarkAlignBytes);
    workspace_ = DeviceBuffer::Empty<float>(workspace_floats_, kBenchmarkAlignBytes);

    context_write_grid_ = tqh::PlanReshapeAndCache(context, aiv_num);
    chunk_write_grid_ = tqh::PlanReshapeAndCache(chunk, aiv_num);

    const size_t cache_elements = elems(pool_blocks * kBlockSize, hkv, d);
    fp16_key_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    fp16_value_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    if (exact_context_) {
      const size_t row = static_cast<size_t>(hkv * d);
      const std::vector<float> key_host = HostTiled(key_pattern, elems(context, hkv, d));
      const std::vector<float> value_host = HostTiled(value_pattern, elems(context, hkv, d));
      std::vector<float> key_cache(cache_elements, 0.0f);
      std::vector<float> value_cache(cache_elements, 0.0f);
      for (int64_t pos = 0; pos < context; ++pos) {
        const size_t slot = static_cast<size_t>(slots_full[static_cast<size_t>(pos)]);
        const size_t src = static_cast<size_t>(pos) * row;
        const size_t dst = slot * row;
        std::copy(key_host.begin() + static_cast<std::ptrdiff_t>(src),
                  key_host.begin() + static_cast<std::ptrdiff_t>(src + row),
                  key_cache.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(value_host.begin() + static_cast<std::ptrdiff_t>(src),
                  value_host.begin() + static_cast<std::ptrdiff_t>(src + row),
                  value_cache.begin() + static_cast<std::ptrdiff_t>(dst));
      }
      fp16_key_cache_.CopyFromHost(FloatToHalf(key_cache).data(), cache_elements * sizeof(Half));
      fp16_value_cache_.CopyFromHost(FloatToHalf(value_cache).data(), cache_elements * sizeof(Half));
    } else {
      TileToDevice(fp16_key_cache_, FloatToHalf(key_pattern));
      TileToDevice(fp16_value_cache_, FloatToHalf(value_pattern));
    }

    std::vector<int8_t> mask(static_cast<size_t>(kCausalMaskSide * kCausalMaskSide), 0);
    for (int64_t row = 0; row < kCausalMaskSide; ++row) {
      for (int64_t col = row + 1; col < kCausalMaskSide; ++col) {
        mask[static_cast<size_t>(row * kCausalMaskSide + col)] = 1;
      }
    }
    mask_ = DeviceBuffer::FromHost(mask, kBenchmarkAlignBytes);

    BuildDescriptors();
    PlanOperators();
  }

  void EnqueuePrefillIngest(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_chunk_.get(), value_chunk_.get(), slots_chunk_.get(), config_.chunk_tokens(),
                      chunk_write_grid_);
  }

  void EnqueuePrefillRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_pf_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_pf_rot_fp32_.get(), config_.chunk_tokens(), config_.model.num_heads, config_.model.head_size,
                     aiv_num_);
  }

  void EnqueuePrefillAttnCore(aclrtStream stream) const { fia_prefill_rotated_->Launch(stream); }

  void EnqueuePrefillRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_pf_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_pf_rot_.get(), config_.chunk_tokens(), config_.model.num_heads, config_.model.head_size,
                     aiv_num_);
  }

  void EnqueuePrefillE2E(aclrtStream stream) const {
    EnqueuePrefillRotateQ(stream);
    EnqueuePrefillAttnCore(stream);
    if (!config_.model.folds_output) {
      EnqueuePrefillRotateO(stream);
    }
  }

  void EnqueuePrefillNative(aclrtStream stream) const { fia_prefill_native_->Launch(stream); }
  void EnqueuePrefillNativeIngest(aclrtStream stream) const { native_write_->Launch(stream); }

  void EnqueueDecodeRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_dec_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size, aiv_num_);
  }

  void EnqueueDecodeFusedCube(aclrtStream stream) const {
    turboquant_mm_fused_decode_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.fused_context_limit, config_.attention_scale(), config_.attention_scale());
  }

  // The unpack ablation of EnqueueDecodeFusedCube: byte-for-byte the same launch on the same buffers, with
  // the KV ingest's codec expand compiled out. Cube path only -- the AIV paged decode has no separate
  // unpack phase to remove, its expand and its dot product are the same instruction stream.
  void EnqueueDecodeNoUnpack(aclrtStream stream) const {
    turboquant_mm_fused_decode_nounpack_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.fused_context_limit, config_.attention_scale(), config_.attention_scale());
  }

  // Option C of tests/research/QJL_3PLUS1_PHASE1.md: byte-for-byte the same launch on the same buffers,
  // with the KV ingest's per-plane Adds replaced by a 16-entry UB Gather. Cube path only, for the same
  // reason the ablation is -- the AIV paged decode has no separate unpack phase.
  void EnqueueDecodeGather(aclrtStream stream) const {
    turboquant_mm_fused_decode_gather_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.fused_context_limit, config_.attention_scale(), config_.attention_scale());
  }

  // The gather expand is a correct variant, not an ablation: its table is the uniform grid, so its operands
  // are the bits the affine expand produces and its output must match the unablated decode exactly. The
  // harness's own checksum only pins a leg against itself across its timed run, so the two are compared
  // here, once per configuration, while the launches are being primed.
  void CheckGatherAgainstBaseline(aclrtStream stream) const {
    EnqueueDecodeAttnCore(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    gather_baseline_checksum_ = DecodeTqChecksum();
    EnqueueDecodeGather(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    gather_checksum_ = DecodeTqChecksum();
    gather_checked_ = true;
  }

  bool gather_checked() const { return gather_checked_; }
  double gather_baseline_checksum() const { return gather_baseline_checksum_; }
  double gather_checksum() const { return gather_checksum_; }

  // The same grid handed the raw fp16 query: the launch rotates every (token, head) into
  // query_dec_prologue_rot_ ahead of its split tasks (on the Cube from 16 vectors), so no rotate_q launch
  // precedes it, and for an unfolded W_o it writes out_dec_tq_ un-rotated, so no rotate_o launch follows.
  void EnqueueDecodeFusedQ(aclrtStream stream) const {
    turboquant_mm_fused_decode_raw_query_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_.get(),
        pi_signs_.get(), rot_tables_.get(), h16_.get(), nullptr, query_dec_prologue_rot_.get(), key_cache_.get(),
        value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(),
        workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.prologue_vectors_per_block, cube_grid_.prologue_cube_chunk_vectors, config_.fused_q_output_stage(),
        cube_grid_.fused_context_limit, config_.attention_scale(), config_.attention_scale());
  }

  // One launch: the in-launch output stage leaves nothing to follow it.
  void EnqueueDecodeFusedQE2E(aclrtStream stream) const { EnqueueDecodeFusedQ(stream); }

  // Both rotation modes over the same cache: the in-launch rotation against rotate_q's word for word, and the
  // two decode outputs in the basis the step hands on (after rotate_o, unfolded) against each other. The two
  // rotations stage the Cube under different rules, so only the outputs are held to a bound.
  struct RotationAgreement {
    size_t rotated_words = 0;
    size_t word_mismatches = 0;
    double output_cosine = 0.0;
  };

  RotationAgreement CompareRotationModes(aclrtStream stream) {
    EnqueueDecodeE2E(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    const std::vector<float> separate_rot = query_dec_rot_.ToHost<float>();
    const std::vector<float> separate_out =
        config_.model.folds_output ? DecodeTqOutput() : out_dec_rot_.ToHost<float>();

    const std::vector<float> unwritten(separate_rot.size(), kUnwrittenSentinel);
    query_dec_prologue_rot_.CopyFromHost(unwritten.data(), unwritten.size() * sizeof(float));
    EnqueueDecodeFusedQ(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    const std::vector<float> fused_rot = query_dec_prologue_rot_.ToHost<float>();

    RotationAgreement agreement;
    agreement.rotated_words = separate_rot.size();
    for (size_t i = 0; i < separate_rot.size(); ++i) {
      agreement.word_mismatches +=
          (i >= fused_rot.size() || std::memcmp(&separate_rot[i], &fused_rot[i], sizeof(float)) != 0) ? 1u : 0u;
    }
    agreement.output_cosine = turboquant_ref::cpu_fidelity(DecodeTqOutput(), separate_out).cosine_similarity;
    return agreement;
  }

  void EnqueueDecodeAttnCore(aclrtStream stream) const {
    if (config_.path == PathMode::kCube) {
      EnqueueDecodeFusedCube(stream);
      return;
    }
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, aiv_grid_.block_dim, query_dec_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_dec_tq_.get(), static_cast<uint32_t>(config_.batch), static_cast<uint32_t>(config_.model.num_heads),
        static_cast<uint32_t>(config_.model.num_kv_heads), static_cast<uint32_t>(config_.model.head_size),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(config_.blocks_per_seq()),
        static_cast<uint32_t>(aiv_grid_.num_splits), aiv_grid_.split_tasks_per_core, aiv_grid_.reduce_tasks_per_core,
        static_cast<uint32_t>(tqh::kFusedContextLimit), config_.attention_scale(), config_.attention_scale());
  }

  void EnqueueDecodeRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_dec_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size, aiv_num_);
  }

  void EnqueueDecodeE2E(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (!config_.model.folds_output) {
      EnqueueDecodeRotateO(stream);
    }
  }

  void EnqueueDecodeNative(aclrtStream stream) const { fia_decode_native_->Launch(stream); }

  void FillCache(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_ctx_.get(), value_ctx_.get(), slots_full_.get(), config_.context_tokens(),
                      context_write_grid_);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  void Prime(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (config_.decodes_in_mode(RotationMode::kFusedPrologue)) {
      EnqueueDecodeFusedQ(stream);
    }
    if (config_.path == PathMode::kCube && UnpackAblationEnabled()) {
      EnqueueDecodeNoUnpack(stream);
    }
    if (config_.path == PathMode::kCube && UnpackGatherEnabled()) {
      EnqueueDecodeGather(stream);
      CheckGatherAgainstBaseline(stream);
    }
    if (prefill_available()) {
      EnqueuePrefillAttnCore(stream);
      EnqueuePrefillNative(stream);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  bool prefill_available() const { return fia_prefill_rotated_ != nullptr && fia_prefill_native_ != nullptr; }
  bool decode_available() const { return fia_decode_native_ != nullptr; }
  bool native_write_available() const { return native_write_ != nullptr; }
  bool exact_context() const { return exact_context_; }
  const std::string& fia_operator() const { return fia_operator_; }
  const std::string& fia_note() const { return fia_note_; }
  const std::string& native_write_note() const { return native_write_note_; }
  int64_t num_splits() const { return num_splits_; }
  uint32_t block_dim() const { return config_.path == PathMode::kCube ? cube_grid_.block_dim : aiv_grid_.block_dim; }

  double ScaleChecksum() const { return bench::ChecksumSum(LeadingElements<float>(scale_plane_)); }
  double PrefillRotatedQueryChecksum() const { return bench::ChecksumSum(LeadingElements<float>(query_pf_rot_fp32_)); }
  double PrefillTqChecksum() const { return bench::ChecksumSum(HalfToFloat(LeadingElements<Half>(out_pf_tq_))); }
  double PrefillNativeChecksum() const { return bench::ChecksumSum(HalfToFloat(LeadingElements<Half>(out_pf_v5_))); }
  double PrefillRotatedOutChecksum() const { return bench::ChecksumSum(LeadingElements<float>(out_pf_rot_)); }
  double NativeCacheChecksum() const {
    return bench::ChecksumSum(HalfToFloat(LeadingElements<Half>(fp16_value_cache_)));
  }
  double DecodeRotatedQueryChecksum() const { return bench::ChecksumSum(LeadingElements<float>(query_dec_rot_)); }
  double DecodeTqChecksum() const { return bench::ChecksumSum(HalfToFloat(LeadingElements<Half>(out_dec_tq_))); }
  double DecodeNativeChecksum() const { return bench::ChecksumSum(HalfToFloat(LeadingElements<Half>(out_dec_v5_))); }
  double DecodeRotatedOutChecksum() const { return bench::ChecksumSum(LeadingElements<float>(out_dec_rot_)); }

  double PrefillPipelineChecksum() const {
    return config_.model.folds_output ? PrefillTqChecksum() : PrefillRotatedOutChecksum();
  }
  double DecodePipelineChecksum() const {
    return config_.model.folds_output ? DecodeTqChecksum() : DecodeRotatedOutChecksum();
  }

  std::vector<float> PrefillTqOutput() const { return HalfToFloat(out_pf_tq_.ToHost<Half>()); }
  std::vector<float> PrefillNativeOutput() const { return HalfToFloat(out_pf_v5_.ToHost<Half>()); }
  std::vector<float> DecodeTqOutput() const { return HalfToFloat(out_dec_tq_.ToHost<Half>()); }
  std::vector<float> DecodeNativeOutput() const { return HalfToFloat(out_dec_v5_.ToHost<Half>()); }

 private:
  static constexpr tqm::TurboQuantMode kCubeMode = tqm::TurboQuantMode::KV4_FP8;

  // Written by CheckGatherAgainstBaseline, which runs from the const Prime.
  mutable double gather_baseline_checksum_ = 0.0;
  mutable double gather_checksum_ = 0.0;
  mutable bool gather_checked_ = false;

  void EnqueueCacheWrite(aclrtStream stream, void* key, void* value, void* slots, int64_t tokens,
                         const tqh::ReshapeAndCacheGrid& grid) const {
    const uint32_t token_count = static_cast<uint32_t>(tokens);
    const uint32_t kv_heads = static_cast<uint32_t>(config_.model.num_kv_heads);
    const uint32_t head_size = static_cast<uint32_t>(config_.model.head_size);
    if (config_.path == PathMode::kCube) {
      turboquant_mm_reshape_and_cache_impl(
          static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, grid.block_dim, key, value, key_cache_.get(),
          value_cache_.get(), scale_plane_.get(), slots, pi_signs_.get(), rot_tables_.get(), write_tables_.get(),
          token_count, kv_heads, head_size, static_cast<uint32_t>(kBlockSize),
          static_cast<uint32_t>(config_.pool_blocks()), grid.tokens_per_core, config_.attention_scale());
      return;
    }
    turboquant_reshape_and_cache_impl(AscendType::FP16, stream, grid.block_dim, key, value, key_cache_.get(),
                                      value_cache_.get(), scale_plane_.get(), slots, pi_signs_.get(),
                                      write_tables_.get(), token_count, kv_heads, head_size,
                                      static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(config_.pool_blocks()),
                                      grid.tokens_per_core, config_.attention_scale());
  }

  void BuildDescriptors() {
    const int64_t d = config_.model.head_size;
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const int64_t chunk = config_.chunk_tokens();
    const int64_t context = config_.context_tokens();
    const int64_t pool_blocks = config_.pool_blocks();

    query_pf_tnd_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_pf_.get()));
    query_pf_rot_tnd_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_pf_rot_half_.get()));
    key_ctx_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_.get()));
    value_ctx_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_.get()));
    key_ctx_rot_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_rot_.get()));
    value_ctx_rot_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_rot_.get()));
    mask_tensor_.reset(new AclnnTensor({kCausalMaskSide, kCausalMaskSide}, ACL_INT8, mask_.get()));
    out_pf_tq_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_pf_tq_.get()));
    out_pf_v5_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_pf_v5_.get()));
    lse_pf_tq_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_pf_tq_.get()));
    lse_pf_v5_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_pf_v5_.get()));

    key_ctx_list_.reset(new AclnnTensorList({key_ctx_tnd_->get()}));
    value_ctx_list_.reset(new AclnnTensorList({value_ctx_tnd_->get()}));
    key_ctx_rot_list_.reset(new AclnnTensorList({key_ctx_rot_tnd_->get()}));
    value_ctx_rot_list_.reset(new AclnnTensorList({value_ctx_rot_tnd_->get()}));

    std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
    std::vector<int64_t> cumulative_kv(static_cast<size_t>(config_.batch));
    for (int64_t seq = 0; seq < config_.batch; ++seq) {
      cumulative_q[static_cast<size_t>(seq)] = (seq + 1) * config_.chunk;
      cumulative_kv[static_cast<size_t>(seq)] = (seq + 1) * config_.seq_len;
    }
    prefill_seq_q_.reset(new AclnnIntArray(cumulative_q));
    prefill_seq_kv_.reset(new AclnnIntArray(cumulative_kv));

    query_dec_tnd_.reset(new AclnnTensor({config_.batch, hq, d}, ACL_FLOAT16, query_dec_.get()));
    out_dec_v5_tensor_.reset(new AclnnTensor({config_.batch, hq, d}, ACL_FLOAT16, out_dec_v5_.get()));
    lse_dec_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_dec_.get()));
    fp16_key_flat_.reset(
        new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16, fp16_key_cache_.get()));
    fp16_value_flat_.reset(
        new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16, fp16_value_cache_.get()));
    fp16_key_list_.reset(new AclnnTensorList({fp16_key_flat_->get()}));
    fp16_value_list_.reset(new AclnnTensorList({fp16_value_flat_->get()}));
    block_table_tensor_.reset(
        new AclnnTensor({config_.batch, config_.blocks_per_seq()}, ACL_INT32, block_tables_.get()));
    {
      std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
      for (int64_t seq = 0; seq < config_.batch; ++seq) {
        cumulative_q[static_cast<size_t>(seq)] = seq + 1;
      }
      decode_seq_q_.reset(new AclnnIntArray(cumulative_q));
      std::vector<int64_t> kv_lens(static_cast<size_t>(config_.batch), config_.seq_len);
      decode_seq_kv_.reset(new AclnnIntArray(kv_lens));
    }

    key_chunk_tensor_.reset(new AclnnTensor({chunk, hkv, d}, ACL_FLOAT16, key_chunk_.get()));
    value_chunk_tensor_.reset(new AclnnTensor({chunk, hkv, d}, ACL_FLOAT16, value_chunk_.get()));
    slots_chunk_tensor_.reset(new AclnnTensor({chunk}, ACL_INT32, slots_chunk_.get()));
    fp16_key_cache_tensor_.reset(
        new AclnnTensor({pool_blocks, kBlockSize, hkv, d}, ACL_FLOAT16, fp16_key_cache_.get()));
    fp16_value_cache_tensor_.reset(
        new AclnnTensor({pool_blocks, kBlockSize, hkv, d}, ACL_FLOAT16, fp16_value_cache_.get()));
  }

  void PlanOperators() {
    if (!NativeEnabled()) {
      fia_note_ = "the native V5 legs were dropped by ASCEND_BENCH_TQ_FIA=0";
      return;
    }
    fia_prefill_rotated_ = PlanPrefill(query_pf_rot_tnd_->get(), key_ctx_rot_list_->get(), value_ctx_rot_list_->get(),
                                       out_pf_tq_tensor_->get(), lse_pf_tq_tensor_->get());
    fia_prefill_native_ = PlanPrefill(query_pf_tnd_->get(), key_ctx_list_->get(), value_ctx_list_->get(),
                                      out_pf_v5_tensor_->get(), lse_pf_v5_tensor_->get());
    fia_decode_native_ = PlanDecode();
    try {
      native_write_.reset(new PlannedOp(PlanAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
          ScatterPaKvCacheOp(), key_chunk_tensor_->get(), fp16_key_cache_tensor_->get(), slots_chunk_tensor_->get(),
          value_chunk_tensor_->get(), fp16_value_cache_tensor_->get(), nullptr, nullptr, nullptr,
          const_cast<char*>(ops::kScatterCacheModeNorm), nullptr, nullptr, nullptr)));
    } catch (const AclError& error) {
      native_write_note_ = std::string(ops::kScatterPaKvCache) + ": " + error.what();
    }
  }

  std::unique_ptr<PlannedOp> PlanPrefill(const aclTensor* query, const aclTensorList* key_list,
                                         const aclTensorList* value_list, const aclTensor* out, const aclTensor* lse) {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query, key_list, value_list, nullptr, mask_tensor_->get(), prefill_seq_q_->get(),
          prefill_seq_kv_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault,
          kFiaNoPaging, 0, false, 0, 0, s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (prompt role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query, key_list, value_list, nullptr, mask_tensor_->get(), prefill_seq_q_->get(),
          prefill_seq_kv_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hq, scale, s950::kFiaUnboundedTokens,
          s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv, kFiaSparseModeRightDownCausal,
          s950::kFiaInnerPreciseDefault, kFiaNoPaging, 0, false, 0, 0, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV2);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV2) + " (prompt role): " + error.what());
    }
    return nullptr;
  }

  std::unique_ptr<PlannedOp> PlanDecode() {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), nullptr, nullptr,
          decode_seq_q_->get(), decode_seq_kv_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          block_table_tensor_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hq, scale, s950::kFiaUnboundedTokens,
          s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv, s950::kFiaSparseModeNone,
          s950::kFiaInnerPreciseDefault, kBlockSize, 0, false, 0, 0, s950::kFiaQueryQuantModeNone,
          s950::kFiaPseTypeDefault, out_dec_v5_tensor_->get(), lse_dec_tensor_->get())));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (incremental role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), nullptr, nullptr,
          decode_seq_q_->get(), decode_seq_kv_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          block_table_tensor_->get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd),
          hkv, s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault, kBlockSize, 0, false, 0, 0,
          out_dec_v5_tensor_->get(), lse_dec_tensor_->get())));
      RecordOperator(ops950::kFusedInferAttentionScoreV2);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV2) + " (incremental role): " + error.what());
    }
    return nullptr;
  }

  void RecordOperator(const char* name) {
    if (fia_operator_.empty()) {
      fia_operator_ = name;
    }
  }
  void AppendNote(const std::string& note) {
    if (!fia_note_.empty()) {
      fia_note_ += "; ";
    }
    fia_note_ += note;
  }

  Config config_;
  int64_t aiv_num_ = 0;
  bool exact_context_ = false;
  int64_t num_splits_ = 1;
  size_t workspace_floats_ = 0;

  tqh::ReshapeAndCacheGrid context_write_grid_;
  tqh::ReshapeAndCacheGrid chunk_write_grid_;
  tqh::PagedAttentionGrid aiv_grid_;
  tqh::FusedDecodeGrid cube_grid_;

  DeviceBuffer key_ctx_, value_ctx_, key_ctx_rot_, value_ctx_rot_;
  DeviceBuffer key_chunk_, value_chunk_;
  DeviceBuffer query_pf_, query_pf_rot_half_, query_pf_rot_fp32_;
  DeviceBuffer out_pf_tq_, out_pf_v5_, out_pf_rot_, lse_pf_tq_, lse_pf_v5_;
  DeviceBuffer query_dec_, query_dec_rot_, query_dec_prologue_rot_;
  DeviceBuffer out_dec_tq_, out_dec_v5_, out_dec_rot_, lse_dec_;
  DeviceBuffer slots_full_, slots_chunk_, block_tables_, context_lens_;
  DeviceBuffer pi_signs_, h16_, rot_tables_, write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_;
  DeviceBuffer fp16_key_cache_, fp16_value_cache_;
  DeviceBuffer mask_;

  std::unique_ptr<AclnnTensor> query_pf_tnd_, query_pf_rot_tnd_;
  std::unique_ptr<AclnnTensor> key_ctx_tnd_, value_ctx_tnd_, key_ctx_rot_tnd_, value_ctx_rot_tnd_;
  std::unique_ptr<AclnnTensor> mask_tensor_, out_pf_tq_tensor_, out_pf_v5_tensor_;
  std::unique_ptr<AclnnTensor> lse_pf_tq_tensor_, lse_pf_v5_tensor_;
  std::unique_ptr<AclnnTensor> query_dec_tnd_, out_dec_v5_tensor_, lse_dec_tensor_;
  std::unique_ptr<AclnnTensor> fp16_key_flat_, fp16_value_flat_, block_table_tensor_;
  std::unique_ptr<AclnnTensor> key_chunk_tensor_, value_chunk_tensor_, slots_chunk_tensor_;
  std::unique_ptr<AclnnTensor> fp16_key_cache_tensor_, fp16_value_cache_tensor_;
  std::unique_ptr<AclnnTensorList> key_ctx_list_, value_ctx_list_, key_ctx_rot_list_, value_ctx_rot_list_;
  std::unique_ptr<AclnnTensorList> fp16_key_list_, fp16_value_list_;
  std::unique_ptr<AclnnIntArray> prefill_seq_q_, prefill_seq_kv_, decode_seq_q_, decode_seq_kv_;

  std::string fia_operator_;
  std::string fia_note_;
  std::string native_write_note_;
  std::unique_ptr<PlannedOp> fia_prefill_rotated_, fia_prefill_native_, fia_decode_native_, native_write_;
};

class RunnerSet {
 public:
  RunnerSet(BenchmarkRunner& primary, const std::vector<Budget>& budgets) : primary_(&primary) {
    if (budgets.empty()) {
      budgets_.push_back(Budget{kShortWarmup, kShortIterations});
    } else {
      budgets_ = budgets;
    }
    BenchmarkOptions options = Options(primary.options(), budgets_.front());
    primary.set_options(options);
    runners_.push_back(&primary);
    for (size_t index = 1; index < budgets_.size(); ++index) {
      std::ostringstream name;
      name << kSuiteName << " -- warmup " << budgets_[index].warmup << ", " << budgets_[index].iterations
           << " iterations";
      owned_.emplace_back(
          new BenchmarkRunner(name.str(), Options(primary.options(), budgets_[index]), primary.stream()));
      runners_.push_back(owned_.back().get());
    }
  }

  BenchmarkRunner& For(const Budget& budget) const {
    for (size_t index = 0; index < budgets_.size(); ++index) {
      if (budgets_[index] == budget) {
        return *runners_[index];
      }
    }
    return *primary_;
  }

  const std::vector<BenchmarkRunner*>& all() const { return runners_; }

  void Fail(const std::string& name, const std::string& why) const { primary_->RecordFailure(name, why); }

  void MirrorFail(const BenchmarkRunner& runner, const std::string& name, const std::string& why) const {
    if (&runner != primary_) {
      primary_->RecordFailure(name, why);
    }
  }

  void ReportExtras() const {
    for (const std::unique_ptr<BenchmarkRunner>& runner : owned_) {
      runner->Report();
    }
  }

 private:
  static BenchmarkOptions Options(const BenchmarkOptions& base, const Budget& budget) {
    BenchmarkOptions options = base;
    options.warmup_iterations = budget.warmup;
    options.timed_iterations = budget.iterations;
    options.pipeline_batch = kPipelineBatch;
    return options;
  }

  BenchmarkRunner* primary_;
  std::vector<Budget> budgets_;
  std::vector<std::unique_ptr<BenchmarkRunner>> owned_;
  std::vector<BenchmarkRunner*> runners_;
};

double RotatedBasisCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueuePrefillAttnCore(stream);
  scenario.EnqueuePrefillNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> plain = scenario.PrefillNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.PrefillTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, plain).cosine_similarity;
}

double DecodeTiePointCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueueDecodeE2E(stream);
  scenario.EnqueueDecodeNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> native = scenario.DecodeNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.DecodeTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, native).cosine_similarity;
}

}  // namespace

void BuildSuite(BenchmarkRunner& primary) {
  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);

  const std::vector<Config> sweep = BuildSweep();
  tqa::PrintBanner(sweep, aiv_num, aiv_queried);
  if (sweep.empty()) {
    return;
  }

  std::vector<Budget> budgets;
  for (const Config& config : sweep) {
    if (std::find(budgets.begin(), budgets.end(), config.budget) == budgets.end()) {
      budgets.push_back(config.budget);
    }
  }
  RunnerSet runners(primary, budgets);
  for (const Budget& budget : budgets) {
    std::printf("[ascend-bench]   budget: warmup %d, %d timed iterations, pipeline_batch %d\n", budget.warmup,
                budget.iterations, kPipelineBatch);
  }
  std::fflush(stdout);

  const bool run_prefill = PhaseEnabled("prefill");
  const bool run_decode = PhaseEnabled("decode");

  size_t basis_check = sweep.size();
  size_t tie_point_check = sweep.size();
  for (size_t index = 0; index < sweep.size(); ++index) {
    const auto weight = [](const Config& config) {
      return config.chunk_tokens() * config.model.num_heads * config.model.head_size;
    };
    if (basis_check == sweep.size() || weight(sweep[index]) < weight(sweep[basis_check])) {
      basis_check = index;
    }
    const bool exact = static_cast<size_t>(sweep[index].context_tokens() * sweep[index].model.num_kv_heads *
                                           sweep[index].model.head_size) <= kExactContextElements;
    if (exact &&
        (tie_point_check == sweep.size() || sweep[index].context_tokens() < sweep[tie_point_check].context_tokens())) {
      tie_point_check = index;
    }
  }

  std::vector<tqa::Traffic> traffic;
  traffic.reserve(sweep.size());
  std::string fia_operator = "none";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    BenchmarkRunner& runner = runners.For(config.budget);

    const DecodeDispatch dispatch = PlanDispatch(config, aiv_num);
    traffic.push_back(ModelTraffic(config, dispatch));

    size_t free_hbm = 0;
    size_t total_hbm = 0;
    const bool known = aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm) == ACL_SUCCESS && free_hbm > 0;
    const double needed = tqa::ScenarioBytes(config);
    if (known && needed > kHbmBudget * static_cast<double>(free_hbm)) {
      std::ostringstream why;
      why << "needs ~" << needed / (1024.0 * 1024.0 * 1024.0) << " GiB of HBM; the budget is " << kHbmBudget * 100.0
          << "% of " << static_cast<double>(free_hbm) / (1024.0 * 1024.0 * 1024.0) << " GiB free";
      for (const char* leg : PrefillLegs()) {
        runner.Skip(CaseName(leg, config), why.str());
      }
      for (const char* leg : DecodeLegs()) {
        runner.Skip(CaseName(leg, config), why.str());
      }
      std::printf("[ascend-bench] %s: skipped -- %s\n", config.id().c_str(), why.str().c_str());
      std::fflush(stdout);
      continue;
    }

    std::unique_ptr<Scenario> scenario;
    try {
      scenario.reset(new Scenario(config, aiv_num));
    } catch (const std::exception& error) {
      runners.Fail(CaseName("setup", config), error.what());
      continue;
    }
    if (!scenario->fia_operator().empty()) {
      fia_operator = scenario->fia_operator();
    }
    std::printf(
        "\n[ascend-bench] %-17s S=%lld B=%lld D=%lld H_Q=%lld H_KV=%lld | %s path, chunk %lld, "
        "blocks/seq %lld, pool %lld, Cfg [K/Tsk/Blk/Red] %s of %lld blocks | native: %s\n",
        config.model.label, static_cast<long long>(config.seq_len), static_cast<long long>(config.batch),
        static_cast<long long>(config.model.head_size), static_cast<long long>(config.model.num_heads),
        static_cast<long long>(config.model.num_kv_heads), PathLabel(config.path), static_cast<long long>(config.chunk),
        static_cast<long long>(config.blocks_per_seq()), static_cast<long long>(config.pool_blocks()),
        dispatch.label().c_str(), static_cast<long long>(dispatch.device_blocks),
        scenario->decode_available() ? scenario->fia_operator().c_str() : "unavailable");
    if (scenario->num_splits() != dispatch.split_k ||
        static_cast<int64_t>(scenario->block_dim()) != dispatch.active_blocks) {
      runners.Fail(CaseName("dispatch", config), "the scenario launches a different grid than Table B reports");
    }
    if (!scenario->prefill_available() || !scenario->decode_available()) {
      std::printf("[ascend-bench]   native operator note: %s\n", scenario->fia_note().c_str());
    }
    std::fflush(stdout);

    try {
      scenario->FillCache(runner.stream());
      scenario->Prime(runner.stream());
    } catch (const std::exception& error) {
      runners.Fail(CaseName("fill", config), error.what());
      continue;
    }

    if (index == basis_check && scenario->prefill_available()) {
      try {
        const double cosine = RotatedBasisCosine(*scenario, config, runner.stream());
        std::printf(
            "[ascend-bench]   rotated-basis identity: FIA(Q~,K~,V~) un-rotated vs FIA(Q,K,V), "
            "cos = %.9f (bound %.4f)\n",
            cosine, kRotatedBasisMinCosine);
        if (!(cosine > kRotatedBasisMinCosine)) {
          runners.Fail(CaseName("rotated_basis_identity", config),
                       "attention in the rotated basis does not un-rotate to attention in the plain one");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("rotated_basis_identity", config), error.what());
      }
    }
    if (index == tie_point_check && scenario->decode_available() && scenario->exact_context()) {
      try {
        const double cosine = DecodeTiePointCosine(*scenario, config, runner.stream());
        std::printf(
            "[ascend-bench]   decode tie-point: TurboQuant un-rotated vs native V5 over the same "
            "context, cos = %.6f (bound %.2f)\n",
            cosine, kDecodeTiePointMinCosine);
        if (!(cosine > kDecodeTiePointMinCosine)) {
          runners.Fail(CaseName("decode_tie_point", config),
                       "the TurboQuant decode and the native decode are not attending over the same context");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("decode_tie_point", config), error.what());
      }
    }
    // Every configuration that times the raw-query decode first checks it against the pre-rotated one: the
    // raw entry had executed on the camodel at B = 1, D = 256 only.
    if (run_decode && config.decodes_in_mode(RotationMode::kFusedPrologue)) {
      try {
        const Scenario::RotationAgreement agreement = scenario->CompareRotationModes(runner.stream());
        std::printf(
            "[ascend-bench]   rotation modes: the in-launch rotation differs from rotate_q in %zu of %zu "
            "fp32 words; decode outputs agree to cos = %.9f (bound %.4f)\n",
            agreement.word_mismatches, agreement.rotated_words, agreement.output_cosine, kRotationModeMinCosine);
        if (!(agreement.output_cosine > kRotationModeMinCosine)) {
          runners.Fail(CaseName("rotation_mode_agreement", config),
                       "the raw-query decode does not reproduce the pre-rotated decode");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("rotation_mode_agreement", config), error.what());
      }
    }
    std::fflush(stdout);

    const tqa::Traffic& model = traffic.back();
    const Scenario* sc = scenario.get();
    const auto run_leg = [&](const char* leg, double flops, double bytes, int tasks,
                             std::function<void(aclrtStream)> launch, std::function<double()> checksum) {
      const std::string name = CaseName(leg, config);
      if (!LegEnabled(leg)) {
        runner.Skip(name, "not in ASCEND_BENCH_TQ_AUDIT_LEGS");
        return;
      }
      try {
        BenchmarkCase benchmark_case;
        benchmark_case.name = name;
        benchmark_case.flops_per_iteration = flops;
        benchmark_case.bytes_per_iteration = bytes;
        benchmark_case.tasks_per_launch = tasks;
        benchmark_case.launch = std::move(launch);
        benchmark_case.checksum = std::move(checksum);
        runner.Run(benchmark_case);
      } catch (const std::exception& error) {
        runner.RecordFailure(name, error.what());
        runners.MirrorFail(runner, name, error.what());
      }
    };
    const int rot_o_tasks = config.model.folds_output ? 0 : 1;

    if (run_decode) {
      const bool separate = config.decodes_in_mode(RotationMode::kSeparate);
      const bool fused_q = config.decodes_in_mode(RotationMode::kFusedPrologue);
      const std::string separate_off = std::string("rotation mode separate is not selected by ") + kRotationModeEnv;
      const std::string fused_q_off =
          config.path != PathMode::kCube
              ? std::string("the raw-query decode exists on the Cube path only")
              : std::string("rotation mode fused_prologue is not selected by ") + kRotationModeEnv;

      if (separate) {
        run_leg(
            kLegDecRotQ, 0.0, model.dec_rot_q, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateQ(s); },
            [sc]() { return sc->DecodeRotatedQueryChecksum(); });
        if (UnpackStandardEnabled()) {
          run_leg(
              kLegDecAttnCore, model.dec_flops, model.dec_attn_core, 1,
              [sc](aclrtStream s) { sc->EnqueueDecodeAttnCore(s); }, [sc]() { return sc->DecodeTqChecksum(); });
        } else {
          runner.Skip(CaseName(kLegDecAttnCore, config),
                      std::string("the unablated decode is not selected by ") + kUnpackEnv);
        }
      } else {
        runner.Skip(CaseName(kLegDecRotQ, config), separate_off);
        runner.Skip(CaseName(kLegDecAttnCore, config), separate_off);
      }

      // The ablated twin of kLegDecAttnCore. It runs under the same rotation mode and the same traffic
      // model, so the two rows differ in the unpack phase and nothing else. Its checksum is recorded like
      // any other leg's and is meaningless on purpose: the launch stages zeros.
      if (!UnpackAblationEnabled()) {
        runner.Skip(CaseName(kLegDecNoUnpack, config),
                    std::string("the unpack ablation is not selected by ") + kUnpackEnv);
      } else if (config.path != PathMode::kCube) {
        runner.Skip(CaseName(kLegDecNoUnpack, config), "the unpack ablation exists on the Cube path only");
      } else if (!separate) {
        runner.Skip(CaseName(kLegDecNoUnpack, config), separate_off);
      } else {
        run_leg(
            kLegDecNoUnpack, model.dec_flops, model.dec_attn_core, 1,
            [sc](aclrtStream s) { sc->EnqueueDecodeNoUnpack(s); }, [sc]() { return sc->DecodeTqChecksum(); });
      }

      // Option C's twin of kLegDecAttnCore. Same rotation mode, same traffic model, same launch shape, so
      // the rows differ in the expand's instruction sequence and nothing else. Unlike the ablation its
      // output is the real one, and the equality against the unablated decode is reported before it runs.
      if (!UnpackGatherEnabled()) {
        runner.Skip(CaseName(kLegDecGather, config), std::string("the gather expand is not selected by ") + kUnpackEnv);
      } else if (config.path != PathMode::kCube) {
        runner.Skip(CaseName(kLegDecGather, config), "the gather expand exists on the Cube path only");
      } else if (!separate) {
        runner.Skip(CaseName(kLegDecGather, config), separate_off);
      } else {
        if (sc->gather_checked()) {
          const double baseline = sc->gather_baseline_checksum();
          const double gathered = sc->gather_checksum();
          std::printf("[ascend-bench]   %s: checksum %.9g against the unablated decode's %.9g -- %s\n",
                      CaseName(kLegDecGather, config).c_str(), gathered, baseline,
                      gathered == baseline ? "equal, as the uniform table requires"
                                           : "DIFFERENT, so the gather expand is not decoding the same grid");
          std::fflush(stdout);
        }
        run_leg(
            kLegDecGather, model.dec_flops, model.dec_attn_core, 1, [sc](aclrtStream s) { sc->EnqueueDecodeGather(s); },
            [sc]() { return sc->DecodeTqChecksum(); });
      }

      if (fused_q) {
        run_leg(
            kLegDecFusedQAttnCore, model.dec_flops, model.dec_fused_q_attn_core, 1,
            [sc](aclrtStream s) { sc->EnqueueDecodeFusedQ(s); }, [sc]() { return sc->DecodeTqChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecFusedQAttnCore, config), fused_q_off);
      }

      if (config.model.folds_output) {
        runner.Skip(CaseName(kLegDecRotO, config),
                    "W_o is folded: the de-rotation is an offline weight transform and costs 0.0 us at runtime");
      } else {
        run_leg(
            kLegDecRotO, 0.0, model.dec_rot_o, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateO(s); },
            [sc]() { return sc->DecodeRotatedOutChecksum(); });
      }

      if (separate) {
        run_leg(
            kLegDecE2E, model.dec_flops, model.dec_e2e, config.decode_launches(RotationMode::kSeparate),
            [sc](aclrtStream s) { sc->EnqueueDecodeE2E(s); }, [sc]() { return sc->DecodePipelineChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecE2E, config), separate_off);
      }

      if (fused_q) {
        run_leg(
            kLegDecFusedQE2E, model.dec_flops, model.dec_fused_q_e2e,
            config.decode_launches(RotationMode::kFusedPrologue),
            [sc](aclrtStream s) { sc->EnqueueDecodeFusedQE2E(s); }, [sc]() { return sc->DecodeTqChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecFusedQE2E, config), fused_q_off);
      }

      if (!sc->decode_available()) {
        runner.Skip(CaseName(kLegDecV5, config), sc->fia_note());
      } else {
        run_leg(
            kLegDecV5, model.dec_flops, model.dec_v5, 1, [sc](aclrtStream s) { sc->EnqueueDecodeNative(s); },
            [sc]() { return sc->DecodeNativeChecksum(); });
      }
    }

    if (run_prefill) {
      run_leg(
          kLegPfIngest, 0.0, model.pf_ingest, 1, [sc](aclrtStream s) { sc->EnqueuePrefillIngest(s); },
          [sc]() { return sc->ScaleChecksum(); });
      run_leg(
          kLegPfRotQ, 0.0, model.pf_rot_q, 1, [sc](aclrtStream s) { sc->EnqueuePrefillRotateQ(s); },
          [sc]() { return sc->PrefillRotatedQueryChecksum(); });

      if (!sc->prefill_available()) {
        for (const char* leg : {kLegPfAttnCore, kLegPfRotO, kLegPfE2E, kLegPfV5}) {
          runner.Skip(CaseName(leg, config), sc->fia_note());
        }
      } else {
        run_leg(
            kLegPfAttnCore, model.pf_flops, model.pf_attn_core, 1,
            [sc](aclrtStream s) { sc->EnqueuePrefillAttnCore(s); }, [sc]() { return sc->PrefillTqChecksum(); });

        if (config.model.folds_output) {
          runner.Skip(CaseName(kLegPfRotO, config),
                      "W_o is folded: the de-rotation is an offline weight transform and costs 0.0 us at runtime");
        } else {
          run_leg(
              kLegPfRotO, 0.0, model.pf_rot_o, 1, [sc](aclrtStream s) { sc->EnqueuePrefillRotateO(s); },
              [sc]() { return sc->PrefillRotatedOutChecksum(); });
        }

        run_leg(
            kLegPfE2E, model.pf_flops, model.pf_e2e, 1 + 1 + rot_o_tasks,
            [sc](aclrtStream s) { sc->EnqueuePrefillE2E(s); }, [sc]() { return sc->PrefillPipelineChecksum(); });
        run_leg(
            kLegPfV5, model.pf_flops, model.pf_v5, 1, [sc](aclrtStream s) { sc->EnqueuePrefillNative(s); },
            [sc]() { return sc->PrefillNativeChecksum(); });
      }

      if (!sc->native_write_available()) {
        runner.Skip(CaseName(kLegPfV5Ingest, config), sc->native_write_note().empty()
                                                          ? std::string("the native write was not planned")
                                                          : sc->native_write_note());
      } else {
        run_leg(
            kLegPfV5Ingest, 0.0, model.pf_v5_ingest, 1, [sc](aclrtStream s) { sc->EnqueuePrefillNativeIngest(s); },
            [sc]() { return sc->NativeCacheChecksum(); });
      }
    }
  }

  runners.ReportExtras();

  if (run_prefill) {
    tqa::PrintTableA(runners.all(), sweep, traffic, fia_operator);
    const std::string path = env::String("ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV");
    if (!path.empty()) {
      tqa::WritePrefillCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  if (run_decode) {
    tqa::PrintTableB(runners.all(), sweep, traffic, fia_operator, aiv_num);
    const std::string path = env::String("ASCEND_BENCH_TQ_AUDIT_DECODE_CSV");
    if (!path.empty()) {
      tqa::WriteDecodeCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  std::printf("\n[ascend-bench] the raw per-case table for the first iteration budget follows below.\n");
  std::fflush(stdout);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend