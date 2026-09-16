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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;
namespace tq = turboquant_ref;
namespace tqm = vllm_ascend::turboquant;

#ifdef VLLM_ASCEND_TEST_TIER_DEVICE
constexpr int64_t kNumHeads = 8;
#else
constexpr int64_t kNumHeads = 4;
#endif
constexpr int64_t kHeadSize = 256;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kMinBlocks = 4;

#ifdef VLLM_ASCEND_TEST_TIER_DEVICE
const int64_t kDefaultContextLens[] = {64, 512, 1024, 2048};
const int64_t kDefaultBatchSizes[] = {1, 8};
#else
const int64_t kDefaultContextLens[] = {16};
const int64_t kDefaultBatchSizes[] = {1};
#endif

std::vector<int64_t> Int64ListFromEnv(const char* name, const int64_t* fallback, size_t fallback_count) {
  const std::vector<int64_t> defaults(fallback, fallback + fallback_count);
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  std::stringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const long long value = std::strtoll(field.c_str(), nullptr, 10);
    if (value > 0) {
      parsed.push_back(static_cast<int64_t>(value));
    }
  }
  if (parsed.empty()) {
    std::printf("[ multimode ] %s='%s' parsed to nothing; using the default sweep\n", name, raw);
    return defaults;
  }
  return parsed;
}

const std::vector<int64_t>& ContextLens() {
  static const std::vector<int64_t> value =
      Int64ListFromEnv("ASCEND_TQ_SIM_CONTEXT", kDefaultContextLens,
                       sizeof(kDefaultContextLens) / sizeof(kDefaultContextLens[0]));
  return value;
}

const std::vector<int64_t>& BatchSizes() {
  static const std::vector<int64_t> value = Int64ListFromEnv(
      "ASCEND_TQ_SIM_BATCH", kDefaultBatchSizes, sizeof(kDefaultBatchSizes) / sizeof(kDefaultBatchSizes[0]));
  return value;
}

int64_t BlocksPerSeq(int64_t context_len) { return (context_len + kBlockSize - 1) / kBlockSize; }

int64_t NumBlocks(int64_t batch, int64_t context_len) {
  return std::max(kMinBlocks, batch * BlocksPerSeq(context_len) * 4);
}
constexpr float kInvSqrtHeadSize = 0.0625f;
constexpr float kAttentionScale = kInvSqrtHeadSize;

constexpr double kSmokeCos = 0.90;

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (na <= 0.0 || nb <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<float> HostAttention(int64_t batch, int64_t context_len, const std::vector<float>& query,
                                 const std::vector<float>& key, const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(batch * kNumHeads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t query_base = b * kNumHeads * kHeadSize;
    const int64_t kv_base = b * context_len * kNumKvHeads * kHeadSize;
    for (int64_t h = 0; h < kNumHeads; ++h) {
      const int64_t kv = h / heads_per_kv;
      std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
      double max_logit = -1e30;
      for (int64_t t = 0; t < context_len; ++t) {
        double dot = 0.0;
        for (int64_t d = 0; d < kHeadSize; ++d) {
          dot += static_cast<double>(query[static_cast<size_t>(query_base + h * kHeadSize + d)]) *
                 static_cast<double>(key[static_cast<size_t>(kv_base + (t * kNumKvHeads + kv) * kHeadSize + d)]);
        }
        logits[static_cast<size_t>(t)] = dot * static_cast<double>(kAttentionScale);
        max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
      }
      double denom = 0.0;
      for (int64_t t = 0; t < context_len; ++t) {
        logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
        denom += logits[static_cast<size_t>(t)];
      }
      for (int64_t t = 0; t < context_len; ++t) {
        const double w = logits[static_cast<size_t>(t)] / denom;
        for (int64_t d = 0; d < kHeadSize; ++d) {
          out[static_cast<size_t>(query_base + h * kHeadSize + d)] += static_cast<float>(
              w * static_cast<double>(value[static_cast<size_t>(kv_base + (t * kNumKvHeads + kv) * kHeadSize + d)]));
        }
      }
    }
  }
  return out;
}

struct Shape {
  int64_t batch = 1;
  int64_t context_len = 0;
  int64_t blocks_per_seq = 0;
  int64_t num_blocks = 0;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> query;
  std::vector<int32_t> slots;
  std::vector<int32_t> block_table;
  std::vector<int32_t> context_lens;
};

struct ModeRun {
  std::vector<float> output;
  std::vector<float> scale_plane;
  int64_t num_splits = 0;
  int64_t packed_bytes = 0;
  vllm_ascend::turboquant::RotateQPlan rotate_plan;
};

ModeRun RunMode(tqm::TurboQuantMode mode, const Shape& shape, aclrtStream stream) {
  ModeRun run;
  run.packed_bytes = tqh::ModePackedBytes(mode, kHeadSize);
  const int64_t batch = shape.batch;
  const int64_t context_len = shape.context_len;

  DeviceBuffer key_dev = DeviceBuffer::FromHost(FloatToHalf(shape.key));
  DeviceBuffer value_dev = DeviceBuffer::FromHost(FloatToHalf(shape.value));
  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(shape.query));
  DeviceBuffer slots_dev = DeviceBuffer::FromHost(shape.slots);
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(shape.block_table);
  DeviceBuffer context_dev = DeviceBuffer::FromHost(shape.context_lens);
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));

  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer write_tables = DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, 1, 0));
  DeviceBuffer decode_tables =
      DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows));

  DeviceBuffer key_cache = DeviceBuffer::Empty<int8_t>(
      tqh::ModePackedCacheBytes(mode, shape.num_blocks, kBlockSize, kNumKvHeads, kHeadSize));
  DeviceBuffer value_cache = DeviceBuffer::Empty<int8_t>(key_cache.size_bytes());
  DeviceBuffer scale_plane =
      DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(shape.num_blocks, kBlockSize, kNumKvHeads));
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(batch * kNumHeads * kHeadSize));
  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(batch * kNumHeads * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(batch * context_len, aiv_num);
  const tqh::FusedDecodeGrid decode_grid = tqh::PlanFusedDecode(batch, kNumHeads, kNumKvHeads, kHeadSize,
                                                                shape.blocks_per_seq, kBlockSize, aiv_num);
  run.num_splits = decode_grid.num_splits;

  DeviceBuffer workspace = DeviceBuffer::Empty<float>(decode_grid.workspace_floats);

  turboquant_mm_reshape_and_cache_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), slots_dev.get(), pi_signs.get(), rot_tables.get(),
      write_tables.get(), static_cast<uint32_t>(batch * context_len), static_cast<uint32_t>(kNumKvHeads),
      static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), write_grid.tokens_per_core,
      kInvSqrtHeadSize);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  run.rotate_plan = tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs.get(), h16.get(),
                                     rot_tables.get(), query_rot.get(), batch, kNumHeads, kHeadSize, aiv_num);

  turboquant_mm_fused_decode_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, decode_grid.block_dim, query_rot.get(), key_cache.get(),
      value_cache.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(), decode_tables.get(),
      workspace.get(), out.get(), static_cast<uint32_t>(batch), static_cast<uint32_t>(kNumHeads),
      static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
      static_cast<uint32_t>(shape.blocks_per_seq), static_cast<uint32_t>(decode_grid.num_splits),
      decode_grid.heads_per_task, decode_grid.tasks_per_block, decode_grid.reduce_tasks_per_block,
      static_cast<uint32_t>(tqh::kFusedContextLimit), kAttentionScale, kInvSqrtHeadSize);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  run.output = tqh::UnrotateHeads(HalfToFloat(out.ToHost<Half>()), kHeadSize);
  run.scale_plane = scale_plane.ToHost<float>();
  return run;
}

Shape MakeShape(int64_t batch, int64_t context_len, DeterministicRandom* rng) {
  Shape shape;
  shape.batch = batch;
  shape.context_len = context_len;
  shape.blocks_per_seq = BlocksPerSeq(context_len);
  shape.num_blocks = NumBlocks(batch, context_len);

  const size_t kv_elems = static_cast<size_t>(batch * context_len * kNumKvHeads * kHeadSize);
  shape.key = rng->NormalHalfExact(kv_elems, 0.0f, 1.0f);
  shape.value = rng->NormalHalfExact(kv_elems, 0.0f, 1.0f);
  shape.query = rng->NormalHalfExact(static_cast<size_t>(batch * kNumHeads * kHeadSize), 0.0f, 1.0f);

  const std::vector<int32_t> permutation = rng->Permutation(static_cast<int32_t>(shape.num_blocks));
  shape.block_table.resize(static_cast<size_t>(batch * shape.blocks_per_seq));
  shape.context_lens.assign(static_cast<size_t>(batch), static_cast<int32_t>(context_len));
  shape.slots.resize(static_cast<size_t>(batch * context_len));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t i = 0; i < shape.blocks_per_seq; ++i) {
      shape.block_table[static_cast<size_t>(b * shape.blocks_per_seq + i)] =
          permutation[static_cast<size_t>(b * shape.blocks_per_seq + i)];
    }
    for (int64_t i = 0; i < context_len; ++i) {
      const int32_t block = shape.block_table[static_cast<size_t>(b * shape.blocks_per_seq + i / kBlockSize)];
      shape.slots[static_cast<size_t>(b * context_len + i)] =
          block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
    }
  }
  return shape;
}

bool ModeSelected(const char* name) {
  const char* raw = std::getenv("ASCEND_TQ_SIM_MODES");
  if (raw == nullptr || *raw == '\0') {
    return true;
  }
  const std::string wanted(raw);
  const std::string needle(name);
  size_t at = wanted.find(needle);
  while (at != std::string::npos) {
    const bool left = at == 0 || wanted[at - 1] == ',';
    const size_t after = at + needle.size();
    const bool right = after == wanted.size() || wanted[after] == ',';
    if (left && right) {
      return true;
    }
    at = wanted.find(needle, at + 1);
  }
  return false;
}

TEST(TurboQuantMultiMode, DispatchAndFidelityAcrossShapes) {
  REQUIRE_ASCEND_950PR();

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  struct ModeCase {
    tqm::TurboQuantMode mode;
    const char* name;
    bool assert_fidelity;
    bool provisional;
  };
  const ModeCase cases[] = {
      {tqm::TurboQuantMode::KV4_FP8, "kv4fp8", true, false},
      {tqm::TurboQuantMode::KV5_FP8, "kv5fp8", true, true},
      {tqm::TurboQuantMode::KV3_FP4, "kv3fp4", false, true},
  };

  std::printf("[ multimode ] tier=%s head_size=%lld heads=%lld kv_heads=%lld block=%lld\n",
              VLLM_ASCEND_TEST_TIER, static_cast<long long>(kHeadSize), static_cast<long long>(kNumHeads),
              static_cast<long long>(kNumKvHeads), static_cast<long long>(kBlockSize));
  std::printf("[ multimode ] sweep: batch x context = %zu x %zu = %zu shape(s)\n", BatchSizes().size(),
              ContextLens().size(), BatchSizes().size() * ContextLens().size());

  for (const int64_t batch : BatchSizes()) {
    for (const int64_t context_len : ContextLens()) {
      DeterministicRandom rng(0x5A17u);
      const Shape shape = MakeShape(batch, context_len, &rng);
      const std::vector<float> reference = HostAttention(batch, context_len, shape.query, shape.key, shape.value);

      std::printf("\n[ multimode ] shape: B=%lld S=%lld pool=%lld blocks/seq=%lld\n",
                  static_cast<long long>(batch), static_cast<long long>(context_len),
                  static_cast<long long>(shape.num_blocks), static_cast<long long>(shape.blocks_per_seq));

      for (const ModeCase& mode_case : cases) {
        if (!ModeSelected(mode_case.name)) {
          std::printf("[ multimode ] %s: not selected by ASCEND_TQ_SIM_MODES; skipped\n", mode_case.name);
          continue;
        }
        if (mode_case.provisional && !CubeWipOptedIn()) {
          std::printf("[ multimode ] %s: still provisional, not re-measured through the corrected staging; "
                      "set VLLM_ASCEND_TQ_CUBE_WIP=1 to run it\n",
                      mode_case.name);
          continue;
        }
        const tqm::TurboQuantModeConfig cfg = tqm::TurboQuantModeConfigOf(mode_case.mode);
        std::printf("[ multimode ] %s: %d bits, %d levels, %lld packed bytes/vector, %s, %s\n", mode_case.name,
                    cfg.bits, cfg.levels, static_cast<long long>(cfg.PackedBytes(kHeadSize)),
                    cfg.operand == tqm::TurboQuantOperand::kFp4E2m1 ? "Cube mad_mx (fp4 e2m1)"
                                                                    : "Cube mad (fp8 e4m3fn)",
                    cfg.is_affine ? "affine INT4" : "Lloyd-Max index");
        ASSERT_TRUE(cfg.IsBurstAligned(kHeadSize))
            << mode_case.name << " slot is not a whole 32-byte burst; every DMA on this path assumes it is";
        ASSERT_EQ(kBlockSize % 64, 0) << "block_size must be a multiple of kCubeTileRows (64): CopyInTile issues "
                                         "one unconditional 64-row DataCopy per tile";

        const ModeRun run = RunMode(mode_case.mode, shape, stream);

        std::printf("[ multimode ] %s: dispatched, splits=%lld\n", mode_case.name,
                    static_cast<long long>(run.num_splits));

        size_t finite = 0;
        double abs_sum = 0.0;
        for (const float v : run.output) {
          if (std::isfinite(v)) {
            ++finite;
          }
          abs_sum += std::fabs(static_cast<double>(v));
        }
        EXPECT_EQ(finite, run.output.size()) << mode_case.name << " produced non-finite output";
        EXPECT_GT(abs_sum, 0.0) << mode_case.name << " produced an identically zero output";

        double scale_sum = 0.0;
        for (const float v : run.scale_plane) {
          scale_sum += std::fabs(static_cast<double>(v));
        }
        EXPECT_GT(scale_sum, 0.0) << mode_case.name << " left the scale plane empty";

        const double cos = Cosine(reference, run.output);
        double worst_cos = cos;
        int64_t worst_seq = 0;
        if (batch > 1) {
          const size_t stride = static_cast<size_t>(kNumHeads * kHeadSize);
          for (int64_t b = 0; b < batch; ++b) {
            const size_t base = static_cast<size_t>(b) * stride;
            const std::vector<float> ref_seq(reference.begin() + static_cast<std::ptrdiff_t>(base),
                                             reference.begin() + static_cast<std::ptrdiff_t>(base + stride));
            const std::vector<float> got_seq(run.output.begin() + static_cast<std::ptrdiff_t>(base),
                                             run.output.begin() + static_cast<std::ptrdiff_t>(base + stride));
            const double seq_cos = Cosine(ref_seq, got_seq);
            if (seq_cos < worst_cos) {
              worst_cos = seq_cos;
              worst_seq = b;
            }
          }
          std::printf("[ multimode ] %s: cos vs fp32 host reference = %.6f (batch), worst sequence %lld = %.6f\n",
                      mode_case.name, cos, static_cast<long long>(worst_seq), worst_cos);
        } else {
          std::printf("[ multimode ] %s: cos vs fp32 host reference = %.6f\n", mode_case.name, cos);
        }

        if (mode_case.assert_fidelity) {
          EXPECT_GT(worst_cos, kSmokeCos)
              << mode_case.name << " decode does not track the host reference at B=" << batch << " S=" << context_len
              << "; this bound is a structural check, not the cos > 0.995 gate -- see the file header";
        }
      }
    }
  }
}

}
}
}
