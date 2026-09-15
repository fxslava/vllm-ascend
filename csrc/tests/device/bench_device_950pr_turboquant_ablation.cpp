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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turboquant_launch.hpp"
#include "turboquant_mirrored_cache.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "turboquant_950pr_ablation (kv4fp8 Cube decode split, cut stage by stage)";

namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;

constexpr tqm::TurboQuantMode kMode = tqm::TurboQuantMode::KV4_FP8;

constexpr int64_t kNumHeads = s950::kNumHeads;
constexpr int64_t kNumKvHeads = s950::kNumKvHeads;
constexpr int64_t kBlockSize = s950::kBlockSize;
constexpr int64_t kQueryTokens = 1;

const int64_t kDefaultHeadSizes[] = {256, 512};
const int64_t kDefaultContextLens[] = {64, 512, 1024, 2048};

constexpr int64_t kMinHeadSize = 64;
constexpr int64_t kMaxHeadSize = 512;
constexpr int64_t kAdapterMaxHeadSize = 256;
constexpr int64_t kContextMultiple = 8;

constexpr double kFidelityCos = 0.90;

constexpr int64_t kDefaultSyncTimeoutMs = 30000;
constexpr int32_t kWaitForever = -1;
constexpr int kHangExitCode = 3;

constexpr int64_t kFloatBytes = 4;
constexpr double kPercent = 100.0;

constexpr uint32_t kBypassSeed = 0xB1A5u;
constexpr int64_t kMinPoolBlocks = 4;
constexpr int64_t kPoolFactor = 4;

uint32_t U32(int64_t value) { return static_cast<uint32_t>(value); }

enum class BypassMode {
  kOff,
  kWithLadder,
  kOnly,
};

BypassMode BypassUnpackMode() {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_BYPASS_UNPACK");
  if (raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") == 0) {
    return BypassMode::kOff;
  }
  if (std::strcmp(raw, "1") == 0) {
    return BypassMode::kWithLadder;
  }
  if (std::strcmp(raw, "only") == 0) {
    return BypassMode::kOnly;
  }
  std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_BYPASS_UNPACK='%s' is not 0, 1 or only; "
              "the bypass pair is off\n",
              raw);
  return BypassMode::kOff;
}

const char* BypassModeLabel(BypassMode mode) {
  switch (mode) {
    case BypassMode::kWithLadder:
      return "on, after the stage ladder";
    case BypassMode::kOnly:
      return "only, the stage ladder is not run";
    case BypassMode::kOff:
      break;
  }
  return "off (--bypass-unpack to add it)";
}

bool ParseDecimalList(const char* raw, int64_t min_value, std::vector<int64_t>* values) {
  constexpr size_t kMaxDigits = 9;
  std::vector<int64_t> parsed;
  std::istringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty() || token.size() > kMaxDigits || token.find_first_not_of("0123456789") != std::string::npos) {
      return false;
    }
    const int64_t value = std::strtoll(token.c_str(), nullptr, 10);
    if (value < min_value) {
      return false;
    }
    parsed.push_back(value);
  }
  if (parsed.empty()) {
    return false;
  }
  *values = parsed;
  return true;
}

bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

std::vector<int64_t> HeadSizes() {
  const std::vector<int64_t> defaults(std::begin(kDefaultHeadSizes), std::end(kDefaultHeadSizes));
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_DIMS");
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  bool valid = ParseDecimalList(raw, 1, &parsed);
  for (const int64_t head_size : parsed) {
    valid = valid && IsPowerOfTwo(head_size) && head_size >= kMinHeadSize && head_size <= kMaxHeadSize;
  }
  if (!valid) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_DIMS='%s' is not a list of powers of two in [%lld, %lld]; "
                "using the default sweep\n",
                raw, static_cast<long long>(kMinHeadSize), static_cast<long long>(kMaxHeadSize));
    return defaults;
  }
  return parsed;
}

std::vector<int64_t> ContextLens() {
  const std::vector<int64_t> defaults(std::begin(kDefaultContextLens), std::end(kDefaultContextLens));
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_CONTEXTS");
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  bool valid = ParseDecimalList(raw, 1, &parsed);
  for (const int64_t context_len : parsed) {
    valid = valid && context_len % kContextMultiple == 0;
  }
  if (!valid) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_CONTEXTS='%s' is not a list of positive multiples of %lld; "
                "using the default sweep\n",
                raw, static_cast<long long>(kContextMultiple));
    return defaults;
  }
  return parsed;
}

tqm::DecodeAblationStage StageAt(int32_t index) { return static_cast<tqm::DecodeAblationStage>(index); }

std::vector<tqm::DecodeAblationStage> Stages() {
  std::vector<tqm::DecodeAblationStage> stages;
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_STAGES");
  if (raw != nullptr && *raw != '\0') {
    std::vector<int64_t> parsed;
    bool valid = ParseDecimalList(raw, 0, &parsed);
    for (const int64_t index : parsed) {
      valid = valid && tqm::DecodeAblationStageIsValid(static_cast<int32_t>(index));
    }
    if (valid) {
      for (const int64_t index : parsed) {
        stages.push_back(StageAt(static_cast<int32_t>(index)));
      }
      return stages;
    }
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_STAGES='%s' is not a list of stages 0..%d; running all\n",
                raw, tqm::kDecodeAblationStageCount - 1);
  }
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    stages.push_back(StageAt(index));
  }
  return stages;
}

int32_t SyncTimeoutMs() {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS");
  int64_t timeout_ms = kDefaultSyncTimeoutMs;
  if (raw != nullptr && *raw != '\0') {
    std::vector<int64_t> parsed;
    if (ParseDecimalList(raw, 0, &parsed) && parsed.size() == 1) {
      timeout_ms = parsed[0];
    } else {
      std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS='%s' is not one non-negative integer; "
                  "using %lld\n",
                  raw, static_cast<long long>(kDefaultSyncTimeoutMs));
    }
  }
  return timeout_ms == 0 ? kWaitForever : static_cast<int32_t>(timeout_ms);
}

std::string ShapeName(int64_t head_size, int64_t context_len) {
  std::ostringstream name;
  name << "d" << head_size << "_s" << context_len;
  return name.str();
}

std::string CaseName(const std::string& shape, tqm::DecodeAblationStage stage) {
  return shape + "_" + tqm::DecodeAblationStageName(stage);
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return 0.0;
  }
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

std::vector<float> HostAttention(int64_t head_size, int64_t context_len, double scale,
                                 const std::vector<float>& query, const std::vector<float>& key,
                                 const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(kNumHeads * head_size), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
  for (int64_t h = 0; h < kNumHeads; ++h) {
    const int64_t kv = h / heads_per_kv;
    double max_logit = -1e30;
    for (int64_t t = 0; t < context_len; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < head_size; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * head_size + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * kNumKvHeads + kv) * head_size + d)]);
      }
      logits[static_cast<size_t>(t)] = dot * scale;
      max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
    }
    double denom = 0.0;
    for (int64_t t = 0; t < context_len; ++t) {
      logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
      denom += logits[static_cast<size_t>(t)];
    }
    for (int64_t t = 0; t < context_len; ++t) {
      const double weight = logits[static_cast<size_t>(t)] / denom;
      for (int64_t d = 0; d < head_size; ++d) {
        out[static_cast<size_t>(h * head_size + d)] += static_cast<float>(
            weight * static_cast<double>(value[static_cast<size_t>((t * kNumKvHeads + kv) * head_size + d)]));
      }
    }
  }
  return out;
}

class AblationScenario {
 public:
  AblationScenario(int64_t head_size, int64_t context_len, int64_t aiv_num)
      : head_size_(head_size),
        context_len_(context_len),
        attention_scale_(static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_size)))) {
    blocks_per_seq_ = (context_len + kBlockSize - 1) / kBlockSize;
    num_blocks_ = std::max<int64_t>(4, blocks_per_seq_ * 4);

    DeterministicRandom rng(0x7451u);
    const size_t kv_elems = static_cast<size_t>(context_len * kNumKvHeads * head_size);
    key_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    value_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    query_host_ = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), 0.0f, 1.0f);

    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks_));
    const std::vector<int32_t> block_table(permutation.begin(),
                                           permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq_));
    std::vector<int32_t> slots(static_cast<size_t>(context_len));
    for (int64_t i = 0; i < context_len; ++i) {
      slots[static_cast<size_t>(i)] = block_table[static_cast<size_t>(i / kBlockSize)] *
                                          static_cast<int32_t>(kBlockSize) +
                                      static_cast<int32_t>(i % kBlockSize);
    }

    key_ = DeviceBuffer::FromHost(FloatToHalf(key_host_), kBenchmarkAlignBytes);
    value_ = DeviceBuffer::FromHost(FloatToHalf(value_host_), kBenchmarkAlignBytes);
    query_ = DeviceBuffer::FromHost(FloatToHalf(query_host_), kBenchmarkAlignBytes);
    slots_ = DeviceBuffer::FromHost(slots, kBenchmarkAlignBytes);
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(kQueryTokens), static_cast<int32_t>(context_len)),
        kBenchmarkAlignBytes);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(head_size), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size),
                                            kBenchmarkAlignBytes);
    aiv_num_ = aiv_num;

    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(head_size, 1), kBenchmarkAlignBytes);
    write_tables_ = DeviceBuffer::FromHost(tqh::ModeTables(kMode, head_size, 1, 0), kBenchmarkAlignBytes);
    decode_tables_ = DeviceBuffer::FromHost(
        tqh::ModeTables(kMode, head_size, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);

    key_cache_ = DeviceBuffer::Empty<int8_t>(
        tqh::ModePackedCacheBytes(kMode, num_blocks_, kBlockSize, kNumKvHeads, head_size), kBenchmarkAlignBytes);
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(num_blocks_, kBlockSize, kNumKvHeads),
                                              kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), kBenchmarkAlignBytes);

    write_grid_ = tqh::PlanReshapeAndCache(context_len, aiv_num);
    decode_grid_ = tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, head_size, blocks_per_seq_, aiv_num);
    ResetWorkspace();
  }

  void FillCache(aclrtStream stream) const {
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, write_grid_.block_dim, key_.get(), value_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), slots_.get(), pi_signs_.get(), rot_tables_.get(),
        write_tables_.get(), U32(context_len_), U32(kNumKvHeads), U32(head_size_), U32(kBlockSize),
        write_grid_.tokens_per_core, attention_scale_);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  void ResetWorkspace() {
    workspace_ = DeviceBuffer::FromHost(std::vector<float>(decode_grid_.workspace_floats, 0.0f), kBenchmarkAlignBytes);
  }

  void RotateQueryOnce(aclrtStream stream) {
    rotate_plan_ = tqh::RotateQuery(stream, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(),
                                    rot_tables_.get(), query_rot_.get(), kQueryTokens, kNumHeads, head_size_,
                                    aiv_num_, true);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  const vllm_ascend::turboquant::RotateQPlan& rotate_plan() const { return rotate_plan_; }

  void EnqueueSplit(tqm::DecodeAblationStage stage, aclrtStream stream) const {
    turboquant_mm_decode_ablation_impl(
        static_cast<int32_t>(stage), AscendType::FP16, stream, decode_grid_.split_block_dim, query_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), U32(kQueryTokens),
        U32(kNumHeads), U32(kNumKvHeads), U32(head_size_), U32(kBlockSize), U32(blocks_per_seq_),
        U32(decode_grid_.num_splits), decode_grid_.split_tasks_per_core, attention_scale_, attention_scale_);
  }

  std::vector<float> CombineAndReadBack(aclrtStream stream) const {
    turboquant_paged_attention_combine_impl(AscendType::FP16, stream, decode_grid_.combine_block_dim,
                                            workspace_.get(), out_.get(), U32(kQueryTokens), U32(kNumHeads),
                                            U32(head_size_), U32(decode_grid_.num_splits),
                                            decode_grid_.combine_tasks_per_core);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    return tqh::UnrotateHeads(HalfToFloat(out_.ToHost<Half>()), head_size_);
  }

  std::vector<float> Reference() const {
    return HostAttention(head_size_, context_len_, static_cast<double>(attention_scale_), query_host_, key_host_,
                         value_host_);
  }

  std::vector<float> Workspace() const { return workspace_.ToHost<float>(); }

  double TileReadBytes() const {
    int64_t tiles = 0;
    for (int64_t block = 0; block < blocks_per_seq_; ++block) {
      const int64_t rows = std::min<int64_t>(kBlockSize, context_len_ - block * kBlockSize);
      tiles += (rows + tqh::kCubeTileRows - 1) / tqh::kCubeTileRows;
    }
    const int64_t row_bytes =
        2 * tqh::ModePackedBytes(kMode, head_size_) + tqh::ScaleSlotFloats(kNumKvHeads) * kFloatBytes;
    return static_cast<double>(kNumKvHeads * tiles * tqh::kCubeTileRows * row_bytes);
  }

  const tqh::CubeDecodeGrid& decode_grid() const { return decode_grid_; }
  int64_t blocks_per_seq() const { return blocks_per_seq_; }

 private:
  int64_t head_size_ = 0;
  int64_t context_len_ = 0;
  float attention_scale_ = 1.0f;
  int64_t blocks_per_seq_ = 0;
  int64_t num_blocks_ = 0;

  std::vector<float> key_host_, value_host_, query_host_;

  DeviceBuffer key_, value_, query_, slots_, block_tables_, context_lens_, pi_signs_;
  DeviceBuffer h16_, query_rot_;
  DeviceBuffer rot_tables_, write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_, out_;
  tqh::ReshapeAndCacheGrid write_grid_;
  tqh::CubeDecodeGrid decode_grid_;
  vllm_ascend::turboquant::RotateQPlan rotate_plan_;
  int64_t aiv_num_ = 1;
};

struct ShapeVerdict {
  int64_t head_size = 0;
  int64_t context_len = 0;
  bool fidelity_checked = false;
  double cos = 0.0;
  std::string distrust;
};

enum class ShapeOutcome {
  kCompleted,
  kStreamWedged,
};

void CheckFidelity(BenchmarkRunner& runner, const AblationScenario& ready, const std::string& shape,
                   ShapeVerdict* verdict) {
  const std::string fidelity_name = shape + "_fidelity";
  try {
    const std::vector<float> output = ready.CombineAndReadBack(runner.stream());
    verdict->cos = Cosine(ready.Reference(), output);
    verdict->fidelity_checked = true;
    if (!(verdict->cos >= kFidelityCos)) {
      std::ostringstream reason;
      reason << "stage 5 plus combine gives cos " << verdict->cos << " against the fp32 host attention, below "
             << kFidelityCos;
      runner.RecordFailure(fidelity_name, reason.str());
      verdict->distrust = reason.str();
    }
  } catch (const std::exception& error) {
    runner.RecordFailure(fidelity_name, error.what());
    verdict->distrust = std::string("fidelity check failed: ") + error.what();
  }
}

ShapeOutcome RunShape(BenchmarkRunner& runner, int64_t aiv_num, const std::vector<tqm::DecodeAblationStage>& stages,
                      int32_t sync_timeout_ms, ShapeVerdict* verdict) {
  const std::string shape = ShapeName(verdict->head_size, verdict->context_len);

  std::unique_ptr<AblationScenario> scenario;
  try {
    scenario.reset(new AblationScenario(verdict->head_size, verdict->context_len, aiv_num));
    scenario->FillCache(runner.stream());
    scenario->RotateQueryOnce(runner.stream());
  } catch (const std::exception& error) {
    for (const tqm::DecodeAblationStage stage : stages) {
      runner.RecordFailure(CaseName(shape, stage), std::string("setup failed: ") + error.what());
    }
    verdict->distrust = "setup failed";
    return ShapeOutcome::kCompleted;
  }

  const tqh::CubeDecodeGrid& grid = scenario->decode_grid();
  std::printf("\n[ascend-bench] %s: blocks_per_seq=%lld splits=%lld grid(split)=%u tasks/core=%u "
              "tile read=%.0f B/launch\n",
              shape.c_str(), static_cast<long long>(scenario->blocks_per_seq()),
              static_cast<long long>(grid.num_splits), grid.split_block_dim, grid.split_tasks_per_core,
              scenario->TileReadBytes());
  std::fflush(stdout);

  AblationScenario& mutable_scenario = *scenario;
  const AblationScenario& ready = mutable_scenario;
  for (const tqm::DecodeAblationStage stage : stages) {
    const bool writes_workspace = stage == tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE;
    const std::string name = CaseName(shape, stage);
    mutable_scenario.ResetWorkspace();

    std::printf("[ascend-bench] %s: first launch, deadline %d ms\n", name.c_str(), sync_timeout_ms);
    std::fflush(stdout);
    ready.EnqueueSplit(stage, runner.stream());
    const aclError synced = aclrtSynchronizeStreamWithTimeout(runner.stream(), sync_timeout_ms);
    if (synced != ACL_SUCCESS) {
      std::ostringstream reason;
      reason << "the first launch did not synchronise within " << sync_timeout_ms << " ms (aclError " << synced
             << "); treating the stage as deadlocked";
      std::printf("\n[ascend-bench] HANG: %s: %s\n", name.c_str(), reason.str().c_str());
      runner.RecordFailure(name, reason.str());
      verdict->distrust = name + " deadlocked";
      return ShapeOutcome::kStreamWedged;
    }

    try {
      BenchmarkCase ablation_case;
      ablation_case.name = name;
      ablation_case.bytes_per_iteration = ready.TileReadBytes();
      ablation_case.tasks_per_launch = 1;
      ablation_case.launch = [&ready, stage](aclrtStream stream) { ready.EnqueueSplit(stage, stream); };
      if (writes_workspace) {
        ablation_case.checksum = [&ready]() { return ChecksumSum(ready.Workspace()); };
      } else {
        ablation_case.checksum = [&ready]() { return ChecksumSumOfSquares(ready.Workspace()); };
      }
      runner.Run(ablation_case);

      if (writes_workspace) {
        CheckFidelity(runner, ready, shape, verdict);
      } else if (ChecksumSumOfSquares(ready.Workspace()) != 0.0) {
        runner.RecordFailure(name, "wrote the workspace: a cut below stage 5 must not reach the partial writeback");
        verdict->distrust = "an ablation gate leaked the writeback";
      }
    } catch (const std::exception& error) {
      runner.RecordFailure(name, error.what());
    }
  }
  return ShapeOutcome::kCompleted;
}

enum class UnpackPath {
  kStandard,
  kBypass,
};

const char* UnpackPathSuffix(UnpackPath path) { return path == UnpackPath::kBypass ? "bypass" : "standard"; }

std::string BypassCaseName(const std::string& shape, UnpackPath path) {
  return shape + "_split_" + UnpackPathSuffix(path);
}

class BypassScenario {
 public:
  BypassScenario(int64_t head_size, int64_t context_len, int64_t aiv_num)
      : head_size_(head_size),
        context_len_(context_len),
        attention_scale_(static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_size)))),
        aiv_num_(aiv_num) {
    blocks_per_seq_ = (context_len + kBlockSize - 1) / kBlockSize;
    const int64_t num_blocks = std::max<int64_t>(kMinPoolBlocks, blocks_per_seq_ * kPoolFactor);

    DeterministicRandom rng(kBypassSeed);
    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks));
    const std::vector<int32_t> block_table(permutation.begin(),
                                           permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq_));
    cache_ = tqh::BuildMirroredKvCache(rng, context_len, num_blocks, kBlockSize, kNumKvHeads, head_size, block_table);
    const std::vector<float> query =
        rng.NormalHalfExact(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), 0.0f, 1.0f);

    query_ = DeviceBuffer::FromHost(FloatToHalf(query), kBenchmarkAlignBytes);
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(kQueryTokens), static_cast<int32_t>(context_len)),
        kBenchmarkAlignBytes);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(head_size), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(head_size, 1), kBenchmarkAlignBytes);
    decode_tables_ = DeviceBuffer::FromHost(
        tqh::ModeTables(kMode, head_size, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size),
                                            kBenchmarkAlignBytes);
    key_packed_ = DeviceBuffer::FromHost(cache_.key_packed, kBenchmarkAlignBytes);
    value_packed_ = DeviceBuffer::FromHost(cache_.value_packed, kBenchmarkAlignBytes);
    key_operands_ = DeviceBuffer::FromHost(cache_.key_operands, kBenchmarkAlignBytes);
    value_operands_ = DeviceBuffer::FromHost(cache_.value_operands, kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::FromHost(cache_.scales, kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), kBenchmarkAlignBytes);

    decode_grid_ = tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, head_size, blocks_per_seq_, aiv_num);
    const std::vector<float> zeros(decode_grid_.workspace_floats, 0.0f);
    standard_workspace_ = DeviceBuffer::FromHost(zeros, kBenchmarkAlignBytes);
    bypass_workspace_ = DeviceBuffer::FromHost(zeros, kBenchmarkAlignBytes);
  }

  void RotateQueryOnce(aclrtStream stream) {
    tqh::RotateQuery(stream, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_rot_.get(), kQueryTokens, kNumHeads, head_size_, aiv_num_, true);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    reference_ = HostAttention(head_size_, context_len_, static_cast<double>(attention_scale_),
                               query_rot_.ToHost<float>(), cache_.key, cache_.value);
  }

  void EnqueueSplit(UnpackPath path, aclrtStream stream) const {
    if (path == UnpackPath::kBypass) {
      turboquant_mm_decode_bypass_unpack_impl(
          AscendType::FP16, stream, decode_grid_.split_block_dim, query_rot_.get(), key_operands_.get(),
          value_operands_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(),
          bypass_workspace_.get(), U32(kQueryTokens), U32(kNumHeads), U32(kNumKvHeads), U32(head_size_),
          U32(kBlockSize), U32(blocks_per_seq_), U32(decode_grid_.num_splits), decode_grid_.split_tasks_per_core,
          attention_scale_, attention_scale_);
      return;
    }
    turboquant_mm_decode_split_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, decode_grid_.split_block_dim, query_rot_.get(),
        key_packed_.get(), value_packed_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), standard_workspace_.get(), U32(kQueryTokens), U32(kNumHeads), U32(kNumKvHeads),
        U32(head_size_), U32(kBlockSize), U32(blocks_per_seq_), U32(decode_grid_.num_splits),
        decode_grid_.split_tasks_per_core, attention_scale_, attention_scale_);
  }

  std::vector<float> Workspace(UnpackPath path) const {
    return (path == UnpackPath::kBypass ? bypass_workspace_ : standard_workspace_).ToHost<float>();
  }

  std::vector<float> CombineAndReadBack(UnpackPath path, aclrtStream stream) const {
    const DeviceBuffer& workspace = path == UnpackPath::kBypass ? bypass_workspace_ : standard_workspace_;
    turboquant_paged_attention_combine_impl(AscendType::FP16, stream, decode_grid_.combine_block_dim,
                                            workspace.get(), out_.get(), U32(kQueryTokens), U32(kNumHeads),
                                            U32(head_size_), U32(decode_grid_.num_splits),
                                            decode_grid_.combine_tasks_per_core);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    return HalfToFloat(out_.ToHost<Half>());
  }

  const std::vector<float>& reference() const { return reference_; }

  double TileReadBytes(UnpackPath path) const {
    int64_t tiles = 0;
    for (int64_t block = 0; block < blocks_per_seq_; ++block) {
      const int64_t rows = std::min<int64_t>(kBlockSize, context_len_ - block * kBlockSize);
      tiles += (rows + tqh::kCubeTileRows - 1) / tqh::kCubeTileRows;
    }
    const int64_t plane_bytes = path == UnpackPath::kBypass ? head_size_ : tqh::ModePackedBytes(kMode, head_size_);
    const int64_t row_bytes = 2 * plane_bytes + tqh::ScaleSlotFloats(kNumKvHeads) * kFloatBytes;
    return static_cast<double>(kNumKvHeads * tiles * tqh::kCubeTileRows * row_bytes);
  }

  const tqh::CubeDecodeGrid& decode_grid() const { return decode_grid_; }

 private:
  int64_t head_size_ = 0;
  int64_t context_len_ = 0;
  float attention_scale_ = 1.0f;
  int64_t aiv_num_ = 1;
  int64_t blocks_per_seq_ = 0;
  tqh::MirroredKvCache cache_;
  std::vector<float> reference_;
  DeviceBuffer query_, block_tables_, context_lens_, pi_signs_, h16_, rot_tables_, decode_tables_, query_rot_;
  DeviceBuffer key_packed_, value_packed_, key_operands_, value_operands_, scale_plane_, out_;
  DeviceBuffer standard_workspace_, bypass_workspace_;
  tqh::CubeDecodeGrid decode_grid_;
};

struct BypassVerdict {
  int64_t head_size = 0;
  int64_t context_len = 0;
  std::string skipped;
  bool compared = false;
  size_t differing_words = 0;
  size_t partial_words = 0;
  double cos_standard = 0.0;
  double cos_bypass = 0.0;
  double read_standard = 0.0;
  double read_bypass = 0.0;
  std::string distrust;
};

ShapeOutcome RunBypassShape(BenchmarkRunner& runner, int64_t aiv_num, int32_t sync_timeout_ms,
                            BypassVerdict* verdict) {
  constexpr UnpackPath kPaths[] = {UnpackPath::kStandard, UnpackPath::kBypass};
  const std::string shape = ShapeName(verdict->head_size, verdict->context_len);

  if (verdict->head_size > kAdapterMaxHeadSize) {
    std::ostringstream reason;
    reason << "head_size " << verdict->head_size << " > " << kAdapterMaxHeadSize
           << ": the bypass tile buffer would push the split past the part's UB";
    verdict->skipped = reason.str();
    for (const UnpackPath path : kPaths) {
      runner.Skip(BypassCaseName(shape, path), verdict->skipped);
    }
    return ShapeOutcome::kCompleted;
  }

  std::unique_ptr<BypassScenario> scenario;
  try {
    scenario.reset(new BypassScenario(verdict->head_size, verdict->context_len, aiv_num));
    scenario->RotateQueryOnce(runner.stream());
  } catch (const std::exception& error) {
    for (const UnpackPath path : kPaths) {
      runner.RecordFailure(BypassCaseName(shape, path), std::string("setup failed: ") + error.what());
    }
    verdict->distrust = "setup failed";
    return ShapeOutcome::kCompleted;
  }
  const BypassScenario& ready = *scenario;
  verdict->read_standard = ready.TileReadBytes(UnpackPath::kStandard);
  verdict->read_bypass = ready.TileReadBytes(UnpackPath::kBypass);

  const tqh::CubeDecodeGrid& grid = ready.decode_grid();
  std::printf("\n[ascend-bench] %s bypass pair: mirrored host cache, splits=%lld grid(split)=%u tasks/core=%u "
              "tile read standard=%.0f B bypass=%.0f B per launch\n",
              shape.c_str(), static_cast<long long>(grid.num_splits), grid.split_block_dim,
              grid.split_tasks_per_core, verdict->read_standard, verdict->read_bypass);
  std::fflush(stdout);

  for (const UnpackPath path : kPaths) {
    const std::string name = BypassCaseName(shape, path);
    std::printf("[ascend-bench] %s: first launch, deadline %d ms\n", name.c_str(), sync_timeout_ms);
    std::fflush(stdout);
    ready.EnqueueSplit(path, runner.stream());
    const aclError synced = aclrtSynchronizeStreamWithTimeout(runner.stream(), sync_timeout_ms);
    if (synced != ACL_SUCCESS) {
      std::ostringstream reason;
      reason << "the first launch did not synchronise within " << sync_timeout_ms << " ms (aclError " << synced
             << "); treating the split as deadlocked";
      std::printf("\n[ascend-bench] HANG: %s: %s\n", name.c_str(), reason.str().c_str());
      runner.RecordFailure(name, reason.str());
      verdict->distrust = name + " deadlocked";
      return ShapeOutcome::kStreamWedged;
    }
    try {
      BenchmarkCase split_case;
      split_case.name = name;
      split_case.bytes_per_iteration = ready.TileReadBytes(path);
      split_case.tasks_per_launch = 1;
      split_case.launch = [&ready, path](aclrtStream stream) { ready.EnqueueSplit(path, stream); };
      split_case.checksum = [&ready, path]() { return ChecksumSum(ready.Workspace(path)); };
      runner.Run(split_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(name, error.what());
      verdict->distrust = name + " failed";
    }
  }

  try {
    const std::vector<float> standard = ready.Workspace(UnpackPath::kStandard);
    const std::vector<float> bypass = ready.Workspace(UnpackPath::kBypass);
    verdict->partial_words = standard.size();
    verdict->differing_words = standard.size() == bypass.size() ? 0u : standard.size();
    for (size_t i = 0; i < standard.size() && i < bypass.size(); ++i) {
      verdict->differing_words += std::memcmp(&standard[i], &bypass[i], sizeof(float)) != 0 ? 1u : 0u;
    }
    verdict->cos_standard = Cosine(ready.reference(), ready.CombineAndReadBack(UnpackPath::kStandard, runner.stream()));
    verdict->cos_bypass = Cosine(ready.reference(), ready.CombineAndReadBack(UnpackPath::kBypass, runner.stream()));
    verdict->compared = true;
    const std::string check_name = shape + "_bypass_check";
    if (verdict->differing_words != 0) {
      std::ostringstream reason;
      reason << verdict->differing_words << " of " << verdict->partial_words
             << " partial words differ: the bypass does not stage the operands the codec unpacks";
      runner.RecordFailure(check_name, reason.str());
      verdict->distrust = reason.str();
    }
    if (!(verdict->cos_standard >= kFidelityCos) || !(verdict->cos_bypass >= kFidelityCos)) {
      std::ostringstream reason;
      reason << "cos standard " << verdict->cos_standard << ", bypass " << verdict->cos_bypass
             << " against the fp32 host attention, below " << kFidelityCos;
      runner.RecordFailure(check_name, reason.str());
      verdict->distrust = reason.str();
    }
  } catch (const std::exception& error) {
    runner.RecordFailure(shape + "_bypass_check", error.what());
    verdict->distrust = std::string("bypass check failed: ") + error.what();
  }
  return ShapeOutcome::kCompleted;
}

TimingMode WaterfallMode(const BenchmarkRunner& runner) {
  const std::vector<TimingMode>& modes = runner.options().modes;
  if (std::find(modes.begin(), modes.end(), TimingMode::kPipelined) != modes.end() || modes.empty()) {
    return TimingMode::kPipelined;
  }
  return modes.front();
}

const BenchmarkResult* FindResult(const BenchmarkRunner& runner, const std::string& case_name, TimingMode mode) {
  for (const BenchmarkResult& result : runner.results()) {
    if (result.case_name == case_name && result.mode == mode && result.latency.sample_count > 0) {
      return &result;
    }
  }
  return nullptr;
}

struct Rung {
  bool present = false;
  double median_us = 0.0;
  double p95_us = 0.0;
  bool has_delta = false;
  double delta_us = 0.0;
};

std::vector<Rung> Ladder(const BenchmarkRunner& runner, TimingMode mode, const std::string& shape) {
  std::vector<Rung> rungs(static_cast<size_t>(tqm::kDecodeAblationStageCount));
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    Rung& rung = rungs[static_cast<size_t>(index)];
    const BenchmarkResult* result = FindResult(runner, CaseName(shape, StageAt(index)), mode);
    if (result == nullptr) {
      continue;
    }
    rung.present = true;
    rung.median_us = result->latency.median_us;
    rung.p95_us = result->latency.p95_us;
    if (index == 0) {
      rung.has_delta = true;
      rung.delta_us = rung.median_us;
    } else if (rungs[static_cast<size_t>(index - 1)].present) {
      rung.has_delta = true;
      rung.delta_us = rung.median_us - rungs[static_cast<size_t>(index - 1)].median_us;
    }
  }
  return rungs;
}

void PrintShapeWaterfall(const BenchmarkRunner& runner, TimingMode mode, const ShapeVerdict& verdict) {
  const std::string shape = ShapeName(verdict.head_size, verdict.context_len);
  const std::vector<Rung> rungs = Ladder(runner, mode, shape);
  const Rung& full = rungs.back();

  std::printf("\n[ascend-bench] %s", shape.c_str());
  if (verdict.fidelity_checked) {
    std::printf("  cos %.6f vs fp32 host", verdict.cos);
  }
  if (verdict.head_size > kAdapterMaxHeadSize) {
    std::printf("  (head_size above the adapter's %lld)", static_cast<long long>(kAdapterMaxHeadSize));
  }
  std::printf("\n");
  if (!verdict.distrust.empty()) {
    std::printf("[ascend-bench]   UNTRUSTED: %s\n", verdict.distrust.c_str());
  }
  std::printf("  %-20s %12s %12s %8s %12s\n", "stage", "latency_us", "delta_us", "share", "p95_us");
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    const Rung& rung = rungs[static_cast<size_t>(index)];
    std::printf("  %-20s ", tqm::DecodeAblationStageName(StageAt(index)));
    if (!rung.present) {
      std::printf("%12s %12s %8s %12s\n", "-", "-", "-", "-");
      continue;
    }
    std::printf("%12.2f ", rung.median_us);
    if (rung.has_delta) {
      std::printf("%+12.2f ", rung.delta_us);
    } else {
      std::printf("%12s ", "-");
    }
    if (rung.has_delta && full.present && full.median_us > 0.0) {
      std::printf("%7.1f%% ", kPercent * rung.delta_us / full.median_us);
    } else {
      std::printf("%8s ", "-");
    }
    std::printf("%12.2f\n", rung.p95_us);
  }
}

void PrintDeltaByContext(const BenchmarkRunner& runner, TimingMode mode, const std::vector<ShapeVerdict>& verdicts,
                         int64_t head_size) {
  std::vector<const ShapeVerdict*> columns;
  for (const ShapeVerdict& verdict : verdicts) {
    if (verdict.head_size == head_size) {
      columns.push_back(&verdict);
    }
  }
  if (columns.empty()) {
    return;
  }
  std::vector<std::vector<Rung>> ladders;
  for (const ShapeVerdict* column : columns) {
    ladders.push_back(Ladder(runner, mode, ShapeName(column->head_size, column->context_len)));
  }

  std::printf("\n[ascend-bench] D=%lld: delta_us by context\n", static_cast<long long>(head_size));
  std::printf("  %-20s", "stage");
  for (const ShapeVerdict* column : columns) {
    const std::string label =
        "S=" + std::to_string(column->context_len) + (column->distrust.empty() ? "" : "!");
    std::printf(" %15s", label.c_str());
  }
  std::printf("\n");
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    std::printf("  %-20s", tqm::DecodeAblationStageName(StageAt(index)));
    for (const std::vector<Rung>& ladder : ladders) {
      const Rung& rung = ladder[static_cast<size_t>(index)];
      if (rung.has_delta) {
        std::printf(" %+15.2f", rung.delta_us);
      } else {
        std::printf(" %15s", "-");
      }
    }
    std::printf("\n");
  }
  std::printf("  %-20s", "total (stage5)");
  for (const std::vector<Rung>& ladder : ladders) {
    if (ladder.back().present) {
      std::printf(" %15.2f", ladder.back().median_us);
    } else {
      std::printf(" %15s", "-");
    }
  }
  std::printf("\n");
}

void PrintWaterfall(const BenchmarkRunner& runner, const std::vector<int64_t>& head_sizes,
                    const std::vector<ShapeVerdict>& verdicts) {
  const TimingMode mode = WaterfallMode(runner);
  std::printf("\n[ascend-bench] ==== decode ablation waterfall (%s median) ====\n", TimingModeLabel(mode));
  std::printf("[ascend-bench] delta = this stage's median minus the one below it; share = delta over stage 5.\n"
              "[ascend-bench] A negative delta is noise or a change in pipe overlap, not a free stage.\n");
  for (const ShapeVerdict& verdict : verdicts) {
    PrintShapeWaterfall(runner, mode, verdict);
  }
  for (const int64_t head_size : head_sizes) {
    PrintDeltaByContext(runner, mode, verdicts, head_size);
  }
  std::printf("  ('!' marks a context whose rows are UNTRUSTED; see its waterfall above.)\n");
  std::fflush(stdout);
}

void PrintBypassReport(const BenchmarkRunner& runner, const std::vector<BypassVerdict>& verdicts) {
  const TimingMode mode = WaterfallMode(runner);
  std::printf("\n[ascend-bench] ==== unpack bypass: DecodeSplit_standard vs DecodeSplit_bypass (%s median) ====\n",
              TimingModeLabel(mode));
  std::printf("[ascend-bench] both splits run on one mirrored host cache; the bypass reads fp8 operands instead of\n"
              "[ascend-bench] packed int4 and skips UnpackAffine. It moves twice the GM bytes, so the share is a\n"
              "[ascend-bench] lower bound on what the unpack costs in place.\n");
  std::printf("  %-14s %12s %12s %12s %8s %8s %12s %12s %10s %9s %9s\n", "shape", "standard_us", "bypass_us",
              "delta_us", "ratio", "share", "read_std_B", "read_byp_B", "partials", "cos_std", "cos_byp");
  for (const BypassVerdict& verdict : verdicts) {
    const std::string shape = ShapeName(verdict.head_size, verdict.context_len);
    std::printf("  %-14s ", shape.c_str());
    if (!verdict.skipped.empty()) {
      std::printf("skipped: %s\n", verdict.skipped.c_str());
      continue;
    }
    const BenchmarkResult* standard = FindResult(runner, BypassCaseName(shape, UnpackPath::kStandard), mode);
    const BenchmarkResult* bypass = FindResult(runner, BypassCaseName(shape, UnpackPath::kBypass), mode);
    if (standard == nullptr || bypass == nullptr || standard->latency.median_us <= 0.0) {
      std::printf("%12s %12s %12s %8s %8s ", "-", "-", "-", "-", "-");
    } else {
      const double ratio = bypass->latency.median_us / standard->latency.median_us;
      std::printf("%12.2f %12.2f %+12.2f %8.4f %7.1f%% ", standard->latency.median_us, bypass->latency.median_us,
                  bypass->latency.median_us - standard->latency.median_us, ratio, kPercent * (1.0 - ratio));
    }
    std::printf("%12.0f %12.0f ", verdict.read_standard, verdict.read_bypass);
    if (verdict.compared) {
      std::printf("%10s %9.6f %9.6f\n", verdict.differing_words == 0 ? "identical" : "DIFFER", verdict.cos_standard,
                  verdict.cos_bypass);
    } else {
      std::printf("%10s %9s %9s\n", "-", "-", "-");
    }
    if (!verdict.distrust.empty()) {
      std::printf("[ascend-bench]   UNTRUSTED: %s\n", verdict.distrust.c_str());
    }
  }
  std::fflush(stdout);
}

void PrintBanner(const std::vector<int64_t>& head_sizes, const std::vector<int64_t>& context_lens,
                 const std::vector<tqm::DecodeAblationStage>& stages, int32_t sync_timeout_ms, int64_t aiv_num,
                 bool aiv_queried, BypassMode bypass_mode) {
  std::printf("[ascend-bench] kv4fp8 decode ablation: vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  std::printf("[ascend-bench] stages, each adding one piece of the split kernel to the one before:\n");
  std::printf("[ascend-bench]   %-18s packed K/V and scale tiles, GM -> UB\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_0_MTE2_ONLY));
  std::printf("[ascend-bench]   %-18s + affine unpack onto the fp8 grid\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_1_UNPACK));
  std::printf("[ascend-bench]   %-18s + query read, Pi rotation, operand cast (per task, not per tile)\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_2_QUERY_PREP));
  std::printf("[ascend-bench]   %-18s + V -> MTE3 edges and UB -> L1 copies\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_3_L1_STAGING));
  std::printf("[ascend-bench]   %-18s + Load2D, Q.K^T Mmad, Fixpipe to UB, AIV/AIC handshake\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_4_SCORE_GEMM));
  std::printf("[ascend-bench]   %-18s + online softmax, P.V GEMM, accumulator, GM writeback (the shipping kernel)\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE));
  std::printf("[ascend-bench] unpack bypass pair (split_standard vs split_bypass, head_size <= %lld): %s\n",
              static_cast<long long>(kAdapterMaxHeadSize), BypassModeLabel(bypass_mode));
  std::printf("[ascend-bench] running:");
  if (bypass_mode == BypassMode::kOnly) {
    std::printf(" (no stages)");
  }
  for (const tqm::DecodeAblationStage stage : stages) {
    if (bypass_mode == BypassMode::kOnly) {
      break;
    }
    std::printf(" %s", tqm::DecodeAblationStageName(stage));
  }
  if (sync_timeout_ms == kWaitForever) {
    std::printf("   first-launch deadline: none\n");
  } else {
    std::printf("   first-launch deadline: %d ms\n", sync_timeout_ms);
  }
  std::printf("[ascend-bench] head sizes:");
  for (const int64_t head_size : head_sizes) {
    std::printf(" %lld%s", static_cast<long long>(head_size), head_size > kAdapterMaxHeadSize ? "*" : "");
  }
  std::printf("   contexts:");
  for (const int64_t context_len : context_lens) {
    std::printf(" %lld", static_cast<long long>(context_len));
  }
  std::printf("\n");
  for (const int64_t head_size : head_sizes) {
    if (head_size > kAdapterMaxHeadSize) {
      std::printf("[ascend-bench]   * above the production adapter's head_size ceiling of %lld and never run before;\n"
                  "[ascend-bench]     its rows count only where the fidelity gate (cos >= %.2f) passes.\n",
                  static_cast<long long>(kAdapterMaxHeadSize), kFidelityCos);
      break;
    }
  }
  std::fflush(stdout);
}

}

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> head_sizes = HeadSizes();
  const std::vector<int64_t> context_lens = ContextLens();
  const std::vector<tqm::DecodeAblationStage> stages = Stages();
  const int32_t sync_timeout_ms = SyncTimeoutMs();

  const BypassMode bypass_mode = BypassUnpackMode();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
  PrintBanner(head_sizes, context_lens, stages, sync_timeout_ms, aiv_num, aiv_queried, bypass_mode);

  if (bypass_mode != BypassMode::kOnly) {
    std::vector<ShapeVerdict> verdicts;
    for (const int64_t head_size : head_sizes) {
      for (const int64_t context_len : context_lens) {
        ShapeVerdict verdict;
        verdict.head_size = head_size;
        verdict.context_len = context_len;
        const ShapeOutcome outcome = RunShape(runner, aiv_num, stages, sync_timeout_ms, &verdict);
        verdicts.push_back(verdict);
        if (outcome == ShapeOutcome::kStreamWedged) {
          PrintWaterfall(runner, head_sizes, verdicts);
          std::printf("[ascend-bench] the stream is wedged: exiting %d without the shared report or teardown, "
                      "both of which would block on it. Rerun the other stages with --stage=.\n",
                      kHangExitCode);
          std::fflush(stdout);
          std::_Exit(kHangExitCode);
        }
      }
    }
    PrintWaterfall(runner, head_sizes, verdicts);
  }

  if (bypass_mode == BypassMode::kOff) {
    return;
  }
  std::vector<BypassVerdict> bypass_verdicts;
  for (const int64_t head_size : head_sizes) {
    for (const int64_t context_len : context_lens) {
      BypassVerdict verdict;
      verdict.head_size = head_size;
      verdict.context_len = context_len;
      const ShapeOutcome outcome = RunBypassShape(runner, aiv_num, sync_timeout_ms, &verdict);
      bypass_verdicts.push_back(verdict);
      if (outcome == ShapeOutcome::kStreamWedged) {
        PrintBypassReport(runner, bypass_verdicts);
        std::printf("[ascend-bench] the stream is wedged: exiting %d without the shared report or teardown.\n",
                    kHangExitCode);
        std::fflush(stdout);
        std::_Exit(kHangExitCode);
      }
    }
  }
  PrintBypassReport(runner, bypass_verdicts);
}

}
}
}
