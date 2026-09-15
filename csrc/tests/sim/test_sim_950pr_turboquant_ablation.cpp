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

#include <dirent.h>
#include <sys/stat.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turboquant_launch.hpp"
#include "turboquant_mirrored_cache.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;

constexpr tqm::TurboQuantMode kMode = tqm::TurboQuantMode::KV4_FP8;
constexpr int64_t kBatch = 1;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadSize = 256;
constexpr int64_t kBlockSize = tqh::kCubeTileRows;
constexpr int64_t kPoolFactor = 4;
constexpr float kInvSqrtHeadSize = 0.0625f;
constexpr float kAttentionScale = kInvSqrtHeadSize;

constexpr int64_t kDefaultContextLen = 256;
constexpr int64_t kDefaultStageTimeoutSeconds = 5400;

constexpr double kSmokeCos = 0.90;

constexpr float kWorkspaceSentinel = -1234.5f;

constexpr int kHangExitCode = 3;

constexpr int64_t kBypassNumHeads = 4;
constexpr int64_t kBypassNumKvHeads = 1;
constexpr int64_t kDefaultBypassContextLen = 256;
constexpr int64_t kFloatBytes = 4;
constexpr double kPercent = 100.0;

std::vector<int64_t> IntListFromEnv(const char* name) {
  std::vector<int64_t> parsed;
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return parsed;
  }
  std::stringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (field.empty() || field.find_first_not_of("0123456789") != std::string::npos) {
      return {};
    }
    parsed.push_back(std::strtoll(field.c_str(), nullptr, 10));
  }
  return parsed;
}

int64_t ContextLenFromEnv(const char* name, int64_t fallback) {
  const std::vector<int64_t> parsed = IntListFromEnv(name);
  if (parsed.size() == 1 && parsed[0] > 0 && parsed[0] % kBlockSize == 0) {
    return parsed[0];
  }
  if (std::getenv(name) != nullptr) {
    std::printf("[ ablation ] %s is not one positive multiple of %lld; using %lld\n", name,
                static_cast<long long>(kBlockSize), static_cast<long long>(fallback));
  }
  return fallback;
}

int64_t ContextLen() { return ContextLenFromEnv("ASCEND_TQ_ABLATION_CONTEXT", kDefaultContextLen); }

std::vector<tqm::DecodeAblationStage> Stages() {
  std::vector<tqm::DecodeAblationStage> stages;
  const std::vector<int64_t> parsed = IntListFromEnv("ASCEND_TQ_ABLATION_STAGES");
  bool valid = !parsed.empty();
  for (const int64_t raw : parsed) {
    valid = valid && tqm::DecodeAblationStageIsValid(static_cast<int32_t>(raw));
  }
  if (valid) {
    for (const int64_t raw : parsed) {
      stages.push_back(static_cast<tqm::DecodeAblationStage>(raw));
    }
    return stages;
  }
  if (std::getenv("ASCEND_TQ_ABLATION_STAGES") != nullptr) {
    std::printf("[ ablation ] ASCEND_TQ_ABLATION_STAGES is not a list of stages 0..%d; running all of them\n",
                tqm::kDecodeAblationStageCount - 1);
  }
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    stages.push_back(static_cast<tqm::DecodeAblationStage>(index));
  }
  return stages;
}

int64_t StageTimeoutSeconds() {
  const std::vector<int64_t> parsed = IntListFromEnv("ASCEND_TQ_ABLATION_STAGE_TIMEOUT_S");
  return parsed.size() == 1 && parsed[0] > 0 ? parsed[0] : kDefaultStageTimeoutSeconds;
}

bool HostFillRequested() {
  const char* raw = std::getenv("ASCEND_TQ_ABLATION_HOST_FILL");
  return raw != nullptr && raw[0] == '1';
}

std::vector<int8_t> PseudoRandomBytes(size_t count, uint32_t seed) {
  std::vector<int8_t> bytes(count);
  uint32_t state = seed;
  for (int8_t& byte : bytes) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<int8_t>(state >> 24);
  }
  return bytes;
}

class LaunchWatchdog {
 public:
  explicit LaunchWatchdog(int64_t budget_seconds) : budget_(budget_seconds), thread_([this] { Watch(); }) {}

  ~LaunchWatchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }

  LaunchWatchdog(const LaunchWatchdog&) = delete;
  LaunchWatchdog& operator=(const LaunchWatchdog&) = delete;

  void Arm(const std::string& what) {
    std::lock_guard<std::mutex> lock(mutex_);
    what_ = what;
    deadline_ = std::chrono::steady_clock::now() + budget_;
    armed_ = true;
    ++generation_;
    wake_.notify_all();
  }

  void Disarm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    ++generation_;
    wake_.notify_all();
  }

 private:
  void Watch() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (!armed_) {
        wake_.wait(lock, [this] { return stop_ || armed_; });
        continue;
      }
      const uint64_t generation = generation_;
      const bool changed =
          wake_.wait_until(lock, deadline_, [this, generation] { return stop_ || generation_ != generation; });
      if (!changed) {
        std::printf("\n[ ablation ] HANG: %s did not return within %lld s; ending the process with %d\n",
                    what_.c_str(), static_cast<long long>(budget_.count()), kHangExitCode);
        std::fflush(stdout);
        std::_Exit(kHangExitCode);
      }
    }
  }

  const std::chrono::seconds budget_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::string what_;
  std::chrono::steady_clock::time_point deadline_;
  bool armed_ = false;
  bool stop_ = false;
  uint64_t generation_ = 0;
  std::thread thread_;
};

struct DumpCensus {
  size_t files = 0;
  size_t non_empty = 0;
  uint64_t bytes = 0;
};

DumpCensus ExceptionDumps() {
  DumpCensus census;
  DIR* dir = opendir(".");
  if (dir == nullptr) {
    return census;
  }
  while (const dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    const bool is_dump = name.size() > 5 && name.compare(name.size() - 5, 5, ".dump") == 0;
    if (!is_dump || name.find("excp_log") == std::string::npos) {
      continue;
    }
    struct stat info;
    if (stat(entry->d_name, &info) != 0) {
      continue;
    }
    ++census.files;
    census.bytes += static_cast<uint64_t>(info.st_size);
    if (info.st_size > 0) {
      ++census.non_empty;
    }
  }
  closedir(dir);
  return census;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
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

std::vector<float> HostAttention(int64_t context_len, int64_t num_heads, int64_t num_kv_heads,
                                 const std::vector<float>& query, const std::vector<float>& key,
                                 const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(num_heads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = num_heads / num_kv_heads;
  std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
  for (int64_t h = 0; h < num_heads; ++h) {
    const int64_t kv = h / heads_per_kv;
    double max_logit = -1e30;
    for (int64_t t = 0; t < context_len; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < kHeadSize; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * kHeadSize + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * num_kv_heads + kv) * kHeadSize + d)]);
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
      const double weight = logits[static_cast<size_t>(t)] / denom;
      for (int64_t d = 0; d < kHeadSize; ++d) {
        out[static_cast<size_t>(h * kHeadSize + d)] += static_cast<float>(
            weight * static_cast<double>(value[static_cast<size_t>((t * num_kv_heads + kv) * kHeadSize + d)]));
      }
    }
  }
  return out;
}

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

enum class UnpackPath {
  kStandard,
  kBypass,
};

const char* UnpackPathName(UnpackPath path) {
  return path == UnpackPath::kBypass ? "DecodeSplit_bypass" : "DecodeSplit_standard";
}

std::vector<UnpackPath> UnpackPaths() {
  const char* raw = std::getenv("ASCEND_TQ_BYPASS_PATHS");
  std::vector<UnpackPath> paths;
  if (raw != nullptr && *raw != '\0') {
    std::stringstream stream(raw);
    std::string field;
    bool valid = true;
    if (std::string(raw) == "none") {
      return paths;
    }
    while (std::getline(stream, field, ',')) {
      if (field == "standard") {
        paths.push_back(UnpackPath::kStandard);
      } else if (field == "bypass") {
        paths.push_back(UnpackPath::kBypass);
      } else {
        valid = false;
      }
    }
    if (valid && !paths.empty()) {
      return paths;
    }
    paths.clear();
    std::printf("[ bypass   ] ASCEND_TQ_BYPASS_PATHS is not 'none' or a list of 'standard' and 'bypass'; "
                "running both\n");
  }
  return {UnpackPath::kStandard, UnpackPath::kBypass};
}

aclrtEvent CreateTimelineEvent(bool* timestamped) {
  aclrtEvent event = nullptr;
  if (aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) == ACL_SUCCESS) {
    *timestamped = true;
    return event;
  }
  event = nullptr;
  *timestamped = false;
  ACL_CHECK(aclrtCreateEvent(&event));
  return event;
}

struct LaunchTiming {
  double event_ms = std::numeric_limits<double>::quiet_NaN();
  double host_s = 0.0;
  bool timestamped = false;
  int elapsed_status = ACL_SUCCESS;
};

template <typename Launch>
LaunchTiming TimeLaunch(aclrtStream stream, LaunchWatchdog& watchdog, const std::string& what, Launch&& launch) {
  LaunchTiming timing;
  bool stop_timestamped = false;
  aclrtEvent start = CreateTimelineEvent(&timing.timestamped);
  aclrtEvent stop = CreateTimelineEvent(&stop_timestamped);
  timing.timestamped = timing.timestamped && stop_timestamped;

  watchdog.Arm(what);
  const auto clock = std::chrono::steady_clock::now();
  ACL_CHECK(aclrtRecordEvent(start, stream));
  launch();
  ACL_CHECK(aclrtRecordEvent(stop, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
  timing.host_s = SecondsSince(clock);
  watchdog.Disarm();

  float milliseconds = 0.0f;
  timing.elapsed_status = static_cast<int>(aclrtEventElapsedTime(&milliseconds, start, stop));
  if (timing.elapsed_status == ACL_SUCCESS) {
    timing.event_ms = static_cast<double>(milliseconds);
  }
  ACL_CHECK_NOTHROW(aclrtDestroyEvent(start));
  ACL_CHECK_NOTHROW(aclrtDestroyEvent(stop));
  return timing;
}

struct PathResult {
  UnpackPath path = UnpackPath::kStandard;
  LaunchTiming split;
  LaunchTiming combine;
  std::vector<float> partials;
  size_t untouched = 0;
  size_t finite = 0;
  size_t outputs = 0;
  double abs_sum = 0.0;
  double cos = 0.0;
  DumpCensus dumps;
};

TEST(TurboQuantDecodeAblation, CutStagesExitCleanOnTheCamodel) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  const int64_t context_len = ContextLen();
  const std::vector<tqm::DecodeAblationStage> stages = Stages();
  const int64_t budget_seconds = StageTimeoutSeconds();
  const int64_t blocks_per_seq = context_len / kBlockSize;
  const int64_t num_blocks = blocks_per_seq * kPoolFactor;

  DeterministicRandom rng(0xAB1Au);
  const size_t kv_elems = static_cast<size_t>(context_len * kNumKvHeads * kHeadSize);
  const std::vector<float> key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> query = rng.NormalHalfExact(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f, 1.0f);
  const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks));
  const std::vector<int32_t> block_table(permutation.begin(),
                                         permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq));
  std::vector<int32_t> slots(static_cast<size_t>(context_len));
  for (int64_t i = 0; i < context_len; ++i) {
    slots[static_cast<size_t>(i)] = block_table[static_cast<size_t>(i / kBlockSize)] *
                                        static_cast<int32_t>(kBlockSize) +
                                    static_cast<int32_t>(i % kBlockSize);
  }

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(context_len, aiv_num);
  const tqh::CubeDecodeGrid grid =
      tqh::PlanCubeDecode(kBatch, kNumHeads, kNumKvHeads, kHeadSize, 1, aiv_num);
  ASSERT_EQ(grid.num_splits, 1) << "the smoke needs every tile in one split";

  std::printf("[ ablation ] tier=%s mode=kv4fp8 B=%lld heads=%lld kv_heads=%lld head_size=%lld block=%lld S=%lld\n",
              VLLM_ASCEND_TEST_TIER, static_cast<long long>(kBatch), static_cast<long long>(kNumHeads),
              static_cast<long long>(kNumKvHeads), static_cast<long long>(kHeadSize),
              static_cast<long long>(kBlockSize), static_cast<long long>(context_len));
  std::printf("[ ablation ] vector cores=%lld%s  grid(split)=%u tasks/core=%u  splits=%lld  tiles in the split=%lld  "
              "pool=%lld blocks\n",
              static_cast<long long>(aiv_num), queried ? "" : " (assumed)", grid.split_block_dim,
              grid.split_tasks_per_core, static_cast<long long>(grid.num_splits),
              static_cast<long long>(context_len / tqh::kCubeTileRows), static_cast<long long>(num_blocks));
  std::printf("[ ablation ] stages:");
  for (const tqm::DecodeAblationStage stage : stages) {
    std::printf(" %s", tqm::DecodeAblationStageName(stage));
  }
  std::printf("  (per-launch hang budget %lld s)\n", static_cast<long long>(budget_seconds));
  std::fflush(stdout);

  DeviceBuffer key_dev = DeviceBuffer::FromHost(FloatToHalf(key));
  DeviceBuffer value_dev = DeviceBuffer::FromHost(FloatToHalf(value));
  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(query));
  DeviceBuffer slots_dev = DeviceBuffer::FromHost(slots);
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(block_table);
  DeviceBuffer context_dev = DeviceBuffer::FromHost(std::vector<int32_t>(1, static_cast<int32_t>(context_len)));
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer write_tables = DeviceBuffer::FromHost(tqh::ModeTables(kMode, kHeadSize, 1, 0));
  DeviceBuffer decode_tables =
      DeviceBuffer::FromHost(tqh::ModeTables(kMode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows));
  const bool host_fill = HostFillRequested();
  const size_t cache_bytes = tqh::ModePackedCacheBytes(kMode, num_blocks, kBlockSize, kNumKvHeads, kHeadSize);
  const size_t scale_floats = tqh::ScalePlaneFloats(num_blocks, kBlockSize, kNumKvHeads);
  DeviceBuffer key_cache = host_fill ? DeviceBuffer::FromHost(PseudoRandomBytes(cache_bytes, 0x4B455931u))
                                     : DeviceBuffer::Empty<int8_t>(cache_bytes);
  DeviceBuffer value_cache = host_fill ? DeviceBuffer::FromHost(PseudoRandomBytes(cache_bytes, 0x56414C31u))
                                       : DeviceBuffer::Empty<int8_t>(cache_bytes);
  DeviceBuffer scale_plane = host_fill ? DeviceBuffer::FromHost(std::vector<float>(scale_floats, 1.0f))
                                       : DeviceBuffer::Empty<float>(scale_floats);
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(kNumHeads * kHeadSize));
  const std::vector<float> sentinel(grid.workspace_floats, kWorkspaceSentinel);

  LaunchWatchdog watchdog(budget_seconds);
  const DumpCensus at_start = ExceptionDumps();
  std::printf("[ ablation ] excp_log dumps before any launch: %zu files, %llu bytes\n", at_start.files,
              static_cast<unsigned long long>(at_start.bytes));

  auto clock = std::chrono::steady_clock::now();
  if (host_fill) {
    std::printf("[ ablation ] cache filled from the host (ASCEND_TQ_ABLATION_HOST_FILL=1); stage 5's cosine is "
                "not checked\n");
  } else {
    watchdog.Arm("the cache write");
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(),
        key_cache.get(), value_cache.get(), scale_plane.get(), slots_dev.get(), pi_signs.get(), rot_tables.get(),
        write_tables.get(), static_cast<uint32_t>(context_len), static_cast<uint32_t>(kNumKvHeads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), write_grid.tokens_per_core,
        kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    watchdog.Disarm();
    std::printf("[ ablation ] cache written in %.1f s\n", SecondsSince(clock));
  }
  std::fflush(stdout);

  const std::vector<float> reference = HostAttention(context_len, kNumHeads, kNumKvHeads, query, key, value);

  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(kBatch) * kNumHeads * kHeadSize);
  watchdog.Arm("the query rotation");
  const vllm_ascend::turboquant::RotateQPlan rotate_plan =
      tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs.get(), h16.get(), rot_tables.get(),
                       query_rot.get(), kBatch, kNumHeads, kHeadSize, aiv_num, true);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  watchdog.Disarm();
  std::printf("[ ablation ] query rotated once for every stage: %s path, %u blocks x %u vectors, chunk %u\n",
              rotate_plan.use_cube ? "cube" : "aiv", rotate_plan.block_dim, rotate_plan.vectors_per_block,
              rotate_plan.vectors_per_chunk);
  std::fflush(stdout);

  for (const tqm::DecodeAblationStage stage : stages) {
    const std::string name = tqm::DecodeAblationStageName(stage);
    const bool full = stage == tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE;

    DeviceBuffer workspace = DeviceBuffer::FromHost(sentinel);

    std::printf("\n[ ablation ] %s: launching\n", name.c_str());
    std::fflush(stdout);
    clock = std::chrono::steady_clock::now();
    watchdog.Arm(name);
    turboquant_mm_decode_ablation_impl(
        static_cast<int32_t>(stage), AscendType::FP16, stream, grid.split_block_dim, query_rot.get(),
        key_cache.get(), value_cache.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(),
        decode_tables.get(), workspace.get(), static_cast<uint32_t>(kBatch),
        static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(blocks_per_seq),
        static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    watchdog.Disarm();
    const double seconds = SecondsSince(clock);

    const DumpCensus census = ExceptionDumps();
    const std::vector<float> partials = workspace.ToHost<float>();
    const size_t untouched =
        static_cast<size_t>(std::count(partials.begin(), partials.end(), kWorkspaceSentinel));
    std::printf("[ ablation ] %s: returned in %.1f s; workspace %zu of %zu words still sentinel; "
                "excp_log dumps %zu files, %zu non-empty, %llu bytes\n",
                name.c_str(), seconds, untouched, partials.size(), census.files, census.non_empty,
                static_cast<unsigned long long>(census.bytes));
    std::fflush(stdout);
    EXPECT_EQ(census.bytes, 0u) << name << ": the camodel has written an exception dump by the end of this stage";

    if (!full) {
      EXPECT_EQ(untouched, partials.size())
          << name << " wrote the workspace: a cut below stage 5 must not reach the partial writeback";
      continue;
    }

    EXPECT_EQ(untouched, 0u) << name << " left partials unwritten";
    watchdog.Arm("the combine after " + name);
    turboquant_paged_attention_combine_impl(AscendType::FP16, stream, grid.combine_block_dim, workspace.get(),
                                            out.get(), static_cast<uint32_t>(kBatch),
                                            static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kHeadSize),
                                            static_cast<uint32_t>(grid.num_splits), grid.combine_tasks_per_core);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    watchdog.Disarm();

    const std::vector<float> output = tqh::UnrotateHeads(HalfToFloat(out.ToHost<Half>()), kHeadSize);
    size_t finite = 0;
    double abs_sum = 0.0;
    for (const float v : output) {
      finite += std::isfinite(v) ? 1u : 0u;
      abs_sum += std::fabs(static_cast<double>(v));
    }
    const double cos = Cosine(reference, output);
    std::printf("[ ablation ] %s: output %zu of %zu finite, cos vs fp32 host reference = %.6f\n", name.c_str(),
                finite, output.size(), cos);
    std::fflush(stdout);
    EXPECT_EQ(finite, output.size()) << name << " produced non-finite output";
    EXPECT_GT(abs_sum, 0.0) << name << " produced an identically zero output";
    if (!host_fill) {
      EXPECT_GT(cos, kSmokeCos) << name << " does not track the host reference; this is a structural bound";
    }
  }
}

TEST(TurboQuantDecodeAblation, BypassUnpackKeepsThePipelineOnTheCamodel) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int32_t index = 0; index < tqh::kMirroredLevels; ++index) {
    const float level = tqh::MirroredLevel(index);
    ASSERT_EQ(tqh::Fp8E4m3fnValue(tqh::Fp8E4m3fnBits(level)), level)
        << "level " << level << " is not exact in fp8_e4m3fn";
  }

  const int64_t context_len = ContextLenFromEnv("ASCEND_TQ_BYPASS_CONTEXT", kDefaultBypassContextLen);
  const std::vector<UnpackPath> paths = UnpackPaths();
  const int64_t budget_seconds = StageTimeoutSeconds();
  const int64_t blocks_per_seq = context_len / kBlockSize;
  const int64_t num_blocks = blocks_per_seq * kPoolFactor;
  const int64_t tiles = context_len / tqh::kCubeTileRows;

  DeterministicRandom rng(0xB1A5u);
  const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks));
  const std::vector<int32_t> block_table(permutation.begin(),
                                         permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq));
  const tqh::MirroredKvCache cache =
      tqh::BuildMirroredKvCache(rng, context_len, num_blocks, kBlockSize, kBypassNumKvHeads, kHeadSize, block_table);
  const std::vector<float> query =
      rng.NormalHalfExact(static_cast<size_t>(kBypassNumHeads * kHeadSize), 0.0f, 1.0f);

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::CubeDecodeGrid grid =
      tqh::PlanCubeDecode(kBatch, kBypassNumHeads, kBypassNumKvHeads, kHeadSize, 1, aiv_num);
  ASSERT_EQ(grid.num_splits, 1) << "the smoke needs every tile in one split";

  const int64_t scale_row_bytes = tqh::ScaleSlotFloats(kBypassNumKvHeads) * kFloatBytes;
  const int64_t standard_read =
      kBypassNumKvHeads * tiles * tqh::kCubeTileRows * (2 * tqh::ModePackedBytes(kMode, kHeadSize) + scale_row_bytes);
  const int64_t bypass_read = kBypassNumKvHeads * tiles * tqh::kCubeTileRows * (2 * kHeadSize + scale_row_bytes);

  std::printf("[ bypass   ] tier=%s mode=kv4fp8 B=%lld heads=%lld kv_heads=%lld head_size=%lld block=%lld S=%lld\n",
              VLLM_ASCEND_TEST_TIER, static_cast<long long>(kBatch), static_cast<long long>(kBypassNumHeads),
              static_cast<long long>(kBypassNumKvHeads), static_cast<long long>(kHeadSize),
              static_cast<long long>(kBlockSize), static_cast<long long>(context_len));
  std::printf("[ bypass   ] vector cores=%lld%s  grid(split)=%u tasks/core=%u  splits=%lld  tiles=%lld  pool=%lld "
              "blocks  (per-launch hang budget %lld s)\n",
              static_cast<long long>(aiv_num), queried ? "" : " (assumed)", grid.split_block_dim,
              grid.split_tasks_per_core, static_cast<long long>(grid.num_splits), static_cast<long long>(tiles),
              static_cast<long long>(num_blocks), static_cast<long long>(budget_seconds));
  std::printf("[ bypass   ] GM -> UB tile traffic per launch: standard %lld B (packed int4 + scales), bypass %lld B "
              "(fp8 operands + scales)\n",
              static_cast<long long>(standard_read), static_cast<long long>(bypass_read));
  std::printf("[ bypass   ] cache: K/V halves mirrored (x[j] == x[j + %lld]), so the packed bytes and the fp8 "
              "operand image agree under either nibble order\n",
              static_cast<long long>(kHeadSize / 2));
  std::fflush(stdout);

  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(query));
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(block_table);
  DeviceBuffer context_dev = DeviceBuffer::FromHost(std::vector<int32_t>(1, static_cast<int32_t>(context_len)));
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer decode_tables =
      DeviceBuffer::FromHost(tqh::ModeTables(kMode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows));
  DeviceBuffer key_packed = DeviceBuffer::FromHost(cache.key_packed);
  DeviceBuffer value_packed = DeviceBuffer::FromHost(cache.value_packed);
  DeviceBuffer key_operands = DeviceBuffer::FromHost(cache.key_operands);
  DeviceBuffer value_operands = DeviceBuffer::FromHost(cache.value_operands);
  DeviceBuffer scale_plane = DeviceBuffer::FromHost(cache.scales);
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(kBypassNumHeads * kHeadSize));
  const std::vector<float> sentinel(grid.workspace_floats, kWorkspaceSentinel);

  LaunchWatchdog watchdog(budget_seconds);
  const DumpCensus at_start = ExceptionDumps();
  std::printf("[ bypass   ] excp_log dumps before any launch: %zu files, %llu bytes\n", at_start.files,
              static_cast<unsigned long long>(at_start.bytes));

  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(kBatch * kBypassNumHeads * kHeadSize));
  watchdog.Arm("the query rotation");
  const vllm_ascend::turboquant::RotateQPlan rotate_plan =
      tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs.get(), h16.get(), rot_tables.get(),
                       query_rot.get(), kBatch, kBypassNumHeads, kHeadSize, aiv_num, true);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  watchdog.Disarm();
  const std::vector<float> reference = HostAttention(context_len, kBypassNumHeads, kBypassNumKvHeads,
                                                     query_rot.ToHost<float>(), cache.key, cache.value);
  std::printf("[ bypass   ] query rotated once for both paths: %s path\n", rotate_plan.use_cube ? "cube" : "aiv");
  std::fflush(stdout);

  std::vector<PathResult> results;
  for (const UnpackPath path : paths) {
    PathResult result;
    result.path = path;
    const std::string name = UnpackPathName(path);
    DeviceBuffer workspace = DeviceBuffer::FromHost(sentinel);

    std::printf("\n[ bypass   ] %s: launching split\n", name.c_str());
    std::fflush(stdout);
    result.split = TimeLaunch(stream, watchdog, name, [&] {
      if (path == UnpackPath::kBypass) {
        turboquant_mm_decode_bypass_unpack_impl(
            AscendType::FP16, stream, grid.split_block_dim, query_rot.get(), key_operands.get(),
            value_operands.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(), decode_tables.get(),
            workspace.get(), static_cast<uint32_t>(kBatch), static_cast<uint32_t>(kBypassNumHeads),
            static_cast<uint32_t>(kBypassNumKvHeads), static_cast<uint32_t>(kHeadSize),
            static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(blocks_per_seq),
            static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);
      } else {
        turboquant_mm_decode_split_impl(
            static_cast<int32_t>(kMode), AscendType::FP16, stream, grid.split_block_dim, query_rot.get(),
            key_packed.get(), value_packed.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(),
            decode_tables.get(), workspace.get(), static_cast<uint32_t>(kBatch),
            static_cast<uint32_t>(kBypassNumHeads), static_cast<uint32_t>(kBypassNumKvHeads),
            static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
            static_cast<uint32_t>(blocks_per_seq), static_cast<uint32_t>(grid.num_splits),
            grid.split_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);
      }
    });
    result.partials = workspace.ToHost<float>();
    result.untouched =
        static_cast<size_t>(std::count(result.partials.begin(), result.partials.end(), kWorkspaceSentinel));
    if (const char* dir = std::getenv("ASCEND_TQ_BYPASS_PARTIALS_DIR")) {
      const std::string file = std::string(dir) + "/" + name + ".partials.f32";
      FILE* sink = std::fopen(file.c_str(), "wb");
      ASSERT_NE(sink, nullptr) << "cannot open " << file;
      const size_t written = std::fwrite(result.partials.data(), sizeof(float), result.partials.size(), sink);
      std::fclose(sink);
      ASSERT_EQ(written, result.partials.size()) << "short write to " << file;
      std::printf("[ bypass   ] %s: partials written to %s\n", name.c_str(), file.c_str());
    }
    std::printf("[ bypass   ] %s: split returned, event %.3f ms (status %d), host %.1f s; workspace %zu of %zu "
                "words still sentinel\n",
                name.c_str(), result.split.event_ms, result.split.elapsed_status, result.split.host_s,
                result.untouched, result.partials.size());
    std::fflush(stdout);

    result.combine = TimeLaunch(stream, watchdog, "the combine after " + name, [&] {
      turboquant_paged_attention_combine_impl(AscendType::FP16, stream, grid.combine_block_dim, workspace.get(),
                                              out.get(), static_cast<uint32_t>(kBatch),
                                              static_cast<uint32_t>(kBypassNumHeads),
                                              static_cast<uint32_t>(kHeadSize),
                                              static_cast<uint32_t>(grid.num_splits), grid.combine_tasks_per_core);
    });
    const std::vector<float> output = HalfToFloat(out.ToHost<Half>());
    result.outputs = output.size();
    for (const float v : output) {
      result.finite += std::isfinite(v) ? 1u : 0u;
      result.abs_sum += std::fabs(static_cast<double>(v));
    }
    result.cos = Cosine(reference, output);
    result.dumps = ExceptionDumps();
    std::printf("[ bypass   ] %s: combine returned, event %.3f ms, host %.1f s; output %zu of %zu finite, "
                "cos vs fp32 host = %.6f; excp_log dumps %zu files, %zu non-empty, %llu bytes\n",
                name.c_str(), result.combine.event_ms, result.combine.host_s, result.finite, result.outputs,
                result.cos, result.dumps.files, result.dumps.non_empty,
                static_cast<unsigned long long>(result.dumps.bytes));
    std::fflush(stdout);

    EXPECT_EQ(result.untouched, 0u) << name << " left partials unwritten";
    EXPECT_EQ(result.dumps.bytes, 0u) << name << ": the camodel has written an exception dump";
    EXPECT_EQ(result.finite, result.outputs) << name << " produced non-finite output";
    EXPECT_GT(result.abs_sum, 0.0) << name << " produced an identically zero output";
    EXPECT_GT(result.cos, kSmokeCos) << name << " does not track the host reference";
    results.push_back(std::move(result));
  }

  std::printf("\n[ bypass   ] %-22s %14s %10s %16s %10s\n", "path", "split_event_ms", "split_s", "combine_event_ms",
              "cos");
  for (const PathResult& result : results) {
    std::printf("[ bypass   ] %-22s %14.3f %10.1f %16.3f %10.6f\n", UnpackPathName(result.path),
                result.split.event_ms, result.split.host_s, result.combine.event_ms, result.cos);
  }
  std::printf("[ bypass   ] event source: %s\n",
              !results.empty() && results.front().split.timestamped
                  ? "ACL_EVENT_TIME_LINE|ACL_EVENT_SYNC"
                  : "aclrtCreateEvent default (no timestamp guarantee)");

  if (results.size() == 2 && results[0].path != results[1].path) {
    const PathResult& standard = results[0].path == UnpackPath::kStandard ? results[0] : results[1];
    const PathResult& bypass = results[0].path == UnpackPath::kBypass ? results[0] : results[1];
    size_t differing = 0;
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < standard.partials.size() && i < bypass.partials.size(); ++i) {
      if (std::memcmp(&standard.partials[i], &bypass.partials[i], sizeof(float)) != 0) {
        ++differing;
        max_abs_diff = std::max(
            max_abs_diff, std::fabs(static_cast<double>(standard.partials[i]) - bypass.partials[i]));
      }
    }
    std::printf("[ bypass   ] partial words differing between the paths: %zu of %zu (max |diff| %.3e)\n",
                differing, standard.partials.size(), max_abs_diff);
    if (standard.split.event_ms > 0.0 && std::isfinite(bypass.split.event_ms)) {
      const double ratio = bypass.split.event_ms / standard.split.event_ms;
      std::printf("[ bypass   ] split device time bypass/standard = %.4f; the removed unpack accounts for %.1f%% "
                  "of DecodeSplit_standard\n",
                  ratio, (1.0 - ratio) * kPercent);
    } else {
      std::printf("[ bypass   ] the runtime timestamped neither split (event elapsed %.3f / %.3f ms): no device time "
                  "from ACL events here, only host wall\n",
                  standard.split.event_ms, bypass.split.event_ms);
    }
    std::printf("[ bypass   ] split host wall bypass/standard = %.4f (%.1f s vs %.1f s)\n",
                standard.split.host_s > 0.0 ? bypass.split.host_s / standard.split.host_s : 0.0,
                bypass.split.host_s, standard.split.host_s);
    EXPECT_EQ(standard.partials.size(), bypass.partials.size());
    EXPECT_EQ(differing, 0u) << "the bypass does not stage the operands the codec unpacks";
  }
  std::fflush(stdout);
}

}
}
}
