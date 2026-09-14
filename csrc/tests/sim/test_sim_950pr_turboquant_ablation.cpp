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

// Camodel smoke of the decode ablation ladder: every DecodeAblationStage of the
// kv4fp8 Cube split, dispatched in turn in one process.
//
// The cut kernels compile cross-core flags and barriers out in pairs. A pair
// removed on one side and not the other does not fault, it hangs -- and on
// silicon it hangs with nothing in the log. The benchmark
// (device/bench_device_950pr_turboquant_ablation.cpp) refuses the camodel, so
// this is the only place a cut kernel runs before the part does.
//
// THE SHAPE: B = 1, 8 query heads over 2 kv heads (the production group of 4),
// head_size 256, block_size 64, S = 256.
//
//   block_size 64   the smallest block CopyInTile supports -- it reads a whole
//                   kCubeTileRows = 64 rows per tile -- so every tile is also a
//                   block and NextTile's block walk runs on every tile.
//   S = 256         four tiles in ONE split. The split count is pinned to 1 so
//                   the tiles are not spread across tasks. Four is the smallest
//                   tile count at which every flag id the pipelined loop spends
//                   fires: the L1 slots wrap at tile 2, and kFlagSlotFree fires
//                   for slot 0 at tile 0 and for slot 1 at tile 1. Three tiles
//                   never frees slot 1.
//
// WHAT IS ASSERTED, per stage, in launch order:
//
//   exits       the launch returns and the stream synchronises within the
//               per-stage budget. A stage that does not is named and the
//               process exits kHangExitCode: a deadlocked stream cannot be
//               recovered for the stages after it.
//   no dumps    no core*excp_log*.dump in the cwd holds a byte. Whether the
//               camodel writes those as it goes or only at exit is not something
//               this file can know, so the run script's census after exit is
//               the authoritative one; this per-stage figure attributes only what
//               it can see.
//   untouched   stages 0-4: the partial workspace -- the split's only GM output
//               -- still holds the sentinel it was filled with.
//   fidelity    stage 5: every partial was written, and split plus combine track
//               an fp32 host attention at cos > kSmokeCos.
//
// Nothing here is a timing. The per-stage durations printed are camodel wall
// clock, there to tell a slow stage from a stuck one.
//
// Knobs:
//   ASCEND_TQ_ABLATION_STAGES=0,4,5          stages to dispatch, in that order (default 0..5)
//   ASCEND_TQ_ABLATION_CONTEXT=192           context, a positive multiple of 64 (default 256)
//   ASCEND_TQ_ABLATION_STAGE_TIMEOUT_S=3600  per-launch hang budget in seconds (default 5400)
//   ASCEND_TQ_ABLATION_HOST_FILL=1           fill the packed cache from the host instead of
//                                            launching the write, which is 20 minutes of
//                                            camodel at S = 256. The flag protocol does not
//                                            read the values; stage 5's cosine is then not
//                                            checked.

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
constexpr int64_t kBlockSize = tqh::kCubeTileRows;  // 64
constexpr int64_t kPoolFactor = 4;
constexpr float kInvSqrtHeadSize = 0.0625f;  // 1 / sqrt(256), exact in fp32
constexpr float kAttentionScale = kInvSqrtHeadSize;

constexpr int64_t kDefaultContextLen = 256;
constexpr int64_t kDefaultStageTimeoutSeconds = 5400;

// The multimode test's structural bound: kv4fp8 measures 0.986 at S = 64, and a
// transposed operand or a mis-staged row lands near zero.
constexpr double kSmokeCos = 0.90;

// Filled into every workspace word before a launch. No partial the split writes
// can equal it: the accumulator is a weighted sum of unit-scale values, and the
// tail is a running max and sum of probabilities.
constexpr float kWorkspaceSentinel = -1234.5f;

// Distinct from gtest's 1 and from timeout(1)'s 124, so the run script can say
// which of the three ended the process.
constexpr int kHangExitCode = 3;

// --- knobs ------------------------------------------------------------------

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

int64_t ContextLen() {
  const std::vector<int64_t> parsed = IntListFromEnv("ASCEND_TQ_ABLATION_CONTEXT");
  if (parsed.size() == 1 && parsed[0] > 0 && parsed[0] % kBlockSize == 0) {
    return parsed[0];
  }
  if (std::getenv("ASCEND_TQ_ABLATION_CONTEXT") != nullptr) {
    std::printf("[ ablation ] ASCEND_TQ_ABLATION_CONTEXT is not one positive multiple of %lld; using %lld\n",
                static_cast<long long>(kBlockSize), static_cast<long long>(kDefaultContextLen));
  }
  return kDefaultContextLen;
}

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

// Deterministic bytes for a host-filled packed cache. Any byte is two valid
// 4-bit codes, so the unpack sees the kind of input the encoder writes.
std::vector<int8_t> PseudoRandomBytes(size_t count, uint32_t seed) {
  std::vector<int8_t> bytes(count);
  uint32_t state = seed;
  for (int8_t& byte : bytes) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<int8_t>(state >> 24);
  }
  return bytes;
}

// --- hang detection ------------------------------------------------------------

// Armed around one launch and its synchronise. If it is not disarmed within the
// budget it names the launch and ends the process, because the thread that
// would record a gtest failure is the one blocked in aclrtSynchronizeStream.
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
  // Last, so everything Watch() touches is constructed before it starts.
  std::thread thread_;
};

// --- camodel exception dumps -------------------------------------------------

struct DumpCensus {
  size_t files = 0;
  size_t non_empty = 0;
  uint64_t bytes = 0;
};

// The camodel's exception dumps in the process cwd: core<N>...excp_log...dump.
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

// --- the host reference ---------------------------------------------------------

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

// fp32 attention for one decode token over the unquantised context. Same
// computation as HostAttention in test_sim_950pr_turboquant_multimode.cpp at
// B = 1: no rotation and no codec, so agreeing with it says the whole pipeline
// is right rather than one stage of it.
std::vector<float> HostAttention(int64_t context_len, const std::vector<float>& query, const std::vector<float>& key,
                                 const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
  for (int64_t h = 0; h < kNumHeads; ++h) {
    const int64_t kv = h / heads_per_kv;
    double max_logit = -1e30;
    for (int64_t t = 0; t < context_len; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < kHeadSize; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * kHeadSize + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * kNumKvHeads + kv) * kHeadSize + d)]);
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
            weight * static_cast<double>(value[static_cast<size_t>((t * kNumKvHeads + kv) * kHeadSize + d)]));
      }
    }
  }
  return out;
}

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

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
  // max_blocks_per_seq = 1 here is what caps PlanCubeDecode's split count at
  // one. The kernel is still handed the real blocks_per_seq below; at B = 1 it
  // only ever indexes row 0 of the block table, so the planner's argument
  // changes the grid and nothing else.
  const tqh::CubeDecodeGrid grid =
      tqh::PlanCubeDecode(kBatch, kNumHeads, kNumKvHeads, kHeadSize, /*max_blocks_per_seq=*/1, aiv_num);
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
  DeviceBuffer write_tables = DeviceBuffer::FromHost(tqh::ModeTables(kMode, kHeadSize, 1, /*nz_rows=*/0));
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

  const std::vector<float> reference = HostAttention(context_len, query, key, value);

  // The rotation is outside the ladder: every rung consumes the same
  // pre-rotated query, so no rung's delta carries any part of it. That is the
  // change this ladder now measures the absence of -- stage 2 used to be the
  // Pi transform and is now the read and the operand scaling alone.
  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(kBatch) * kNumHeads * kHeadSize);
  watchdog.Arm("the query rotation");
  const vllm_ascend::turboquant::RotateQPlan rotate_plan =
      tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs.get(), h16.get(), rot_tables.get(),
                       query_rot.get(), kBatch, kNumHeads, kHeadSize, aiv_num, /*input_exact_in_half=*/true);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  watchdog.Disarm();
  std::printf("[ ablation ] query rotated once for every stage: %s path, %u blocks x %u vectors, chunk %u\n",
              rotate_plan.use_cube ? "cube" : "aiv", rotate_plan.block_dim, rotate_plan.vectors_per_block,
              rotate_plan.vectors_per_chunk);
  std::fflush(stdout);

  for (const tqm::DecodeAblationStage stage : stages) {
    const std::string name = tqm::DecodeAblationStageName(stage);
    const bool full = stage == tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE;

    // Fresh sentinel before every stage, so no stage's check depends on what
    // ran before it and stage 5's "every partial written" is not satisfied by a
    // leftover.
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

    // The combine writes the rotated basis; the folded W_o un-rotates in
    // production and UnrotateHeads does it here.
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

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
