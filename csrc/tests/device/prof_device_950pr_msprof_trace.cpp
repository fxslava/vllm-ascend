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

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_audit_models.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace trace {
namespace {

namespace tqa = turboquant_audit;
namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;

constexpr const char* kTag = "[tq-trace]";
constexpr const char* kLegTurboQuant = "TQ_Pipeline";
constexpr const char* kLegNative = "V5_Native";
constexpr const char* kStartSuffix = "_Start";
constexpr const char* kEndSuffix = "_End";
constexpr const char* kAbortSuffix = "_Aborted";

constexpr int64_t kBlockSize = s950::kDefaultBlockSize;
constexpr int64_t kCausalMaskSide = 2048;
constexpr int64_t kFiaSparseModeRightDownCausal = 3;
constexpr int64_t kFiaNoPaging = 0;
constexpr int64_t kPatternTokens = 1024;
constexpr int64_t kContextMultiple = 8;
constexpr int64_t kAnyPositive = 1;
constexpr size_t kMaxListDigits = 9;
constexpr size_t kReadbackElements = 1u << 20;
constexpr size_t kPathCapacity = 4096;
constexpr uint32_t kSeed = 0x7AC3u;
constexpr std::chrono::milliseconds kTimelineGap(10);
constexpr tqm::TurboQuantMode kCubeMode = tqm::TurboQuantMode::KV4_FP8;

constexpr const char* kModelsFlag = "--models=";
constexpr const char* kContextsFlag = "--contexts=";
constexpr const char* kBatchesFlag = "--batches=";
constexpr const char* kModeFlag = "--mode=";
constexpr const char* kModelsEnv = "ASCEND_TQ_TRACE_MODELS";
constexpr const char* kContextsEnv = "ASCEND_TQ_TRACE_CONTEXTS";
constexpr const char* kBatchesEnv = "ASCEND_TQ_TRACE_BATCHES";
constexpr const char* kModeEnv = "ASCEND_TQ_TRACE_MODE";
constexpr const char* kDefaultContexts = "2048,32768";
constexpr const char* kDefaultBatches = "1,4";
constexpr const char* kDefaultMode = "decode";

constexpr const char* kMsprofOutput = "./prof/tq_trace";
constexpr const char* kMstxLibrary = "libms_tools_ext.so";
constexpr const char* kMstxUnderToolkit = "/tools/mstx/lib64/libms_tools_ext.so";
constexpr const char* kDefaultToolkit = "/usr/local/Ascend/ascend-toolkit/latest";
constexpr uint64_t kMstxToolInvalidId = 0x0;
constexpr uint64_t kMstxToolMsprofId = 0x1000;

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

enum class Phase { kDecode, kPrefill };

const char* PhaseLabel(Phase phase) { return phase == Phase::kDecode ? "decode" : "prefill"; }

struct Options {
  std::vector<tqa::ModelSpec> models;
  std::vector<int64_t> contexts;
  std::vector<int64_t> batches;
  std::vector<Phase> phases;
  std::string mode;
};

std::vector<std::string> SplitCsv(const std::string& raw) {
  std::vector<std::string> fields;
  std::istringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (!field.empty()) {
      fields.push_back(field);
    }
  }
  return fields;
}

std::string EnvOr(const char* name, const char* fallback) {
  const char* raw = std::getenv(name);
  return raw == nullptr || *raw == '\0' ? std::string(fallback) : std::string(raw);
}

bool TakeFlag(const char* arg, const char* flag, std::string* value) {
  const size_t length = std::strlen(flag);
  if (std::strncmp(arg, flag, length) != 0) {
    return false;
  }
  *value = arg + length;
  return true;
}

std::string ModelChoices(const std::vector<tqa::ModelSpec>& models) {
  std::ostringstream text;
  for (size_t index = 0; index < models.size(); ++index) {
    text << (index == 0 ? "" : ", ") << models[index].label << " (" << models[index].key << ')';
  }
  return text.str();
}

bool ParseModels(const std::string& raw, std::vector<tqa::ModelSpec>* models, std::string* error) {
  const std::vector<tqa::ModelSpec> known = tqa::Models();
  if (raw.empty()) {
    *models = known;
    return true;
  }
  for (const std::string& field : SplitCsv(raw)) {
    const auto match = std::find_if(known.begin(), known.end(), [&field](const tqa::ModelSpec& model) {
      return field == model.label || field == model.key;
    });
    if (match == known.end()) {
      *error = "unknown model '" + field + "'; expected one of " + ModelChoices(known);
      return false;
    }
    const bool seen = std::any_of(models->begin(), models->end(), [&match](const tqa::ModelSpec& model) {
      return std::strcmp(model.key, match->key) == 0;
    });
    if (!seen) {
      models->push_back(*match);
    }
  }
  if (models->empty()) {
    *error = "the list is empty";
    return false;
  }
  return true;
}

bool ParseCounts(const std::string& raw, int64_t multiple, std::vector<int64_t>* values, std::string* error) {
  const std::vector<std::string> fields = SplitCsv(raw);
  if (fields.empty()) {
    *error = "the list is empty";
    return false;
  }
  for (const std::string& field : fields) {
    const bool digits =
        field.size() <= kMaxListDigits && field.find_first_not_of("0123456789") == std::string::npos;
    const int64_t value = digits ? static_cast<int64_t>(std::strtoll(field.c_str(), nullptr, 10)) : 0;
    if (value <= 0 || value % multiple != 0) {
      std::ostringstream why;
      why << "'" << field << "' is not a positive integer";
      if (multiple > kAnyPositive) {
        why << " multiple of " << multiple;
      }
      *error = why.str();
      return false;
    }
    if (std::find(values->begin(), values->end(), value) == values->end()) {
      values->push_back(value);
    }
  }
  return true;
}

bool ParseMode(const std::string& raw, Options* options, std::string* error) {
  if (raw == "decode") {
    options->phases = {Phase::kDecode};
  } else if (raw == "prefill") {
    options->phases = {Phase::kPrefill};
  } else if (raw == "both") {
    options->phases = {Phase::kDecode, Phase::kPrefill};
  } else {
    *error = "'" + raw + "' is not one of decode, prefill, both";
    return false;
  }
  options->mode = raw;
  return true;
}

void PrintUsage(const char* argv0) {
  std::printf(
      "usage: %s [--models=<list>] [--contexts=<list>] [--batches=<list>] [--mode=decode|prefill|both]\n"
      "\n"
      "One msprof timeline of the TurboQuant attention chain against aclnnFusedInferAttentionScoreV5.\n"
      "Every shape launches each leg exactly once -- no warmup, no timing loop:\n"
      "  TQ_Pipeline  npu_turboquant_rotate_q -> TQ_FusedDecode, one launch [-> rotate_o, unfolded W_o only]\n"
      "  V5_Native    GetWorkspaceSize + workspace -> aclnnFusedInferAttentionScoreV5\n"
      "with a %lld ms gap after each leg.\n"
      "\n"
      "  --models=<list>    any of %s.\n"
      "                     Default: all three. Env %s.\n"
      "  --contexts=<list>  context lengths S, positive multiples of %lld. Default %s. Env %s.\n"
      "  --batches=<list>   batch sizes B. Default %s. Env %s.\n"
      "  --mode=<mode>      decode, prefill (one chunked step, C = min(S, %lld) query tokens), or both.\n"
      "                     Default %s. Env %s.\n"
      "\n"
      "Flags override the environment. The decode path, GLM-5.2's head size and the prefill chunk\n"
      "follow ASCEND_BENCH_TQ_AUDIT_PATH, _GLM_D and _CHUNK, as bench_device_950pr_turboquant does.\n"
      "The markers are recorded only under msprof --msproftx=on; the binary prints that command.\n",
      argv0, static_cast<long long>(kTimelineGap.count()), ModelChoices(tqa::Models()).c_str(), kModelsEnv,
      static_cast<long long>(kContextMultiple), kDefaultContexts, kContextsEnv, kDefaultBatches, kBatchesEnv,
      static_cast<long long>(tqa::kPrefillChunkTokens), kDefaultMode, kModeEnv);
}

bool Reject(const char* what, const std::string& error, int* exit_code) {
  std::fprintf(stderr, "%s %s: %s\n", kTag, what, error.c_str());
  *exit_code = kExitUsage;
  return false;
}

bool ParseCommandLine(int argc, char** argv, Options* options, int* exit_code) {
  std::string models = EnvOr(kModelsEnv, "");
  std::string contexts = EnvOr(kContextsEnv, kDefaultContexts);
  std::string batches = EnvOr(kBatchesEnv, kDefaultBatches);
  std::string mode = EnvOr(kModeEnv, kDefaultMode);
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (TakeFlag(arg, kModelsFlag, &models) || TakeFlag(arg, kContextsFlag, &contexts) ||
        TakeFlag(arg, kBatchesFlag, &batches) || TakeFlag(arg, kModeFlag, &mode)) {
      continue;
    }
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      *exit_code = kExitOk;
      return false;
    }
    std::fprintf(stderr, "%s unrecognised argument '%s'\n", kTag, arg);
    PrintUsage(argv[0]);
    *exit_code = kExitUsage;
    return false;
  }
  std::string error;
  if (!ParseModels(models, &options->models, &error)) {
    return Reject("--models", error, exit_code);
  }
  if (!ParseCounts(contexts, kContextMultiple, &options->contexts, &error)) {
    return Reject("--contexts", error + " (see TURBOQUANT_TESTS.md 13.20)", exit_code);
  }
  if (!ParseCounts(batches, kAnyPositive, &options->batches, &error)) {
    return Reject("--batches", error, exit_code);
  }
  if (!ParseMode(mode, options, &error)) {
    return Reject("--mode", error, exit_code);
  }
  return true;
}

struct TraceConfig {
  tqa::ModelSpec model{};
  int64_t seq_len = 0;
  int64_t batch = 0;
  int64_t chunk = 0;
  Phase phase = Phase::kDecode;
  tqa::PathMode path = tqa::PathMode::kAiv;

  int64_t context_tokens() const { return batch * seq_len; }
  int64_t chunk_tokens() const { return batch * chunk; }
  int64_t blocks_per_seq() const { return tqh::CeilDiv(seq_len, kBlockSize); }
  int64_t pool_blocks() const { return blocks_per_seq() * batch; }
  float attention_scale() const { return 1.0f / std::sqrt(static_cast<float>(model.head_size)); }

  std::string label() const {
    std::ostringstream text;
    text << model.label << " S=" << seq_len << " B=" << batch << ' ' << PhaseLabel(phase);
    return text.str();
  }
};

std::vector<TraceConfig> BuildShapes(const Options& options) {
  std::vector<TraceConfig> shapes;
  for (const tqa::ModelSpec& model : options.models) {
    for (const int64_t seq_len : options.contexts) {
      for (const int64_t batch : options.batches) {
        for (const Phase phase : options.phases) {
          TraceConfig config;
          config.model = model;
          config.seq_len = seq_len;
          config.batch = batch;
          config.chunk = tqa::PrefillChunk(seq_len);
          config.phase = phase;
          config.path = tqa::SelectPath(model);
          shapes.push_back(config);
        }
      }
    }
  }
  return shapes;
}

size_t Elements(int64_t a, int64_t b, int64_t c) {
  return static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(c);
}

template <typename T>
void TileToDevice(const DeviceBuffer& dst, const std::vector<T>& pattern) {
  const size_t total = dst.size_bytes();
  const size_t chunk = pattern.size() * sizeof(T);
  if (chunk == 0) {
    return;
  }
  for (size_t offset = 0; offset < total; offset += chunk) {
    const size_t bytes = std::min(chunk, total - offset);
    ACL_CHECK(aclrtMemcpy(static_cast<char*>(dst.get()) + offset, dst.capacity_bytes() - offset, pattern.data(),
                          bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }
}

std::vector<float> Rotated(std::vector<float> values, int64_t head_size) {
  const std::vector<int8_t> signs = turboquant_ref::cpu_pi_sign_vector(static_cast<int>(head_size));
  const size_t row = static_cast<size_t>(head_size);
  for (size_t base = 0; base + row <= values.size(); base += row) {
    turboquant_ref::cpu_apply_pi(values.data() + base, static_cast<int>(head_size), signs.data());
  }
  return values;
}

template <typename T>
std::vector<T> Leading(const DeviceBuffer& buffer) {
  std::vector<T> host(std::min(buffer.size_bytes() / sizeof(T), kReadbackElements));
  if (!host.empty()) {
    buffer.CopyToHost(host.data(), host.size() * sizeof(T));
  }
  return host;
}

double Magnitude(const std::vector<float>& values) {
  double sum = 0.0;
  for (const float value : values) {
    sum += std::fabs(static_cast<double>(value));
  }
  return sum;
}

double HalfMagnitude(const DeviceBuffer& buffer) { return Magnitude(HalfToFloat(Leading<Half>(buffer))); }
double FloatMagnitude(const DeviceBuffer& buffer) { return Magnitude(Leading<float>(buffer)); }

using MstxMarkFn = void (*)(const char*, aclrtStream);
using MstxRangeStartFn = uint64_t (*)(const char*, aclrtStream);
using MstxRangeEndFn = void (*)(uint64_t);
using MstxGetToolIdFn = void (*)(uint64_t*);
using ProfMarkExFn = int (*)(const char*, size_t, aclrtStream);

std::vector<std::string> MstxCandidates() {
  std::vector<std::string> candidates = {kMstxLibrary};
  for (const char* variable : {"ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"}) {
    const char* root = std::getenv(variable);
    if (root != nullptr && *root != '\0') {
      candidates.push_back(std::string(root) + kMstxUnderToolkit);
    }
  }
  candidates.push_back(std::string(kDefaultToolkit) + kMstxUnderToolkit);
  return candidates;
}

class TraceMarkers {
 public:
  TraceMarkers() {
    for (const std::string& candidate : MstxCandidates()) {
      void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (handle == nullptr) {
        continue;
      }
      const MstxMarkFn mark = reinterpret_cast<MstxMarkFn>(dlsym(handle, "mstxMarkA"));
      const MstxRangeStartFn start = reinterpret_cast<MstxRangeStartFn>(dlsym(handle, "mstxRangeStartA"));
      const MstxRangeEndFn end = reinterpret_cast<MstxRangeEndFn>(dlsym(handle, "mstxRangeEnd"));
      if (mark == nullptr || start == nullptr || end == nullptr) {
        dlclose(handle);
        continue;
      }
      mstx_mark_ = mark;
      mstx_range_start_ = start;
      mstx_range_end_ = end;
      mstx_get_tool_id_ = reinterpret_cast<MstxGetToolIdFn>(dlsym(handle, "mstxGetToolId"));
      source_ = candidate;
      return;
    }
    prof_mark_ex_ = reinterpret_cast<ProfMarkExFn>(dlsym(RTLD_DEFAULT, "aclprofMarkEx"));
    if (prof_mark_ex_ != nullptr) {
      source_ = "aclprofMarkEx";
    }
  }

  bool has_ranges() const { return mstx_range_start_ != nullptr; }
  bool available() const { return mstx_mark_ != nullptr || prof_mark_ex_ != nullptr; }
  const std::string& source() const { return source_; }

  uint64_t ToolId() const {
    uint64_t id = kMstxToolInvalidId;
    if (mstx_get_tool_id_ != nullptr) {
      mstx_get_tool_id_(&id);
    }
    return id;
  }

  void Mark(const std::string& message, aclrtStream stream) const {
    if (mstx_mark_ != nullptr) {
      mstx_mark_(message.c_str(), stream);
      return;
    }
    if (prof_mark_ex_ != nullptr) {
      static_cast<void>(prof_mark_ex_(message.c_str(), message.size(), stream));
    }
  }

  uint64_t RangeStart(const std::string& name, aclrtStream stream) const {
    if (mstx_range_start_ == nullptr) {
      Mark(name + kStartSuffix, stream);
      return 0;
    }
    return mstx_range_start_(name.c_str(), stream);
  }

  void RangeEnd(uint64_t id, const std::string& name, aclrtStream stream) const {
    if (mstx_range_end_ == nullptr) {
      Mark(name + kEndSuffix, stream);
      return;
    }
    if (id != 0) {
      mstx_range_end_(id);
    }
  }

 private:
  MstxMarkFn mstx_mark_ = nullptr;
  MstxRangeStartFn mstx_range_start_ = nullptr;
  MstxRangeEndFn mstx_range_end_ = nullptr;
  MstxGetToolIdFn mstx_get_tool_id_ = nullptr;
  ProfMarkExFn prof_mark_ex_ = nullptr;
  std::string source_;
};

class TraceRange {
 public:
  TraceRange(const TraceMarkers& markers, std::string name, aclrtStream stream)
      : markers_(markers), name_(std::move(name)), stream_(stream), id_(markers_.RangeStart(name_, stream_)) {}
  ~TraceRange() { markers_.RangeEnd(id_, name_, stream_); }

  TraceRange(const TraceRange&) = delete;
  TraceRange& operator=(const TraceRange&) = delete;

 private:
  const TraceMarkers& markers_;
  std::string name_;
  aclrtStream stream_;
  uint64_t id_;
};

void Synchronize(const TraceMarkers& markers, aclrtStream stream, const char* name) {
  const TraceRange range(markers, name, nullptr);
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

using InitHugeMemThreadLocalFn = int (*)(void*, bool);
using UnInitHugeMemThreadLocalFn = void (*)(void*, bool);

class HugeMemThreadScope {
 public:
  HugeMemThreadScope() {
    const OpApiLibrary& library = OpApiLibrary::Instance();
    const InitHugeMemThreadLocalFn initialise =
        reinterpret_cast<InitHugeMemThreadLocalFn>(library.Resolve("InitHugeMemThreadLocal"));
    uninitialise_ = reinterpret_cast<UnInitHugeMemThreadLocalFn>(library.Resolve("UnInitHugeMemThreadLocal"));
    if (initialise != nullptr) {
      static_cast<void>(initialise(nullptr, false));
    }
  }
  ~HugeMemThreadScope() {
    if (uninitialise_ != nullptr) {
      uninitialise_(nullptr, false);
    }
  }

  HugeMemThreadScope(const HugeMemThreadScope&) = delete;
  HugeMemThreadScope& operator=(const HugeMemThreadScope&) = delete;

 private:
  UnInitHugeMemThreadLocalFn uninitialise_ = nullptr;
};

const AclnnOp& FiaV5() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV5);
  return op;
}

struct NativeLaunch {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  DeviceBuffer workspace;
};

template <typename WorkspaceSizeFn, typename... Args>
void PlanNative(const AclnnOp& op, NativeLaunch* launch, Args... args) {
  if (!op.available()) {
    throw AclError(op.unavailable_reason().c_str(), __FILE__, __LINE__, -1);
  }
  const int status = reinterpret_cast<WorkspaceSizeFn>(op.get_workspace_size_fn())(
      args..., &launch->workspace_size, &launch->executor);
  if (status != 0) {
    const std::string label = op.name() + "GetWorkspaceSize";
    throw AclError(label.c_str(), __FILE__, __LINE__, status);
  }
  if (launch->workspace_size > 0) {
    launch->workspace.Allocate(static_cast<size_t>(launch->workspace_size));
  }
}

void LaunchNative(const AclnnOp& op, const NativeLaunch& launch, aclrtStream stream) {
  const int status = reinterpret_cast<AclnnLaunchFn>(op.launch_fn())(launch.workspace.get(), launch.workspace_size,
                                                                     launch.executor, stream);
  if (status != 0) {
    throw AclError(op.name().c_str(), __FILE__, __LINE__, status);
  }
}

class Scenario {
 public:
  Scenario(const TraceConfig& config, int64_t aiv_num)
      : config_(config),
        aiv_num_(aiv_num),
        rng_(kSeed + static_cast<uint32_t>(config.seq_len + 7 * config.batch + 31 * config.model.head_size)),
        pi_signs_(DeviceBuffer::FromHost(tqh::PiSigns(config.model.head_size), kBenchmarkAlignBytes)),
        h16_(DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes)),
        rot_tables_(DeviceBuffer::FromHost(tqh::CodecTables(config.model.head_size, 1), kBenchmarkAlignBytes)) {}
  virtual ~Scenario() = default;

  Scenario(const Scenario&) = delete;
  Scenario& operator=(const Scenario&) = delete;

  virtual std::string Route() const = 0;
  virtual void TurboQuantLeg(const TraceMarkers& markers, aclrtStream stream) = 0;
  virtual void NativeLeg(const TraceMarkers& markers, aclrtStream stream) = 0;
  virtual double TurboQuantMagnitude() const = 0;
  virtual double NativeMagnitude() const = 0;

 protected:
  void EnqueueRotation(aclrtStream stream, const DeviceBuffer& input, const DeviceBuffer& output,
                       int64_t tokens) const {
    tqh::RotateQuery(stream, AscendType::FP16, input.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     output.get(), tokens, config_.model.num_heads, config_.model.head_size, aiv_num_);
  }

  const TraceConfig config_;
  const int64_t aiv_num_;
  DeterministicRandom rng_;
  DeviceBuffer pi_signs_;
  DeviceBuffer h16_;
  DeviceBuffer rot_tables_;
};

class DecodeScenario final : public Scenario {
 public:
  DecodeScenario(const TraceConfig& config, int64_t aiv_num, aclrtStream stream) : Scenario(config, aiv_num) {
    const int64_t d = config.model.head_size;
    const int64_t hq = config.model.num_heads;
    const int64_t hkv = config.model.num_kv_heads;
    const int64_t batch = config.batch;
    const int64_t context = config.context_tokens();
    const int64_t blocks_per_seq = config.blocks_per_seq();
    const int64_t pool_blocks = config.pool_blocks();
    const int64_t pattern = std::min<int64_t>(context, kPatternTokens);

    const std::vector<Half> key_pattern = FloatToHalf(rng_.NormalHalfExact(Elements(pattern, hkv, d), 0.0f, 1.0f));
    const std::vector<Half> value_pattern =
        FloatToHalf(rng_.NormalHalfExact(Elements(pattern, hkv, d), 0.0f, 1.0f));
    key_ctx_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_ctx_, key_pattern);
    TileToDevice(value_ctx_, value_pattern);

    const size_t cache_elements = Elements(pool_blocks * kBlockSize, hkv, d);
    fp16_key_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    fp16_value_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    TileToDevice(fp16_key_cache_, key_pattern);
    TileToDevice(fp16_value_cache_, value_pattern);

    query_ = DeviceBuffer::FromHost(FloatToHalf(rng_.NormalHalfExact(Elements(batch, hq, d), 0.0f, 1.0f)),
                                    kBenchmarkAlignBytes);
    query_rot_ = DeviceBuffer::Empty<float>(Elements(batch, hq, d), kBenchmarkAlignBytes);
    out_tq_ = DeviceBuffer::Empty<Half>(Elements(batch, hq, d), kBenchmarkAlignBytes);
    if (!config.model.folds_output) {
      out_rot_ = DeviceBuffer::Empty<float>(Elements(batch, hq, d), kBenchmarkAlignBytes);
    }
    out_v5_ = DeviceBuffer::Empty<Half>(Elements(batch, hq, d), kBenchmarkAlignBytes);
    lse_v5_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    const std::vector<int32_t> block_table = rng_.Permutation(static_cast<int32_t>(pool_blocks));
    std::vector<int32_t> slots(static_cast<size_t>(context));
    for (int64_t seq = 0; seq < batch; ++seq) {
      for (int64_t pos = 0; pos < config.seq_len; ++pos) {
        const int32_t block = block_table[static_cast<size_t>(seq * blocks_per_seq + pos / kBlockSize)];
        slots[static_cast<size_t>(seq * config.seq_len + pos)] =
            block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(pos % kBlockSize);
      }
    }
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);
    slots_ = DeviceBuffer::FromHost(slots, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(batch), static_cast<int32_t>(config.seq_len)), kBenchmarkAlignBytes);

    size_t workspace_floats = 0;
    if (config.path == tqa::PathMode::kCube) {
      key_cache_ = DeviceBuffer::Empty<int8_t>(
          tqh::ModePackedCacheBytes(kCubeMode, pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::ModeTables(kCubeMode, d, 1, 0), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(
          tqh::ModeTables(kCubeMode, d, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);
      cube_grid_ = tqh::PlanFusedDecode(batch, hq, hkv, d, blocks_per_seq, kBlockSize, aiv_num);
      num_splits_ = cube_grid_.num_splits;
      workspace_floats = cube_grid_.workspace_floats;
    } else {
      key_cache_ =
          DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, tqh::kTileRows), kBenchmarkAlignBytes);
      aiv_grid_ = tqh::PlanPagedAttention(batch, hq, d, blocks_per_seq, kBlockSize, aiv_num);
      num_splits_ = aiv_grid_.num_splits;
      workspace_floats = aiv_grid_.workspace_floats;
    }
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ =
        DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(pool_blocks, kBlockSize, hkv), kBenchmarkAlignBytes);
    workspace_ = DeviceBuffer::Empty<float>(workspace_floats, kBenchmarkAlignBytes);

    query_tensor_.reset(new AclnnTensor({batch, hq, d}, ACL_FLOAT16, query_.get()));
    out_v5_tensor_.reset(new AclnnTensor({batch, hq, d}, ACL_FLOAT16, out_v5_.get()));
    lse_v5_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_v5_.get()));
    key_view_.reset(new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16,
                                    fp16_key_cache_.get()));
    value_view_.reset(new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16,
                                      fp16_value_cache_.get()));
    block_table_tensor_.reset(new AclnnTensor({batch, blocks_per_seq}, ACL_INT32, block_tables_.get()));
    key_list_.reset(new AclnnTensorList({key_view_->get()}));
    value_list_.reset(new AclnnTensorList({value_view_->get()}));
    std::vector<int64_t> cumulative_q(static_cast<size_t>(batch));
    for (int64_t seq = 0; seq < batch; ++seq) {
      cumulative_q[static_cast<size_t>(seq)] = seq + 1;
    }
    seq_q_.reset(new AclnnIntArray(cumulative_q));
    seq_kv_.reset(new AclnnIntArray(std::vector<int64_t>(static_cast<size_t>(batch), config.seq_len)));

    FillCache(stream);
  }

  std::string Route() const override {
    std::ostringstream text;
    if (config_.path == tqa::PathMode::kCube) {
      text << "Cube: rotate_q -> FusedDecode (one launch)";
    } else {
      text << "AIV: rotate_q -> PagedAttention fused (one launch)";
    }
    text << (config_.model.folds_output ? "" : " -> rotate_o") << ", splits " << num_splits_ << ", blocks/seq "
         << config_.blocks_per_seq();
    return text.str();
  }

  void TurboQuantLeg(const TraceMarkers& markers, aclrtStream stream) override {
    {
      const TraceRange range(markers, "TQ_rotate_q", stream);
      EnqueueRotation(stream, query_, query_rot_, config_.batch);
    }
    {
      // One marker name on both paths, so the timeline compares like with like against V5_Native.
      const TraceRange range(markers, "TQ_FusedDecode", stream);
      if (config_.path == tqa::PathMode::kCube) {
        EnqueueCubeFused(stream);
      } else {
        EnqueueAivPagedAttention(stream);
      }
    }
    if (!config_.model.folds_output) {
      const TraceRange range(markers, "TQ_rotate_o", stream);
      EnqueueRotation(stream, out_tq_, out_rot_, config_.batch);
    }
    Synchronize(markers, stream, "TQ_sync");
  }

  void NativeLeg(const TraceMarkers& markers, aclrtStream stream) override {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    NativeLaunch launch;
    {
      const TraceRange range(markers, "V5_GetWorkspaceSize", nullptr);
      PlanNative<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), &launch, query_tensor_->get(), key_list_->get(), value_list_->get(), nullptr, nullptr,
          seq_q_->get(), seq_kv_->get(), nullptr,
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, block_table_tensor_->get(),
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr,
          hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault,
          kBlockSize, 0, false, 0, 0, s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault,
          out_v5_tensor_->get(), lse_v5_tensor_->get());
    }
    {
      const TraceRange range(markers, "V5_Launch", stream);
      LaunchNative(FiaV5(), launch, stream);
    }
    Synchronize(markers, stream, "V5_sync");
  }

  double TurboQuantMagnitude() const override {
    return config_.model.folds_output ? HalfMagnitude(out_tq_) : FloatMagnitude(out_rot_);
  }
  double NativeMagnitude() const override { return HalfMagnitude(out_v5_); }

 private:
  void FillCache(aclrtStream stream) const {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(config_.context_tokens(), aiv_num_);
    const uint32_t tokens = static_cast<uint32_t>(config_.context_tokens());
    const uint32_t kv_heads = static_cast<uint32_t>(config_.model.num_kv_heads);
    const uint32_t head_size = static_cast<uint32_t>(config_.model.head_size);
    if (config_.path == tqa::PathMode::kCube) {
      turboquant_mm_reshape_and_cache_impl(static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, grid.block_dim,
                                           key_ctx_.get(), value_ctx_.get(), key_cache_.get(), value_cache_.get(),
                                           scale_plane_.get(), slots_.get(), pi_signs_.get(), rot_tables_.get(),
                                           write_tables_.get(), tokens, kv_heads, head_size,
                                           static_cast<uint32_t>(kBlockSize), grid.tokens_per_core,
                                           config_.attention_scale());
    } else {
      turboquant_reshape_and_cache_impl(AscendType::FP16, stream, grid.block_dim, key_ctx_.get(), value_ctx_.get(),
                                        key_cache_.get(), value_cache_.get(), scale_plane_.get(), slots_.get(),
                                        pi_signs_.get(), write_tables_.get(), tokens, kv_heads, head_size,
                                        static_cast<uint32_t>(kBlockSize), grid.tokens_per_core,
                                        config_.attention_scale());
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  void EnqueueCubeFused(aclrtStream stream) const {
    turboquant_mm_fused_decode_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        static_cast<uint32_t>(tqh::kFusedContextLimit), config_.attention_scale(), config_.attention_scale());
  }

  void EnqueueAivPagedAttention(aclrtStream stream) const {
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, aiv_grid_.block_dim, query_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_tq_.get(), static_cast<uint32_t>(config_.batch), static_cast<uint32_t>(config_.model.num_heads),
        static_cast<uint32_t>(config_.model.num_kv_heads), static_cast<uint32_t>(config_.model.head_size),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(config_.blocks_per_seq()),
        static_cast<uint32_t>(aiv_grid_.num_splits), aiv_grid_.split_tasks_per_core, aiv_grid_.reduce_tasks_per_core,
        static_cast<uint32_t>(tqh::kFusedContextLimit), config_.attention_scale(), config_.attention_scale());
  }

  DeviceBuffer key_ctx_, value_ctx_, fp16_key_cache_, fp16_value_cache_;
  DeviceBuffer query_, query_rot_, out_tq_, out_rot_, out_v5_, lse_v5_;
  DeviceBuffer block_tables_, slots_, context_lens_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, write_tables_, decode_tables_, workspace_;
  tqh::PagedAttentionGrid aiv_grid_;
  tqh::FusedDecodeGrid cube_grid_;
  int64_t num_splits_ = 1;

  std::unique_ptr<AclnnTensor> query_tensor_, out_v5_tensor_, lse_v5_tensor_;
  std::unique_ptr<AclnnTensor> key_view_, value_view_, block_table_tensor_;
  std::unique_ptr<AclnnTensorList> key_list_, value_list_;
  std::unique_ptr<AclnnIntArray> seq_q_, seq_kv_;
};

class PrefillScenario final : public Scenario {
 public:
  PrefillScenario(const TraceConfig& config, int64_t aiv_num) : Scenario(config, aiv_num) {
    const int64_t d = config.model.head_size;
    const int64_t hq = config.model.num_heads;
    const int64_t hkv = config.model.num_kv_heads;
    const int64_t batch = config.batch;
    const int64_t context = config.context_tokens();
    const int64_t chunk = config.chunk_tokens();
    const int64_t pattern = std::min<int64_t>(context, kPatternTokens);

    const std::vector<float> key_pattern = rng_.NormalHalfExact(Elements(pattern, hkv, d), 0.0f, 1.0f);
    const std::vector<float> value_pattern = rng_.NormalHalfExact(Elements(pattern, hkv, d), 0.0f, 1.0f);
    key_ctx_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    key_ctx_rot_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_rot_ = DeviceBuffer::Empty<Half>(Elements(context, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_ctx_, FloatToHalf(key_pattern));
    TileToDevice(value_ctx_, FloatToHalf(value_pattern));
    TileToDevice(key_ctx_rot_, FloatToHalf(Rotated(key_pattern, d)));
    TileToDevice(value_ctx_rot_, FloatToHalf(Rotated(value_pattern, d)));

    const int64_t query_pattern = std::min<int64_t>(chunk, kPatternTokens);
    const std::vector<float> query_host = rng_.NormalHalfExact(Elements(query_pattern, hq, d), 0.0f, 1.0f);
    query_ = DeviceBuffer::Empty<Half>(Elements(chunk, hq, d), kBenchmarkAlignBytes);
    query_rot_half_ = DeviceBuffer::Empty<Half>(Elements(chunk, hq, d), kBenchmarkAlignBytes);
    TileToDevice(query_, FloatToHalf(query_host));
    TileToDevice(query_rot_half_, FloatToHalf(Rotated(query_host, d)));
    query_rot_ = DeviceBuffer::Empty<float>(Elements(chunk, hq, d), kBenchmarkAlignBytes);

    out_tq_ = DeviceBuffer::Empty<Half>(Elements(chunk, hq, d), kBenchmarkAlignBytes);
    if (!config.model.folds_output) {
      out_rot_ = DeviceBuffer::Empty<float>(Elements(chunk, hq, d), kBenchmarkAlignBytes);
    }
    out_v5_ = DeviceBuffer::Empty<Half>(Elements(chunk, hq, d), kBenchmarkAlignBytes);
    lse_tq_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);
    lse_v5_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    std::vector<int8_t> mask(static_cast<size_t>(kCausalMaskSide * kCausalMaskSide), 0);
    for (int64_t row = 0; row < kCausalMaskSide; ++row) {
      for (int64_t col = row + 1; col < kCausalMaskSide; ++col) {
        mask[static_cast<size_t>(row * kCausalMaskSide + col)] = 1;
      }
    }
    mask_ = DeviceBuffer::FromHost(mask, kBenchmarkAlignBytes);

    query_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_.get()));
    query_rot_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_rot_half_.get()));
    key_tensor_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_.get()));
    value_tensor_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_.get()));
    key_rot_tensor_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_rot_.get()));
    value_rot_tensor_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_rot_.get()));
    mask_tensor_.reset(new AclnnTensor({kCausalMaskSide, kCausalMaskSide}, ACL_INT8, mask_.get()));
    out_tq_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_tq_.get()));
    out_v5_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_v5_.get()));
    lse_tq_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_tq_.get()));
    lse_v5_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_v5_.get()));
    key_list_.reset(new AclnnTensorList({key_tensor_->get()}));
    value_list_.reset(new AclnnTensorList({value_tensor_->get()}));
    key_rot_list_.reset(new AclnnTensorList({key_rot_tensor_->get()}));
    value_rot_list_.reset(new AclnnTensorList({value_rot_tensor_->get()}));

    std::vector<int64_t> cumulative_q(static_cast<size_t>(batch));
    std::vector<int64_t> cumulative_kv(static_cast<size_t>(batch));
    for (int64_t seq = 0; seq < batch; ++seq) {
      cumulative_q[static_cast<size_t>(seq)] = (seq + 1) * config.chunk;
      cumulative_kv[static_cast<size_t>(seq)] = (seq + 1) * config.seq_len;
    }
    seq_q_.reset(new AclnnIntArray(cumulative_q));
    seq_kv_.reset(new AclnnIntArray(cumulative_kv));
  }

  std::string Route() const override {
    std::ostringstream text;
    text << "chunk C=" << config_.chunk << ": rotate_q -> FIA V5 over the rotated basis"
         << (config_.model.folds_output ? "" : " -> rotate_o");
    return text.str();
  }

  void TurboQuantLeg(const TraceMarkers& markers, aclrtStream stream) override {
    {
      const TraceRange range(markers, "TQ_rotate_q", stream);
      EnqueueRotation(stream, query_, query_rot_, config_.chunk_tokens());
    }
    NativeLaunch launch;
    {
      const TraceRange range(markers, "TQ_FIA_V5_GetWorkspaceSize", nullptr);
      PlanPrefill(&launch, query_rot_tensor_->get(), key_rot_list_->get(), value_rot_list_->get(),
                  out_tq_tensor_->get(), lse_tq_tensor_->get());
    }
    {
      const TraceRange range(markers, "TQ_FIA_V5_Launch", stream);
      LaunchNative(FiaV5(), launch, stream);
    }
    if (!config_.model.folds_output) {
      const TraceRange range(markers, "TQ_rotate_o", stream);
      EnqueueRotation(stream, out_tq_, out_rot_, config_.chunk_tokens());
    }
    Synchronize(markers, stream, "TQ_sync");
  }

  void NativeLeg(const TraceMarkers& markers, aclrtStream stream) override {
    NativeLaunch launch;
    {
      const TraceRange range(markers, "V5_GetWorkspaceSize", nullptr);
      PlanPrefill(&launch, query_tensor_->get(), key_list_->get(), value_list_->get(), out_v5_tensor_->get(),
                  lse_v5_tensor_->get());
    }
    {
      const TraceRange range(markers, "V5_Launch", stream);
      LaunchNative(FiaV5(), launch, stream);
    }
    Synchronize(markers, stream, "V5_sync");
  }

  double TurboQuantMagnitude() const override {
    return config_.model.folds_output ? HalfMagnitude(out_tq_) : FloatMagnitude(out_rot_);
  }
  double NativeMagnitude() const override { return HalfMagnitude(out_v5_); }

 private:
  void PlanPrefill(NativeLaunch* launch, const aclTensor* query, const aclTensorList* key,
                   const aclTensorList* value, const aclTensor* out, const aclTensor* lse) const {
    PlanNative<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
        FiaV5(), launch, query, key, value, nullptr, mask_tensor_->get(), seq_q_->get(), seq_kv_->get(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        config_.model.num_heads, static_cast<double>(config_.attention_scale()), s950::kFiaUnboundedTokens,
        s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), config_.model.num_kv_heads,
        kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault, kFiaNoPaging, 0, false, 0, 0,
        s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, out, lse);
  }

  DeviceBuffer key_ctx_, value_ctx_, key_ctx_rot_, value_ctx_rot_;
  DeviceBuffer query_, query_rot_half_, query_rot_;
  DeviceBuffer out_tq_, out_rot_, out_v5_, lse_tq_, lse_v5_;
  DeviceBuffer mask_;

  std::unique_ptr<AclnnTensor> query_tensor_, query_rot_tensor_;
  std::unique_ptr<AclnnTensor> key_tensor_, value_tensor_, key_rot_tensor_, value_rot_tensor_;
  std::unique_ptr<AclnnTensor> mask_tensor_, out_tq_tensor_, out_v5_tensor_, lse_tq_tensor_, lse_v5_tensor_;
  std::unique_ptr<AclnnTensorList> key_list_, value_list_, key_rot_list_, value_rot_list_;
  std::unique_ptr<AclnnIntArray> seq_q_, seq_kv_;
};

std::unique_ptr<Scenario> MakeScenario(const TraceConfig& config, int64_t aiv_num, aclrtStream stream) {
  if (config.phase == Phase::kDecode) {
    return std::unique_ptr<Scenario>(new DecodeScenario(config, aiv_num, stream));
  }
  return std::unique_ptr<Scenario>(new PrefillScenario(config, aiv_num));
}

void RunLeg(const TraceMarkers& markers, aclrtStream stream, const char* leg, const std::function<void()>& body) {
  markers.Mark(std::string(leg) + kStartSuffix, stream);
  try {
    std::unique_ptr<TraceRange> range;
    if (markers.has_ranges()) {
      range.reset(new TraceRange(markers, leg, stream));
    }
    body();
  } catch (...) {
    markers.Mark(std::string(leg) + kAbortSuffix, stream);
    throw;
  }
  markers.Mark(std::string(leg) + kEndSuffix, stream);
}

std::string ShapeDetail(const TraceConfig& config) {
  std::ostringstream text;
  text << "D=" << config.model.head_size << " H_Q=" << config.model.num_heads << " H_KV=" << config.model.num_kv_heads
       << (config.model.folds_output ? " W_o folded" : " W_o unfolded");
  return text.str();
}

bool RunShape(const TraceConfig& config, size_t index, size_t count, const TraceMarkers& markers, aclrtStream stream,
              int64_t aiv_num) {
  const std::string label = config.label();
  std::printf("%s [%zu/%zu] %s | %s", kTag, index + 1, count, label.c_str(), ShapeDetail(config).c_str());
  std::fflush(stdout);
  bool clean = false;
  try {
    const TraceRange shape_range(markers, label, stream);
    std::unique_ptr<Scenario> scenario;
    {
      const TraceRange range(markers, "setup", stream);
      scenario = MakeScenario(config, aiv_num, stream);
    }
    std::printf(" | %s | dispatching", scenario->Route().c_str());
    std::fflush(stdout);
    RunLeg(markers, stream, kLegTurboQuant, [&] { scenario->TurboQuantLeg(markers, stream); });
    std::this_thread::sleep_for(kTimelineGap);
    RunLeg(markers, stream, kLegNative, [&] { scenario->NativeLeg(markers, stream); });
    double turboquant = 0.0;
    double native = 0.0;
    {
      const TraceRange range(markers, "readback", stream);
      turboquant = scenario->TurboQuantMagnitude();
      native = scenario->NativeMagnitude();
    }
    clean = std::isfinite(turboquant) && turboquant > 0.0 && std::isfinite(native) && native > 0.0;
    std::printf(" -> done, sum|out| TQ %.4e V5 %.4e%s\n", turboquant, native,
                clean ? "" : " -- EMPTY OR NON-FINITE OUTPUT");
    const TraceRange range(markers, "teardown", stream);
    scenario.reset();
  } catch (const std::exception& error) {
    std::printf(" -> FAILED: %s\n", error.what());
    clean = false;
  }
  std::fflush(stdout);
  std::this_thread::sleep_for(kTimelineGap);
  return clean;
}

std::string JoinModels(const std::vector<tqa::ModelSpec>& models) {
  std::ostringstream text;
  for (size_t index = 0; index < models.size(); ++index) {
    text << (index == 0 ? "" : ",") << models[index].label;
  }
  return text.str();
}

std::string JoinCounts(const std::vector<int64_t>& values) {
  std::ostringstream text;
  for (size_t index = 0; index < values.size(); ++index) {
    text << (index == 0 ? "" : ",") << values[index];
  }
  return text.str();
}

std::string SelfExecutable(const char* argv0) {
  char path[kPathCapacity];
  const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (length <= 0) {
    return argv0;
  }
  path[length] = '\0';
  return path;
}

std::string MsprofCommand(const Options& options, const std::string& executable) {
  std::ostringstream command;
  command << "msprof --output=" << kMsprofOutput << " --msproftx=on --task-time=l1 --runtime-api=on " << executable
          << ' ' << kModelsFlag << JoinModels(options.models) << ' ' << kContextsFlag << JoinCounts(options.contexts)
          << ' ' << kBatchesFlag << JoinCounts(options.batches) << ' ' << kModeFlag << options.mode;
  return command.str();
}

void PrintBanner(const Options& options, const std::vector<TraceConfig>& shapes, const std::string& command) {
  std::printf("%s TurboQuant attention chain vs aclnnFusedInferAttentionScoreV5, one msprof timeline\n", kTag);
  std::printf("%s   one launch per leg, no warmup, no timing loop, %lld ms gap after each leg\n", kTag,
              static_cast<long long>(kTimelineGap.count()));
  std::printf("%s   %zu shapes: models %s, contexts %s, batches %s, mode %s\n", kTag, shapes.size(),
              JoinModels(options.models).c_str(), JoinCounts(options.contexts).c_str(),
              JoinCounts(options.batches).c_str(), options.mode.c_str());
  std::printf("%s   the first shape also carries each kernel's and the operator's first-launch cost\n", kTag);
  if (std::find(options.phases.begin(), options.phases.end(), Phase::kPrefill) != options.phases.end()) {
    std::printf("%s   prefill: FIA reads a host-rotated fp16 query, because the fp32 -> fp16 narrowing after\n"
                "%s   rotate_q has no header-verified operator; both prefill legs plan FIA V5, so the second\n"
                "%s   may reuse operator state the first created\n",
                kTag, kTag, kTag);
  }
  std::printf("%s collect the trace with:\n%s   %s\n", kTag, kTag, command.c_str());
  std::printf("%s then export it with:\n%s   msprof --export=on --output=%s\n", kTag, kTag, kMsprofOutput);
  std::fflush(stdout);
}

void PrintRuntime(const AscendDevice& device, int64_t aiv_num, bool aiv_queried, const TraceMarkers& markers) {
  std::printf("%s device %d, soc '%s', vector cores %lld%s\n", kTag, device.device_id(), device.soc_name().c_str(),
              static_cast<long long>(aiv_num), aiv_queried ? "" : " (assumed; the runtime declined to answer)");
  const AclnnOp& fia = FiaV5();
  if (fia.available()) {
    std::printf("%s native operator %s from %s\n", kTag, fia.name().c_str(), fia.source().c_str());
  } else {
    std::printf("%s native operator unavailable: %s\n", kTag, fia.unavailable_reason().c_str());
  }
  if (markers.has_ranges()) {
    const uint64_t tool = markers.ToolId();
    const char* tool_note = tool == kMstxToolMsprofId    ? " (msprof)"
                            : tool == kMstxToolInvalidId ? " (none reported; outside msprof --msproftx=on the "
                                                           "markers record nothing)"
                                                         : "";
    std::printf("%s markers: stream-bound mstx ranges and marks from %s, tool id 0x%llx%s\n", kTag,
                markers.source().c_str(), static_cast<unsigned long long>(tool), tool_note);
  } else if (markers.available()) {
    std::printf("%s markers: %s did not resolve; aclprofMarkEx point marks only, ranges become _Start/_End marks\n",
                kTag, kMstxLibrary);
  } else {
    std::printf("%s markers: NONE -- neither %s nor aclprofMarkEx resolved; the timeline will carry no labels\n",
                kTag, kMstxLibrary);
  }
  std::fflush(stdout);
}

int Run(int argc, char** argv) {
  Options options;
  int exit_code = kExitOk;
  if (!ParseCommandLine(argc, argv, &options, &exit_code)) {
    return exit_code;
  }
  const std::vector<TraceConfig> shapes = BuildShapes(options);
  PrintBanner(options, shapes, MsprofCommand(options, SelfExecutable(argv[0])));

  if (IsRunningOnSimulator()) {
    std::printf("%s CAModel loaded (%s); this harness traces silicon, skipping\n", kTag,
                SimulatorEvidence().c_str());
    return bench::kBenchmarkSkipExitCode;
  }
  std::unique_ptr<AscendDevice> device;
  try {
    device.reset(new AscendDevice());
  } catch (const std::exception& error) {
    std::printf("%s no usable Ascend device, skipping:\n%s\n", kTag, error.what());
    return bench::kBenchmarkSkipExitCode;
  }
  if (!s950::IsAscend950PrSocName(device->soc_name())) {
    std::printf("%s attached device reports soc '%s'; this harness targets an Ascend 950PR, skipping\n", kTag,
                device->soc_name().c_str());
    return bench::kBenchmarkSkipExitCode;
  }

  size_t failures = 0;
  {
    const HugeMemThreadScope huge_mem;
    bool aiv_queried = false;
    const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
    const TraceMarkers markers;
    PrintRuntime(*device, aiv_num, aiv_queried, markers);
    const aclrtStream stream = device->stream();
    for (size_t index = 0; index < shapes.size(); ++index) {
      if (!RunShape(shapes[index], index, shapes.size(), markers, stream, aiv_num)) {
        ++failures;
      }
    }
  }
  std::printf("%s %zu of %zu shapes dispatched both legs with non-empty output\n", kTag, shapes.size() - failures,
              shapes.size());
  return failures == 0 ? kExitOk : kExitFailure;
}

}
}
}
}

int main(int argc, char** argv) { return ::vllm_ascend::test::trace::Run(argc, argv); }
