// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#if __has_include(<aclnnop/aclnn_grouped_matmul_v5.h>)
  #include <aclnnop/aclnn_grouped_matmul_v5.h>
#endif

#include "aclnn_ops.hpp"
#include "device_tensor.hpp"
#include "dsv4_device_case.hpp"
#include "test_harness.hpp"

namespace vllm_ascend::test {
namespace {
namespace d = dsv4;
constexpr int kWarmup = 50;
constexpr int kIterations = 200;
constexpr size_t kOperations = 7;
// Signature matches the CANN 9.2 official aclnn_grouped_matmul_v5.h.
// Some 9.1 toolkit bundles ship libopapi without individual operator headers.
#if __has_include(<aclnnop/aclnn_grouped_matmul_v5.h>)
using GmmV5Fn = decltype(&aclnnGroupedMatmulV5GetWorkspaceSize);
#else
using GmmV5Fn = int (*)(const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensorList*,
                        const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensorList*,
                        const aclTensor*, const aclTensorList*, const aclTensorList*, const aclTensorList*, int64_t,
                        int64_t, int64_t, int64_t, aclIntArray*, aclTensorList*, aclTensorList*, aclTensorList*,
                        uint64_t*, aclOpExecutor**);
#endif
using CatFn = int (*)(const aclTensorList*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
using ClampFn = int (*)(const aclTensor*, const aclScalar*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
using CastFn = int (*)(const aclTensor*, aclDataType, aclTensor*, uint64_t*, aclOpExecutor**);

class Events {
 public:
  Events() {
    ACL_CHECK(aclrtCreateEvent(&start_));
    try {
      ACL_CHECK(aclrtCreateEvent(&stop_));
    } catch (...) {
      ACL_CHECK_NOTHROW(aclrtDestroyEvent(start_));
      throw;
    }
  }
  ~Events() {
    ACL_CHECK_NOTHROW(aclrtDestroyEvent(start_));
    ACL_CHECK_NOTHROW(aclrtDestroyEvent(stop_));
  }
  template <typename Launch>
  double Measure(aclrtStream stream, Launch launch) {
    ACL_CHECK(aclrtRecordEvent(start_, stream));
    launch();
    ACL_CHECK(aclrtRecordEvent(stop_, stream));
    ACL_CHECK(aclrtSynchronizeEvent(stop_));
    float milliseconds = 0;
    ACL_CHECK(aclrtEventElapsedTime(&milliseconds, start_, stop_));
    if (!(milliseconds > 0) || !std::isfinite(milliseconds)) throw std::runtime_error("invalid event time");
    return milliseconds * 1000.0;
  }

 private:
  aclrtEvent start_ = nullptr, stop_ = nullptr;
};

std::vector<float> DequantTransposed(const std::vector<uint8_t>& w, const std::vector<uint8_t>& s, int64_t rows,
                                     int64_t cols) {
  std::vector<float> values(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c)
      values[static_cast<size_t>(c * rows + r)] =
          d::UnpackFp4(w.data() + r * cols / 2, c) *
          d::E8m0ToScale(s[static_cast<size_t>(r * cols / d::kFp4Block + c / d::kFp4Block)]);
  return values;
}

class Bf16Tensor {
 public:
  Bf16Tensor() = default;
  Bf16Tensor(const std::vector<int64_t>& dims, const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t j = 0; j < bits.size(); ++j) bits[j] = d::FloatToBf16Bits(values[j]);
    if (bits.size() != ElementCount(dims)) throw std::invalid_argument("BF16 tensor size mismatch");
    buffer_ = DeviceBuffer::FromHost(bits);
    tensor_ = AclnnTensor(dims, ACL_BF16, buffer_.get());
  }
  aclTensor* get() const { return tensor_.get(); }

 private:
  DeviceBuffer buffer_;
  AclnnTensor tensor_;
};

// Official grouped-matmul V5 baseline: pre-dequantized BF16 operands, FP32
// projection outputs, FP32 clamp/SwiGLU, BF16 activation before the down GEMM.
// Runtime availability and BF16->FP32 output tuples must pass planning/parity;
// no precision-changing fallback is used if the installed runtime rejects them.
class AclnnBaseline {
 public:
  explicit AclnnBaseline(const d::Problem& p)
      : h_(p.hidden),
        i_(p.inter),
        matmul_("aclnnGroupedMatmulV5"),
        clamp_("aclnnClamp"),
        swiglu_(ops::kSwiGlu),
        cast_("aclnnCast"),
        cat_("aclnnCat"),
        lo_(-d::kSwigluLimit),
        hi_(d::kSwigluLimit) {
    for (const auto* op : {&matmul_, &clamp_, &swiglu_, &cast_, &cat_})
      if (!op->available()) throw std::runtime_error(op->unavailable_reason());
    std::vector<float> x(p.x.size());
    for (size_t j = 0; j < x.size(); ++j) x[j] = d::Bf16BitsToFloat(p.x[j]);
    x_ = Bf16Tensor({1, h_}, x);
    w1_ = Bf16Tensor({h_, i_}, DequantTransposed(p.w1, p.w1_scale, i_, h_));
    w3_ = Bf16Tensor({h_, i_}, DequantTransposed(p.w3, p.w3_scale, i_, h_));
    w2_ = Bf16Tensor({i_, h_}, DequantTransposed(p.w2, p.w2_scale, h_, i_));
    gate_ = DeviceTensor::FloatEmpty({1, i_});
    combined_ = DeviceTensor::FloatEmpty({1, 2 * i_});
    clamped_ = DeviceTensor::FloatEmpty({1, i_});
    up_ = DeviceTensor::FloatEmpty({1, i_});
    activated_ = DeviceTensor::FloatEmpty({1, i_});
    bf16_.Allocate(static_cast<size_t>(2 * i_));
    activated_bf16_ = AclnnTensor({1, i_}, ACL_BF16, bf16_.get());
    down_ = DeviceTensor::FloatEmpty({1, h_});
    inputs_[0] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{x_.get()});
    inputs_[1] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{x_.get()});
    inputs_[2] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{activated_bf16_.get()});
    weights_[0] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{w1_.get()});
    weights_[1] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{w3_.get()});
    weights_[2] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{w2_.get()});
    outputs_[0] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{gate_.get()});
    outputs_[1] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{up_.get()});
    outputs_[2] = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{down_.get()});
    cat_inputs_ = std::make_unique<AclnnTensorList>(std::vector<const aclTensor*>{clamped_.get(), up_.get()});
  }
  ~AclnnBaseline() { ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream())); }
  aclrtStream stream() const { return stream_.get(); }
  void Prepare() {
    PlanGemm(0, 0);
    PlanGemm(1, 1);
    Plan(2, clamp_, reinterpret_cast<ClampFn>(clamp_.get_workspace_size_fn()), gate_.get(), lo_.get(), hi_.get(),
         clamped_.get());
    Plan(3, cat_, reinterpret_cast<CatFn>(cat_.get_workspace_size_fn()), cat_inputs_->get(), int64_t{-1},
         combined_.get());
    Plan(4, swiglu_, reinterpret_cast<ops::SwiGluWorkspaceFn>(swiglu_.get_workspace_size_fn()), combined_.get(),
         int64_t{-1}, activated_.get());
    Plan(5, cast_, reinterpret_cast<CastFn>(cast_.get_workspace_size_fn()), activated_.get(), ACL_BF16,
         activated_bf16_.get());
    PlanGemm(6, 2);
    const size_t required = static_cast<size_t>(*std::max_element(sizes_.begin(), sizes_.end()));
    if (workspace_.size_bytes() < required) workspace_.Allocate(required);
  }
  void Enqueue() {
    // Executors are one-shot: Prepare is called afresh before each sample,
    // outside its event interval. Workspace reuse is ordered by this stream.
    for (size_t j = 0; j < kOperations; ++j) {
      auto fn = reinterpret_cast<AclnnLaunchFn>(operators_[j]->launch_fn());
      ACL_CHECK(fn(workspace_.get(), sizes_[j], executors_[j], stream()));
    }
  }
  void Check(const d::DeviceOutputs& want) {
    ACL_CHECK(aclrtSynchronizeStream(stream()));
    const auto gate = gate_.ToFloat();
    const auto combined = combined_.ToFloat();
    const auto down = down_.ToFloat();
    const auto active = bf16_.ToHost<uint16_t>();
    d::DeviceOutputs got = {std::vector<uint16_t>(gate.size()), std::vector<uint16_t>(gate.size()), active,
                            std::vector<uint16_t>(down.size())};
    for (size_t j = 0; j < gate.size(); ++j) {
      got[0][j] = d::FloatToBf16Bits(gate[j]);
      got[1][j] = d::FloatToBf16Bits(combined[static_cast<size_t>(i_) + j]);
    }
    for (size_t j = 0; j < down.size(); ++j) got[3][j] = d::FloatToBf16Bits(down[j]);
    for (size_t t = 0; t < got.size(); ++t)
      for (size_t j = 0; j < got[t].size(); ++j)
        if (!std::isfinite(d::Bf16BitsToFloat(got[t][j])) ||
            d::Bf16UlpDistance(got[t][j], want[t][j]) > d::kDeviceMaxUlp)
          throw std::runtime_error("ACLNN baseline failed golden parity; refusing timings");
  }
  size_t AllocatedBytes() const {
    // Owned tensor payload + alignment/allocator slack + shared peak workspace.
    size_t bytes = static_cast<size_t>(6 * h_ * i_ + 6 * h_ + 26 * i_);
    bytes += 11 * (2 * kDeviceAlignBytes);
    if (!workspace_.empty()) bytes += workspace_.capacity_bytes() + workspace_.alignment();
    return bytes;
  }

 private:
  void PlanGemm(size_t op_index, size_t leg) {
    auto fn = reinterpret_cast<GmmV5Fn>(matmul_.get_workspace_size_fn());
    // A single expert per tensor list: splitItem=0, groupType=-1 (no split),
    // groupListType=0, actType=0. Activation is a separate clamped SwiGLU.
    Plan(op_index, matmul_, fn, inputs_[leg]->get(), weights_[leg]->get(), nullptr, nullptr, nullptr, nullptr, nullptr,
         nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{0}, int64_t{-1}, int64_t{0}, int64_t{0}, nullptr,
         outputs_[leg]->get(), nullptr, nullptr);
  }
  template <typename Fn, typename... Args>
  void Plan(size_t j, const AclnnOp& op, Fn fn, Args... args) {
    operators_[j] = &op;
    ACL_CHECK(fn(args..., &sizes_[j], &executors_[j]));
  }
  d::ExpertStream stream_;
  int64_t h_, i_;
  AclnnOp matmul_, clamp_, swiglu_, cast_, cat_;
  AclnnScalar lo_, hi_;
  Bf16Tensor x_, w1_, w2_, w3_;
  DeviceTensor gate_, up_, clamped_, combined_, activated_, down_;
  DeviceBuffer bf16_, workspace_;
  AclnnTensor activated_bf16_;
  std::array<std::unique_ptr<AclnnTensorList>, 3> inputs_, weights_, outputs_;
  std::unique_ptr<AclnnTensorList> cat_inputs_;
  std::array<const AclnnOp*, kOperations> operators_{};
  std::array<uint64_t, kOperations> sizes_{};
  std::array<aclOpExecutor*, kOperations> executors_{};
};

template <typename Prepare, typename Launch>
double Time(aclrtStream stream, Prepare prepare, Launch launch) {
  for (int j = 0; j < kWarmup; ++j) {
    prepare();
    launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
  Events events;
  std::vector<double> samples;
  for (int j = 0; j < kIterations; ++j) {
    prepare();
    samples.push_back(events.Measure(stream, launch));
  }
  std::sort(samples.begin(), samples.end());
  return (samples[kIterations / 2 - 1] + samples[kIterations / 2]) / 2;
}

void Run() {
  for (int64_t hidden : {int64_t{4096}, d::kProductionHidden}) {
    const auto p = d::MakeDeviceProblem(hidden, d::kProductionInter, 0xD540);
    const auto want = d::Golden(p);
    d::DeviceExpert custom(p);
    custom.Enqueue();
    const auto got = custom.Read();
    for (size_t t = 0; t < got.size(); ++t)
      for (size_t j = 0; j < got[t].size(); ++j)
        if (!std::isfinite(d::Bf16BitsToFloat(got[t][j])) ||
            d::Bf16UlpDistance(got[t][j], want[t][j]) > d::kDeviceMaxUlp)
          throw std::runtime_error("custom failed parity; refusing timings");
    AclnnBaseline baseline(p);
    baseline.Prepare();
    baseline.Enqueue();
    baseline.Check(want);
    const double custom_us = Time(custom.stream(), [] {}, [&] { custom.Enqueue(); });
    const double baseline_us = Time(baseline.stream(), [&] { baseline.Prepare(); }, [&] { baseline.Enqueue(); });
    if (custom.Read() != got) throw std::runtime_error("custom changed bits during timing");
    baseline.Check(want);
    const double flops = 6.0 * hidden * p.inter;
    const double custom_bytes = 3.0 * hidden * p.inter * (0.5 + 1.0 / d::kFp4Block) + 4.0 * hidden + 6.0 * p.inter;
    const double baseline_bytes = 6.0 * hidden * p.inter;
    std::printf("\nH=%lld I=%lld warmup=%d samples=%d median ACL event time\n", static_cast<long long>(hidden),
                static_cast<long long>(p.inter), kWarmup, kIterations);
    std::printf(
        "+------------------------+-----------+---------+----------+-----------+--------------+\n"
        "| implementation         | us        | speedup | TFLOP/s  | eff GB/s  | owned HBM MiB|\n"
        "+------------------------+-----------+---------+----------+-----------+--------------+\n");
    std::printf("| custom packed FP4      | %9.3f | %7.3f | %8.4f | %9.3f | %12.3f |\n", custom_us,
                baseline_us / custom_us, flops / custom_us / 1e6, custom_bytes / custom_us / 1e3,
                custom.AllocatedBytes() / 1048576.0);
    std::printf("| ACLNN V5 dense BF16    | %9.3f |   1.000 | %8.4f | %9.3f | %12.3f |\n", baseline_us,
                flops / baseline_us / 1e6, baseline_bytes / baseline_us / 1e3, baseline.AllocatedBytes() / 1048576.0);
    std::puts(
        "+------------------------+-----------+---------+----------+-----------+--------------+\n"
        "ACLNN V5 uses pre-dequantized resident BF16 weights; dequantization/upload excluded.\n"
        "Bandwidth is logical minimum traffic, not measured DRAM transactions.\n"
        "Owned HBM excludes CANN internal caches. Peak utilization/saturation needs SKU peaks and msprof.");
  }
}
}  // namespace
}  // namespace vllm_ascend::test

int main() {
  using namespace vllm_ascend::test;
  if (IsRunningOnSimulator()) {
    std::puts("Refusing simulator timings");
    return 77;
  }
  auto& env = AscendTestEnvironment::Instance();
  env.SetUp();
  int status = 0;
  try {
    if (!env.available() || !env.is_950pr()) {
      std::printf("No physical Ascend950PR: %s\n", env.unavailable_reason().c_str());
      status = 77;
    } else {
      if (env.device().device_id() != 0) throw std::runtime_error("benchmark requires npu:0");
      Run();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    status = 1;
  }
  env.TearDown();
  return status;
}
