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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "hadamard_spike.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "cube_hadamard_950pr (Walsh-Hadamard: Cube-factorised against AIV-only)";

namespace {

namespace hs = hadamard_spike;

const int64_t kDefaultDims[] = {64, 128, 256, 512};
const int64_t kDefaultBatches[] = {1, 8, 16, 32};

constexpr double kFloatBytes = 4.0;

std::vector<int64_t> ParseList(const char* env_name, const int64_t* defaults, size_t default_count) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || *value == '\0') {
    return std::vector<int64_t>(defaults, defaults + default_count);
  }
  std::vector<int64_t> out;
  std::istringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    out.push_back(std::strtoll(token.c_str(), nullptr, 10));
  }
  if (out.empty()) {
    return std::vector<int64_t>(defaults, defaults + default_count);
  }
  return out;
}

class Shape {
 public:
  Shape(int64_t dim, int64_t num_vectors)
      : dim_(dim),
        num_vectors_(num_vectors),
        elements_(static_cast<size_t>(dim * num_vectors)),
        vectors_per_chunk_(hs::HadamardVectorsPerChunk(dim, num_vectors)),
        inv_sqrt_dim_(hs::InvSqrtDim(dim)),
        input_(DeviceBuffer::FromHost(hs::SyntheticBatch(dim, num_vectors), kBenchmarkAlignBytes)),
        h16_(DeviceBuffer::FromHost(hs::Hadamard16Half(), kBenchmarkAlignBytes)),
        tables_(DeviceBuffer::FromHost(hs::EarlyStageTables(dim), kBenchmarkAlignBytes)),
        output_(DeviceBuffer::Empty<float>(elements_, kBenchmarkAlignBytes)) {
    ACL_CHECK(aclrtMemset(output_.get(), output_.size_bytes(), 0, output_.size_bytes()));
  }

  void EnqueueHybrid(aclrtStream stream, uint32_t variant) const {
    sim_hadamard_hybrid_impl(stream, input_.get(), h16_.get(), output_.get(), static_cast<uint32_t>(dim_),
                             static_cast<uint32_t>(num_vectors_), static_cast<uint32_t>(vectors_per_chunk_),
                             variant, inv_sqrt_dim_);
  }

  void EnqueueAiv(aclrtStream stream) const {
    sim_hadamard_aiv_impl(stream, input_.get(), tables_.get(), output_.get(), static_cast<uint32_t>(dim_),
                          static_cast<uint32_t>(num_vectors_), inv_sqrt_dim_);
  }

  std::vector<float> Output() const { return output_.ToHost<float>(); }

  int64_t dim() const { return dim_; }
  int64_t num_vectors() const { return num_vectors_; }
  int64_t vectors_per_chunk() const { return vectors_per_chunk_; }

  double bytes_per_iteration() const { return 2.0 * kFloatBytes * static_cast<double>(elements_); }

 private:
  int64_t dim_;
  int64_t num_vectors_;
  size_t elements_;
  int64_t vectors_per_chunk_;
  float inv_sqrt_dim_;
  DeviceBuffer input_;
  DeviceBuffer h16_;
  DeviceBuffer tables_;
  DeviceBuffer output_;
};

}

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> dims =
      ParseList("ASCEND_BENCH_HADAMARD_DIMS", kDefaultDims, std::size(kDefaultDims));
  const std::vector<int64_t> batches =
      ParseList("ASCEND_BENCH_HADAMARD_BATCHES", kDefaultBatches, std::size(kDefaultBatches));

  std::printf("[ascend-bench] Cube-factorised Walsh-Hadamard. Four legs per shape: aiv (all stages on the\n"
              "[ascend-bench]   vector unit), single (lower four as one fp16 Mmad), hilo (two Mmads, the\n"
              "[ascend-bench]   configuration that clears 1e-4) and dualdst (hilo with both subcores on the\n"
              "[ascend-bench]   residual). Read aiv against dualdst; see this file's header for what the\n"
              "[ascend-bench]   GM round trip does to that ratio.\n");
  std::fflush(stdout);

  std::vector<std::unique_ptr<Shape>> shapes;
  shapes.reserve(dims.size() * batches.size());

  for (const int64_t dim : dims) {
    for (const int64_t num_vectors : batches) {
      std::unique_ptr<Shape> owned;
      try {
        owned.reset(new Shape(dim, num_vectors));
      } catch (const std::exception& error) {
        for (const char* leg : {"aiv", "single", "hilo", "dualdst"}) {
          runner.RecordFailure(hs::CaseLabel(dim, num_vectors, leg), error.what());
        }
        continue;
      }
      shapes.push_back(std::move(owned));
      const Shape& shape = *shapes.back();

      std::printf("[ascend-bench] D=%lld V=%lld: chunk=%lld vectors, %lld chunk(s), Cube m=%lld k=16 n=16\n",
                  static_cast<long long>(dim), static_cast<long long>(num_vectors),
                  static_cast<long long>(shape.vectors_per_chunk()),
                  static_cast<long long>(num_vectors / shape.vectors_per_chunk()),
                  static_cast<long long>(shape.vectors_per_chunk() * (dim / hs::kTile)));
      std::fflush(stdout);

      struct Leg {
        const char* label;
        bool hybrid;
        uint32_t variant;
      };
      const Leg legs[] = {
          {"aiv", false, 0u},
          {"single", true, hs::kHybridSingleMmad},
          {"hilo", true, hs::kHybridHiLo},
          {"dualdst", true, hs::kHybridHiLo | hs::kHybridDualDst},
      };

      for (const Leg& leg : legs) {
        const std::string name = hs::CaseLabel(dim, num_vectors, leg.label);
        if (leg.variant == (hs::kHybridHiLo | hs::kHybridDualDst) &&
            !hs::HadamardDualDstApplies(dim, num_vectors)) {
          runner.Skip(name, "chunk holds one vector; the dual-destination Fixpipe does not apply");
          continue;
        }
        try {
          BenchmarkCase bench_case;
          bench_case.name = name;
          bench_case.bytes_per_iteration = shape.bytes_per_iteration();
          if (leg.hybrid) {
            const uint32_t variant = leg.variant;
            bench_case.launch = [&shape, variant](aclrtStream stream) { shape.EnqueueHybrid(stream, variant); };
          } else {
            bench_case.launch = [&shape](aclrtStream stream) { shape.EnqueueAiv(stream); };
          }
          bench_case.checksum = [&shape]() { return ChecksumSum(shape.Output()); };
          runner.Run(bench_case);
        } catch (const std::exception& error) {
          runner.RecordFailure(name, error.what());
        }
      }
    }
  }
}

}
}
}
