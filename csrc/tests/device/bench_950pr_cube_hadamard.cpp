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

/*
 * EXPLORATORY SPIKE, timed: what the Cube factorisation of the Walsh-Hadamard
 * transform is actually worth on silicon.
 *
 * Four legs at every (D, V) in {64, 128, 256, 512} x {1, 8, 16, 32}:
 *
 *   aiv       all log2(D) butterfly stages on the vector unit, one vector at a
 *             time, in the shape TurboQuantCodec runs them. The baseline, and
 *             the only leg here that is a model of something the project
 *             already ships.
 *   single    lower four stages as one fp16 Mmad. Fastest and least accurate;
 *             it is in the sweep so the cost of the hi/lo split can be read off
 *             directly rather than inferred.
 *   hilo      the same with x = hi + lo, two Mmads into one L0C. The
 *             configuration that clears 1e-4 - see
 *             device/test_device_950pr_cube_hadamard.cpp - and therefore the
 *             one whose time matters.
 *   dualdst   hilo with Fixpipe's dualDstCtl = 0b01, so both vector subcores
 *             run the residual instead of one. Skipped at V = 1, where a chunk
 *             is a single vector and the M split would land inside its tile.
 *
 * THE COMPARISON THIS BINARY EXISTS TO MAKE is aiv against dualdst at fixed
 * (D, V): same input, same output, same GM traffic, differing only in where the
 * lower four stages ran. aiv against single isolates the operand grid, and hilo
 * against single prices the accuracy.
 *
 * WHAT THE NUMBERS DO AND DO NOT COVER. Every leg here moves its input in from
 * GM and its result back out, because that is the only shape a standalone
 * kernel can have. In the decode the rotation happens on data already in UB, so
 * the GM traffic is common overhead that flatters both sides and compresses the
 * ratio between them. The hybrid additionally pays a UB -> L1 staging the
 * AIV-only form does not - a cost TurboQuant milestone 2 is separately trying
 * to remove. Read the ratio as a lower bound on the vector-unit saving, not as
 * a decode speedup.
 *
 * CAModel evidence, for orientation only, at D = 256, V = 16: 3.2x fewer vector
 * instructions per subcore and 2.2x fewer ticks for dualdst against aiv, of
 * which 1.44x is the dual-destination Fixpipe alone. A functional simulator
 * cannot produce a time, which is why this file exists.
 *
 * OUTPUT
 *
 *   --csv=<path>   writes the per-case table. Defaults to
 *                  hadamard_benchmark_results.csv in the working directory;
 *                  --csv= (empty) turns it off. ASCEND_BENCH_CSV does the same
 *                  and the flag wins. The `case` column is d<D>_v<V>_<leg>, so
 *                  it splits on '_' into the three sweep axes.
 *
 *   ASCEND_BENCH_HADAMARD_DIMS=256,512      restrict the D sweep
 *   ASCEND_BENCH_HADAMARD_BATCHES=16,32     restrict the V sweep
 *   everything else is the shared ASCEND_BENCH_* set; see common/benchmark.hpp.
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

/*
 * One shape's device-side state, allocated once and reused by every leg at that
 * shape. The timed region must not allocate, so the buffers, the constant
 * images and the launch arguments all exist before the first launch.
 *
 * The hybrid and the AIV leg write the same output buffer. That is deliberate:
 * they compute the same function, so their checksums are directly comparable
 * and a leg that has quietly stopped computing the transform shows up as a
 * checksum that does not match its neighbours'.
 */
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

  // Read and written per launch: the fp32 input in, the fp32 result out. The
  // fp16 staging copy and the Cube operands live in L1 and L0 and never reach
  // HBM, and H_16 is 512 bytes read once per launch - below the noise of a
  // bandwidth figure, so it is not counted.
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

}  // namespace

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

  // Held for the whole suite: a case's launch closure captures its shape by
  // reference and the runner replays it long after this loop has moved on.
  std::vector<std::unique_ptr<Shape>> shapes;
  shapes.reserve(dims.size() * batches.size());

  for (const int64_t dim : dims) {
    for (const int64_t num_vectors : batches) {
      std::unique_ptr<Shape> owned;
      try {
        owned.reset(new Shape(dim, num_vectors));
      } catch (const std::exception& error) {
        // One shape that will not allocate must not cost the rest of the sweep.
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
          // The kernel would silently take the single-destination path, so the
          // row would duplicate hilo under a name that says otherwise.
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
          // The transform is idempotent in the sense that matters here: the
          // same input goes in every launch, so the output must come back
          // bit-identical. A leg whose checksum moves between the warmup and
          // the last timed iteration is racing, not slow.
          bench_case.checksum = [&shape]() { return ChecksumSum(shape.Output()); };
          runner.Run(bench_case);
        } catch (const std::exception& error) {
          runner.RecordFailure(name, error.what());
        }
      }
    }
  }
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
