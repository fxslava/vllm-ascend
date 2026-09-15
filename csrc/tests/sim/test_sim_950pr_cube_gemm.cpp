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
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "test_harness.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;

constexpr int64_t kC0 = 32;

constexpr int64_t kHeadSize = 256;
constexpr int64_t kTileRows = 64;
constexpr int64_t kGroupHeads = 16;

float ExactFp8(int64_t i) { return static_cast<float>((i % 9) - 4) * 0.5f; }

uint8_t Fp8Bits(float v) {
  struct Entry {
    float value;
    uint8_t bits;
  };
  static const Entry kTable[] = {
      {0.0f, 0x00},  {0.5f, 0x30},  {1.0f, 0x38},  {1.5f, 0x3C}, {2.0f, 0x40},
      {-0.5f, 0xB0}, {-1.0f, 0xB8}, {-1.5f, 0xBC}, {-2.0f, 0xC0},
  };
  for (const Entry& e : kTable) {
    if (e.value == v) {
      return e.bits;
    }
  }
  return 0x00;
}

int64_t NzElems(int64_t rows, int64_t cols) { return rows * ((cols + kC0 - 1) / kC0) * kC0; }

std::vector<int8_t> ToNz(const std::vector<float>& nd, int64_t rows, int64_t cols) {
  std::vector<int8_t> out(static_cast<size_t>(NzElems(rows, cols)), 0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(tqh::NzOffset(r, c, rows))] =
          static_cast<int8_t>(Fp8Bits(nd[static_cast<size_t>(r * cols + c)]));
    }
  }
  return out;
}

struct Case {
  const char* name;
  int64_t m;
  int64_t k;
  int64_t n;
  bool b_is_nk;
  uint32_t variant;
};

double RunCase(const Case& c, aclrtStream stream) {
  std::vector<float> a_nd(static_cast<size_t>(c.m * c.k));
  for (size_t i = 0; i < a_nd.size(); ++i) {
    a_nd[i] = ExactFp8(static_cast<int64_t>(i));
  }
  std::vector<float> b_kn(static_cast<size_t>(c.k * c.n));
  for (size_t i = 0; i < b_kn.size(); ++i) {
    b_kn[i] = ExactFp8(static_cast<int64_t>(i) + 3);
  }

  std::vector<float> reference(static_cast<size_t>(c.m * c.n), 0.0f);
  for (int64_t r = 0; r < c.m; ++r) {
    for (int64_t col = 0; col < c.n; ++col) {
      float acc = 0.0f;
      for (int64_t t = 0; t < c.k; ++t) {
        acc += a_nd[static_cast<size_t>(r * c.k + t)] * b_kn[static_cast<size_t>(t * c.n + col)];
      }
      reference[static_cast<size_t>(r * c.n + col)] = acc;
    }
  }

  std::vector<float> b_staged;
  int64_t b_rows = 0;
  int64_t b_cols = 0;
  if (c.b_is_nk) {
    b_rows = c.n;
    b_cols = c.k;
    b_staged.resize(static_cast<size_t>(b_rows * b_cols));
    for (int64_t t = 0; t < c.k; ++t) {
      for (int64_t col = 0; col < c.n; ++col) {
        b_staged[static_cast<size_t>(col * c.k + t)] = b_kn[static_cast<size_t>(t * c.n + col)];
      }
    }
  } else {
    b_rows = c.k;
    b_cols = c.n;
    b_staged = b_kn;
  }

  DeviceBuffer a_dev = DeviceBuffer::FromHost(ToNz(a_nd, c.m, c.k));
  DeviceBuffer b_dev = DeviceBuffer::FromHost(ToNz(b_staged, b_rows, b_cols));
  DeviceBuffer c_dev = DeviceBuffer::Empty<float>(static_cast<size_t>(c.m * c.n));
  ACL_CHECK(aclrtMemset(c_dev.get(), c_dev.size_bytes(), 0, c_dev.size_bytes()));

  turboquant_cube_gemm_probe_impl(
      stream, a_dev.get(), b_dev.get(), c_dev.get(), static_cast<uint32_t>(c.m), static_cast<uint32_t>(c.k),
      static_cast<uint32_t>(c.n), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kTileRows),
      static_cast<uint32_t>(NzElems(c.m, c.k)), static_cast<uint32_t>(NzElems(b_rows, b_cols)),
      static_cast<uint32_t>(c.m * c.n), c.b_is_nk ? 1u : 0u, c.variant);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const std::vector<float> got = c_dev.ToHost<float>();
  double worst = 0.0;
  size_t worst_at = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i] - reference[i]));
    if (err > worst) {
      worst = err;
      worst_at = i;
    }
  }
  std::printf("[ cube-gemm ] %-28s m=%lld k=%lld n=%lld  B staged [%s]  variant %u  max|err| = %g\n", c.name,
              static_cast<long long>(c.m), static_cast<long long>(c.k), static_cast<long long>(c.n),
              c.b_is_nk ? "n,k" : "k,n", c.variant, worst);
  if (worst != 0.0) {
    std::printf("[ cube-gemm ]   first worst element %zu: got %g, want %g\n", worst_at, got[worst_at],
                reference[worst_at]);
  }
  std::fflush(stdout);

  EXPECT_EQ(worst, 0.0) << c.name
                        << " does not reproduce the host product, so TurboQuantCubeMm's transcription of the "
                           "fractal contract is wrong on this CANN";
  return worst;
}

double ProbeCase(const Case& c, aclrtStream stream) {
  std::vector<float> a_nd(static_cast<size_t>(c.m * c.k));
  for (size_t i = 0; i < a_nd.size(); ++i) {
    a_nd[i] = ExactFp8(static_cast<int64_t>(i));
  }
  std::vector<float> b_kn(static_cast<size_t>(c.k * c.n));
  for (size_t i = 0; i < b_kn.size(); ++i) {
    b_kn[i] = ExactFp8(static_cast<int64_t>(i) + 3);
  }
  std::vector<float> reference(static_cast<size_t>(c.m * c.n), 0.0f);
  for (int64_t r = 0; r < c.m; ++r) {
    for (int64_t col = 0; col < c.n; ++col) {
      float acc = 0.0f;
      for (int64_t t = 0; t < c.k; ++t) {
        acc += a_nd[static_cast<size_t>(r * c.k + t)] * b_kn[static_cast<size_t>(t * c.n + col)];
      }
      reference[static_cast<size_t>(r * c.n + col)] = acc;
    }
  }

  DeviceBuffer a_dev = DeviceBuffer::FromHost(ToNz(a_nd, c.m, c.k));
  DeviceBuffer b_dev = DeviceBuffer::FromHost(ToNz(b_kn, c.k, c.n));
  DeviceBuffer c_dev = DeviceBuffer::Empty<float>(static_cast<size_t>(c.m * c.n));
  ACL_CHECK(aclrtMemset(c_dev.get(), c_dev.size_bytes(), 0, c_dev.size_bytes()));

  turboquant_cube_gemm_probe_impl(
      stream, a_dev.get(), b_dev.get(), c_dev.get(), static_cast<uint32_t>(c.m), static_cast<uint32_t>(c.k),
      static_cast<uint32_t>(c.n), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kTileRows),
      static_cast<uint32_t>(NzElems(c.m, c.k)), static_cast<uint32_t>(NzElems(c.k, c.n)),
      static_cast<uint32_t>(c.m * c.n), 0u, c.variant);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const std::vector<float> got = c_dev.ToHost<float>();
  double worst = 0.0;
  size_t nonzero = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i] - reference[i])));
    if (got[i] != 0.0f) {
      ++nonzero;
    }
  }
  std::printf("[ sweep ] variant %u: max|err| = %-12g  nonzero %zu/%zu\n", c.variant, worst, nonzero, got.size());
  std::fflush(stdout);
  return worst;
}

TEST(CubeGemmContract, LoadBFromKnVariantSweep) {
  REQUIRE_ASCEND_950PR();
  if (std::getenv("ASCEND_TQ_GEMM_SWEEP") == nullptr) {
    GTEST_SKIP() << "set ASCEND_TQ_GEMM_SWEEP=1 to sweep the LoadBFromKn parameters";
  }
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  std::printf("[ sweep ] context GEMM shape m=%lld k=%lld n=%lld, B staged [k,n]\n",
              static_cast<long long>(kGroupHeads), static_cast<long long>(kTileRows),
              static_cast<long long>(kHeadSize));
  int exact = -1;
  for (uint32_t v = 0; v <= 8; ++v) {
    const Case c{"sweep", kGroupHeads, kTileRows, kHeadSize, false, v};
    if (ProbeCase(c, stream) == 0.0 && exact < 0) {
      exact = static_cast<int>(v);
    }
  }
  if (exact >= 0) {
    std::printf("[ sweep ] FIRST EXACT VARIANT: %d\n", exact);
  } else {
    std::printf("[ sweep ] no variant reproduced the host product\n");
  }
  std::fflush(stdout);
}

TEST(CubeGemmContract, ContextGemmRepeated) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_CUBE_WIP_OPT_IN("The repeated context GEMM",
                          "it is a diagnostic for a launch-order effect in the Cube path.");
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  std::printf("[ repeat ] same context GEMM, three launches, variant 0\n");
  for (int i = 0; i < 3; ++i) {
    const Case c{"repeat", kGroupHeads, kTileRows, kHeadSize, false, 0};
    std::printf("[ repeat ] launch %d: ", i);
    ProbeCase(c, stream);
  }
  std::fflush(stdout);
}

TEST(CubeGemmContract, ScoreGemmBStagedNk) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_CUBE_WIP_OPT_IN("The score GEMM's [n, k] B form",
                          "it does not reproduce the host product, while the [k, n] form in the case below "
                          "does so exactly -- so what is wrong is confined to LoadBFromNk.");
  const Case c{"score GEMM (Q . K^T)", kGroupHeads, kHeadSize, kTileRows, true, 0};
  RunCase(c, AscendTestEnvironment::Instance().stream());
}

TEST(CubeGemmContract, ContextGemmBStagedKn) {
  REQUIRE_ASCEND_950PR();
  const Case c{"context GEMM (P . V)", kGroupHeads, kTileRows, kHeadSize, false, 0};
  RunCase(c, AscendTestEnvironment::Instance().stream());
}

}
}
}
