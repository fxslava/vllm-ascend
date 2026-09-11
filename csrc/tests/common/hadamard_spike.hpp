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

// Host side of the Cube-Hadamard spike: the constant images the kernels read,
// the chunk planning they require of their caller, and the launcher
// declarations. Shared by the sim test, the device test and the benchmark so
// the three cannot disagree about a table or a chunk size - a disagreement
// there would show up as a fidelity failure and be read as a kernel bug.
//
// EXPLORATORY SPIKE. The kernels are csrc/tests/sim/sim_hadamard_hybrid_kernels.cpp,
// which is test-owned device code; nothing here is on the decode path and
// nothing here is built into the wheel.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vllm_ascend {

/*
 * Declared here rather than in common/turboquant_launch.hpp on purpose: that
 * header is the suite's contract with the production kernels, and this spike
 * has no business in it. The symbols come from the same
 * libvllm_ascend_turboquant.so, which lists the spike's kernel source only in
 * the tests' build.
 *
 *   headDim          a power of two in [64, 512]
 *   vectorsPerChunk  divides numVectors; use HadamardVectorsPerChunk
 *   variant          a mask of HadamardVariant
 *   invSqrtDim       1 / sqrt(headDim), so the device needs no square root
 */
void sim_hadamard_hybrid_impl(void *stream, void *input, void *h16, void *output, uint32_t headDim,
                              uint32_t numVectors, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim);

void sim_hadamard_aiv_impl(void *stream, void *input, void *tables, void *output, uint32_t headDim,
                           uint32_t numVectors, float invSqrtDim);

namespace test {
namespace hadamard_spike {

// The Cube fractal quantum, and the order of the constant matrix the Mmad
// multiplies by. Mirrors kTile in the kernel.
constexpr int64_t kTile = 16;

// Butterfly strides below this straddle a 32 B fp32 block and need a Gather
// shuffle rather than a strided Add/Sub pair. Mirrors kEarlyStages.
constexpr int64_t kEarlyStages = 3;

// Elements one chunk may hold. The kernel's UB working set is 14 bytes per
// element - fp32 in, fp16 cast, fp32 product, fp32 ping-pong - so this is
// 56 KB of the subcore's UB, and it is the shape the camodel result at
// D = 256, V = 16 was measured on. Raising it trades headroom for fewer
// chunks; it is a constant rather than a computed fit because a wrong fit
// shows up as InitBuffer handing back a base of 0 and stores decoding as DDR,
// which does not fault.
constexpr int64_t kMaxChunkElements = 4096;

// Variant bits, mirrored from the kernel.
enum HadamardVariant : uint32_t {
  kHybridSingleMmad = 0x0u,
  // Two Mmads accumulating into one L0C, x = hi + lo, both halves fp16.
  kHybridHiLo = 0x1u,
  // Fixpipe dualDstCtl = 0b01: half the product into each subcore's UB. The
  // kernel ignores it when a chunk holds an odd number of vectors, because the
  // M split would then land inside a vector's tile.
  kHybridDualDst = 0x2u,
  // The un-pipelined loop: every chunk is a self-contained stage -> Mmad ->
  // Fixpipe -> residual with one L1 and one UB slot, so the AIV waits out every
  // Cube stage and the Cube waits out every vector stage. Kept only so the
  // pipelined default can be measured against it at the SAME chunking; it
  // computes exactly the same thing, bit for bit.
  kHybridLockstep = 0x4u,
};

// fp16 bit patterns for the only two values H_16 contains. Both are exact, so
// the Cube stage introduces no error of its own.
constexpr uint16_t kHalfOne = 0x3C00;
constexpr uint16_t kHalfMinusOne = 0xBC00;

// H_16[i][j] = (-1)^popcount(i & j): Sylvester's construction, symmetric, so
// the [n, k] image the Cube loads and the [k, n] image the product needs are
// the same bytes. Row major, which for a 16-column half operand is also NZ.
inline std::vector<uint16_t> Hadamard16Half() {
  std::vector<uint16_t> h(static_cast<size_t>(kTile * kTile));
  for (int64_t i = 0; i < kTile; ++i) {
    for (int64_t j = 0; j < kTile; ++j) {
      unsigned bits = static_cast<unsigned>(i & j);
      int parity = 0;
      while (bits != 0) {
        parity ^= static_cast<int>(bits & 1u);
        bits >>= 1;
      }
      h[static_cast<size_t>(i * kTile + j)] = parity ? kHalfMinusOne : kHalfOne;
    }
  }
  return h;
}

// The shuffle tables the AIV comparison needs for strides 1, 2 and 4: per
// stage, `dim` fp32 sign lanes then `dim` Gather byte offsets. Same image
// layout as turboquant_host::CodecTables builds for the production codec,
// restated here so the spike does not depend on that builder's other sections.
inline std::vector<int32_t> EarlyStageTables(int64_t dim) {
  constexpr int32_t kWord = 4;
  std::vector<int32_t> tables;
  tables.reserve(static_cast<size_t>(kEarlyStages * 2 * dim));
  for (int64_t stage = 0; stage < kEarlyStages; ++stage) {
    const int64_t stride = static_cast<int64_t>(1) << stage;
    for (int64_t c = 0; c < dim; ++c) {
      const float sign = static_cast<float>(1 - 2 * ((c / stride) & 1));
      int32_t bits = 0;
      std::memcpy(&bits, &sign, sizeof(bits));
      tables.push_back(bits);
    }
    for (int64_t c = 0; c < dim; ++c) {
      tables.push_back(static_cast<int32_t>(kWord * (c ^ stride)));
    }
  }
  return tables;
}

/*
 * Vectors the kernel may hold in flight at this shape.
 *
 * Both arguments are powers of two here, so the minimum of the UB fit and the
 * batch always divides the batch and every chunk is full. A caller that ever
 * passes a non-power-of-two batch has to check that itself: the kernel divides
 * numVectors by this and would silently drop the remainder.
 */
inline int64_t HadamardVectorsPerChunk(int64_t dim, int64_t num_vectors) {
  const int64_t fits = kMaxChunkElements / dim;
  const int64_t chunk = fits < num_vectors ? fits : num_vectors;
  return chunk < 1 ? 1 : chunk;
}

/*
 * The smallest chunk that still exercises the macro-pipeline.
 *
 * HadamardVectorsPerChunk maximises the chunk, which at D = 256, V = 16 is the
 * whole batch: one chunk, a prolog and an epilogue and no steady state at all.
 * A pipelined kernel measured there is the lockstep kernel. Four vectors per
 * chunk gives four chunks - two prolog/epilogue iterations and two that visit
 * every slot-reuse edge, including the kFlagOperandsFree and kFlagProductFree
 * waits, which a three-chunk run would only half reach.
 *
 * Still even, so the dual-destination Fixpipe applies and both subcores work.
 */
constexpr int64_t kPipelineVectorsPerChunk = 4;

// True when the dual-destination Fixpipe can actually split this shape. An odd
// chunk - in practice a batch of one - would split a vector's tile between the
// subcores, so the kernel falls back to the single-destination path and the
// variant measures the same thing as kHybridHiLo.
inline bool HadamardDualDstApplies(int64_t dim, int64_t num_vectors) {
  return HadamardVectorsPerChunk(dim, num_vectors) % 2 == 0;
}

inline float InvSqrtDim(int64_t dim) {
  return 1.0f / std::sqrt(static_cast<float>(dim));
}

// A deterministic unit-scale batch. 24 significant bits, so nothing in it is
// exactly representable in fp16 and the operand grid is measured rather than
// dodged. The LCG is the one the CPU reference uses for its sign vector, so
// there is one generator in the project and no second seed to drift.
inline std::vector<float> SyntheticBatch(int64_t dim, int64_t num_vectors, uint32_t seed = 0x9E3779B9u) {
  std::vector<float> x(static_cast<size_t>(dim * num_vectors));
  uint32_t state = seed;
  for (size_t i = 0; i < x.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    x[i] = static_cast<float>(static_cast<double>(state >> 8) / 8388608.0 - 1.0);
  }
  return x;
}

// Largest absolute difference against the golden, and where it is. Accumulated
// in double so the comparison is not itself a source of error.
struct Deviation {
  double max_abs = 0.0;
  double rms = 0.0;
  size_t worst_at = 0;
};

inline Deviation Compare(const std::vector<float>& got, const std::vector<float>& want) {
  Deviation d;
  double sumsq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    sumsq += err * err;
    if (err > d.max_abs) {
      d.max_abs = err;
      d.worst_at = i;
    }
  }
  d.rms = std::sqrt(sumsq / static_cast<double>(want.size()));
  return d;
}

// "d256_v16_hilo_dualdst". One token per axis, so a CSV of these splits on '_'
// and a table of them sorts sensibly.
inline std::string CaseLabel(int64_t dim, int64_t num_vectors, const char* variant) {
  return "d" + std::to_string(dim) + "_v" + std::to_string(num_vectors) + "_" + variant;
}

}  // namespace hadamard_spike
}  // namespace test
}  // namespace vllm_ascend
