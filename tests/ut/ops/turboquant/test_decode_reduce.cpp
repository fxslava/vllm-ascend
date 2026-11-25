// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Host evidence for launch coverage and dense reduction padding, not a device test.
#include "csrc/attention/turboquant/common/turboquant_gemm_reduce.hpp"
#include "csrc/attention/turboquant/common/turboquant_tiling.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gr = ascend_micro::gemm_reduce;
namespace fd = ascend_micro::fused_decode;

namespace {

constexpr unsigned kHeads = 24;
constexpr unsigned kMaxGrid = 32;
constexpr int kFusedContextLimit = 4096;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_output_coverage(const std::vector<std::int32_t>& contexts, unsigned splits) {
  fd::DecodeTilingRequest r;
  r.tokens = static_cast<unsigned>(contexts.size());
  r.num_heads = kHeads;
  r.num_kv_heads = 4;
  r.heads_per_task = 6;
  r.splits = splits;
  r.max_blocks = 66;
  const auto launch = fd::decode_launch_split(r, kMaxGrid);
  r.grid = launch.grid;
  r.tasks_per_block = launch.tasks_per_block;
  r.reduce_tasks_per_block = launch.reduce_tasks_per_block;
  std::vector<std::uint32_t> image(fd::decode_tiling_words(r.tokens, splits, r.grid));
  require(fd::build_decode_tiling(r, contexts.data(), nullptr, image.data()) == fd::DecodeTilingStatus::kOk,
          "build reduction dispatch");
  const auto* dispatch = reinterpret_cast<const fd::TurboQuantDecodeDispatch*>(
      image.data() + fd::decode_dispatch_word_base(r.tokens, splits));
  const unsigned total = r.tokens * kHeads;
  std::vector<unsigned> visits(total, 0);
  const auto needs_combine = [&](unsigned task) { return splits > 1 && contexts[task / kHeads] > kFusedContextLimit; };
  for (unsigned block = 0; block < r.grid; ++block) {
    const unsigned start = block * r.reduce_tasks_per_block;
    const unsigned count = gr::kReduceTaskCount(total, block, r.reduce_tasks_per_block);
    if (!count) continue;
    require(dispatch[block].reduce_token * kHeads + dispatch[block].reduce_head == start,
            "vector cursor agrees with GEMM block range");
    const unsigned half = (count + 1) / 2;
    require(dispatch[block].reduce_token_hi * kHeads + dispatch[block].reduce_head_hi == start + half,
            "second vector cursor agrees with block half");
    unsigned task = start;
    unsigned batches = 0;
    while (task < start + count) {
      if (!needs_combine(task)) {
        ++task;
        continue;
      }
      const unsigned first = task++;
      unsigned live = 1;
      while (task < start + count && needs_combine(task) && !gr::kGemmBatchBreaks(task - 1, task, live)) {
        ++live;
        ++task;
      }
      require(++batches <= gr::kMaxGemmBatchesPerBlock, "batch capacity covers block");
      for (unsigned sub = 0; sub < 2; ++sub) {
        for (unsigned row = 0; row < gr::kGemmLiveRowsOf(live, sub); ++row) {
          const unsigned output = first + gr::kGemmFirstTaskOf(live, sub) + row;
          require(needs_combine(output) && ++visits[output] == 1,
                  "GEMM output ownership is disjoint and excludes fused requests");
        }
      }
    }
  }
  for (unsigned task = 0; task < total; ++task)
    require(visits[task] == unsigned(needs_combine(task)), "every split head is written once");

  if (contexts.size() == 2 && contexts[0] == 136 && splits == 3) {
    require(launch.grid == 24 && launch.tasks_per_block == 1 && launch.reduce_tasks_per_block == 2,
            "reported S=3 launch geometry");
  }
}

float dense_product(const std::vector<float>& operand, unsigned packed_row, unsigned splits, unsigned column) {
  float sum = 0.0f;
  // Deliberately multiply all K lanes. Skipping zero weights would hide the bug.
  for (unsigned row = 0; row < gr::kGemmBatchRows; ++row) {
    for (unsigned split = 0; split < gr::kGemmKPerRow; ++split) {
      const float weight = row == packed_row && split < splits ? 1.0f / (split + 1) : 0.0f;
      sum += weight * operand[(row * gr::kGemmKPerRow + split) * gr::kGemmSliceCols + column];
    }
  }
  return sum;
}

void check_poisoned_padding(unsigned live, unsigned splits, float poison) {
  std::vector<float> operand(gr::kGemmKMax * gr::kGemmSliceCols, poison);
  for (unsigned sub = 0; sub < 2; ++sub) {
    for (unsigned row = 0; row < gr::kGemmLiveRowsOf(live, sub); ++row) {
      const unsigned packed = gr::kGemmFirstRowOf(sub) + row;
      const unsigned task = gr::kGemmFirstTaskOf(live, sub) + row;
      for (unsigned split = 0; split < gr::kGemmKPerRow; ++split) {
        for (unsigned col = 0; col < gr::kGemmSliceCols; ++col) {
          // Model DN2NZ's finite live data and its within-row C0 zero padding.
          operand[(packed * gr::kGemmKPerRow + split) * gr::kGemmSliceCols + col] =
              split < splits ? float((task + 1) * (split + 1)) : 0.0f;
        }
      }
    }
  }
  if (live < gr::kGemmBatchRows)
    require(!std::isfinite(dense_product(operand, 0, splits, 0)),
            "uncleared padding must reproduce zero-times-poison contamination");
  for (unsigned sub = 0; sub < 2; ++sub) {
    const unsigned first = gr::kGemmPaddingFirstRowOf(live, sub);
    const unsigned end = first + gr::kGemmPaddingRowsOf(live, sub);
    require(end == gr::kGemmFirstRowOf(sub) + gr::kGemmPanelRows, "padding stops at the static half boundary");
    for (unsigned at = first * gr::kSliceFractalFloats; at < end * gr::kSliceFractalFloats; ++at) operand[at] = 0.0f;
  }
  for (unsigned sub = 0; sub < 2; ++sub) {
    for (unsigned row = 0; row < gr::kGemmLiveRowsOf(live, sub); ++row) {
      const unsigned packed = gr::kGemmFirstRowOf(sub) + row;
      const unsigned task = gr::kGemmFirstTaskOf(live, sub) + row;
      const float expected = float((task + 1) * splits);
      for (unsigned col = 0; col < gr::kGemmSliceCols; ++col) {
        const float actual = dense_product(operand, packed, splits, col);
        require(std::isfinite(actual) && std::abs(actual - expected) < 0.0001f,
                "cleared padding preserves every live output");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    for (unsigned splits = 1; splits <= gr::kMaxSplits; ++splits) {
      for (const std::vector<std::int32_t>& contexts : {std::vector<std::int32_t>{8223},
                                                        {136, 8223},
                                                        {264, 8223},
                                                        {1032, 8223},
                                                        {8223, 136},
                                                        {4096, 4097, 8192},
                                                        {4097, 8192, 8193}})
        check_output_coverage(contexts, splits);
      for (unsigned live = 1; live <= gr::kGemmBatchRows; ++live) {
        for (const float poison : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()})
          check_poisoned_padding(live, splits, poison);
      }
    }
  } catch (const std::exception& error) {
    std::printf("FAIL decode reduction: %s\n", error.what());
    return 1;
  }
  std::printf(
      "PASS decode reduction: mixed-batch output coverage; NaN/Inf padding, "
      "live rows 1..16, splits 1..8\n");
}
