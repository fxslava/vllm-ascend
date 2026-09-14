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

// End-to-end fidelity of the TurboQuant 4-bit KV cache, measured on an Ascend
// 950PR or on the CANN camodel simulator standing in for one.
//
// ONE PASS, ON PURPOSE: the camodel simulates the pipeline cycle by cycle, so a
// launch that is microseconds on silicon is minutes here. Every shape is the
// smallest that still exercises the thing it is there to exercise:
//
//   head_size 64      the bottom of the supported range; the Walsh-Hadamard is
//                     still four stages deep, so both its block-strided and its
//                     Gather-shuffled halves run
//   block_size 16     exactly one kTileRows tile
//   context 32        two blocks through a non-identity block table, so the
//                     online softmax carries state across a block boundary
//   4 heads / 2 kv    a GQA group of two
//
// WHAT IS COMPARED. Three outputs of the same decode step:
//
//   quantised   the two TurboQuant kernels on the device.
//   control     the same attention with an unquantised fp16 KV cache through
//               aclnnFusedInferAttentionScoreV2. Attempted, not required: on an
//               Ascend950 the planning call returns 361001 because the V1..V4
//               family is withdrawn (see aclnn_ops_950pr.hpp).
//   exact       fp32 attention over the same fp16 inputs, on the host. Always
//               available, which is why the assertions hang off it.
//
// WHAT IS ASSERTED:
//
//   quantised vs the CPU TurboQuant reference   -   tight (cos > 0.999)
//       Both run the identical algorithm, so anything beyond fp16 output
//       rounding is a kernel bug. This is the regression detector.
//
//   quantised vs exact fp32                     -   a floor (SNR >= 12 dB)
//       The scheme's error, not the kernel's. The floor sits well below what
//       the codec achieves because the number moves with the data; what it
//       catches is gross breakage -- a dropped un-rotation, a scale read from
//       the wrong lane, a block table ignored.
//
// Timing is printed and never asserted.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tq = turboquant_ref;
namespace tqh = turboquant_host;
namespace s = shapes950;

// --- the one shape this file runs -------------------------------------------

constexpr int64_t kHeadSize = 64;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kNumHeads = 4;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 4;
constexpr int64_t kContextLen = 32;
constexpr int64_t kBlocksPerSeq = kContextLen / kBlockSize;
constexpr int64_t kQueryTokens = 1;
constexpr float kAttentionScale = 0.125f;   // 1 / sqrt(64)
constexpr float kInvSqrtHeadSize = 0.125f;  // the kernels take this rather than doing a scalar sqrt

// --- the bounds -------------------------------------------------------------

// Kernel against the CPU reference. The two run the same arithmetic; the only
// licensed difference is the single fp16 rounding at the store.
constexpr double kMinKernelCosine = 0.999;
constexpr double kMaxKernelRelativeL2 = 5e-3;

// The codec against exact fp32. The bounds sit well clear of what this shape
// measures, because the number moves with the data.
//
// They do not discriminate "the Pi rotation was dropped": on iid Gaussian
// channels a bare absmax quantiser scores the same, since what the rotation
// buys is protection against anisotropic and outlier-heavy channels, which
// test_host_turboquant_fidelity.cpp measures on real activations instead. What
// they do catch is an un-rotation that never happened.
constexpr double kMinCosine = 0.97;
constexpr double kMinSnrDb = 12.0;
constexpr double kMaxRelativeL2 = 0.30;

// --- host-side pieces --------------------------------------------------------

// Exact fp32 paged attention over the unquantised context. Same arrangement as
// cpu_paged_attention_turboquant so the two differ only in the codec.
void ExactAttention(const std::vector<float>& query, const std::vector<float>& key, const std::vector<float>& value,
                    std::vector<float>* out) {
  const int64_t group = kNumHeads / kNumKvHeads;
  out->assign(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f);

  for (int64_t head = 0; head < kNumHeads; ++head) {
    const int64_t kv_head = head / group;
    std::vector<float> scores(static_cast<size_t>(kContextLen));
    float running_max = -std::numeric_limits<float>::infinity();

    for (int64_t i = 0; i < kContextLen; ++i) {
      const size_t k_base = static_cast<size_t>((i * kNumKvHeads + kv_head) * kHeadSize);
      double dot = 0.0;
      for (int64_t c = 0; c < kHeadSize; ++c) {
        dot += static_cast<double>(query[static_cast<size_t>(head * kHeadSize + c)]) * key[k_base + c];
      }
      scores[static_cast<size_t>(i)] = static_cast<float>(dot) * kAttentionScale;
      running_max = std::max(running_max, scores[static_cast<size_t>(i)]);
    }

    float denom = 0.0f;
    for (int64_t i = 0; i < kContextLen; ++i) {
      scores[static_cast<size_t>(i)] = std::exp(scores[static_cast<size_t>(i)] - running_max);
      denom += scores[static_cast<size_t>(i)];
    }
    for (int64_t i = 0; i < kContextLen; ++i) {
      const float weight = scores[static_cast<size_t>(i)] / denom;
      const size_t v_base = static_cast<size_t>((i * kNumKvHeads + kv_head) * kHeadSize);
      for (int64_t c = 0; c < kHeadSize; ++c) {
        (*out)[static_cast<size_t>(head * kHeadSize + c)] += weight * value[v_base + c];
      }
    }
  }
}

void PrintMetrics(const char* label, const tq::FidelityMetrics& m) {
  std::printf("  %-40s cos=%.6f  snr=%7.2f dB  relL2=%.6f\n", label, m.cosine_similarity, m.snr_db, m.relative_l2);
}

const AclnnOp& FusedInferAttentionOp() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
  return op;
}

// One decode step's inputs, all fp16-exact so the device and the host see the
// identical bit patterns and the only difference measured is the arithmetic.
struct Inputs {
  std::vector<float> key;    // [context_len, num_kv_heads, head_size]
  std::vector<float> value;  // same
  std::vector<float> query;  // [1, num_heads, head_size]
  std::vector<int32_t> slots;
  std::vector<int32_t> block_table;
};

Inputs MakeInputs() {
  DeterministicRandom rng(0x5A17u);
  Inputs in;
  const size_t kv_elems = static_cast<size_t>(kContextLen * kNumKvHeads * kHeadSize);
  in.key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  in.value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  in.query = rng.NormalHalfExact(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f, 1.0f);

  // A non-identity block table, so a decode that ignores paging reads the wrong
  // blocks rather than accidentally reading the right ones.
  const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(kNumBlocks));
  in.block_table.assign(permutation.begin(), permutation.begin() + kBlocksPerSeq);

  in.slots.resize(static_cast<size_t>(kContextLen));
  for (int64_t i = 0; i < kContextLen; ++i) {
    const int32_t block = in.block_table[static_cast<size_t>(i / kBlockSize)];
    in.slots[static_cast<size_t>(i)] = block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
  }
  return in;
}

// Milliseconds of wall clock around a synchronised launch. Reported, not
// asserted: under the camodel this is the simulator's speed.
template <typename Fn>
double TimeMs(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

class TurboQuantSimulatorFidelity : public ::testing::Test {};

}  // namespace

TEST_F(TurboQuantSimulatorFidelity, SingleDecodePassQuantisedVersusExact) {
  REQUIRE_ASCEND_950PR();

  const Inputs in = MakeInputs();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);

  std::printf("\n[turboquant/sim] single decode pass on '%s'\n",
              AscendTestEnvironment::Instance().soc_name().c_str());
  std::printf("  head_size=%lld heads=%lld kv_heads=%lld block_size=%lld context=%lld blocks=%lld aiv=%lld%s\n",
              static_cast<long long>(kHeadSize), static_cast<long long>(kNumHeads),
              static_cast<long long>(kNumKvHeads), static_cast<long long>(kBlockSize),
              static_cast<long long>(kContextLen), static_cast<long long>(kNumBlocks),
              static_cast<long long>(aiv_num), aiv_queried ? "" : " (assumed, runtime declined)");

  // --- 1. device memory for the quantised path -------------------------------
  //
  // Plain contiguous ND: the packed cache is
  // [num_blocks, block_size, num_kv_heads, head_size / 2] int8 and the scale
  // plane is [num_blocks, block_size, scale_slot] fp32, indexed by token.

  DeviceBuffer key_dev = DeviceBuffer::FromHost(FloatToHalf(in.key));
  DeviceBuffer value_dev = DeviceBuffer::FromHost(FloatToHalf(in.value));
  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(in.query));
  DeviceBuffer slots_dev = DeviceBuffer::FromHost(in.slots);
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(in.block_table);
  DeviceBuffer context_lens_dev =
      DeviceBuffer::FromHost(std::vector<int32_t>{static_cast<int32_t>(kContextLen)});
  DeviceBuffer pi_signs_dev = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));
  DeviceBuffer h16_dev = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  // The rotated query the split kernel now reads instead of rotating for
  // itself. fp32, same element count as the query.
  DeviceBuffer query_rot_dev =
      DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize));

  // The write path expands one vector per call and the decode path expands a
  // kTileRows tile; their table images differ and are not interchangeable.
  DeviceBuffer write_tables_dev = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer decode_tables_dev = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, tqh::kTileRows));

  DeviceBuffer key_cache_dev =
      DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize));
  DeviceBuffer value_cache_dev = DeviceBuffer::Empty<int8_t>(key_cache_dev.size_bytes());
  DeviceBuffer scale_plane_dev =
      DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(kNumBlocks, kBlockSize, kNumKvHeads));
  DeviceBuffer quantised_out_dev = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize));

  const tqh::PagedAttentionGrid decode_grid =
      tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, kBlocksPerSeq, aiv_num);
  DeviceBuffer workspace_dev = DeviceBuffer::Empty<float>(decode_grid.workspace_floats);

  // --- 2. the quantised invocation -------------------------------------------

  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(kContextLen, aiv_num);
  const double write_ms = TimeMs([&] {
    turboquant_reshape_and_cache_impl(
        AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(), key_cache_dev.get(),
        value_cache_dev.get(), scale_plane_dev.get(), slots_dev.get(), pi_signs_dev.get(), write_tables_dev.get(),
        static_cast<uint32_t>(kContextLen), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), write_grid.tokens_per_core, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  });

  // Two launches on one stream: split writes one partial per (token, head,
  // sequence split), combine reduces them. They cannot share a launch - an
  // in-kernel barrier only orders co-resident blocks - so stream order is the
  // barrier, and the single synchronise below covers both.
  const double decode_ms = TimeMs([&] {
    // Three launches on one stream now: the rotation, then the split, then the
    // combine. The rotation has to come first and stream order is what says so.
    tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs_dev.get(), h16_dev.get(),
                     write_tables_dev.get(), query_rot_dev.get(), kQueryTokens, kNumHeads, kHeadSize, aiv_num,
                     /*input_exact_in_half=*/true);
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, decode_grid.split_block_dim, decode_grid.combine_block_dim, query_rot_dev.get(),
        key_cache_dev.get(), value_cache_dev.get(), scale_plane_dev.get(), block_table_dev.get(),
        context_lens_dev.get(), pi_signs_dev.get(), decode_tables_dev.get(), workspace_dev.get(),
        quantised_out_dev.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
        static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(kBlocksPerSeq), static_cast<uint32_t>(decode_grid.num_splits),
        decode_grid.split_tasks_per_core, decode_grid.combine_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  });

  const std::vector<float> quantised = HalfToFloat(quantised_out_dev.ToHost<Half>());

  // --- 3. the unquantised control --------------------------------------------
  //
  // Same decode, same paging, fp16 KV cache, through the stock CANN operator the
  // plugin's DecodeOnly path calls. FiaKeyCacheView reproduces the head-axis
  // flattening _get_fia_params does before the call.

  std::vector<float> control;
  double control_ms = 0.0;
  std::string control_note;

  if (!FusedInferAttentionOp().available()) {
    control_note = "not run: " + FusedInferAttentionOp().unavailable_reason();
  } else {
    // The unquantised cache, written on the host into the same slots the
    // TurboQuant write path used, so both legs read the identical paging.
    std::vector<float> fp16_key_cache(
        static_cast<size_t>(kNumBlocks * kBlockSize * kNumKvHeads * kHeadSize), 0.0f);
    std::vector<float> fp16_value_cache(fp16_key_cache.size(), 0.0f);
    for (int64_t pos = 0; pos < kContextLen; ++pos) {
      const size_t slot = static_cast<size_t>(in.slots[static_cast<size_t>(pos)]);
      for (int64_t kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
        const size_t src = static_cast<size_t>((pos * kNumKvHeads + kv_head) * kHeadSize);
        const size_t dst = (slot * static_cast<size_t>(kNumKvHeads) + static_cast<size_t>(kv_head)) *
                           static_cast<size_t>(kHeadSize);
        std::copy(in.key.begin() + static_cast<std::ptrdiff_t>(src),
                  in.key.begin() + static_cast<std::ptrdiff_t>(src + kHeadSize),
                  fp16_key_cache.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(in.value.begin() + static_cast<std::ptrdiff_t>(src),
                  in.value.begin() + static_cast<std::ptrdiff_t>(src + kHeadSize),
                  fp16_value_cache.begin() + static_cast<std::ptrdiff_t>(dst));
      }
    }

    DeviceBuffer control_key = DeviceBuffer::FromHost(FloatToHalf(fp16_key_cache));
    DeviceBuffer control_value = DeviceBuffer::FromHost(FloatToHalf(fp16_value_cache));
    DeviceBuffer control_out = DeviceBuffer::Empty<Half>(static_cast<size_t>(kNumHeads * kHeadSize));
    DeviceBuffer softmax_lse = DeviceBuffer::Empty<Half>(1);

    const std::vector<int64_t> cache_view = s::FiaKeyCacheView(kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize);
    AclnnTensor key_flat(cache_view, ACL_FLOAT16, control_key.get());
    AclnnTensor value_flat(cache_view, ACL_FLOAT16, control_value.get());
    AclnnTensorList key_list({key_flat.get()});
    AclnnTensorList value_list({value_flat.get()});

    AclnnTensor query_tnd({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, query_dev.get());
    AclnnTensor block_table_tensor({kQueryTokens, kBlocksPerSeq}, ACL_INT32, block_table_dev.get());
    AclnnTensor context_tensor({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, control_out.get());
    AclnnTensor lse_tensor({1}, ACL_FLOAT16, softmax_lse.get());
    AclnnIntArray actual_seq_lengths(std::vector<int64_t>{kQueryTokens});
    AclnnIntArray actual_seq_lengths_kv(std::vector<int64_t>{kContextLen});

    // The control is best-effort: its argument list has never been executed on
    // this part, and a camodel run may have no binary kernel for the operator.
    // A failure is reported and the exact-path comparison carries on.
    try {
      control_ms = TimeMs([&] {
        RunAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
            FusedInferAttentionOp(), stream, query_tnd.get(), key_list.get(), value_list.get(),
            /*pse_shift=*/nullptr, /*atten_mask=*/nullptr, actual_seq_lengths.get(), actual_seq_lengths_kv.get(),
            /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr,
            /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr,
            block_table_tensor.get(), /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr,
            /*key_antiquant_scale=*/nullptr, /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr,
            /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr,
            /*actual_shared_prefix_len=*/nullptr, kNumHeads, static_cast<double>(kAttentionScale),
            s::kFiaUnboundedTokens, s::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), kNumKvHeads,
            s::kFiaSparseModeNone, s::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0,
            /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0, context_tensor.get(),
            lse_tensor.get());
      });
      control = HalfToFloat(control_out.ToHost<Half>());
    } catch (const AclError& error) {
      control.clear();
      control_note = std::string("not run: ") + error.what();
    }
  }

  // --- 4. the exact path and the reference -----------------------------------

  std::vector<float> exact;
  ExactAttention(in.query, in.key, in.value, &exact);

  // The CPU TurboQuant reference, reading back the cache the device actually
  // wrote. Using the device's cache rather than a separately quantised one is
  // deliberate: it isolates the decode kernel, so a disagreement here is the
  // decode and not the write path.
  const std::vector<int8_t> device_key_cache = key_cache_dev.ToHost<int8_t>();
  const std::vector<int8_t> device_value_cache = value_cache_dev.ToHost<int8_t>();
  const std::vector<float> device_scale_plane = scale_plane_dev.ToHost<float>();
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(kHeadSize));

  std::vector<float> reference(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f);
  tq::cpu_paged_attention_turboquant(in.query.data(), device_key_cache.data(), device_value_cache.data(),
                                     device_scale_plane.data(), in.block_table.data(),
                                     static_cast<int>(kContextLen), static_cast<int>(kNumHeads),
                                     static_cast<int>(kNumKvHeads), static_cast<int>(kHeadSize),
                                     static_cast<int>(kBlockSize), kAttentionScale, signs.data(), reference.data());

  // --- 5. the report ----------------------------------------------------------

  const tq::FidelityMetrics vs_exact = tq::cpu_fidelity(quantised, exact);
  const tq::FidelityMetrics vs_reference = tq::cpu_fidelity(quantised, reference);

  std::printf("  splits=%lld  write %.2f ms, decode %.2f ms (2 launches)\n",
              static_cast<long long>(decode_grid.num_splits), write_ms, decode_ms);
  PrintMetrics("4-bit NPU vs exact fp32", vs_exact);
  PrintMetrics("4-bit NPU vs CPU TurboQuant reference", vs_reference);

  if (!control.empty()) {
    const tq::FidelityMetrics control_vs_exact = tq::cpu_fidelity(control, exact);
    const tq::FidelityMetrics quantised_vs_control = tq::cpu_fidelity(quantised, control);
    std::printf("  unquantised fp16 control: %.2f ms\n", control_ms);
    PrintMetrics("fp16 control vs exact fp32", control_vs_exact);
    PrintMetrics("4-bit NPU vs fp16 control", quantised_vs_control);

    // The control is the device's own fp16 error floor. If it is worse than the
    // 4-bit path against the same reference, the control did not compute the
    // attention this test thinks it did, and reading anything into the
    // comparison would be wrong.
    EXPECT_GT(control_vs_exact.snr_db, vs_exact.snr_db)
        << "the unquantised fp16 control is no more accurate than the 4-bit path; the control's argument list is "
           "the unverified one from aclnn_ops_950pr.hpp and is the first thing to doubt";
  } else {
    std::printf("  unquantised fp16 control: %s\n", control_note.c_str());
    std::printf("    the exact fp32 path below is the reference either way; the control only adds the device's\n"
                "    own fp16 error floor to the report.\n");
  }
  std::fflush(stdout);

  // --- 6. the assertions ------------------------------------------------------

  // Tight: same algorithm on both sides.
  EXPECT_GT(vs_reference.cosine_similarity, kMinKernelCosine)
      << "the decode kernel and the CPU reference run the same arithmetic; a disagreement in direction is a kernel "
         "bug, not quantisation";
  EXPECT_LT(vs_reference.relative_l2, kMaxKernelRelativeL2)
      << "the decode kernel and the CPU reference disagree in magnitude by more than fp16 output rounding allows";

  // A floor: the 4-bit scheme's own error against exact attention.
  EXPECT_GT(vs_exact.cosine_similarity, kMinCosine)
      << "the 4-bit output points somewhere else entirely, which quantisation noise does not do - suspect the "
         "un-rotation or the scale lane";
  EXPECT_GT(vs_exact.snr_db, kMinSnrDb) << "4-bit TurboQuant fidelity has fallen well below the ~18 dB this shape "
                                           "measures on the host";
  EXPECT_LT(vs_exact.relative_l2, kMaxRelativeL2) << "4-bit TurboQuant relative L2 error has grown past its floor";
}

}  // namespace test
}  // namespace vllm_ascend
