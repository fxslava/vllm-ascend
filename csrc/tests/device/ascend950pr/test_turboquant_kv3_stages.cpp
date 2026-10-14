/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant/dev_dump_utils.hpp"
#include "turboquant/turboquant_launch.hpp"

namespace vllm_ascend {

// Compiled only into the C++ test kernel library (VLLM_ASCEND_TQ_TEST_KERNELS).
void turboquant_mm_debug_flat_kv3fp4_impl(void* stream, void* key, void* value, void* key_cache, void* value_cache,
                                          void* scale_cache, void* slot_mapping, void* pi_signs, void* rot_tables,
                                          void* mode_tables, void* debug_gm, uint32_t num_kv_heads, uint32_t head_size,
                                          uint32_t block_size, float inv_sqrt_len);

namespace test {
namespace {

namespace tq = turboquant_ref;
namespace tqh = turboquant_host;
namespace dump = turboquant_dump;
using turboquant::TurboQuantMode;

constexpr size_t kHeadSize = 256;
constexpr size_t kPackedBytes = 96;
constexpr size_t kContextLen = 16;
// The logical test covers 16 rows; cache allocation still reserves a full Cube tile.
constexpr size_t kBlockSize = static_cast<size_t>(tqh::kCubeTileRows);
constexpr uint32_t kInputSeed = 0x3B17u;

enum class Stage { kFlat, kCache, kScale };

struct ReferenceImage {
  std::vector<int8_t> flat_key;
  std::vector<int8_t> flat_value;
  std::vector<float> scales;
};

ReferenceImage BuildReference(const std::vector<float>& key, const std::vector<float>& value, size_t rows,
                              size_t kv_heads) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(kHeadSize));
  const size_t plane_bytes = rows * kv_heads * kPackedBytes;
  ReferenceImage ref;
  ref.flat_key.assign(plane_bytes, 0);
  ref.flat_value.assign(plane_bytes, 0);
  const size_t scale_stride = static_cast<size_t>(tqh::ScaleSlotFloats(kv_heads));
  ref.scales.assign(rows * scale_stride, 0.0f);

  for (size_t plane = 0; plane < 2; ++plane) {
    const std::vector<float>& input = plane == 0 ? key : value;
    std::vector<int8_t>& flat = plane == 0 ? ref.flat_key : ref.flat_value;
    for (size_t token = 0; token < rows; ++token) {
      for (size_t head = 0; head < kv_heads; ++head) {
        const size_t vector_base = (token * kv_heads + head) * kHeadSize;
        const std::ptrdiff_t vector_offset = static_cast<std::ptrdiff_t>(vector_base);
        std::vector<float> rotated(input.begin() + vector_offset,
                                   input.begin() + vector_offset + static_cast<std::ptrdiff_t>(kHeadSize));
        tq::cpu_apply_pi(rotated.data(), static_cast<int>(kHeadSize), signs.data());
        const float scale = tq::cpu_rms_scale(rotated.data(), static_cast<int>(kHeadSize));
        ref.scales[token * scale_stride + plane * kv_heads + head] = scale;
        std::vector<int32_t> bins(kHeadSize);
        for (size_t column = 0; column < kHeadSize; ++column) {
          const float scaled = rotated[column] / scale;
          int32_t bin = 0;
          for (const float threshold : turboquant::TurboQuantModeTraits<TurboQuantMode::KV3_FP4>::kThresholds) {
            bin += scaled > threshold ? 1 : 0;
          }
          bins[column] = bin;
        }
        const std::vector<int8_t> encoded = tq::cpu_pack_kv3_bins(bins);
        for (size_t byte = 0; byte < kPackedBytes; ++byte) {
          flat[(token * kv_heads + head) * kPackedBytes + byte] = encoded[byte];
        }
      }
    }
  }
  return ref;
}

void RunStage(Stage stage, size_t kv_heads, size_t plane, const char* test_case_name) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  ASSERT_EQ(static_cast<size_t>(tqh::ModePackedBytes(TurboQuantMode::KV3_FP4, kHeadSize)), kPackedBytes);
  ASSERT_TRUE(kv_heads == 1 || kv_heads == 2);
  ASSERT_LE(kContextLen, kBlockSize);
  const size_t rows = stage == Stage::kFlat ? 1 : kContextLen;
  const size_t cache_rows = stage == Stage::kFlat ? rows : kBlockSize;
  DeterministicRandom rng(kInputSeed);
  const std::vector<float> key = rng.NormalHalfExact(rows * kv_heads * kHeadSize, 0.0f, 1.0f);
  const std::vector<float> value = rng.NormalHalfExact(rows * kv_heads * kHeadSize, 0.0f, 1.0f);
  const ReferenceImage expected = BuildReference(key, value, rows, kv_heads);
  const std::vector<int32_t> slots = [&] {
    std::vector<int32_t> result(rows);
    for (size_t token = 0; token < rows; ++token) result[token] = static_cast<int32_t>(token);
    return result;
  }();

  DeviceBuffer key_gm = DeviceBuffer::FromHost(FloatToHalf(key));
  DeviceBuffer value_gm = DeviceBuffer::FromHost(FloatToHalf(value));
  DeviceBuffer key_cache = DeviceBuffer::Empty<int8_t>(cache_rows * kv_heads * kPackedBytes);
  DeviceBuffer value_cache = DeviceBuffer::Empty<int8_t>(cache_rows * kv_heads * kPackedBytes);
  DeviceBuffer scale_cache =
      DeviceBuffer::Empty<float>(cache_rows * static_cast<size_t>(tqh::ScaleSlotFloats(kv_heads)));
  DeviceBuffer slots_gm = DeviceBuffer::FromHost(slots);
  DeviceBuffer signs_gm = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));
  DeviceBuffer rotation_gm = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer mode_gm = DeviceBuffer::FromHost(tqh::ModeTables(TurboQuantMode::KV3_FP4, kHeadSize, 1, 0));
  DeviceBuffer flat_debug_gm = DeviceBuffer::Empty<int8_t>(2 * kv_heads * kPackedBytes);
  void* stream = AscendTestEnvironment::Instance().stream();
  const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(kHeadSize));

  if (stage == Stage::kFlat) {
    turboquant_mm_debug_flat_kv3fp4_impl(
        stream, key_gm.get(), value_gm.get(), key_cache.get(), value_cache.get(), scale_cache.get(), slots_gm.get(),
        signs_gm.get(), rotation_gm.get(), mode_gm.get(), flat_debug_gm.get(), static_cast<uint32_t>(kv_heads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), inv_sqrt_len);
  } else {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(rows, tqh::VectorCoreNum(nullptr));
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(TurboQuantMode::KV3_FP4), AscendType::FP16, stream, grid.block_dim, key_gm.get(),
        value_gm.get(), key_cache.get(), value_cache.get(), scale_cache.get(), slots_gm.get(), signs_gm.get(),
        rotation_gm.get(), mode_gm.get(), static_cast<uint32_t>(rows), static_cast<uint32_t>(kv_heads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), 1, grid.tokens_per_core, inv_sqrt_len);
  }
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const char* plane_label = plane == 0 ? "KEY" : "VALUE";
  const std::string stage_label = stage == Stage::kFlat ? "STAGE_1_FLAT_" + std::string(plane_label)
                                  : stage == Stage::kCache ? "STAGE_2_CACHE_" + std::string(plane_label)
                                                        : "STAGE_3_SCALE_" + std::string(plane_label);
  size_t mismatches = 0;
  if (stage == Stage::kFlat) {
    mismatches =
        dump::DumpPackedStage(test_case_name, stage_label.c_str(), flat_debug_gm,
                              plane == 0 ? expected.flat_key : expected.flat_value, rows, kv_heads, kPackedBytes,
                              false, plane * kv_heads * kPackedBytes);
  } else if (stage == Stage::kCache) {
    // KV3 cache bytes use slot * (kv_heads * packed_bytes) + head * packed_bytes + byte.
    mismatches =
        dump::DumpPackedStage(test_case_name, stage_label.c_str(), plane == 0 ? key_cache : value_cache,
                              plane == 0 ? expected.flat_key : expected.flat_value, rows, kv_heads, kPackedBytes,
                              false);
  } else {
    mismatches =
        dump::DumpScaleStage(test_case_name, stage_label.c_str(), scale_cache, expected.scales, rows, kv_heads, plane);
  }
  EXPECT_EQ(mismatches, 0u) << "Compact diff: " << dump::DumpFilePath(test_case_name).string();
}

// Run one case with --gtest_filter=TurboQuantKv3Stages.<case>; every invocation
// writes only csrc/tests/dev_dumps/<case>.txt for easy remote transfer.
TEST(TurboQuantKv3Stages, Stage1FlatKeyH1) { RunStage(Stage::kFlat, 1, 0, "Stage1FlatKeyH1"); }
TEST(TurboQuantKv3Stages, Stage1FlatValueH1) { RunStage(Stage::kFlat, 1, 1, "Stage1FlatValueH1"); }
TEST(TurboQuantKv3Stages, Stage1FlatKeyH2) { RunStage(Stage::kFlat, 2, 0, "Stage1FlatKeyH2"); }
TEST(TurboQuantKv3Stages, Stage1FlatValueH2) { RunStage(Stage::kFlat, 2, 1, "Stage1FlatValueH2"); }
TEST(TurboQuantKv3Stages, Stage2CacheWriteKeyH1) { RunStage(Stage::kCache, 1, 0, "Stage2CacheWriteKeyH1"); }
TEST(TurboQuantKv3Stages, Stage2CacheWriteValueH1) { RunStage(Stage::kCache, 1, 1, "Stage2CacheWriteValueH1"); }
TEST(TurboQuantKv3Stages, Stage2CacheWriteKeyH2) { RunStage(Stage::kCache, 2, 0, "Stage2CacheWriteKeyH2"); }
TEST(TurboQuantKv3Stages, Stage2CacheWriteValueH2) { RunStage(Stage::kCache, 2, 1, "Stage2CacheWriteValueH2"); }
TEST(TurboQuantKv3Stages, Stage3ScaleKeyH1) { RunStage(Stage::kScale, 1, 0, "Stage3ScaleKeyH1"); }
TEST(TurboQuantKv3Stages, Stage3ScaleValueH1) { RunStage(Stage::kScale, 1, 1, "Stage3ScaleValueH1"); }
TEST(TurboQuantKv3Stages, Stage3ScaleKeyH2) { RunStage(Stage::kScale, 2, 0, "Stage3ScaleKeyH2"); }
TEST(TurboQuantKv3Stages, Stage3ScaleValueH2) { RunStage(Stage::kScale, 2, 1, "Stage3ScaleValueH2"); }

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
