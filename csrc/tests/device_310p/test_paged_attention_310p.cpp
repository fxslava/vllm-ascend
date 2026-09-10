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

// Paged attention and the KV cache write on Ascend 310P, fp16.
//
// Mirrors two call sites:
//   torch_npu._npu_reshape_and_cache
//     vllm_ascend/device/device_op.py :: Ascend310PDeviceAdaptor.reshape_and_cache
//   torch_npu._npu_paged_attention
//     vllm_ascend/_310p/attention/attention_v1.py :: forward_paged_attention
//
// The 310P KV cache is not the [2, num_blocks, block_size, num_kv_heads,
// head_size] layout the generic Ascend backend uses. get_kv_cache_shape on the
// 310P backend returns
//     (2, num_blocks, num_kv_heads * head_size / 16, block_size, 16)
// and the runner allocates each half with acl_format=ACL_FORMAT_FRACTAL_NZ. The
// suite is layered so the layout arithmetic, the cache write and the attention
// numerics fail independently of one another.

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

using reference::PagedAttentionShape;
using reference::PagedKvLayout;

// The runner allocates both halves of the KV cache with
// acl_format=ACL_FORMAT_FRACTAL_NZ over the already-decomposed 4-D shape, so
// the descriptors handed to the operator carry the same tag. If a CANN release
// rejects it, ACL_FORMAT_ND is the one thing to try first.
constexpr aclFormat kKvCacheFormat = ACL_FORMAT_FRACTAL_NZ;

// aclnnReshapeAndCache does not exist on CANN 9.1.0 (it is an ATB operator).
// aclnnScatterPaKvCache is the aclnn route to the same paged KV write, and is
// what BaseDeviceAdaptor.reshape_and_cache calls via npu_scatter_pa_kv_cache.
const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

// cacheMode is a mutable char* in the prototype, so it needs a real array.
char kCacheModeNorm[] = "Norm";

std::vector<int64_t> KvCacheDims(const PagedKvLayout& layout) {
  return {layout.num_blocks, layout.fractal_rows(), layout.block_size, shapes::kKvCacheFractalWidth};
}

struct DecodeCase {
  const char* label;
  int64_t num_seqs;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_size;
  int64_t block_size;
  int64_t max_context_len;
};

// Builds a scattered block table plus per-sequence context lengths. Physical
// blocks are drawn from a shuffled pool so that logical order never matches
// physical order, which is what catches an indexing bug that a sequential table
// would hide.
struct PagedLayoutFixture {
  PagedKvLayout layout;
  PagedAttentionShape shape;
  std::vector<int32_t> block_table;
  std::vector<int32_t> context_lens;
  int64_t total_cached_tokens = 0;
  std::vector<int32_t> slot_mapping;  // one slot per cached token, in fill order
};

PagedLayoutFixture BuildPagedLayout(const DecodeCase& test_case, DeterministicRandom* random) {
  PagedLayoutFixture fixture;

  fixture.layout.block_size = test_case.block_size;
  fixture.layout.num_kv_heads = test_case.num_kv_heads;
  fixture.layout.head_size = test_case.head_size;

  const int64_t blocks_per_seq =
      (test_case.max_context_len + test_case.block_size - 1) / test_case.block_size;
  fixture.shape.max_blocks_per_seq = blocks_per_seq;
  // Over-allocate the pool so the shuffle has room to scatter.
  fixture.layout.num_blocks = test_case.num_seqs * blocks_per_seq + 4;

  fixture.shape.num_seqs = test_case.num_seqs;
  fixture.shape.num_heads = test_case.num_heads;
  fixture.shape.num_kv_heads = test_case.num_kv_heads;
  fixture.shape.head_size = test_case.head_size;
  fixture.shape.block_size = test_case.block_size;
  fixture.shape.scale = 1.0f / std::sqrt(static_cast<float>(test_case.head_size));

  const std::vector<int32_t> pool = random->Permutation(static_cast<int32_t>(fixture.layout.num_blocks));

  fixture.block_table.assign(static_cast<size_t>(test_case.num_seqs * blocks_per_seq), 0);
  fixture.context_lens.assign(static_cast<size_t>(test_case.num_seqs), 0);

  int32_t next_block = 0;
  for (int64_t seq = 0; seq < test_case.num_seqs; ++seq) {
    // Vary the context length per sequence so partially filled trailing blocks
    // are exercised alongside exactly-full ones.
    const int64_t context_len =
        (seq == 0) ? test_case.max_context_len
                   : static_cast<int64_t>(random->IntInRange(1, static_cast<int32_t>(test_case.max_context_len)));
    fixture.context_lens[static_cast<size_t>(seq)] = static_cast<int32_t>(context_len);

    for (int64_t block = 0; block < blocks_per_seq; ++block) {
      fixture.block_table[static_cast<size_t>(seq * blocks_per_seq + block)] =
          pool[static_cast<size_t>(next_block++)];
    }

    for (int64_t position = 0; position < context_len; ++position) {
      const int64_t logical_block = position / test_case.block_size;
      const int64_t block_offset = position % test_case.block_size;
      const int32_t physical_block =
          fixture.block_table[static_cast<size_t>(seq * blocks_per_seq + logical_block)];
      fixture.slot_mapping.push_back(
          static_cast<int32_t>(physical_block * test_case.block_size + block_offset));
    }
  }

  fixture.total_cached_tokens = static_cast<int64_t>(fixture.slot_mapping.size());
  return fixture;
}

// -----------------------------------------------------------------------------
// Host-only checks on the 310P cache layout
// -----------------------------------------------------------------------------

TEST(PagedKvLayout, MatchesGetKvCacheShapeOn310P) {
  // AscendAttentionBackend310.get_kv_cache_shape:
  //   (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)
  for (const shapes::AttentionHeads& heads : shapes::GqaConfigurations()) {
    for (int64_t block_size : shapes::Supported310PBlockSizes()) {
      if (!shapes::IsValid310PBlockSize(block_size, heads.head_size)) {
        continue;
      }
      PagedKvLayout layout;
      layout.num_blocks = 8;
      layout.block_size = block_size;
      layout.num_kv_heads = heads.num_kv_heads;
      layout.head_size = heads.head_size;

      EXPECT_EQ(layout.hidden() % shapes::kKvCacheFractalWidth, 0)
          << heads.label << ": num_kv_heads * head_size must be a multiple of 16";
      EXPECT_EQ(layout.fractal_rows(), layout.hidden() / shapes::kKvCacheFractalWidth) << heads.label;
      EXPECT_EQ(layout.ElementCount(),
                static_cast<size_t>(layout.num_blocks * layout.fractal_rows() * block_size *
                                    shapes::kKvCacheFractalWidth))
          << heads.label;
    }
  }
}

TEST(PagedKvLayout, OffsetsAreUniqueAndInBounds) {
  // A collision here would silently make two heads alias in the cache, which is
  // the kind of bug an end-to-end accuracy test reports as "slightly worse".
  PagedKvLayout layout;
  layout.num_blocks = 3;
  layout.block_size = 64;
  layout.num_kv_heads = 4;
  layout.head_size = 128;

  std::set<size_t> seen;
  for (int64_t block = 0; block < layout.num_blocks; ++block) {
    for (int64_t offset = 0; offset < layout.block_size; ++offset) {
      for (int64_t head = 0; head < layout.num_kv_heads; ++head) {
        for (int64_t dim = 0; dim < layout.head_size; ++dim) {
          const size_t index = reference::NzCacheOffset(layout, block, offset, head, dim);
          ASSERT_LT(index, layout.ElementCount())
              << "block=" << block << " offset=" << offset << " head=" << head << " dim=" << dim;
          ASSERT_TRUE(seen.insert(index).second)
              << "duplicate cache offset " << index << " at block=" << block << " offset=" << offset
              << " head=" << head << " dim=" << dim;
        }
      }
    }
  }
  EXPECT_EQ(seen.size(), layout.ElementCount());
}

TEST(PagedKvLayout, ReshapeAndCacheRoundTripsThroughTheReference) {
  PagedKvLayout layout;
  layout.num_blocks = 4;
  layout.block_size = 64;
  layout.num_kv_heads = 2;
  layout.head_size = 128;

  DeterministicRandom random(0x4b564341u);  // "KVCA"
  const int64_t num_tokens = 100;

  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(num_tokens * layout.num_kv_heads * layout.head_size), 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(
      static_cast<size_t>(num_tokens * layout.num_kv_heads * layout.head_size), 0.0f, 1.0f);

  std::vector<int32_t> slot_mapping(static_cast<size_t>(num_tokens));
  const std::vector<int32_t> slots =
      random.Permutation(static_cast<int32_t>(layout.num_blocks * layout.block_size));
  for (int64_t i = 0; i < num_tokens; ++i) {
    slot_mapping[static_cast<size_t>(i)] = slots[static_cast<size_t>(i)];
  }

  std::vector<float> key_cache(layout.ElementCount(), 0.0f);
  std::vector<float> value_cache(layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, slot_mapping, layout, &key_cache, &value_cache);

  for (int64_t token = 0; token < num_tokens; ++token) {
    const int32_t slot = slot_mapping[static_cast<size_t>(token)];
    const int64_t block = slot / layout.block_size;
    const int64_t offset = slot % layout.block_size;
    for (int64_t head = 0; head < layout.num_kv_heads; ++head) {
      for (int64_t dim = 0; dim < layout.head_size; ++dim) {
        const size_t source = static_cast<size_t>((token * layout.num_kv_heads + head) * layout.head_size + dim);
        const size_t index = reference::NzCacheOffset(layout, block, offset, head, dim);
        ASSERT_EQ(key_cache[index], key[source]) << "token=" << token << " head=" << head << " dim=" << dim;
        ASSERT_EQ(value_cache[index], value[source]) << "token=" << token << " head=" << head << " dim=" << dim;
      }
    }
  }
}

TEST(PagedAttentionReference, SingleTokenContextReturnsThatValue) {
  // With one key in the context, softmax is 1 and the output is exactly v.
  PagedKvLayout layout;
  layout.num_blocks = 1;
  layout.block_size = 64;
  layout.num_kv_heads = 1;
  layout.head_size = 64;

  PagedAttentionShape shape;
  shape.num_seqs = 1;
  shape.num_heads = 1;
  shape.num_kv_heads = 1;
  shape.head_size = 64;
  shape.block_size = 64;
  shape.max_blocks_per_seq = 1;
  shape.scale = 1.0f / 8.0f;

  DeterministicRandom random(0x53314b56u);  // "S1KV"
  const std::vector<float> query = random.NormalHalfExact(64, 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(64, 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(64, 0.0f, 1.0f);

  std::vector<float> key_cache(layout.ElementCount(), 0.0f);
  std::vector<float> value_cache(layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, {0}, layout, &key_cache, &value_cache);

  std::vector<float> out;
  reference::PagedAttentionDecode(query, key_cache, value_cache, {0}, {1}, layout, shape, &out);

  ASSERT_EQ(out.size(), 64u);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], value[i], 1e-5f) << "at index " << i;
  }
}

TEST(PagedAttentionReference, IdenticalKeysGiveTheMeanOfTheValues) {
  // Equal scores make softmax uniform, so the output is the arithmetic mean of
  // the value vectors. This pins the softmax normalisation independently of the
  // dot-product path.
  PagedKvLayout layout;
  layout.num_blocks = 1;
  layout.block_size = 64;
  layout.num_kv_heads = 1;
  layout.head_size = 16;

  PagedAttentionShape shape;
  shape.num_seqs = 1;
  shape.num_heads = 1;
  shape.num_kv_heads = 1;
  shape.head_size = 16;
  shape.block_size = 64;
  shape.max_blocks_per_seq = 1;
  shape.scale = 0.25f;

  const int64_t context_len = 8;
  const std::vector<float> query(16, 0.5f);

  DeterministicRandom random(0x4d45414eu);  // "MEAN"
  std::vector<float> key(static_cast<size_t>(context_len * 16), 0.25f);  // every key identical
  const std::vector<float> value = random.NormalHalfExact(static_cast<size_t>(context_len * 16), 0.0f, 1.0f);

  std::vector<int32_t> slot_mapping(static_cast<size_t>(context_len));
  for (int64_t i = 0; i < context_len; ++i) {
    slot_mapping[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }

  std::vector<float> key_cache(layout.ElementCount(), 0.0f);
  std::vector<float> value_cache(layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, slot_mapping, layout, &key_cache, &value_cache);

  std::vector<float> out;
  reference::PagedAttentionDecode(query, key_cache, value_cache, {0}, {static_cast<int32_t>(context_len)}, layout,
                                  shape, &out);

  for (int64_t dim = 0; dim < 16; ++dim) {
    float mean = 0.0f;
    for (int64_t position = 0; position < context_len; ++position) {
      mean += value[static_cast<size_t>(position * 16 + dim)];
    }
    mean /= static_cast<float>(context_len);
    EXPECT_NEAR(out[static_cast<size_t>(dim)], mean, 1e-5f) << "at dim " << dim;
  }
}

TEST(PagedAttentionShapes, BlockSizesRespectThe310PProduct) {
  // block_size * head_size must stay within 128 * 128 or the runner drops down
  // to a smaller block. Confirm the combinations the suite uses are legal.
  for (const shapes::AttentionHeads& heads : shapes::GqaConfigurations()) {
    bool any_legal = false;
    for (int64_t block_size : shapes::Supported310PBlockSizes()) {
      if (shapes::IsValid310PBlockSize(block_size, heads.head_size)) {
        any_legal = true;
      }
    }
    EXPECT_TRUE(any_legal) << heads.label << ": no supported block size fits the 128*128 product limit";
  }
}

// -----------------------------------------------------------------------------
// Device: KV cache write
// -----------------------------------------------------------------------------

class ReshapeAndCache310PTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(ReshapeAndCache310PTest, WritesTheSameBytesAsTheHostScatter) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(ScatterPaKvCacheOp());

  const DecodeCase& test_case = GetParam();
  DeterministicRandom random(0x52414331u);  // "RAC1"

  PagedLayoutFixture fixture = BuildPagedLayout(test_case, &random);
  const int64_t num_tokens = fixture.total_cached_tokens;

  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(num_tokens * test_case.num_kv_heads * test_case.head_size), 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(
      static_cast<size_t>(num_tokens * test_case.num_kv_heads * test_case.head_size), 0.0f, 1.0f);

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor key_device =
      DeviceTensor::Half({num_tokens, test_case.num_kv_heads, test_case.head_size}, key);
  DeviceTensor value_device =
      DeviceTensor::Half({num_tokens, test_case.num_kv_heads, test_case.head_size}, value);
  DeviceTensor slot_device = DeviceTensor::Int32({num_tokens}, fixture.slot_mapping);

  const std::vector<int64_t> cache_dims = KvCacheDims(fixture.layout);
  DeviceTensor key_cache_device = DeviceTensor::HalfEmpty(cache_dims, kKvCacheFormat);
  DeviceTensor value_cache_device = DeviceTensor::HalfEmpty(cache_dims, kKvCacheFormat);

  // Argument order follows the header exactly: key, keyCache, slotMapping,
  // then value, valueCache, then the optional tensors and modes.
  RunAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
      ScatterPaKvCacheOp(), stream, key_device.get(), key_cache_device.get(), slot_device.get(),
      value_device.get(), value_cache_device.get(), static_cast<const aclTensor*>(nullptr),
      static_cast<const aclTensor*>(nullptr), static_cast<const aclTensor*>(nullptr), kCacheModeNorm,
      static_cast<char*>(nullptr), static_cast<const aclIntArray*>(nullptr),
      static_cast<const aclIntArray*>(nullptr));

  std::vector<float> expected_key_cache(fixture.layout.ElementCount(), 0.0f);
  std::vector<float> expected_value_cache(fixture.layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, fixture.slot_mapping, fixture.layout, &expected_key_cache,
                             &expected_value_cache);

  // The values are already fp16-exact and only copied, so this must match bit
  // for bit; a non-zero tolerance here would hide a layout error.
  const Tolerance exact{0.0, 0.0, "cache write is a pure copy of fp16-exact values"};
  EXPECT_TENSORS_ALLCLOSE(key_cache_device.ToFloatFromHalf(), expected_key_cache, exact);
  EXPECT_TENSORS_ALLCLOSE(value_cache_device.ToFloatFromHalf(), expected_value_cache, exact);
}

// -----------------------------------------------------------------------------
// Device: paged attention decode
// -----------------------------------------------------------------------------

class PagedAttention310PTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(PagedAttention310PTest, MatchesCpuReference) {
  // aclnnPagedAttention does not exist on CANN 9.1.0. torch_npu._npu_paged_attention
  // is an ATB operator (atb::PagedAttentionOperation in libatb.so), which is a
  // C++ object API the aclnn RunAclnn path cannot drive. The aclnn alternative,
  // aclnnIncreFlashAttentionV4, is declared and verified in aclnn_ops.hpp but
  // expects a different paged KV layout from the 310P 5-D NZ cache, so wiring it
  // here would mean guessing that layout.
  //
  // The NZ layout arithmetic and the attention reference this test would compare
  // against are covered by the host-only PagedKvLayout and PagedAttentionReference
  // suites above, which do run.
  GTEST_SKIP() << "aclnnPagedAttention is not provided by CANN 9.1.0; "
                  "torch_npu._npu_paged_attention is backed by ATB (libatb.so). "
                  "See csrc/tests/common/aclnn_ops.hpp for the verified "
                  "aclnnIncreFlashAttentionV4 prototype and what it would take.";
}

TEST_P(PagedAttention310PTest, AttendsOnlyWithinTheContextLength) {
  // aclnnPagedAttention does not exist on CANN 9.1.0. torch_npu._npu_paged_attention
  // is an ATB operator (atb::PagedAttentionOperation in libatb.so), which is a
  // C++ object API the aclnn RunAclnn path cannot drive. The aclnn alternative,
  // aclnnIncreFlashAttentionV4, is declared and verified in aclnn_ops.hpp but
  // expects a different paged KV layout from the 310P 5-D NZ cache, so wiring it
  // here would mean guessing that layout.
  //
  // The NZ layout arithmetic and the attention reference this test would compare
  // against are covered by the host-only PagedKvLayout and PagedAttentionReference
  // suites above, which do run.
  GTEST_SKIP() << "aclnnPagedAttention is not provided by CANN 9.1.0; "
                  "torch_npu._npu_paged_attention is backed by ATB (libatb.so). "
                  "See csrc/tests/common/aclnn_ops.hpp for the verified "
                  "aclnnIncreFlashAttentionV4 prototype and what it would take.";
}

std::string DecodeTestName(const ::testing::TestParamInfo<DecodeCase>& info) { return info.param.label; }

// Head counts follow the Qwen3.5 GQA splits; block sizes are the two the 310P
// kernel supports. max_context_len values straddle block boundaries so both
// exactly-full and partially-filled trailing blocks appear.
const DecodeCase kDecodeCases[] = {
    DecodeCase{"seq1_h28_kv4_d128_b64_ctx64", 1, 28, 4, 128, 64, 64},
    DecodeCase{"seq4_h28_kv4_d128_b64_ctx200", 4, 28, 4, 128, 64, 200},
    DecodeCase{"seq4_h32_kv8_d128_b64_ctx512", 4, 32, 8, 128, 64, 512},
    DecodeCase{"seq2_h32_kv8_d128_b128_ctx512", 2, 32, 8, 128, 128, 512},
    DecodeCase{"seq8_h16_kv2_d64_b64_ctx63", 8, 16, 2, 64, 64, 63},
    DecodeCase{"seq2_h16_kv2_d64_b128_ctx300", 2, 16, 2, 64, 128, 300},
    DecodeCase{"seq3_h8_kv8_d128_b64_ctx128", 3, 8, 8, 128, 64, 128},
};

INSTANTIATE_TEST_SUITE_P(Qwen35, ReshapeAndCache310PTest, ::testing::ValuesIn(kDecodeCases), DecodeTestName);
INSTANTIATE_TEST_SUITE_P(Qwen35, PagedAttention310PTest, ::testing::ValuesIn(kDecodeCases), DecodeTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
