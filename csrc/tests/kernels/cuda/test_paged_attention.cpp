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

// Paged KV cache scatter and decode attention on the native CUDA C backend.
//
// Worth stating plainly: the decode half of test_paged_attention_310p.cpp is
// permanently skipped, because CANN 9.1.0 exposes no aclnn paged attention and
// torch_npu._npu_paged_attention is an ATB C++ object API the aclnn path cannot
// drive. Until that changes, this file is the only place in the suite where
// reference::PagedAttentionDecode is checked against a real kernel rather than
// against itself.
//
// The cache here is the dense 4-D GPU layout
// [num_blocks, num_kv_heads, block_size, head_dim], selected on the reference
// side with PagedKvCacheLayout::kDense4D, so the scatter and the decode are
// compared against exactly the same two host functions the Ascend tests use.

#include <gtest/gtest.h>

#include <cmath>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "cpu_reference.hpp"
#include "cuda_device_tensor.hpp"
#include "cuda_kernels.hpp"
#include "cuda_runtime.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"

namespace vllm_ascend {
namespace test {
namespace {

using reference::PagedAttentionShape;
using reference::PagedKvCacheLayout;
using reference::PagedKvLayout;

struct DecodeCase {
  const char* label;
  int64_t num_seqs;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_size;
  int64_t block_size;
  int64_t max_context_len;
};

// Without a printer GTest describes a parameter struct as a raw byte dump,
// which `--gtest_list_tests` appends to every case name and ctest then carries
// into its own test names. The label is the whole shape, so print that.
void PrintTo(const DecodeCase& test_case, std::ostream* stream) { *stream << test_case.label; }

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

  fixture.layout.kind = PagedKvCacheLayout::kDense4D;
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
        (seq == 0)
            ? test_case.max_context_len
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

std::vector<int64_t> KvCacheDims(const PagedKvLayout& layout) {
  return {layout.num_blocks, layout.num_kv_heads, layout.block_size, layout.head_size};
}

// -----------------------------------------------------------------------------
// Host-only checks on the dense cache layout
// -----------------------------------------------------------------------------

TEST(DensePagedKvLayout, OffsetsAreUniqueAndInBounds) {
  // A collision here would silently make two heads alias in the cache, which is
  // the kind of bug an end-to-end accuracy test reports as "slightly worse".
  PagedKvLayout layout;
  layout.kind = PagedKvCacheLayout::kDense4D;
  layout.num_blocks = 3;
  layout.block_size = 64;
  layout.num_kv_heads = 4;
  layout.head_size = 128;

  std::set<size_t> seen;
  for (int64_t block = 0; block < layout.num_blocks; ++block) {
    for (int64_t offset = 0; offset < layout.block_size; ++offset) {
      for (int64_t head = 0; head < layout.num_kv_heads; ++head) {
        for (int64_t dim = 0; dim < layout.head_size; ++dim) {
          const size_t index = reference::CacheOffset(layout, block, offset, head, dim);
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

TEST(DensePagedKvLayout, IsRowMajorInTheTrailingHeadDimension) {
  // The whole point of the dense layout over the 310P fractal one is that a
  // (block, head, position) row is contiguous, which is what lets the scatter
  // and the decode kernels read a head vector as one coalesced run.
  PagedKvLayout layout;
  layout.kind = PagedKvCacheLayout::kDense4D;
  layout.num_blocks = 2;
  layout.block_size = 16;
  layout.num_kv_heads = 3;
  layout.head_size = 64;

  const size_t base = reference::CacheOffset(layout, 1, 5, 2, 0);
  for (int64_t dim = 0; dim < layout.head_size; ++dim) {
    EXPECT_EQ(reference::CacheOffset(layout, 1, 5, 2, dim), base + static_cast<size_t>(dim))
        << "dim=" << dim;
  }
}

TEST(DensePagedAttentionReference, SingleTokenContextReturnsThatValue) {
  // With one cached position the softmax is a single weight of 1, so the output
  // is exactly the cached value regardless of the query.
  PagedKvLayout layout;
  layout.kind = PagedKvCacheLayout::kDense4D;
  layout.num_blocks = 2;
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
  shape.scale = 0.125f;

  DeterministicRandom random(0x53474c31u);  // "SGL1"
  const std::vector<float> query = random.NormalHalfExact(64, 0.0f, 1.0f);
  const std::vector<float> cached_value = random.NormalHalfExact(64, 0.0f, 1.0f);

  std::vector<float> key_cache(layout.ElementCount(), 0.0f);
  std::vector<float> value_cache(layout.ElementCount(), 0.0f);
  for (int64_t dim = 0; dim < 64; ++dim) {
    value_cache[reference::CacheOffset(layout, 1, 0, 0, dim)] = cached_value[static_cast<size_t>(dim)];
  }

  const std::vector<int32_t> block_table = {1};
  const std::vector<int32_t> context_lens = {1};

  std::vector<float> out;
  reference::PagedAttentionDecode(query, key_cache, value_cache, block_table, context_lens, layout, shape,
                                  &out);

  for (int64_t dim = 0; dim < 64; ++dim) {
    EXPECT_NEAR(out[static_cast<size_t>(dim)], cached_value[static_cast<size_t>(dim)], 1e-6f)
        << "dim=" << dim;
  }
}

// -----------------------------------------------------------------------------
// Device: KV cache scatter
// -----------------------------------------------------------------------------

class PagedCacheScatterCudaTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(PagedCacheScatterCudaTest, WritesTheSameBytesAsTheHostScatter) {
  REQUIRE_CUDA_DEVICE();

  const DecodeCase& test_case = GetParam();
  DeterministicRandom random(0x53434154u);  // "SCAT"
  const PagedLayoutFixture fixture = BuildPagedLayout(test_case, &random);

  const size_t token_elements = static_cast<size_t>(fixture.total_cached_tokens *
                                                    test_case.num_kv_heads * test_case.head_size);
  const std::vector<float> key = random.NormalHalfExact(token_elements, 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(token_elements, 0.0f, 1.0f);

  cudaStream_t stream = CUDATestEnvironment::Instance().stream();

  CudaDeviceTensor key_device =
      CudaDeviceTensor::Half({fixture.total_cached_tokens, test_case.num_kv_heads, test_case.head_size}, key);
  CudaDeviceTensor value_device = CudaDeviceTensor::Half(
      {fixture.total_cached_tokens, test_case.num_kv_heads, test_case.head_size}, value);
  CudaDeviceTensor slot_device =
      CudaDeviceTensor::Int32({fixture.total_cached_tokens}, fixture.slot_mapping);

  // HalfEmpty zeroes the allocation, which matches the zero-initialised host
  // caches below; untouched slots therefore have to agree as well.
  CudaDeviceTensor key_cache_device = CudaDeviceTensor::HalfEmpty(KvCacheDims(fixture.layout));
  CudaDeviceTensor value_cache_device = CudaDeviceTensor::HalfEmpty(KvCacheDims(fixture.layout));

  cuda::LaunchPagedCacheScatterHalf(key_device.half_data(), value_device.half_data(),
                                    slot_device.int32_data(), key_cache_device.half_data(),
                                    value_cache_device.half_data(), fixture.total_cached_tokens,
                                    test_case.num_kv_heads, test_case.head_size, test_case.block_size,
                                    stream);
  CUDATestEnvironment::Instance().device().SynchronizeStream();

  std::vector<float> expected_key_cache(fixture.layout.ElementCount(), 0.0f);
  std::vector<float> expected_value_cache(fixture.layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, fixture.slot_mapping, fixture.layout, &expected_key_cache,
                             &expected_value_cache);

  // The scatter moves fp16 bits without converting, so this is an equality
  // check, not an approximate one.
  const Tolerance exact{0.0, 0.0, "the scatter is a pure fp16 move, so it must be bit-exact"};
  EXPECT_TENSORS_ALLCLOSE(key_cache_device.ToFloatFromHalf(), expected_key_cache, exact);
  EXPECT_TENSORS_ALLCLOSE(value_cache_device.ToFloatFromHalf(), expected_value_cache, exact);
}

TEST_P(PagedCacheScatterCudaTest, SkipsTokensWithANegativeSlot) {
  REQUIRE_CUDA_DEVICE();

  // vLLM marks padded tokens with a negative slot. Writing them would corrupt
  // block 0, which is a live block for some other sequence.
  const DecodeCase& test_case = GetParam();
  DeterministicRandom random(0x50414421u);  // "PAD!"
  PagedLayoutFixture fixture = BuildPagedLayout(test_case, &random);

  // Mark every other token as padding.
  std::vector<int32_t> slot_mapping = fixture.slot_mapping;
  for (size_t i = 1; i < slot_mapping.size(); i += 2) {
    slot_mapping[i] = -1;
  }

  const size_t token_elements = static_cast<size_t>(fixture.total_cached_tokens *
                                                    test_case.num_kv_heads * test_case.head_size);
  const std::vector<float> key = random.NormalHalfExact(token_elements, 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(token_elements, 0.0f, 1.0f);

  cudaStream_t stream = CUDATestEnvironment::Instance().stream();

  CudaDeviceTensor key_device =
      CudaDeviceTensor::Half({fixture.total_cached_tokens, test_case.num_kv_heads, test_case.head_size}, key);
  CudaDeviceTensor value_device = CudaDeviceTensor::Half(
      {fixture.total_cached_tokens, test_case.num_kv_heads, test_case.head_size}, value);
  CudaDeviceTensor slot_device = CudaDeviceTensor::Int32({fixture.total_cached_tokens}, slot_mapping);

  CudaDeviceTensor key_cache_device = CudaDeviceTensor::HalfEmpty(KvCacheDims(fixture.layout));
  CudaDeviceTensor value_cache_device = CudaDeviceTensor::HalfEmpty(KvCacheDims(fixture.layout));

  cuda::LaunchPagedCacheScatterHalf(key_device.half_data(), value_device.half_data(),
                                    slot_device.int32_data(), key_cache_device.half_data(),
                                    value_cache_device.half_data(), fixture.total_cached_tokens,
                                    test_case.num_kv_heads, test_case.head_size, test_case.block_size,
                                    stream);
  CUDATestEnvironment::Instance().device().SynchronizeStream();

  std::vector<float> expected_key_cache(fixture.layout.ElementCount(), 0.0f);
  std::vector<float> expected_value_cache(fixture.layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, slot_mapping, fixture.layout, &expected_key_cache,
                             &expected_value_cache);

  const Tolerance exact{0.0, 0.0, "the scatter is a pure fp16 move, so it must be bit-exact"};
  EXPECT_TENSORS_ALLCLOSE(key_cache_device.ToFloatFromHalf(), expected_key_cache, exact);
  EXPECT_TENSORS_ALLCLOSE(value_cache_device.ToFloatFromHalf(), expected_value_cache, exact);
}

// -----------------------------------------------------------------------------
// Device: paged attention decode
// -----------------------------------------------------------------------------

// Everything a decode launch needs, with the caches already filled on the host
// so the scatter kernel is not on the critical path of an attention failure.
struct DecodeFixture {
  PagedLayoutFixture layout;
  std::vector<float> query;
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  int64_t max_context_len = 0;
};

DecodeFixture BuildDecodeFixture(const DecodeCase& test_case, uint32_t seed) {
  DecodeFixture fixture;
  DeterministicRandom random(seed);
  fixture.layout = BuildPagedLayout(test_case, &random);

  fixture.query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_seqs * test_case.num_heads * test_case.head_size), 0.0f, 1.0f);

  const size_t token_elements = static_cast<size_t>(fixture.layout.total_cached_tokens *
                                                    test_case.num_kv_heads * test_case.head_size);
  const std::vector<float> key = random.NormalHalfExact(token_elements, 0.0f, 1.0f);
  const std::vector<float> value = random.NormalHalfExact(token_elements, 0.0f, 1.0f);

  fixture.key_cache.assign(fixture.layout.layout.ElementCount(), 0.0f);
  fixture.value_cache.assign(fixture.layout.layout.ElementCount(), 0.0f);
  reference::ReshapeAndCache(key, value, fixture.layout.slot_mapping, fixture.layout.layout,
                             &fixture.key_cache, &fixture.value_cache);

  for (int32_t context_len : fixture.layout.context_lens) {
    if (context_len > fixture.max_context_len) {
      fixture.max_context_len = context_len;
    }
  }
  return fixture;
}

std::vector<float> RunDecodeOnDevice(const DecodeCase& test_case, const DecodeFixture& fixture,
                                     const std::vector<float>& key_cache,
                                     const std::vector<float>& value_cache) {
  cudaStream_t stream = CUDATestEnvironment::Instance().stream();
  const PagedKvLayout& layout = fixture.layout.layout;

  CudaDeviceTensor query_device = CudaDeviceTensor::Half(
      {test_case.num_seqs, test_case.num_heads, test_case.head_size}, fixture.query);
  CudaDeviceTensor key_cache_device = CudaDeviceTensor::Half(KvCacheDims(layout), key_cache);
  CudaDeviceTensor value_cache_device = CudaDeviceTensor::Half(KvCacheDims(layout), value_cache);
  CudaDeviceTensor block_table_device = CudaDeviceTensor::Int32(
      {test_case.num_seqs, fixture.layout.shape.max_blocks_per_seq}, fixture.layout.block_table);
  CudaDeviceTensor context_lens_device =
      CudaDeviceTensor::Int32({test_case.num_seqs}, fixture.layout.context_lens);
  CudaDeviceTensor out_device =
      CudaDeviceTensor::HalfEmpty({test_case.num_seqs, test_case.num_heads, test_case.head_size});

  cuda::LaunchPagedAttentionDecodeV1Half(
      query_device.half_data(), key_cache_device.half_data(), value_cache_device.half_data(),
      block_table_device.int32_data(), context_lens_device.int32_data(), out_device.half_data(),
      test_case.num_seqs, test_case.num_heads, test_case.num_kv_heads, test_case.head_size,
      test_case.block_size, fixture.layout.shape.max_blocks_per_seq, fixture.max_context_len,
      fixture.layout.shape.scale, stream);
  CUDATestEnvironment::Instance().device().SynchronizeStream();

  return out_device.ToFloatFromHalf();
}

class PagedAttentionCudaTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(PagedAttentionCudaTest, MatchesCpuReference) {
  REQUIRE_CUDA_DEVICE();

  const DecodeCase& test_case = GetParam();
  const DecodeFixture fixture = BuildDecodeFixture(test_case, 0x50414744u);  // "PAGD"

  const std::vector<float> actual =
      RunDecodeOnDevice(test_case, fixture, fixture.key_cache, fixture.value_cache);

  std::vector<float> expected;
  reference::PagedAttentionDecode(fixture.query, fixture.key_cache, fixture.value_cache,
                                  fixture.layout.block_table, fixture.layout.context_lens,
                                  fixture.layout.layout, fixture.layout.shape, &expected);

  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kPagedAttentionTolerance);
}

TEST_P(PagedAttentionCudaTest, AttendsOnlyWithinTheContextLength) {
  REQUIRE_CUDA_DEVICE();

  // Positions past a sequence's context length belong to no one yet. Filling
  // them with a large value would dominate the softmax if the kernel read them,
  // so an unchanged output is direct evidence that it does not.
  const DecodeCase& test_case = GetParam();
  const DecodeFixture fixture = BuildDecodeFixture(test_case, 0x43545820u);  // "CTX "

  const std::vector<float> baseline =
      RunDecodeOnDevice(test_case, fixture, fixture.key_cache, fixture.value_cache);

  std::vector<float> poisoned_key_cache = fixture.key_cache;
  std::vector<float> poisoned_value_cache = fixture.value_cache;

  const PagedKvLayout& layout = fixture.layout.layout;
  const int64_t blocks_per_seq = fixture.layout.shape.max_blocks_per_seq;

  for (int64_t seq = 0; seq < test_case.num_seqs; ++seq) {
    const int64_t context_len = fixture.layout.context_lens[static_cast<size_t>(seq)];
    for (int64_t position = context_len; position < blocks_per_seq * test_case.block_size; ++position) {
      const int64_t logical_block = position / test_case.block_size;
      const int64_t block_offset = position % test_case.block_size;
      const int32_t physical_block =
          fixture.layout.block_table[static_cast<size_t>(seq * blocks_per_seq + logical_block)];

      for (int64_t head = 0; head < test_case.num_kv_heads; ++head) {
        for (int64_t dim = 0; dim < test_case.head_size; ++dim) {
          const size_t index = reference::CacheOffset(layout, physical_block, block_offset, head, dim);
          // Well inside fp16 range, and large enough that a single leaked
          // position would move the output by far more than the tolerance.
          poisoned_key_cache[index] = 40.0f;
          poisoned_value_cache[index] = -40.0f;
        }
      }
    }
  }

  const std::vector<float> poisoned =
      RunDecodeOnDevice(test_case, fixture, poisoned_key_cache, poisoned_value_cache);

  const Tolerance exact{0.0, 0.0, "reading out of context changes nothing, so the outputs must match bit for bit"};
  EXPECT_TENSORS_ALLCLOSE(poisoned, baseline, exact);
}

std::string DecodeTestName(const ::testing::TestParamInfo<DecodeCase>& info) { return info.param.label; }

// Head counts follow the Qwen3.5 GQA splits. Block sizes are the two the Ascend
// 310P kernel supports, kept here so the two backends cover the same paging
// granularities even though CUDA imposes no such limit. max_context_len values
// straddle block boundaries so both exactly-full and partially-filled trailing
// blocks appear.
const DecodeCase kDecodeCases[] = {
    DecodeCase{"seq1_h28_kv4_d128_b64_ctx64", 1, 28, 4, 128, 64, 64},
    DecodeCase{"seq4_h28_kv4_d128_b64_ctx200", 4, 28, 4, 128, 64, 200},
    DecodeCase{"seq4_h32_kv8_d128_b64_ctx512", 4, 32, 8, 128, 64, 512},
    DecodeCase{"seq2_h32_kv8_d128_b128_ctx512", 2, 32, 8, 128, 128, 512},
    DecodeCase{"seq8_h16_kv2_d64_b64_ctx63", 8, 16, 2, 64, 64, 63},
    DecodeCase{"seq2_h16_kv2_d64_b128_ctx300", 2, 16, 2, 64, 128, 300},
    DecodeCase{"seq3_h8_kv8_d128_b64_ctx128", 3, 8, 8, 128, 64, 128},
};

INSTANTIATE_TEST_SUITE_P(Qwen35, PagedCacheScatterCudaTest, ::testing::ValuesIn(kDecodeCases),
                         DecodeTestName);
INSTANTIATE_TEST_SUITE_P(Qwen35, PagedAttentionCudaTest, ::testing::ValuesIn(kDecodeCases), DecodeTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
