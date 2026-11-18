/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

// Compile the production NZ ingress geometry without CANN or a device. Small
// cache fixtures cannot detect offsets wrapping into another physical page.
#define TURBOQUANT_INGRESS_NZ 1
#include "csrc/attention/turboquant/op_kernel/turboquant_reshape_batch.cpp"

#include <cstdio>
#include <limits>

int main() {
  namespace bf = turboquant_bf16;
  namespace tq = vllm_ascend::turboquant;
  constexpr std::uint64_t byte_boundary = std::uint64_t{1} << 32;
  constexpr std::uint32_t first_wrapping_slot = byte_boundary / (bf::kHeads * bf::kNzRowBytes);
  constexpr std::uint32_t slots[] = {
      first_wrapping_slot - bf::kTokens,
      first_wrapping_slot,
      first_wrapping_slot + bf::kTokens,
      2 * first_wrapping_slot + 3 * bf::kTokens,
      // Also cross the 32-bit scale-element boundary at eight floats/slot.
      (std::uint32_t{1} << 29) + bf::kTokens,
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - bf::kTokens + 1,
  };
  for (const auto slot : slots) {
    constexpr std::uint32_t block_size = 128;
    const auto address =
        bf::CalculateStripOffsets(slot, bf::kHeads, bf::kNzHeadBytes, block_size, slot / block_size + 1);
    for (std::uint32_t head = 0; head < bf::kHeads; ++head) {
      for (std::uint32_t kv = 0; kv < 2; ++kv) {
        for (std::uint32_t row = 0; row < bf::kTokens; ++row) {
          const auto scale = (kv == 0 ? address.key_scale : address.value_scale) + head + row * bf::kNzScaleSlot;
          if (scale != tq::NzTiledScaleIndex(slot + row, head, kv, bf::kHeads)) {
            std::printf("FAIL: scale address wrapped at slot %u\n", slot);
            return 1;
          }
        }
      }
      for (std::uint32_t group = 0; group < bf::kNzGroups; ++group) {
        for (std::uint32_t row = 0; row < bf::kTokens; ++row) {
          const auto actual =
              address.page_bytes + head * bf::kNzHeadBytes + group * bf::kNzGroupBytes + row * bf::kNzC0;
          const auto expected = tq::NzTiledPackedByte(slot + row, head, group * bf::kNzC0, bf::kHeads, bf::kNzRowBytes);
          if (actual != expected) {
            std::printf("FAIL: slot %u, head %u, group %u, row %u: writer %llu, reader %llu\n", slot, head, group, row,
                        static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expected));
            return 1;
          }
        }
      }
    }
  }
  std::puts("PASS: prefill addresses agree with decode beyond the 4 GiB boundary");
  return 0;
}
