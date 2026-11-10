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

// Shared test scaffolding: the failure harness, bit-level float helpers, guarded host buffers
// and the RFC 3.1 strip index maps.
//
// Self-contained C++17. No CANN, no CCE, no device types, so every CPU gate and every host leg
// of a device runner includes this one header rather than keeping a copy of the arithmetic.

#ifndef TURBOQUANT_TEST_COMMON_HPP
#define TURBOQUANT_TEST_COMMON_HPP

#include "csrc/attention/turboquant/op_kernel/common/turboquant_layout.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tq_test {

namespace tq = vllm_ascend::turboquant;

// R, the strip height of the RFC 3.1 layout, taken from the production contract rather than
// restated: a test that carried its own 16 could agree with a kernel that had moved.
constexpr int kStripHeight = static_cast<int>(tq::kStripTokens);
constexpr int kPlanes = 2;  // 0 = key, 1 = value
constexpr int kNibblesPerByte = static_cast<int>(tq::kPackFactor);
constexpr int kNibbleMin = -8;
constexpr int kNibbleMax = 7;

// ------------------------------------------------------------------ the failure harness
// Two idioms, deliberately different. `require` throws, for a precondition whose failure makes
// the rest of the run meaningless (a bad fixture, a refused launch). `check` counts, for an
// assertion whose failure should still let the remaining assertions report -- a gate that stops
// at the first wrong nibble tells you far less than one that says how many are wrong and where.
inline void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FIXTURE/PRECONDITION FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

struct Failures {
    int count = 0;
    void check(bool ok, const char* what)
    {
        if (!ok) {
            std::fprintf(stderr, "  FAIL %s\n", what);
            ++count;
        }
    }
    bool clean() const { return count == 0; }
};

// ------------------------------------------------------------------ bit-level helpers
inline std::uint32_t float_bits(float value)
{
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

inline float bits_float(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline float bf16_to_float(std::uint16_t raw)
{
    return bits_float(static_cast<std::uint32_t>(raw) << 16);
}

inline std::uint16_t float_to_bf16_rne(float value)
{
    const std::uint32_t bits = float_bits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu))
        return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// A BF16 significand is 8 bits, so every integer of magnitude <= 256 is exact. That is the
// property the bit-exact fixtures rest on, and it is checked rather than assumed.
inline bool bf16_is_exact(float value)
{
    return bf16_to_float(float_to_bf16_rne(value)) == value;
}

// An integer as a BF16 bit pattern, refusing anything the narrowing would round.
inline std::uint16_t exact_bf16(int value)
{
    const float f = static_cast<float>(value);
    require((float_bits(f) & 0xFFFFu) == 0u, "activation is not exact in BF16");
    return static_cast<std::uint16_t>(float_bits(f) >> 16);
}

inline float half_to_float(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        if (mantissa == 0) return bits_float(sign);
        // Subnormal: renormalise into the FP32 exponent range.
        exponent = 1;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x3FFu;
        return bits_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
    }
    if (exponent == 0x1Fu) return bits_float(sign | 0x7F800000u | (mantissa << 13));
    return bits_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

inline std::uint16_t float_to_half_rne(float value)
{
    const std::uint32_t bits = float_bits(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127;
    const std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent == 128) {  // inf / nan
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (exponent < -24) return static_cast<std::uint16_t>(sign);
    if (exponent < -14) {
        // Subnormal half: shift the implicit bit in and round half to even.
        const std::uint32_t shift = static_cast<std::uint32_t>(-exponent - 14);
        const std::uint32_t full = mantissa | 0x800000u;
        const std::uint32_t half_bits = full >> (shift + 13);
        const std::uint32_t rest = full & ((1u << (shift + 13)) - 1u);
        const std::uint32_t tie = 1u << (shift + 12);
        std::uint32_t out = half_bits;
        if (rest > tie || (rest == tie && (out & 1u))) ++out;
        return static_cast<std::uint16_t>(sign | out);
    }
    const std::uint32_t biased = static_cast<std::uint32_t>(exponent + 15) << 10;
    const std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t rest = mantissa & 0x1FFFu;
    std::uint32_t out = biased | half_mantissa;
    if (rest > 0x1000u || (rest == 0x1000u && (out & 1u))) ++out;
    return static_cast<std::uint16_t>(sign | out);
}

// ------------------------------------------------------------------ guarded host buffers
// A host allocation with a canary band on each side, for the CPU gates that write through a
// pointer the way a kernel would. The device-side equivalent is in the NPU runners, which guard
// Global Memory the same way.
constexpr std::size_t kCanaryBytes = 64;
constexpr std::uint32_t kCanaryWord = 0xDEADBEEFu;

template <class T, std::size_t Alignment = 64>
class GuardedBuffer {
    static_assert(Alignment >= kCanaryBytes && (Alignment & (Alignment - 1)) == 0);
    static_assert(Alignment >= alignof(T));
    static_assert(std::is_trivially_copyable<T>::value);

public:
    explicit GuardedBuffer(std::size_t count) : count_(count)
    {
        const auto bytes = count_ * sizeof(T);
        raw_ = static_cast<std::byte*>(
            ::operator new(Alignment + bytes + kCanaryBytes, std::align_val_t{Alignment}));
        data_ = reinterpret_cast<T*>(raw_ + Alignment);
        std::uninitialized_default_construct_n(data_, count_);
        fill_canary(raw_ + Alignment - kCanaryBytes);
        fill_canary(reinterpret_cast<std::byte*>(data_) + bytes);
    }
    ~GuardedBuffer()
    {
        std::destroy_n(data_, count_);
        ::operator delete(raw_, std::align_val_t{Alignment});
    }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
    T& operator[](std::size_t i)
    {
        require(i < count_, "guarded buffer index out of range");
        return data_[i];
    }
    const T& operator[](std::size_t i) const
    {
        require(i < count_, "guarded buffer index out of range");
        return data_[i];
    }
    bool canaries_intact() const
    {
        return check_canary(raw_ + Alignment - kCanaryBytes) &&
               check_canary(reinterpret_cast<const std::byte*>(data_) + count_ * sizeof(T));
    }

private:
    static void fill_canary(std::byte* at)
    {
        for (std::size_t i = 0; i < kCanaryBytes; i += sizeof(kCanaryWord))
            std::memcpy(at + i, &kCanaryWord, sizeof(kCanaryWord));
    }
    static bool check_canary(const std::byte* at)
    {
        for (std::size_t i = 0; i < kCanaryBytes; i += sizeof(kCanaryWord)) {
            std::uint32_t word = 0;
            std::memcpy(&word, at + i, sizeof(word));
            if (word != kCanaryWord) return false;
        }
        return true;
    }
    std::size_t count_;
    std::byte* raw_ = nullptr;
    T* data_ = nullptr;
};

// ------------------------------------------------------------------ the RFC 3.1 strip layout
// The physical geometry of one paged 4-bit cache allocation. `block_size` is S, the page height
// in tokens, and must be a whole number of strips.
struct Shape {
    int num_requests = 1;
    int num_tokens = 16;
    int num_kv_heads = 4;
    int head_size = 256;
    int block_size = 16;  // S
    int num_blocks = 8;   // P

    int strips_per_block() const { return block_size / kStripHeight; }
    int strips() const { return num_tokens / kStripHeight; }
    int head_bytes() const { return block_size * head_size / kNibblesPerByte; }
    int page_bytes() const { return num_kv_heads * head_bytes(); }
    std::size_t strip_bytes() const
    {
        return static_cast<std::size_t>(kStripHeight) * head_size / kNibblesPerByte;
    }
    std::size_t cache_bytes() const
    {
        return static_cast<std::size_t>(num_blocks) * page_bytes();
    }
    std::size_t scale_floats() const
    {
        return static_cast<std::size_t>(kPlanes) * num_blocks * num_kv_heads * block_size;
    }
    // invSqrtLen folds into the stored scale, never into the quantised value: scaling Y and
    // absmax together leaves every nibble unchanged.
    float inv_sqrt_len() const { return 1.0f / std::sqrt(static_cast<float>(head_size)); }
    bool valid() const
    {
        return num_tokens > 0 && num_kv_heads > 0 && num_blocks > 0 && block_size > 0 &&
               head_size % static_cast<int>(tq::kOperandC0) == 0 &&
               block_size % kStripHeight == 0 && num_tokens % kStripHeight == 0 &&
               (block_size * head_size) % kNibblesPerByte == 0;
    }
};

// key nibble = ((block*H + head)*S + token)*d + channel
inline std::size_t key_nibble(const Shape& s, int block, int head, int token, int channel)
{
    return (static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size + token) *
               s.head_size +
           channel;
}

// value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
inline std::size_t value_nibble(const Shape& s, int block, int head, int token, int channel)
{
    const std::size_t page =
        static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size * s.head_size;
    return page +
           static_cast<std::size_t>(token / kStripHeight) *
               (static_cast<std::size_t>(kStripHeight) * s.head_size) +
           static_cast<std::size_t>(channel) * kStripHeight + (token % kStripHeight);
}

inline std::size_t strip_nibble(const Shape& s, int plane, int block, int head, int token,
                                int channel)
{
    return plane == 0 ? key_nibble(s, block, head, token, channel)
                      : value_nibble(s, block, head, token, channel);
}

// scale index = ((kv*P + block)*H + head)*S + token
inline std::size_t scale_index(const Shape& s, int plane, int block, int head, int token)
{
    return ((static_cast<std::size_t>(plane) * s.num_blocks + block) * s.num_kv_heads + head) *
               s.block_size +
           token;
}

// The byte offset of the (block, head, strip) read-modify-write window the scatter writer owns:
// kStripHeight * head_size / 2 contiguous bytes.
inline std::size_t strip_byte_offset(const Shape& s, int block, int head, int strip)
{
    return (static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size * s.head_size +
            static_cast<std::size_t>(strip) * kStripHeight * s.head_size) /
           kNibblesPerByte;
}

// ------------------------------------------------------------------ nibble access
inline void write_nibble(std::int8_t* cache, std::size_t nibble, int value)
{
    const std::size_t byte = nibble / 2;
    const auto nib = static_cast<std::uint8_t>(value & 0xF);
    auto raw = static_cast<std::uint8_t>(cache[byte]);
    if (nibble % 2 == 0)
        raw = static_cast<std::uint8_t>((raw & 0xF0u) | nib);
    else
        raw = static_cast<std::uint8_t>((raw & 0x0Fu) | static_cast<std::uint8_t>(nib << 4));
    cache[byte] = static_cast<std::int8_t>(raw);
}

// Sign-extended from 4 bits: the packing is signed INT4 in [-8, 7].
inline int read_nibble(const std::int8_t* cache, std::size_t nibble)
{
    const auto raw = static_cast<std::uint8_t>(cache[nibble / 2]);
    const int nib = (nibble % 2 == 0) ? (raw & 0x0F) : (raw >> 4);
    return nib >= 8 ? nib - 16 : nib;
}

}  // namespace tq_test

#endif  // TURBOQUANT_TEST_COMMON_HPP
