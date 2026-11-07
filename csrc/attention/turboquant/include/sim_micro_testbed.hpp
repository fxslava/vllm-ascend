#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../routines/routine_cube_hadamard16.hpp"

namespace ascend_micro::sim {

constexpr std::size_t kCanaryBytes = 64;
constexpr std::uint32_t kCanaryWord = 0xDEADBEEF;

inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

inline void assert_aligned(const void* pointer, std::size_t alignment = 32)
{
    require(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0,
            "mock buffer alignment violation");
}

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
        assert_aligned(data_, Alignment);
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
    T& operator[](std::size_t index)
    {
        require(index < count_, "mock buffer index out of range");
        return data_[index];
    }
    const T& operator[](std::size_t index) const
    {
        require(index < count_, "mock buffer index out of range");
        return data_[index];
    }

    bool canaries_intact() const
    {
        const auto* head = raw_ + Alignment - kCanaryBytes;
        const auto* tail = reinterpret_cast<const std::byte*>(data_) + count_ * sizeof(T);
        return check_canary(head) && check_canary(tail);
    }

private:
    static void fill_canary(std::byte* location)
    {
        for (std::size_t i = 0; i < kCanaryBytes; i += sizeof(kCanaryWord))
            std::memcpy(location + i, &kCanaryWord, sizeof(kCanaryWord));
    }
    static bool check_canary(const std::byte* location)
    {
        for (std::size_t i = 0; i < kCanaryBytes; i += sizeof(kCanaryWord)) {
            std::uint32_t word = 0;
            std::memcpy(&word, location + i, sizeof(word));
            if (word != kCanaryWord) return false;
        }
        return true;
    }
    std::size_t count_;
    std::byte* raw_ = nullptr;
    T* data_ = nullptr;
};

#define ASSERT_CANARY_INTACT(mock_tensor) \
    ::ascend_micro::sim::require((mock_tensor).canaries_intact(), \
                                  "canary guard band was overwritten")

inline std::uint32_t float_bits(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bits_float(std::uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline float half_to_float(std::uint16_t h)
{
    const std::uint32_t sign = (std::uint32_t(h & 0x8000) << 16);
    std::uint32_t exponent = (h >> 10) & 31;
    std::uint32_t fraction = h & 1023;
    if (exponent == 0) {
        if (fraction == 0) return bits_float(sign);
        int e = -14;
        while ((fraction & 1024) == 0) { fraction <<= 1; --e; }
        fraction &= 1023;
        return bits_float(sign | (std::uint32_t(e + 127) << 23) | (fraction << 13));
    }
    if (exponent == 31) return bits_float(sign | 0x7f800000u | (fraction << 13));
    return bits_float(sign | ((exponent + 112) << 23) | (fraction << 13));
}

inline std::uint16_t float_to_half_rne(float value)
{
    const auto bits = float_bits(value);
    const std::uint16_t sign = std::uint16_t((bits >> 16) & 0x8000);
    const int exponent = int((bits >> 23) & 255) - 127;
    const std::uint32_t fraction = bits & 0x7fffff;
    if (exponent == 128)
        return std::uint16_t(sign | (fraction ? 0x7e00 : 0x7c00));
    if (exponent > 15) return std::uint16_t(sign | 0x7c00);
    if (exponent < -24) return sign;
    if (exponent < -14) {
        const std::uint32_t mantissa = fraction | 0x800000;
        const int shift = -exponent - 1;
        const auto quotient = mantissa >> shift;
        const auto remainder = mantissa & ((1u << shift) - 1);
        const auto halfway = 1u << (shift - 1);
        const auto rounded = quotient +
            (remainder > halfway || (remainder == halfway && (quotient & 1)));
        return std::uint16_t(sign | rounded);
    }
    const auto rounded = fraction + 0xfff + ((fraction >> 13) & 1);
    return std::uint16_t(sign | ((exponent + 15) << 10) | (rounded >> 13));
}

inline std::uint16_t float_to_bfloat16_rne(float value)
{
    const auto bits = float_bits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu))
        return std::uint16_t((bits >> 16) | 0x40u);
    return std::uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// bf16 is the TOP half of the fp32 word, so widening is a shift and nothing else: no exponent
// rebias, no subnormal loop, and every bf16 value -- subnormals and NaN payloads included -- is
// an exact fp32 value. The asymmetry with half_to_float above is the format's, not an omission.
//
// REFINEMENT_OF_COMBAT_KERNELS.md section 3 asks that a bf16 output be checked at its original
// WORD POSITION, which is why callers compare the device's std::uint16_t against
// float_to_bfloat16_rne(expected) directly and use this only to print what they got: a comparison
// that widened both sides first would accept a halfword swap, since the low half of a correct
// fp32 result is not the value and would not round-trip -- but a comparison that accepted either
// half would. The bit distance between two bf16 words of the same sign IS their ULP distance.
inline float bfloat16_to_float(std::uint16_t b)
{
    return bits_float(std::uint32_t(b) << 16);
}

// Independent scalar oracle: row-major X and canonical Sylvester H16.
inline std::array<float, 256> golden_hadamard16(const std::uint16_t* x)
{
    std::array<float, 256> result{};
    for (unsigned row = 0; row < 16; ++row) {
        for (unsigned col = 0; col < 16; ++col) {
            float sum = 0;
            for (unsigned k = 0; k < 16; ++k) {
                unsigned intersect = k & col;
                unsigned parity = 0;
                while (intersect) { parity ^= (intersect & 1u); intersect >>= 1; }
                const float sign = parity ? -1.0f : 1.0f;
                sum += half_to_float(x[row * 16 + k]) * sign;
            }
            result[row * 16 + col] = sum;
        }
    }
    return result;
}

// Single-fractal NZ is row-major within the 16x16 tile; ZN is column-major.
// MockCubeEngine models L0A x L0B -> L0C and direct Fixpipe NZ -> ND.
struct MockCubeEngine {
    const std::uint16_t* l0a_nz;
    const std::uint16_t* l0b_zn;
    float* l0c_nz;
    void* output_nd;
    unsigned mmads = 0;
    unsigned fixpipes = 0;
    bool pending = false;

    void mmad()
    {
        require(!pending, "L0C is still owned by Fixpipe");
        for (unsigned row = 0; row < 16; ++row)
            for (unsigned col = 0; col < 16; ++col) {
                float sum = 0;
                for (unsigned k = 0; k < 16; ++k)
                    sum += half_to_float(l0a_nz[row * 16 + k]) *
                           half_to_float(l0b_zn[col * 16 + k]);
                l0c_nz[row * 16 + col] = sum;
            }
        ++mmads;
        pending = true;
    }

    template <OutflowMode Mode>
    void fixpipe()
    {
        require(pending, "Fixpipe must follow Mmad");
        for (unsigned i = 0; i < 256; ++i) {
            if constexpr (Mode == OutflowMode::Fp32)
                static_cast<float*>(output_nd)[i] = l0c_nz[i];
            else if constexpr (Mode == OutflowMode::Fp16)
                static_cast<std::uint16_t*>(output_nd)[i] = float_to_half_rne(l0c_nz[i]);
            else
                static_cast<std::uint16_t*>(output_nd)[i] = float_to_bfloat16_rne(l0c_nz[i]);
        }
        ++fixpipes;
        pending = false;
    }
};

}  // namespace ascend_micro::sim
