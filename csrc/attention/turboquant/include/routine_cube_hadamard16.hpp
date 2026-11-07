#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "tensor_api/tensor.h"
#endif

namespace ascend_micro {

// H16 is the product of four Sylvester butterfly factors. The caller supplies
// the resulting 16x16 sign matrix in L0B/ZN; no scalar coefficient generation
// or elementwise arithmetic occurs on the AIC.
enum class OutflowMode : std::uint8_t { Fp32 = 0, Fp16 = 1, Bf16 = 2 };

constexpr unsigned kHadamardOrder = 16;
constexpr unsigned kFp16TileBytes = 16 * 16 * 2;
constexpr unsigned kFp32TileBytes = 16 * 16 * 4;
constexpr unsigned kL1BankSkewBytes = 32;
constexpr unsigned kL1BOffsetBytes = kFp16TileBytes + kL1BankSkewBytes;

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_AIC_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_AIC_INLINE inline
#endif

// The two calls only dispatch asynchronous Cube and Fixpipe work. The engine
// owns its already-ready L0A/L0B operands and the caller owns completion of
// GM -> L1 -> L0 staging and final GM visibility. Mmad/Fixpipe use matching
// unitFlag=3 on the same 16x16 L0C fractal.
template <OutflowMode Mode, class Engine>
ASCEND_MICRO_AIC_INLINE void cube_hadamard16(Engine& engine)
{
    static_assert(Mode == OutflowMode::Fp32 || Mode == OutflowMode::Fp16 ||
                  Mode == OutflowMode::Bf16, "invalid outflow mode");
    engine.mmad();
    engine.template fixpipe<Mode>();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

// Native 950PR Tensor API backend. These pointers must point to complete
// 16x16 fractals: L0A NZ and L0B ZN, both 512-byte aligned; L0C NZ, 64-byte
// aligned; GM output at least 32-byte aligned. Inputs must not alias output.
// The caller must arrange cross-pipeline completion before entering this leaf.
template <class Output>
struct NativeCubeEngine {
    __ca__ half* l0a;
    __cb__ half* l0b;
    __cc__ float* l0c;
    __gm__ Output* output;

    ASCEND_MICRO_AIC_INLINE void mmad()
    {
        using namespace AscendC::Te;
        auto a = MakeTensor(MakeMemPtr(l0a), MakeFrameLayout<NZLayoutPtn, half>(16, 16));
        auto b = MakeTensor(MakeMemPtr(l0b), MakeFrameLayout<ZNLayoutPtn, half>(16, 16));
        auto c = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(16, 16));
        AscendC::SetMMRowMajor();  // NZ -> ND Fixpipe consumes N first.
        MmadParams params(16, 16, 16, 3, true);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(params), c, a, b);
    }

    template <OutflowMode Mode>
    ASCEND_MICRO_AIC_INLINE void fixpipe()
    {
        using namespace AscendC::Te;
        auto c = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(16, 16));
        auto y = MakeTensor(MakeMemPtr(output), MakeFrameLayout<NDExtLayoutPtn>(16, 16));
        FixpipeParams params;
        params.unitFlag = 3;
        Copy(MakeCopy(CopyL0C2GM{}).with(params), y, c);
    }
};

template <OutflowMode Mode, class Output>
ASCEND_MICRO_AIC_INLINE void cube_hadamard16_native(
    __ca__ half* l0a, __cb__ half* l0b, __cc__ float* l0c,
    __gm__ Output* output)
{
    NativeCubeEngine<Output> engine{l0a, l0b, l0c, output};
    cube_hadamard16<Mode>(engine);
}

#endif

#undef ASCEND_MICRO_AIC_INLINE
}  // namespace ascend_micro
