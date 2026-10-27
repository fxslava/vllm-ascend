#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "tensor_api/tensor.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_CUBE_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_CUBE_INLINE inline
#endif

constexpr std::uint32_t kCubeBankSkewBytes = 32;

// Each phase is an asynchronous leaf. The caller completes L1->L0 transfers
// before dispatching the Cube phase, and completes Fixpipe before UB consumers.
template <class Engine>
ASCEND_MICRO_CUBE_INLINE void cube_matmul_stage_l0(Engine& engine)
{
    engine.load_a();
    engine.load_b();
}

template <class Engine>
ASCEND_MICRO_CUBE_INLINE void cube_matmul_compute_dump(Engine& engine)
{
    engine.mma_dispatch();
    engine.fixpipe_to_ub();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

template <int M>
struct NativeCubeMatmulEngine {
    static_assert(M == 16 || M == 32, "supported M tiles: 16 or 32");
    static constexpr int N = 16;
    static constexpr int K = 16;
    __cbuf__ half* l1_a;
    __cbuf__ half* l1_b;
    __ca__ half* l0a;
    __cb__ half* l0b;
    __cc__ float* l0c;
    __ubuf__ float* ub_output;

    __aicore__ inline void load_a()
    {
        using namespace AscendC::Te;
        auto src = MakeTensor(MakeMemPtr(l1_a), MakeFrameLayout<NZLayoutPtn, half>(M, K));
        auto dst = MakeTensor(MakeMemPtr(l0a), MakeFrameLayout<NZLayoutPtn, half>(M, K));
        Copy(MakeCopy(CopyL12L0A{}), dst, src);
    }
    __aicore__ inline void load_b()
    {
        using namespace AscendC::Te;
        auto src = MakeTensor(MakeMemPtr(l1_b), MakeFrameLayout<ZNLayoutPtn, half>(K, N));
        auto dst = MakeTensor(MakeMemPtr(l0b), MakeFrameLayout<ZNLayoutPtn, half>(K, N));
        Copy(MakeCopy(CopyL12L0B{}), dst, src);
    }
    __aicore__ inline void mma_dispatch();
    __aicore__ inline void fixpipe_to_ub()
    {
        using namespace AscendC::Te;
        auto src = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(M, N));
        auto dst = MakeTensor(MakeMemPtr(ub_output), MakeFrameLayout<NDExtLayoutPtn>(M, N));
        FixpipeParams params;
        params.unitFlag = 3;
        Copy(MakeCopy(CopyL0C2UB{}).with(params), dst, src);
    }
};

template <int M>
__aicore__ inline void NativeCubeMatmulEngine<M>::mma_dispatch()
{
    using namespace AscendC::Te;
    auto a = MakeTensor(MakeMemPtr(l0a), MakeFrameLayout<NZLayoutPtn, half>(M, K));
    auto b = MakeTensor(MakeMemPtr(l0b), MakeFrameLayout<ZNLayoutPtn, half>(K, N));
    auto c = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(M, N));
    AscendC::SetMMRowMajor();
    MmadParams params(M, N, K, 3, true);
    Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(params), c, a, b);
}

#endif

#undef ASCEND_MICRO_CUBE_INLINE
}  // namespace ascend_micro
