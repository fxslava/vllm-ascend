#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_COMMON_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_COMMON_H

#include "kernel_operator.h"
#include "turboquant_layout.h"

namespace vllm_ascend {
namespace turboquant {

constexpr uint8_t kSubBlockSyncMode = 0x2;

constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncEvent()
{
    const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}

__aicore__ inline void SyncVectorToMte3() { SyncEvent<AscendC::HardEvent::V_MTE3>(); }
__aicore__ inline void SyncMte3ToVector() { SyncEvent<AscendC::HardEvent::MTE3_V>(); }
__aicore__ inline void SyncMte2ToVector() { SyncEvent<AscendC::HardEvent::MTE2_V>(); }
__aicore__ inline void SyncVectorToMte2() { SyncEvent<AscendC::HardEvent::V_MTE2>(); }
__aicore__ inline void SyncMte1ToMatrix() { SyncEvent<AscendC::HardEvent::MTE1_M>(); }
__aicore__ inline void SyncMatrixToFixpipe() { SyncEvent<AscendC::HardEvent::M_FIX>(); }

__aicore__ inline void SyncScalarToVector() { SyncEvent<AscendC::HardEvent::S_V>(); }

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b) { return static_cast<uint16_t>((a + b - 1) / b); }

__aicore__ inline uint32_t ScaleSlotFloats(uint32_t numKvHeads)
{
    return CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock;
}

__aicore__ inline uint32_t MixBlockIdx()
{
    return static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
}

__aicore__ inline void BroadcastScalar(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src)
{
    AscendC::Brcb(dst, src, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
}

__aicore__ inline void BroadcastSub(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                    const AscendC::LocalTensor<float> &scalarBlock, const uint32_t count)
{
    constexpr uint8_t kRepBlocks = static_cast<uint8_t>(kFp32PerRepeat / kFp32PerBlock);
    const uint32_t repeats = count / kFp32PerRepeat;
    if (repeats > 0) {
        AscendC::Sub(dst, src, scalarBlock, static_cast<uint64_t>(kFp32PerRepeat), static_cast<uint8_t>(repeats),
                     {1, 1, 0, kRepBlocks, kRepBlocks, 0});
    }
    const uint32_t tail = count - repeats * kFp32PerRepeat;
    if (tail > 0) {
        const uint32_t base = repeats * kFp32PerRepeat;
        AscendC::Sub(dst[base], src[base], scalarBlock, static_cast<uint64_t>(tail), 1, {1, 1, 0, 0, 0, 0});
    }
}

__aicore__ inline void RepeatButterfly(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                       const uint32_t stride, const uint32_t groups)
{
    const uint32_t lanes = stride < kFp32PerRepeat ? stride : kFp32PerRepeat;
    const uint8_t rep = static_cast<uint8_t>(2 * stride / kFp32PerBlock);
    const AscendC::BinaryRepeatParams params{1, 1, 1, rep, rep, rep};
    for (uint32_t done = 0; done < groups; done += kMaxRepeatTimes) {
        const uint8_t batch = static_cast<uint8_t>((groups - done) < kMaxRepeatTimes ? (groups - done) : kMaxRepeatTimes);
        const uint32_t base = done * 2 * stride;
        for (uint32_t lane = 0; lane < stride; lane += lanes) {
            const uint32_t lo = base + lane;
            const uint32_t hi = lo + stride;
            AscendC::Add(dst[lo], src[lo], src[hi], static_cast<uint64_t>(lanes), batch, params);
            AscendC::Sub(dst[hi], src[lo], src[hi], static_cast<uint64_t>(lanes), batch, params);
        }
    }
}

}
}

#endif
