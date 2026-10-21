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

// Kernel-side helpers every TurboQuant operator shares: the cross-pipe event idiom, block indexing, and the
// one-block broadcasts. Vector chains issue without intra-pipe barriers (ccec tracks RAW/WAR/WAW within a
// pipe); a PipeBarrier<PIPE_V> stays only where a UB buffer is reinterpreted or written twice over the same
// lanes, or between the FWHT's ping-pong passes, and says which next to it.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_COMMON_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_COMMON_H

#include "kernel_operator.h"
#include "turboquant_layout.h"

namespace vllm_ascend {
namespace turboquant {

// The CrossCoreSetFlag mode that releases both AIV subcores of a MIX block (CANN's NotifyEvent default).
constexpr uint8_t kSubBlockSyncMode = 0x2;

// A Fixpipe that lands the Cube's product rows in a vector subcore's UB, row-major.
constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

// PipeBarrier orders a pipe against itself only; a hand-off between pipes needs a hard event.
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
// For the one thing the scalar unit writes into UB that the vector pipe then reads: a lane
// the vector pipe cannot address because it is not on a 32-byte boundary. See MaskTailLanes.
__aicore__ inline void SyncScalarToVector() { SyncEvent<AscendC::HardEvent::S_V>(); }

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b) { return static_cast<uint16_t>((a + b - 1) / b); }

__aicore__ inline uint32_t ScaleSlotFloats(uint32_t numKvHeads)
{
    return CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock;
}

// The AIC block index both halves of a MIX launch agree on: an AIV subcore sees GetBlockIdx() as
// block * subblockdim + subblock.
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
