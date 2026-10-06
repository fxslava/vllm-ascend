// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "../../moe/dsv4_moe_expert/op_kernel/dsv4_moe_expert_vector_compat.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void dsv4_gather_index_probe(GM_ADDR input, GM_ADDR output)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuf, offsetsBuf, gatheredBuf;
    constexpr uint32_t SOURCE_COUNT = 2048;
    constexpr uint32_t MAX_COUNT = 256;
    pipe.InitBuffer(sourceBuf, SOURCE_COUNT * sizeof(float));
    pipe.InitBuffer(offsetsBuf, MAX_COUNT * sizeof(int32_t));
    pipe.InitBuffer(gatheredBuf, MAX_COUNT * sizeof(float));
    auto source = sourceBuf.Get<float>();
    auto offsets = offsetsBuf.Get<int32_t>();
    auto gathered = gatheredBuf.Get<float>();
    GlobalTensor<float> inputGm;
    GlobalTensor<uint32_t> outputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    DataCopy(source, inputGm, SOURCE_COUNT);
    const auto loadEvent = pipe.FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(loadEvent);
    WaitFlag<HardEvent::MTE2_V>(loadEvent);
    const auto storeEvent = pipe.FetchEventID(HardEvent::V_MTE3);
    const auto reuseEvent = pipe.FetchEventID(HardEvent::MTE3_V);
    uint32_t outputBase = 0;
    // All merge/split lengths, including the final eight-lane merge. Poison
    // the scratch before each call so missing initialisation cannot pass.
    for (uint32_t count = 8; count <= MAX_COUNT; count *= 2) {
        Duplicate(offsets, static_cast<int32_t>(0x7FC00000), MAX_COUNT);
        Dsv4MoeExpertOp::CreateGatherIndices(offsets, count);
        ShiftLeft(offsets, offsets, 5, count); // 32-byte reduction slots
        PipeBarrier<PIPE_V>();
        Gather(gathered, source, offsets.ReinterpretCast<uint32_t>(), 0, count);
        SetFlag<HardEvent::V_MTE3>(storeEvent);
        WaitFlag<HardEvent::V_MTE3>(storeEvent);
        DataCopy(outputGm[outputBase], offsets.ReinterpretCast<uint32_t>(), count);
        DataCopy(outputGm[outputBase + count], gathered.ReinterpretCast<uint32_t>(), count);
        outputBase += 2 * count;
        SetFlag<HardEvent::MTE3_V>(reuseEvent);
        WaitFlag<HardEvent::MTE3_V>(reuseEvent);
    }
    // SplitColumns' odd address is (2*i+1)*sizeof(float), also in bytes.
    Dsv4MoeExpertOp::CreateGatherIndices(offsets, MAX_COUNT);
    ShiftLeft(offsets, offsets, 3, MAX_COUNT);
    PipeBarrier<PIPE_V>();
    Adds(offsets, offsets, static_cast<int32_t>(sizeof(float)), MAX_COUNT);
    PipeBarrier<PIPE_V>();
    Gather(gathered, source, offsets.ReinterpretCast<uint32_t>(), 0, MAX_COUNT);
    SetFlag<HardEvent::V_MTE3>(storeEvent);
    WaitFlag<HardEvent::V_MTE3>(storeEvent);
    DataCopy(outputGm[outputBase], offsets.ReinterpretCast<uint32_t>(), MAX_COUNT);
    DataCopy(outputGm[outputBase + MAX_COUNT], gathered.ReinterpretCast<uint32_t>(), MAX_COUNT);
    SetFlag<HardEvent::MTE3_V>(reuseEvent);
    WaitFlag<HardEvent::MTE3_V>(reuseEvent);
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void dsv4_gather_index_probe_launch(void* stream, void* input, void* output)
{
    dsv4_gather_index_probe<<<1, nullptr, stream>>>(input, output);
}
#endif
