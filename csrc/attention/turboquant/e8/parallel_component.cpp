#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "routines/routine_e8_bf16_dequant_simt.hpp"
#include "routines/routine_e8_int8_dequant_simt.hpp"
namespace {
constexpr unsigned kRows=16,kLabels=512,kElements=4096;
constexpr unsigned kLabelOffset=0,kScaleOffset=1280,kBf16Offset=1600,kInt8Offset=10048;
constexpr unsigned kArenaBytes=14400;
static_assert(kInt8Offset+kElements+256<=kArenaBytes);
static_assert(kArenaBytes<=216*1024);
}
extern "C" __global__ __aicore__ void e8_parallel_component(
    __gm__ uint16_t* labels,__gm__ float* scales,
    __gm__ uint16_t* bf16,__gm__ int8_t* int8) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    __ubuf__ uint8_t arena[kArenaBytes] __attribute__((aligned(32)));
    const auto base=static_cast<uint32_t>(reinterpret_cast<uint64_t>(arena));
    auto localLabels=AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECIN,base+kLabelOffset,kLabels);
    auto localScales=AscendC::LocalTensor<float>(AscendC::TPosition::VECIN,base+kScaleOffset,kRows);
    auto localBf16=AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECOUT,base+kBf16Offset,kElements);
    auto localInt8=AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT,base+kInt8Offset,kElements);
    auto dummy=AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN,base+1024,32);
    AscendC::GlobalTensor<uint16_t> inputLabels;
    AscendC::GlobalTensor<bfloat16_t> outputBf16;
    AscendC::GlobalTensor<float> inputScales;AscendC::GlobalTensor<int8_t> outputInt8;
    inputLabels.SetGlobalBuffer(labels);inputScales.SetGlobalBuffer(scales);
    outputBf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bf16));outputInt8.SetGlobalBuffer(int8);
    AscendC::DataCopy(localLabels,inputLabels,kLabels);
    AscendC::DataCopy(localScales,inputScales,kRows);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    namespace bf=ascend_micro::e8;
    namespace i8=ascend_micro::e8_int8_simt;
    bf::Dequant<bfloat16_t,kElements,bf::Backend::SIMT,
        bf::Representation::Arithmetic,128,bf::TableLayout::AoS,bf::Shaping::ResidualBits>(
            localLabels,localScales,localBf16,dummy,dummy);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    i8::Dequant<int8_t,kElements,i8::Backend::SIMT,
        i8::Representation::Arithmetic,128,i8::TableLayout::AoS,i8::Shaping::ResidualBits>(
            localLabels,localScales,localInt8,dummy,dummy);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(outputBf16,localBf16,kElements);
    AscendC::DataCopy(outputInt8,localInt8,kElements);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void e8_parallel_component(
    __gm__ uint16_t*,__gm__ float*,__gm__ uint16_t*,__gm__ int8_t*);
extern "C" void launch_e8_parallel_component(void* stream,uint16_t* labels,
    float* scales,uint16_t* bf16,int8_t* int8) {
    e8_parallel_component<<<1,nullptr,stream>>>(labels,scales,bf16,int8);
}
#endif
