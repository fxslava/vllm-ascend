#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "kernel_operator.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
extern "C" __global__ __aicore__ void e8_fused_decode_fp32(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                                           GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t,
                                                           float) {}
#pragma clang diagnostic pop
#endif
#include "../decode/e8/e8_fused_decode.cpp"
