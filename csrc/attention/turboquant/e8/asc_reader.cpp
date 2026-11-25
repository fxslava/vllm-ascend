// The sealed reader uses legacy generated stubs. ASC requires a host-visible
// entry definition; the device pass still compiles the sealed source verbatim.
#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "kernel_operator.h"
#if !defined(__CCE_AICORE__)
// CANN's generated ASC host entry replaces this body and adds unused launch
// parameters. Limit the diagnostic exception to these generated entry stubs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
extern "C" __global__ __aicore__ void e8_fused_decode_fp32(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, float) {}
#if !defined(__CCE_AICORE__)
#pragma clang diagnostic pop
#endif
#endif
#include "e8_fused_decode.cpp"
