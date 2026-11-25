#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "kernel_operator.h"
#if !defined(__CCE_AICORE__)
// CANN's generated ASC host entry replaces this body and adds unused launch
// parameters. Limit the diagnostic exception to these generated entry stubs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
extern "C" __global__ __aicore__ void e8_parallel_component(
    __gm__ uint16_t*, __gm__ float*, __gm__ uint16_t*, __gm__ int8_t*) {}
#if !defined(__CCE_AICORE__)
#pragma clang diagnostic pop
#endif
#endif
#include "parallel_component.cpp"
