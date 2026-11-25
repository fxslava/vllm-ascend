#pragma once
#include <cstdint>

// Fixed verified frame: 512 UINT16 labels, 16 FP32 scales, 4096 BF16 words
// and 4096 signed INT8 output bytes. Buffers are owned by the caller.
extern "C" void launch_e8_parallel_component(void* stream, std::uint16_t* labels,
    float* scales, std::uint16_t* bf16, std::int8_t* int8);
