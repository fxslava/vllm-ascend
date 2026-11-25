#pragma once
#include <cstdint>

extern "C" void launch_e8_parallel_component(void* stream, std::uint16_t* labels, float* scales,
                                             std::uint16_t* bf16, std::int8_t* int8);
