#pragma once

#include "../include/turboquant_fused_decode_opt.hpp"

namespace vllm_integration_layer {
using FusedDecodeLaunch = ascend_micro::fused_decode_opt::Launch;
}  // namespace vllm_integration_layer
