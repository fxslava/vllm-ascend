#include "turboquant_tiling.h"

static_assert(vllm_integration_layer::FusedDecodeLaunch{}.head_size ==
                  ascend_micro::fused_decode_opt::kHeadSize,
              "the staged tiling descriptor must remain default constructible");
