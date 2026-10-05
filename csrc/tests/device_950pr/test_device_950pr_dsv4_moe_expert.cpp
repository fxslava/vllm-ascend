// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#ifdef VLLM_ASCEND_DSV4_SIM_SUITE
constexpr bool kExpertSimulator = true;
#else
constexpr bool kExpertSimulator = false;
#endif

#include "dsv4_production_suite.hpp"

namespace vllm_ascend::test {
namespace {
INSTANTIATE_TEST_SUITE_P(
    DeepSeekV4, Dsv4Production,
    ::testing::Values(ExpertCase{704, dsv4::kProductionInter, ExpertHardware::k950PR, kExpertSimulator},
                      ExpertCase{4096, dsv4::kProductionInter, ExpertHardware::k950PR, kExpertSimulator},
                      ExpertCase{dsv4::kProductionHidden, dsv4::kProductionInter, ExpertHardware::k950PR,
                                 kExpertSimulator}));
}  // namespace
}  // namespace vllm_ascend::test
