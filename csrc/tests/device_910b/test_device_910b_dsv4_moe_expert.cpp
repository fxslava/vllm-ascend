// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "dsv4_production_suite.hpp"

namespace vllm_ascend::test {
namespace {
INSTANTIATE_TEST_SUITE_P(DeepSeekV4, Dsv4Production,
                         ::testing::Values(ExpertCase{704, dsv4::kProductionInter, ExpertHardware::k910B, false},
                                           ExpertCase{4096, dsv4::kProductionInter, ExpertHardware::k910B, false},
                                           ExpertCase{dsv4::kProductionHidden, dsv4::kProductionInter,
                                                      ExpertHardware::k910B, false}));
}  // namespace
}  // namespace vllm_ascend::test
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::vllm_ascend::test::RegisterAscendTestEnvironment();
  return RUN_ALL_TESTS();
}
