# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
#
# Tier 3 - device. Dispatch by SoC family: each leg configures only when the
# root selected it, so a 950PR tree cannot configure the 310P leg and vice
# versa, and a `*310*` binary cannot exist in a 950PR build tree. See the
# architecture-isolation notes in the root CMakeLists.txt.

if(VLLM_ASCEND_TESTS_TIER_DEVICE)
  add_subdirectory(ascend950pr)
endif()

if(VLLM_ASCEND_TESTS_TIER_DEVICE310)
  add_subdirectory(ascend310p)
endif()
