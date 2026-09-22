#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Experimental A5 MLA decode optimization. Split each request's Block Table
    # while broadcasting Query, then merge Local FIA outputs with AttentionUpdate.
    # It is disabled by default until the end-to-end accuracy/performance gates pass.
    "VLLM_ASCEND_MLA_FIA_SPLIT": lambda: bool(int(os.getenv("VLLM_ASCEND_MLA_FIA_SPLIT", "0"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_ffn_combine/mega_moe`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine/mega_moe` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # `mega_moe` can be used only for moe layer with W8A8/W4A8/bf16(none quant), EP<=64, non-dynamic-eplb.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Route AscendAttentionBackend to the TurboQuant 4-bit KV cache implementation
    # (vllm_ascend/attention/turboquant_v1.py) and pack its KV cache to
    # head_size // 2 int8 elements. 0 (default): off. 1: on. It applies to every
    # layer on AscendAttentionBackend, so enable it only for dense-attention models
    # the TurboQuant kernels support; it cannot be combined with decode context
    # parallel.
    "ENABLE_TURBOQUANT": lambda: bool(int(os.getenv("ENABLE_TURBOQUANT", "0"))),
    # Decode the TurboQuant 4-bit KV cache with the kv4fp8 Cube kernels: one launch per step
    # that rotates the raw query and, for an o_proj without Pi folded in, un-rotates the output
    # and applies an attn_output_gate itself (vllm_ascend/attention/turboquant_v1.py).
    # 1 (default): on wherever the build has the Cube kernels (Ascend 950) and the model runs
    # in float16; everywhere else the AIV decode is used regardless. 0: always the AIV decode,
    # with separate query and output rotation launches. The two write different cache layouts,
    # so the value is read once per layer, when its attention impl is built.
    "VLLM_ASCEND_TURBOQUANT_CUBE_DECODE": lambda: bool(int(os.getenv("VLLM_ASCEND_TURBOQUANT_CUBE_DECODE", "1"))),
    # Check every slot the TurboQuant writer is handed against the cache it writes into, before
    # the launch, and drain the launch afterwards (vllm_ascend/attention/turboquant_v1.py).
    # 0 (default): off. 1: on. The values live on the device, so the check costs a device-to-host
    # copy and a synchronisation on every cache write of every layer -- diagnostic only, never a
    # serving default. The kernels drop an unusable slot either way; this is what turns that silent
    # drop into a named error, and the two synchronisations bracket the write so an asynchronous
    # fault is attributed to the right side of it rather than to a later operator.
    "VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": lambda: bool(int(os.getenv("VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS", "0"))),
    # Append every TurboQuant operator invocation -- its tensor signatures, pointer
    # alignments, slot/block index ranges and scalar launch arguments -- as one JSON
    # record per line to TURBOQUANT_TRACE_PATH
    # (vllm_ascend/attention/turboquant_trace.py). 0 (default): off. 1: on. Diagnostic
    # only, never a serving default: summarising the index tensors costs a
    # device-to-host copy per call site, and the file is line-buffered so a record
    # survives the process abort an asynchronous kernel fault causes. The file is
    # written directly rather than through `logger`, because a worker process's log
    # output is redirected somewhere that may never be flushed.
    "TURBOQUANT_CAPTURE_SIGNATURES": lambda: bool(int(os.getenv("TURBOQUANT_CAPTURE_SIGNATURES", "0"))),
    # Skip every TurboQuant kernel launch and leave the destination tensors zeroed
    # (vllm_ascend/attention/turboquant_trace.py). 0 (default): off. 1: on. The engine
    # then walks through prefill and every decode step of every layer without ever
    # reaching an AI core, so TURBOQUANT_CAPTURE_SIGNATURES can record the real
    # production configuration of a run that otherwise aborts at its first faulting
    # launch. The outputs are meaningless: this produces no tokens worth reading.
    "TURBOQUANT_DRY_RUN": lambda: bool(int(os.getenv("TURBOQUANT_DRY_RUN", "0"))),
    # Where TURBOQUANT_CAPTURE_SIGNATURES appends its records. Defaults to
    # /workspace/turboquant_trace.log, which is the container's bind mount. A path that
    # cannot be opened falls back to stderr rather than losing the capture.
    "TURBOQUANT_TRACE_PATH": lambda: os.getenv("TURBOQUANT_TRACE_PATH", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
