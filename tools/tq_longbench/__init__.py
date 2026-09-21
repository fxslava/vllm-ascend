#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""A standalone long-context evaluation harness for the TurboQuant KV cache.

PyTorch, ``torch_npu`` and the Ascend C operators; ``transformers`` for the
tokenizer and the config, and nothing else.  No vLLM is imported -- see
:mod:`tq_longbench._ascend` for how the cache layout is shared with the plugin
without one.

Nothing is imported eagerly here: the submodules reach ``torch_npu``,
``transformers`` and ``datasets`` in different combinations, and a test that
only needs the metrics should not need a device runtime.
"""

from __future__ import annotations

__all__ = [
    "build_turboquant_ops",
    "cpu_reference",
    "diagnose",
    "engine",
    "families",
    "glm4",
    "hf_bridge",
    "kv_cache",
    "kv_dump",
    "layers",
    "ops",
    "preflight",
    "probe",
    "reference",
    "run_benchmark",
    "run_eval",
    "run_longbench",
    "smoke_dense",
    "smoke_glm",
    "smoke_hf",
    "tasks",
]
