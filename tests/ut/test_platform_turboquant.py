# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-time handling of ``ENABLE_TURBOQUANT=1``.

The graph mode is the part worth a test: ``FULL_DECODE_ONLY`` is the platform's
own default for a decode-heavy run, so without the fallback every TurboQuant
run captures its attention layers, and the dry-run emulator cannot be captured
(a torch op with no NPU kernel falls back to the host, and a host copy on a
capturing stream fails with ``aclrtMemcpy`` 107030).
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config.compilation import CompilationMode, CUDAGraphMode

from vllm_ascend.platform import _apply_turboquant_graph_mode


def _config(cudagraph_mode: CUDAGraphMode, mode: CompilationMode = CompilationMode.VLLM_COMPILE):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode, mode=mode),
    )


def _enabled(value: str):
    return patch.dict(os.environ, {"ENABLE_TURBOQUANT": value})


@pytest.mark.parametrize(
    "full_mode",
    [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_a_full_graph_falls_back_to_piecewise(full_mode):
    vllm_config = _config(full_mode)
    with _enabled("1"):
        _apply_turboquant_graph_mode(vllm_config)
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_without_piecewise_compilation_the_fallback_is_no_graph():
    """There is nothing to split out of an uncompiled model, so capture nothing."""
    vllm_config = _config(CUDAGraphMode.FULL_DECODE_ONLY, mode=CompilationMode.NONE)
    with _enabled("1"):
        _apply_turboquant_graph_mode(vllm_config)
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_a_mode_that_already_works_is_left_alone(mode):
    vllm_config = _config(mode)
    with _enabled("1"):
        _apply_turboquant_graph_mode(vllm_config)
    assert vllm_config.compilation_config.cudagraph_mode == mode


def test_it_is_inert_without_the_env_switch():
    """Every other model keeps the platform's own default."""
    vllm_config = _config(CUDAGraphMode.FULL_DECODE_ONLY)
    with _enabled("0"):
        _apply_turboquant_graph_mode(vllm_config)
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


def test_check_and_update_config_applies_it_after_the_compilation_modes():
    """Order matters: ``_update_compilation_modes`` would overwrite an earlier clamp."""
    import inspect

    from vllm_ascend import platform

    source = inspect.getsource(platform.NPUPlatform.check_and_update_config)
    update = source.index("_update_compilation_modes(vllm_config, ascend_config)")
    clamp = source.index("_apply_turboquant_graph_mode(vllm_config)")
    setup = source.index("_setup_compile_backend(")
    assert update < clamp < setup
