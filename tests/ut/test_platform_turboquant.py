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
"""Startup validation for the TurboQuant 4-bit KV cache.

The backend serves two of the five Ascend attention states, and the scheduler
picks a state per batch. Without a startup check the engine accepts a
configuration it cannot serve, runs until a prefill and a decode land in one
batch, and raises from inside ``forward_impl`` -- a crash under load standing
in for a configuration error. These tests pin the refusals and, just as
importantly, pin that a configuration the backend *can* serve is left alone.
"""

from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform


def _config(
    cache_dtype="int4_per_token_head",
    block_size=128,
    chunked_prefill=False,
    prefix_caching=False,
    speculative=None,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=block_size,
            enable_prefix_caching=prefix_caching,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked_prefill),
        speculative_config=speculative,
    )


class TestTurboQuantStartupGating(TestBase):
    def test_a_supported_configuration_passes(self):
        NPUPlatform._validate_turboquant_config(_config())

    def test_chunked_prefill_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(_config(chunked_prefill=True))
        message = str(caught.exception)
        self.assertIn("chunked prefill", message)
        self.assertIn("--no-enable-chunked-prefill", message)

    def test_prefix_caching_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(_config(prefix_caching=True))
        message = str(caught.exception)
        self.assertIn("prefix caching", message)
        self.assertIn("--no-enable-prefix-caching", message)

    def test_speculative_decoding_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(
                _config(speculative=SimpleNamespace(method="mtp"))
            )
        self.assertIn("speculative decoding", str(caught.exception))

    def test_every_unsupported_option_is_named_at_once(self):
        """One startup error should list all of them, not the first one found."""
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(
                _config(
                    chunked_prefill=True,
                    prefix_caching=True,
                    speculative=SimpleNamespace(method="mtp"),
                )
            )
        message = str(caught.exception)
        for expected in ("chunked prefill", "prefix caching", "speculative decoding"):
            self.assertIn(expected, message)

    def test_a_block_size_that_is_not_a_whole_tile_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(_config(block_size=100))
        self.assertIn("multiple of 16", str(caught.exception))

    def test_every_kernel_legal_block_size_is_accepted(self):
        for block_size in (16, 32, 64, 128):
            NPUPlatform._validate_turboquant_config(_config(block_size=block_size))

    def test_a_vllm_turboquant_preset_is_refused_with_the_reason(self):
        """vLLM's own presets interleave fp16 scales per head and budget a
        smaller page than this two-plane layout writes."""
        with self.assertRaises(ValueError) as caught:
            NPUPlatform._validate_turboquant_config(_config(cache_dtype="turboquant_4bit_nc"))
        message = str(caught.exception)
        self.assertIn("turboquant_4bit_nc", message)
        self.assertIn("int4_per_token_head", message)


class TestGatingOnlyAppliesToTurboQuant(TestBase):
    def test_an_unrelated_run_is_untouched(self):
        """Without TurboQuant these options are perfectly legal."""
        NPUPlatform._validate_turboquant_config(
            _config(
                cache_dtype="auto",
                chunked_prefill=True,
                prefix_caching=True,
                speculative=SimpleNamespace(method="mtp"),
                block_size=100,
            )
        )

    def test_the_env_var_route_is_gated_too(self):
        """ENABLE_TURBOQUANT=1 activates the backend whatever the cache dtype."""
        with patch("vllm_ascend.envs.ENABLE_TURBOQUANT", True):
            with self.assertRaises(ValueError) as caught:
                NPUPlatform._validate_turboquant_config(
                    _config(cache_dtype="auto", chunked_prefill=True)
                )
            self.assertIn("chunked prefill", str(caught.exception))


class TestEnablementDetection(TestBase):
    def test_every_route_is_recognised(self):
        from vllm_ascend.utils import is_turboquant_cache_dtype, turboquant_enabled

        self.assertTrue(is_turboquant_cache_dtype("int4_per_token_head"))
        self.assertTrue(is_turboquant_cache_dtype("turboquant_4bit_nc"))
        self.assertFalse(is_turboquant_cache_dtype("auto"))
        self.assertFalse(is_turboquant_cache_dtype(None))

        self.assertTrue(turboquant_enabled(_config()))
        self.assertFalse(turboquant_enabled(_config(cache_dtype="auto")))
        self.assertFalse(turboquant_enabled(None))

        with patch("vllm_ascend.envs.ENABLE_TURBOQUANT", True):
            self.assertTrue(turboquant_enabled(None))
