# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force the real NPU runtime in UTs, bypassing the npu-smi availability probe.

Load before conftest initialization with:
    python -m pytest -p tests.force_npu tests/ut/attention/ -k turboquant

This does not initialize or mock torch_npu. Missing runtime dependencies and
NPU errors are left visible to the caller.
"""


def pytest_report_header():
    return "UT runtime: forced NPU (npu-smi availability probe skipped)"
