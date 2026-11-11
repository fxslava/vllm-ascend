# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the device-free TurboQuant tests on a host with no NPU and no vllm.

    python tests/ut/attention/run_turboquant_device_free.py

Arguments are passed through to pytest, so a single file or ``-k`` expression
works::

    python tests/ut/attention/run_turboquant_device_free.py -k schema -v

The files listed below check things that need only ``torch``: the three-plane
cache layout arithmetic, the kernel/golden dispatch seam, the in-code profiler's
bookkeeping, and the ``ops.def`` schemas
in ``csrc/attention/turboquant/torch_bindings.cpp` against the Python stubs that
call them. They are ordinary members of ``tests/ut/attention`` and run there in
CI like everything else; this script only makes them reachable somewhere that
cannot host the real suite, by installing :mod:`stub_vllm` first and skipping
``tests/ut/conftest.py``, which calls ``adapt_patch()`` and needs vllm.

Where vllm is installed this script refuses to run and tells you to use pytest
directly, so it can never shadow a real run.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Device-free by construction: no torch_npu call, no CANN, no NPU. Anything
# added here must hold to that, or it will fail only on the hosts this script
# exists for. test_turboquant_live_write_npu.py is deliberately absent: it is
# the live writer gate and needs real hardware.
DEVICE_FREE_TESTS = (
    "test_turboquant_op_schemas.py",
    "test_turboquant_dispatch_seam.py",
    "test_turboquant_strip_cache.py",
    "test_turboquant_dry_run_codec.py",
    "test_turboquant_kernel_cache_layout.py",
    "test_turboquant_layout.py",
    "test_turboquant_nz_ingress.py",
    "test_turboquant_profile.py",
)


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(HERE))

    import stub_vllm

    if stub_vllm.vllm_is_importable():
        print(
            "vllm is importable here, so this script is not what you want. Run the "
            "tests directly:\n\n"
            f"    pytest {' '.join(f'tests/ut/attention/{name}' for name in DEVICE_FREE_TESTS)}\n",
            file=sys.stderr,
        )
        return 2

    stub_vllm.install()

    import pytest

    # --noconftest: tests/ut/conftest.py calls adapt_patch(), which imports the
    # vllm_ascend patch tree and from there vllm itself. None of these six files
    # uses a fixture from it. --import-mode=importlib keeps the test modules out
    # of the way of the stubbed `vllm_ascend` package tree in sys.modules.
    return pytest.main(
        [
            "--noconftest",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            *(str(HERE / name) for name in DEVICE_FREE_TESTS),
            *argv,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
