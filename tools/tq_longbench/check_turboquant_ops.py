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
"""Which TurboQuant library this process would load, and whether its schema is current.

    python tools/tq_longbench/check_turboquant_ops.py

Seconds, no checkpoint, no dataset, no device. Run it before a benchmark: a stale
registration is otherwise found by the dispatcher, from inside a decode step, minutes
after the weights loaded, as "expected at most 17 argument(s) but received 18".

What it prints:

* every path the loader would try, in order, and which of them exist;
* whether the operators were already registered before the loader looked -- which is the
  failure mode on a host with an older extension installed, because registration is global
  and permanent and a fresh library cannot replace what got in first;
* the mapped files that could have registered them, out of ``/proc/self/maps``, since the
  dispatcher records no provenance for a schema;
* the argument count of each probed operator and whether it carries ``lse``.

Exit 0 when the registered operators carry ``lse``; 1 when they do not; 2 when nothing
registered them at all. Nothing here needs ``setup.py`` or ``pip install``: the library it
looks for is the one ``build_turboquant_ops.py`` produces.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """Appended, so the standard library keeps precedence over ``tools/bisect``."""
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)


_bootstrap_path()

import os  # noqa: E402

from tq_longbench._ascend import (  # noqa: E402
    TURBOQUANT_BUILD_DIR_ENV,
    TURBOQUANT_KERNELS_LIB_NAME,
    TURBOQUANT_LIB_ENV,
    TURBOQUANT_LIB_NAME,
    assert_turboquant_schema_is_fresh,
    load_turboquant_library,
    turboquant_library_candidates,
    turboquant_mapped_libraries,
    turboquant_schema_report,
)


def main() -> int:
    print("=== environment ===")
    for name in (TURBOQUANT_LIB_ENV, TURBOQUANT_BUILD_DIR_ENV, "LD_LIBRARY_PATH", "ASCEND_HOME_PATH"):
        print(f"  {name} = {os.environ.get(name, '<unset>')}")

    print()
    print("=== candidates, in the order the loader tries them ===")
    for candidate in turboquant_library_candidates():
        kernels = candidate.parent / TURBOQUANT_KERNELS_LIB_NAME
        mark = "found  " if candidate.is_file() else "missing"
        beside = "" if not candidate.is_file() else f"  [kernels beside it: {'yes' if kernels.is_file() else 'NO'}]"
        print(f"  {mark} {candidate}{beside}")

    # Imported here rather than at the top: the message above is the useful part on a host
    # where importing torch is itself the thing that fails.
    import torch  # noqa: PLC0415

    print()
    print(f"=== registration (torch {torch.__version__}) ===")

    def served() -> bool:
        namespace = getattr(torch.ops, "_C_ascend", None)
        return namespace is not None and hasattr(namespace, "npu_turboquant_reshape_and_cache")

    before = served()
    print(f"  registered before the loader looked: {before}")
    outcome = load_turboquant_library(served)
    print(f"  loader: registered={outcome.registered}  {outcome.detail}")

    print("  mapped files that could have registered these operators:")
    for path in turboquant_mapped_libraries() or ["(no mapped file is named after either)"]:
        print(f"    {path}")

    print()
    print("=== schema ===")
    for line in turboquant_schema_report():
        print(f"  {line}")

    if not outcome.registered:
        print()
        print("RESULT: nothing registered the TurboQuant operators.")
        print(f"  Build {TURBOQUANT_LIB_NAME} with tools/tq_longbench/build_turboquant_ops.py --soc-version")
        print(f"  Ascend950PR_9599, then point ${TURBOQUANT_LIB_ENV} or ${TURBOQUANT_BUILD_DIR_ENV} at it.")
        return 2

    try:
        assert_turboquant_schema_is_fresh()
    except RuntimeError as error:
        print()
        print("RESULT: STALE")
        print(error)
        return 1

    print()
    print("RESULT: OK -- the registered operators carry the lse out-tensor.")
    if before:
        print("  Note: they were already registered before the loader looked, so this run is using")
        print("  whatever got in first. That is fine because the schema is current; it would not be")
        print("  correctable if it were not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
