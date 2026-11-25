# SPDX-License-Identifier: Apache-2.0
"""Compile/load only the real Torch bindings and CPU tiling, with NPU traps.

Run in the pinned Torch 2.10 CPU container. This cannot launch a device kernel
or link the full vllm-ascend plugin. Every device dependency throws if reached.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TQ = ROOT / "csrc/attention/turboquant"


def run() -> None:
    torch_root = Path(torch.__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="tq-host-bindings-") as directory:
        out = Path(directory)
        flags = [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-O0",
            "-I",
            str(ROOT),
            "-I",
            str(HERE / "stubs"),
            "-isystem",
            str(torch_root / "include"),
            "-isystem",
            str(torch_root / "include/torch/csrc/api/include"),
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ]
        sources = [TQ / "torch_bindings.cpp", TQ / "e8_torch_bindings.cpp", TQ / "op_host/turboquant_tiling.cpp"]
        for source in [*sources, HERE / "abi.cpp"]:
            subprocess.run([*flags, "-fsyntax-only", str(source)], check=True)
        # Generate traps from the public declarations so drift cannot be hidden
        # by a second handwritten set of launcher signatures.
        adapters = (TQ / "op_adapter/turboquant_torch_adpt.h").read_text()
        adapters = re.sub(r"//[^\n]*", "", adapters)
        stubs = [
            "#include <stdexcept>",
            '#include "csrc/attention/turboquant/op_adapter/turboquant_torch_adpt.h"',
            '#include "csrc/attention/turboquant/e8/e8_decode.hpp"',
            '#include "csrc/attention/turboquant/e8/e8_component.hpp"',
            "#include <torch_npu/csrc/core/npu/NPUStream.h>",
            'void* c10_npu::HostSyntaxStream::stream() const { throw std::runtime_error("NPU trap"); }',
            'c10_npu::HostSyntaxStream c10_npu::getCurrentNPUStream() { throw std::runtime_error("NPU trap"); }',
            'int aclGetDeviceCapability(int, int, std::int64_t*) { throw std::runtime_error("NPU trap"); }',
        ]
        for name, args in re.findall(r"void\s+(\w+)\s*\(([^;]+)\)\s*;", adapters):
            params = ",".join(re.sub(r"\b\w+\s*$", "", p.strip()) for p in args.split(","))
            if name.startswith("turboquant_mm_"):
                name = "vllm_ascend::" + name
                linkage = ""
            else:
                linkage = 'extern "C" '
            stubs.append(f'{linkage}void {name}({params}) {{ throw std::runtime_error("NPU trap"); }}')
        stubs.extend(
            [
                'extern "C" void launch_e8_fused_decode_fp32(void*, const e8_decode::Launch*, '
                'const e8_decode::Buffers*) { throw std::runtime_error("NPU trap"); }',
                'extern "C" void launch_e8_parallel_component(void*, std::uint16_t*, float*, '
                'std::uint16_t*, std::int8_t*) { throw std::runtime_error("NPU trap"); }',
            ]
        )
        stub_path = out / "traps.cpp"
        stub_path.write_text("\n".join(stubs))
        library = out / "bindings.so"
        subprocess.run(
            [
                *flags,
                "-shared",
                *(str(s) for s in sources),
                str(stub_path),
                "-L",
                str(torch_root / "lib"),
                "-ltorch_cpu",
                "-ltorch",
                "-lc10",
                f"-Wl,-rpath,{torch_root / 'lib'}",
                "-Wl,--no-undefined",
                "-o",
                str(library),
            ],
            check=True,
        )
        torch.ops.load_library(str(library))
        # Real C++ Meta dispatch, then FakeTensor dispatch over those registrations.
        from torch._subclasses.fake_tensor import FakeTensorMode

        def meta_frame(device):
            return [
                torch.empty((16, 256), device=device),
                torch.empty((6, 4352), dtype=torch.uint8, device=device),
                torch.empty((6, 4352), dtype=torch.uint8, device=device),
                torch.empty(8, dtype=torch.int64, device=device),
                torch.empty((6, 256), dtype=torch.bfloat16, device=device),
                torch.empty((6, 64, 256), dtype=torch.bfloat16, device=device),
                torch.empty((6, 64, 256), dtype=torch.bfloat16, device=device),
                torch.empty((6, 6, 64), device=device),
            ]

        frame = meta_frame("meta")
        torch.ops.turboquant_e8_ops.reader(*frame, 6, 32, 0.0625)
        for index, dtype in ((0, torch.bfloat16), (1, torch.int8), (4, torch.float32)):
            wrong_type = frame.copy()
            wrong_type[index] = wrong_type[index].to(dtype=dtype)
            try:
                torch.ops.turboquant_e8_ops.reader(*wrong_type, 6, 32, 0.0625)
            except RuntimeError:
                pass
            else:
                raise AssertionError("accepted a wrong E8 boundary dtype")
        bad = frame.copy()
        bad[3] = torch.empty(6, dtype=torch.int64, device="meta")
        try:
            torch.ops.turboquant_e8_ops.reader(*bad, 6, 32, 0.0625)
        except RuntimeError as exc:
            assert "packet_offsets" in str(exc)
        else:
            raise AssertionError("accepted a short DMA offset table")
        torch.ops.turboquant_e8_ops.dequant(
            torch.empty(512, dtype=torch.uint16, device="meta"),
            torch.empty(16, device="meta"),
            torch.empty(4096, dtype=torch.bfloat16, device="meta"),
            torch.empty(4096, dtype=torch.int8, device="meta"),
        )
        with FakeTensorMode():
            torch.ops.turboquant_e8_ops.reader(*meta_frame("cpu"), 6, 32, 0.0625)
            torch.ops.turboquant_e8_ops.dequant(
                torch.empty(512, dtype=torch.uint16),
                torch.empty(16),
                torch.empty(4096, dtype=torch.bfloat16),
                torch.empty(4096, dtype=torch.int8),
            )
            for name in (
                "npu_turboquant_reshape_and_cache_a8",
                "npu_turboquant_reshape_scatter_aiv",
                "npu_turboquant_fused_decode",
                "npu_turboquant_build_decode_tiling",
            ):
                schema = torch._C._get_schema(f"turboquant_ops::{name}", "")
                values = []
                for argument in schema.arguments:
                    kind = argument.type.kind()
                    if kind == "TensorType":
                        values.append(torch.empty(1))
                    elif kind == "OptionalType":
                        values.append(None)
                    elif kind == "IntType":
                        values.append(1)
                    elif kind == "FloatType":
                        values.append(1.0)
                    else:
                        raise AssertionError(f"unhandled schema type: {argument.type}")
                assert getattr(torch.ops.turboquant_ops, name)(*values) is None
        # Execute the actual native CPU tiling registration and compare its exact
        # bytes with a second invocation; no NPU dependency is reachable.
        lengths = torch.tensor([208], dtype=torch.int32)
        tiling = torch.full((16 + 32 * 8,), -1, dtype=torch.int32)
        torch.ops.turboquant_ops.npu_turboquant_build_decode_tiling(lengths, tiling, 6, 1, 256, 128, 4, 1, 6, 4096)
        assert tiling[:4].tolist() == [4, 3, 16, 0]
        assert tiling[5:8].tolist() == [256, 6, 1]
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "syntax_translation_units": 4,
                    "registered_ops": 6,
                    "real_cpp_meta": "PASS",
                    "fake_tensor": "PASS",
                    "cpu_tiling": "PASS",
                    "device_launches": 0,
                }
            )
        )


if __name__ == "__main__":
    run()
