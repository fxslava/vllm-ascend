# SPDX-License-Identifier: Apache-2.0
"""Compile and exercise CPU/Meta bindings with throwing device dependencies."""

from __future__ import annotations

import importlib.abc
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TQ = ROOT / "csrc/attention/turboquant"


class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("torch_npu", "vllm_ascend.vllm_ascend_C")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise ImportError(f"Host binding checks block importing {fullname}")
        return None


def write_traps(path: Path) -> None:
    headers = ("turboquant_launch.h", "e8_decode.hpp", "e8_component.hpp")
    stubs = [
        "#include <stdexcept>",
        "#include <acl/acl.h>",
        "#include <torch_npu/csrc/core/npu/NPUStream.h>",
        *(f'#include "csrc/attention/turboquant/common/{header}"' for header in headers),
        'void* c10_npu::HostSyntaxStream::stream() const { throw std::runtime_error("NPU trap"); }',
        'c10_npu::HostSyntaxStream c10_npu::getCurrentNPUStream() { throw std::runtime_error("NPU trap"); }',
        'int aclGetDeviceCapability(int, int, std::int64_t*) { throw std::runtime_error("NPU trap"); }',
    ]
    for header in headers:
        declarations = (TQ / "common" / header).read_text()
        for name, args in re.findall(r"\bvoid\s+((?:launch_|turboquant_mm_)\w+)\s*\(([^;]+)\)\s*;", declarations):
            params = ", ".join(re.sub(r"\b\w+\s*$", "", part.strip()) for part in args.split(","))
            symbol = f"vllm_ascend::{name}" if name.startswith("turboquant_mm_") else f'extern "C" void {name}'
            if name.startswith("turboquant_mm_"):
                symbol = f"void {symbol}"
            stubs.append(f'{symbol}({params}) {{ throw std::runtime_error("NPU trap"); }}')
    path.write_text("\n".join(stubs) + "\n")


def check_packet_dispatch(torch) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema_name = "turboquant_e8_ops::dequant_packet"
    schema = torch._C._get_schema(schema_name, "")
    assert [arg.name for arg in schema.arguments] == ["packets", "bf16", "int8"]
    assert schema.arguments[0].alias_info is None and not schema.returns
    bf16_alias, int8_alias = (arg.alias_info for arg in schema.arguments[1:])
    assert bf16_alias.is_write and int8_alias.is_write and bf16_alias.before_set.isdisjoint(int8_alias.before_set)
    for dispatch in ("Meta", "PrivateUse1"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(schema_name, dispatch)

    def operands(device):
        return (
            torch.empty((2, 4352), dtype=torch.uint8, device=device),
            torch.empty((2, 64, 256), dtype=torch.bfloat16, device=device),
            torch.empty((2, 64, 256), dtype=torch.int8, device=device),
        )

    op = torch.ops.turboquant_e8_ops.dequant_packet
    packets, bf16, int8 = operands("meta")
    assert op(packets, bf16, int8) is None
    for bad_packets, bad_bf16 in ((packets[:, :-1], bf16), (packets, bf16.transpose(1, 2))):
        try:
            op(bad_packets, bad_bf16, int8)
        except RuntimeError as error:
            assert "E8 packet" in str(error)
        else:
            raise AssertionError("malformed E8 packet dequantization reached dispatch")
    with FakeTensorMode():
        assert op(*operands("cpu")) is None

        def packet_decode(packets, bf16, int8):
            op(packets, bf16, int8)
            return bf16, int8

        compiled = torch.compile(packet_decode, backend="eager", fullgraph=True)
        fake_packets, fake_bf16, fake_int8 = operands("cpu")
        actual_bf16, actual_int8 = compiled(fake_packets, fake_bf16, fake_int8)
        assert actual_bf16 is fake_bf16 and actual_int8 is fake_int8
        placeholder = torch.empty(1)
        native = torch.ops.turboquant_ops
        for write, tensor_count in (
            (native.npu_turboquant_reshape_and_cache_a8, 8),
            (native.npu_turboquant_reshape_scatter_aiv, 7),
        ):
            try:
                write(*([placeholder] * tensor_count), 1.0, 2)
            except RuntimeError as error:
                assert "device E8 encoding is gated" in str(error)
            else:
                raise AssertionError("accepted an unverified device encoder")


def run() -> None:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    assert not any(name == "torch_npu" or name.startswith("torch_npu.") for name in sys.modules)
    sys.meta_path.insert(0, BlockDeviceImports())
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    flags = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
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
    sources = [
        TQ / "bindings/torch_bindings.cpp",
        TQ / "bindings/torch_ingress.cpp",
        TQ / "bindings/torch_decode.cpp",
        TQ / "common/turboquant_tiling.cpp",
    ]
    for source in [*sources, HERE / "abi.cpp"]:
        subprocess.run([*flags, "-fsyntax-only", str(source)], check=True)
    with tempfile.TemporaryDirectory(prefix="tq-host-bindings-") as directory:
        traps = Path(directory) / "traps.cpp"
        write_traps(traps)
        library = Path(directory) / "bindings.so"
        subprocess.run(
            [
                *flags,
                "-fPIC",
                "-shared",
                *map(str, sources),
                str(traps),
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
        check_packet_dispatch(torch)
    assert not any(name == "torch_npu" or name.startswith("torch_npu.") for name in sys.modules)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "syntax_translation_units": len(sources) + 1,
                "packet_meta_fake_fullgraph": "PASS",
                "device_launches": 0,
            }
        )
    )


if __name__ == "__main__":
    run()
