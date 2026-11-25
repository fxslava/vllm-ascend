# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


class BuildExt:
    def initialize_options(self):
        pass


def setup_helpers(**overrides):
    # Loading setup.py directly would run dependency detection and setup().
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in {"normalize_soc_version", "cmake_build_ext"}]
    namespace = {"build_ext": BuildExt, "os": os, "Path": Path, **overrides}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("supplied", "canonical"),
    [
        ("aScEnD950pR_9599", "Ascend950PR_9599"),
        ("DAV-3510", "Ascend950PR_9599"),
        (" ASCEND950 ", "ascend950"),
        ("ASCEND910B2", "ascend910b2"),
        ("910C", "ascend910_9392"),
        ("310P", "ascend310p1"),
    ],
)
def test_canonical_soc(supplied, canonical):
    assert setup_helpers()["normalize_soc_version"](supplied) == canonical


def test_editable_outputs_include_only_cmake_installed_targets(tmp_path):
    expected = ("vllm_ascend_C.so", "libvllm_ascend_turboquant.so", "libvllm_ascend_turboquant_e8.so")
    build_temp = tmp_path / "cmake-build"
    build_temp.mkdir()
    (build_temp / "unselected.so").touch()
    build_lib = tmp_path / "build-lib"
    package = tmp_path / "vllm_ascend"
    installs = []

    def install(command, cwd):
        assert command[:4] == ["cmake", "--install", ".", "--prefix"]
        assert command[5:] == ["--component", "vllm_ascend_native"]
        prefix = Path(command[4])
        installs.append(prefix)
        manifest = Path(cwd) / "install_manifest_vllm_ascend_native.txt"
        manifest.write_text("\n".join(str(prefix / target) for target in expected), encoding="utf-8")

    helpers = setup_helpers(
        subprocess=SimpleNamespace(check_call=install), envs=SimpleNamespace(COMPILE_CUSTOM_KERNELS=True)
    )
    command = helpers["cmake_build_ext"]()
    command.initialize_options()
    command.build_temp = str(build_temp)
    command.build_lib = str(build_lib)
    command.get_finalized_command = lambda _: SimpleNamespace(get_package_dir=lambda _: str(package))
    destination = build_lib / "vllm_ascend"
    command._native_outputs = command.install_native_targets(str(destination))
    command.copy_extensions_to_source()

    assert installs == [destination, package]
    assert command.get_outputs() == [str(destination / target) for target in expected]
    assert command.get_output_mapping() == {str(destination / target): str(package / target) for target in expected}
