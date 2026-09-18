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
"""Build the TurboQuant operators alone, for Ascend 950, without building vllm-ascend.

    python tools/tq_longbench/build_turboquant_ops.py --soc-version Ascend950PR_9599

Configures ``csrc/attention/turboquant/standalone/CMakeLists.txt`` -- the three
TurboQuant kernel sources, the host tiling and one binding file -- and installs

    tools/tq_longbench/lib/libvllm_turboquant_cube.so
    tools/tq_longbench/lib/libvllm_turboquant_cube_kernels.so

where :func:`tq_longbench._ascend.load_turboquant_library` finds them. The first
registers ``torch.ops._C_ascend.npu_turboquant_*`` -- the same schemas as the full
extension, from the same source -- when ``torch.ops.load_library`` opens it.

**The SoC must be a full variant** (``Ascend950PR_9599``, not ``Ascend950PR``):
the variant's platform config fixes the core counts the kernels are compiled for,
and CANN ships nine 950PR variants. It comes from ``--soc-version``, else
``$SOC_VERSION``, else ``npu-smi`` the way ``setup.py`` reads it.

Two properties of CANN's ``ascendc_library()`` are enforced rather than left to
fail obscurely: every build starts from an empty build directory (it cannot build
incrementally -- a rebuild dies with ``ld.lld: unknown file type``), and a source
tree under a dot-directory such as ``.claude/worktrees/`` is refused (its object
glob skips dot-directories and the link dies with ``TooFewObj``).

Imports neither ``torch`` nor ``torch_npu``: both are located on disk, because
importing ``torch_npu`` on a build host with no NPU can fail outright.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_SOURCE_DIR = REPO_ROOT / "csrc" / "attention" / "turboquant" / "standalone"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "lib"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build" / "tq_standalone"

LIBRARY_NAME = "libvllm_turboquant_cube.so"

#: ``ascendc_library()`` needs Makefiles: under Ninja, CANN's extract_host_stub.py dies with a KeyError.
CMAKE_GENERATOR = "Unix Makefiles"

_DEFAULT_ASCEND_HOME = Path("/usr/local/Ascend/ascend-toolkit/latest")
_ASCEND_HOME_ENVS = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
_SOC_VERSION_ENV = "SOC_VERSION"
_SOC_FAMILY = "ascend950"
_CMAKE_CACHE = "CMakeCache.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/build_turboquant_ops.py",
        description="Build libvllm_turboquant_cube.so (TurboQuant kernels + torch bindings) for Ascend 950.",
    )
    parser.add_argument(
        "--soc-version",
        default=None,
        help="full CANN SoC variant, e.g. Ascend950PR_9599 (default: $SOC_VERSION, else detected with npu-smi)",
    )
    parser.add_argument("--ascend-home", type=Path, default=None, help="CANN toolkit root (default: $ASCEND_HOME_PATH)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where the .so files go")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="scratch build tree, emptied first (default: build/tq_standalone/<soc>)",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--build-type", default="Release", choices=("Release", "RelWithDebInfo", "Debug"))
    parser.add_argument("--dry-run", action="store_true", help="print the cmake commands and stop")
    return parser


# ------------------------------------------------------------------ inputs


def ascend_home(explicit: Path | None) -> Path:
    """The CANN toolkit root: the flag, then the environment, then the default install."""
    candidates = [explicit] if explicit else []
    candidates += [Path(os.environ[name]) for name in _ASCEND_HOME_ENVS if os.environ.get(name)]
    candidates.append(_DEFAULT_ASCEND_HOME)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(
        f"no CANN toolkit found (looked at {', '.join(map(str, candidates))}); pass --ascend-home or "
        "source <cann>/set_env.sh"
    )


def platform_variants(home: Path) -> list[str]:
    """Every Ascend 950 variant this CANN has a platform config for."""
    return sorted(
        {ini.stem for ini in home.glob("*/data/platform_config/*.ini") if ini.stem.lower().startswith(_SOC_FAMILY)}
    )


def detect_soc_version() -> str | None:
    """``npu-smi``'s chip and NPU name, joined as ``setup.py`` joins them for a 950."""
    try:
        listing = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, check=True).stdout
        npu_id = _field(listing, "NPU ID")
        board = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", npu_id], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    chip, npu = _field(board, "Chip Name", required=False), _field(board, "NPU Name", required=False)
    if chip and npu and "950" in chip:
        return f"{chip}_{npu}"
    return None


def _field(lines: str, key: str, required: bool = True) -> str:
    for line in lines.splitlines():
        name, _, value = line.partition(":")
        if name.strip() == key:
            return value.strip()
    if required:
        raise ValueError(f"npu-smi printed no {key!r}")
    return ""


def resolve_soc_version(explicit: str | None, variants: list[str], detect=detect_soc_version) -> str:
    """The variant to compile for, refusing a family name that does not pick one."""
    soc = explicit or os.environ.get(_SOC_VERSION_ENV) or detect()
    choices = ", ".join(variants) or "none found in this CANN"
    if not soc:
        raise SystemExit(f"no SoC variant: pass --soc-version or set {_SOC_VERSION_ENV} (choices: {choices})")
    if not soc.lower().startswith(_SOC_FAMILY):
        raise SystemExit(f"the TurboQuant Cube decode is Ascend 950 only, got {soc!r} (choices: {choices})")
    if "_" not in soc:
        raise SystemExit(
            f"{soc!r} is a family, not a variant: the variant fixes the core counts the kernels are built for. "
            f"Choose one of: {choices}"
        )
    known = {variant.lower(): variant for variant in variants}
    if variants and soc.lower() not in known:
        raise SystemExit(f"this CANN has no platform config for {soc!r}; choices: {choices}")
    return known.get(soc.lower(), soc)


def package_dir(name: str) -> Path:
    """Where an installed package lives, found without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{name} is not installed in {sys.executable}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def refuse_dot_directories(path: Path) -> None:
    """``ascendc_library()`` globs its objects with ``**``, which never enters a dot-directory."""
    hidden = [part for part in path.resolve().parts if part.startswith(".") and part not in (".", "..")]
    if hidden:
        raise SystemExit(
            f"{path} sits under {hidden[0]!r}: CANN's recompile_binary.py globs its objects with '**', which skips "
            "dot-directories, and the link fails with TooFewObj. Build from a checkout outside dot-directories."
        )


# ------------------------------------------------------------------- build


def cmake_commands(
    soc: str, home: Path, torch_dir: Path, torch_npu_dir: Path, build_dir: Path, output_dir: Path, args
) -> list[list[str]]:
    """Configure, build and install, as argv lists."""
    configure = [
        "cmake",
        "-S",
        str(STANDALONE_SOURCE_DIR),
        "-B",
        str(build_dir),
        "-G",
        CMAKE_GENERATOR,
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DSOC_VERSION={soc}",
        f"-DASCEND_HOME_PATH={home}",
        f"-DTORCH_NPU_PATH={torch_npu_dir}",
        f"-DCMAKE_PREFIX_PATH={torch_dir / 'share' / 'cmake'}",
        f"-DCMAKE_INSTALL_PREFIX={output_dir.resolve()}",
    ]
    build = ["cmake", "--build", str(build_dir), "-j", str(max(1, args.jobs))]
    install = ["cmake", "--install", str(build_dir)]
    return [configure, build, install]


def build_environment(home: Path) -> dict[str, str]:
    """The toolkit on ``PATH`` and ``LD_LIBRARY_PATH``, as ``set_env.sh`` would leave them."""
    env = dict(os.environ)
    env["ASCEND_HOME_PATH"] = str(home)
    env.setdefault("ASCEND_TOOLKIT_HOME", str(home))
    tools = [home / "bin", home / "compiler" / "ccec_compiler" / "bin"]
    env["PATH"] = os.pathsep.join([*(str(p) for p in tools if p.is_dir()), env.get("PATH", "")])
    libraries = [home / "lib64", home / "compiler" / "lib64"]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(p) for p in libraries if p.is_dir()), env.get("LD_LIBRARY_PATH", "")]
    )
    return env


def fresh_build_dir(build_dir: Path) -> None:
    """Empty ``build_dir``, but only if it is empty already or a CMake tree this script could have made."""
    if build_dir.exists():
        if any(build_dir.iterdir()) and not (build_dir / _CMAKE_CACHE).is_file():
            raise SystemExit(f"{build_dir} is not empty and holds no {_CMAKE_CACHE}; refusing to delete it")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    home = ascend_home(args.ascend_home)
    soc = resolve_soc_version(args.soc_version, platform_variants(home))
    refuse_dot_directories(STANDALONE_SOURCE_DIR)
    build_dir = args.build_dir or DEFAULT_BUILD_ROOT / soc.lower()
    commands = cmake_commands(
        soc, home, package_dir("torch"), package_dir("torch_npu"), build_dir, args.output_dir, args
    )

    print(f"[build_turboquant_ops] SoC {soc}, CANN {home}, build tree {build_dir}", file=sys.stderr)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    fresh_build_dir(build_dir)
    env = build_environment(home)
    for command in commands:
        subprocess.run(command, check=True, env=env)
    library = args.output_dir.resolve() / LIBRARY_NAME
    if not library.is_file():
        raise SystemExit(f"the build finished but {library} is missing")
    print(f"[build_turboquant_ops] built {library}", file=sys.stderr)
    print(str(library))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
