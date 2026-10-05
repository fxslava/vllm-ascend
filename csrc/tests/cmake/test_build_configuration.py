# SPDX-License-Identifier: Apache-2.0
"""Toolkit-free regression tests for CANN precedence and offline GTest setup."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULES = Path(__file__).resolve().parent


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def configure(self, module, definitions=(), environment=None):
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(configuration NONE)\n"
            f'include("{(MODULES / module).as_posix()}")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "${ASCEND_HOME_PATH}")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/major.txt" "${CANN_VERSION_MAJOR}")\n'
        )
        env = os.environ.copy()
        env.pop("ASCEND_HOME_PATH", None)
        env.pop("ASCEND_TOOLKIT_HOME", None)
        env.update(environment or {})
        return subprocess.run(
            ["cmake", "-S", str(source), "-B", str(self.root / "build"), *definitions],
            env=env, capture_output=True, text=True, check=False,
        )

    def toolkit(self, name):
        root = self.root / name
        header = root / "include/acl/acl.h"
        header.parent.mkdir(parents=True)
        header.touch()
        return root

    def test_explicit_toolkit_wins(self):
        explicit = self.toolkit("explicit")
        result = self.configure("CannToolkit.cmake", [f"-DASCEND_HOME_PATH={explicit}"],
                                {"ASCEND_TOOLKIT_HOME": str(self.toolkit("environment"))})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "build/resolved.txt").read_text(), str(explicit))

    def test_toolkit_environment_precedes_home_environment(self):
        preferred = self.toolkit("preferred")
        result = self.configure("CannToolkit.cmake", environment={
            "ASCEND_TOOLKIT_HOME": str(preferred),
            "ASCEND_HOME_PATH": str(self.toolkit("secondary")),
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "build/resolved.txt").read_text(), str(preferred))

    def test_invalid_explicit_toolkit_has_checked_path(self):
        missing = self.root / "missing"
        result = self.configure("CannToolkit.cmake", [f"-DASCEND_HOME_PATH={missing}"],
                                {"ASCEND_TOOLKIT_HOME": str(self.toolkit("valid"))})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(missing), result.stderr)
        self.assertIn("Explicit ASCEND_HOME_PATH is invalid", result.stderr)

    def test_offline_source_works_with_fetch_disabled(self):
        gtest = self.root / "googletest"
        (gtest / "googletest").mkdir(parents=True)
        (gtest / "googletest/CMakeLists.txt").touch()
        (gtest / "CMakeLists.txt").write_text(
            "add_library(gtest INTERFACE)\nadd_library(gtest_main INTERFACE)\n"
        )
        result = self.configure("GoogleTestProvider.cmake", [
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={gtest}",
            "-DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local GoogleTest", result.stdout)

    def test_invalid_offline_source_is_actionable(self):
        result = self.configure("GoogleTestProvider.cmake", [
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={self.root / 'missing'}",
            "-DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid offline GoogleTest source", result.stderr)

    def assert_major(self, root, expected, extra=()):
        result = self.configure("CannVersion.cmake", [f"-DASCEND_HOME_PATH={root}", *extra])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "build/major.txt").read_text(), str(expected))

    def test_cann8_product_version_precedes_component_version(self):
        root = self.toolkit("latest")
        (root / "version.info").write_text("Version=7.6.0.1.220\nversion_dir=8.0.0\n")
        self.assert_major(root, 8)

    def test_cann9_compiler_metadata(self):
        root = self.toolkit("latest")
        (root / "compiler").mkdir()
        (root / "compiler/version.info").write_text("Version=9.2.0-beta.2\nversion_dir=cann\n")
        self.assert_major(root, 9)

    def test_cann_version_path_fallback(self):
        self.assert_major(self.toolkit("8.0.0"), 8)

    @unittest.skipIf(os.name == "nt", "Directory symlink creation may require administrator rights")
    def test_latest_symlink_resolves_version_path(self):
        root = self.toolkit("9.1.0")
        latest = self.root / "latest"
        latest.symlink_to(root, target_is_directory=True)
        self.assert_major(latest, 9)

    def test_manual_cann_version_overrides_metadata(self):
        root = self.toolkit("latest")
        (root / "version.info").write_text("Version=9.2.0-beta.2\n")
        self.assert_major(root, 8, ["-DCANN_VERSION_MAJOR=8"])

    def test_unknown_cann_version_requests_override(self):
        result = self.configure("CannVersion.cmake", [f"-DASCEND_HOME_PATH={self.toolkit('latest')}"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pass -DCANN_VERSION_MAJOR=8 or 9", " ".join(result.stderr.split()))

    def test_invalid_cann_major_is_rejected(self):
        result = self.configure("CannVersion.cmake", ["-DCANN_VERSION_MAJOR=banana"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN_VERSION_MAJOR must be an integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
