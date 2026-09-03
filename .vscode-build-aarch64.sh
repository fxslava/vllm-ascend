set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PATH="${ASCEND_TOOLKIT_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"

cd /vllm-workspace/vllm-ascend
cmake -S csrc/tests -B build/csrc-tests-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=/vllm-workspace/vllm-ascend/cmake/aarch64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=ascend310p3

cmake --build build/csrc-tests-aarch64 -j"$(nproc)"

# csrc/tests/CMakeLists.txt writes one absolute path per line for every
# executable it built. Driving strip from that instead of a hand-maintained list
# keeps it from being handed the gtest_discover_tests *_include.cmake files that
# sit alongside the binaries, which is what produced the "file format not
# recognized" warnings, and picks up new bench_* targets without another edit.
xargs -r -a build/csrc-tests-aarch64/test_binaries.txt aarch64-linux-gnu-strip --strip-unneeded
