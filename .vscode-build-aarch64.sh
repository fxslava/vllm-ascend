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

aarch64-linux-gnu-strip --strip-unneeded build/csrc-tests-aarch64/test_rmsnorm_310p build/csrc-tests-aarch64/test_rotary_embedding_310p build/csrc-tests-aarch64/test_activation_swiglu_310p build/csrc-tests-aarch64/test_paged_attention_310p build/csrc-tests-aarch64/test_matmul_310p
