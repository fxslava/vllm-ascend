set -euo pipefail

cd /vllm-workspace/vllm-ascend

echo "=== Packaging AArch64 Test Binaries ==="
mkdir -p dist/ascend310p3-aarch64-tests
cp -f build/csrc-tests-aarch64/test_*310p dist/ascend310p3-aarch64-tests/
tar -czvf ascend310p3-aarch64-csrc-tests.tar.gz -C dist ascend310p3-aarch64-tests
echo "=== Packaging Complete ==="