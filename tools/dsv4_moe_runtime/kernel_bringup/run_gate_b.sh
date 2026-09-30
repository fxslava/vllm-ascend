#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Gate B: build the CPU-interpreter harness, run the kernel, compare against
# the golden. Runs INSIDE the vendor container; see run_gate_b_docker.ps1 /
# the README for the wrapper.
#
#   bash run_gate_b.sh [io-dir] [hidden] [inter]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IO_DIR="${1:-/work/sim_test}"
HIDDEN="${2:-256}"
INTER="${3:-128}"
SOC="${SOC_TIKICPULIB:-Ascend950PR_9579}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
T="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

echo "=== [1/4] golden generation (hidden=$HIDDEN inter=$INTER) ==="
mkdir -p "$IO_DIR"
python3 "$HERE/gen_golden.py" gen --out-dir "$IO_DIR" --hidden "$HIDDEN" --inter "$INTER"

echo
echo "=== [2/4] configure + build the CPU-interpreter harness ==="
BUILD_DIR="${BUILD_DIR:-/tmp/dsv4_sim_build}"
rm -rf "$BUILD_DIR"
cmake -S "$HERE" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$T/tools/tikicpulib/lib64/cmake" \
      -DSOC_VERSION="$SOC" \
      -Wno-dev
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "=== [3/4] run the kernel on the CPU interpreter ==="
"$BUILD_DIR/dsv4_moe_expert_sim" "$IO_DIR" "$HIDDEN" "$INTER"

echo
echo "=== [4/4] Gate B comparison ==="
python3 "$HERE/gen_golden.py" cmp \
        --golden "$IO_DIR/golden.bin" \
        --actual "$IO_DIR/actual.bin" \
        --meta   "$IO_DIR/meta.json"
