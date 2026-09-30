#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Gate C: build the kernel as device object code, run it on the CANN camodel
# under npusim/cannsim, and emit the simulator report. Runs INSIDE the vendor
# container.
#
#   bash run_gate_c.sh [io-dir] [hidden] [inter]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IO_DIR="${1:-/work/sim_test_c}"
HIDDEN="${2:-256}"
INTER="${3:-128}"
# Two different identifiers, deliberately:
#   SOC   the toolchain / camodel part, which is what names the directories
#         under tools/simulator and what ascendc_library compiles for. The
#         simulator ships no Ascend950PR_9579, and CANN folds every C310 part
#         onto the 9599 series (tikicpulib-config.cmake product_map()), so 9599
#         is the documented stand-in and ticks are labelled approximate.
#   NPUSIM_SOC  what `npusim record -s` accepts, which is the FAMILY name only:
#         it rejects Ascend950PR_9599 with "Supported: ['Ascend950',
#         'Ascend950DT']".
SOC="${SOC_SIM:-Ascend950PR_9599}"
NPUSIM_SOC="${NPUSIM_SOC:-Ascend950}"
BUILD_DIR="${BUILD_DIR:-/tmp/dsv4_gatec_build}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SIM_LIB="$ASCEND_HOME_PATH/tools/simulator/$SOC/lib"

echo "=== [1/4] golden data ==="
mkdir -p "$IO_DIR"
python3 "$HERE/gen_golden.py" gen --out-dir "$IO_DIR" --hidden "$HIDDEN" --inter "$INTER"

echo
echo "--- camodel libraries for $SOC ---"
ls "$SIM_LIB" 2>&1 | tr '\n' ' '; echo

echo
echo "=== [2/4] build device object code for $SOC (RUN_MODE=sim) ==="
rm -rf "$BUILD_DIR"
cmake -S "$HERE/gate_c" -B "$BUILD_DIR" -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DSOC_VERSION="$SOC" \
      -DRUN_MODE=sim \
      -Wno-dev
cmake --build "$BUILD_DIR" -j"$(nproc)"

# The camodel directory has to precede the toolkit's so libruntime_camodel.so
# wins over libruntime.so; $BUILD_DIR/lib is where ascendc_library leaves the
# kernel library the executable needs at load time.
export LD_LIBRARY_PATH="$SIM_LIB:$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"

echo
echo "=== [3/5] plain camodel run (correctness incl. pipe synchronisation) ==="
rm -f "$IO_DIR/actual.bin"
timeout 1200 "$BUILD_DIR/dsv4_gate_c" "$IO_DIR" "$HIDDEN" "$INTER"

echo
echo "=== [4/5] Gate C numerics ==="
if [ -f "$IO_DIR/actual.bin" ]; then
    python3 "$HERE/gen_golden.py" cmp --golden "$IO_DIR/golden.bin" \
            --actual "$IO_DIR/actual.bin" --meta "$IO_DIR/meta.json"
    cp "$IO_DIR/actual.bin" "$IO_DIR/actual_camodel.bin"
else
    echo "no actual.bin produced"
fi

echo
echo "=== [5/5] profiled run under npusim record (-s $NPUSIM_SOC) ==="
REPORT_DIR="$IO_DIR/simreport"
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"
( cd "$REPORT_DIR" && timeout 1800 "$ASCEND_HOME_PATH/bin/cannsim" record \
    -s "$NPUSIM_SOC" \
    -o "$REPORT_DIR" \
    -g \
    -n 0 \
    "$BUILD_DIR/dsv4_gate_c $IO_DIR $HIDDEN $INTER" ) 2>&1 | tail -40

echo
echo "--- report tree ---"
find "$REPORT_DIR" -type f | head -40
