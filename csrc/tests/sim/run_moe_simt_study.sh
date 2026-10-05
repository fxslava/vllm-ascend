#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail
build=$(realpath "${1:?pass the configured simulator build directory}")
output=$(realpath -m "${2:?pass a fresh results directory}")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output"
# CANN 9.2 npusim maps this CLI alias to Ascend950PR_9599/camodel.
# Check npusim.log before treating results as 9599 evidence on another version.
for hidden in 256 512; do
  for path in simd simt simt-cg simt-lut; do
    name="${path}_${hidden}x128"
    mkdir -p "$output/$name/work"
    cd "$output/$name/work"
    npusim record -s Ascend950DT -n 0 -g vf -o "$output/$name" \
      "$build/sim/bench_sim_950pr_moe_simt_vs_simd $path $hidden 128" \
      >"$output/$name.log" 2>&1
    grep -q 'PASS path=' "$output/$name.log"
    grep -q 'Ascend950PR_9599/camodel' "$output/$name.log"
    python3 "$source_dir/extract_moe_simt_metrics.py" "$output/$name" \
      --elements "$((hidden * 128))" --output "$output/$name.json"
  done
done
