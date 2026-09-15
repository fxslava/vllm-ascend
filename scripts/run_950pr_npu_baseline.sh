#!/bin/bash
# Usage: ./scripts/run_950pr_npu_baseline.sh [BUILD_DIR]
set -o pipefail

BUILD="${1:-build/kv4fp8-950-npu}"
DEV="$BUILD/device"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="artifacts/npu-baseline-$STAMP"

if [ -z "$ASCEND_HOME_PATH" ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
mkdir -p "$OUT"

echo "########## Ascend 950PR baseline  $STAMP ##########"
echo "build:  $BUILD"
echo "output: $OUT"

echo
echo "===== preflight ====="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>&1 | tee "$OUT/npu-smi.log" | head -20
else
  echo "WARNING: npu-smi not found -- cannot confirm a 950PR is attached."
fi

for b in test_device_950pr_turboquant test_device_950pr_turboquant_multimode \
         bench_device_950pr_turboquant; do
  if [ ! -x "$DEV/$b" ]; then
    echo "MISSING: $DEV/$b -- build with RUN_MODE=npu first."
    exit 1
  fi
done
echo "all three binaries present."

export VLLM_ASCEND_TQ_CUBE_WIP=1

echo
echo "===== stage 1: AIV-only bare-metal correctness (control) ====="
"$DEV/test_device_950pr_turboquant" > "$OUT/stage1_aiv.log" 2>&1
rc1=$?
echo "exit $rc1"
grep -E "\[  *(OK|FAILED|PASSED|SKIPPED) *\]" "$OUT/stage1_aiv.log" | tail -12

echo
echo "===== stage 2: Cube decode fidelity (kv4fp8) ====="
export ASCEND_TQ_SIM_MODES=kv4fp8
cos_fail=0
for S in 64 512; do
  echo "--- S=$S ---"
  ASCEND_TQ_SIM_CONTEXT="$S" "$DEV/test_device_950pr_turboquant_multimode" \
    > "$OUT/stage2_cos_s$S.log" 2>&1
  rc=$?
  cos=$(grep -oE "cos vs fp32 host reference = [-0-9.naif]+" "$OUT/stage2_cos_s$S.log" \
        | tail -1 | awk '{print $NF}')
  echo "exit $rc   cos=${cos:-<none>}"
  awk -v c="${cos:-0}" -v s="$S" 'BEGIN{
    if (c+0 > 0.90) printf "  PASS  S=%s cos=%s > 0.90\n", s, c;
    else            printf "  FAIL  S=%s cos=%s <= 0.90\n", s, c;
  }'
  awk -v c="${cos:-0}" 'BEGIN{ exit (c+0 > 0.90) ? 0 : 1 }' || cos_fail=1
  grep -cE "vec_err_idata_inf_nan|unrecognize ldst|su_ccu|misalign" \
    "$OUT/stage2_cos_s$S.log" | xargs -I{} echo "  device error lines: {}"
done

echo
echo "===== stage 3: decode latency baseline ====="
export ASCEND_BENCH_CSV="$OUT/bench.csv"
ASCEND_BENCH_TQ_AUDIT_PHASES=decode \
ASCEND_BENCH_TQ_AUDIT_S=2048,32768 \
ASCEND_BENCH_TQ_AUDIT_B=1,4 \
ASCEND_BENCH_TQ_AUDIT_DECODE_CSV="$OUT/decode_audit.csv" \
  "$DEV/bench_device_950pr_turboquant" > "$OUT/stage3_bench.log" 2>&1
rc3=$?
echo "exit $rc3  (77 = no usable 950PR attached)"
sed -n '/TABLE B/,/Speedup is V5_Decode/p' "$OUT/stage3_bench.log"
grep -E "^  (Qwen|DeepSeek|GLM)" "$OUT/stage3_bench.log" | tail -20

echo
echo "===== stage 4: msprof pipe utilisation ====="
if command -v msprof >/dev/null 2>&1; then
  ASCEND_BENCH_TQ_AUDIT_PHASES=decode ASCEND_BENCH_TQ_AUDIT_S=2048 \
  ASCEND_BENCH_TQ_AUDIT_B=1 ASCEND_BENCH_TQ_AUDIT_MODELS=dsv4 \
  msprof --application="$DEV/bench_device_950pr_turboquant" \
         --output="$OUT/prof" \
         --ai-core=on \
         --aic-metrics=PipeUtilization \
         --task-time=l1 \
         --runtime-api=on \
         > "$OUT/stage4_msprof.log" 2>&1
  echo "exit $?   data under $OUT/prof"
  echo "summarise with:  msprof --export=on --output=$OUT/prof"
else
  echo "msprof not on PATH; skipping. It ships at \$ASCEND_HOME_PATH/bin/msprof."
fi

echo
echo "########## summary ##########"
echo "stage 1 AIV control      exit $rc1"
echo "stage 2 Cube fidelity    $([ "$cos_fail" -eq 0 ] && echo 'cos > 0.90 at both S' || echo 'BELOW GATE -- see logs')"
echo "stage 3 benchmark        exit $rc3"
echo "logs and CSV under $OUT"
