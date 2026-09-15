#!/bin/bash
# -----------------------------------------------------------------------------
# Ascend 950PR: hardware correctness check and performance baseline.
#
# Run this ON a host with a physical 950PR attached, against a tree built with
# RUN_MODE=npu. It is the first thing to run on silicon after the Cube-native
# decode was corrected on the camodel (see TURBOQUANT_TESTS.md section 13.8),
# and the numbers it produces are the baseline to beat once the manual NZ
# staging is removed.
#
#   ./scripts/run_950pr_npu_baseline.sh [BUILD_DIR]
#
# BUILD_DIR defaults to build/kv4fp8-950-npu. Configure it with:
#
#   cmake -S csrc/tests -B build/kv4fp8-950-npu -G "Unix Makefiles" \
#     -DCMAKE_BUILD_TYPE=Release -DENABLE_ASCEND_950PR=ON \
#     -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=npu \
#     -DVLLM_ASCEND_TESTS_WERROR=ON
#   cmake --build build/kv4fp8-950-npu -j 8
#
# Build the DEFAULT target, not a list of --target flags: naming more than one
# makes the ascendc_library ExternalProject re-enter and fail against its own
# just-built objects with "ld.lld: unknown file type".
# -----------------------------------------------------------------------------
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

# --- preflight ---------------------------------------------------------------
#
# A real part, not a camodel. test_device_950pr_turboquant refuses the simulator
# at run time by reading /proc/self/maps, but the benchmark only exits 77, so
# check here rather than discover it in the results.
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

# The Cube legs are on by default in the benchmark; the multimode test still
# gates behind the same variable, so set it explicitly for both.
export VLLM_ASCEND_TQ_CUBE_WIP=1

# --- stage 1: the AIV-only path, as a control --------------------------------
#
# This drives turboquant_kernels.cpp only and touches nothing that changed in
# the Cube fix. If it fails, the problem is the machine or the build, not the
# decode, and stages 2-4 are not worth reading.
echo
echo "===== stage 1: AIV-only bare-metal correctness (control) ====="
"$DEV/test_device_950pr_turboquant" > "$OUT/stage1_aiv.log" 2>&1
rc1=$?
echo "exit $rc1"
grep -E "\[  *(OK|FAILED|PASSED|SKIPPED) *\]" "$OUT/stage1_aiv.log" | tail -12

# --- stage 2: the Cube decode's fidelity, S=64 and S=512 ---------------------
#
# cos against an fp32 host reference. The camodel measured 0.982351 at S=64;
# tq_multimode_calibration.py puts this rate at 0.98785 at S=512. Silicon
# should land in the same place -- a materially different number means the
# camodel and the part disagree, which is a finding in itself.
#
# S=64 is a single 64-row Cube tile. S=512 is the first shape that exercises
# multi-tile addressing and real paging, so run both: a pass at 64 alone does
# not cover the tile loop.
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
  # The bound is the test's own structural gate, restated so the summary below
  # does not depend on reading the gtest output.
  awk -v c="${cos:-0}" -v s="$S" 'BEGIN{
    if (c+0 > 0.90) printf "  PASS  S=%s cos=%s > 0.90\n", s, c;
    else            printf "  FAIL  S=%s cos=%s <= 0.90\n", s, c;
  }'
  awk -v c="${cos:-0}" 'BEGIN{ exit (c+0 > 0.90) ? 0 : 1 }' || cos_fail=1
  grep -cE "vec_err_idata_inf_nan|unrecognize ldst|su_ccu|misalign" \
    "$OUT/stage2_cos_s$S.log" | xargs -I{} echo "  device error lines: {}"
done

# --- stage 3: the performance baseline ---------------------------------------
#
# This is the number the NZ-staging work has to beat. The binary is an
# end-to-end audit now (csrc/tests/TURBOQUANT_TESTS.md 7.5): Table B breaks a
# decode step into rotate-q, split, combine and rotate-o, sums them into TQ_E2E
# and divides the native aclnnFusedInferAttentionScoreV5 decode by it.
#
# Restricted to the two short regimes and the small batches. The full sweep is
# 27 configurations across four context regimes up to 1M and both phases, which
# is hours; this stage wants a decode baseline that finishes.
#
# Leave ASCEND_BENCH_WARMUP alone: it no longer reaches this binary (the audit
# pins its own 5/20 and 1/3 budgets), but the other stages still read it, and a
# checksummed case reads its reference on the first warmup launch.
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
# To the footer, not to the first blank line: the table's banner has blank lines
# in it, so a /^$/ range stops before any row is printed.
sed -n '/TABLE B/,/Speedup is V5_Decode/p' "$OUT/stage3_bench.log"
grep -E "^  (Qwen|DeepSeek|GLM)" "$OUT/stage3_bench.log" | tail -20

# --- stage 4: where the time goes --------------------------------------------
#
# PipeUtilization splits Cube against the MTE and vector pipes, which is the
# measurement that decides whether removing the manual NZ staging is worth it:
# if MTE3 and the vector unpack dominate, it is.
#
# task-time=l1 gives per-kernel durations. Profiling multiplies run time, so
# this uses a single context length rather than the sweep.
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

# --- summary -----------------------------------------------------------------
echo
echo "########## summary ##########"
echo "stage 1 AIV control      exit $rc1"
echo "stage 2 Cube fidelity    $([ "$cos_fail" -eq 0 ] && echo 'cos > 0.90 at both S' || echo 'BELOW GATE -- see logs')"
echo "stage 3 benchmark        exit $rc3"
echo "logs and CSV under $OUT"
