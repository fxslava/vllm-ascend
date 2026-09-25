#!/usr/bin/env bash
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
# Stage 1 of the Ascend 950PR TurboQuant hardware gate: the bare-metal kernel
# checks, run straight through the CANN runtime with no Python, no PyTorch and no
# vLLM. It is the gate a benchmark run should never be started without.
#
# Nothing here needs a checkpoint or a dataset. Everything here needs the part.

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Defaults
# ---------------------------------------------------------------------------- #

# Where the RUN_MODE=npu tier was built. `device/` and `lib/` live under it.
BUILD_DIR="${TQ_BUILD_DIR:-build/csrc-tests-npu}"

# The context lengths every stage sweeps. 64 is one tile and one split; 512 is
# several tiles inside the fused limit. Both are seconds on silicon.
CONTEXTS="${TQ_CONTEXTS:-64,512}"

OUT_DIR=""
RUN_BENCH=1
ALLOW_SIMULATOR=0

# The decode kernel against the CPU reference **running the same arithmetic over
# the same quantised cache**. This is a kernel-correctness gate: the two agree to
# fp16 output rounding or the kernel is wrong. It is the bound compiled into
# test_device_950pr_turboquant as kMinDecodeCosine and is re-asserted here so the
# script halts on it even if a future build loosens the constant.
MIN_DECODE_COSINE="${TQ_MIN_DECODE_COSINE:-0.999}"

# The Cube (kv4fp8) decode against *exact fp32 attention over unquantised K/V*.
# A different question, and a much looser number: this is the 4-bit cache's own
# quantisation error, measured at cos 0.98629 (min over four seeds, d=256, S=512)
# by scripts/tq_multimode_calibration.py. Gating it at 0.999 would fail every
# correct build, so it is gated just under the measured floor instead.
MIN_CUBE_QUANT_COSINE="${TQ_MIN_CUBE_QUANT_COSINE:-0.98}"

# One context above the kernel's fused limit (4096), so the lse case reaches
# TurboQuantPartialReducer's write and not only the fused writer. A token is split
# -- and so takes its pair out of the reducer -- only past that limit, which
# CONTEXTS above deliberately does not reach.
LSE_SPLIT_CONTEXT="${TQ_LSE_SPLIT_CONTEXT:-8192}"

# Which models and shapes the unpack ablation prices. Kept to one small decode
# point: this stage is asking "does the ablated launch still run and still land on
# the same traffic", not "what is the throughput curve" -- that is stage 2's job.
BENCH_MODELS="${TQ_BENCH_MODELS:-dsv4}"
BENCH_S="${TQ_BENCH_S:-2048}"
BENCH_B="${TQ_BENCH_B:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/run_950pr_baremetal_validation.sh [options]

Stage 1 of the 950PR TurboQuant gate: bare-metal kernel validation. No Python,
no checkpoint, no dataset -- just the CANN runtime and the part.

What it measures, in order:

  0  npu-smi preflight               the part is attached, and which one
  1  binaries present                the RUN_MODE=npu tier was actually built
  2  AIV decode vs CPU reference     write path, packed cache bytes, decode
                                     fidelity, cache geometry, slot containment,
                                     launch determinism -- and the two lse cases:
                                       * the decode's (max, mass) out-tensor
                                         against an independent host derivation,
                                         over --contexts plus one context above
                                         the kernel's fused limit so that the
                                         reducer's write is covered too
                                       * that attaching it leaves the decode's
                                         own output bit-identical
                                     GATE: decode cos > MIN_DECODE_COSINE
                                     (kernel vs reference, same quantised cache)
  3  Cube kv4fp8 quantisation        the one-launch Cube decode against exact
                                     fp32 attention over unquantised K/V
                                     GATE: cos > MIN_CUBE_QUANT_COSINE
                                     (this is the 4-bit cache's own error, NOT a
                                      kernel gate -- see the constants in-file)
  4  unpack ablation                 ASCEND_BENCH_TQ_UNPACK=both: the shipping
                                     decode and the same launch with the KV
                                     nibble expand compiled out, at identical
                                     traffic. Prices the unpack phase.

Options:
  --build-dir DIR      RUN_MODE=npu build tree (default: build/csrc-tests-npu,
                       or $TQ_BUILD_DIR)
  --contexts LIST      comma-separated context lengths (default: 64,512)
  --lse-split-context N
                       the extra context the lse case sweeps so that a token is
                       actually split and its pair comes out of the in-launch
                       reducer (default: 8192; the kernel fuses at or below 4096)
  --out DIR            where logs and CSVs land
                       (default: reports/npu_950pr_baremetal_<timestamp>)
  --skip-bench         stop after stage 3; do not run the unpack ablation
  --allow-simulator    let the device tests run against a CAModel. They refuse
                       one by default, and a skipped test is treated as a
                       FAILURE here, because a skip exits 0 and would otherwise
                       read as a pass.
  -h, --help           this text

Exit: 0 all gates passed. 1 a gate failed or a binary is missing. 2 no NPU.
EOF
}

# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) BUILD_DIR="${2:?--build-dir needs a directory}"; shift 2 ;;
    --contexts) CONTEXTS="${2:?--contexts needs a list}"; shift 2 ;;
    --lse-split-context) LSE_SPLIT_CONTEXT="${2:?--lse-split-context needs a number}"; shift 2 ;;
    --out) OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
    --skip-bench) RUN_BENCH=0; shift ;;
    --allow-simulator) ALLOW_SIMULATOR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-reports/npu_950pr_baremetal_$STAMP}"
DEV="$BUILD_DIR/device"
mkdir -p "$OUT_DIR"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -r "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi
# The kernel library sits beside the binaries rather than on the system path. Tolerant
# of a build directory that is not there: stage 1 is what reports that, with the cmake
# line to fix it, and it must not be pre-empted by a bare `cd` failure under set -e.
if [ -d "$BUILD_DIR/lib" ]; then
  export LD_LIBRARY_PATH="$(cd "$BUILD_DIR/lib" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ "$ALLOW_SIMULATOR" -eq 1 ]; then
  export ASCEND_TEST_ALLOW_SIMULATOR=1
fi

FAILURES=0
note_failure() {
  echo "  !! $1"
  FAILURES=$((FAILURES + 1))
}

banner() {
  echo
  echo "=============================================================================="
  echo "$1"
  echo "=============================================================================="
}

echo "##############################################################################"
echo "# Ascend 950PR TurboQuant -- stage 1, bare-metal kernel validation"
echo "# $STAMP"
echo "##############################################################################"
echo "build dir            : $BUILD_DIR"
echo "contexts             : $CONTEXTS"
echo "lse split context    : $LSE_SPLIT_CONTEXT   (above the 4096 fused limit, so the reducer writes too)"
echo "output               : $OUT_DIR"
echo "decode gate          : cos > $MIN_DECODE_COSINE   (kernel vs CPU reference, same cache)"
echo "cube quant gate      : cos > $MIN_CUBE_QUANT_COSINE      (kv4fp8 vs exact fp32, unquantised K/V)"
echo "unpack ablation      : $([ "$RUN_BENCH" -eq 1 ] && echo 'stage 4 enabled' || echo 'skipped (--skip-bench)')"
echo "ASCEND_HOME_PATH     : ${ASCEND_HOME_PATH:-<unset>}"

# ---------------------------------------------------------------------------- #
# Stage 0: preflight
# ---------------------------------------------------------------------------- #

banner "stage 0: npu-smi preflight"
if ! command -v npu-smi >/dev/null 2>&1; then
  echo "npu-smi is not on PATH. This stage measures silicon and there is none here."
  echo "Source the CANN environment, or run with --allow-simulator if you really mean"
  echo "to drive a CAModel (hours per case)."
  [ "$ALLOW_SIMULATOR" -eq 1 ] || exit 2
else
  if npu-smi info > "$OUT_DIR/npu-smi.log" 2>&1; then
    head -20 "$OUT_DIR/npu-smi.log"
    CHIPS="$(grep -cE '^\| *[0-9]+ +[0-9]+ ' "$OUT_DIR/npu-smi.log" || true)"
    echo "  npu-smi reports $CHIPS chip row(s); full output in $OUT_DIR/npu-smi.log"
  else
    echo "npu-smi exited non-zero; see $OUT_DIR/npu-smi.log"
    [ "$ALLOW_SIMULATOR" -eq 1 ] || exit 2
  fi
fi

# ---------------------------------------------------------------------------- #
# Stage 1: binaries
# ---------------------------------------------------------------------------- #

banner "stage 1: binaries"
REQUIRED=(test_device_950pr_turboquant test_device_950pr_turboquant_multimode)
[ "$RUN_BENCH" -eq 1 ] && REQUIRED+=(bench_device_950pr_turboquant)
for b in "${REQUIRED[@]}"; do
  if [ -x "$DEV/$b" ]; then
    echo "  ok      $DEV/$b"
  else
    echo "  MISSING $DEV/$b"
    echo
    echo "Build the npu tier first:"
    echo "  cmake -S csrc/tests -B $BUILD_DIR -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \\"
    echo "        -DVLLM_ASCEND_TESTS_WERROR=ON -DENABLE_ASCEND_950PR=ON \\"
    echo "        -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=npu \\"
    echo "        -DVLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS=ON"
    echo "  cmake --build $BUILD_DIR -j\$(nproc)"
    exit 1
  fi
done

# ---------------------------------------------------------------------------- #
# Stage 2: AIV decode, lse out-tensor, and everything else bare metal
# ---------------------------------------------------------------------------- #

banner "stage 2: AIV decode vs the CPU reference, and the lse out-tensor"
LOG2="$OUT_DIR/stage2_baremetal.log"
set +e
ASCEND_TQ_BARE_METAL_CONTEXTS="$CONTEXTS" \
ASCEND_TQ_LSE_SPLIT_CONTEXT="$LSE_SPLIT_CONTEXT" \
  "$DEV/test_device_950pr_turboquant" > "$LOG2" 2>&1
RC2=$?
set -e
echo "  exit $RC2"
grep -E "^\[ *(RUN|OK|FAILED|SKIPPED) *\]|^\[ *PASSED|^\[ *FAILED" "$LOG2" | tail -30 || true
echo
echo "  --- fidelity and lse lines ---"
grep -E "^  (decode|lse|lse-off|write|determinism)" "$LOG2" || true

# A skipped test exits 0. That would read as a pass, so it is a failure here.
SKIPPED="$(grep -cE "^\[ *SKIPPED *\]" "$LOG2" || true)"
if [ "${SKIPPED:-0}" -gt 0 ]; then
  note_failure "$SKIPPED case(s) SKIPPED -- this build refused the device it was given"
  grep -E "Test targets a physical Ascend 950PR" "$LOG2" | head -2 || true
fi
[ "$RC2" -eq 0 ] || note_failure "test_device_950pr_turboquant exited $RC2"

# The decode cosine gate, re-asserted here rather than trusted to the binary.
DECODE_LINES="$(grep -cE "^  decode S=" "$LOG2" || true)"
if [ "${DECODE_LINES:-0}" -eq 0 ]; then
  note_failure "no 'decode S=... cos=...' line was produced -- the decode case never ran"
else
  echo
  echo "  --- decode cosine gate (> $MIN_DECODE_COSINE) ---"
  if ! grep -E "^  decode S=" "$LOG2" |
       awk -v bound="$MIN_DECODE_COSINE" '
         match($0, /cos=[0-9.]+/) {
           cos = substr($0, RSTART + 4, RLENGTH - 4) + 0
           s = $2
           if (cos > bound) { printf "    PASS  %s cos=%.6f\n", s, cos }
           else             { printf "    FAIL  %s cos=%.6f <= %s\n", s, cos, bound; bad = 1 }
         }
         END { exit bad ? 1 : 0 }'; then
    note_failure "a decode cosine fell to or below $MIN_DECODE_COSINE"
  fi
fi

# The lse bound, likewise: the binary prints the drift and its own bound.
LSE_LINES="$(grep -cE "worst \|d log-mass\|=" "$LOG2" || true)"
if [ "${LSE_LINES:-0}" -eq 0 ]; then
  note_failure "no lse drift line was produced -- this build has no lse out-tensor case"
else
  echo
  echo "  --- lse log-mass drift (against each line's own bound) ---"
  if ! grep -E "worst \|d log-mass\|=" "$LOG2" |
       awk '
         match($0, /log-mass\|=[0-9.eE+-]+/) {
           drift = substr($0, RSTART + 10, RLENGTH - 10) + 0
           bound = 0
           if (match($0, /\(bound [0-9.]+\)/)) {
             bound = substr($0, RSTART + 7, RLENGTH - 8) + 0
           }
           s = $2
           if (bound > 0 && drift < bound) { printf "    PASS  %s |d log-mass|=%.6f < %.3f\n", s, drift, bound }
           else { printf "    FAIL  %s |d log-mass|=%.6f (bound %.3f)\n", s, drift, bound; bad = 1 }
         }
         END { exit bad ? 1 : 0 }'; then
    note_failure "the decode's reported softmax mass drifted past its bound"
  fi
fi

# ---------------------------------------------------------------------------- #
# Stage 3: the Cube decode's quantisation fidelity
# ---------------------------------------------------------------------------- #

banner "stage 3: kv4fp8 Cube decode vs exact fp32 attention"
echo "  This is the 4-bit cache's own error, not a kernel gate. kv4fp8 measures"
echo "  cos 0.98629 at worst on the host calibration; the gate is $MIN_CUBE_QUANT_COSINE."
CUBE_FAIL=0
IFS=',' read -r -a CONTEXT_LIST <<< "$CONTEXTS"
for S in "${CONTEXT_LIST[@]}"; do
  LOG3="$OUT_DIR/stage3_cube_s$S.log"
  set +e
  ASCEND_TQ_SIM_MODES=kv4fp8 ASCEND_TQ_SIM_CONTEXT="$S" ASCEND_TQ_SIM_BATCH=1 \
    "$DEV/test_device_950pr_turboquant_multimode" > "$LOG3" 2>&1
  RC3=$?
  set -e
  COS="$(grep -oE "cos vs fp32 host reference = [-0-9.naife]+" "$LOG3" | tail -1 | awk '{print $NF}' || true)"
  echo "  S=$S exit $RC3   cos=${COS:-<none>}"
  if [ -z "${COS:-}" ]; then
    note_failure "S=$S produced no cosine -- see $LOG3"
    CUBE_FAIL=1
    continue
  fi
  if awk -v c="$COS" -v b="$MIN_CUBE_QUANT_COSINE" 'BEGIN { exit (c + 0 > b + 0) ? 0 : 1 }'; then
    echo "    PASS  S=$S cos=$COS > $MIN_CUBE_QUANT_COSINE"
  else
    echo "    FAIL  S=$S cos=$COS <= $MIN_CUBE_QUANT_COSINE"
    CUBE_FAIL=1
  fi
  DEVICE_ERRORS="$(grep -cE "vec_err_idata_inf_nan|unrecognize ldst|su_ccu|misalign" "$LOG3" || true)"
  [ "${DEVICE_ERRORS:-0}" -eq 0 ] || echo "    device error lines in the log: $DEVICE_ERRORS"
done
[ "$CUBE_FAIL" -eq 0 ] || note_failure "the kv4fp8 Cube decode fell below $MIN_CUBE_QUANT_COSINE"

# ---------------------------------------------------------------------------- #
# Stage 4: the unpack ablation
# ---------------------------------------------------------------------------- #

if [ "$RUN_BENCH" -eq 1 ]; then
  banner "stage 4: unpack ablation (ASCEND_BENCH_TQ_UNPACK=both)"
  echo "  dec_attn_core is the shipping decode; dec_nounpack is the same launch with the"
  echo "  KV nibble expand compiled out and every byte of traffic left in place, so the"
  echo "  difference between the two rows is the unpack phase. The ablated launch stages"
  echo "  zeros, so its output checksum means nothing by construction."
  LOG4="$OUT_DIR/stage4_unpack.log"
  set +e
  ASCEND_BENCH_TQ_UNPACK=both \
  ASCEND_BENCH_TQ_AUDIT_PHASES=decode \
  ASCEND_BENCH_TQ_AUDIT_MODELS="$BENCH_MODELS" \
  ASCEND_BENCH_TQ_AUDIT_S="$BENCH_S" \
  ASCEND_BENCH_TQ_AUDIT_B="$BENCH_B" \
  ASCEND_BENCH_TQ_AUDIT_DECODE_CSV="$OUT_DIR/decode_audit.csv" \
  ASCEND_BENCH_CSV="$OUT_DIR/bench.csv" \
    "$DEV/bench_device_950pr_turboquant" > "$LOG4" 2>&1
  RC4=$?
  set -e
  echo "  exit $RC4  (77 = no usable 950PR attached)"
  if [ "$RC4" -eq 77 ]; then
    note_failure "the benchmark refused the device (77): it will not run on a CAModel"
  elif [ "$RC4" -ne 0 ]; then
    note_failure "bench_device_950pr_turboquant exited $RC4 -- see $LOG4"
  fi
  echo
  echo "  --- the ablation pair ---"
  grep -E "dec_attn_core|dec_nounpack" "$LOG4" | head -20 || true
  if [ -s "$OUT_DIR/decode_audit.csv" ]; then
    echo "  decode CSV: $OUT_DIR/decode_audit.csv ($(wc -l < "$OUT_DIR/decode_audit.csv") lines)"
  else
    note_failure "no decode CSV was written; the audit legs did not run"
  fi
  echo
  echo "  --- in-bench cosine bounds (the benchmark asserts its own) ---"
  grep -E "cos = [0-9.]+ \(bound" "$LOG4" || echo "    (none reported)"
fi

# ---------------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------------- #

banner "summary"
echo "logs and CSVs : $OUT_DIR"
if [ "$FAILURES" -eq 0 ]; then
  echo "result        : PASS -- every gate cleared. Stage 2 may run."
  exit 0
fi
echo "result        : FAIL -- $FAILURES gate(s) did not clear. Do not run stage 2 on this build."
exit 1
