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
# Stage 2 of the Ascend 950PR TurboQuant hardware gate: the task and latency
# sweep for the uncompressed attention sinks, on the lse out-tensor.
#
# Run scripts/run_950pr_baremetal_validation.sh first. This stage costs a weight
# load per rung and does not check the kernels; stage 1 does.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------- #
# Defaults
# ---------------------------------------------------------------------------- #

MODEL_PATH=""
DEVICE="${TQ_DEVICE:-npu}"

# turboquant_cube is the one-launch kv4fp8 decode and the path that ships on the
# 950. turboquant_aiv is the vector-unit decode -- the same cache at three
# launches per step, and the A/B for "is the Cube doing what it should".
# (There is no backend called turboquant_vector; aiv is its name here.)
BACKENDS="${TQ_BACKENDS:-turboquant_cube}"

# The ablation. 0 is the 4-bit cache exactly as it ships; 4 keeps the first four
# tokens of each sequence uncompressed and merges them from the decode's own
# softmax statistics. One pass runs the whole ladder at both.
SINK_TOKENS="${TQ_SINK_TOKENS:-0,4}"

CONTEXTS="${TQ_CONTEXTS:-2048,8192,32768}"

# Sinks need a dense prefill: a batched_decode prefill attends through the 4-bit
# cache, which the plugin's own backend never does. Pinned rather than left to
# the per-rung policy, because the policy can choose batched_decode at a long
# rung and RunnerConfig then refuses the run several minutes in.
PREFILL_MODE="${TQ_PREFILL_MODE:-dense_staging}"

DTYPE="${TQ_DTYPE:-float16}"
BLOCK_SIZE="${TQ_BLOCK_SIZE:-128}"
DEPTHS="${TQ_DEPTHS:-0.0,0.25,0.5,0.75,1.0}"
MAX_NEW_TOKENS="${TQ_MAX_NEW_TOKENS:-48}"

OUT_DIR=""
RUN_LONGBENCH=0
LONGBENCH_TASKS="${TQ_LONGBENCH_TASKS:-narrativeqa,multifieldqa_en}"
LONGBENCH_LIMIT="${TQ_LONGBENCH_LIMIT:-8}"
DATASET_DIR="${TQ_DATASET_DIR:-datasets/longbench}"
REUSE_RUNNER=0
SKIP_PREFLIGHT=0
SKIP_OPS_CHECK=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_950pr_sink_lse_benchmark.sh --model-path DIR [options]

Stage 2 of the 950PR TurboQuant gate: what the uncompressed attention sinks cost
and what they buy, across a context ladder, with the merge reading the decode's
own lse out-tensor instead of recomputing the softmax denominator on the host.

What it measures (tools/tq_longbench/run_benchmark.py, one pass per backend):

  retrieval   needle-in-a-haystack at each depth, per (backend, context, sinks)
  TTFT        prompt in, first token chosen: prefill plus the argmax over logits
  decode p50  microseconds per token, median over the run
  decode p99  the first step of a sequence carries first-touch costs no later
              step does, so it lands here and not in p50; both are reported
  KV MB       what the quantised pool holds, against the dense equivalent
  sink MB     the uncompressed side-car, on top of KV MB
  peak MB     the allocator's own high-water mark

Read the sinks>0 rows for their SCORES. Their p50 is not comparable with the
sinks=0 rows above them: the merge adds its own launches to every decode step,
and a sink layer gives up the single-launch Cube fusion because the merge has to
see the raw rotated accumulator.

Options:
  --model-path DIR     required: config.json + safetensors shards
  --backends LIST      comma-separated (default: turboquant_cube).
                       turboquant_aiv is the vector-unit A/B.
  --sink-tokens LIST   comma-separated counts to sweep (default: 0,4)
  --contexts LIST      comma-separated prompt lengths (default: 2048,8192,32768)
  --device DEV         npu, npu:N (default: npu)
  --prefill-mode MODE  dense_staging | batched_decode (default: dense_staging;
                       a positive sink count needs dense_staging)
  --dtype DT           float16 | bfloat16 | float32 (default: float16)
  --depths LIST        needle depths in [0,1] (default: 0.0,0.25,0.5,0.75,1.0)
  --max-new-tokens N   generation budget per item (default: 48)
  --reuse-runner       one runner per backend sized for the longest rung instead
                       of one per rung. Much faster; the memory columns then
                       describe that size at every rung rather than the rung's.
  --longbench          also run run_longbench.py for task scores
  --longbench-tasks L  (default: narrativeqa,multifieldqa_en)
  --longbench-limit N  items per task per rung (default: 8)
  --dataset-dir DIR    LongBench JSONL directory (default: datasets/longbench);
                       missing files fall back to synthetic stubs, which measure
                       the harness and not the model
  --skip-ops-check     skip the TurboQuant operator preflight. It costs seconds and
                       catches a stale extension winning the registration, which the
                       dispatcher would otherwise report as an argument count from
                       inside a decode step. Only skip it if it is itself wrong.
  --skip-preflight     skip the one-layer probe of every attention path. NOT
                       advised on a 950: the probe is what negotiates the
                       dense_staging pool's backend, because this part refuses
                       op-plugin's FIA (EZ9903) and has to stage through
                       cann_dense.
  --out DIR            where the reports land
                       (default: reports/npu_950pr_sink_lse_<timestamp>)
  --dry-run            print the commands and the environment, run nothing (the
                       operator preflight is skipped too: it imports torch)
  -h, --help           this text

Exit: 0 every rung completed. 1 a rung failed. 2 bad arguments.
EOF
}

# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #

while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?--model-path needs a directory}"; shift 2 ;;
    --backends) BACKENDS="${2:?--backends needs a list}"; shift 2 ;;
    --sink-tokens) SINK_TOKENS="${2:?--sink-tokens needs a list}"; shift 2 ;;
    --contexts) CONTEXTS="${2:?--contexts needs a list}"; shift 2 ;;
    --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
    --prefill-mode) PREFILL_MODE="${2:?--prefill-mode needs a value}"; shift 2 ;;
    --dtype) DTYPE="${2:?--dtype needs a value}"; shift 2 ;;
    --depths) DEPTHS="${2:?--depths needs a list}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?--max-new-tokens needs a number}"; shift 2 ;;
    --reuse-runner) REUSE_RUNNER=1; shift ;;
    --longbench) RUN_LONGBENCH=1; shift ;;
    --longbench-tasks) LONGBENCH_TASKS="${2:?--longbench-tasks needs a list}"; shift 2 ;;
    --longbench-limit) LONGBENCH_LIMIT="${2:?--longbench-limit needs a number}"; shift 2 ;;
    --dataset-dir) DATASET_DIR="${2:?--dataset-dir needs a directory}"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --skip-ops-check) SKIP_OPS_CHECK=1; shift ;;
    --out) OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "--model-path is required." >&2
  usage >&2
  exit 2
fi
if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$MODEL_PATH" ]; then
  echo "--model-path '$MODEL_PATH' is not a directory." >&2
  exit 2
fi

case "$SINK_TOKENS" in
  *[!0-9,]*) echo "--sink-tokens must be comma-separated integers, got '$SINK_TOKENS'" >&2; exit 2 ;;
esac
if [ "$SINK_TOKENS" != "0" ] && [ "$PREFILL_MODE" != "dense_staging" ]; then
  echo "a positive sink count needs --prefill-mode dense_staging: a batched_decode prefill" >&2
  echo "attends through the 4-bit cache, which the plugin's TurboQuant backend never does." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-reports/npu_950pr_sink_lse_$STAMP}"

# ---------------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------------- #

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -r "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

# Where the RUN_MODE=npu csrc/tests tier was built. The loader searches it for the
# standalone binding, and the dynamic linker searches it for the kernel library, so a run
# consumes the test artifacts already on disk without anything being installed.
#
# What that tree does *not* hold is worth stating, because it is the thing that catches
# people out: its lib/libvllm_ascend_turboquant.so is the Ascend C kernel library and
# nothing else. csrc/tests is configured without Python, PyTorch or torch_npu on purpose,
# so that file links no libc10 and registers no torch.ops schema -- handing it to
# torch.ops.load_library succeeds and registers nothing. The operators come only from
# libvllm_turboquant_cube.so, which tools/tq_longbench/build_turboquant_ops.py produces.
# No setup.py and no pip install on either path.
TEST_BUILD_DIR="${TQ_TEST_BUILD_DIR:-$REPO_ROOT/build/csrc-tests-npu}"
export ASCEND_TQ_BUILD_DIR="$TEST_BUILD_DIR"
export LD_LIBRARY_PATH="$TEST_BUILD_DIR:$TEST_BUILD_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The caching allocator fragments badly across a ladder that grows its KV pool
# per rung, and an unquantised dense_staging pool at 32k is the largest single
# block a run asks for. Only set where the operator has not already chosen.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The harness runs with no vLLM in the process and asserts it; nothing here may
# pull it in through a plugin autoload.
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------- #
# Banner
# ---------------------------------------------------------------------------- #

echo "##############################################################################"
echo "# Ascend 950PR TurboQuant -- stage 2, uncompressed attention sinks on the lse"
echo "# out-tensor: retrieval, TTFT, decode percentiles and HBM across a ladder"
echo "# $STAMP"
echo "##############################################################################"
echo "model            : $MODEL_PATH"
echo "device           : $DEVICE"
echo "backends         : $BACKENDS"
echo "sink sweep       : $SINK_TOKENS   (0 = the 4-bit cache as it ships)"
echo "contexts         : $CONTEXTS"
echo "prefill mode     : $PREFILL_MODE"
echo "dtype            : $DTYPE, block size $BLOCK_SIZE"
echo "runner           : $([ "$REUSE_RUNNER" -eq 1 ] && echo 'one per backend (--reuse-runner)' || echo 'one per rung')"
echo "preflight        : $([ "$SKIP_PREFLIGHT" -eq 1 ] && echo 'SKIPPED -- dense_staging will not be negotiated' || echo 'on')"
echo "longbench        : $([ "$RUN_LONGBENCH" -eq 1 ] && echo "$LONGBENCH_TASKS (limit $LONGBENCH_LIMIT)" || echo 'not run (--longbench)')"
echo "output           : $OUT_DIR"
echo "allocator        : PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"
echo "test artifacts   : ASCEND_TQ_BUILD_DIR=$ASCEND_TQ_BUILD_DIR"
echo
echo "VLLM_ASCEND_TQ_SINK_TOKENS is published by the harness itself, once per rung,"
echo "before each runner is built. Do not set it here; it would be overwritten."

# Informational only, and deliberately never fatal. npu-smi goes through DCMI, which
# fails inside a shared container (error -8005) on a host whose devices the run can still
# open; refusing there would block a working setup. What decides whether the operators can
# run is the ops preflight below and the runner's own device open, not this.
echo
if command -v npu-smi >/dev/null 2>&1; then
  if npu-smi info > "/tmp/npu-smi.$$" 2>&1; then
    echo "--- npu-smi (first 12 lines) ---"
    head -12 "/tmp/npu-smi.$$" || true
  else
    echo "--- npu-smi exited non-zero; continuing ---"
    echo "  Common inside a shared container: DCMI returns -8005 while the devices still open."
    head -4 "/tmp/npu-smi.$$" 2>/dev/null | sed 's/^/  /' || true
  fi
else
  echo "--- npu-smi is not on PATH; continuing ---"
fi

# The one check that is fatal, and the cheap one. A host with an older extension reachable
# can have it win the registration -- which is global and permanent, so a fresh library
# cannot replace it -- and the dispatcher would otherwise refuse a decode on an argument
# count minutes into the run, naming neither the stale file nor the argument that grew.
OPS_CHECK_LOG="/tmp/tq-ops-check.$$"
if [ "$DRY_RUN" -eq 0 ] && [ "$SKIP_OPS_CHECK" -eq 0 ]; then
  echo
  echo "--- TurboQuant operator preflight (tools/tq_longbench/check_turboquant_ops.py) ---"
  set +e
  "$PYTHON" "$REPO_ROOT/tools/tq_longbench/check_turboquant_ops.py" > "$OPS_CHECK_LOG" 2>&1
  OPS_RC=$?
  set -e
  if [ "$OPS_RC" -eq 0 ]; then
    grep -E "^  (found|missing) |^  npu_turboquant|^RESULT" "$OPS_CHECK_LOG" | sed 's/^/  /' || true
  else
    sed 's/^/  /' "$OPS_CHECK_LOG"
    echo
    echo "The TurboQuant operators this process would launch are not the ones this checkout"
    echo "builds (check exit $OPS_RC). Stopping here rather than after the weights load."
    echo "Re-run with --skip-ops-check to proceed anyway."
    rm -f "/tmp/npu-smi.$$"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------- #
# The runs
# ---------------------------------------------------------------------------- #

BENCH_ARGS=(
  "$REPO_ROOT/tools/tq_longbench/run_benchmark.py"
  --model-path "$MODEL_PATH"
  --device "$DEVICE"
  --backends "$BACKENDS"
  --contexts "$CONTEXTS"
  --depths "$DEPTHS"
  --sink-tokens "$SINK_TOKENS"
  --prefill-mode "$PREFILL_MODE"
  --dtype "$DTYPE"
  --block-size "$BLOCK_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
[ "$REUSE_RUNNER" -eq 1 ] && BENCH_ARGS+=(--reuse-runner)
[ "$SKIP_PREFLIGHT" -eq 1 ] && BENCH_ARGS+=(--skip-preflight)

LONGBENCH_ARGS=(
  "$REPO_ROOT/tools/tq_longbench/run_longbench.py"
  --model-path "$MODEL_PATH"
  --device "$DEVICE"
  --backends "$BACKENDS"
  --contexts "$CONTEXTS"
  --tasks "$LONGBENCH_TASKS"
  --limit "$LONGBENCH_LIMIT"
  --dataset-dir "$DATASET_DIR"
  --sink-tokens "$SINK_TOKENS"
  --prefill-mode "$PREFILL_MODE"
  --dtype "$DTYPE"
  --block-size "$BLOCK_SIZE"
)
[ "$SKIP_PREFLIGHT" -eq 1 ] && LONGBENCH_ARGS+=(--skip-preflight)

# Prints an argv so that the line can be pasted back into a shell, quoting only a
# word that whitespace would otherwise split. %q would escape the comma in every
# list and print a line that is correct but unreadable.
show_cmd() {
  printf '  %s' "$1"
  shift
  for word in "$@"; do
    case "$word" in
      *[[:space:]]*) printf " '%s'" "$word" ;;
      *) printf ' %s' "$word" ;;
    esac
  done
  echo
}

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "--- dry run: nothing was executed ---"
  echo "output would be: $OUT_DIR"
  echo
  echo "stage 2a:"
  show_cmd "$PYTHON" "${BENCH_ARGS[@]}" --out-file "$OUT_DIR/bench.jsonl"
  if [ "$RUN_LONGBENCH" -eq 1 ]; then
    echo
    echo "stage 2b:"
    show_cmd "$PYTHON" "${LONGBENCH_ARGS[@]}" --out-file "$OUT_DIR/longbench.jsonl"
  fi
  # A hook for the argument check: with TQ_ARGV_DIR set, the argv each stage would
  # hand its CLI is written out one word per line, so a checker can feed it to that
  # CLI's own parser instead of reconstructing it and drifting from this file.
  if [ -n "${TQ_ARGV_DIR:-}" ]; then
    mkdir -p "$TQ_ARGV_DIR"
    printf '%s\n' "${BENCH_ARGS[@]:1}" --out-file "$OUT_DIR/bench.jsonl" > "$TQ_ARGV_DIR/bench_argv.txt"
    printf '%s\n' "${LONGBENCH_ARGS[@]:1}" --out-file "$OUT_DIR/longbench.jsonl" > "$TQ_ARGV_DIR/longbench_argv.txt"
    echo
    echo "argv written to $TQ_ARGV_DIR"
  fi
  exit 0
fi

mkdir -p "$OUT_DIR"
{
  echo "stamp=$STAMP"
  echo "model=$MODEL_PATH"
  echo "device=$DEVICE"
  echo "backends=$BACKENDS"
  echo "sink_tokens=$SINK_TOKENS"
  echo "contexts=$CONTEXTS"
  echo "prefill_mode=$PREFILL_MODE"
  echo "dtype=$DTYPE"
  echo "block_size=$BLOCK_SIZE"
  echo "reuse_runner=$REUSE_RUNNER"
  echo "skip_preflight=$SKIP_PREFLIGHT"
  echo "git_head=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "ascend_home=${ASCEND_HOME_PATH:-unset}"
  echo "alloc_conf=$PYTORCH_NPU_ALLOC_CONF"
} > "$OUT_DIR/run_manifest.txt"
[ -r "/tmp/npu-smi.$$" ] && mv "/tmp/npu-smi.$$" "$OUT_DIR/npu-smi.log"
# The preflight's whole output, not only the lines the banner showed: it names every path
# the loader tried and every mapped file that could have registered the operators, which
# is what a later reader of this report needs to know which build produced these numbers.
[ -r "$OPS_CHECK_LOG" ] && mv "$OPS_CHECK_LOG" "$OUT_DIR/turboquant_ops_check.log"

FAILURES=0

echo
echo "=============================================================================="
echo "stage 2a: retrieval, TTFT, decode percentiles and HBM  (run_benchmark.py)"
echo "=============================================================================="
echo "  JSONL    -> $OUT_DIR/bench.jsonl   (one record per item, flushed as it completes)"
echo "  summary  -> $OUT_DIR/bench_summary.txt"
echo "  progress -> $OUT_DIR/bench_progress.log"
set +e
"$PYTHON" "${BENCH_ARGS[@]}" --out-file "$OUT_DIR/bench.jsonl" \
  > >(tee "$OUT_DIR/bench_summary.txt") \
  2> >(tee "$OUT_DIR/bench_progress.log" >&2)
RC_BENCH=$?
set -e
wait
echo "  exit $RC_BENCH"
if [ "$RC_BENCH" -ne 0 ]; then
  FAILURES=$((FAILURES + 1))
  echo "  !! run_benchmark.py failed. A ladder that dies at a long rung still leaves the"
  echo "     shorter ones in bench.jsonl; the last lines of bench_progress.log say where."
  tail -20 "$OUT_DIR/bench_progress.log" || true
fi

if [ "$RUN_LONGBENCH" -eq 1 ]; then
  echo
  echo "=============================================================================="
  echo "stage 2b: LongBench task scores  (run_longbench.py)"
  echo "=============================================================================="
  echo "  JSONL    -> $OUT_DIR/longbench.jsonl"
  echo "  tables   -> $OUT_DIR/longbench_summary.txt"
  echo "  progress -> $OUT_DIR/longbench_progress.log"
  if [ ! -d "$DATASET_DIR" ]; then
    echo "  NOTE: '$DATASET_DIR' does not exist. Every task will fall back to synthetic"
    echo "  stubs, which measure the harness and say nothing about the model."
  fi
  set +e
  "$PYTHON" "${LONGBENCH_ARGS[@]}" --out-file "$OUT_DIR/longbench.jsonl" \
    > >(tee "$OUT_DIR/longbench_summary.txt") \
    2> >(tee "$OUT_DIR/longbench_progress.log" >&2)
  RC_LB=$?
  set -e
  wait
  echo "  exit $RC_LB"
  if [ "$RC_LB" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
    tail -20 "$OUT_DIR/longbench_progress.log" || true
  fi
fi

# ---------------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------------- #

echo
echo "=============================================================================="
echo "summary"
echo "=============================================================================="
for f in bench.jsonl longbench.jsonl; do
  if [ -s "$OUT_DIR/$f" ]; then
    echo "  $f: $(wc -l < "$OUT_DIR/$f") record(s)"
  fi
done
echo "  reports: $OUT_DIR"
echo
echo "  The A/B is between the sink_tokens fields of the records, at equal backend and"
echo "  context. Compare scores; do not compare p50 across them."
if [ "$FAILURES" -eq 0 ]; then
  echo "  result : PASS -- every stage completed"
  exit 0
fi
echo "  result : FAIL -- $FAILURES stage(s) failed"
exit 1
