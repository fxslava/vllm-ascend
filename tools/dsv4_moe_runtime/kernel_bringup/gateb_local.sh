#!/usr/bin/env bash
# Build Gate B and (optionally) run it against the golden, with loud results.
# Lives in the container at /tmp/gateb.sh; usage:
#   /tmp/gateb.sh build           -> compile, print first errors or BUILD_OK
#   /tmp/gateb.sh run [h] [i]     -> run sim + gen_golden cmp
set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
BUILD=/tmp/dsv4_sim_build
IO=/tmp/dsv4_gateb
HIDDEN=${2:-256}
INTER=${3:-128}

case "${1:-build}" in
  build)
    cd "$BUILD" || exit 2
    if cmake --build . > /tmp/gateb_build.log 2>&1; then
      echo BUILD_OK
    else
      echo BUILD_FAILED
      grep -m6 -E 'error' /tmp/gateb_build.log
      exit 1
    fi
    ;;
  run)
    python3 /workspace/tools/dsv4_moe_runtime/kernel_bringup/gen_golden.py gen       --out-dir "$IO" --hidden "$HIDDEN" --inter "$INTER" > /dev/null || exit 2
    cd /tmp || exit 2
    if ! ./dsv4_sim_build/dsv4_moe_expert_sim "$IO" "$HIDDEN" "$INTER" > /tmp/gateb_run.log 2>&1; then
      echo SIM_FAILED
      grep -m4 -E 'ERROR|Abort' /tmp/gateb_run.log
      exit 1
    fi
    python3 /workspace/tools/dsv4_moe_runtime/kernel_bringup/gen_golden.py cmp \
      --golden "$IO/golden.bin" --actual "$IO/actual.bin" --meta "$IO/meta.json"
    ;;
esac
