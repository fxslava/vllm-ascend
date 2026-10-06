# DeepSeek-V4 offline 910B regression

This standalone project compiles only the shipping MoE expert and its Gather
index regression. It requires the CANN 8.0.0 **aarch64** toolkit, an aarch64
cross compiler and an environment capable of executing aarch64 toolkit tools
and CAModel. No physical NPU is needed for `RUN_MODE=sim`.

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux
# The 8.0.0 package uses setenv.bash.
source "$ASCEND_HOME_PATH/bin/setenv.bash"
cmake -S csrc/tests/sim_910b -B build_910b_test/arch220_regression_sim \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1 -DRUN_MODE=sim
cmake --build build_910b_test/arch220_regression_sim \
  --target test_dsv4_indices test_dsv4_expert -j2
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/simulator/Ascend910B1/lib:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib/linux/aarch64:$LD_LIBRARY_PATH"
export CAMODEL_LOG_PATH="$PWD/build_910b_test/camodel_indices"
mkdir -p "$CAMODEL_LOG_PATH"
timeout --signal=TERM --kill-after=5s 900s \
  build_910b_test/arch220_regression_sim/test_dsv4_indices
export CAMODEL_LOG_PATH="$PWD/build_910b_test/camodel_expert"
mkdir -p "$CAMODEL_LOG_PATH"
timeout --signal=TERM --kill-after=5s 900s \
  build_910b_test/arch220_regression_sim/test_dsv4_expert
```

Both kernels compile with `-Werror --cce-auto-sync=off`; host tests also use
`-Wall -Wextra -Werror`. CTest applies a 900-second timeout to each test.
Use separate log directories so CAModel traces do not overwrite one another.

The index test poisons scratch before generating lengths 8, 16, 32, 64, 128 and
256. It checks exact integer offsets and gathered float bits for 32-byte
reduction slots and odd-column addresses. Gather's offset tensor contains
**bytes**, including for `LocalTensor<float>`; see the
[CANN 8.0.0 Gather contract](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0092.html).

The expert test uses hidden=64, inter=64 and seed=704 with finite inputs. It
compares all four BF16 outputs to the CPU oracle and requires at most two ULP,
zero mismatches outside that tolerance and zero NaN/Inf values. It launches
the shipping source through `common/dsv4_moe_expert_kernels.cpp`.

After passing simulation and inspecting its logs, configure a separate build
directory with `-DRUN_MODE=npu` to compile and link the same targets against
the device runtime and development stubs. `Ascend910B1` is the concrete SoC
name used by this CANN package for the requested 910B target. An offline link
check does not establish numerical parity on physical 910B silicon.
