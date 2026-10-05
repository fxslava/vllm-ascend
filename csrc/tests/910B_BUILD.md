# Ascend 910B standalone build

Run from the repository root in Linux Bash on a host with the CANN toolkit,
compiler package and NPU driver installed. GoogleTest must be unpacked at
`$HOME/googletest` (the repository root containing `googletest/CMakeLists.txt`).
Use a fresh build directory when changing the SoC or toolkit.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
cmake -S csrc/tests -B build_910b_test \
  -DASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}" \
  -DSOC_VERSION=Ascend910B4 \
  -DRUN_MODE=npu \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$HOME/googletest" \
  -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF \
  -DVLLM_ASCEND_TESTS_WERROR=ON \
  -DVLLM_ASCEND_TESTS_BUILD_DSV4_KERNELS=ON \
  -DVLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS=OFF &&
cmake --build build_910b_test --target dsv4_device_targets -j4
```

`dsv4_device_targets` builds both `test_device_910b_dsv4_moe_expert` and
`bench_device_910b_dsv4_moe_expert`. Either can also be named individually with
`--target`. Substitute B1, B2 or B3 for B4 to match the installed part.

```bash
ctest --test-dir build_910b_test \
  -R '^test_device_910b_dsv4_moe_expert$' -V
ctest --test-dir build_910b_test \
  -R '^bench_device_910b_dsv4_moe_expert$' -V
```

Both use device 0. The comparative benchmark checks the custom kernel and
the ACLNN GroupedMatmulV5 baseline against the golden oracle before timing.
An unavailable operator or rejected BF16-to-FP32 output tuple fails explicitly;
the benchmark does not change precision to obtain a timing result.

Toolkit discovery honors an explicit `ASCEND_HOME_PATH` first, followed by
`ASCEND_TOOLKIT_HOME`, environment `ASCEND_HOME_PATH`, and
`/usr/local/Ascend/ascend-toolkit/latest`. An invalid explicit path fails rather
than silently selecting another installation. Offline GoogleTest discovery also
accepts `/tmp/googletest`, `csrc/tests/third_party/googletest`, or an installed
GTest package. `VLLM_ASCEND_TESTS_FETCH_GTEST=OFF` prevents network access.

910B selects the physical device tier and excludes `device_310p`,
`device_950pr`, TurboQuant device kernels and 950PR probes. Host codec tests
remain available. DSV4 compilation always uses `--cce-auto-sync=off -Werror`;
GoogleTest compilation does not inherit those flags.

CANN major detection reads toolkit version metadata, including
`compiler/version.info`, then tries the toolkit path and its resolved symlink
target. On CANN 8.0, `version_dir=8.0.0` takes precedence over the internal
component `Version=7.6.0.1.220`. An unknown version produces an error with the
checked paths. Override detection with `-DCANN_VERSION_MAJOR=8` when needed.

The major definition reaches device precompilation, device compilation and
host launcher compilation. CANN 8 uses `WholeReduceSum` and `CompareScalar`;
CANN 9 uses `ReduceRepeat<ReduceType::SUM>` and `Compares`. The legacy reduction
writes directly to UB and introduces no scalar UB reads, work buffers or event
changes. See Huawei's CANN 8.0 references for
[WholeReduceSum](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0081.html)
and [CompareScalar](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0068.html).

Link-time driver stubs are searched in the toolkit's stub/devlib directories,
after installed driver directories. Stubs and simulator directories are never
placed on device executable RUNPATH. Stub linkage permits compilation on a
development host; execution still requires the real driver and physical NPU.

## Validation

Validated with CANN 9.2.0-beta.2 and GCC 11.4 under WSL:

- Clean offline B4 configure and compilation of both device executables passed
  with Werror, without exporting extra library search paths.
- B1/B2/B3 configure manifests contain both 910B targets and no 310P or 950PR
  device targets; explicitly requesting TurboQuant ON is overridden to OFF.
- All five host CTest entries passed, including five toolkit/offline
  configuration regression checks.
- The two host DSV4 executables passed (30 GTest cases total).
- Device executable RUNPATH contains real toolkit paths and the built kernel
  directory, with no stub or simulator paths.

No NPU is attached to this development host. Attempted physical-runtime startup
aborted before `main` in CANN initialization with
`basic_string::_S_construct null not valid`; hardware results and ACLNN baseline
support on 910B remain unverified.
