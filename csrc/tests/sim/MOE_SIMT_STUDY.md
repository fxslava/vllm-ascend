# DeepSeek-V4 MoE hybrid SIMT/SIMD evaluation — Ascend 950PR

The test-owned E2M1/E8M0 hybrid probe, numerical suite, CAModel benchmark,
metrics exporter and CMake wiring are implemented. CANN 9.2.0-beta.2 builds
both targets with `-Werror` and `--cce-auto-sync=off`. The `Ascend950DT`
npusim alias resolves to `Ascend950PR_9599/camodel` in this installation.
All performance figures below are completed model clocks, excluding host time.
Pending simulation was terminated at the user's request. Dispatch-control
slopes and incomplete cache-bypass tile results are excluded from this report.

## Validated metrics

| Workload | SIMD cycles | Cached SIMT cycles | Speedup |
| --- | ---: | ---: | ---: |
| 1,024-element unpack | 7,055 | 6,688 | 1.055x |
| 512-column tile slice, 4,096 elements | 12,646 | 11,853 | 1.067x |
| 256x128 controlled projection, 32,768 elements | 72,207 | 55,235 | 1.307x |

Explicit UB payload falls from 102,016 bytes to 34,304 bytes: 67,712 bytes
saved, or **66.4% reduction**. These are source allocations, excluding compiler,
TPipe, register spills and Data Cache overhead; dynamic UB allocation is zero.
The full controlled projection saves 23.5% of model clocks.

The 1,024-element `asc_ldcg` unpack probe takes 8,668 cycles, **29.6% more
than cached SIMT** and 22.9% more than SIMD. Cached loads therefore beat bypass
in the validated comparison. This is not a universal cache-policy guarantee.
The register-LUT candidate takes 6,685 cycles with the same reported instruction
counts as bit assembly, giving no meaningful advantage over the reference path.

## Architectural conclusions

- A single AIV executes SIMD or SIMT at a time. Ping-pong UB buffers cannot
  overlap those computations on that AIV. Independent MTE or AIC work is needed
  for useful overlap; this probe uses conservative phase drains.
- Decode in SIMT registers and stage dense FP32 directly into UB. Use coarse
  4,096-value tiles, then SIMD contraction and canonical `SwiGLU(up, gate)`,
  gate clamp `[-10,10]`, and `CAST_ROUND` to BF16.
- `asc_ldcg` bypasses Data Cache but still uses L2. Direct GM addressing is not
  proof of L2 bypass. The public installed API does not establish an uncached
  L2-bypass byte-load implementation.
- Exact isolated mode-switch latency, exclusive stalls, measured GM bandwidth,
  Data Cache hit rate and peak register/spill counts are not established by
  these results. Missing metrics remain null rather than inferred.

## Numerical validation and scope

All 16 E2M1 nibbles in both positions and all 256 E8M0 codes passed across all
four decoders, including subnormal scale 0 and NaN 255. Finite decoded BF16
values match exactly. Completed contraction/SwiGLU checks satisfy two BF16 ULP
and `RateDiff=0`. Host oracle tests (20) and metric-parser tests (2) passed.
Targeted formatting, shell syntax and whitespace checks passed; full
`format.sh ci` remains unavailable because WSL lacks `pre-commit`.

The benchmark uses `x=1` and identical gate/up weights, includes dense output
stores, and excludes separate W1/W3 streams, down projection, Cube consumption
and routing. Its SIMD comparator materializes natural-order decoded weights;
it is not the shipping expert's post-reduction scaling schedule. The measured
1.307x gain applies to this controlled projection, not complete MoE inference.
The research precedents use different scale/output contracts or the 9589 model;
they informed candidate selection but were not reused as DeepSeek-V4 results.

## Engineering recommendation

**Advance cached SIMT bit-assembly unpacking as an experimental DeepSeek-V4
MoE integration candidate on Ascend 950PR.** The 1.307x controlled-projection
speedup and 66.4% UB relief justify further work. Retain cached loads and coarse
handoffs; do not select `asc_ldcg` or claim same-AIV computational overlap.
Production deployment requires separate W1/W3 and downstream Cube/down
projection validation, register/spill characterization, and real-device
end-to-end accuracy and throughput measurements. No production kernel was
changed by this study.

## Artifacts and reproduction

- `test_sim_950pr_moe_simt_unpack.cpp`: numerical and geometry coverage.
- `bench_sim_950pr_moe_simt_vs_simd.cpp`: selectable SIMD/SIMT benchmark.
- `moe_simt_study_results.json`: extracted validated clocks and reports.
- `run_moe_simt_study.sh`: opt-in sequential harness; no further run is pending.

```bash
cmake -S csrc/tests -B build_sim_simt \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann \
  -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim \
  -DVLLM_ASCEND_TESTS_WERROR=ON
cmake --build build_sim_simt --target moe_simt_targets -j4
```

[Official asc_ldcg contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/memory_access_functions/asc_ldcg.md)
confirms that Data Cache bypass still uses L2. The installed mixed-mode toolchain
recognizes `__simt_vf__` and `asc_vf_call` without unsupported `-fsimt`; its legacy
launcher uses static TPipe allocation and three launch fields.
