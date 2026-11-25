# Verified champion integration — 2026-11-25

The requested champions are integrated in `C:/vllm-ascend` and synced into
`C:/Ascend/turboquant-core` on `codex/verified-wave-champions`. No full device
plugin build, device link, CAModel run or physical NPU execution was attempted.
The result is a source/host-validation handoff for yellow-zone testing, not a
new device-parity or hardware-performance certification.

## Sources and changes

The root manifest was read before the handoff. All eleven root-manifest seals
were verified. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the source
paths and SHA-256 hashes; all six imported native/reshape files in core and all
eleven sealed E8 files in both repositories match their verified sources.

| Component | Verified source | Historical CAModel result | Integration |
| --- | --- | --- | --- |
| Native decode | Wave-v11 integration `3e57bfa`; production files equal protected `7c31a112` | 34,701 ticks, 208 rows × 6 heads, 0 BF16 ULP | Paired decode schedule, register softmax, ingress, matching UB/reducer budgets and telemetry closure |
| Batch writer | Frozen Slot03 `3adb895`; source SHA `bcbdcc866230bc72bf47ba3ec4fd025e5a063cd9333a58ee558f37e2924e89ad` | 63,629 ticks, six strips | Scalar metadata hoist; six-strip shares for multiples of 96 tokens, one-strip shares otherwise |
| E8 dequantizer | Wave-3 `review_packet_v5/candidate/source` | 14,252 ticks, exact BF16/INT8 fixture | Sealed parallel SIMT routines and fixed 16-row entry |
| E8 reader | Same sealed Wave-3 packet | 287,101 ticks | Sealed cache-basis reader; K UB group stride 1056 bytes |

No Wave-2/Wave-3 reshape mutant or censored E8 encoder is promoted. Frozen
worktrees, harness probes, manifests and historical artifacts were not edited.

The current core is already the extracted standalone source layout: the old
`kernels/ops/turboquant/vllm_integration_layer` path is absent. The integration
maps its `src/{decode,prefill,scatter}`, `src/host` and `include/turboquant` to
vLLM's `op_kernel`, `op_host`, `include` and `op_adapter`. There is no separate
TurboQuant launcher in `csrc/kernels`; root CMake explicitly includes this tree.

vLLM's existing 64-bit cache address arithmetic and ASC host stub definitions
are retained. The batch Tensor API uses the champion's nondeprecated CANN
9.2.1 spellings. Host-only diagnostic pragmas are confined to compiler-generated
ASC entry stubs with unused injected parameters; device bodies remain strict.
Core's imported native/reshape implementations are byte-identical to the
verified files. E8 device sources are byte-identical in both repositories;
ASC adaptation lives in two separate wrapper files.

[MODIFIED_FILES.json](MODIFIED_FILES.json) is the complete per-file inventory
with normalized-LF hashes, excluding the report files themselves. The 40 vLLM
implementation/test/documentation files comprise CMake, native kernel/services
and ABI headers, the E8 source closure, the E8 C++/Python bindings, host checker
scripts, the CPU contract-test harness and integration documentation. Core's
24 files comprise the six champion files, host launch declarations, the E8
component/CMake gates, golden-root configuration and the inherited telemetry
range-test correction. Four JSON evidence files and this report are added to
each repository. Per-file Git attributes preserve the sealed E8 bytes; the
reader's inherited final blank line is retained with a file-specific whitespace
exception rather than changing its verified hash.

## ABI and signature verification

All checked declarations, definitions, template call sites and Python argument
orders align. This covers six Torch schemas, native writer/decode launchers,
E8 launchers, native tiling descriptors and both cache formats. It does not
claim that an unperformed device-plugin link or hardware launch succeeded.

| Boundary | Verified contract |
| --- | --- |
| Native ingress | Contiguous BF16 `[tokens, 4, 256]`; INT32 slot mapping; INT8 containers for paired signed INT4 nibbles; separate FP32 scales; BF16 table bits carried through the existing INT32-pointer launcher ABI |
| Native decode | BF16 raw query, FP32 rotated-query/workspace storage, INT32 block table and context lengths; existing basis/gate/output-stage arguments preserved; `BITS == 4` codec remains distinct from E8 |
| Native cache addressing | Existing NZ byte/nibble and slot-major scale maps retained; 64-bit large-cache offsets covered by the >4-GiB C++ gate |
| Native tiling | 64-byte, 64-aligned descriptor; 32-byte dispatch record; every descriptor word offset asserted against `offsetof`; the actual CPU builder is linked and executed |
| Native UB | `kUbEnd=142336`, basis 46432 bytes and reducer 3360 bytes move together; shared compile-time budget assertions pass |
| E8 GM planes | Separate UINT8 `[packets, 4352]` K/V planes; 2048 little-endian UINT16 labels in bytes 0..4095, 64 FP32 scales in bytes 4096..4351; effective two bits per coordinate |
| E8 reader | One KV task, six cache-basis FP32 heads padded to `[16,256]`, padding rows zero; even 4..256 tiles; final 1..64 live rows; BF16 output and K/V witnesses, FP32 score witnesses |
| E8 metadata | INT64 storage for UINT64 nonnegative packet byte offsets, prepared from validated packet IDs; allocation rounded to four entries for the 32-byte DMA burst; not the native INT32 block table |
| E8 internal layout | K UB stride 1056 bytes; no GM packet expansion; UB 158464 bytes, L1 282816 bytes; complete padded K-to-L1 address mapping tested |
| E8 component | 512 UINT16 labels, 16 FP32 scales, 4096 BF16 words and 4096 INT8 bytes; exact compiled declaration/definition types |
| Torch/custom ops | Four native registrations plus separate `turboquant_e8_ops.reader/dequant`; mutations, argument order, Meta/FakeTensor dispatch and actual Torch 2.10 registration checked |

These paths call direct Ascend C launchers with storage pointers. There is no
`aclTensor`/ACLNN conversion in this closure to verify. Host binding checks use
real ATen/Torch headers; only the opaque NPU stream/capability declarations and
device launcher implementations are replaced by throwing traps.

E8 runtime calls require caller-owned, disjoint output/witness buffers and
prevalidated offsets. They allocate no tensors and perform no device-to-host
metadata reads. Packet packing and offset creation are explicit CPU preparation
helpers. The existing native constant-cache warmup and CPU tiling staging remain
setup/metadata work; this integration does not claim cold-start zero allocation.

The E8 reader's output remains in cache basis. The native raw-query attention
backend is not automatically switched to E8: doing so would require the held
encoder and separately verified query/output composition. The new explicit E8
operations preserve the verified reader contract without implying that coverage.

## Executed host validation

[VALIDATION.json](VALIDATION.json) contains results and gated-file inventories.

| Check | Result |
| --- | --- |
| vLLM device-free Python suite | **599 passed**, CPU Torch 2.14 |
| Existing latent-codec CPU suite | **13 passed**, package-bootstrap stubs; fused quantizer mocked with CPU oracle |
| vLLM host C++ pytest gates | **7 passed**, GCC C++17 `-Wall -Wextra -Werror` |
| Core CTest | **9/9 passed**, including all eight golden decode cases and exhaustive E8 arithmetic/layout gates |
| vLLM CANN syntax-only checks | **10/10 passed**, five TUs × host/device roles |
| Core CANN syntax-only checks | **10/10 passed**, same native/E8 closure with public declarations forced into the TUs |
| Real Torch 2.10 binding check | **Passed**: four syntax TUs, isolated host-only shared library, six registrations, C++ Meta/FakeTensor and CPU tiling; device dependencies throw if reached |
| Original analyzer `46f16829` | **Zero new diagnostics in 8/8 scopes**, independently parsed verified/integrated sources, including NZ reshape |
| Ruff and whitespace | Changed Python files pass Ruff; both `git diff --check` checks pass |

The E8 CPU check exhausts 65536 labels × 16 boundary scales, including tie
rules, BF16 RNE and physical INT8 rounding. Packet tests separately verify
little-endian label bytes, embedded-scale offsets, strided views and padded
offset-table allocations. The telemetry test's range override previously
examined unrelated split-token records; the imported existing fix restricts
that check to the intended fused token. The separate complete golden dataset
is reused without overwriting checked-in fixtures.

Analyzer raw results are **not clean**. For example, the E8 reader still has
34 raw fatal and 3 warning findings in this all-functions run. They match the
independently parsed sealed source. [ANALYZER_SUMMARY.json](ANALYZER_SUMMARY.json)
retains every scope's raw counts. Differential fingerprints preserve code,
severity, message and multiplicity, ignoring locations and embedded `(line N)`
references only; no category is suppressed. Static analysis is not a device
correctness proof or a complete SIMT coverage claim.

## Exact commands

PowerShell, from `C:/vllm-ascend`:

```powershell
python tests/ut/attention/run_turboquant_device_free.py -q
```

The additional latent run uses the existing device-free package bootstrap:

```powershell
python -c "import sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0,str(root/'tests/ut/attention')); import stub_vllm; stub_vllm.install(); stub_vllm._module('vllm_ascend.quantization.methods',root/'vllm_ascend/quantization/methods'); stub_vllm._module('vllm_ascend.quantization.methods.kv_cache',root/'vllm_ascend/quantization/methods/kv_cache'); import pytest; raise SystemExit(pytest.main(['--noconftest','--import-mode=importlib','-q','tests/ut/quantization/methods/kv_cache/turboquant/test_latent.py']))"
```

Inside WSL Ubuntu-22.04 as `dmin`:

```bash
cd /mnt/c/vllm-ascend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest -q tests/ut/ops/test_turboquant_contract_gates.py

cmake -S /mnt/c/Ascend/turboquant-core -B /home/dmin/build/tq-wave-host-20261125 -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF -DTQ_GOLDEN_DUMP_ROOT=/home/dmin/work/checkpoint1_20261123/golden_validation/dumps
cmake --build /home/dmin/build/tq-wave-host-20261125 -j4
ctest --test-dir /home/dmin/build/tq-wave-host-20261125 --output-on-failure

docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -v /mnt/c/vllm-ascend:/workspace/vllm-ascend --entrypoint python quay.io/ascend/vllm-ascend:v0.26.0rc1-a5 /workspace/vllm-ascend/tests/host/turboquant/check_bindings.py

bash /mnt/c/ascend-micro-harness/scripts/camodel_build_phase_guard.sh compile flock --shared /home/dmin/ascend-sync-schedule-ablation/queue/checkpoint1-exclusive.lock python3 /mnt/c/vllm-ascend/tests/host/turboquant/check_kernels.py --sdk /home/dmin/Ascend-9.2.1/cann-9.2.1 --out /mnt/c/Ascend/turboquant-integration-work/syntax-final
bash /mnt/c/ascend-micro-harness/scripts/camodel_build_phase_guard.sh compile flock --shared /home/dmin/ascend-sync-schedule-ablation/queue/checkpoint1-exclusive.lock python3 /mnt/c/vllm-ascend/tests/host/turboquant/check_kernels.py --sdk /home/dmin/Ascend-9.2.1/cann-9.2.1 --out /mnt/c/Ascend/turboquant-integration-work/syntax-core --core /mnt/c/Ascend/turboquant-core

python3 /mnt/c/vllm-ascend/tests/host/turboquant/check_analyzer.py --analyzer /mnt/c/ascend-kernel-analyzer --native /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core --reshape /mnt/c/Users/Admin/.codex/worktrees/reshape-wave1-selected/turboquant-core --e8 /mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/wave3/w3_e8_dec/review_packet_v5/candidate/source --out /mnt/c/Ascend/turboquant-integration-work/analyzer
```

The last analyzer summary was regenerated with `--reuse` after independently
parsing both sides; it reruns only newer inputs. Raw JSON and per-role syntax
logs remain under `C:/Ascend/turboquant-integration-work`.

## Gated checks and handoff limits

Seven broader host worker/config/capacity/packing/operator test files could not
collect. The normal UT bootstrap was attempted with the installed image, then
with `C:/vllm` mounted over its upstream source:

```bash
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -v /mnt/c/vllm-ascend:/workspace/vllm-ascend -w /workspace/vllm-ascend --entrypoint python quay.io/ascend/vllm-ascend:v0.26.0rc1-a5 -m pytest -q tests/ut/quantization/methods/kv_cache/turboquant tests/ut/worker/test_turboquant_v2.py tests/ut/worker/test_turboquant_hybrid_alloc.py tests/ut/test_platform_turboquant.py
```

The image lacks upstream `CircularBufferSpec`. Adding
`-v /mnt/c/vllm:/vllm-workspace/vllm` instead fails on missing upstream
`_approximate_gcd`. These are **dependency-gated, not passing or hardware-gated**;
the independent latent file was subsequently executed successfully. A matching
upstream vLLM installation is needed to finish those integration tests.

Live tiling/write NPU tests, one-card finalize/ACLGraph tests and all core NPU
runners remain **hardware-gated**. No mock device output is counted as numerical
kernel parity. The normal `format.sh ci` could not complete: its existing CRLF
line endings fail under WSL, and a read-only LF copy reaches a missing
`pre-commit` executable. This does not replace the passing scoped Ruff and
whitespace checks.

Only task-owned files are committed, with sign-off, in each repository.
Tracked working trees are clean. The pre-existing untracked core
`reports/checkpoint1_20261123_i1/` directory and vLLM's `U+F07C`-prefixed `vllm`
file are deliberately retained. Thus an entirely empty `git status` is not
claimed. No files were deleted or hidden to manufacture that result.
