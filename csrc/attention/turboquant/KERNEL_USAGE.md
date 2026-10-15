# TurboQuant kernel usage matrix

This matrix describes the **current source after KV3_FP4 v1 plumbing**. The
production boundary is the v1 `TurboQuantAttentionBackend` in
`vllm_ascend/attention/turboquant_v1.py`. The C++ Torch schemas are in
`op_adapter/turboquant_torch_ops.h`; their validating launchers are in
`op_adapter/turboquant_torch_adpt.h`. `tools/tq_longbench/ops.py` uses those
same Torch operators as a separate benchmark harness. `csrc/tests` links the
host launchers directly.

| Bucket | Device kernel / C++ launcher | Production route and other verified callers |
| --- | --- | --- |
| `[PROD_INFERENCE]` | `turboquant_reshape_and_cache_{half,bfloat16_t}` / `turboquant_reshape_and_cache_impl` | `npu_turboquant_reshape_and_cache` from `turboquant_v1.py` AIV writer; also `tq_longbench/ops.py` and `csrc/tests/{sim,device}`. |
| `[PROD_INFERENCE]` | `turboquant_paged_attention_fused_{half,bfloat16_t}` / `turboquant_paged_attention_impl` | `npu_turboquant_paged_attention` from `turboquant_v1.py` AIV decode; also longbench and C++ tests. |
| `[PROD_INFERENCE]` | `turboquant_rotate_q_impl` and its FP16/BF16 instances | Query and output rotation in `turboquant_v1.py`; KV3 Cube adapter rotates the query before fused decode; also longbench and C++ tests. |
| `[PROD_INFERENCE]` | `turboquant_mm_reshape_and_cache_kv4fp8_half` / `turboquant_mm_reshape_and_cache_impl(mode=4)` | `npu_turboquant_cube_reshape_and_cache` from `turboquant_v1.py` on the FP16 Cube path; also longbench and C++ tests. |
| `[PROD_INFERENCE]` | `turboquant_mm_fused_decode_kv4fp8_half` / `turboquant_mm_fused_decode_impl(mode=4)` | Built in the wheel and directly used by C++ tests; the v1 Python Cube path uses the **raw-query** variant below instead. Thus this symbol is production-built, but not the v1 decode call target. |
| `[PROD_INFERENCE]` | `turboquant_mm_fused_decode_raw_query_kv4fp8_half` / `turboquant_mm_fused_decode_raw_query_impl` | **Called by v1 inference**: `turboquant_v1.py:forward_cube_decode` → `npu_turboquant_cube_decode` → adapter raw-query launcher. Also `tq_longbench/ops.py`, `csrc/tests/common/turboquant/turboquant_fused_harness.hpp`, device benchmark and simulator tests. Its placement in a benchmark-only list would be incorrect. |
| `[PROD_INFERENCE]` (preliminary) | `turboquant_mm_reshape_and_cache_kv3fp4_half` and `turboquant_mm_fused_decode_kv3fp4_half` / shared mode launchers with `mode=3` | Previously behind `VLLM_ASCEND_TQ_TEST_KERNELS`, used by `test_sim_turboquant_multimode.cpp`. Now built in the wheel and reached only for v1 `turboquant_int3` or `kv3fp4`, FP16 Cube/MIX, with pre-rotated query and in-tree mode tables. Device correctness and quality still need NPU validation. |
| `[LONGBENCH_AND_TEST_MODES]` | `turboquant_mm_reshape_and_cache_kv5fp8_half`, `turboquant_mm_fused_decode_kv5fp8_half` / shared mode launchers with `mode=5` | Remain under `VLLM_ASCEND_TQ_TEST_KERNELS`; `test_sim_turboquant_multimode.cpp` and the common C++ fused harness exercise them. No standard Python dtype dispatch. |
| `[LONGBENCH_AND_TEST_MODES]` | `turboquant_mm_reshape_and_cache_lloydmax_kv4fp8_half`, `turboquant_mm_fused_decode_lloydmax_kv4fp8_half` / dedicated Lloyd-Max launchers | Test-guarded; `csrc/tests/common/turboquant/turboquant_fused_harness.hpp` and `csrc/tests/device/ascend950pr/bench/bench_turboquant.cpp`. `DecodeUnpack::kLloydMaxLut` changes the codebook and requires the matching writer. No Python dtype dispatch. |
| `[ABLATION_AND_UNIT_PROBES]` | `turboquant_mm_fused_decode_nounpack_kv4fp8_half` / `turboquant_mm_fused_decode_nounpack_impl` | Test-guarded `DecodeUnpack::kBypass`, called by `bench_turboquant.cpp`; output is intentionally not meaningful. |
| `[ABLATION_AND_UNIT_PROBES]` | `turboquant_mm_fused_decode_gather_kv4fp8_half` / `turboquant_mm_fused_decode_gather_impl` | Test-guarded `DecodeUnpack::kGatherLut`, called by `bench_turboquant.cpp`. |
| `[ABLATION_AND_UNIT_PROBES]` | `turboquant_cube_gemm_probe_fp8` / `turboquant_cube_gemm_probe_impl` | Test-guarded; `test_sim_cube_gemm.cpp` only. |

The workspace-size and vector-core-count Torch registrations are host helpers,
not device kernels. `tests/ut/attention/turboquant_cpu_ops.py` registers CPU or
Meta stand-ins for the production schemas; unit tests in `tests/ut/attention/`
call or mock them. The longbench script does not call KV5, Lloyd-Max, or the
ablation launchers by name; those are linked from `csrc/tests`. In particular,
the `raw_query` KV4 symbol is an active v1 inference target despite its use in
the benchmark harness.

For KV3, `TurboQuantModeConfig::PackedBytes(D)` is `3 * D / 8` (low two-bit
plane followed by one-bit plane), matching `TurboQuantModeCodec<KV3_FP4>`.
The KV3 Cube cache stores each token as a row-major plane of
`AlignUp(num_kv_heads * PackedBytes(D), 32)` bytes. A 128-wide head packs to
48 bytes: two heads occupy a 96-byte plane, while one head uses a 64-byte
plane with 16 bytes of row padding. Cache tensors expose the plane as
`[num_blocks, block_size, 1, packed_plane]`; head offsets within the row are
still `kv_head * PackedBytes(D)`. The decoder reads aligned whole-token rows
from GM and selects heads in UB. Blocks remain divisible by 64. NZ-tiled
storage is used by the affine KV4-FP8 Cube path.
