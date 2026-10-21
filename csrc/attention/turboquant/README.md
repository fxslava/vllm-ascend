# TurboQuant inference kernels (Ascend 950PR)

The 4-bit KV cache decode for Ascend 950PR / DaVinci v3 (`dav-3510`, CANN 9.2+), migrated onto the
`v0.23.0` base as inference code only. Nothing here is a test runner, a verification harness, a CPU
golden model, a simulator mock or a benchmark script.

## Layout

| Path | What it holds |
| --- | --- |
| `op_kernel/*.cpp` | The device kernels and, in the same translation unit, their host launchers |
| `op_kernel/common/` | Codecs, mode traits, the layout contract and the UB budget |
| `op_kernel/cube/` | The Cube Matmul service and the raw-query rotation prologue |
| `op_kernel/vector/` | The vector half of the decode: ingest ring, online softmax, reducer |
| `op_host/` | Host tiling: how a launch is cut into blocks and tasks, and every buffer size the caller owns |
| `op_adapter/` | The `torch.ops._C_ascend` schemas and their validating launchers |

## Build

`CMakeLists.txt` builds `vllm_ascend_turboquant` for `SOC_VERSION` matching `ascend950` only, through
the CANN 9.2 **ASC language** toolchain (`asc-config.cmake`, `enable_language(ASC)`,
`CMAKE_ASC_ARCHITECTURES` of `dav-3510`) rather than the legacy `ascendc.cmake` path. Those two
headers define `ascendc_library()` differently, so exactly one is loaded per configure; the legacy
one still serves every other SoC, whose `vllm_ascend_kernels` depends on it.

`VLLM_ENABLE_TURBOQUANT` gates the adapter and the AIV operators, `VLLM_ENABLE_TURBOQUANT_CUBE` the
kv4fp8 Cube operators. `vllm_ascend_C` links `vllm_ascend_turboquant`; nothing links back, because no
kernel translation unit includes anything from `op_adapter/`.

## What the ASC toolchain changed

Under `ascendc_library()` from the legacy path, a `<<<>>>` launch resolved to a generated host stub
taking `void *`, and the kernel's own `__gm__` parameter types were never checked against it. Under
the ASC language both halves of the translation unit go through bisheng and the launch is checked
directly, which accepts exactly one spelling: an unqualified `uint8_t *`, promoted into the `__gm__`
address space at the call. A `void *` is rejected, and so is a `reinterpret_cast` into `__gm__`.

So the launchers keep the plain `void *` signatures the adapter has to declare -- the adapter is built
by the host compiler, which does not know `__gm__` -- and narrow to `uint8_t *` at the call through
`AsBytes()` in `op_kernel/common/turboquant_launch.h`.

## Production surface

The shipping library exports eight host launchers, all with `void *` signatures:

```
turboquant_reshape_and_cache_impl              turboquant_mm_reshape_and_cache_impl
turboquant_paged_attention_impl                turboquant_mm_fused_decode_impl
turboquant_rotate_q_impl                       turboquant_mm_fused_decode_raw_query_impl
turboquant_mm_fused_decode_opt_impl            turboquant_mm_fused_decode_opt_raw_query_impl
```

`npu_turboquant_cube_decode` reaches the kv4fp8 cache through `turboquant_mm_fused_decode_raw_query_impl`,
which rotates the query inside the launch. The two `_opt_` entries are the modulo-scheduled decode: built,
but not selected by the adapter.

The Lloyd-Max codebook, the KV5_FP8 mode, the unpack ablations and the GEMM probes stay behind
`VLLM_ASCEND_TQ_TEST_KERNELS`, which no build here defines, so none of them is in the library.

## UB budget

`FusedDecodeUbBytes` in `op_kernel/common/turboquant_layout.h` mirrors the decode's UB allocations --
`TurboQuantVectorDecodeService::InitBuffers`, the codec's `Init` and the query prologue's UB half, in
allocation order. The adapter refuses a launch that would exceed the 256 KiB partition, because the
device does not: `InitBuffer` hands out UB past the partition without complaint and the overflow
surfaces only as an AIV out-of-bounds fault, or as one buffer reading another's bytes.

It is a mirror, not a derivation. A buffer added to either side without a line there makes it an
underestimate.

At the head sizes `CheckHeadSize` accepts (64 to 256, powers of two) the decode peaks at about 67% of
the partition, so today the check is a regression guard rather than an active limit.
