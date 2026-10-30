# TurboQuant kernels on Ascend 950

The ASC build includes the BF16 reshape-and-cache writer and the optimized fused
decode in `libvllm_ascend_turboquant`. The writer is also exposed through
`torch.ops._C_ascend.npu_turboquant_reshape_and_cache`. The decode has C++ launchers
declared in `op_adapter/turboquant_fused_decode.h`.

Both kernels use the writer's existing cache contract:

- Four KV heads, 256 channels, and whole aligned 16-token strips. Cache pages can
  contain any positive multiple of 16 tokens.
- K nibbles are token-major within each `(physical_page, head)`.
- V nibbles are channel-major within a 16-token strip, with strips consecutive
  within each `(physical_page, head)`.
- Each byte stores two consecutive signed INT4 values, low nibble first.
- FP32 scales have layout `[2][num_blocks][num_kv_heads][block_size]`.
  Reconstruction is exactly `signed_nibble * scale`.

Ingestion uses a recycled, fenced mode-4 handshake and drains its engine queues
before another execution wave reuses local memory. A complete context is one
launch, including contexts wider than 32 strips. The single-shot device entry
remains available as a synchronization regression control. GM stores and the
existing in-page strip offsets are preserved.

Decode reads this cache directly. It expands signed nibbles onto the FP8 Cube
operand grid and stages its L1 operands without converting or rewriting GM.
There is no affine centroid offset or codebook gain. Each tile loads the two
contiguous scale slices for its head, and pages shorter than a 64-row decode tile
are padded in local memory. Context tails are masked by the existing softmax.

The C++ launchers accept `numBlocks` immediately after `blockSize`. This is the
physical cache capacity, including unused pages; it locates the V scale plane.
Pass mode `4`, type `AscendType::FP16`, and attention scale `1 / sqrt(256)`.
The pre-rotated entry reads FP32 queries and writes FP16 output in the rotated
basis. `modeTables` is unused and may be null.

For raw queries, the basis is the writer's
`Pi = (H8 x I32) diag(piSigns) (I8 x H32) / 16`. Output stages apply its inverse
before optional gating; this transform is generally not self-inverse. The
`rotTables` argument contains the shuffle tables consumed by
`TurboQuantCodec4`, independently of the writer's BF16 resident matrix tables.
The compatibility arguments `h16` and `prologueCubeChunkVectors` are unused for
this basis; the raw-query prologue runs on the Vector cores.

Launch tiling is supplied by the C++ caller:

- `headsPerTask` is clamped to `[1, min(16, numHeads / numKvHeads)]`.
- The task count is `numTokens * numKvHeads * numSplits *`
  `ceil((numHeads / numKvHeads) / headsPerTask)`.
- `tasksPerBlock` must cover that count across the grid. For reduction, cover
  `numTokens * numHeads` with `reduceTasksPerBlock`. For raw queries, cover the
  same vector count with `prologueVectorsPerBlock`.
- Allocate FP32 workspace of at least
  `numTokens * numHeads * numSplits * (headSize + 16)` elements when splitting.
  Allocate FP32 `queryRot` scratch of `numTokens * numHeads * headSize` elements
  for raw queries. All GM buffers must be burst-aligned.
- Keep the decode grid within the physical core count: its raw-query and split
  reduction paths use a grid-wide synchronization.

## Validation

Run the host-only contract regressions without loading the NPU test fixtures:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --noconftest -q tests/ut/ops/test_turboquant_kernel_contracts.py
```

The standalone regression writes real caches, decodes them, and compares against
a CPU attention reference reconstructed from those exact bytes and planar scales.
Build it against the kernel library from an Ascend 950 build:

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. -I"$ASCEND_HOME_PATH/include" \
  tests/e2e/pull_request/one_card/turboquant_cache_decode.cpp \
  -L/path/to/build -lvllm_ascend_turboquant -L"$ASCEND_HOME_PATH/lib64" -lascendcl \
  -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,/path/to/build \
  -o /tmp/turboquant_cache_decode
/tmp/turboquant_cache_decode contracts
/tmp/turboquant_cache_decode 16 2 25 1
/tmp/turboquant_cache_decode 128 33 521 1
/tmp/turboquant_cache_decode 128 33 521 2
/tmp/turboquant_cache_decode 16 2 25 1 0
/tmp/turboquant_cache_decode 16 2 25 1 1
/tmp/turboquant_cache_decode 16 2 25 1 2
```

Arguments are page size, ingestion strip count, decode context length, and split
count. An optional fifth argument selects the raw-query output stage: `0`
rotated, `1` unrotated, or `2` gated with a zero gate. The runner also reports
decode wall time for hardware profiling. For CAModel, build and link both the
kernel library and runner against CANN's simulator runtime for
`Ascend950PR_9599`; setting `LD_LIBRARY_PATH` alone does not select that backend.
Simulator wall time is not a device performance measurement.
