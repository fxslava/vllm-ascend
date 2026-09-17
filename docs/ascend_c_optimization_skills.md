# Ascend C Optimization Skills: arch35 (Ascend 950PR) Kernels

This guide collects the engineering rules learned while bringing the TurboQuant KV-cache decode up on the
Ascend 950PR (arch35). It covers four topics: pipe synchronization, keeping both AIV subcores busy, GM
memory bandwidth, and a single-launch decode that writes its output directly.

Each rule names the code that applies it today and the evidence behind it. Treat those two parts
separately:

- **Camodel evidence** proves functional behavior only, for example that results are bit-exact or that
  a run does not deadlock. The camodel does not model HBM throughput or host dispatch cost, and its ACL
  events report 0 ms.
- **Silicon evidence** is the only valid source for latency. The motivating numbers are user-reported
  silicon measurements: the TurboQuant decode chain takes about 850 us, against 14.16 us for native
  `aclnnFusedInferAttentionScoreV5`. Of that gap, about 48 us is inter-kernel dispatch and stream syncs,
  about 11 us is `Combine` re-reading GM partials, and about 31 us is vector time inflated by barriers.
  No msprof trace from this repository reproduces these numbers yet.

Source files referenced below:

| File | Role |
| --- | --- |
| `op_host/turboquant_tiling.{h,cpp}` | Torch-free launch plans: `PlanPagedAttention`, `PlanFusedDecode`, `PlanRotateQ`, buffer sizes |
| `op_kernel/common/turboquant_layout.h` | Every constant the kernels and the tiling share (tiles, flags, fused limit) |
| `op_kernel/common/turboquant_common.h` | `SyncEvent`, `VecBarrier<ENABLED>`, `MixBlockIdx`, broadcasts, `RepeatButterfly` |
| `op_kernel/common/turboquant_codec_950.h`, `turboquant_codec_mx.h` | The 4-bit and multi-mode codecs |
| `op_kernel/cube/turboquant_cube_service.h` | `TurboQuantCubeMm` (L1 -> L0 load, Mmad, dual-destination Fixpipe) and `TurboQuantCubeDecodeService` |
| `op_kernel/vector/turboquant_vector_service.h` | `TurboQuantTileBurst`, `WriteNormalizedHeads`, `TurboQuantPartialReducer`, `TurboQuantVectorDecodeService` |
| `op_kernel/turboquant_paged_attention.cpp` | Shipping AIV decode (`TurboQuantPagedAttentionSplit`) and the 4-bit cache write |
| `op_kernel/turboquant_rotate_q.cpp` | Query rotation `Pi = D H D`: Cube (Mmad + Fixpipe) and AIV paths |
| `op_kernel/turboquant_fused_decode.cpp` | Cube multi-mode decode (test-only): `TurboQuantFusedDecode` composes the two services |
| `op_adapter/turboquant_torch_adpt.h` | The Torch ops; plans come from `op_host` |

All paths are under `csrc/attention/turboquant/`. The split/combine decode, its separate combine kernel
and the decode ablation ladder are retired; no host launcher issues more than one decode launch.

---

## 1. Event Synchronization Matrix

arch35 runs each kernel across several pipes: MTE1/MTE2/MTE3 for memory transfers, V for the vector
unit, M for the Cube Mmad, FIX for the Fixpipe, and S for the scalar unit. Which synchronization
primitive to use depends on whether data crosses a pipe or stays inside one.

### 1.1 Primitive by situation

| Situation | Primitive | Example in tree |
| --- | --- | --- |
| Data crosses a pipe boundary on the same core | Hard event: `SetFlag<EVENT>(ev)` + `WaitFlag<EVENT>(ev)`, with `ev = GetTPipePtr()->FetchEventID(EVENT)` | `SyncEvent<>()` / `SyncVectorToMte3()` in `op_kernel/common/turboquant_common.h` |
| AIC ↔ AIV handshake (Cube product ready, slot free) | `CrossCoreSetFlag<0x2, PIPE_X>(id)` / `CrossCoreWaitFlag(id)` | `kFlagOperandsReady/Free`, `kFlagProductReady/Free` in `TurboQuantRotateQCube` |
| Ping-pong butterfly stage (`src`/`dst` swapped after each stride) | `PipeBarrier<PIPE_V>()` — **always kept** | `FastWalshHadamardTransform` in the codec; `Residual` in rotate_q |
| Dependent arithmetic inside one V pipe (`Mul`, `Muls`, `Adds`, `Sub`, `Exp`, `Cast`, reduces) | **No barrier.** The kv4fp8 decode and writer chains carry none; older kernels still switch theirs through `VecBarrier<kSwitch>()`, which compiles to nothing when the switch is false | `TurboQuantVectorDecodeService`, `UnpackAffine`, `ApplyPi<false>` |
| One UB buffer rewritten through another type, or two overlapping writes whose order is the result | `PipeBarrier<PIPE_V>()` — **kept** | `Encode`'s bin lanes (float → uint32 → int32); `BeginTask` |
| Kernel `Init` staging, and GM write-back after compute | `PipeBarrier<PIPE_ALL>()` | `Init()` of every operator; `WriteOutput` |
| Leaving the split phase before an in-launch reduction | `AscendC::SyncAll<true>()` | `TURBOQUANT_PAGED_ATTENTION_FUSED_DECLARE` |

### 1.2 Hard events in use

| Event | Direction | Where it gates |
| --- | --- | --- |
| `MTE2_V` | GM read → vector | After `DataCopy(qIn, queryGm_…)`, and after a whole-row KV read by `TurboQuantTileBurst` |
| `V_MTE2` | vector → GM read | Before re-reading GM for chunks > 0 (`StageOperands`), and before a whole-row KV burst |
| `MTE2_MTE3` / `MTE3_MTE2` | GM read ↔ L1 write | Hadamard `h16` staged into `b1_`; rotated query written back to GM |
| `V_MTE3` / `MTE3_V` | vector ↔ L1 or GM write | Cast operands before `DataCopy` into `aHi1_` / `aLo1_`; `Residual` write-back |
| `M_MTE1` / `MTE1_M` | Cube ↔ L1 → L2 load | Surrounds `LoadA` / `LoadB` in `LoadCubeOperands` |
| `FIX_M` / `M_FIX` | Cube ↔ Fixpipe | Before `Mmad`, and between `Mmad` and `Fixpipe` in `MmadAndFixpipe` |

After `Fixpipe`, the kernel still issues `PipeBarrier<PIPE_FIX>()`, and a cross-core
`kFlagProductReady` releases the AIV `Residual`.

**`FIX_V` / `V_FIX` do not exist on this part.** `kernel_event.h` lowers them only under
`__NPU_ARCH__ == 5102`. On arch35 (3510) `SetFlag<FIX_V>` and `WaitFlag<FIX_V>` fall to an empty
`default:` assert and emit nothing. In a MIX kernel the Fixpipe product also lands on another core.
The hand-off that works is a cross-core flag set on `PIPE_FIX` by the AIC and waited for by the AIV
(`kFlagScoresReady`, `kFlagContextReady` in the fused decode).

### 1.3 The barrier rule and its A/B guard

A barrier between dependent intra-pipe arithmetic ops costs vector time. The user's silicon breakdown
attributes about 31 us to barrier inflation. The claim that ccec tracks intra-pipe RAW/WAR hazards
without barriers is user-stated. It contradicts CANN's own `sigmoid_v100_impl.h`, which puts
`PipeBarrier<PIPE_V>` between dependent vector ops. For that reason:

- Removable barriers go through `template <bool ENABLED> VecBarrier()` (`turboquant_codec_950.h`), keyed
  by a per-kernel `constexpr bool` such as `kRotateQVecBarriers = false`. This keeps a one-line A/B
  against the barriered build.
- Ping-pong barriers stay unconditional. After each butterfly stage the buffers swap, and the next stage
  reads what the previous one wrote.
- **Camodel evidence (barriers off):** `test_sim_950pr_turboquant_rotate_q` is bit-identical to the CPU
  reference. The AIV path gives 0 of 2,048 elements differing at N=8. The Cube path gives 0 of 16,384 at
  N=64 (D=256, dual-destination variant `0x2`). The HiLo residual variant (`kHiLo`) is not requested by
  production planning, so it has **not** been run with barriers off.
- **Shipping AIV decode:** `AccumulateTile`, `RowSums` and `WriteOutput` in `turboquant_paged_attention.cpp`
  switch 26 barriers, plus those in the broadcast helpers they call, behind
  `kPagedAttentionVecBarriers = false`. With barriers off, `test_sim_950pr_turboquant_kernels` still
  reports cos 1.000000, SNR 72.87 dB and relL2 2.27e-4 against the CPU reference. That matches the
  barriered build to printed precision, but it is not a bit-level comparison. One barrier stays in
  `ComputeSplit`: it separates `Duplicate(state, 0)` from `Duplicate(runMax, -inf)`, which write
  overlapping memory, so their order is a real write-after-write dependency.
- **Fused Cube decode:** `TurboQuantFusedDecode` runs with `kFusedVecBarriers = false`.
  - **History.** On one kv4fp8 tile it was bit-identical (0 of 1,024 elements) to the fully barriered
    split + combine it replaced.
  - **Barriers removed.** Since TURBOQUANT_TESTS.md 13.29, the decode's own chains (softmax, logits,
    query quantisation, accumulation, tail mask, affine unpack) carry no switchable barriers at all.
  - **The A/B reference.** The test-only instance `turboquant_mm_fused_decode_barriered_impl` now
    differs only inside the broadcast and cast helpers that still take the switch. All five fused
    cases are still bit-identical to it.
- **kv4fp8 cache writer:** `kBarrierFreeWriter<MODE>` covers both chains.
  - **Rotation.** It runs `ApplyPi<false>`.
  - **Encode.** It keeps only the four barriers that mark a reinterpretation.
  - **Tables.** They are handed over with `SyncMte2ToVector()` instead of `PipeBarrier<PIPE_ALL>`.
  - **Result.** The written cache is byte-identical to the host encoder (case (e)).
  - **Cost.** On the camodel the purge moved the writer launch by -1.4%; its time is
    vector-function dispatch.
- **Overlapping initialisations keep a real barrier.** Three sites zero a block and then overwrite
  part of it, and each keeps an unconditional barrier:
  - `BeginTask` and `TurboQuantPartialReducer::Reduce`, which write `-inf` into a lane of the block;
  - the writer's `Duplicate -> Gather` over its scale lanes.

### 1.4 Deadlock diagnosis

`CrossCoreWaitFlag` defaults to `PIPE_S`, so a missing `CrossCoreSetFlag` makes the scalar unit spin. On
the camodel, read the printable tail of each `core<N>.{cubecore0,veccore0,veccore1}.ccu_log.dump`. A
deadlock looks like one `WAIT_FLAG_DEV` line repeating with an unchanging `instr.id`, and its
`wait_flag_id` names the flag. A clean run ends every core on `instr.name=END`, and every `excp_log.dump`
is 0 B.

---

## 2. Subcore Load-Balancing

Each arch35 AIV core has two vector subcores, `GetSubBlockIdx() ∈ {0, 1}`, and each subcore has its own
UB. A kernel that gates work behind `GetSubBlockIdx() == 0` (the old `IsPrimarySubcore()` pattern)
leaves half the vector units idle.

### 2.1 AIV-only kernels: index tasks across all subcores

Launch with `blockDim` sized for `aiv_num` subcores (64 on the camodel: 32 cores × 2 subcores). Then
partition tasks by `GetBlockIdx()` directly:

```cpp
uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
uint32_t end = std::min(start + tasksPerCore, tasks);
```

`TurboQuantPagedAttentionSplit::Process` does this. When a per-core base is needed instead, as in
`TurboQuantRotateQAiv::Process`, compute `MixBlockIdx() = GetBlockIdx() / GetSubBlockNum()`. Then stride
by `GetSubBlockNum()`, starting at `base + GetSubBlockIdx()`.

### 2.2 Cube → AIV: dual-destination Fixpipe

When the Cube produces the rows, one `Fixpipe` feeds both subcores' UBs:

```cpp
AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(tile, chunkRows, paddedRows, tile);
if (DualDst()) {                 // variant has kDualDst and vectorsPerChunk is even
    fp.dualDstCtl = 0b01;        // == 1
    fp.subBlockId = false;
}
AscendC::Fixpipe<float, float, kFixpipeToUb>(prodBuf_[slot].Get<float>(), acc, fp);
AscendC::PipeBarrier<PIPE_FIX>();
```

Rows are then partitioned contiguously (`TurboQuantRotateQCube::MyVectors` / `MyChunkOffset`):

| Subcore | Rows owned | GM write offset |
| --- | --- | --- |
| `GetSubBlockIdx() == 0` | `[0, M/2)` | `(blockBase + chunk·M) · D` |
| `GetSubBlockIdx() == 1` | `[M/2, M)` | `(blockBase + chunk·M + M/2) · D` |

Each subcore runs `Residual` (butterflies plus the scaled sign flip) on its half only, then writes its
rows to GM.

**Pitfall:** `DualDst()` also requires an even `vectorsPerChunk`. If it is odd, subcore 0 takes every row
and subcore 1 idles. Plan chunk sizes to be even (`PlanRotateQ` yields tile multiples of 16).

**Camodel evidence:** Cube rotation with `variant 0x2` is bit-identical to both the AIV path and the CPU
reference (`test_sim_950pr_turboquant_rotate_q`). The ~1.44x share of the D=256 FWHT speed-up attributed
to the dual-destination Fixpipe is a camodel tick measurement, not silicon latency.

**The Cube decode uses the same split.** In `TurboQuantFusedDecode`, a task is (token, kv head,
split, head chunk) with M heads. Subcore 0 owns heads `[0, ceil(M/2))` and stages the K plane into
L1; subcore 1 owns the rest and stages the V plane, so neither subcore idles while the other unpacks.
`TurboQuantCubeMm::GemmScores<true>` / `GemmContext<true>` pad M to even and Fixpipe with
`dualDstCtl = 1`. Each subcore then runs the softmax, accumulation and GM writeback for its own
heads. Both subcores still set every AIV → AIC flag, because the AIC's wait needs both.

**Alternative partitions:** a contiguous half is what ships. Even/odd head interleaving (subcore 0 takes
even heads) is equally valid when rows map to heads. It avoids an offset calculation but makes the GM
write-back non-contiguous, which works against rule 3.

---

## 3. Memory Bandwidth Conservation (MTE2 / GM)

The HBM controller throttles short DMA reads. The user reports that a 32 B group-major tile read drops
throughput to 44.1 GB/s, under 3% of the 1.6 TB/s peak. **Never fragment MTE2 reads below 128 B.**

### 3.1 Rules

1. Read GM in wide contiguous bursts of at least `kMinBurstBytes = 128` B. Prefer one `DataCopy` over a
   whole tile or plane rather than one per head or per group.
2. Store GM in the order its reader wants, so the read needs no permutation at all. kv4fp8's cache
   is NZ-tiled by the cache write (§3.3), and its decode reads one contiguous burst per (tile, kv head).
   Where the layout cannot change, do the NZ or head permutation in the vector unit, after the burst.
   Three options:
   - `DataCopy(UB, UB, DataCopyParams)`, a block move that CANN lowers on `PIPE_V` (`CopyUbufToUbuf`)
   - `Gather` from a contiguous UB tensor
   - the native signed-nibble `Cast` + `DeInterleave`
3. Bracket the burst with `V_MTE2` → `MTE2_V`, so the vector unit does not consume the UB before MTE2
   has filled it.

### 3.2 The KV tile read (`TurboQuantTileBurst`)

| Case | Read | Burst size |
| --- | --- | --- |
| `numKvHeads == 1` | `DataCopy(kv, keyCacheGm_[rowOff], kTileRows · packedBytes)`, and the same for V | 16 · D/2 B, contiguous |
| `numKvHeads > 1`, `packedBytes ≥ 128` (D ≥ 256) | `DataCopy(kv, keyCacheGm_[cacheOff], rowParams_)`: one head's `packedBytes` per row, skipping the other heads | ≥ 128 B per row |
| `numKvHeads > 1`, `packedBytes < 128` (`wholeRowRead_`) | Whole row plane into `rowsBuf_` (`kTileRows · packedPlane` B), then `DataCopy(kv, rows[headOff], rowParams_)` UB → UB | 16 · H_kv · D/2 B, contiguous |

`rowParams_` is `DataCopyParams{kTileRows, packedBytes/32, (packedPlane − packedBytes)/32, 0}`. The step
and skip fields count 32 B units, so `packedBytes` must be a multiple of 32.

The scale plane is read the same way, with one contiguous `DataCopy(scales, scaleCacheGm_[scaleOff],
kTileRows · scaleSlot)`.

**Trade-off:** `wholeRowRead_` reads every KV head's bytes for each task. That costs UB space (`rowsBuf_`
holds both planes) and some redundant bytes, in exchange for keeping the controller in burst mode. The
test shape (D=64, H_kv=2) exercises this path.

**Who uses it.** `TurboQuantTileBurst` (vector service) implements the three cases. The AIV decode
reads a tile's K and V planes through it. The Cube decode's row-major modes (kv3fp4, kv5fp8) read one
plane per subcore through it and unpack in bands. kv4fp8 does not use it; see §3.3.

### 3.3 NZ-tiled storage (kv4fp8): the permutation moves to the cache write

Cube L1 is always NZ: `nz(r, c) = (c / 32) · rows · 32 + r · 32 + c % 32`. A row-major cache therefore
needs a permutation on every decode tile. kv4fp8 pays for it once per token, when the cache is written.

- **Layout** (`NzTiledPackedByte` in `turboquant_layout.h`, keyed by `kStoresNzTiles<MODE>`). Each
  physical block is `[64-row tile][kv head][32 B column group][tile row][byte]`, so the packed plane
  of one (tile, kv head) is the contiguous `[D/64, 64, 32]` image. The block size must be a multiple
  of 64. The cache and scale-plane sizes are unchanged. The scale plane stays row-major.
- **Write** (`TurboQuantModeReshapeAndCache::WriteNzTiled`). One strided UB → GM burst per plane per
  token, `DataCopyParams{H_kv · D/64, 1, 0, 63}`. Consecutive (kv head, group) cells of one tile row
  are exactly one tile apart. The 32 B cells are the price, paid once per token.
- **Read** (`ReadPlane`). One contiguous GM → UB burst of `64 · D/2` B (8,192 B at D 256) at
  `(row · H_kv + kvHead · 64) · D/2`, at every block size. This never falls back to the 32 B
  group-major decode read that §3 forbids.
- **Unpack** (`UnpackPlaneToL1`). `Cast<half, int4b_t>` + `DeInterleave` + the plane expand work byte
  by byte. So a chunk's low nibbles land in place on NZ groups `[0, D/64)` and its high nibbles on
  `[D/64, D/32)`, one tile plane further on. The staging buffer is the L1 operand as unpacked, and one
  MTE3 burst stages it. There is no UB → UB permute loop and no separate operand buffer (8 KB less UB).
- **Host pin.** `TurboQuantNzTiledCache.TilesAreContiguousNzImagesAndTheWriterBurstHitsThem` checks
  that the layout is a bijection, that each read window holds exactly its tile's cells in NZ order,
  and that the writer's strided burst lands on them. It covers H_kv 1/2/8, D 64/128/256 and block
  64/128.
- **Executed.** `TurboQuantFusedDecode.DecodesTheCacheTheKernelWriterWrote` (case (e)) runs the
  kv4fp8 kernel writer and checks the result. With independent vector halves, the written planes
  match the host encoder in all 131,072 bytes. Lane 0 is the low nibble: swapping the lanes makes
  29,700 bytes differ. The fused decode of that cache matches its golden.
- **Cycles (camodel).** A tile stages in 29-36% fewer cycles than with the UB permute, and the fused
  launch spans fell 9-20% on cases (a) to (d) (TURBOQUANT_TESTS.md 13.28).

---

## 4. Online Softmax Direct Writeback (retiring Split-Combine)

The old decode ran two launches, *split* and *combine*. The split launch wrote per-segment partials
(`acc`, running max, running log-sum) to a GM workspace. A separate combine launch read those partials
back and reduced them. For B=1 and S ≤ 4096 this paid a second dispatch, a stream sync and a GM
re-read, and bought nothing.

### 4.1 Rule

For contexts with `blocks · blockSize ≤ kFusedContextLimit (4096)`, run the whole softmax online in UB
in **one launch**, with no workspace and no combine. Normalize at the end and write the output token
straight to GM.

### 4.2 Planning (`turboquant_torch_adpt.h`)

```cpp
const int64_t blocks = std::max<int64_t>(1, maxBlocksPerSeq);
int64_t num_splits = 1;
if (blocks * blockSize > kFusedContextLimit) {
    num_splits = std::max(CeilDiv(aivNum, base_tasks), CeilDiv(blocks * blockSize, kFusedContextLimit));
    num_splits = std::min(num_splits, std::min<int64_t>(kMaxSequenceSplits, blocks));
}
...
plan.workspace_floats = num_splits > 1 ? split_tasks * PartialStride(headSize) : 0;
plan.reduce_tasks_per_core = num_splits > 1 ? CeilDiv(base_tasks, block_dim) : 0;
```

When `workspace_floats == 0`, the adapter passes `nullptr` for the workspace.
`TurboQuantLaunchContract.GridPlansMatchTheAdapterArithmetic` asserts `num_splits == 1` and
`workspace_floats == 0` for every shape that fits the limit.

### 4.3 In-kernel mechanics (`TurboQuantPagedAttentionSplit`)

For each tile, `AccumulateTile` keeps a numerically stable running softmax in UB:

1. Compute `scores = (K̂ · q) · kScale · scale` and mask invalid rows to `−inf`.
2. Compute `newMax = max(runMax, max(scores))` and `alpha = exp(runMax − newMax)`.
3. Compute `probs = exp(scores − newMax)`.
4. Update `runSum = runSum · alpha + Σ probs`.
5. Update `acc = acc · alpha + Σ_rows probs · vScale · V̂`, then set `runMax = newMax`.

`WriteOutput` then runs once per (token, head):

```cpp
Adds(runSum, runSum, kEps, 1);                   // 1 / (Σ + ε)
Duplicate(invSum, 1.0f, kFp32PerBlock); Div(invSum, invSum, bSum, kFp32PerBlock);
BroadcastMul(acc, acc, invSum, headSize_);
Cast(out, acc, CAST_RINT, headSize_);            // fp32 → fp16/bf16 in UB
SyncEvent<HardEvent::V_MTE3>();
DataCopy(outputGm_[(token · numHeads + head) · headSize], out, headSize_);
SyncEvent<HardEvent::MTE3_V>();
```

`TurboQuantFusedDecode` does the same on the Cube path, per subcore, for that subcore's heads. It
normalizes all of them with one repeat `Mul` and writes them to GM in one `heads.mine · D` burst.

### 4.4 Long contexts stay one launch

When `S > 4096`, the split phase still writes partials. The same launcher then calls
`AscendC::SyncAll` and hands every split token to `TurboQuantPartialReducer` (vector service), which
reads each partial in one `partialStride` burst, merges the splits with the online-softmax recurrence and
closes out through `WriteNormalizedHeads`. Both decodes use it; there is no combine kernel and no second
host launch.

### 4.5 Evidence (camodel, Ascend950PR_9589, one process each, all `excp_log.dump` 0 B)

| Test | Path | Result |
| --- | --- | --- |
| `TurboQuantKernels.PagedAttentionMatchesTheCpuReference` | AIV fused (splits = 1), aiv = 64, barriers off | cos 1.000000, SNR 72.87 dB, relL2 2.27e-4 |
| `TurboQuantFusedDecode.IsBitIdenticalToItsBarrieredInstance` | Cube fused, kv4fp8, H_Q 4, H_KV 2, D 256, S 64: 1 launch, block_dim 2 | 0 of 1,024 elements differ from the barriered instance; cos 0.999407 vs fp32, the value the retired split + combine produced; no sentinel value left |
| `TurboQuantSimulatorFidelity.SingleDecodePassQuantisedVersusExact` | AIV, forced splits = 2, reduction by `TurboQuantPartialReducer` | vs CPU TurboQuant: cos 1.000000, SNR 74.00 dB, relL2 1.99e-4; vs exact fp32: cos 0.991182 (4-bit quantization error) |
| `TurboQuantFusedDecode.DecodesTheCacheTheKernelWriterWrote` | kv4fp8 kernel writer (NZ-tiled, barrier-free) → Cube fused, H_Q 4, H_KV 2, S 64 | written planes byte-identical to the host encoder (0 of 131,072); fused = barriered; cos 0.999576 vs fp32 |
| `TurboQuantMultiMode.DispatchAndFidelityAcrossShapes` | Cube fused, kv4fp8, S 16 (one partial tile, the `Min` mask) | cos 0.991817 vs fp32, measured on the row-major cache; not re-run since kv4fp8 became NZ-tiled (§3.3) |

The residual error against the CPU reference is fp16 output rounding. Neither test is a latency
measurement.

---

## 5. Checklist for a New arch35 Kernel

- [ ] Every pipe crossing has a hard event. Every AIC ↔ AIV handoff has a cross-core flag, and each
      `CrossCoreWaitFlag` has a matching `CrossCoreSetFlag` on every path.
- [ ] No `PipeBarrier<PIPE_V>` between dependent arithmetic ops. The only barriers kept are for
      ping-pong buffers, a buffer rewritten through another type, and overlapping writes whose order
      is the result.
- [ ] No `HardEvent` that the part does not lower: `FIX_V` / `V_FIX` compile to nothing on arch35.
- [ ] No `GetSubBlockIdx() == 0` gating. Tasks are indexed across all subcores, or a Cube feeds both
      through `dualDstCtl = 0b01` with an even row count.
- [ ] No GM read below 128 B. Prefer storing GM in the reader's order (§3.3); otherwise permutations
      happen in UB.
- [ ] Decode for S ≤ 4096 is one launch: no workspace, no combine, output written to GM from UB.
- [ ] Camodel runs stay at S ≤ 256, one process per case. Latency claims come from a silicon msprof
      trace (`prof_device_950pr_msprof_trace`), never from camodel ticks.

## 6. Open Items

1. Capture a silicon msprof trace of both fused decodes (`prof_device_950pr_msprof_trace`, whose AIV
   and Cube legs now each issue one decode launch) and compare them against
   `aclnnFusedInferAttentionScoreV5`. No number in this guide is a silicon latency measured in this
   repository.
2. Run the fused Cube kernel on the paths the camodel smoke does not reach: kv3fp4 and kv5fp8, head
   chunking (`H_Q / H_KV > 16`), and the in-launch reduction above 4096. The reduction and deep
   contexts belong on silicon.
3. Measure the shipping AIV decode bit-for-bit against its barriered build. The camodel check so far
   matches only to printed precision.
4. Re-run `test_sim_950pr_turboquant_multimode` for kv4fp8. The NZ-tiled writer itself has now executed
   (fused case (e)), but the multimode figure in §4.5 still predates the layout.
5. Measure vector-function dispatch latency on silicon. On the camodel it moves fused spans by up to
   ~3% between builds that differ only in code layout, and it is most of the kv4fp8 writer's time.
