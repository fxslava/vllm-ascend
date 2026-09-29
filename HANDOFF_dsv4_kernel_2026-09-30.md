# HANDOFF: DSV4 MoE Expert Kernel — 2026-09-30

## 1. Goal (unchanged)
Build a real Ascend C kernel for the DeepSeek V4 Flash MoE expert,
replacing DummyExpertKernelRunner. Validate on CPU simulator at reduced
geometry before scaling to production. Target: Ascend 950PR.

## 2. Environment (verified this session)
- WSL2 Ubuntu-22.04, native dockerd 29.7.2
- Image: quay.io/ascend/vllm-ascend:v0.26.0rc1-a5
- Image presets SOC_VERSION=ascend950dt_9582 (WRONG variant).
  Must override with: -e SOC_VERSION=ascend950pr_9579
- No NPU attached to this workstation. All work is simulator-only.
  Do NOT attempt device passthrough.
- Docker storage at stop: images 63.63 GB total (12.74 GB reclaimable),
  build cache 33.09 GB, 5 containers (31.5 MB). Disk backing WSL volume
  has ~817 GB free — ENOSPC is NOT a risk on the volume; the 8 GB
  ceiling in older notes referred to docker storage, which was not
  consumed by these builds (all artifacts go to the mounted volume).
- torch_npu 2.10.0.post4 imports WITHOUT a driver if
  /usr/local/Ascend/cann-9.1.0/x86_64-linux/simulator/dav_3510/camodel
  is prepended to LD_LIBRARY_PATH (simulator camodel provides
  libascend_hal.so). Benign warning "can not use command: npu-smi info".
- `ascendebug` does NOT exist anywhere in this image (full-FS search).
  The equivalent tooling that IS present:
    * CPU backend: tools/cpudebug + tools/tikicpulib (interpreter
      cmake package; `tikicpulib_ascend950pr_9599` serves the whole
      C310/ascend950 family INCLUDING ascend950pr_9579 — verified in
      tools/cpudebug/cmake/cpudebug-config.cmake PRODUCT_TYPE_LIST_C310_)
    * simulator backend: `cannsim` (renamed `npusim`) in
      $ASCEND_HOME_PATH/bin — `record` subcommand wraps a user app.
- Simulator SoC inventory (x86_64-linux/simulator/): NO
  ascend950pr_9579. Present 950PR siblings: 9571-9578, 957b/c/d, 9581-
  9589, 958a/b, 9591/9592/9595/9596/9599, 95A1/95A2, 950x/y/z. Also
  dav_3510 (the arch-level model). CANN itself maps 9579 to the
  ascend950pr_9599 series for CPU debug — use that (or 9599/9582) as
  the documented approximation for Gate C, per the task's fallback rule.
- Source tree synced from:
    /mnt/d/Projects/vllm-ascend/csrc/moe/dsv4_moe_expert/
  to:
    ~/dsv4-kernel/csrc/moe/dsv4_moe_expert/
  WARNING: this sync has been error-prone. One build ran against a
  stale tiling.cpp due to a malformed sync command, and all synced
  files arrive with CRLF (the D: tree is Windows-checked-out); every
  file must be `sed -i "s/\r$//"`-ed or bash/cmake steps break with
  `$'\r': command not found`. Always verify hashes + line endings
  after sync, before building.

## 3. Reduced geometry (locked for this milestone)
hidden=256, moe_inter_dim=128, block-32 E8M0 scales.
  w1, w3: packed FP4 [128,128] uint8 + scales [128,8] E8M0
  w2:     packed FP4 [256,64]  uint8 + scales [256,4] E8M0
  gate_out, up_out, activated: [1,128] bf16
  down_out: [1,256] bf16
  x: [1,256] bf16
Structure must match production: same block size, same 2-FP4-per-byte
packing, same scale granularity. Only dims shrink.
Nibble convention (kernel AND golden must agree): byte i holds logical
element 2i in the LOW nibble, 2i+1 in the HIGH nibble. E8M0 byte b
decodes to 2^(b-127) exactly (b=0 -> fp32 subnormal 2^-127 via
0x00400000; b=0xFF -> NaN per OCP MX).

## 4. Build command (working shape — VERIFIED SUCCESSFUL)
docker run --rm -v $HOME/dsv4-kernel:/work \
  -e SOC_VERSION=ascend950pr_9579 \
  --entrypoint bash \
  quay.io/ascend/vllm-ascend:v0.26.0rc1-a5 -c "
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd /work
    CUSTOM_OPS=dsv4_moe_expert bash csrc/build_aclnn.sh /work ascend950pr_9579
  "

Prerequisites baked into ~/dsv4-kernel (already done, do not redo):
- csrc/third_party/catlass extracted from the image's checkout (the
  local submodule is empty; without catlass/include/ the script tries
  a git fetch that fails — /work is not a git repo).
- csrc/third_party/ascend_protobuf and abseil-cpp REPLACED with the
  image's extracted trees (as container root, then chmod -R u+w). The
  build's own tarball extraction produced a protobuf tree whose
  external build failed repeatably ("opening dependency file
  ...cc.o.d: No such file or directory"); the image's trees build fine.
- All workdir text files CRLF-stripped (see section 2 warning).
- The CUSTOM_OPS env-override hook in csrc/build_aclnn.sh (captured
  CUSTOM_OPS before the per-SoC default lists clobber it). This hook
  exists in /mnt/d/.../csrc/build_aclnn.sh (modified, NOT yet
  committed — see section 9) and in the WSL workdir copy.
- WSL workdir ~135 MB source + build artifacts; full single-op build
  takes ~6-8 minutes wall clock.

## 5. Current state (CORRECTED — supersedes the earlier "kernel fails" state)
The kernel COMPILES and the FULL BUILD SUCCEEDS. Frozen log
~/dsv4-kernel/build_full.log.frozen (1045 lines) is the successful run:
"installer finished" present, zero ninja FAILED lines.

Historical compile failures, ALL NOW FIXED (do not re-diagnose):
- TBufPosition does not exist on CANN 9.1 arch35 -> use
  `TBuf<TPosition::VECCALC>` / `TQue<TPosition::VECIN, N>` (verified
  against csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp).
- Kernel-side tiling struct must be a PLAIN header
  (op_kernel/dsv4_moe_expert_tiling_data.h) mirroring the host
  BEGIN_TILING_DATA_DEF field order; GET_TILING_DATA_WITH_STRUCT reads
  the raw bytes. (situ_mx_quant does the same.)
- Host tiling: TilingContext::GetInputShape returns
  gert::StorageShape* -> use ->GetStorageShape(); fill the generated
  struct via set_* accessors; only def.cpp goes into
  target_sources(op_host_aclnn) — tiling.cpp is picked up by
  add_modules_sources_with_soc, infershape.cpp is not compiled by the
  open-project path at all.
- expf is NOT available in device code -> vector `Exp()` API with
  SetFlag/WaitFlag<HardEvent::S_V>/<V_S> sync (mirrors chunk_kda_fwd
  RunExp2; event id 7 used).
- bisheng backend REJECTS scalar bfloat16_t casts ("not support bf16
  type cast") -> ALL bf16<->fp32 conversions are explicit bit
  manipulation (uint16 views for GM/UB copies; Bf16BitsToFloat /
  FloatToBf16Bits with round-to-nearest-even, specials truncated) —
  byte-identical to the golden's numpy helpers.
- torch adapter: at::kUInt8 -> torch::kUInt8.

Gate A status at stop (three of four checks green):
- build succeeds: YES (BUILD_SCRIPT_EXIT=0; the exit line was echoed to
  docker stdout, the log itself ends with the installer listing).
- op appears in vllm_ascend/_cann_ops_custom: YES —
  vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend950/
  dsv4_moe_expert/Dsv4MoeExpert_<hash>.{o,json},
  kernel/config/ascend950/{binary_info_config.json,dsv4_moe_expert.json},
  op_api/include/aclnnop/aclnn_dsv4_moe_expert.h, proto header,
  dynamic impl .py. nm -D on libcust_opapi.so shows exported
  `aclnnDsv4MoeExpert` + `aclnnDsv4MoeExpertGetWorkspaceSize`;
  Dsv4MoeExpert strings present in opsproto + opmaster libs.
- torch.ops._C_ascend.dsv4_moe_expert: the standalone mini-binding
  (~/dsv4-kernel/gatecheck/mini_binding.cpp + build.sh) COMPILED
  against torch+torch_npu headers. The final python resolution step
  (`torch.ops.load_library(...); print(torch.ops._C_ascend...)`) was
  cancelled mid-run by the session stop — re-run it, it takes seconds
  (bash /work/gatecheck/build.sh in a container; needs the simulator
  camodel LD_LIBRARY_PATH which build.sh already sets).
- NOTE for interpreting this check on real hosts:
  vllm_ascend/utils.py enable_custom_op() deliberately returns False
  for A5/Ascend950 (FIXME referencing vllm-ascend issue #7157 —
  "custom op compilation and execution are partially available in
  ASCEND950"). So on a real 950PR the stock loader will not register
  torch.ops._C_ascend.* regardless of our op. The op IS wired into
  csrc/torch_binding.cpp (include + ops.def + ops.impl) like every
  other 950 op; the mini-binding is the isolated proof.

## 6. Exact next step (resume here)
1. Re-run the cancelled resolution check:
   docker run --rm -v $HOME/dsv4-kernel:/work --entrypoint bash \
     quay.io/ascend/vllm-ascend:v0.26.0rc1-a5 /work/gatecheck/build.sh
   Expect "RESOLVED: ...dsv4_moe_expert" + schema print.
2. Gate B (CPU numerics). ascendebug does not exist; drive the CPU
   interpreter directly:
     - Golden data: python3 tools/dsv4_moe_runtime/kernel_bringup/
       gen_golden.py gen --out-dir <dir>  (writes input.bin/golden.bin/
       meta.json; the script also self-compares exactly; run `cmp` with
       kernel outputs to produce z.txt with per-buffer FpDiff/RateDiff
       in bf16-ULP terms — pass: FpDiff<=2 ULP, RateDiff<=1e-2).
     - Kernel execution: compile op_kernel/dsv4_moe_expert.cpp in CPU
       mode against tikicpulib (find_package(tikicpulib) alias
       tikicpulib::Ascend950PR_9579) and launch via
       AscendC::RunKernelFunctionOnCpu (see
       tools/cpudebug/include/cpu_debug_launch.h + kern_fwk.h).
       Feeds: x/w1/w2/w3/scales from input.bin regions in kernel-arg
       order (offsets in meta.json), outputs dumped as four .bin files.
       The kernel is AIV-only, block_dim=1, tiling struct is 3 packed
       int64 (hiddenSize, interSize, blockSize=32) — build the 24-byte
       tiling buffer by hand if the host tiling path is not driven.
3. Gate C (simulator): cannsim/npusim record --soc-version Ascend950
   (or the closest 950PR variant; 9579 absent — see section 2), wrap
   the CPU-mode driver app; read total tick + per-pipe breakdown from
   the generated report/debug_op.log. Report unpack-tick/GEMM-tick
   ratio (flag if unpack > 40%).

## 7. Validation gates (unchanged, in this order)
- Gate A: build succeeds; op appears in _cann_ops_custom listing;
  torch.ops._C_ascend.<op_name> resolves (mini-binding workaround
  above; on stock loaders A5 is disabled by design — issue #7157).
- Gate B: CPU-backend numerics (via cpudebug/tikicpulib since
  ascendebug is absent). Report FpDiff and RateDiff.
  Pass: FpDiff <= 2 ULP of output dtype, RateDiff <= eps.
  Do NOT proceed to Gate C if Gate B fails.
- Gate C: simulator via cannsim/npusim (block-num 1, timeout budget
  1200 s). Report total tick, and unpack-tick / GEMM-tick ratio.

## 8. Process notes (lessons from this session)
- DO NOT use sleep N to wait for builds. Poll with early exit:
    for i in $(seq 1 30); do
      grep -qE "BUILD_SCRIPT_EXIT=|\[ERROR\] TBE" $LOG && break
      sleep 10
    done
    tail -40 $LOG
- The build log contains BOTH ninja FAILED markers AND [ERROR] TBE
  markers. A grep for "FAILED" alone misses ccec crashes entirely.
  Always check both. (grep treats the log as binary — use grep -a.)
- Verify source sync before every build. Silent sync failures have
  already caused at least one wasted build cycle.
- Do not rebuild the entire custom op set. Use CUSTOM_OPS=<name>.
- If ENOSPC occurs, run `docker builder prune` on the host.
  Do not delete files inside the running container.
- CRLF: every file crossing /mnt/d -> WSL must be stripped (see §2).
- CANN writes root-owned read-only files into the mounted workdir;
  cleaning them requires docker run --rm ... rm -rf as root.
- The protobuf/abseil external builds are flaky from the tarball
  extraction; the image's pre-extracted trees are the reliable source.
- Output dtype decision (per task note): outputs are bf16 (matching
  the ExpertKernelRunner scratchpad contract exactly). FP8 output is
  the production follow-up; the golden + kernel both compute fp32
  internally and round to bf16 RTNE via identical bit-level helpers.
- Do not trust log timestamps: TBE replays cached error info with old
  timestamps; check ninja step numbers ([N/72]) and artifact existence
  instead.

## 9. Artifacts left behind
- ~/dsv4-kernel/build_full.log.frozen — frozen build log (SUCCESSFUL
  run, 1045 lines), DO NOT TOUCH
- ~/dsv4-kernel/csrc/moe/dsv4_moe_expert/ — current WSL-side source
  (identical to D: side)
- ~/dsv4-kernel/gatecheck/{mini_binding.cpp,build.sh,mini_binding.so}
  — Gate A torch-resolution check (compiled .so present)
- ~/dsv4-kernel/vllm_ascend/_cann_ops_custom/ — INSTALLED vendor
  package from the successful build (root-owned files inside)
- /mnt/d/Projects/vllm-ascend/csrc/moe/dsv4_moe_expert/ — D: side,
  committed to git in the WIP commit preceding this document.
- UNCOMMITTED but on disk on D: (review + commit next session):
    * csrc/build_aclnn.sh — CUSTOM_OPS override hook + op added to the
      ascend950 default list (REQUIRED for the §4 build command)
    * csrc/torch_binding.cpp — include + ops.def/ops.impl registration
      for dsv4_moe_expert (3 additions, mirrors situ_mx_quant)
    * tools/dsv4_moe_runtime/kernel_bringup/gen_golden.py — deliverable
      3 golden generator/comparator (self-test passed on Windows host)
- This handoff document, committed immediately after the WIP commit.

## 10. What NOT to do next session
- Do not modify DummyExpertKernelRunner.
- Do not attempt NPU passthrough — no hardware on this host.
- Do not scale geometry until Gate B passes at reduced size.
- Do not rebuild the whole custom op set; use CUSTOM_OPS=<name>.
- Do not trust ambient SOC_VERSION inside the image; always override.
- Do NOT re-diagnose the kernel compile — it builds (see §5); the
  remaining work is validation (Gates B/C) and the 30-second torch
  resolution re-run.
- Do not use `ascendebug` — it is not in the image; use cpudebug/
  tikicpulib + cannsim/npusim (see §2, §6).

## 11. Open questions for next session
- Simulator for ascend950pr_9579: ABSENT (verified — see §2 inventory).
  Planned fallback: CANN's own mapping puts 9579 in the
  tikicpulib_ascend950pr_9599 series for CPU; for Gate C use
  Ascend950PR_9599 (or 9582, which exists under the PR name) and label
  tick numbers as approximate. Confirm which variant npusim accepts.
- Exact CPU-mode launch recipe: RunKernelFunctionOnCpu exists but the
  stub-registration boilerplate (stub_def.h/stub_reg.h) has not been
  wired into a driver main() yet — UNKNOWN — verify next session.
- Vector `Exp()` numerics vs np.exp on the boundary cases (Gate B may
  show sporadic 1-ULP bf16 diffs; criterion allows 2).
- unpack/GEMM tick attribution: whether npusim reports give per-phase
  or per-pipe granularity sufficient for the 40% ratio, or whether the
  kernel needs profiling markers — UNKNOWN — verify next session.
- enable_custom_op()/A5 policy (issue #7157): how vllm-ascend plans to
  re-enable custom ops on 950 — affects the eventual integration path.
