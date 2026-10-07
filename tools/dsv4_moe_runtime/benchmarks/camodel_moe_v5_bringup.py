"""Single-layer DeepSeek-V4 Flash MoE bring-up on the Ascend 950PR runtime.

Torch-free on purpose: the tq950-sim container ships CANN 9.2.0 without
torch, so this runner drives libascendcl + libopapi directly through the same
ctypes bindings the torch harness uses (``AclnnLibrary`` plus the shared
plan-arg builders and transcribed argtypes in ``hardware.v5_ops_moe``).

Stages, in the execution order of one DSV4 MoE layer::

    GATING       aclnnMoeGatingTopKV2    [T,256]fp32 logits -> expertIdx [T,6]
    DISPATCH     aclnnMoeInitRoutingV4   hidden fp8 -> expandedX [T*6, H] +
                 device expertTokensCountOrCumsum int64 [256]
    EXPERT_GEMM  aclnnGroupedMatmulV5    FP4 weights (40) + UE8M0 block-32 (37)
                 scales, FP8 activations -> [T*6, I] bf16

``--stages plan`` (default) runs the two-phase protocol up to
``GetWorkspaceSize``: symbol resolution, descriptor shapes/dtypes and tiling
against the deployed CANN, no kernels launched. ``--stages launch``
additionally enqueues every stage on the ACL stream with the planned
workspaces; the GMM groupList is the dispatch kernel's device-born cumsum.
``--sync`` waits and verifies ``cumsum[-1] == T*6`` -- under CAModel that
simulates the kernels and is ORDERS OF MAGNITUDE slower than real silicon.

Env (tq950-sim, repo mounted at /workspace)::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    SIM=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/simulator/Ascend950PR_9599
    export LD_LIBRARY_PATH="$SIM/lib:$SIM/camodel:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH"
    cd /workspace && timeout 600 python3 -m \\
        tools.dsv4_moe_runtime.benchmarks.camodel_moe_v5_bringup --preset micro --stages launch

Exit code 0 and the ``MOE_V5_BRINGUP_OK`` marker are printed iff every
requested stage passed on every operator.
"""

from __future__ import annotations

import argparse
import ctypes
import random
import struct
import sys
from dataclasses import dataclass, field

from ..hardware.aclnn_binding import (
    ACL_DT_BF16,
    ACL_DT_FLOAT32,
    ACL_DT_FLOAT8_E4M3FN,
    ACL_DT_FLOAT8_E8M0,
    ACL_DT_FP4X2_E2M1,
    ACL_DT_INT32,
    ACL_DT_INT64,
    AclnnLibrary,
    MOE_V5_LAUNCH_SYMBOLS,
    MOE_V5_PLAN_SYMBOLS,
)
from ..hardware.v5_ops_moe import (
    _GROUPED_MATMUL_V5_PLAN_ARGTYPES,
    _MOE_GATING_TOP_K_V2_PLAN_ARGTYPES,
    _MOE_INIT_ROUTING_V4_PLAN_ARGTYPES,
    DSV4_MOE_PROFILE,
    Dsv4GatingConfig,
    GroupedMatmulV5Config,
    grouped_matmul_v5_plan_args,
    moe_gating_top_k_v2_plan_args,
    moe_init_routing_v4_plan_args,
)

ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
PLAN_BACKING_CAP = 1 << 20  # plan mode backs the full shapes with <=1 MiB dummies

ARG_TYPES = {
    "GATING": _MOE_GATING_TOP_K_V2_PLAN_ARGTYPES,
    "DISPATCH": _MOE_INIT_ROUTING_V4_PLAN_ARGTYPES,
    "EXPERT_GEMM": _GROUPED_MATMUL_V5_PLAN_ARGTYPES,
}

PRESETS = {
    # Full DeepSeek-V4 Flash layer geometry (plan-scale validation).
    "dsv4": {"tokens": 8, "hidden": DSV4_MOE_PROFILE.hidden_size, "inter": DSV4_MOE_PROFILE.moe_intermediate_size},
    # CAModel-sized: identical routing topology (256 experts, top-6), shrunk
    # GEMMs so the kernel simulation stays inside a smoke budget.
    "micro": {"tokens": 4, "hidden": 512, "inter": 128},
}


@dataclass(frozen=True)
class LayerGeometry:
    tokens: int
    hidden: int
    inter: int
    experts: int = DSV4_MOE_PROFILE.num_routed_experts
    top_k: int = DSV4_MOE_PROFILE.num_experts_per_tok

    @property
    def rows(self) -> int:
        return self.tokens * self.top_k

    @property
    def weight_bytes(self) -> int:
        return self.experts * self.inter * self.hidden // 2  # FP4: 2 elements/byte


class AclRuntime:
    """Minimal libascendcl owner: init, stream, device memory, memcpy."""

    def __init__(self, library: AclnnLibrary, device_id: int = 0):
        self._acl = library.raw_ascendcl()
        if self._acl is None:
            raise RuntimeError(
                "libascendcl.so unavailable; export LD_LIBRARY_PATH with the toolkit lib64 "
                "and the Ascend950PR_9599 simulator dirs (see module docstring)"
            )
        self._memcpy = self._bind(
            "aclrtMemcpy",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32],
        )
        self._bind("aclInit", ctypes.c_int, [ctypes.c_char_p])(None)
        self._bind("aclrtSetDevice", ctypes.c_int, [ctypes.c_int32])(device_id)
        self._stream = ctypes.c_void_p(0)
        self._bind("aclrtCreateStream", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)])(ctypes.byref(self._stream))
        self._allocations: list[ctypes.c_void_p] = []

    def _bind(self, name: str, restype, argtypes):
        fn = getattr(self._acl, name, None)
        if fn is None:
            raise RuntimeError(f"{name} not found in libascendcl")
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def allocate(self, size: int) -> ctypes.c_void_p:
        malloc = self._bind(
            "aclrtMalloc", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int32]
        )
        pointer = ctypes.c_void_p(0)
        status = malloc(ctypes.byref(pointer), size, ACL_MEM_MALLOC_HUGE_FIRST)
        if status != ACL_SUCCESS or not pointer.value:
            raise RuntimeError(f"aclrtMalloc({size} B) failed with aclError {status}")
        self._allocations.append(pointer)
        return pointer

    def copy_in(self, pointer: ctypes.c_void_p, data: bytes) -> None:
        status = self._memcpy(
            pointer, len(data), ctypes.cast(data, ctypes.c_void_p), len(data), ACL_MEMCPY_HOST_TO_DEVICE
        )
        if status != ACL_SUCCESS:
            raise RuntimeError(f"aclrtMemcpy H2D failed with aclError {status}")

    def copy_out(self, pointer: ctypes.c_void_p, size: int) -> bytes:
        buffer = (ctypes.c_char * size)()
        status = self._memcpy(
            ctypes.cast(buffer, ctypes.c_void_p), size, pointer, size, ACL_MEMCPY_DEVICE_TO_HOST
        )
        if status != ACL_SUCCESS:
            raise RuntimeError(f"aclrtMemcpy D2H failed with aclError {status}")
        return bytes(buffer)

    def synchronize(self) -> None:
        status = self._bind("aclrtSynchronizeStream", ctypes.c_int, [ctypes.c_void_p])(self._stream)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"aclrtSynchronizeStream failed with aclError {status}")

    def stream(self) -> ctypes.c_void_p:
        return self._stream

    def teardown(self) -> None:
        free = self._bind("aclrtFree", ctypes.c_int, [ctypes.c_void_p])
        for pointer in reversed(self._allocations):
            free(pointer)
        self._allocations.clear()
        self._bind("aclrtDestroyStream", ctypes.c_int, [ctypes.c_void_p])(self._stream)
        self._bind("aclrtResetDevice", ctypes.c_int, [ctypes.c_int32])(0)
        self._bind("aclFinalize", ctypes.c_int, [])()


def deterministic_bytes(length: int, seed: int) -> bytes:
    rng = random.Random(seed)
    if hasattr(rng, "randbytes"):
        return rng.randbytes(length)
    return rng.getrandbits(length * 8).to_bytes(length, "little")


def seeded_expert_indices(tokens: int, top_k: int, experts: int, seed: int) -> bytes:
    """Valid int32 expert indices (launch-order safety net before gating runs)."""
    rng = random.Random(seed + 1)
    return struct.pack(f"<{tokens * top_k}i", *(rng.randrange(experts) for _ in range(tokens * top_k)))


def _buffer_plans(geometry: LayerGeometry) -> dict:
    """name -> (dims, aclDataType, bytes, filler seed or None)."""
    blocks = geometry.hidden // DSV4_MOE_PROFILE.fp_scale_block
    return {
        "logits": ([geometry.tokens, geometry.experts], ACL_DT_FLOAT32, geometry.tokens * geometry.experts * 4, 704),
        "bias": ([geometry.experts], ACL_DT_FLOAT32, geometry.experts * 4, 706),
        "gate_y": ([geometry.tokens, geometry.top_k], ACL_DT_FLOAT32, geometry.tokens * geometry.top_k * 4, None),
        "expert_idx": ([geometry.tokens, geometry.top_k], ACL_DT_INT32, geometry.tokens * geometry.top_k * 4, None),
        "hidden": ([geometry.tokens, geometry.hidden], ACL_DT_FLOAT8_E4M3FN, geometry.tokens * geometry.hidden, 707),
        "expanded_x": ([geometry.rows, geometry.hidden], ACL_DT_FLOAT8_E4M3FN, geometry.rows * geometry.hidden, None),
        "row_idx": ([geometry.rows], ACL_DT_INT32, geometry.rows * 4, None),
        "expanded_scale": ([geometry.rows], ACL_DT_FLOAT32, geometry.rows * 4, None),
        "cumsum": ([geometry.experts], ACL_DT_INT64, geometry.experts * 8, None),
        "x_scale": ([geometry.rows, blocks], ACL_DT_FLOAT8_E8M0, geometry.rows * blocks, 708),
        "weight": (
            [geometry.experts, geometry.inter, geometry.hidden],
            ACL_DT_FP4X2_E2M1,
            geometry.weight_bytes,
            709,
        ),
        "wgt_scale": (
            [geometry.experts, geometry.inter, blocks],
            ACL_DT_FLOAT8_E8M0,
            geometry.experts * geometry.inter * blocks,
            710,
        ),
        "gmm_out": ([geometry.rows, geometry.inter], ACL_DT_BF16, geometry.rows * geometry.inter * 2, None),
    }


@dataclass
class StageContext:
    """Device buffers, descriptors and tensor lists for one MoE layer."""

    geometry: LayerGeometry
    pointers: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)
    lists: dict = field(default_factory=dict)
    tuning: object = None


def build_stage_context(library: AclnnLibrary, runtime: AclRuntime, geometry: LayerGeometry, launch: bool) -> StageContext:
    context = StageContext(geometry=geometry)
    for name, (dims, code, nbytes, filler) in _buffer_plans(geometry).items():
        backing = nbytes if launch else min(nbytes, PLAN_BACKING_CAP)
        pointer = runtime.allocate(backing)
        if filler is not None:
            runtime.copy_in(pointer, deterministic_bytes(min(nbytes, backing), filler))
        context.pointers[name] = pointer
        context.tensors[name] = library.create_tensor_raw(pointer, dims, code)
    # Valid expert indices sit in the buffer until the gating kernel (launched
    # ahead of dispatch on the same stream) rewrites them.
    runtime.copy_in(
        context.pointers["expert_idx"],
        seeded_expert_indices(geometry.tokens, geometry.top_k, geometry.experts, 704),
    )
    list_names = ("expanded_x", "weight", "x_scale", "wgt_scale", "gmm_out")
    context.lists = {name: library.create_tensor_list([context.tensors[name]]) for name in list_names}
    context.tuning = library.create_int_array([geometry.top_k])
    return context


def run_stage(library, runtime, argtypes, plan_symbol, launch_symbol, args, do_launch: bool) -> str:
    plan_fn = ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(library.resolve(plan_symbol))
    workspace_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)
    try:
        status = plan_fn(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
    except ctypes.ArgumentError as exc:
        return f"plan FAILED conversion: {exc}"
    if status != ACL_SUCCESS:
        return f"plan FAILED aclnnStatus {status}"
    if not do_launch:
        return f"plan ok (workspace {workspace_size.value} B)"
    launch_fn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(
        library.resolve(launch_symbol)
    )
    workspace = runtime.allocate(workspace_size.value) if workspace_size.value else ctypes.c_void_p(0)
    status = launch_fn(workspace, workspace_size.value, executor, runtime.stream())
    if status != ACL_SUCCESS:
        return f"launch FAILED aclnnStatus {status}"
    return f"launch ok (workspace {workspace_size.value} B)"


def stage_arguments(context: StageContext) -> dict:
    tensors, lists = context.tensors, context.lists
    return {
        "GATING": (
            moe_gating_top_k_v2_plan_args(
                tensors["logits"], tensors["gate_y"], tensors["expert_idx"], bias=tensors["bias"], config=Dsv4GatingConfig()
            ),
            "aclnnMoeGatingTopKV2GetWorkspaceSize",
            "aclnnMoeGatingTopKV2",
        ),
        "DISPATCH": (
            moe_init_routing_v4_plan_args(
                tensors["hidden"],
                tensors["expert_idx"],
                tensors["expanded_x"],
                tensors["row_idx"],
                tensors["cumsum"],
                expanded_scale_out=tensors["expanded_scale"],
            ),
            "aclnnMoeInitRoutingV4GetWorkspaceSize",
            "aclnnMoeInitRoutingV4",
        ),
        "EXPERT_GEMM": (
            grouped_matmul_v5_plan_args(
                lists["expanded_x"].pointer,
                lists["weight"].pointer,
                tensors["cumsum"],
                lists["gmm_out"].pointer,
                scale_list=lists["wgt_scale"].pointer,
                per_token_scale_list=lists["x_scale"].pointer,
                tuning_config=context.tuning,
                config=GroupedMatmulV5Config(expected_tokens_per_expert=context.geometry.top_k),
            ),
            "aclnnGroupedMatmulV5GetWorkspaceSize",
            "aclnnGroupedMatmulV5",
        ),
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="DSV4 Flash single-layer MoE bring-up (aclnn V5, 950PR)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="dsv4")
    parser.add_argument("--tokens", type=int, default=None, help="override the preset token count")
    parser.add_argument("--stages", choices=("plan", "launch"), default="plan")
    parser.add_argument("--sync", action="store_true", help="wait for kernel completion and verify the cumsum")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    geometry = LayerGeometry(
        tokens=args.tokens or PRESETS[args.preset]["tokens"],
        hidden=PRESETS[args.preset]["hidden"],
        inter=PRESETS[args.preset]["inter"],
    )
    launch = args.stages == "launch"
    print(f"[moe-v5-bringup] preset={args.preset} geometry={geometry} stages={args.stages} sync={args.sync}")
    library = AclnnLibrary()
    if not library.loaded:
        print("[moe-v5-bringup] FAIL: libopapi.so not loaded")
        return 1
    missing = [symbol for symbol in (*MOE_V5_PLAN_SYMBOLS, *MOE_V5_LAUNCH_SYMBOLS) if library.resolve(symbol) is None]
    if missing:
        print(f"[moe-v5-bringup] FAIL: missing V5 MoE symbols: {missing}")
        return 1
    runtime = AclRuntime(library, args.device)
    statuses: dict[str, str] = {}
    try:
        context = build_stage_context(library, runtime, geometry, launch)
        for name in ("GATING", "DISPATCH", "EXPERT_GEMM"):
            stage_args, plan_symbol, launch_symbol = stage_arguments(context)[name]
            statuses[name] = run_stage(
                library, runtime, ARG_TYPES[name], plan_symbol, launch_symbol, stage_args, launch
            )
            print(f"[moe-v5-bringup]   {name:11s} {statuses[name]}")
            if "FAILED" in statuses[name]:
                break
        if args.sync and launch and all("FAILED" not in status for status in statuses.values()):
            runtime.synchronize()
            raw = runtime.copy_out(context.pointers["cumsum"], geometry.experts * 8)
            cumsum = struct.unpack(f"<{geometry.experts}q", raw)
            expected = geometry.rows
            verdict = "ok" if cumsum[-1] == expected else f"FAILED cumsum[-1]={cumsum[-1]} != {expected}"
            statuses["VERIFY"] = verdict
            print(f"[moe-v5-bringup]   VERIFY      cumsum[-1]=={expected}: {verdict}")
    finally:
        runtime.teardown()
    ok = bool(statuses) and all("FAILED" not in status for status in statuses.values())
    print(f"[moe-v5-bringup] {'MOE_V5_BRINGUP_OK' if ok else 'MOE_V5_BRINGUP_FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
