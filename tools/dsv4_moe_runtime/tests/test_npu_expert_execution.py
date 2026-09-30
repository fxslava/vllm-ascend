"""Hardware execution of ``dsv4_moe_expert`` on a physical Ascend 950PR.

Regression class: everything in this file needs real silicon and is skipped
everywhere else. The suite that does run without hardware
(``csrc/tests/host``) settles the arithmetic; the CPU interpreter settles the
vector math; neither can speak to what the part actually does with the kernel's
hand-written pipe synchronisation, its HBM traffic, or its behaviour across
concurrent streams. That is what these cases are for.

Two things make this file unusual, and both are deliberate:

* **It bypasses the vllm-ascend custom-op gate.** ``enable_custom_op()`` in
  ``vllm_ascend/utils.py`` returns ``False`` for A5/Ascend950 by design (see
  vllm-project/vllm-ascend#7157), so ``torch.ops._C_ascend.dsv4_moe_expert``
  does not exist on a 950PR no matter how the op was built. The kernel is
  reached instead through the custom op package directly -- a Torch extension
  registering the op if one is pointed at, otherwise the aclnn C entry points
  in ``libcust_opapi.so`` resolved ahead of CANN's own ``libopapi.so``.

* **The numeric oracle is imported from the golden generator by path**, not
  reimplemented. ``kernel_bringup`` is a script directory rather than a
  package, so it is loaded with importlib. One source of numeric truth: the
  same code produces the camodel goldens and the expectations here.

Environment:
    DSV4_CUST_OPAPI_PATH   libcust_opapi.so from the built op package
                           (default: <install>/vllm_ascend/_cann_ops_custom/...)
    DSV4_TORCH_BINDING     optional .so registering torch.ops._C_ascend
    DSV4_NPU_DEVICE        device string, default "npu:0"
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from ..hardware.aclnn_binding import AclnnLibrary
from .conftest import forbid_torch_allocations

# --------------------------------------------------------------------------
# Reduced bring-up geometry. Matches csrc/tests and the camodel goldens, and is
# what the host tiling function accepts: the production 4096/2048 shape needs
# 4 MiB of packed weight per projection against a 96 KiB single-load budget and
# is rejected by design.
# --------------------------------------------------------------------------
HIDDEN = 256
INTER = 128
FP4_BLOCK = 32
FP4_PER_BYTE = 2
SWIGLU_LIMIT = 10.0
MAX_ULP = 2  # the Gate B criterion, in bf16 ULPs

DEVICE = os.environ.get("DSV4_NPU_DEVICE", "npu:0")


# --------------------------------------------------------------------------
# Availability gates -- every one of these skips with a reason that names what
# is missing, so a red suite on a dev box is never mistaken for a kernel bug.
# --------------------------------------------------------------------------
def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    backend = getattr(torch, "npu", None)
    try:
        return bool(backend is not None and backend.is_available() and backend.device_count() > 0)
    except Exception:
        return False


requires_npu = pytest.mark.skipif(not _npu_available(), reason="no Ascend NPU visible to torch_npu")


def _golden_module() -> Any:
    """Import kernel_bringup/gen_golden.py by path; it is not a package."""
    path = Path(__file__).resolve().parents[1] / "kernel_bringup" / "gen_golden.py"
    if not path.exists():
        pytest.skip(f"golden generator not found at {path}")
    spec = importlib.util.spec_from_file_location("dsv4_gen_golden", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cust_opapi_candidates() -> list[Path]:
    override = os.environ.get("DSV4_CUST_OPAPI_PATH", "")
    if override:
        return [Path(override)]
    roots = [Path(__file__).resolve().parents[3] / "vllm_ascend" / "_cann_ops_custom"]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found += sorted(root.rglob("libcust_opapi.so"))
    return found


# --------------------------------------------------------------------------
# The invoker: the #7157 bypass, in the two forms the task allows.
# --------------------------------------------------------------------------
_PLAN_ARGTYPES = [ctypes.c_void_p] * 11 + [ctypes.c_void_p, ctypes.c_void_p]


class Dsv4ExpertAclnn:
    """Drives the kernel through the custom package's aclnn entry points.

    Mirrors the two-call CANN contract used by ``hardware/v5_ops.py``: a host
    ``GetWorkspaceSize`` plan call returns a single-shot executor and the
    workspace it needs, then the launch consumes both. The workspace is
    reserved once here and never grown inside a step -- growing it would be an
    allocation in the decode path, which the zero-allocation case below
    forbids outright.
    """

    PLAN = "aclnnDsv4MoeExpertGetWorkspaceSize"
    LAUNCH = "aclnnDsv4MoeExpert"

    def __init__(self, library: AclnnLibrary, device: str, workspace_bytes: int = 1 << 20):
        self._library = library
        self._device = device
        plan = library.resolve(self.PLAN)
        launch = library.resolve(self.LAUNCH)
        if plan is None or launch is None:
            raise RuntimeError(
                f"{self.PLAN}/{self.LAUNCH} not resolvable from the loaded custom package; "
                "build it with CUSTOM_OPS=dsv4_moe_expert bash csrc/build_aclnn.sh"
            )
        self._plan_fn = ctypes.CFUNCTYPE(ctypes.c_int, *_PLAN_ARGTYPES)(plan)
        launch.restype = ctypes.c_int
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        self._launch_fn = launch
        self._workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device=device)

    def __call__(self, tensors: dict[str, torch.Tensor], stream_pointer: int | None = None) -> None:
        handles = [self._library.create_tensor(tensors[name]) for name in _ARG_ORDER]
        try:
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p(0)
            status = self._plan_fn(
                *[h.pointer for h in handles],
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if status != 0:
                raise RuntimeError(f"{self.PLAN} failed with aclnnStatus {status}")
            if workspace_size.value > self._workspace.numel():
                raise RuntimeError(
                    f"plan wants {workspace_size.value} workspace bytes, only "
                    f"{self._workspace.numel()} reserved; re-create the invoker (AOT), "
                    "never grow inside a step"
                )
            pointer = (
                ctypes.c_void_p(self._workspace.data_ptr()) if workspace_size.value > 0 else ctypes.c_void_p(0)
            )
            status = self._launch_fn(pointer, workspace_size.value, executor, stream_pointer)
            if status != 0:
                raise RuntimeError(f"{self.LAUNCH} failed with aclnnStatus {status}")
        finally:
            for handle in handles:
                self._library.destroy_tensor(handle)


_ARG_ORDER = (
    "x", "w1", "w2", "w3", "w1_scale", "w2_scale", "w3_scale",
    "gate_out", "up_out", "activated", "down_out",
)


@pytest.fixture(scope="module")
def invoker() -> Any:
    """The #7157 bypass. Prefers a Torch binding, falls back to raw aclnn."""
    binding = os.environ.get("DSV4_TORCH_BINDING", "")
    if binding:
        torch.ops.load_library(binding)
        op = getattr(getattr(torch.ops, "_C_ascend", None), "dsv4_moe_expert", None)
        if op is None:
            pytest.skip(f"{binding} loaded but did not register _C_ascend.dsv4_moe_expert")

        def call_torch(tensors: dict[str, torch.Tensor], stream_pointer: int | None = None) -> None:
            op(*[tensors[name] for name in _ARG_ORDER])

        return call_torch

    library = AclnnLibrary()
    if not library.loaded:
        pytest.skip("libopapi.so not loaded; this case needs the CANN runtime on an Ascend host")
    candidates = _cust_opapi_candidates()
    if not candidates:
        pytest.skip(
            "libcust_opapi.so not found; set DSV4_CUST_OPAPI_PATH or build the op package "
            "(CUSTOM_OPS=dsv4_moe_expert bash csrc/build_aclnn.sh <root> ascend950pr_9579)"
        )
    library.load_custom_package(candidates[0])
    try:
        return Dsv4ExpertAclnn(library, DEVICE)
    except RuntimeError as exc:
        pytest.skip(str(exc))


# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------
def _make_inputs(golden: Any, seed: int, *, gate_bias: float | None = None) -> dict[str, torch.Tensor]:
    """Host-side operands, in the kernel's argument order.

    ``gate_bias`` drives every gate activation past +/-100 by pinning w1 to the
    extreme E2M1 code with a large block scale -- the extreme-bounds case.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x_bits = golden.float_to_bf16_bits(rng.standard_normal(HIDDEN, dtype=np.float32))
    packed_gate = (INTER, HIDDEN // FP4_PER_BYTE)
    packed_down = (HIDDEN, INTER // FP4_PER_BYTE)
    scale_gate = (INTER, HIDDEN // FP4_BLOCK)
    scale_down = (HIDDEN, INTER // FP4_BLOCK)

    def bytes_for(shape: tuple[int, int]) -> np.ndarray:
        return rng.integers(0, 256, size=shape, dtype=np.int64).astype(np.uint8)

    def scales_for(shape: tuple[int, int]) -> np.ndarray:
        return rng.integers(127 - 6, 127 + 7, size=shape, dtype=np.int64).astype(np.uint8)

    w1, w3, w2 = bytes_for(packed_gate), bytes_for(packed_gate), bytes_for(packed_down)
    w1s, w3s, w2s = scales_for(scale_gate), scales_for(scale_gate), scales_for(scale_down)

    if gate_bias is not None:
        # Code 0xF is -6.0; a 2^5 block scale drives |gate| past 100 for every
        # element, well outside the +/-10 clamp.
        w1[:] = 0xFF if gate_bias < 0 else 0x77  # 0x7 is +6.0
        w1s[:] = 127 + 5
        x_bits = golden.float_to_bf16_bits(np.full(HIDDEN, 2.0, dtype=np.float32))

    host = {
        "x": torch.from_numpy(x_bits.copy()).view(torch.bfloat16).reshape(1, HIDDEN),
        "w1": torch.from_numpy(w1), "w2": torch.from_numpy(w2), "w3": torch.from_numpy(w3),
        "w1_scale": torch.from_numpy(w1s), "w2_scale": torch.from_numpy(w2s),
        "w3_scale": torch.from_numpy(w3s),
    }
    return host


def _device_operands(host: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move inputs to HBM and allocate the four caller-owned outputs once."""
    tensors = {name: value.to(DEVICE) for name, value in host.items()}
    tensors["gate_out"] = torch.empty((1, INTER), dtype=torch.bfloat16, device=DEVICE)
    tensors["up_out"] = torch.empty((1, INTER), dtype=torch.bfloat16, device=DEVICE)
    tensors["activated"] = torch.empty((1, INTER), dtype=torch.bfloat16, device=DEVICE)
    tensors["down_out"] = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    return tensors


def _reference(golden: Any, host: dict[str, torch.Tensor]) -> dict[str, Any]:
    import numpy as np

    def u8(name: str) -> np.ndarray:
        return host[name].numpy()

    x_bits = host["x"].view(torch.uint16).reshape(-1).numpy()
    return golden.reference_expert(
        x_bits, u8("w1"), u8("w2"), u8("w3"), u8("w1_scale"), u8("w2_scale"), u8("w3_scale"),
        HIDDEN, INTER,
    )


def _max_ulp(actual: torch.Tensor, expected_bits: Any) -> int:
    import numpy as np

    got = actual.cpu().view(torch.uint16).reshape(-1).numpy().astype(np.int64)
    want = np.asarray(expected_bits, dtype=np.uint16).astype(np.int64)

    def ordinal(bits: np.ndarray) -> np.ndarray:
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7FFF), bits + 0x8000)

    return int(np.max(np.abs(ordinal(got) - ordinal(want)))) if got.size else 0


# --------------------------------------------------------------------------
# 1. Loader bypass
# --------------------------------------------------------------------------
@requires_npu
def test_stock_loader_really_is_disabled_on_a5() -> None:
    """The premise of this whole file: document that the gate is shut.

    If this ever fails, #7157 was resolved upstream and the bypass below can be
    retired in favour of torch.ops._C_ascend.dsv4_moe_expert.
    """
    from vllm_ascend.utils import enable_custom_op

    if enable_custom_op():
        pytest.skip("custom ops are enabled on this part; the #7157 bypass is no longer needed")
    assert getattr(getattr(torch.ops, "_C_ascend", None), "dsv4_moe_expert", None) is None


@requires_npu
def test_bypass_resolves_the_kernel(invoker: Any) -> None:
    """The bypass produced something callable -- the fixture skips if not."""
    assert callable(invoker)


# --------------------------------------------------------------------------
# 2. Numerical accuracy on real HBM
# --------------------------------------------------------------------------
@requires_npu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_golden_on_hardware(invoker: Any, seed: int) -> None:
    golden = _golden_module()
    host = _make_inputs(golden, seed)
    tensors = _device_operands(host)
    invoker(tensors)
    torch.npu.synchronize()

    want = _reference(golden, host)
    for name in ("gate_out", "up_out", "activated", "down_out"):
        ulp = _max_ulp(tensors[name], want[name])
        assert ulp <= MAX_ULP, f"{name}: FpDiff {ulp} ULP on hardware (limit {MAX_ULP})"


# --------------------------------------------------------------------------
# 3. Repeated-launch determinism
# --------------------------------------------------------------------------
@requires_npu
def test_repeated_launches_are_bit_identical(invoker: Any) -> None:
    """The synchronisation test, and the only one that can find a missing flag.

    The kernel compiles with --cce-auto-sync=off and stages in TBuf rather than
    TQue, so every cross-pipe dependency is hand-written. A missing flag does
    not usually corrupt a result outright -- it makes the result depend on how
    the pipes happened to interleave, which no single-run correctness check can
    see. 100 launches of identical input on one stream; any diff is a flag.
    """
    golden = _golden_module()
    host = _make_inputs(golden, seed=3)
    tensors = _device_operands(host)

    invoker(tensors)
    torch.npu.synchronize()
    first = {name: tensors[name].clone() for name in ("gate_out", "up_out", "activated", "down_out")}

    for iteration in range(1, 100):
        for name in first:
            tensors[name].fill_(0)
        invoker(tensors)
        torch.npu.synchronize()
        for name, baseline in first.items():
            assert torch.equal(tensors[name], baseline), (
                f"launch {iteration} differs from launch 0 on {name} with identical input: "
                "a cross-pipe flag is missing (--cce-auto-sync=off inserts none)"
            )


# --------------------------------------------------------------------------
# 4. Multi-expert concurrency
# --------------------------------------------------------------------------
@requires_npu
def test_six_concurrent_streams_keep_scratchpads_isolated(invoker: Any) -> None:
    """Six experts in flight at once, each with its own scratchpad rows.

    A kernel that leaked state through a shared buffer -- or left a hardware
    event set for the next launch to inherit -- shows up here and nowhere else:
    every stream's result must equal what that same input produces alone.
    """
    golden = _golden_module()
    num_streams = 6
    problems = [_make_inputs(golden, seed=10 + i) for i in range(num_streams)]
    operands = [_device_operands(p) for p in problems]

    # Serial reference pass first: what each problem produces undisturbed.
    serial: list[dict[str, torch.Tensor]] = []
    for tensors in operands:
        invoker(tensors)
        torch.npu.synchronize()
        serial.append({n: tensors[n].clone() for n in ("gate_out", "up_out", "activated", "down_out")})

    for tensors in operands:
        for name in ("gate_out", "up_out", "activated", "down_out"):
            tensors[name].fill_(0)

    streams = [torch.npu.Stream() for _ in range(num_streams)]
    for stream, tensors in zip(streams, operands):
        with torch.npu.stream(stream):
            invoker(tensors, stream_pointer=ctypes.c_void_p(stream.npu_stream).value)
    for stream in streams:
        stream.synchronize()
    torch.npu.synchronize()

    for index, (tensors, expected) in enumerate(zip(operands, serial)):
        for name, baseline in expected.items():
            assert torch.equal(tensors[name], baseline), (
                f"stream {index} produced a different {name} when run concurrently: "
                "scratchpad isolation is broken"
            )


# --------------------------------------------------------------------------
# 5. Extreme activation bounds
# --------------------------------------------------------------------------
@requires_npu
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_extreme_gate_is_clamped_and_never_traps(invoker: Any, sign: float) -> None:
    """Gate activations past +/-100 must hit the 10.0 clamp, not an inf.

    Without the DeepSeek-V4 clamp, exp(-gate) overflows for gate < -88 and the
    result is correct only because 1/(1+inf) is exactly 0 -- and the hardware
    raises FP status on every such element. With the clamp the exponential
    stays near exp(10) ~= 2.2e4 and nothing saturates.
    """
    golden = _golden_module()
    host = _make_inputs(golden, seed=4, gate_bias=sign)
    tensors = _device_operands(host)
    invoker(tensors)
    torch.npu.synchronize()

    gate = tensors["gate_out"].float().cpu()
    activated = tensors["activated"].float().cpu()
    down = tensors["down_out"].float().cpu()

    assert torch.isfinite(gate).all(), "gate_out is not finite"
    assert gate.abs().max().item() > 100.0, (
        "the extreme-bounds problem no longer drives the gate past +/-100; re-tune _make_inputs"
    )
    assert torch.isfinite(activated).all(), "activated went non-finite: the clamp did not hold"
    assert torch.isfinite(down).all(), "down_out went non-finite"

    # Past the clamp the activation is bounded by |swiglu_limit * up|, so it
    # cannot track the runaway gate.
    up_magnitude = tensors["up_out"].float().cpu().abs().max().item()
    assert activated.abs().max().item() <= SWIGLU_LIMIT * up_magnitude * 1.01

    want = _reference(golden, host)
    assert _max_ulp(tensors["activated"], want["activated"]) <= MAX_ULP


# --------------------------------------------------------------------------
# 6. Zero-allocation during the compute loop
# --------------------------------------------------------------------------
@requires_npu
def test_compute_loop_allocates_nothing(invoker: Any) -> None:
    """The ExpertKernelRunner contract: every tensor is caller-owned.

    Two independent checks, because either alone is easy to fool: torch's own
    allocators are booby-trapped for the duration (catching a Python-level
    allocation), and the NPU allocator's reserved byte count must not move
    (catching one made underneath it).
    """
    golden = _golden_module()
    host = _make_inputs(golden, seed=6)
    tensors = _device_operands(host)

    invoker(tensors)  # warm: first call may plan/reserve
    torch.npu.synchronize()

    before = torch.npu.memory_allocated(DEVICE)
    with forbid_torch_allocations():
        for _ in range(16):
            invoker(tensors)
    torch.npu.synchronize()
    after = torch.npu.memory_allocated(DEVICE)

    assert after == before, f"compute loop allocated {after - before} bytes on device"
