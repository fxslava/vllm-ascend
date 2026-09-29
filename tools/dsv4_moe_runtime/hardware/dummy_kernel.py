"""Safe on-device surrogate for the native DeepSeek-V4 FP4 expert kernel.

Why this exists: the production expert projection is a fused FP4 (block-32
E8M0 scales) GEMM + SwiGLU kernel that vllm-ascend does not ship yet. Driving
the real aclnn weight-quant family against synthetic slot bytes during the
static-runtime bring-up risks Ascend 950PR hardware faults long before any
numerics could be validated. :class:`DummyExpertKernelRunner` executes the
*same call signature, the same operand buffers and the same memory invariants*
as the future native kernel while only issuing verified, dtype-safe primitives
(``sum`` reductions with ``out=``, in-place ``copy_``/``mul_``/``sigmoid``):

* consumes the pre-sliced expert slot views (``w1``/``w2``/``w3`` packed FP4
  bytes plus their E8M0 scale views) and the pre-allocated scratchpad rows
  strictly in place -- the caller owns every tensor, nothing is returned;
* replaces each FP4 GEMM with a bounded byte-digest reduction (weights and
  scales folded into one ``int32`` row digest, masked into bf16-safe range)
  scaled by a sanitized, clamped reduction of the input row, so outputs are
  deterministic and can never diverge to NaN/Inf even on garbage activations;
* populates the intermediate SwiGLU buffers (``gate_out``/``up_out``/
  ``activated``) and writes the down projection into ``down_out``;
* optionally emulates the representative native-kernel latency
  (150-250 us per expert) with a ``perf_counter`` busy-wait -- ``time.sleep``
  cannot honor microsecond budgets on every host (Windows scheduler
  granularity is ~1-15 ms) and a busy-wait allocates nothing.

Reference-checkable: because the surrogate is plain torch arithmetic, tests
recompute the expected digests and compare element for element.
"""

from __future__ import annotations

import time
from typing import Protocol

import torch

from dsv4_moe_runtime.core.layout import FP4_BLOCK_SIZE, FP4_ELEMS_PER_BYTE, ExpertTensorLayout

MIN_REPRESENTATIVE_LATENCY_US = 150.0
MAX_REPRESENTATIVE_LATENCY_US = 250.0
DEFAULT_EXPERT_LATENCY_US = 200.0
# Row digests stay inside [0, 2**15) so the bf16 destination rows can never
# overflow regardless of the synthetic weight bytes staged into the slot.
DIGEST_MASK = 0x7FFF
# Activation rows are clamped into this window before the input reduction so
# uninitialized bf16 garbage (huge magnitudes, NaN, Inf) cannot leak through.
INPUT_CLAMP_MAGNITUDE = 64.0


class ExpertKernelRunner(Protocol):
    """Dependency seam for one expert's fused projection kernel.

    The signature mirrors exactly what the future native vllm-ascend kernel
    will receive per expert per decode step: the hidden-state row ``x``
    ``[1, hidden]`` bf16, the slot's packed FP4 weight views and E8M0 block
    scale views (``uint8``), and four pre-allocated scratchpad rows. The
    implementation must be write-only into caller storage and allocate
    nothing at call time.
    """

    def execute_expert(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_scale: torch.Tensor,
        gate_out: torch.Tensor,
        up_out: torch.Tensor,
        activated: torch.Tensor,
        down_out: torch.Tensor,
        label: str,
    ) -> None: ...


class DummyExpertKernelRunner:
    """Deterministic, allocation-free surrogate expert kernel.

    All working storage (row digests, sanitized activation row, scalar
    factors) is allocated once in ``__init__`` from the layout geometry;
    ``execute_expert`` performs only in-place / ``out=`` operations.
    """

    def __init__(
        self,
        layout: ExpertTensorLayout,
        device: str = "cpu",
        latency_us: float = 0.0,
    ):
        if latency_us < 0.0:
            raise ValueError(f"latency_us must be >= 0, got {latency_us}")
        self._layout = layout
        self._device = torch.device(device)
        self._latency_us = float(latency_us)
        hidden_spec = layout.spec_for("w1")
        self._hidden_size = hidden_spec.cols
        self._inter_size = hidden_spec.rows
        self._digest_inter = torch.empty(self._inter_size, dtype=torch.int32, device=self._device)
        self._digest_hidden = torch.empty(self._hidden_size, dtype=torch.int32, device=self._device)
        self._scale_digest_inter = torch.empty(self._inter_size, dtype=torch.int32, device=self._device)
        self._scale_digest_hidden = torch.empty(self._hidden_size, dtype=torch.int32, device=self._device)
        # bf16 so the in-place multiplies below never cross dtypes (torch
        # refuses f32 -> bf16 downcasts in in-place ops).
        self._x_factor = torch.empty(1, dtype=torch.bfloat16, device=self._device)
        self._x_sanitized = torch.empty(1, self._hidden_size, dtype=torch.bfloat16, device=self._device)
        self.executions = 0
        self.burned_us_total = 0.0

    @property
    def latency_us(self) -> float:
        return self._latency_us

    def execute_expert(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_scale: torch.Tensor,
        gate_out: torch.Tensor,
        up_out: torch.Tensor,
        activated: torch.Tensor,
        down_out: torch.Tensor,
        label: str,
    ) -> None:
        self._validate_operands(
            x, w1, w2, w3, w1_scale, w2_scale, w3_scale, gate_out, up_out, activated, down_out, label
        )

        # --- gate (w1) and up (w3) legs: digest "GEMM" into [1, I] rows. ---
        self._compute_input_factor(x)
        self._digest_rows(w1, w1_scale, self._digest_inter, self._scale_digest_inter)
        gate_out[0].copy_(self._digest_inter)
        gate_out.mul_(self._x_factor)
        self._digest_rows(w3, w3_scale, self._digest_inter, self._scale_digest_inter)
        up_out[0].copy_(self._digest_inter)
        up_out.mul_(self._x_factor)

        # --- SwiGLU surrogate into the intermediate buffer, in place. ---
        torch.sigmoid(gate_out[0], out=activated[0])
        activated[0].mul_(gate_out[0])
        activated[0].mul_(up_out[0])

        # --- down (w2) leg: digest into [1, H], scaled by the activated row. ---
        self._digest_rows(w2, w2_scale, self._digest_hidden, self._scale_digest_hidden)
        down_out[0].copy_(self._digest_hidden)
        torch.sum(activated, dim=1, dtype=torch.bfloat16, out=self._x_factor)
        down_out.mul_(self._x_factor)

        self._burn_latency()
        self.executions += 1

    # ------------------------------------------------------------- internals

    def _compute_input_factor(self, x: torch.Tensor) -> None:
        """Bounded bf16 row reduction of the input: NaN/Inf-free by construction.

        The magnitude (``abs``) keeps gate/up projections non-negative so the
        SwiGLU surrogate stays non-degenerate (``sigmoid(gate)`` in (0.5, 1)
        instead of saturating to 0 for negative factors).
        """
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, out=self._x_sanitized)
        self._x_sanitized.clamp_(-INPUT_CLAMP_MAGNITUDE, INPUT_CLAMP_MAGNITUDE)
        torch.sum(self._x_sanitized, dim=1, dtype=torch.bfloat16, out=self._x_factor)
        self._x_factor.abs_()

    def _digest_rows(
        self, weight: torch.Tensor, scale: torch.Tensor, digest: torch.Tensor, scale_digest: torch.Tensor
    ) -> None:
        """Row-wise uint8 digest (weights + scales) masked into bf16-safe range."""
        torch.sum(weight, dim=1, dtype=torch.int32, out=digest)
        torch.sum(scale, dim=1, dtype=torch.int32, out=scale_digest)
        digest.add_(scale_digest)
        digest.bitwise_and_(DIGEST_MASK)

    def _burn_latency(self) -> None:
        if self._latency_us <= 0.0:
            return
        budget_seconds = self._latency_us * 1e-6
        start = time.perf_counter()
        while time.perf_counter() - start < budget_seconds:
            pass  # busy-wait: microsecond-faithful, zero allocation
        self.burned_us_total += self._latency_us

    def _validate_operands(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_scale: torch.Tensor,
        gate_out: torch.Tensor,
        up_out: torch.Tensor,
        activated: torch.Tensor,
        down_out: torch.Tensor,
        label: str,
    ) -> None:
        hidden_half = self._hidden_size // FP4_ELEMS_PER_BYTE
        inter_half = self._inter_size // FP4_ELEMS_PER_BYTE
        hidden_blocks = self._hidden_size // FP4_BLOCK_SIZE
        inter_blocks = self._inter_size // FP4_BLOCK_SIZE
        expected = {
            "x": (x, (1, self._hidden_size), torch.bfloat16),
            "w1": (w1, (self._inter_size, hidden_half), torch.uint8),
            "w3": (w3, (self._inter_size, hidden_half), torch.uint8),
            "w2": (w2, (self._hidden_size, inter_half), torch.uint8),
            "w1_scale": (w1_scale, (self._inter_size, hidden_blocks), torch.uint8),
            "w3_scale": (w3_scale, (self._inter_size, hidden_blocks), torch.uint8),
            "w2_scale": (w2_scale, (self._hidden_size, inter_blocks), torch.uint8),
            "gate_out": (gate_out, (1, self._inter_size), torch.bfloat16),
            "up_out": (up_out, (1, self._inter_size), torch.bfloat16),
            "activated": (activated, (1, self._inter_size), torch.bfloat16),
            "down_out": (down_out, (1, self._hidden_size), torch.bfloat16),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{label}/{name}: expected shape {shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != dtype:
                raise ValueError(f"{label}/{name}: expected dtype {dtype}, got {tensor.dtype}")
            if tensor.device != self._device:
                raise ValueError(f"{label}/{name}: expected device {self._device}, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{label}/{name}: operand must be contiguous")
