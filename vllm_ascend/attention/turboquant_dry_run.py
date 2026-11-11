#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""TurboQuant dry-run attention backend: a PyTorch model of the Ascend C kernels.

Why this file exists
--------------------
The TurboQuant operators (``kernels/ops/turboquant/vllm_integration_layer`` of
the ascend-micro-harness) are still under development, so the vLLM-side
routing, cache carve and metadata plumbing have nothing to run against. This
backend is that something: every kernel in the ingestion and decode path has a
native PyTorch stand-in that runs on ``device="npu"`` and reproduces the
kernel's *arithmetic contract* -- the same rotation, the same power-of-two
scale, the same nibble packing, the same cache byte layout -- so a dry run
exercises the real quantisation error and the real cache geometry, just slowly.

It is deliberately **not** a monolith. Every operation goes through
:class:`TurboQuantOpsAdapter`, which dispatches it either to a ``_kernel_*``
entry point -- whose parameter list is the staged C launcher's, and which is a
stub until the ``TORCH_LIBRARY_FRAGMENT`` lands -- or to the ``_golden_*``
PyTorch reference built on :class:`TurboQuantEmulator`. Progressive replacement
is therefore one stage at a time, with the rest still golden, which is also how
a newly landed kernel gets differentially diffed against its own model.

Ingestion is two kernels, not one, and the phase decides which: a prefill step
hands whole page strips to the batch writer, a decode step hands one token per
request to the AIV scatter writer. The phase picks the *owner*, not the whole
dispatch -- a real prefill step is 220 tokens of one prompt or several requests
concatenated, which is not a whole number of the batch writer's 16-token
strips, so the prefill path plans its step across both writers
(``turboquant_plan_strip_ingestion``).

The arithmetic contract
-----------------------
``Pi`` is the rotation the kernels apply in UB, factored over a head of ``D``
channels in groups of ``TURBOQUANT_GROUP`` (32, the INT8 K0 of the arch-3510
fractal, so one Cube contraction is exactly one K fractal)::

    Pi = (H_{D/32} (x) I_32) . diag(pi_signs) . (I_{D/32} (x) H_32)

    Pi[c1*32 + c0][g*32 + i] = H8[c1][g] . pi_signs[g*32 + c0] . H32[c0][i]

with ``H`` the unnormalised Sylvester Hadamard, ``H[i][j] = (-1)^popcount(i&j)``.
``Pi`` is *not* an involution (unlike the ``D H D`` form the earlier 910B branch
used): ``Pi^T Pi = D I``, so the un-rotation is the transpose, which the kernels
reach by running the butterfly cascade in the opposite stage order ("Egress"
versus "Ingress" in ``routine_vector_rotate_butterfly.hpp``).

The ``1/sqrt(D)`` that makes the pair orthonormal is never applied to a
quantised value. On ingestion it is folded into the stored fp32 scale, and on
egress it is applied again by the un-rotation -- the two halves meeting as
``Pi^T Pi / D == I``.

Per rotated row of ``D`` channels, with ``2^e <= absmax < 2^(e+1)``::

    scale = 2^(e - 2) * 1/sqrt(D)      stored fp32, one per (plane, page, head, token)
    q     = min(rint_even(y * 2^(2 - e)), 7)     signed INT4, two per byte, low nibble first

Both powers of two come out of the fp32 exponent field, so ``y * 2^(2-e)`` is
exact and the single round-to-nearest-even is the only rounding in the path:
nibbles and scales are bit-comparable against the kernel rather than
tolerance-comparable. ``absmax == 0`` takes the explicit ``scale = 1.0, q = 0``
branch, because the exponent field of fp32 zero would otherwise borrow a scale
out of nothing. The clamp is from above only: ``|y * 2^(2-e)| <= 8`` and
round-to-nearest-even can reach ``+8``, which INT4 cannot hold, while ``-8`` is
representable and is a legitimate code.

What the dry run does *not* model
---------------------------------
* The decode runs in the **un-rotated** basis: K and V are gathered,
  dequantised, un-rotated and handed to ``npu_fused_infer_attention_score``.
  The kernel instead rotates the query and contracts against the rotated cache
  (``Pi q . Pi k == D q . k``). The quantisation error is identical; the
  floating-point summation order is not.
* Nothing here is a performance model. A dry run is for routing, shapes,
  metadata and accuracy, never for latency.

What the scale plane does *not* need reconciling
------------------------------------------------
An earlier revision of this note said the AIV scatter writer used a
"burst-padded per-token slot" for its scales, unlike the
``(2, num_blocks, num_kv_heads, block_size)`` fp32 plane modelled here. It does
not, and has not since the audited revision: both writers index

    ((kv * num_blocks + block) * num_kv_heads + head) * block_size + token

-- ``CalculateStripAddr`` in op_kernel/turboquant_reshape_batch.cpp, and the
``scale_base`` of ``WriteKeyPlane`` / ``WriteValuePlane`` in
op_kernel/turboquant_reshape_scatter.cpp, which ``DataCopyPad``s one float per
token into it because a single float is not a burst. That is exactly
:func:`turboquant_scale_cache_shape`, so the plane agrees element for element.

The burst-padded slot is real, but it belongs to the **fused decode's** NZ
cache: ``NzTiledScaleIndex`` is ``slot * NzScaleSlotFloats(H) + kv * H + head``,
slot-major and rounded up to a whole fp32 block
(op_kernel/common/turboquant_layout.h). It is one of the five axes on which that
cache differs from the writers', and nothing in the ingestion path has to
reconcile it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_GROUP,
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_PLANES,
    TURBOQUANT_VALUE_PLANE,
    turboquant_packed_cache_shape,
    turboquant_scale_cache_shape,
)
from vllm_ascend.attention.utils import notify_kv_cache_written

# ---------------------------------------------------------------------------
# The layout contract. Every constant here is shared with the kernels and with
# op_kernel/common/turboquant_layout.h; none of them is a tuning knob.
# ---------------------------------------------------------------------------

# Seed of the +-1 diagonal of Pi. Fixed so that every rank, every layer and
# every restart agree on the rotation: the cache is only ever read back by the
# same transform that wrote it.
TURBOQUANT_PI_SEED: Final[int] = 0x5F3759DF

# The sign vector comes from an explicit LCG (numerical-recipes constants)
# rather than from torch.Generator, so the host C++ reference produces the
# identical vector without either side depending on the other's RNG.
_LCG_MULTIPLIER: Final[int] = 1664525
_LCG_INCREMENT: Final[int] = 1013904223
_LCG_MODULUS_MASK: Final[int] = 0xFFFFFFFF
_LCG_SIGN_BIT: Final[int] = 16

# The signed INT4 grid. The packing itself (TURBOQUANT_PACK_FACTOR), the plane
# count and indices, and the group width live in turboquant_layout: the KV
# cache spec budgets a page from the same numbers this module writes into it.
TURBOQUANT_NIBBLE_MIN: Final[int] = -8
TURBOQUANT_NIBBLE_MAX: Final[int] = 7
TURBOQUANT_NIBBLE_MASK: Final[int] = 0xF
TURBOQUANT_NIBBLE_LEVELS: Final[int] = 16

# The strip the audited writers work in: `kTokens` in
# op_kernel/turboquant_reshape_batch.cpp and `kStripHeight` in
# op_kernel/turboquant_reshape_scatter.cpp, a compile-time constant of both
# schedules. It is the unit the channel-major value plane is channel-major
# *inside*, and the granularity of its read-modify-write.
TURBOQUANT_STRIP_TOKENS: Final[int] = 16

# scale = 2^(e - TURBOQUANT_HEADROOM_BITS): three bits of INT4 magnitude plus
# the sign, so the largest channel of a row lands on +-4.
TURBOQUANT_HEADROOM_BITS: Final[int] = 2

# The IEEE-754 binary32 fields the quantiser reads and writes directly. Both
# of its powers of two come out of the exponent field, which is how the
# kernel does it (`bits & 0x7F800000`) and what keeps the dry run's nibbles
# bit-comparable rather than tolerance-comparable.
_FP32_MANTISSA_BITS: Final[int] = 23
_FP32_EXPONENT_MASK: Final[int] = 0xFF
_FP32_EXPONENT_BIAS: Final[int] = 127
_FP32_MAX_BIASED_EXPONENT: Final[int] = 254

# Environment switches. ENABLE_TURBOQUANT routes the backend and the cache
# spec, and lives in turboquant_layout with the rest of the shared contract so
# that reading it does not import this module; the two below steer this module
# only.
TURBOQUANT_STAGES_ENV: Final[str] = "TURBOQUANT_KERNEL_STAGES"
TURBOQUANT_NAMESPACE_ENV: Final[str] = "TURBOQUANT_OPS_NAMESPACE"

# torch.ops namespace the kernels register into. The in-tree AscendC operators
# use "_C_ascend"; these have their own, defined by the TORCH_LIBRARY_FRAGMENT in
# csrc/attention/turboquant/torch_bindings.cpp, which registers the two cache
# writers and the fused decode. The remaining four entry points below are still
# unregistered, and the name stays configurable so a staging build can park the
# fragment elsewhere without a code change.
TURBOQUANT_DEFAULT_NAMESPACE: Final[str] = "turboquant_ops"


class TurboQuantKernelStage(str, Enum):
    """The stages a C++ kernel can take over, one at a time.

    ``RESHAPE_AND_CACHE`` and ``DECODE`` are the fused entry points: when either
    is switched on it subsumes the finer stages under it, because the fused
    kernel does rotation, quantisation and the cache write (or gather, dequant
    and the attention itself) inside one launch.
    """

    ROTATE = "rotate"
    QUANTIZE = "quantize"
    SCATTER = "scatter"
    GATHER = "gather"
    DEQUANTIZE = "dequantize"
    RESHAPE_AND_CACHE = "reshape_and_cache"
    DECODE = "decode"


class TurboQuantValueLayout(str, Enum):
    """The cache layout a page is in, which is a property of whoever wrote it.

    This is not a tuning knob and not a global the golden path has to be told
    about: :meth:`TurboQuantOpsAdapter.value_layout` derives it from which
    writer owns the page, and the readers below follow it. Two writers means two
    layouts, and the one the cache is in is the one the last writer used.

    ``TOKEN_MAJOR`` -- the golden model's own cache. Planes are
    ``(num_blocks, block_size, num_kv_heads, head_size / 2)`` and both of them
    are token-major inside a ``(page, head)`` panel: nibble index
    ``token * head_size + channel``. A token's nibbles are a contiguous run, so
    a per-token scatter writes them without touching another token's bytes,
    which is what lets PyTorch write this layout at all.

    ``CHANNEL_MAJOR`` -- the audited strip cache that both NPU writers write
    (``op_kernel/turboquant_reshape_batch.cpp``, ``..._scatter.cpp``). It
    differs from ``TOKEN_MAJOR`` in two ways at once, and both matter to a
    reader:

    * the page axis order is ``(num_blocks, num_kv_heads, block_size,
      head_size / 2)`` -- head before token -- because a ``(block, head)`` page
      has to be one contiguous MTE3 burst. The key plane is token-major inside
      it, nibble ``token * head_size + channel``
      (``turboquant_reshape_scatter.cpp``, the ``key_base`` burst).
    * the value plane is channel-major **inside a 16-token strip**, not across
      the page::

          nibble = (token // 16) * (16 * head_size)
                 + channel * 16
                 + (token % 16)

      which is ``StripWordOffset`` plus the within-strip index of
      ``turboquant_reshape_scatter.cpp``. A channel owns 16 consecutive nibbles,
      one per token, so a channel is 8 bytes and a strip is 2048 of them at
      ``head_size`` 256. Token ``t`` lands in the low nibble of its byte when
      ``t`` is even and the high nibble when it is odd, so a per-token write is
      a read-modify-write of the strip; the kernel splices it in with register
      bitwise masks and :meth:`TurboQuantOpsAdapter._write_slots` does it as a
      tensor read-modify-write of the touched strips.

    ``block_size`` is a multiple of 16 in this layout, and **both** writers take
    any such multiple. Each carries a strip index inside the page: the scatter
    writer as ``item.strip`` in ``StripWordOffset`` and ``item.token_in_block``
    in its key burst, the batch writer as

        block_idx      = slot0 / block_size
        strip_in_block = (slot0 % block_size) / 16

    in ``CalculateStripAddr`` (op_kernel/turboquant_reshape_batch.cpp), which
    offsets the packed base and both scale bases by it.

    The batch writer *used* to need ``block_size == 16`` -- its only address was
    a page base, so at block_size 64 all four strips of a page computed the same
    base and overwrote each other -- and an earlier revision of this docstring
    said so. It is worth stating what changed, because ``block_size == 16`` is
    **incompatible with the fused decode**, whose closed-form tile cursor needs a
    power-of-two multiple of 64 (``valid_launch``). Under the old constraint the
    three operators had no common page size at all; under the current writers
    ``block_size = 128`` -- an ordinary vLLM page, and what the decode's own
    end-to-end fixture uses -- satisfies all three at once.

    Caveat: no device test covers it. The writers' runner pins
    ``kBlockSize = 16`` (tests/npu/test_live_write.cpp), so ``strip_in_block`` is
    0 in every run that has ever been made and the batch writer's in-page offset
    is code without evidence. See csrc/attention/turboquant/torch_bindings.cpp,
    which checks the multiple rather than the equality.
    """

    TOKEN_MAJOR = "token_major"
    CHANNEL_MAJOR = "channel_major"


class TurboQuantIngestionPhase(str, Enum):
    """Which of the two cache writers owns a step's K/V.

    ``PREFILL`` -- more tokens than requests, so a request contributes a run of
    tokens. The staged batch writer takes whole physical pages and is the only
    writer that may assume ``slot_mapping`` is one page-aligned contiguous
    strip.

    ``DECODE``  -- one token per request. The staged AIV scatter writer takes
    arbitrary slots, one token at a time, and is the only one a per-token write
    is sound in.
    """

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class TurboQuantOpsConfig:
    """Which stages are kernel-backed, and where the kernels live."""

    ops_namespace: str = TURBOQUANT_DEFAULT_NAMESPACE
    kernel_stages: frozenset[TurboQuantKernelStage] = frozenset()
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.TOKEN_MAJOR

    @classmethod
    def from_env(cls) -> TurboQuantOpsConfig:
        """Read the configuration out of the environment.

        ``TURBOQUANT_KERNEL_STAGES`` is a comma-separated list of
        :class:`TurboQuantKernelStage` values, or ``all``, or ``none``. ``none``
        is the default: the kernels exist only on an ``ascend950`` build, and a
        run has to ask for them before any stage leaves the golden model.
        """
        raw = os.getenv(TURBOQUANT_STAGES_ENV, "none").strip().lower()
        if raw in ("", "none", "0", "off"):
            stages: frozenset[TurboQuantKernelStage] = frozenset()
        elif raw in ("all", "1", "on"):
            stages = frozenset(TurboQuantKernelStage)
        else:
            stages = cls._parse_stages(raw)
        return cls(
            ops_namespace=os.getenv(TURBOQUANT_NAMESPACE_ENV, TURBOQUANT_DEFAULT_NAMESPACE),
            kernel_stages=stages,
        )

    @staticmethod
    def _parse_stages(raw: str) -> frozenset[TurboQuantKernelStage]:
        known = {stage.value: stage for stage in TurboQuantKernelStage}
        selected: set[TurboQuantKernelStage] = set()
        for name in (part.strip() for part in raw.split(",")):
            if not name:
                continue
            stage = known.get(name)
            if stage is None:
                raise ValueError(
                    f"{TURBOQUANT_STAGES_ENV} names an unknown stage {name!r}; "
                    f"known stages are {sorted(known)} plus 'all' and 'none'"
                )
            selected.add(stage)
        return frozenset(selected)

    def enables(self, stage: TurboQuantKernelStage) -> bool:
        return stage in self.kernel_stages


# ---------------------------------------------------------------------------
# The PyTorch fallback math.
# ---------------------------------------------------------------------------


def _exp2_fp32(exponent: torch.Tensor) -> torch.Tensor:
    """``2^exponent`` as fp32, assembled from the exponent field.

    Exact for every exponent a normal binary32 can hold, and -- unlike
    ``torch.ldexp`` -- an integer shift and a bitcast, both of which have NPU
    kernels. The biased exponent is clamped into the normal range rather than
    allowed to wrap into a denormal, an infinity or a NaN. Only two kinds of row
    reach a clamp: a zero row, whose scale
    :meth:`TurboQuantEmulator.quantize` overwrites from its explicit branch, and
    a row whose ``absmax`` is itself denormal, which saturates at the smallest
    normal power of two instead of underflowing to zero -- a row that small
    quantises to zero codes either way.
    """
    biased = torch.clamp(exponent.to(torch.int32) + _FP32_EXPONENT_BIAS, 1, _FP32_MAX_BIASED_EXPONENT)
    return (biased << _FP32_MANTISSA_BITS).contiguous().view(torch.float32)


def _sylvester_hadamard(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``H[i][j] = (-1)^popcount(i & j)``, the unnormalised Sylvester Hadamard."""
    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    index = torch.arange(order, device=device, dtype=torch.int64)
    common = index.unsqueeze(1) & index.unsqueeze(0)
    parity = torch.zeros_like(common)
    for bit in range(order.bit_length()):
        parity = parity ^ ((common >> bit) & 1)
    return torch.where(parity.bool(), -1.0, 1.0).to(dtype)


def turboquant_pi_signs(head_size: int, device: torch.device) -> torch.Tensor:
    """The deterministic +-1 diagonal of ``Pi`` for ``head_size`` channels.

    An explicit LCG, seeded by ``TURBOQUANT_PI_SEED + head_size`` so that two
    different head sizes in one (hybrid) model do not share a sign prefix.
    """
    state = (TURBOQUANT_PI_SEED + head_size) & _LCG_MODULUS_MASK
    bits: list[float] = []
    for _ in range(head_size):
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & _LCG_MODULUS_MASK
        bits.append(1.0 if (state >> _LCG_SIGN_BIT) & 1 else -1.0)
    return torch.tensor(bits, dtype=torch.float32, device=device)


class TurboQuantEmulator:
    """Native-PyTorch model of the TurboQuant codec, in the kernels' arithmetic.

    One instance per ``(head_size, device)``; the backend keeps it on the layer,
    so the rotation matrix and the sign diagonal are built once per layer and
    then only read.
    """

    def __init__(
        self,
        head_size: int,
        device: torch.device,
        pi_signs: torch.Tensor | None = None,
        group: int = TURBOQUANT_GROUP,
    ) -> None:
        if head_size % group:
            raise ValueError(f"head_size {head_size} must be a multiple of the group width {group}")
        groups = head_size // group
        if groups & (groups - 1):
            raise ValueError(
                f"head_size / group = {groups} must be a power of two: the outer factor is H_{groups}"
            )
        self.head_size: Final[int] = head_size
        self.group: Final[int] = group
        self.groups: Final[int] = groups
        self.device: Final[torch.device] = device
        # 1/sqrt(D): folded into the stored scale on ingestion and applied again
        # by the un-rotation, the two halves composing Pi^T Pi / D == I.
        self.inv_sqrt_len: Final[float] = 1.0 / math.sqrt(float(head_size))
        self.pi_signs: torch.Tensor = (
            turboquant_pi_signs(head_size, device)
            if pi_signs is None
            else pi_signs.to(device=device, dtype=torch.float32)
        )
        if self.pi_signs.numel() != head_size:
            raise ValueError(f"pi_signs has {self.pi_signs.numel()} entries, expected {head_size}")
        self._rotation: torch.Tensor | None = None
        self._codec_tables: torch.Tensor | None = None

    # -- the rotation ------------------------------------------------------

    @property
    def rotation(self) -> torch.Tensor:
        """``Pi`` as a dense ``[D, D]`` fp32 matrix, built once and cached.

        A dense matmul is the point. The butterfly cascade is what the kernel
        does; modelling it with the same loop structure would let a wrong stage
        order hide behind a matching bug. ``D <= 256`` keeps the matrix at
        256 KiB.
        """
        if self._rotation is None:
            eye_group = torch.eye(self.group, device=self.device, dtype=torch.float32)
            eye_groups = torch.eye(self.groups, device=self.device, dtype=torch.float32)
            h_inner = _sylvester_hadamard(self.group, self.device, torch.float32)
            h_outer = _sylvester_hadamard(self.groups, self.device, torch.float32)
            inner = torch.kron(eye_groups, h_inner)  # I_{D/32} (x) H_32
            outer = torch.kron(h_outer, eye_group)  # H_{D/32} (x) I_32
            self._rotation = outer @ torch.diag(self.pi_signs) @ inner
        return self._rotation

    # -- the Cube's resident operands --------------------------------------
    #
    # Both are pure functions of the head size, and both used to be rebuilt on
    # every launch of every layer -- eleven device kernels and two allocations
    # per layer per decode step, inside what has to become a capturable region.
    # They are cached here rather than on the layer because the ops adapter has
    # no layer to read: `_ensure_statics` publishes these same objects as
    # `layer._turboquant_codec_tables` and `layer._turboquant_h16`, so a layer
    # dump and a launch cannot disagree about what rotated the cache.

    @property
    def codec_tables(self) -> torch.Tensor:
        """The Cube's resident BF16 rotation operand, built once per emulator."""
        if self._codec_tables is None:
            self._codec_tables = _build_codec_tables(self)
        return self._codec_tables

    @property
    def h16(self) -> torch.Tensor:
        """The Cube's resident ``[16, 16]`` fp16 sign matrix.

        Held in the module-level cache rather than on the instance: it depends
        on the device alone, so two head sizes of a hybrid model share one.
        """
        return _h16_operand(self.device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``y = Pi x`` over the last axis, unnormalised, in fp32.

        Unnormalised because the quantiser's exponent is extracted from exactly
        this value in the kernel; ``1/sqrt(D)`` is the stored scale's business.
        """
        return x.to(torch.float32) @ self.rotation.transpose(0, 1)

    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        """``x = Pi^T y / sqrt(D)``, the egress direction.

        Applied to a dequantised cache value -- which already carries one
        ``1/sqrt(D)`` from its stored scale -- this reconstructs the activation:
        ``Pi^T Pi / D == I``.
        """
        return (y.to(torch.float32) @ self.rotation) * self.inv_sqrt_len

    # -- the quantiser -----------------------------------------------------

    def quantize(self, rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic per-row INT4 quantisation of a rotated panel.

        ``rotated`` is ``[..., D]``. Returns ``(codes, scales)``: ``codes`` INT8
        in ``[-8, 7]`` shaped like the input, ``scales`` fp32 shaped ``[...]`` --
        one per row, which is one per ``(token, head)`` of a K or V plane.

        The exponent is read straight out of the fp32 exponent field, which is
        what the kernel does (``bits & 0x7F800000``): for a normal
        ``absmax = 1.f * 2^(biased - 127)``, ``2^e <= absmax < 2^(e+1)`` means
        ``e = biased - 127``. ``torch.frexp`` and ``torch.ldexp`` state the same
        thing and were what this used to read, but neither has an NPU kernel:
        they fall back to the CPU, a fallback is a device-to-host copy, and that
        copy is illegal while an ACL graph stream is capturing (``aclrtMemcpy``
        107030, "Stream during the capture stage is not supported"). An integer
        shift and a bitcast run on the device in every mode. Denormal rows are
        the one input the two spellings disagree on, and on those the field is
        the authority: the kernel reads the field too.
        """
        rotated_fp32 = rotated.to(torch.float32)
        absmax = rotated_fp32.abs().amax(dim=-1, keepdim=True).contiguous()
        zero_row = absmax == 0.0
        biased_exponent = (absmax.view(torch.int32) >> _FP32_MANTISSA_BITS) & _FP32_EXPONENT_MASK
        exponent = biased_exponent - _FP32_EXPONENT_BIAS
        ones = torch.ones_like(absmax)
        # 2^(e-2) and its exact reciprocal 2^(2-e): no divide, no reciprocal.
        pow_scale = _exp2_fp32(exponent - TURBOQUANT_HEADROOM_BITS)
        inv_scale = _exp2_fp32(TURBOQUANT_HEADROOM_BITS - exponent)
        # torch.round is round-half-to-even, matching the kernel's CAST_RINT.
        codes = torch.round(rotated_fp32 * inv_scale)
        codes = torch.clamp(codes, TURBOQUANT_NIBBLE_MIN, TURBOQUANT_NIBBLE_MAX)
        codes = torch.where(zero_row, torch.zeros_like(codes), codes)
        # A zero row stores exactly 1.0, without the 1/sqrt(D) fold -- the
        # kernel's explicit branch, kept bit-identical even though its codes are
        # zero and the value is therefore immaterial to the dequantisation.
        scales = torch.where(zero_row, ones, pow_scale * self.inv_sqrt_len)
        return codes.to(torch.int8), scales.squeeze(-1)

    def dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """``codes * scales`` in fp32: the rotated activation, ``Pi x / sqrt(D)``."""
        return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)

    # -- the packing -------------------------------------------------------

    @staticmethod
    def pack(codes: torch.Tensor) -> torch.Tensor:
        """Pack INT4 codes two to a byte along the last axis, low nibble first.

        ``[..., D]`` INT8 in ``[-8, 7]`` -> ``[..., D/2]`` INT8 bytes.
        """
        if codes.shape[-1] % TURBOQUANT_PACK_FACTOR:
            raise ValueError(f"cannot pack an odd channel count {codes.shape[-1]}")
        nibbles = codes.to(torch.int16) & TURBOQUANT_NIBBLE_MASK
        pairs = nibbles.reshape(*nibbles.shape[:-1], nibbles.shape[-1] // TURBOQUANT_PACK_FACTOR, 2)
        packed = pairs[..., 0] | (pairs[..., 1] << 4)
        return packed.to(torch.uint8).view(torch.int8)

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`pack`: ``[..., D/2]`` bytes -> ``[..., D]`` INT8 codes."""
        raw = packed.view(torch.uint8).to(torch.int16)
        low = raw & TURBOQUANT_NIBBLE_MASK
        high = (raw >> 4) & TURBOQUANT_NIBBLE_MASK
        pair = torch.stack((low, high), dim=-1)
        signed = torch.where(pair >= 8, pair - TURBOQUANT_NIBBLE_LEVELS, pair)
        return signed.reshape(*packed.shape[:-1], packed.shape[-1] * TURBOQUANT_PACK_FACTOR).to(torch.int8)

    # -- the composite round trip, for tests and offline reproduction -------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x -> (packed bytes, fp32 scales)``: rotate, quantise, pack."""
        codes, scales = self.quantize(self.rotate(x))
        return self.pack(codes), scales

    def decode(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """``(packed bytes, fp32 scales) -> x_hat``: unpack, dequantise, un-rotate."""
        return self.inverse_rotate(self.dequantize(self.unpack(packed), scales))


# ---------------------------------------------------------------------------
# The kernel entry points and their golden models.
# ---------------------------------------------------------------------------

# ``TurboQuantMode`` and the ``AscendType`` of the query and output are not
# here: the fused decode's binding fixes the mode to KV4_FP8, the only mode the
# operator is validated in, and reads the element type off the output dtype.
#
# ``TurboQuantOutputStage`` (op_kernel/common/turboquant_layout.h): where a
# rotation leaves its result. 0 is the rotated basis (ingress, ``Pi``), 1 is the
# model's own basis (egress, ``Pi^T / sqrt(D)``), 2 also divides by
# ``1 + exp(-gate)`` and is the only stage besides 1 that reads the gate buffer.
TURBOQUANT_STAGE_ROTATED_BASIS: Final[int] = 0
TURBOQUANT_STAGE_UNROTATED: Final[int] = 1
TURBOQUANT_STAGE_GATED: Final[int] = 2

# The order of the Cube's resident ``h16`` sign matrix, the product of four
# Sylvester butterfly factors (routines/routine_cube_hadamard16.hpp). It is fp16
# for both element types: its entries are +-1 and the kernel converts it on
# ingest where the Cube chunk operand is not fp16.
TURBOQUANT_H16_ORDER: Final[int] = 16

# The element type of the Cube's resident rotation operand (``tables`` for the
# batch writer, ``rot_tables`` for the fused decode). BF16, because
# ``stage_blocks_nz`` reads it as ``bfloat16_t`` and L1 reserves two bytes an
# entry; see :func:`_codec_tables` for why an INT8 blob is both the wrong
# arithmetic and a 16 KB over-read, and for why the decode cannot detect it.
TURBOQUANT_TABLE_DTYPE: Final[torch.dtype] = torch.bfloat16

# ---------------------------------------------------------------------------
# What the kernels are compile-time specialised to.
#
# These are not tuning knobs and not defaults: outside them a staged kernel is
# not a slow path, it is a wrong answer with every canary intact. All three
# operators pin the head size, and the two writers are additionally unrolled
# over a fixed KV head count (``kHeads`` in
# op_kernel/turboquant_reshape_batch.cpp drives ``kChunks = 2 * kHeads`` chunk
# phases by template argument).
#
# The binding TORCH_CHECKs every one of these, so a violating launch raises
# rather than corrupting -- but it raises from inside an attention forward, one
# step into a server that has already loaded weights and allocated a cache.
# :meth:`AscendTurboQuantDryRunBackendImpl.__init__` checks them again at layer
# construction so an impossible model is refused at load instead.
# ---------------------------------------------------------------------------

# ``kHeadSize`` in both writers, and ``dim != kHeadSize`` in the decode's
# ``valid_launch``. The golden path has no such constraint -- any multiple of
# TURBOQUANT_GROUP whose group count is a power of two works -- so this applies
# only to a kernel-backed stage.
TURBOQUANT_KERNEL_HEAD_SIZE: Final[int] = 256

# ``kHeads`` in op_kernel/turboquant_reshape_batch.cpp and the scatter writer's
# panel geometry: the two ingestion kernels are specialised to exactly this many
# KV heads.
TURBOQUANT_KERNEL_KV_HEADS: Final[int] = 4

# ``kv_heads > 4`` is refused by the decode's ``valid_launch``. The decode
# tolerates fewer, unlike the writers, which need exactly
# TURBOQUANT_KERNEL_KV_HEADS.
TURBOQUANT_DECODE_MAX_KV_HEADS: Final[int] = 4

# The query and output element type the kernels have instances for. The fused
# decode carries one pair of template instantiations,
# ``..._kv4fp8_bf16`` and ``..._raw_query_kv4fp8_bf16``, and its ``_impl``
# dispatch RETURNS WITHOUT LAUNCHING for any other ``AscendType`` -- which would
# leave ``output`` whatever the allocator gave it and report success. Both
# writers likewise take ``bfloat16_t`` activations only. The binding refuses a
# non-bf16 query outright for that reason; this constant is what the refusal at
# layer construction names.
TURBOQUANT_KERNEL_DTYPE: Final[torch.dtype] = torch.bfloat16

# The kernel entry points, as op_adapter/turboquant_torch_adpt.h declares them.
TURBOQUANT_OP_ROTATE: Final[str] = "npu_turboquant_rotate"
TURBOQUANT_OP_QUANTIZE_PACK: Final[str] = "npu_turboquant_quantize_pack"
TURBOQUANT_OP_UNPACK_DEQUANT: Final[str] = "npu_turboquant_unpack_dequant"
TURBOQUANT_OP_RESHAPE_AND_CACHE: Final[str] = "npu_turboquant_reshape_and_cache_a8"
TURBOQUANT_OP_RESHAPE_SCATTER: Final[str] = "npu_turboquant_reshape_scatter_aiv"
TURBOQUANT_OP_GATHER_PAGES: Final[str] = "npu_turboquant_gather_pages"
TURBOQUANT_OP_FUSED_DECODE: Final[str] = "npu_turboquant_fused_decode"
TURBOQUANT_OP_BUILD_DECODE_TILING: Final[str] = "npu_turboquant_build_decode_tiling"

# Which stage switch gates each entry point. The two ingestion entries are
# gated apart because they are two kernels: the batch writer owns whole
# physical pages and the AIV scatter writer owns one token.
#
# Three of these seven are registered by
# csrc/attention/turboquant/torch_bindings.cpp on an ascend950 build:
# RESHAPE_AND_CACHE, RESHAPE_SCATTER and FUSED_DECODE. The other four have no
# kernel in the audited subtree -- the rotation and the codec live inside the
# fused decode's prologue and inside both writers rather than as standalone
# operators, and nothing materialises a gathered page in production -- so their
# stages resolve to nothing and always run golden. Switching one on is not an
# error: it warns once and falls back, which is the same path a build without
# the extension takes.
TURBOQUANT_KERNEL_OPS: Final[dict[str, TurboQuantKernelStage]] = {
    TURBOQUANT_OP_ROTATE: TurboQuantKernelStage.ROTATE,
    TURBOQUANT_OP_QUANTIZE_PACK: TurboQuantKernelStage.QUANTIZE,
    TURBOQUANT_OP_UNPACK_DEQUANT: TurboQuantKernelStage.DEQUANTIZE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE: TurboQuantKernelStage.RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER: TurboQuantKernelStage.SCATTER,
    TURBOQUANT_OP_GATHER_PAGES: TurboQuantKernelStage.GATHER,
    TURBOQUANT_OP_FUSED_DECODE: TurboQuantKernelStage.DECODE,
}

# The fused decode's Split-K width. One split is the fused path: it emits its
# context straight out of the decode pass and never reaches
# ``AscendC::SyncAll<false>()``, which on dav-3510 only completes when every
# launched block is co-resident -- so a wider launch is also a grid capped at the
# physical block count. Widening this is a tuning decision that needs a device.
TURBOQUANT_DECODE_SPLITS: Final[int] = 1

# ``heads_per_task`` is clamped to the GQA group width, and the operator refuses
# more than this many (``valid_launch`` in include/turboquant_fused_decode_opt.hpp).
TURBOQUANT_DECODE_MAX_HEADS_PER_TASK: Final[int] = 16


# ---------------------------------------------------------------------------
# The decode's buffer geometry
# ---------------------------------------------------------------------------
#
# Mirrors op_host/turboquant_tiling.h and op_kernel/common/turboquant_layout.h.
# The authority is C++ -- the device reads the descriptor words by offset and
# nothing in the toolchain diagnoses two sides disagreeing about which word is
# which -- so none of this computes a descriptor. It computes *sizes*, which is
# the one thing the caller has to know now that the buffers are its own, and the
# op re-derives every size and ``TORCH_CHECK``s what it was handed. These
# constants exist so a Python caller can size a buffer without a C++ call; the
# op refusing an undersized buffer is what makes a drift here an error rather
# than an out-of-bounds read.

# ``sizeof(TurboQuantDecodeTilingData) / 4`` and
# ``sizeof(TurboQuantDecodeDispatch) / 4``.
TURBOQUANT_DECODE_TILING_WORDS: Final[int] = 16
TURBOQUANT_DECODE_DISPATCH_WORDS: Final[int] = 8

# ``kDecodeMaxGrid`` and ``kDecodeMaxSplits`` in
# csrc/attention/turboquant/torch_bindings.cpp.
TURBOQUANT_DECODE_MAX_GRID: Final[int] = 32
TURBOQUANT_DECODE_MAX_SPLITS: Final[int] = 8

# ``kPartialTail = 2 * kFp32PerBlock``: the running max and running sum lanes a
# Split-K partial carries past its ``head_size`` accumulator.
TURBOQUANT_PARTIAL_TAIL: Final[int] = 16
# The depth of the builder's host staging ring. Two is the minimum that makes
# overwriting a slot safe while its own upload is still in flight: by the time
# the host returns to a slot, the whole of the following step has been enqueued
# behind that slot's copy on an in-order stream. Named rather than inline so the
# capture-contract gate reads the same number production uses.
TURBOQUANT_STAGING_SLOTS: Final[int] = 2


def turboquant_decode_heads_per_task(num_heads: int, num_kv_heads: int) -> int:
    """``heads_per_task`` for one launch, derived in exactly one place.

    The op clamps this again and the tiling builder clamps it a third time
    (``build_decode_tiling``), so the three agreeing is what makes the
    descriptor's ``chunks_per_group`` describe the work the launch actually
    splits into. Deriving it twice from the same formula is how they agree.
    """
    group_heads = num_heads // num_kv_heads if num_kv_heads else num_heads
    return min(TURBOQUANT_DECODE_MAX_HEADS_PER_TASK, max(group_heads, 1))


def turboquant_decode_tiling_words(tokens: int, num_splits: int, grid: int) -> int:
    """``decode_tiling_words(tokens, splits, grid)``: descriptors then dispatch."""
    return tokens * num_splits * TURBOQUANT_DECODE_TILING_WORDS + grid * TURBOQUANT_DECODE_DISPATCH_WORDS


def turboquant_tiling_upper_bound_words(max_num_reqs: int, max_splits: int) -> int:
    """The largest tiling buffer any launch of up to ``max_num_reqs`` can need.

    The exact size depends on ``grid``, which is the op's own derivation from
    the task count. Sizing to the bound instead removes the need for Python to
    reproduce that derivation at all: ``grid`` is capped at
    :data:`TURBOQUANT_DECODE_MAX_GRID`, so the dispatch half has a constant
    bound and the descriptor half is linear in the request count. The cost is
    ``kDecodeMaxGrid * 8`` words -- 1 KiB -- of slack, which buys one buffer for
    every capture size instead of one per size.
    """
    return turboquant_decode_tiling_words(max_num_reqs, max_splits, TURBOQUANT_DECODE_MAX_GRID)


def turboquant_decode_workspace_floats(tokens: int, num_heads: int, head_size: int, num_splits: int) -> int:
    """The Split-K partial workspace, as ``TurboQuantPartialReducer`` addresses it."""
    return tokens * num_heads * num_splits * (head_size + TURBOQUANT_PARTIAL_TAIL)


def turboquant_decode_query_rot_floats(tokens: int, num_heads: int, head_size: int) -> int:
    """The fp32 rotated query image the basis prologue writes."""
    return tokens * num_heads * head_size


# ---------------------------------------------------------------------------
# Loading the translation unit that carries the registrations.
# ---------------------------------------------------------------------------

# The four the fragment in csrc/attention/turboquant/torch_bindings.cpp
# registers. The other four entry points above have no kernel behind them yet,
# so their absence is not worth reporting.
#
# BUILD_DECODE_TILING is not in TURBOQUANT_KERNEL_OPS and has no stage switch:
# it is not an alternative to a golden model, it is the fused decode's own
# descriptor builder. It is registered with the decode and absent with it, so
# the metadata builder treats its absence as "there is no decode to describe"
# rather than as a fallback.
TURBOQUANT_REGISTERED_OPS: Final[tuple[str, ...]] = (
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
)

# Tri-state so the import is attempted once per process: True it landed, False
# it did not, None not tried yet.
_TURBOQUANT_EXTENSION_LOADED: bool | None = None


def _is_op_registered(namespace: str, op_name: str) -> bool:
    """Whether ``namespace::op_name`` has a kernel in the dispatcher.

    Deliberately NOT ``getattr(torch.ops, namespace, None) is not None``:
    ``torch.ops.__getattr__`` manufactures an ``_OpNamespace`` for whatever name
    it is handed and never raises, so that test passes for a namespace nothing
    ever registered. Worse, the manufactured object is cached, so the namespace
    then shows up in ``dir(torch.ops)`` -- which :meth:`_resolve_kernel` causes
    on its first call -- and a diagnostic built on either would claim these
    kernels are present on a build that has none of them.

    ``torch._C._dispatch_has_kernel`` asks the dispatcher, which is what a
    ``TORCH_LIBRARY_FRAGMENT`` actually writes into. It is private, hence the
    fallback onto the same resolution ``_resolve_kernel`` performs -- where
    ``_OpNamespace.__getattr__`` does raise for an operator it cannot find.
    """
    try:
        return bool(torch._C._dispatch_has_kernel(f"{namespace}::{op_name}"))
    except (AttributeError, RuntimeError):
        try:
            getattr(getattr(torch.ops, namespace), op_name)
        except (AttributeError, RuntimeError):
            return False
        return True


def load_turboquant_extension() -> bool:
    """Import the extension whose static initialisers define the ops.

    ``TORCH_LIBRARY_FRAGMENT(turboquant_ops, ops)`` in
    ``csrc/attention/turboquant/torch_bindings.cpp`` runs when
    ``vllm_ascend.vllm_ascend_C`` is imported and not before, so
    ``torch.ops.turboquant_ops`` does not exist until something imports that
    module. CMakeLists.txt adds the binding TU only on an ``ascend950`` build,
    so on every other part the namespace is absent by construction and that
    absence is the intended fallback signal -- there is nothing to load and this
    returns False without complaint.

    On an ascend950 build the module has to be imported from here, because the
    usual loader never will. :func:`vllm_ascend.utils.enable_custom_op` is that
    loader, and on this part it returns False at its capability check *without
    importing anything*: the A5 hardware profile deliberately withholds
    ``HardwareCapability.RUNTIME_CUSTOM_OPS`` while the other AscendC operators
    are only partially available there (vllm-ascend issue #7157). The TurboQuant
    ops are gated by ``TURBOQUANT_KERNEL_STAGES`` instead, so they do not want
    that switch -- and since every consumer of the ops it does disable asks
    ``enable_custom_op()`` rather than ``torch.ops``, registering their schemas
    as a side effect of this import leaves them exactly as unreachable as
    before. Registering a schema is not executing it.

    ``bootstrap_custom_op_env()`` and then the explicit import is the pattern the
    A5 operator tests already use -- see
    ``tests/e2e/nightly/single_node/ops/singlecard_ops/test_scatter_nd_update_sk.py``
    and ``test_add_rms_norm_bias_a5.py``, whose comments name this same
    capability gap.

    This is deliberately NOT done from ``vllm_ascend/__init__.py``. That module
    is imported in every process on every part, and importing the extension at
    interpreter start is the one thing ``enable_custom_op``'s laziness exists to
    prevent: it initialises CANN's RTS component while
    ``ASCEND_RT_VISIBLE_DEVICES`` can still be changed before
    ``torch.npu.set_device()``. Doing it here keeps the cost on the runs that
    asked for a kernel stage.
    """
    global _TURBOQUANT_EXTENSION_LOADED
    if _TURBOQUANT_EXTENSION_LOADED is not None:
        return _TURBOQUANT_EXTENSION_LOADED

    # Two steps, two handlers: the bootstrap only prepends the vendor op_api
    # search paths, so losing it must not stop the import that actually carries
    # the registrations. Collapsing these would let an unrelated failure here
    # look exactly like a build with no kernels in it.
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: PLC0415

        if not torch.compiler.is_compiling():
            bootstrap_custom_op_env()
    except ImportError as exc:
        logger.debug("TurboQuant: bootstrap_custom_op_env is unavailable (%s); importing anyway.", exc)

    try:
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

        _TURBOQUANT_EXTENSION_LOADED = True
    except (ImportError, OSError) as exc:
        # Not a failure in itself: a build for any other SoC has no binding TU
        # and a source checkout has no extension at all. OSError as well as
        # ImportError because a built extension that cannot dlopen its CANN
        # dependencies raises that instead. The caller reports the consequence
        # -- the missing operators -- once, with the stages that wanted them.
        _TURBOQUANT_EXTENSION_LOADED = False
        logger.debug("TurboQuant: vllm_ascend.vllm_ascend_C did not import (%s).", exc)
    return _TURBOQUANT_EXTENSION_LOADED


class TurboQuantOpsAdapter:
    """The seam between the TurboQuant NPU kernels and their PyTorch reference.

    Every operation is a two-branch method: :meth:`_has_kernel` decides, and the
    branches are a ``_kernel_*`` entry point whose parameter list is the staged
    C launcher's, and a ``_golden_*`` PyTorch implementation that is the only
    path a dry run takes today. Adopting a kernel is one stage flip in
    :class:`TurboQuantOpsConfig`; bisecting a suspect one is the flip back.

    The entry points are the declaration surface of
    ``kernels/ops/turboquant/vllm_integration_layer/op_adapter`` in the
    ascend-micro-harness. Three of them are registered on an ``ascend950`` build
    -- ``npu_turboquant_reshape_and_cache_a8``,
    ``npu_turboquant_reshape_scatter_aiv`` and ``npu_turboquant_fused_decode``,
    by the fragment in ``csrc/attention/turboquant/torch_bindings.cpp``, whose
    schemas these ``_kernel_*`` parameter lists match argument for argument. The
    other four (``rotate``, ``quantize_pack``, ``unpack_dequant``,
    ``gather_pages``) have no kernel behind them yet, so their bodies are the
    call they will make and they always fall back.
    """

    def __init__(
        self,
        config: TurboQuantOpsConfig | None = None,
    ) -> None:
        self.config: TurboQuantOpsConfig = config if config is not None else TurboQuantOpsConfig.from_env()
        self._emulators: dict[tuple[int, str], TurboQuantEmulator] = {}
        self._missing_kernels: set[str] = set()
        if self.config.kernel_stages:
            # Only a run that asked for a kernel stage pays for the import, and
            # only such a run is entitled to complain that it is missing.
            load_turboquant_extension()
            self._report_namespace()

    def _report_namespace(self) -> None:
        """Say once which of the registered ops arrived, and what to look at if not.

        :meth:`_resolve_kernel` already warns per operation as each is first
        reached, but it cannot distinguish a build that registered nothing from
        one whose kernels are there and simply lacks the stage being asked for.
        This separates them up front, while the distinction is still cheap to
        act on: nothing registered points at the extension or at the ascend950
        build of it, a partial set points at the fragment.
        """
        stages = sorted(stage.value for stage in self.config.kernel_stages)
        namespace = self.config.ops_namespace
        missing = [op for op in TURBOQUANT_REGISTERED_OPS if not _is_op_registered(namespace, op)]
        if not missing:
            logger.debug(
                "TurboQuant: torch.ops.%s carries %s for stages %s.",
                namespace,
                list(TURBOQUANT_REGISTERED_OPS),
                stages,
            )
            return
        logger.warning(
            "TurboQuant stages %s are switched on but torch.ops.%s is missing %s, which will "
            "run the PyTorch golden model instead. These registrations are static initialisers "
            "of vllm_ascend.vllm_ascend_C (import of it %s here), and CMakeLists.txt compiles "
            "csrc/attention/turboquant/torch_bindings.cpp only when SOC_VERSION matches "
            "ascend950 -- so on any other part they are absent by construction. Operators the "
            "dispatcher does have under this namespace: %s",
            stages,
            namespace,
            missing,
            "succeeded" if _TURBOQUANT_EXTENSION_LOADED else "failed",
            [op for op in TURBOQUANT_REGISTERED_OPS if op not in missing] or "none",
        )

    # -- plumbing ----------------------------------------------------------

    def emulator_for(self, head_size: int, device: torch.device) -> TurboQuantEmulator:
        """The emulator for one ``(head_size, device)``, built once.

        Keyed rather than per-layer because the rotation is a property of the
        head size alone: every layer of a model shares one matrix, and a hybrid
        model with two head sizes gets one each.
        """
        key = (head_size, str(device))
        emulator = self._emulators.get(key)
        if emulator is None:
            emulator = TurboQuantEmulator(head_size, device)
            self._emulators[key] = emulator
        return emulator

    def _resolve_kernel(self, op_name: str) -> Callable[..., Any] | None:
        """The registered entry point for ``op_name``, or ``None`` to run golden.

        A stage switched on whose kernel is not registered is a configuration
        error the caller can survive, so it warns once and falls back: a dry run
        that dies on a missing kernel tells you nothing you did not know.
        """
        if not self.config.enables(TURBOQUANT_KERNEL_OPS[op_name]):
            return None
        namespace = getattr(torch.ops, self.config.ops_namespace, None)
        kernel = getattr(namespace, op_name, None) if namespace is not None else None
        if kernel is None:
            if op_name not in self._missing_kernels:
                self._missing_kernels.add(op_name)
                logger.warning(
                    "TurboQuant stage %s is switched on but torch.ops.%s.%s is not registered; "
                    "falling back to the PyTorch golden model.",
                    TURBOQUANT_KERNEL_OPS[op_name].value,
                    self.config.ops_namespace,
                    op_name,
                )
            return None
        return kernel

    def _has_kernel(self, op_name: str) -> bool:
        """Whether ``op_name`` runs on the NPU for this configuration."""
        return self._resolve_kernel(op_name) is not None

    def _require_kernel(self, op_name: str) -> Callable[..., Any]:
        """The entry point, for a ``_kernel_*`` body :meth:`_has_kernel` cleared."""
        kernel = self._resolve_kernel(op_name)
        if kernel is None:
            raise RuntimeError(f"torch.ops.{self.config.ops_namespace}.{op_name} is not registered")
        return kernel

    def value_layout(self) -> TurboQuantValueLayout:
        """The layout the cache is in, which is whichever writer owns the page.

        Either NPU writer puts the cache in the audited strip layout
        (:class:`TurboQuantValueLayout`), so this is derived rather than
        configured: a run that switches an ingestion stage on does not also have
        to be told what that stage writes, and the golden reader follows without
        a second switch to keep in step.

        ``config.value_layout`` is consulted only when both writers are golden,
        which is how a dry run can still ask the golden writer to refuse the
        channel-major read-modify-write it cannot do.
        """
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE) or self._has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            return TurboQuantValueLayout.CHANNEL_MAJOR
        return self.config.value_layout

    # -- rotation ----------------------------------------------------------

    def rotate(self, x: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """Ingress rotation ``y = Pi x`` over the last axis: ``[..., D] -> [..., D]`` fp32."""
        if self._has_kernel(TURBOQUANT_OP_ROTATE):
            return self._kernel_rotate(
                x=x,
                pi_signs=emulator.pi_signs,
                rot_tables=_codec_tables(emulator),
                h16=_h16_operand(emulator.device),
                output_stage=TURBOQUANT_STAGE_ROTATED_BASIS,
                inv_sqrt_len=1.0,
            )
        return self._golden_rotate(x, emulator)

    def inverse_rotate(self, y: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """Egress rotation ``x = Pi^T y / sqrt(D)``: ``[..., D] -> [..., D]`` fp32."""
        if self._has_kernel(TURBOQUANT_OP_ROTATE):
            return self._kernel_rotate(
                x=y,
                pi_signs=emulator.pi_signs,
                rot_tables=_codec_tables(emulator),
                h16=_h16_operand(emulator.device),
                output_stage=TURBOQUANT_STAGE_UNROTATED,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
        return self._golden_inverse_rotate(y, emulator)

    def _kernel_rotate(
        self,
        x: torch.Tensor,
        pi_signs: torch.Tensor,
        rot_tables: torch.Tensor,
        h16: torch.Tensor,
        output_stage: int,
        inv_sqrt_len: float,
    ) -> torch.Tensor:
        """The butterfly cascade: ``[..., D]`` in, ``[..., D]`` fp32 out.

        No standalone C launcher stages the rotation. It is the fused decode's
        prologue -- ``RawQueryBuffers::{pi_signs, rot_tables, h16}`` plus
        ``Launch::output_stage``, over
        op_kernel/vector/routine_vector_rotate_butterfly.hpp -- and it is inside
        both cache writers. ``output_stage`` picks the stage order, which is what
        makes the pair ``Pi`` and ``Pi^T`` rather than an involution;
        ``inv_sqrt_len`` is applied by the egress direction only.
        """
        return self._require_kernel(TURBOQUANT_OP_ROTATE)(x, pi_signs, rot_tables, h16, output_stage, inv_sqrt_len)

    def _golden_rotate(self, x: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """``y = Pi x``, unnormalised fp32: the quantiser reads this exponent."""
        return emulator.rotate(x)

    def _golden_inverse_rotate(self, y: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """``x = Pi^T y / sqrt(D)``, fp32: with the stored scale's fold, ``Pi^T Pi / D == I``."""
        return emulator.inverse_rotate(y)

    # -- quantisation and packing ------------------------------------------

    def quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[..., D]`` rotated fp32 -> (``[..., D/2]`` INT8 bytes, ``[...]`` fp32 scales)."""
        if self._has_kernel(TURBOQUANT_OP_QUANTIZE_PACK):
            return self._kernel_quantize_pack(rotated=rotated, inv_sqrt_len=emulator.inv_sqrt_len)
        return self._golden_quantize_pack(rotated, emulator)

    def unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        """(``[..., D/2]`` bytes, ``[...]`` fp32 scales) -> ``[..., D]`` fp32, still rotated."""
        if self._has_kernel(TURBOQUANT_OP_UNPACK_DEQUANT):
            return self._kernel_unpack_dequant(packed=packed, scales=scales)
        return self._golden_unpack_dequant(packed, scales, emulator)

    def _kernel_quantize_pack(self, rotated: torch.Tensor, inv_sqrt_len: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row INT4 quantisation and nibble packing of a rotated panel.

        No standalone C launcher: the codec is inside both cache writers
        (op_kernel/common/turboquant_codec_mx.h). ``inv_sqrt_len`` is the fold the
        stored fp32 scale carries, so the egress un-rotation does not repeat it.
        """
        return self._require_kernel(TURBOQUANT_OP_QUANTIZE_PACK)(rotated, inv_sqrt_len)

    def _kernel_unpack_dequant(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Nibble unpack and dequantisation: ``[..., D/2]`` bytes -> ``[..., D]`` fp32.

        No standalone C launcher: the decoder is
        op_kernel/common/routine_turboquant_dequant4.hpp, which the fused decode
        reads through UB rather than materialising.
        """
        return self._require_kernel(TURBOQUANT_OP_UNPACK_DEQUANT)(packed, scales)

    def _golden_quantize_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[..., D]`` fp32 -> (``[..., D/2]`` INT8, ``[...]`` fp32), one scale per row."""
        codes, scales = emulator.quantize(rotated)
        return emulator.pack(codes), scales

    def _golden_unpack_dequant(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        """(``[..., D/2]`` INT8, ``[...]`` fp32) -> ``[..., D]`` fp32, in the rotated basis."""
        return emulator.dequantize(emulator.unpack(packed), scales)

    # -- ingestion: the prefill batch writer -------------------------------

    def reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """Ingest a contiguous batch of ``[T, H, D]`` K/V into whole cache pages.

        The prefill phase. ``slot_mapping`` is ``[T]`` int32 of
        ``page * block_size + token``. ``T`` is whatever the scheduler handed
        this step -- 220 tokens of one prompt, or several requests concatenated
        -- while the batch writer only takes whole 16-token strips whose slots
        are strip-aligned and consecutive. Reconciling the two is
        :meth:`_kernel_ingest_prefill`, which plans the step and dispatches it
        across both audited writers. The golden model needs no plan: it writes
        the slots it is given, one token at a time.
        """
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE):
            self._kernel_ingest_prefill(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
            )
            return
        self._golden_reshape_and_cache_prefill(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    def _kernel_ingest_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """Ingest an arbitrary token count through the two audited writers.

        The batch writer is specialised to whole 16-token strips whose slots are
        strip-aligned and consecutive, and refuses any other token count
        outright::

            npu_turboquant_reshape_and_cache_a8: num_tokens (220) must be a
            whole number of 16-token strips

        A real prompt is 220 tokens, and a real batch is several requests whose
        boundaries fall inside a strip, so the step is planned
        (:func:`turboquant_plan_strip_ingestion`) and dispatched piecewise:
        whole strips to the batch writer, a strip-aligned partial strip to the
        batch writer with its last token replicated into the slots the request
        already owns past its own length, and anything else to the scatter
        writer, whose read-modify-write is the only thing allowed to touch a
        strip that already holds an earlier step's tokens.

        Every token this step was given lands exactly once, through the same
        codec, at the slot it was given -- so a step that already conformed is
        byte-for-byte what it was before this split existed, and takes its own
        tensors through untouched rather than a slice of them.

        The plan needs the slot values on the host, so this costs one
        device-to-host copy of an int32 ``[T]`` and one Python pass over it per
        ingestion call -- the pass itself measures 0.3 ms at 2048 tokens and
        1.1 ms at 8192, per layer. The copy is the one
        ``VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS``
        pays in csrc/attention/turboquant/torch_bindings.cpp, where that
        validation is opt-in precisely because it is a synchronisation. This
        backend is for metadata and accuracy rather than latency, and the
        alternative is a writer that silently lands a strip in the wrong page.
        """
        plan = turboquant_plan_strip_ingestion(slot_mapping.tolist(), int(key_cache.shape[1]))
        tables = _codec_tables(emulator)

        if plan.is_whole_batch:
            self._kernel_reshape_and_cache_a8(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
                tables=tables,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
            return

        for run in plan.runs:
            self._kernel_reshape_and_cache_a8(
                key=_strip_run_rows(key, run),
                value=_strip_run_rows(value, run),
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=_strip_run_slots(slot_mapping, run),
                pi_signs=emulator.pi_signs,
                tables=tables,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )

        # One launch per wave, and the public decode entry rather than the
        # scatter shim: a build that staged the batch writer on without the
        # scatter writer still has to land these tokens, and the golden writer
        # is what this file falls back to everywhere else.
        for wave in plan.waves:
            rows = torch.as_tensor(wave, dtype=torch.long, device=slot_mapping.device)
            self.reshape_scatter_decode(
                key=key.index_select(0, rows).contiguous(),
                value=value.index_select(0, rows).contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping.index_select(0, rows).contiguous(),
                emulator=emulator,
            )

    def _kernel_reshape_and_cache_a8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        pi_signs: torch.Tensor,
        tables: torch.Tensor,
        inv_sqrt_len: float,
    ) -> None:
        """The batch writer: rotate, quantise, pack and write whole pages.

        ``key`` and ``value`` are ``[T, H, D]`` bf16; ``key_cache`` and
        ``value_cache`` are ``(num_blocks, block_size, H, D/2)`` INT8 in the
        strip layout and ``scale_cache`` is ``(2, num_blocks, H, block_size)``
        fp32, for any ``block_size`` that is a multiple of 16 -- see
        :class:`TurboQuantValueLayout` for why this writer no longer needs
        ``block_size == 16`` and why that matters.

        ``tables`` is the Cube's resident **BF16** operand, ``2 * (D/32)``
        row-major 32x32 blocks of ±1 (:func:`_codec_tables`, which is also where
        the consequences of getting its dtype wrong are written down).
        ``pi_signs`` is folded into it, so this kernel does not read it --
        literally: its body is ``(void)piSigns``. An empty tensor is the honest
        way to say so, and because nothing dereferences it, a wrongly shaped
        ``pi_signs`` is invisible *here* while being fatal to the scatter writer,
        which does read it.

        There is no mode-4 channel-policy argument any more. The audited writer
        is mode-2 only -- its three cross-core channels are counters that bank
        unconsumed announcements, so recycling an id across pipeline generations
        is sound by construction and needs no single-shot fallback to bisect
        against. The ``reuse_mode4_flags`` flag and the
        ``..._mode4_single_shot`` entry point it selected are both gone from the
        kernel, so the schema no longer advertises the knob.

        The C launcher (``launch_turboquant_reshape_and_cache_bf16``,
        op_kernel/turboquant_reshape_batch.cpp) also takes ``num_tokens``,
        ``num_kv_heads``, ``head_size``, ``block_size``, ``num_blocks`` and
        ``tokens_per_core``. The schema does not: the first five come off the
        tensor shapes in csrc/attention/turboquant/torch_bindings.cpp, and
        ``tokens_per_core`` is derived there rather than accepted -- a caller
        that could choose it could contradict the schedule.

        ``tokens_per_core`` is **not** a compile-time constant of the kernel,
        which an earlier revision of this docstring claimed. It is the width of
        one grid block's share, and the kernel's seamless chunk pipeline runs
        ``tokens_per_core / 16`` strips as one uninterrupted chunk stream so that
        MTE2 does not drain at a strip boundary. The binding pins it to 16 today,
        which is a legal geometry and the one the goldens were taken at, but it
        is one strip per block -- the stream is one strip long and the refactor
        buys nothing. Raising it is a measurement, not a correctness change, and
        the constraint is only that ``num_tokens`` stay a whole multiple of it
        (``WholeStrips``, which the launcher throws on).
        """
        self._require_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE)(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            pi_signs,
            tables,
            inv_sqrt_len,
        )

    def _golden_reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """Rotate, quantise and write ``[T, H, D]`` K/V at the slots given."""
        self._write_slots(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    # -- ingestion: the decode scatter writer ------------------------------

    def reshape_scatter_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """Ingest one ``[T, H, D]`` token per request, scattered by slot.

        The decode phase. ``T`` is the batch size and ``slot_mapping`` is ``[T]``
        int32 with no contiguity of any kind. Negative slots are padding and are
        skipped, which is what a graph-capture dummy run feeds.
        """
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            self._kernel_reshape_scatter_aiv(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
            return
        self._golden_reshape_scatter_decode(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    def _kernel_reshape_scatter_aiv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        pi_signs: torch.Tensor,
        inv_sqrt_len: float,
    ) -> None:
        """The AIV scatter writer: one token per request, rotated and packed in UB.

        Same tensors and same strip cache as
        :meth:`_kernel_reshape_and_cache_a8`, with no contiguity contract on
        ``slot_mapping``: the value plane's write is a read-modify-write of one
        2048-byte strip, spliced in by register bitwise masks, which is the whole
        reason this kernel exists.

        The C launcher (``launch_turboquant_reshape_scatter_aiv``,
        op_kernel/turboquant_reshape_scatter.cpp) also takes ``block_dim``,
        ``cu_seqlens``, ``num_requests``, ``num_kv_heads``, ``head_size``,
        ``block_size``, ``num_blocks``, ``item_begin`` and ``item_count``. The
        schema takes none of them:
        csrc/attention/turboquant/torch_bindings.cpp derives the geometry from
        the tensor shapes and the work split from the device
        (``total_items = num_requests * num_kv_heads``, ``item_begin = 0``,
        ``item_count = total_items``, ``block_dim = min(total_items,
        aiv_core_num)``), and passes ``cu_seqlens`` as null -- the prefix sum a
        multi-token-per-request step would need. The canary runner's
        ``launch_probe`` is gone from the kernel: it was a synthetic buffer that
        published the block-index registers back to a host runner, and nothing
        here ever read it.
        """
        self._require_kernel(TURBOQUANT_OP_RESHAPE_SCATTER)(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            pi_signs,
            inv_sqrt_len,
        )

    def _golden_reshape_scatter_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """Rotate, quantise and scatter one token per request by slot."""
        self._write_slots(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    def _write_slots(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """The codec and the slot write both golden ingestion phases share.

        ``key`` and ``value`` are ``[T, H, D]``; ``slot_mapping`` is ``[T]``
        int32 of ``page * block_size + token``, negative for padding. The two
        phases differ in which kernel owns the write, not in what lands: both
        produce the same nibbles and the same fp32 scales.

        In the strip layout the value plane's write is a read-modify-write: a
        channel owns 16 consecutive nibbles, one per token, so token ``t`` shares
        a byte with ``t ^ 1`` and writing one token without the other's nibble
        would corrupt it. Only the touched ``(page, head, strip)`` panels are
        read back, modified and written, so every untouched lane survives --
        which is what makes this safe for a decode step that writes one token
        into a strip the previous step filled.
        """
        codes_key, key_scales = emulator.quantize(self.rotate(key, emulator))
        codes_value, value_scales = emulator.quantize(self.rotate(value, emulator))

        block_size = key_cache.shape[1]
        slots = slot_mapping.to(torch.int64)
        valid = slots >= 0
        if not bool(valid.all()):
            slots = slots[valid]
            codes_key = codes_key[valid]
            codes_value = codes_value[valid]
            key_scales = key_scales[valid]
            value_scales = value_scales[valid]
        if slots.numel() == 0:
            return

        pages = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots - pages * block_size

        if self.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR:
            self._write_strip_planes(codes_key, codes_value, key_cache, value_cache, pages, offsets)
        else:
            for packed, cache in (
                (emulator.pack(codes_key), key_cache),
                (emulator.pack(codes_value), value_cache),
            ):
                _flat_slot_view(cache)[slots] = packed

        # The scale plane is the one thing both layouts index identically:
        # ((kv * num_blocks + page) * num_kv_heads + head) * block_size + token,
        # which is scale_cache[plane][page, :, token].
        for plane, scales in (
            (TURBOQUANT_KEY_PLANE, key_scales),
            (TURBOQUANT_VALUE_PLANE, value_scales),
        ):
            scale_cache[plane][pages, :, offsets] = scales.to(scale_cache.dtype)

    def _write_strip_planes(
        self,
        codes_key: torch.Tensor,
        codes_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Write both planes of the audited strip cache, values read-modify-write.

        ``codes_*`` are ``[T, H, D]`` INT8 in ``[-8, 7]``; ``pages`` and
        ``offsets`` are ``[T]`` int64 of the page and the token's index inside
        it. The planes are the same buffers as always -- what changes is how
        they are addressed: ``(num_blocks, num_kv_heads, block_size, D/2)`` for
        the token-major key plane and
        ``(num_blocks, num_kv_heads, block_size/16, D, 8)`` for the value
        plane's ``[strip, channel, token % 16]`` nibbles.
        """
        num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
        head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
        if block_size % TURBOQUANT_STRIP_TOKENS:
            raise ValueError(
                f"the strip cache is written 16 tokens at a time, so block_size {block_size} must be a "
                f"multiple of {TURBOQUANT_STRIP_TOKENS}"
            )
        strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS

        # The key plane is token-major inside the page, so one token's bytes are
        # a contiguous run and no read-modify-write is needed.
        key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
        key_pages[pages, :, offsets] = TurboQuantEmulator.pack(codes_key)

        # The value plane: unpack only the pages this step touches, set the one
        # token lane of every channel, and pack them back.
        strips = torch.div(offsets, TURBOQUANT_STRIP_TOKENS, rounding_mode="floor")
        lanes = offsets - strips * TURBOQUANT_STRIP_TOKENS
        value_pages = _require_contiguous(value_cache, "value").view(
            num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
        )
        touched, inverse = torch.unique(pages, return_inverse=True)
        panel = TurboQuantEmulator.unpack(value_pages[touched])
        # [U, H, strips, D, 16] indexed [page, head, strip, channel, token % 16];
        # one token is one lane of every channel, which is codes_value[t].
        panel[inverse, :, strips, :, lanes] = codes_value
        value_pages[touched] = TurboQuantEmulator.pack(panel)

    # -- the cache read ----------------------------------------------------

    def gather(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather whole pages for a batch out of the paged cache.

        Returns ``(packed_key, key_scales, packed_value, value_scales)`` with the
        packed planes ``[B, S_pad, H, D/2]`` INT8 and the scales ``[B, S_pad, H]``
        fp32, where ``S_pad = max_blocks_per_seq * block_size``. Positions past a
        request's context length are whatever the allocator last left there; the
        caller masks them.

        The return shape is the same whichever layout the cache is in: this is
        where a mixed configuration -- an NPU writer and a golden reader -- is
        reconciled, so the un-packing and un-rotation above it never learn which
        writer ran.
        """
        if self._has_kernel(TURBOQUANT_OP_GATHER_PAGES):
            return self._kernel_gather_pages(
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
            )
        if self.value_layout() is TurboQuantValueLayout.CHANNEL_MAJOR:
            return self._golden_gather_pages_strip(key_cache, value_cache, scale_cache, block_tables)
        return self._golden_gather_pages(key_cache, value_cache, scale_cache, block_tables)

    def _kernel_gather_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise a batch's packed pages, for validating a gather in isolation.

        No standalone C launcher and no production caller: the fused decode
        streams pages through UB instead of materialising them.
        """
        return self._require_kernel(TURBOQUANT_OP_GATHER_PAGES)(key_cache, value_cache, scale_cache, block_tables)

    def _golden_gather_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``block_tables`` ``[B, max_blocks]`` int32 -> the four ``[B, S_pad, ...]`` panels."""
        block_size = key_cache.shape[1]
        pages = block_tables.to(torch.int64)
        batch_size, max_blocks = pages.shape
        within = torch.arange(block_size, device=pages.device, dtype=torch.int64)
        slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

        packed_key = _flat_slot_view(key_cache)[slots]
        packed_value = _flat_slot_view(value_cache)[slots]
        # scale_cache[plane][page] is (num_kv_heads, block_size); transpose the
        # gathered pages into token-major and flatten the page axis away.
        num_kv_heads = scale_cache.shape[2]

        def gathered_scales(plane: int) -> torch.Tensor:
            return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

        return (
            packed_key,
            gathered_scales(TURBOQUANT_KEY_PLANE),
            packed_value,
            gathered_scales(TURBOQUANT_VALUE_PLANE),
        )

    def _golden_gather_pages_strip(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read a cache an NPU writer wrote, into the same four panels as above.

        Inverts both differences the strip layout has from the golden one: the
        page axis order, and the value plane's channel-major-inside-a-strip
        nibble index. The buffer is the same bytes either way; what changes is
        how they are read.

        ``block_tables`` is ``[B, max_blocks]`` int32. The returned panels are
        ``[B, S_pad, H, D/2]`` INT8 and ``[B, S_pad, H]`` fp32, token-major, so
        the caller's unpack and un-rotation are unchanged.
        """
        num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
        head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
        if block_size % TURBOQUANT_STRIP_TOKENS:
            raise ValueError(
                f"the strip cache is laid out 16 tokens at a time, so block_size {block_size} must be a "
                f"multiple of {TURBOQUANT_STRIP_TOKENS}"
            )
        strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS
        pages = block_tables.to(torch.int64)
        batch_size = pages.shape[0]

        # The key plane is token-major inside the page: only the page axis order
        # differs, so lift the token axis back out in front of the head axis.
        key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
        packed_key = key_pages[pages].permute(0, 1, 3, 2, 4).reshape(batch_size, -1, num_kv_heads, packed_bytes)

        # The value plane is [strip, channel, token % 16] nibbles. Undo it on
        # CODES rather than on bytes: tokens 2t and 2t+1 share a byte, so a
        # byte-level transpose would have to split nibble pairs, while unpacking
        # first makes it an ordinary axis permutation.
        value_pages = _require_contiguous(value_cache, "value").view(
            num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
        )
        # [B, max_blocks, H, strips, D, 16] -> [B, max_blocks, strips, 16, H, D],
        # so flattening (strips, 16) gives token = strip * 16 + lane.
        codes = TurboQuantEmulator.unpack(value_pages[pages])
        token_major = codes.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, -1, num_kv_heads, head_size)
        packed_value = TurboQuantEmulator.pack(token_major.contiguous())

        # The scale plane is indexed identically in both layouts, so only the
        # token axis has to come forward.
        def gathered_scales(plane: int) -> torch.Tensor:
            return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

        return (
            packed_key,
            gathered_scales(TURBOQUANT_KEY_PLANE),
            packed_value,
            gathered_scales(TURBOQUANT_VALUE_PLANE),
        )

    def dequantize_cache(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        emulator: TurboQuantEmulator,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gathered packed pages -> dense activations in ``dtype``, in the model's basis."""
        rotated = self.unpack_and_dequantize(packed, scales, emulator)
        return self.inverse_rotate(rotated, emulator).to(dtype)

    # -- the fused decode --------------------------------------------------

    def fused_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        emulator: TurboQuantEmulator,
        num_heads: int,
        num_kv_heads: int,
        scale: float,
        output: torch.Tensor,
        tiling: torch.Tensor,
        workspace: torch.Tensor,
        query_rot: torch.Tensor,
    ) -> torch.Tensor | None:
        """The one-launch decode, or ``None`` when it did not run.

        ``query`` is ``[B, N, D]`` and ``output`` is ``[B, N, D]`` in the model's
        element type. ``None`` is the caller's signal to gather, dequantise and
        attend instead: that fallback is a whole attention rather than one
        operation, so it lives in the backend.

        ``tiling``, ``workspace`` and ``query_rot`` are the three buffers the op
        used to allocate inside its own launch and now takes as inputs. They
        belong to the metadata builder
        (:class:`AscendTurboQuantMetadataBuilder`), which sizes them once from
        the maximum geometry and refills ``tiling`` once per step for every
        layer.

        **All three are required, and none has a default.** A default would have
        to be a buffer allocated here, and for ``tiling`` that is the worst
        available failure mode: an unfilled tiling buffer is all zeros, every
        descriptor reads ``num_tiles == 0``, and the kernel skips every worker
        and returns having written nothing. The decode would report success over
        an untouched output. So a caller without them is a programming error and
        says so.
        """
        if not self._has_kernel(TURBOQUANT_OP_FUSED_DECODE):
            return None
        return self._kernel_fused_decode(
            query=query,
            pi_signs=emulator.pi_signs,
            rot_tables=emulator.codec_tables,
            h16=emulator.h16,
            gate=None,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
            output=output,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_splits=TURBOQUANT_DECODE_SPLITS,
            heads_per_task=turboquant_decode_heads_per_task(num_heads, num_kv_heads),
            output_stage=TURBOQUANT_STAGE_UNROTATED,
            scale=scale,
            inv_sqrt_len=emulator.inv_sqrt_len,
        )

    def _kernel_fused_decode(
        self,
        query: torch.Tensor,
        pi_signs: torch.Tensor,
        rot_tables: torch.Tensor,
        h16: torch.Tensor,
        gate: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        tiling: torch.Tensor,
        workspace: torch.Tensor,
        query_rot: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        num_splits: int,
        heads_per_task: int,
        output_stage: int,
        scale: float,
        inv_sqrt_len: float,
    ) -> torch.Tensor:
        """The raw-query fused decode, writing through ``output``.

        ``query`` and ``output`` are ``[B, N, 256]`` fp16 or bf16, ``block_tables``
        ``[B, max_blocks]`` int32 and ``context_lens`` ``[B]`` int32.
        ``rot_tables`` is the Cube's resident INT8 operand and ``h16`` the 16x16
        fp16 sign matrix; ``gate`` is read only at ``output_stage`` 2.
        ``output_stage`` names a ``TurboQuantOutputStage``, and the decode path
        here wants its context back in the model's own basis, so it passes 1.

        Returns ``output`` so the caller can tell a launch from the ``None`` of a
        golden fallback. The op itself returns nothing: like the two writers it
        writes through its ``Tensor!`` argument.

        Six of the launcher's inputs are built inside the op rather than passed.
        ``mode`` is fixed to ``TurboQuantMode::KV4_FP8``, the only mode the fused
        decode is validated in. ``element_type`` is **not** read off the output
        dtype as this docstring once said: the operator has one pair of bf16
        instantiations and its ``_impl`` dispatch returns without launching for
        any other ``AscendType``, so mapping an fp16 query would leave ``output``
        whatever the allocator gave it and report success. The binding therefore
        refuses a non-bf16 query outright and passes ``AscendType::BF16``
        unconditionally (:data:`TURBOQUANT_KERNEL_DTYPE`).

        THREE OF THEM USED TO BE, and are arguments now, because a buffer the op
        allocates has a fresh address on every launch and a captured ACL graph
        replays the address it saw:

        * ``tiling``, ``decode_tiling_words(tokens, splits, grid)`` int32 of
          descriptors. Built by ``build_decode_tiling`` as before, but once per
          *step* for every layer rather than once per layer, through
          ``npu_turboquant_build_decode_tiling``. The op reads it and checks its
          size; it no longer reads the context lengths on the host at all.
        * ``workspace``, the Split-K partials,
          ``tokens * heads * splits * (D + kPartialTail)`` fp32.
        * ``query_rot``, the fp32 rotated query image the prologue writes and the
          decode then reads -- it is not ``query`` and the two are not
          interchangeable.

        Each is sized from the launch geometry, so the op still owns the
        arithmetic: it derives the sizes and ``TORCH_CHECK``s what it was given
        rather than trusting the caller. An undersized ``tiling`` was previously
        impossible and is now the one new failure mode, which is why it is a
        named check and not a hang.

        What remains the binding's, and is now a function-local static keyed by
        ``(head_size, device_index)`` rather than a per-launch build:

        * the query basis codec's constant pool, a pure function of
          ``head_size``;
        * ``mode_tables``, the decode codec's own constant pool. It is a required
          positional argument of the launcher, and for ``KV4_FP8`` it is never
          dereferenced: ``ConstTableWords`` returns 0 for an affine mode with the
          gather LUT off, so ``Init`` reads no words from it
          (op_kernel/common/turboquant_codec_mx.h). That is a property of the
          mode and not of the ABI, so the binding hands over a small zeroed
          buffer rather than a null pointer -- the operator's own fixture does
          the same (``modes.Upload(std::vector<std::uint32_t>(8, 0))``), and a
          mode that did read it would otherwise fault.

        ``rot_tables`` is passed, and is in the same position: the query basis
        hands it to ``TurboQuantModeCodec::Init``, which reads nothing from it
        under this mode. It is given the real operand anyway so the two sides
        never disagree about its dtype -- see :func:`_codec_tables`.

        THIS OP CANNOT READ A CACHE EITHER WRITER WROTE, not directly. The
        writers produce the strip layout; this reads the NZ-tiled layout, and the
        two disagree on four independent axes -- byte order, nibble pairing, the
        value plane's token axis and the scale plane -- not just on
        ``block_size``. The binding TORCH_CHECKs what it can, so the mismatch is
        a refused launch naming the conflict rather than an attention over bytes
        in the wrong order.

        :func:`vllm_ascend.attention.turboquant_nz_ingress.turboquant_repack_strip_to_nz_tiled`
        converts a writer-produced cache into the three planes this op reads, and
        is what makes the decode reachable at all today. It is a reference
        conversion for exercising and gating the decode -- a whole extra pass
        over the pages it touches -- and deliberately not wired into this path:
        the real fix is a write-time NZ ingress kernel in the harness. See that
        module's header for the four axes and the evidence for each.
        """
        self._require_kernel(TURBOQUANT_OP_FUSED_DECODE)(
            query,
            pi_signs,
            rot_tables,
            h16,
            gate,
            key_cache,
            value_cache,
            scale_cache,
            block_tables,
            context_lens,
            tiling,
            workspace,
            query_rot,
            output,
            num_heads,
            num_kv_heads,
            num_splits,
            heads_per_task,
            output_stage,
            scale,
            inv_sqrt_len,
        )
        return output


# ---------------------------------------------------------------------------
# Meeting the batch writer's strip contract from an arbitrary prefill step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboQuantStripRun:
    """One batch-writer launch: tokens ``[start, stop)`` plus ``pad`` replicas.

    ``pad`` is how many copies of token ``stop - 1`` complete a trailing partial
    strip to :data:`TURBOQUANT_STRIP_TOKENS`. Their slots are the ones that
    follow ``slot_mapping[stop - 1]`` inside the same physical page -- a page the
    request owns and has not filled yet -- and nothing reads them: a reader stops
    at the sequence length, and the next step overwrites them, whether that is
    the next prefill chunk or a decode token spliced in by the scatter writer.
    """

    start: int
    stop: int
    pad: int


@dataclass(frozen=True)
class TurboQuantStripPlan:
    """How one ingestion step is split between the two cache writers.

    ``runs`` go to the batch writer in order, ``waves`` to the scatter writer,
    one launch each. A wave never holds two tokens of one ``(page, strip)``: the
    scatter writer's value-plane write is a read-modify-write of the whole
    16-token strip, so two tokens of one strip in one launch would race and lose
    an update.
    """

    runs: tuple
    waves: tuple
    num_tokens: int

    @property
    def is_whole_batch(self) -> bool:
        """True when one unpadded batch-writer launch covers every token.

        The caller then passes its own tensors straight through, which is what
        keeps an already-conforming step byte-for-byte what it was before the
        split existed.
        """
        return not self.waves and self.runs == (TurboQuantStripRun(0, self.num_tokens, 0),)

    @property
    def num_launches(self) -> int:
        """Kernel launches this plan costs, over both writers."""
        return len(self.runs) + len(self.waves)


def turboquant_plan_strip_ingestion(
    slots: Sequence[int], block_size: int, strip: int = TURBOQUANT_STRIP_TOKENS
) -> TurboQuantStripPlan:
    """Split one step's ``slot_mapping`` between the batch and scatter writers.

    ``slots`` is ``slot_mapping`` as host ints -- ``page * block_size + token``,
    negative for padding. The batch writer reads one slot per strip on the
    scalar unit and derives both the page and the strip inside it from that slot
    alone, so a strip it writes must be 16 tokens whose slots are strip-aligned
    and consecutive. That is also what keeps a strip inside one page, given
    ``block_size % 16 == 0``, and it is a per-strip contract rather than a
    per-step one: strips may jump between pages freely, which is what lets one
    run cover a prefill spanning several non-adjacent blocks.

    Three shapes come out of a real step:

    * **Whole strips.** The bulk of a prefill, coalesced into as few runs as
      possible, across page boundaries included.
    * **A strip-aligned partial strip.** The last ``T % 16`` tokens of a request,
      at the end of the batch or at a request boundary inside it. Completed to a
      whole strip by replicating its last token, so the batch writer can take it
      as one more strip -- see :class:`TurboQuantStripRun` for why the replicas
      are harmless, and note that the pad is refused outright unless every
      padded slot is free of a token this very step writes.
    * **Anything else.** A chunked-prefill continuation that does not start on a
      strip, a decode token carried in a mixed batch, a padding slot from a dummy
      capture run. Its strip may already hold another step's tokens, which only a
      read-modify-write may touch, so these go to the scatter writer -- one launch
      per token of a shared strip, which is the price of not corrupting the rest
      of the strip around them.

    Replication rather than zeros is deliberate: an all-zero row has no finite
    power-of-two scale, and a NaN parked in the scale plane would survive into a
    gathered page and out through the dequantisation before any mask reached it.
    """
    num_tokens = len(slots)
    runs: list[TurboQuantStripRun] = []
    scatter: list[int] = []
    # Only a pad candidate needs this, and a conforming step has none, so it is
    # built on the first one rather than on every call.
    written: set[int] | None = None

    def run_length(start: int, limit: int) -> int:
        """Consecutive non-negative slots from ``start``, capped at ``limit``."""
        if start >= num_tokens or slots[start] < 0:
            return 0
        length = 1
        while length < limit and start + length < num_tokens and slots[start + length] == slots[start] + length:
            length += 1
        return length

    def aligned(start: int) -> bool:
        return start < num_tokens and slots[start] >= 0 and slots[start] % strip == 0

    def whole_strip(start: int) -> bool:
        return aligned(start) and run_length(start, strip) == strip

    def pad_for_partial_strip(start: int) -> int:
        """Replicas that complete a strip-aligned partial strip at ``start``, or 0."""
        if not aligned(start):
            return 0
        tail = run_length(start, strip)
        if not 0 < tail < strip:
            return 0
        pad = strip - tail
        last = slots[start] + tail - 1
        # Implied by ``slots[start] % strip == 0`` and ``block_size % strip == 0``,
        # and checked anyway: a page geometry that broke it would put the pad in
        # the next page, which belongs to someone else.
        if block_size <= 0 or (last + pad) // block_size != last // block_size:
            return 0
        # The pad must not land on a slot this step writes for real. It cannot,
        # for a request whose tail this is -- but a caller that repeated a slot,
        # or ordered one out of sequence, would otherwise have a replica
        # silently beat the real token to it.
        nonlocal written
        if written is None:
            written = {slot for slot in slots if slot >= 0}
        if any(slots[start] + offset in written for offset in range(tail, strip)):
            return 0
        return pad

    index = 0
    while index < num_tokens:
        if whole_strip(index):
            stop = index + strip
            while whole_strip(stop):
                stop += strip
            runs.append(TurboQuantStripRun(index, stop, 0))
            index = stop
            continue
        pad = pad_for_partial_strip(index)
        if pad:
            stop = index + (strip - pad)
            runs.append(TurboQuantStripRun(index, stop, pad))
            index = stop
            continue
        scatter.append(index)
        index += 1

    return TurboQuantStripPlan(
        runs=tuple(runs),
        waves=_strip_scatter_waves(slots, scatter, strip),
        num_tokens=num_tokens,
    )


def _strip_scatter_waves(slots: Sequence[int], tokens: Sequence[int], strip: int) -> tuple:
    """Group scatter-writer tokens so that no wave holds two tokens of one strip."""
    waves: list[list[int]] = []
    depth: dict[int, int] = {}
    for token in tokens:
        slot = slots[token]
        if slot < 0:
            # The writer skips a negative slot, so it reads and writes nothing
            # and cannot collide with anything.
            wave = 0
        else:
            wave = depth.get(slot // strip, 0)
            depth[slot // strip] = wave + 1
        while len(waves) <= wave:
            waves.append([])
        waves[wave].append(token)
    return tuple(tuple(wave) for wave in waves)


def _strip_run_rows(activations: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    """``[T, H, D]`` -> the run's rows, its trailing partial strip replicated out."""
    window = activations[run.start : run.stop]
    if not run.pad:
        return window
    return torch.cat((window, window[-1:].expand(run.pad, *window.shape[1:]))).contiguous()


def _strip_run_slots(slot_mapping: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    """``[T]`` int32 -> the run's slots, continued through its replicated tail."""
    window = slot_mapping[run.start : run.stop]
    if not run.pad:
        return window
    pad = torch.arange(1, run.pad + 1, dtype=window.dtype, device=window.device) + window[-1]
    return torch.cat((window, pad)).contiguous()


def _flat_slot_view(cache: torch.Tensor) -> torch.Tensor:
    """``(num_blocks, block_size, H, D/2)`` -> ``(num_blocks * block_size, H, D/2)``.

    A view, never a copy: a scatter into a reshaped *copy* would be silently
    dropped, so a non-contiguous plane is an error rather than a slow path.
    """
    if not cache.is_contiguous():
        raise RuntimeError(
            "the TurboQuant packed cache plane must be contiguous to be addressed by slot; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache.view(cache.shape[0] * cache.shape[1], *cache.shape[2:])


def _require_contiguous(cache: torch.Tensor, name: str) -> torch.Tensor:
    """A cache plane that is about to be re-viewed by page, or an error.

    A view of a non-contiguous plane would be a copy, and a scatter into a copy
    is silently dropped -- so this is an error rather than a slow path.
    """
    if not cache.is_contiguous():
        raise RuntimeError(
            f"the TurboQuant {name} plane must be contiguous to be addressed by page; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache


def _codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    """The Cube's resident rotation operand for ``emulator``, built once.

    A thin accessor over :attr:`TurboQuantEmulator.codec_tables`, kept as a free
    function because that is how the fixtures and the rotate stubs reach it. The
    operand it returns is documented on :func:`_build_codec_tables`.

    **It is a cache, and that is load-bearing.** This used to build the operand
    on every call, and ``fused_decode`` calls it once per layer per decode step:
    eleven device kernels and two allocations inside the region an ACL graph has
    to capture. ``emulator_for`` keys emulators by ``(head_size, device)``, so
    one model has one of these however many layers read it.
    """
    return emulator.codec_tables


def _build_codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    """Build the Cube's resident rotation operand, ``2 * (D/32)`` row-major 32x32 BF16 blocks.

    ``tables[s * groups + g][c0][i] = (s ? -1 : +1) * pi_signs[g*32 + c0] * H32[c0][i]``:
    the sign diagonal and the inner Hadamard factor, pre-composed, with the
    outer factor supplied by which block the kernel picks
    (``parity(c1 & g) * groups + g``). Built here so the dry run hands the
    kernel-backed stages exactly the operand ``tq_oracle::build_tables`` audits.

    **BF16, NOT INT8**, and the distinction is load-bearing rather than
    cosmetic. The batch writer stages this blob as the Cube's BF16 L1 operand::

        // op_kernel/turboquant_reshape_batch.cpp
        ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(
            reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);

    ``stage_blocks_nz<Blocks, Dim>`` reads ``Blocks * Dim * Dim`` *bfloat16_t*
    elements -- 16384 of them at ``D`` 256, so 32768 bytes, which is what
    ``kTableBytes = kTableElems * kElemBytes`` reserves in L1 with
    ``kElemBytes == 2``. An INT8 tensor of the same element count is half those
    bytes, so handing one over would both feed the Cube ±1 bytes reinterpreted
    as BF16 -- a wrong rotation, not a slightly wrong one -- and read 16 KB past
    the end of the operand.

    The launcher's parameter is still declared ``const std::int32_t*`` and the
    kernel still reinterprets it, which is exactly how this went wrong: the
    declared pointer type says nothing about the element type, and neither side
    of a ``reinterpret_cast`` is checkable. The authority on what the device
    wants is the operator's own runner, which converts the oracle's int8 blob on
    the way up -- ``tests/npu/test_live_write.cpp``,
    ``tb[i] = to_bf16(table_blob[i])`` into a buffer sized
    ``table_blob.size() * sizeof(std::uint16_t)``.

    Every entry is ±1, exact in BF16, so this is a representation change and not
    a rounding: the operand the Cube contracts against is bit-identical to the
    oracle's, element for element.

    ``npu_turboquant_fused_decode`` takes the same blob as ``rot_tables``, where
    it reaches ``TurboQuantModeCodec::Init`` and is **not read at all** for
    ``KV4_FP8`` -- ``ConstTableWords`` returns 0 for an affine mode with the
    gather LUT off (op_kernel/common/turboquant_codec_mx.h). So the decode would
    have accepted either dtype, and a decode that runs is no evidence that this
    operand is right. The prefill is where it bites.
    """
    device = emulator.device
    h_inner = _sylvester_hadamard(emulator.group, device, torch.float32)
    signs = emulator.pi_signs.reshape(emulator.groups, emulator.group)
    # [groups, 32, 32]: signs broadcast over the inner factor's rows.
    positive = signs.unsqueeze(-1) * h_inner.unsqueeze(0)
    tables = torch.stack((positive, -positive), dim=0)
    return tables.reshape(TURBOQUANT_PLANES * emulator.groups, emulator.group, emulator.group).to(
        TURBOQUANT_TABLE_DTYPE
    )


# One ``h16`` per device, for the process. Keyed by the device's string form
# rather than by the object, because ``torch.device("npu")`` and
# ``torch.device("npu", 0)`` are different objects naming one device.
_H16_CACHE: dict[str, torch.Tensor] = {}


def _h16_operand(device: torch.device) -> torch.Tensor:
    """The Cube's resident ``h16`` operand, a ``[16, 16]`` fp16 matrix of ``+-1``.

    The product of four Sylvester butterfly factors, which is the unnormalised
    Sylvester Hadamard of order 16 (``routines/routine_cube_hadamard16.hpp``:
    the caller supplies the sign matrix, the AIC generates no coefficients).
    fp16 for both query element types -- the kernel converts it on ingest where
    the Cube chunk operand is not fp16.

    Cached per device for the same reason :func:`_codec_tables` is: it is a
    constant, and building it once per layer per step put ``arange``, three
    bitwise ops, ``zeros_like`` and ``where`` inside the decode launch.
    """
    key = str(device)
    operand = _H16_CACHE.get(key)
    if operand is None:
        operand = _sylvester_hadamard(TURBOQUANT_H16_ORDER, device, torch.float16)
        _H16_CACHE[key] = operand
    return operand


# ---------------------------------------------------------------------------
# The metadata and its builder
# ---------------------------------------------------------------------------


@dataclass
class AscendTurboQuantMetadata(AscendMetadata):
    """``AscendMetadata`` plus the fused decode's four per-step device operands.

    All four are views into buffers :class:`AscendTurboQuantMetadataBuilder`
    allocates once and refreshes in place, so their addresses are the same on
    every step of a given engine. That is the property an ACL graph replay
    needs: the replay re-executes only device work, so a Python-side refresh of
    a *reallocated* buffer never runs again after capture.
    """

    # The step's KV lengths on the device, FIA padding rows zeroed. See
    # :meth:`AscendTurboQuantMetadataBuilder._turboquant_context_lens` for why
    # this is not ``seq_lens`` and not ``seq_lens_gpu``.
    turboquant_context_lens: torch.Tensor = None  # type: ignore[assignment]
    # The decode tiling descriptors, filled once per step for every layer.
    turboquant_tiling: torch.Tensor = None  # type: ignore[assignment]
    # The Split-K partial workspace and the fp32 rotated query image: pure
    # kernel scratch, never read or written by the host.
    turboquant_workspace: torch.Tensor = None  # type: ignore[assignment]
    turboquant_query_rot: torch.Tensor = None  # type: ignore[assignment]
    # The token extent the builder filled the host-written buffers for.
    #
    # **This is the single authority for the launch extent**, and it exists
    # because the fill side and the read side otherwise derive it from
    # different quantities. The builder fills one descriptor row per row of
    # ``seq_lens``; the operator derives its own ``tokens`` from
    # ``query.size(0)`` and indexes the descriptor array by it -- and the
    # dispatch records sit at ``decode_dispatch_word_base(tokens, splits)``,
    # an offset that *moves with the token count*. So a launch whose
    # ``tokens`` exceeds the extent the host filled does not read a short
    # buffer and trip its size check; it reads descriptors out of the dispatch
    # array, reads dispatch records out of words nobody wrote, and reports
    # success. Publishing the number the host used, and slicing the launch
    # with it, makes the two agree by construction instead of by the
    # coincidence that a decode step has one token per request.
    turboquant_num_tokens: int = 0


class AscendTurboQuantMetadataBuilder(AscendAttentionMetadataBuilder):
    """The shared builder plus the fused decode's launch buffers.

    **Why a builder at all.** Three of the fused decode's operands used to be
    allocated inside its own launch, once per layer per step, and a fourth --
    the tiling descriptors -- was *built on the host and uploaded* there too.
    Both are per-batch quantities, not per-layer ones: on a 60-layer model one
    decode step did the same host work and the same host-to-device copy sixty
    times. This builder does each once, into a buffer it owns.

    **Why the builder and not ``get_capture_resource``.**
    ``vllm_ascend.compilation.updatable_graph.get_capture_resource`` is the
    repository's mechanism for a capture-scoped buffer, and it is what
    ``full_graph_fia`` uses for the FIA workspace. It is the right owner for a
    buffer only the device touches, and the wrong one here, for two reasons.
    The tiling buffer is written by the *host* before every replay, and a
    ``get_capture_resource`` buffer is allocated during capture, owned by the
    graph's memory pool and weak-ref'd at ``capture_end`` -- it exists so the
    graph has a pointer, not so the host can refill it. And outside capture
    ``get_capture_resource`` simply calls its factory
    (``updatable_graph.py:189``), so it would keep allocating per step in the
    mode TurboQuant is actually served in. A buffer allocated here, before any
    capture, is strongly owned, host-writable and has one address for the
    engine's life -- the same contract ``block_tables`` and ``seq_lens`` already
    have.

    **Why one buffer and not one per capture size.** Every size is exactly
    computable from ``max_num_reqs``, and a step uses a prefix, so a single
    maximum-sized allocation serves every capture size. The tiling buffer's
    exact extent also depends on the ``grid`` the operator derives, which Python
    deliberately does not reproduce; sizing to
    :func:`turboquant_tiling_upper_bound_words` costs 1 KiB of slack and removes
    that duplication.

    **What is not done here.** Nothing is captured and no graph mode changes.
    ``NPUPlatform._apply_turboquant_graph_mode`` still holds an
    ``ENABLE_TURBOQUANT=1`` run to PIECEWISE or NONE, and still should: the
    golden fallback paths this backend keeps are not capturable, and the cache
    the ingestion writers produce is not the cache the fused decode reads.
    """

    metadata_cls: type[AscendMetadata] = AscendTurboQuantMetadata

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._num_kv_heads = int(kv_cache_spec.num_kv_heads)
        self._head_size = int(kv_cache_spec.head_size)
        self._block_size = int(kv_cache_spec.block_size)
        self._num_heads = int(vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config))
        self._num_splits = TURBOQUANT_DECODE_SPLITS
        self._heads_per_task = turboquant_decode_heads_per_task(self._num_heads, self._num_kv_heads)

        # Allocated once, at engine start, outside any capture. Every step
        # writes a prefix of the first and the kernels write prefixes of the
        # other two.
        self._context_lens = torch.zeros(self._max_num_reqs, dtype=torch.int32, device=device)
        self._tiling = torch.zeros(
            turboquant_tiling_upper_bound_words(self._max_num_reqs, TURBOQUANT_DECODE_MAX_SPLITS),
            dtype=torch.int32,
            device=device,
        )
        self._workspace = torch.zeros(
            turboquant_decode_workspace_floats(self._max_num_reqs, self._num_heads, self._head_size, self._num_splits),
            dtype=torch.float32,
            device=device,
        )
        self._query_rot = torch.zeros(
            turboquant_decode_query_rot_floats(self._max_num_reqs, self._num_heads, self._head_size),
            dtype=torch.float32,
            device=device,
        )
        # The host staging for the zero-padded lengths: one contiguous int32
        # run, so the per-step upload is a single H2D rather than a scatter.
        #
        # **Pinned, and a ring of two.** The upload is ``non_blocking``, so the
        # copy is enqueued on the step's stream and completes later, while the
        # host races ahead to build the next step's metadata -- and with async
        # scheduling that next build overlaps this step's execution by design.
        # A single staging buffer would therefore be rewritten while its own
        # copy was still in flight, which is a torn upload, not a stale one.
        # Pinning is what makes the copy genuinely asynchronous (from pageable
        # memory the runtime stages through its own bounce buffer), and depth
        # two is what makes the overwrite safe: by the time the host returns to
        # slot N again, the whole of step N+1 has been enqueued behind step N's
        # copy on an in-order stream, so that copy has retired.
        self._staging_slots = TURBOQUANT_STAGING_SLOTS
        self._staging_step = 0
        staging = [torch.zeros(self._max_num_reqs, dtype=torch.int32, device="cpu") for _ in range(self._staging_slots)]
        try:
            staging = [slot.pin_memory() for slot in staging]
            self._staging_pinned = True
        except (RuntimeError, NotImplementedError):
            # No pinned allocator on this host. The ring is then pointless but
            # harmless, and the copy below becomes blocking, which is correct
            # for the same reason -- it just costs a host wait per step.
            self._staging_pinned = False
        self._context_lens_host_ring = staging

    # -- the per-step work -------------------------------------------------

    def _build_backend_metadata(
        self,
        common_attn_metadata: Any,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, Any]:
        """The four extra metadata fields, once per step.

        This is the shared builder's documented extension point, which is why
        nothing in ``AscendAttentionMetadataBuilder`` changes: a non-TurboQuant
        backend never reaches this method.
        """
        context_lens, host = self._refresh_context_lens(common_attn_metadata, seq_lens)
        tokens = int(context_lens.numel())
        self._reject_multi_token_steps(common_attn_metadata, tokens)
        self._refresh_tiling(host, tokens)
        return {
            "turboquant_context_lens": context_lens,
            "turboquant_tiling": self._tiling,
            "turboquant_workspace": self._workspace,
            "turboquant_query_rot": self._query_rot,
            "turboquant_num_tokens": tokens,
        }

    def _reject_multi_token_steps(self, common_attn_metadata: Any, tokens: int) -> None:
        """Refuse a step that carries more query rows than descriptor rows.

        The tiling descriptor array is indexed *by token*, and this builder
        fills one row per row of ``seq_lens`` -- that is, one per request. For
        an ordinary decode step those are the same number. For a speculative or
        MTP step they are not: the runner pads tokens to
        ``num_reqs_padded * uniform_decode_query_len``
        (``model_runner_v1.py:1095``), so the query carries
        ``decode_token_per_req`` rows per request while ``seq_lens`` still
        carries one length per request.

        Tiling that step correctly means expanding each request's length into
        one length per accepted token, which this builder does not do. The
        alternative to refusing is mis-tiling: the extra query rows would read
        descriptors the host never wrote. So this is a guard, not a limitation
        being introduced -- the decode has never been able to serve such a step,
        and nothing else in this repository gates TurboQuant against
        speculative decoding, so without this it was silent.

        **Not planned**, and the reason is not effort. Expanding the lengths is
        Python, but ``build_decode_tiling`` reads
        ``block_table + token * max_blocks``
        (``turboquant_tiling.cpp:123``) while the engine's block table has one
        row per *request*. At one token per request those coincide; past that
        the operator needs either a token-to-request operand -- a schema and
        kernel change -- or a block table the host repeats per token every
        step, which is the per-step host-to-device copy this whole change set
        removed. ``TURBOQUANT_GRAPH_CAPTURE_RFC.md`` records the decision and
        the sign-off question.
        """
        per_req = int(getattr(common_attn_metadata, "decode_token_per_req", 1) or 1)
        if per_req > 1:
            raise RuntimeError(
                "the TurboQuant fused decode cannot tile a multi-token decode step: this step "
                f"carries {per_req} query rows per request over {tokens} descriptor rows, and the "
                "tiling descriptors are indexed by token while this builder fills one row per "
                "request. Speculative decoding and MTP need the per-request context lengths "
                "expanded per accepted token first."
            )

    def _refresh_context_lens(
        self, common_attn_metadata: Any, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's KV lengths, on the device and on the host staging slot.

        **This is the padded-row contract.** The row it saves is the FIA dummy
        request. ``build()`` appends one when the TND layout needs the q-derived
        batch size bumped, with a KV length of **1** and a zero block-table row
        (``attention_v1.py:387-390``), deliberately, because FIA validates
        ``actualSeqLengthsKv`` and refuses a zero. A length of 1 is not harmless
        to this decode: ``build_decode_tiling`` writes a *live* descriptor for
        it -- one tile over logical page 0 -- so that row attends over whatever
        block 0 holds and writes a nonzero result. Downstream trimming is what
        hides it.

        The fused decode needs the opposite convention: a dead row must be the
        all-zero descriptor, which is what ``num_tiles == 0`` means and is the
        single predicate both the Cube and the Vector role branch on to skip a
        worker. So the two views differ by exactly this: FIA keeps its ones,
        TurboQuant gets zeros past ``num_reqs``.

        Note which rows this is *not* about. The cudagraph padding rows between
        ``num_reqs`` and ``num_reqs_padded`` are already zero -- the runner
        does ``self.seq_lens[num_reqs:].fill_(0)``
        (``model_runner_v1.py:1616``) and zeroes their block-table rows at
        ``:3594`` -- and ``common_attn_metadata.num_reqs`` is itself the
        *padded* count (``:3680``). So ``live`` here is already the padded
        count and this zeroing reaches exactly the FIA dummy beyond it, which
        is the one row carrying a 1. Both facts matter: if the runner ever
        stopped zeroing, a stale length from a longer previous batch would make
        a padding row live, and over the split-K limit it would also flip that
        row's fuse decision (``turboquant_tiling.cpp:93``).

        Returns both views because the caller needs both: the device prefix is
        what the launch reads, and the host slot is what ``build_decode_tiling``
        reads to derive each worker's tile count.
        """
        num_reqs = int(common_attn_metadata.num_reqs)
        padded = int(seq_lens.shape[0])
        if padded > self._max_num_reqs:
            raise RuntimeError(
                f"the TurboQuant builder was sized for {self._max_num_reqs} requests but this step "
                f"carries {padded}. max_num_seqs is what sizes the launch buffers."
            )
        live = min(num_reqs, padded)
        host = self._context_lens_host_ring[self._staging_step % self._staging_slots]
        self._staging_step += 1
        host[:live] = seq_lens[:live].to(torch.int32)
        if padded > live:
            # The FIA dummy requests. Zero, not one: see the docstring.
            host[live:padded] = 0
        # Asynchronous only where the staging is pinned: a non-blocking copy out
        # of pageable memory is the one combination that is neither fast nor
        # safe, because the runtime bounces it through a buffer of its own while
        # the host believes it may reuse the source.
        self._context_lens[:padded].copy_(host[:padded], non_blocking=self._staging_pinned)
        return self._context_lens[:padded], host

    def _refresh_tiling(self, host_context_lens: torch.Tensor, tokens: int) -> None:
        """Fill the tiling descriptors for this step, for every layer at once.

        A no-op when the operator is absent, which is every build that is not an
        ``ascend950`` one: the decode then runs its golden fallback and reads no
        descriptors. Filling a buffer nothing reads would only be a way to fail
        on hosts the kernels do not exist on.

        The host half of this is ``build_decode_tiling``, unchanged and still
        the authority -- the operator is a thin wrapper that calls it on a host
        staging buffer and copies the result into ``self._tiling``. Python does
        not reproduce any of its arithmetic.
        """
        namespace = getattr(torch.ops, TURBOQUANT_DEFAULT_NAMESPACE, None)
        entry = getattr(namespace, TURBOQUANT_OP_BUILD_DECODE_TILING, None) if namespace else None
        if entry is None:
            return
        entry(
            host_context_lens[:tokens],
            self._tiling,
            self._num_heads,
            self._num_kv_heads,
            self._head_size,
            self._block_size,
            self.max_num_blocks_per_req,
            self._num_splits,
            self._heads_per_task,
        )


# ---------------------------------------------------------------------------
# The backend descriptor.
# ---------------------------------------------------------------------------


# vLLM resolves a backend both ways: by the class path this platform's
# `get_attn_backend_cls` returns, and by looking the name `get_name` reports up
# in `AttentionBackendEnum`. Registering the class against the member whose
# name it reports is what keeps those two answers the same object. The second
# argument is deliberately left out: in vLLM 0.30 it is `class_path`, not a
# device, so passing "ASCEND" here would register the string "ASCEND" as this
# backend's import path and skip the decoration entirely.
@register_backend(AttentionBackendEnum.TURBOQUANT)
class AscendTurboQuantDryRunBackend(AscendAttentionBackend):
    """Backend descriptor for the TurboQuant dry run.

    Everything but the cache geometry and the implementation class is inherited:
    the metadata builder, the block-size contract and the block copy/swap
    helpers are the stock ones, which is the point -- a dry run that needed its
    own scheduler plumbing would not be testing the integration.
    """

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # Must be an AttentionBackendEnum member name: vLLM indexes the enum
        # with this string, and the lookup raises on anything else.
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type[AscendTurboQuantDryRunBackendImpl]:
        return AscendTurboQuantDryRunBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        """The TurboQuant builder, which owns the fused decode's launch buffers.

        Not the shared builder any more: three of the decode's operands and its
        tiling descriptors are per-batch quantities that used to be rebuilt per
        layer inside the launch. See
        :class:`AscendTurboQuantMetadataBuilder`.
        """
        return AscendTurboQuantMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        # Prefill context parallel gathers dense K/V before the cache write; the
        # packed planes are outside that feature matrix.
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        """``(2, num_blocks, block_size, num_kv_heads, head_size // 2)`` INT8.

        Two signed INT4 codes per byte, token-major inside a ``(page, head)``
        panel. The fp32 scale plane is *not* part of this shape: vLLM's KV cache
        spec describes one shape per layer, so the third plane is carved from
        the same page budget by the model runner
        (``AscendTurboQuantAttentionSpec`` and
        :meth:`NPUModelRunner._carve_turboquant_planes`) and shaped by
        :func:`turboquant_scale_cache_shape`. When it is absent the backend
        still falls back to a private scale plane, see
        :meth:`AscendTurboQuantDryRunBackendImpl._ensure_scale_cache`.
        """
        return turboquant_packed_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_scale_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int) -> tuple[int, ...]:
        return turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)

    # The block movers below exist because the scale plane is plane-major:
    # ``(2, num_blocks, num_kv_heads, block_size)``, so its block dimension is
    # 1 and not 0. The stock movers index dimension 0 of every tensor in the
    # layer's tuple, which for this plane would move *planes* instead of blocks
    # -- a copy that silently leaves a page's scales behind, which turns a
    # preemption or a prefix-cache copy into wrong numbers rather than an
    # error. All three planes move together here.
    _SCALE_PLANE_INDEX: int = 2

    @classmethod
    def swap_blocks(
        cls,
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        for index, (src_cache, dst_cache) in enumerate(zip(src_kv_cache, dst_kv_cache)):
            if index == cls._SCALE_PLANE_INDEX:
                dst_cache[:, dst_indices] = src_cache[:, src_indices].to(dst_cache.device)
            else:
                dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

    @classmethod
    def copy_blocks(
        cls,
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]
        for kv_cache in kv_caches:
            for index, cache in enumerate(kv_cache):
                if index == cls._SCALE_PLANE_INDEX:
                    cache[:, dst_indices] = cache[:, src_indices]
                else:
                    cache[dst_indices] = cache[src_indices]


# ---------------------------------------------------------------------------
# In-code profiling of the dry-run forward.
#
# Switched on by TURBOQUANT_PROFILE=1 and off otherwise, at the cost of one
# bool test per forward.
#
# This *exports* a trace; it does not summarise one. torch_npu's profiler has
# no in-memory aggregation -- `prof.key_averages()` does not exist on it, which
# is where Kineto's in-process event tree would have been -- because the CANN
# profiler stages its records on disk and parses them out of band. So the only
# way to read a profiled forward is to give the profiler a trace handler and
# open what it writes offline: Ascend Insight, or chrome://tracing on the
# exported JSON.
#
# This is a measurement harness, not a serving configuration. A profiler
# start/stop per layer per step costs far more than anything it measures, and
# the artefacts accumulate under the trace directory for as long as the server
# runs.
# ---------------------------------------------------------------------------

TURBOQUANT_PROFILE_ENV: Final[str] = "TURBOQUANT_PROFILE"

_PROFILE_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

# Where ``tensorboard_trace_handler`` writes. Relative, so for a served model
# it lands wherever the server process was started.
TURBOQUANT_PROFILE_TRACE_DIR: Final[str] = "./turboquant_traces"

# Logged once, on the first forward that is actually profiled.
TURBOQUANT_PROFILE_NOTICE: Final[str] = (
    "TurboQuant profiling active. Ascend CANN does not support in-memory key_averages. "
    "Traces are being dumped to %s for offline analysis in Ascend Insight or chrome://tracing."
)


def turboquant_profile_enabled() -> bool:
    """Whether ``TURBOQUANT_PROFILE`` asks for the in-code profiler."""
    return os.getenv(TURBOQUANT_PROFILE_ENV, "0").strip().lower() in _PROFILE_TRUTHY


def _stream_is_capturing() -> bool:
    """Whether an ACL graph capture is in progress on the current stream.

    Checked directly rather than through ``forward_context.capturing``, which
    stays set for the whole forward once the first piece has been captured and
    would switch profiling off for every TurboQuant call in a piecewise run --
    exactly the configuration this is meant to profile. Returns ``False`` on a
    driverless host, where the function exists and answers.
    """
    with suppress(AttributeError, RuntimeError):
        return bool(torch.npu.is_current_stream_capturing())  # type: ignore[attr-defined]
    return False


# ---------------------------------------------------------------------------
# The implementation.
# ---------------------------------------------------------------------------

# auto | v1 | v2. On Ascend 950 the V1..V4 FIA entry points are refused
# (EZ9903) and only npu_fused_infer_attention_score_v2 exists; auto discovers
# that from the first refusal instead of hardcoding a SoC test.
TURBOQUANT_FIA_VERSION_ENV: Final[str] = "TURBOQUANT_FIA_VERSION"


class AscendTurboQuantDryRunBackendImpl(AscendAttentionBackendImpl):
    """PyTorch dry run of the TurboQuant attention path.

    Ingestion (``reshape_and_cache``) is the kernels' contract exactly: rotate,
    quantise per token and head, pack to nibbles, scatter by slot. The attention
    itself (``forward_impl``) gathers those pages back, dequantises and
    un-rotates them, and hands dense K/V to
    ``npu_fused_infer_attention_score`` -- so the numbers carry the real
    quantisation error while the schedule is nothing like the kernel's.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        adapter: TurboQuantOpsAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        # The base __init__ validates kv_cache_dtype against the C8 static
        # quantisation recipe and raises for a non-float cache without one.
        # TurboQuant is dynamic per token and carries no recipe, so the base is
        # told the model dtype and the packed cache dtype is re-declared below.
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        self.requested_kv_cache_dtype: str = kv_cache_dtype
        self.model_dtype: torch.dtype = self.kv_cache_dtype
        self.kv_cache_dtype = torch.int8
        self.enable_c8_quant = False

        if head_size % TURBOQUANT_GROUP:
            raise ValueError(
                f"TurboQuant needs head_size to be a multiple of {TURBOQUANT_GROUP}, got {head_size}"
            )
        if alibi_slopes is not None:
            raise NotImplementedError("TurboQuant has no ALiBi path: the decode kernel takes no slope operand.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TurboQuant has no sliding-window path: the fused decode reads whole pages from the block table."
            )
        if sinks is not None:
            raise NotImplementedError(
                "learnable attention sinks need the kernel's softmax denominator as an out-tensor; "
                "the dequantising dry run does not expose one."
            )
        if self.use_bnsd_kv_cache:
            raise NotImplementedError(
                "the TurboQuant packed cache is token-major ND; VLLM_KV_CACHE_LAYOUT must not be HND or LBHNC."
            )

        self.adapter: TurboQuantOpsAdapter = adapter if adapter is not None else TurboQuantOpsAdapter()
        self._check_kernel_geometry()
        self.emulator: TurboQuantEmulator | None = None
        self.scale_cache: torch.Tensor | None = None
        self._use_fia_v2: bool | None = None
        self._warned_scale_carve = False
        self._profile: bool = turboquant_profile_enabled()
        # info_once caches on its arguments, so every one of them has to be
        # hashable: a list of stage names raises TypeError before it is logged.
        staged = ", ".join(sorted(stage.value for stage in self.adapter.config.kernel_stages))
        logger.info_once(
            "TurboQuant dry run active: packed INT4 KV cache, kernel stages=%s, ops namespace=%s",
            staged or "none (all golden)",
            self.adapter.config.ops_namespace,
        )

    # -- what the staged kernels can actually run --------------------------

    def _check_kernel_geometry(self) -> None:
        """Refuse a model shape a staged kernel cannot run, at layer construction.

        Only the stages that are switched on are checked, and only against their
        own kernel's specialisation: the golden path has none of these limits,
        so a dry run with no kernel stage is unaffected by any of it.

        The binding TORCH_CHECKs every one of these too, and that is the check
        that matters -- it is the one a direct caller cannot skip. This one
        exists because of *when* it fires. The binding's refusal arrives from
        inside an attention forward, one decode step into a server that has
        already loaded weights, sized a KV cache from
        :func:`turboquant_page_size_bytes` and warmed up; it names a tensor
        shape rather than the model. Raising here instead means an impossible
        combination is a startup error naming the stage that cannot run it.

        Deliberately NOT a warn-and-fall-back, which is how a missing
        registration is handled (:meth:`TurboQuantOpsAdapter._resolve_kernel`).
        A kernel that is absent has a golden model to fall back to and the run
        stays correct; a kernel that is present and mis-specialised writes one
        head's rows into another head's page with every canary intact, and the
        operand that would silently do it is exactly what this refuses.
        """
        config = self.adapter.config
        writers = (
            TurboQuantKernelStage.RESHAPE_AND_CACHE,
            TurboQuantKernelStage.SCATTER,
        )
        staged_writers = [stage for stage in writers if config.enables(stage)]
        decode_staged = config.enables(TurboQuantKernelStage.DECODE)
        if not staged_writers and not decode_staged:
            return

        staged = list(staged_writers)
        if decode_staged:
            staged.append(TurboQuantKernelStage.DECODE)
        named = ", ".join(sorted(stage.value for stage in staged))

        # `block_size` is deliberately not checked here: it is a property of the
        # cache the model runner allocates, not of the layer, and it is not
        # known yet. The binding checks it per call -- a multiple of 16 for
        # either writer, a power-of-two multiple of 64 for the decode -- and
        # :class:`TurboQuantValueLayout` records why 128 is the value that
        # satisfies all three.

        # Every one of the three kernels pins the head size: `kHeadSize` in both
        # writers (the scatter writer returns without writing anything when it
        # disagrees, which is worse than a fault) and `dim != kHeadSize` in the
        # decode's valid_launch.
        if self.head_size != TURBOQUANT_KERNEL_HEAD_SIZE:
            raise ValueError(
                f"TurboQuant kernel stages {named} are switched on, but all three kernels are "
                f"specialised to head_size {TURBOQUANT_KERNEL_HEAD_SIZE} and this model's is "
                f"{self.head_size}. The golden model has no such constraint: unset "
                f"{TURBOQUANT_STAGES_ENV} to run this shape."
            )

        # The two writers are unrolled over a fixed KV head count -- kChunks =
        # 2 * kHeads chunk phases by template argument -- and nothing in the
        # kernel checks the runtime argument against it.
        if staged_writers and self.num_kv_heads != TURBOQUANT_KERNEL_KV_HEADS:
            raise ValueError(
                f"TurboQuant ingestion stages {named} are switched on, but both cache writers are "
                f"specialised to exactly {TURBOQUANT_KERNEL_KV_HEADS} KV heads and this model has "
                f"{self.num_kv_heads}. The kernel does not check this: outside it the writer reads "
                f"one head's rows and writes another head's page, with every canary intact."
            )

        # The decode tolerates fewer heads than the writers need, but not more.
        if decode_staged and not 0 < self.num_kv_heads <= TURBOQUANT_DECODE_MAX_KV_HEADS:
            raise ValueError(
                f"the TurboQuant fused decode refuses num_kv_heads outside 1.."
                f"{TURBOQUANT_DECODE_MAX_KV_HEADS} (valid_launch), and this model has "
                f"{self.num_kv_heads}. This is the constraint a GQA model is most likely to hit; "
                f"unset {TURBOQUANT_STAGES_ENV} or drop the 'decode' stage to run golden."
            )
        if decode_staged and self.num_heads % max(self.num_kv_heads, 1):
            raise ValueError(
                f"the TurboQuant fused decode needs num_heads ({self.num_heads}) to be a multiple "
                f"of num_kv_heads ({self.num_kv_heads})."
            )

        # One pair of bf16 instantiations, and the dispatch returns without
        # launching for anything else -- see TURBOQUANT_KERNEL_DTYPE.
        if self.model_dtype != TURBOQUANT_KERNEL_DTYPE:
            raise ValueError(
                f"TurboQuant kernel stages {named} are switched on, but every kernel is "
                f"{TURBOQUANT_KERNEL_DTYPE} only and this model runs in {self.model_dtype}. The "
                f"fused decode's dispatch returns WITHOUT LAUNCHING for any other element type, "
                f"which would leave the attention output unwritten and report success, so this is "
                f"refused rather than attempted."
            )

    # -- static per-layer state -------------------------------------------

    def _ensure_statics(self, layer: AttentionLayer, device: torch.device) -> TurboQuantEmulator:
        """Bind the rotation's constants, once per layer.

        They live on the layer rather than on ``self`` so that a kernel-backed
        stage and its golden model read the same objects, and so that a layer dump
        carries the rotation that produced its cache.
        """
        emulator = self.adapter.emulator_for(self.head_size, device)
        if self.emulator is not emulator:
            self.emulator = emulator
            layer._turboquant_pi_signs = emulator.pi_signs
            layer._turboquant_rotation = emulator.rotation
            layer._turboquant_codec_tables = emulator.codec_tables
            layer._turboquant_h16 = emulator.h16
            layer._turboquant_inv_sqrt_len = emulator.inv_sqrt_len
        return emulator

    def _ensure_scale_cache(self, kv_cache: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        """Bind the fp32 scale plane, preferring the allocator's third plane.

        A three-plane carve is the production arrangement: ``kv_cache`` arrives
        as ``[key_cache, value_cache, scale_cache]`` and nothing is allocated
        here. Until the KV cache spec grows that plane -- and on the hybrid
        allocation path it cannot yet, because Qwen3.5's full-attention layers
        share one flat buffer with its Gated DeltaNet layers -- the dry run
        allocates it itself, sized off the packed planes, and says so once.
        """
        if self.scale_cache is not None:
            return self.scale_cache
        if len(kv_cache) >= 3 and isinstance(kv_cache[2], torch.Tensor):
            scale_cache = kv_cache[2]
            expected = turboquant_scale_cache_shape(
                self.key_cache.shape[0], self.key_cache.shape[1], self.key_cache.shape[2]
            )
            if tuple(scale_cache.shape) != expected:
                raise ValueError(
                    f"the allocator's scale plane has shape {tuple(scale_cache.shape)}, expected {expected}"
                )
            if scale_cache.dtype is not torch.float32:
                raise ValueError(f"the scale plane must be fp32, got {scale_cache.dtype}")
            self.scale_cache = scale_cache
            return scale_cache

        num_blocks, block_size, num_kv_heads = self.key_cache.shape[:3]
        if not self._warned_scale_carve:
            self._warned_scale_carve = True
            logger.warning(
                "TurboQuant dry run is allocating its own fp32 scale plane %s (%.1f MiB per layer): the "
                "KV cache spec carved only two planes. The production path carves three. "
                "TODO: extend the Ascend KV cache spec so the allocator owns this plane.",
                turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads),
                TURBOQUANT_PLANES * num_blocks * num_kv_heads * block_size * 4 / 2**20,
            )
        self.scale_cache = torch.zeros(
            turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads),
            dtype=torch.float32,
            device=self.key_cache.device,
        )
        return self.scale_cache

    # -- ingestion ---------------------------------------------------------

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate, quantise and write this step's K/V into the packed cache.

        The signature is the base class's, including the four-tuple return, so
        the stock ``forward`` drives this without knowing it is quantising. The
        ingestion phase is picked by :meth:`_ingestion_phase`: the two kernels
        that own the write are different kernels, and only the prefill one may
        assume a contiguous page strip.
        """
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return query, key, value, output
        if kv_cache is None or len(kv_cache) < 2:
            return query, key, value, output

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            # A KV-sharing layer consumes another layer's cache; writing its own
            # K/V would overwrite shared slots.
            if self.is_kv_producer and attn_metadata.reshape_cache_event is not None:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return query, key, value, output

        emulator = self.emulator
        if emulator is None:
            raise RuntimeError("TurboQuant statics were never bound; forward() must run _ensure_statics first.")
        scale_cache = self._ensure_scale_cache(kv_cache)

        ingest = (
            self.adapter.reshape_and_cache_prefill
            if self._ingestion_phase(attn_metadata, num_actual_tokens) is TurboQuantIngestionPhase.PREFILL
            else self.adapter.reshape_scatter_decode
        )
        ingest(
            key=key[:num_actual_tokens].contiguous(),
            value=value[:num_actual_tokens].contiguous(),
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=scale_cache,
            slot_mapping=attn_metadata.slot_mapping[:num_actual_tokens].to(torch.int32),
            emulator=emulator,
        )
        notify_kv_cache_written()
        return query, key, value, output

    @staticmethod
    def _ingestion_phase(attn_metadata: AscendMetadata, num_actual_tokens: int) -> TurboQuantIngestionPhase:
        """Which writer owns this step: more tokens than requests means prefill.

        A decode step ingests exactly one token per request, so the token count
        equals the batch size; anything longer is carrying prefill. The batch
        size is read off ``seq_lens_list``, which is a host list on purpose (the
        metadata keeps it off the device). A metadata object that does not carry
        it, or carries ``None`` -- a dummy capture run, or a builder that has not
        filled it -- reads as zero requests and therefore as prefill, which is
        the branch that plans its own dispatch and so assumes nothing about the
        slots it is given: a dummy run's all-negative slot mapping reaches the
        scatter writer, which skips it, rather than the batch writer, which
        refuses it.
        """
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None)
        num_requests = len(seq_lens_list) if seq_lens_list is not None else 0
        if num_actual_tokens > num_requests:
            return TurboQuantIngestionPhase.PREFILL
        return TurboQuantIngestionPhase.DECODE

    # -- the cache read ----------------------------------------------------

    def _gather_dense_kv(
        self, attn_metadata: AscendMetadata, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense ``[B, S_pad, H, D]`` K and V for the batch, out of the packed cache.

        ``S_pad = max_blocks_per_seq * block_size``. Positions past a request's
        context length are zeroed rather than left alone: an unwritten page
        carries whatever the allocator last had there, and FIA honouring
        ``actual_seq_lengths_kv`` is not a reason to feed it a NaN.
        """
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )
        emulator = self.emulator
        assert emulator is not None

        block_tables = attn_metadata.block_tables[:batch_size]
        packed_key, key_scales, packed_value, value_scales = self.adapter.gather(
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=block_tables,
        )
        dense_key = self.adapter.dequantize_cache(packed_key, key_scales, emulator, dtype)
        dense_value = self.adapter.dequantize_cache(packed_value, value_scales, emulator, dtype)

        padded_len = dense_key.shape[1]
        positions = torch.arange(padded_len, device=dense_key.device, dtype=torch.int32)
        seq_lens = attn_metadata.seq_lens[:batch_size].to(device=dense_key.device, dtype=torch.int32)
        live = (positions.unsqueeze(0) < seq_lens.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        zero = torch.zeros((), dtype=dtype, device=dense_key.device)
        return torch.where(live, dense_key, zero), torch.where(live, dense_value, zero)

    # -- attention ---------------------------------------------------------

    def _fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_layout: str,
        actual_seq_lengths: Any,
        actual_seq_lengths_kv: Any,
        atten_mask: torch.Tensor | None,
        sparse_mode: int,
    ) -> torch.Tensor:
        """``npu_fused_infer_attention_score``, V1 or V2, on dense K/V.

        Ascend 950 refuses the V1..V4 entry points (EZ9903) and exposes only
        ``_v2``; ``TURBOQUANT_FIA_VERSION=auto`` learns which one works from the
        first refusal rather than from a SoC test, and remembers it.
        """
        if self._use_fia_v2 is None:
            requested = os.getenv(TURBOQUANT_FIA_VERSION_ENV, "auto").strip().lower()
            if requested == "v2":
                self._use_fia_v2 = True
            elif requested == "v1":
                self._use_fia_v2 = False
            else:
                self._use_fia_v2 = not hasattr(torch_npu, "npu_fused_infer_attention_score")

        if not self._use_fia_v2:
            try:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=atten_mask,
                    input_layout=input_layout,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=sparse_mode,
                )
                return attn_output
            except RuntimeError as refusal:
                logger.warning(
                    "npu_fused_infer_attention_score was refused (%s); retrying on _v2 for the rest of this run.",
                    refusal,
                )
                self._use_fia_v2 = True

        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            atten_mask=atten_mask,
            input_layout=input_layout,
            actual_seq_qlen=actual_seq_lengths,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            sparse_mode=sparse_mode,
        )
        return attn_output

    def _forward_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """DecodeOnly: the fused kernel, or gather the packed context and run FIA.

        Every tensor the fused launch takes is either a Tier-0 constant on the
        emulator or a Tier-1 buffer the metadata builder owns, and every scalar
        is configuration or geometry. Nothing here reads device memory on the
        host and nothing here allocates, which is what makes the launch legal on
        a capturing stream -- see ``TURBOQUANT_GRAPH_CAPTURE_RFC.md`` §3 and the
        gate in ``tests/ut/attention/test_turboquant_capture_contract.py``.

        ``context_lens`` is **not** ``attn_metadata.seq_lens``. That field is the
        CPU mirror (``_select_seq_lens`` prefers ``_seq_lens_cpu``), so passing
        it made the binding copy it to the device on every launch of every
        layer. The builder publishes ``turboquant_context_lens`` instead: the
        same lengths in a persistent device buffer, with the FIA padding rows
        zeroed rather than set to 1.

        The extent comes from the builder too, for the reason on
        :attr:`AscendTurboQuantMetadata.turboquant_num_tokens`: it has to be
        the same number the host filled descriptors for, and
        ``len(seq_lens_list)`` is a different quantity that merely happens to
        agree. It is also a Python ``len`` on a host list, which no
        ``TorchFunctionMode`` gate can see, so taking it out of the launch is
        a G10 improvement the allowlist could never have caught.
        """
        emulator = self.emulator
        assert emulator is not None
        batch_size = self._decode_num_tokens(attn_metadata)
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )

        tiling, workspace, query_rot = self._decode_buffers(attn_metadata, batch_size)
        fused = self.adapter.fused_decode(
            query=query[:batch_size],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=attn_metadata.block_tables[:batch_size],
            context_lens=self._decode_context_lens(attn_metadata, batch_size),
            emulator=emulator,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            output=output[:batch_size],
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
        )
        if fused is not None:
            return output

        dense_key, dense_value = self._gather_dense_kv(attn_metadata, batch_size, query.dtype)
        attn_output = self._fused_infer_attention(
            query=query[:batch_size].unsqueeze(1),
            key=dense_key,
            value=dense_value,
            input_layout="BSND",
            actual_seq_lengths=None,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            atten_mask=None,
            sparse_mode=0,
        )
        output[:batch_size] = attn_output.view(batch_size, self.num_heads, self.head_size)
        return output

    @staticmethod
    def _decode_num_tokens(attn_metadata: AscendMetadata) -> int:
        """The launch extent: the token count the builder filled buffers for.

        Falls back to ``len(seq_lens_list)`` only for a metadata object built by
        the base builder, which should not happen because the backend names its
        own. The two agree for a decode step -- ``build()`` pads the list and
        the tensor together -- but they are not the same quantity, and only one
        of them is the number the descriptors were written for.
        """
        tokens = int(getattr(attn_metadata, "turboquant_num_tokens", 0) or 0)
        if tokens:
            return tokens
        return len(attn_metadata.seq_lens_list)

    @staticmethod
    def _decode_context_lens(attn_metadata: AscendMetadata, batch_size: int) -> torch.Tensor:
        """The device context lengths for this step, zero-padded.

        Published by :class:`AscendTurboQuantMetadataBuilder`. The fallback to
        ``seq_lens_gpu`` covers a metadata object built by the base builder --
        which should not happen, since the backend names its own builder, but a
        device tensor is the right answer either way and the alternative is an
        ``AttributeError`` deep inside a launch. There is no fallback to
        ``seq_lens``: that is the CPU mirror and is what this change removes.
        """
        context_lens = getattr(attn_metadata, "turboquant_context_lens", None)
        if context_lens is None:
            context_lens = attn_metadata.seq_lens_gpu
        if context_lens is None:
            raise RuntimeError(
                "the TurboQuant decode has no device-side context lengths. "
                "AscendTurboQuantMetadataBuilder publishes turboquant_context_lens; "
                "a metadata object without it and without seq_lens_gpu cannot be decoded "
                "without a host-to-device copy inside the launch."
            )
        return context_lens[:batch_size]

    def _decode_buffers(
        self, attn_metadata: AscendMetadata, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(tiling, workspace, query_rot)`` for this step, narrowed to its extent.

        The builder sizes all three from ``max_num_reqs`` once, so a step uses a
        prefix of each. ``narrow`` keeps the base address, which is the whole
        point: the op's size checks want the exact extent while a captured graph
        wants the address to be the one it recorded, and a prefix view gives
        both.

        ``tiling`` is passed whole. The op checks it with ``>=`` because its
        exact size depends on the ``grid`` the op itself derives, and the
        descriptor array starts at word zero either way.
        """
        splits = TURBOQUANT_DECODE_SPLITS
        tiling = getattr(attn_metadata, "turboquant_tiling", None)
        workspace = getattr(attn_metadata, "turboquant_workspace", None)
        query_rot = getattr(attn_metadata, "turboquant_query_rot", None)
        if tiling is None or workspace is None or query_rot is None:
            raise RuntimeError(
                "the TurboQuant decode found no builder-owned launch buffers on its metadata "
                "(turboquant_tiling / turboquant_workspace / turboquant_query_rot). They are "
                "published by AscendTurboQuantMetadataBuilder; the operator cannot allocate them "
                "itself without giving every step a fresh address, which a captured graph cannot "
                "follow."
            )
        need_workspace = turboquant_decode_workspace_floats(batch_size, self.num_heads, self.head_size, splits)
        need_query_rot = turboquant_decode_query_rot_floats(batch_size, self.num_heads, self.head_size)
        if workspace.numel() < need_workspace or query_rot.numel() < need_query_rot:
            raise RuntimeError(
                f"the TurboQuant launch buffers are too small for {batch_size} tokens: workspace holds "
                f"{workspace.numel()} floats (needs {need_workspace}) and query_rot {query_rot.numel()} "
                f"(needs {need_query_rot}). The builder sizes them from max_num_reqs."
            )
        return tiling, workspace.narrow(0, 0, need_workspace), query_rot.narrow(0, 0, need_query_rot)

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """PrefillNoCache: attend over the dense K/V this step produced.

        Full precision on purpose. There is no context in the cache to read, the
        kernel path does the same (its prefill runs dense in TND), and a
        round trip through the codec here would only add error the kernel does
        not have.
        """
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        attn_output = self._fused_infer_attention(
            query=query[:num_tokens],
            key=key[:num_tokens].contiguous(),
            value=value[:num_tokens].contiguous(),
            input_layout="TND",
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
            atten_mask=attn_metadata.attn_mask,
            sparse_mode=3 if attn_metadata.causal else 0,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output

    def _forward_mixed(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Every state that reads context out of the cache with a query longer than one.

        ChunkedPrefill, PrefillCacheHit and SpecDecoding: pad the ragged query to
        ``[B, max_q, N, D]``, dequantise the whole context, and attend with an
        explicit causal mask aligned to each request's ``L_kv - L_q`` offset.

        NOTE: untested on device -- there is no NPU on the development host. The
        decode and ingestion paths are the ones the bring-up gates on; this one
        exists so a chunked-prefill run reaches an answer instead of an
        exception, and it should be read as such until it has run.
        """
        emulator = self.emulator
        assert emulator is not None

        query_lens_cpu = torch.tensor(attn_metadata.actual_seq_lengths_q, dtype=torch.int64)
        query_lens_cpu = torch.diff(query_lens_cpu, prepend=torch.zeros(1, dtype=torch.int64))
        batch_size = int(query_lens_cpu.numel())
        total_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
        max_query_len = int(query_lens_cpu.max())

        # Row -> (request, position in request), built on the host: CANN 9.2 has
        # no repeat_interleave kernel for this shape (aclnn 561103).
        request_of_row = torch.repeat_interleave(torch.arange(batch_size), query_lens_cpu)
        row_starts = torch.cumsum(query_lens_cpu, dim=0) - query_lens_cpu
        position_in_request = torch.arange(total_tokens) - row_starts[request_of_row]
        request_of_row = request_of_row.to(query.device)
        position_in_request = position_in_request.to(query.device)

        padded_query = torch.zeros(
            (batch_size, max_query_len, self.num_heads, self.head_size),
            dtype=query.dtype,
            device=query.device,
        )
        padded_query[request_of_row, position_in_request] = query[:total_tokens]

        dense_key, dense_value = self._gather_dense_kv(attn_metadata, batch_size, query.dtype)
        padded_kv_len = dense_key.shape[1]

        # True == masked. Query position j of a request whose context is L_kv
        # and whose new slice is L_q may see keys up to L_kv - L_q + j.
        seq_lens = attn_metadata.seq_lens[:batch_size].to(device=query.device, dtype=torch.int64)
        query_lens = query_lens_cpu.to(query.device)
        key_positions = torch.arange(padded_kv_len, device=query.device, dtype=torch.int64)
        query_positions = torch.arange(max_query_len, device=query.device, dtype=torch.int64)
        horizon = (seq_lens - query_lens).unsqueeze(1) + query_positions.unsqueeze(0)
        # [B, 1, max_q, max_kv], the head axis broadcast.
        atten_mask = (key_positions.view(1, 1, -1) > horizon.unsqueeze(-1)).unsqueeze(1)

        attn_output = self._fused_infer_attention(
            query=padded_query,
            key=dense_key,
            value=dense_value,
            input_layout="BSND",
            actual_seq_lengths=query_lens_cpu.tolist(),
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            atten_mask=atten_mask,
            sparse_mode=0,
        )
        attn_output = attn_output.view(batch_size, max_query_len, self.num_heads, self.head_size)
        output[:total_tokens] = attn_output[request_of_row, position_in_request]
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Route by attention state. The base class's FIA entry points are not
        reachable from here: they would read the packed planes as dense K/V."""
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            return self._forward_prefill_no_cache(query, key, value, attn_metadata, output)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_decode(query, attn_metadata, output)
        return self._forward_mixed(query, attn_metadata, output)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bind the rotation's constants, then run the stock forward.

        The base class owns the order (cache write, then attention), the padding
        contract and the output buffer; only the two methods above change.

        With ``TURBOQUANT_PROFILE=1`` the stock forward runs inside
        ``torch_npu.profiler``, which exports its trace to
        :data:`TURBOQUANT_PROFILE_TRACE_DIR` for offline analysis. Not profiled:
        a call with no ``attn_metadata`` (a dummy/warm-up forward, where there is
        no attention to measure) and a call made during an ACL graph capture,
        where starting a profiler would be measuring the capture and risking it.
        """
        if attn_metadata is not None:
            self._ensure_statics(layer, query.device)
        call: dict[str, Any] = {
            "layer": layer,
            "query": query,
            "key": key,
            "value": value,
            "kv_cache": kv_cache,
            "attn_metadata": attn_metadata,
            "output": output,
            "output_scale": output_scale,
            "output_block_scale": output_block_scale,
        }
        if not self._profile or attn_metadata is None or _stream_is_capturing():
            return super().forward(**call)
        return self._profiled_forward(call)

    # -- profiling ---------------------------------------------------------

    def _profiled_forward(self, call: dict[str, Any]) -> torch.Tensor:
        """Run the stock forward under the profiler and let it export its trace.

        Nothing is read back. The trace handler owns the records, and
        ``prof.step()`` inside the context is what tells it a step is complete,
        so it chunks and flushes one artefact set per profiled forward instead
        of one for the whole run.

        There is no fallback around this: with the handler doing the writing
        there is no in-memory call left to fail on a torch_npu that lacks one,
        and a profiler that cannot even be constructed has already switched
        itself off in :meth:`_open_profiler`.
        """
        profiler = self._open_profiler()
        if profiler is None:
            return super().forward(**call)
        logger.info_once(TURBOQUANT_PROFILE_NOTICE, TURBOQUANT_PROFILE_TRACE_DIR)
        with profiler as prof:
            attn_output = super().forward(**call)
            prof.step()
        return attn_output

    def _open_profiler(self) -> Any | None:
        """A CPU+NPU profiler that exports its trace, or ``None`` if unavailable.

        ``on_trace_ready`` is the whole of it: torch_npu has no in-memory
        ``key_averages()``, so a profiler with no handler records a step and
        then has nothing any caller can read. ``tensorboard_trace_handler`` is
        what writes the CANN artefacts and the chrome-trace JSON, and its
        absence counts as the profiler being unavailable -- a build without it
        cannot be read at all.

        Everything else is left at its default: op names and device times are
        the default profiler level, and every knob left alone is one fewer
        version-dependent keyword.
        """
        profiler_module = getattr(torch_npu, "profiler", None)
        activity = getattr(profiler_module, "ProfilerActivity", None)
        trace_handler = getattr(profiler_module, "tensorboard_trace_handler", None)
        if profiler_module is None or activity is None or trace_handler is None:
            self._disable_profiling("torch_npu.profiler is not available in this build")
            return None
        if not hasattr(profiler_module, "profile"):
            self._disable_profiling("torch_npu.profiler has no profile()")
            return None
        try:
            return profiler_module.profile(
                activities=[activity.CPU, activity.NPU],
                on_trace_ready=trace_handler(TURBOQUANT_PROFILE_TRACE_DIR),
                record_shapes=False,
                with_stack=False,
            )
        except Exception as error:
            self._disable_profiling(error)
            return None

    def _disable_profiling(self, reason: object) -> None:
        self._profile = False
        logger.warning(
            "TurboQuant in-code profiling is switching itself off for this layer (%s stays set, "
            "the attention path is unaffected): %s",
            TURBOQUANT_PROFILE_ENV,
            reason,
        )
