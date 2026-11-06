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

It is deliberately **not** a monolith. Every stage goes through
:class:`TurboQuantOpsAdapter`, which per stage either calls the C++ binding
(once it exists and the stage is switched on) or falls back to
:class:`TurboQuantEmulator`. Progressive replacement is therefore one stage at
a time, with the remaining stages still emulated -- which is also how a newly
landed kernel gets differentially diffed against its own model.

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
* The scale plane is modelled at the staged kernels' layout
  (``(2, num_blocks, num_kv_heads, block_size)`` fp32), not the burst-padded
  per-token slot the AIV scatter writer uses.
* Nothing here is a performance model. A dry run is for routing, shapes,
  metadata and accuracy, never for latency.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
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

# torch.ops namespace the kernels register into once their TORCH_LIBRARY
# fragment lands. The in-tree AscendC operators use "_C_ascend"; the staging
# subtree registers nothing yet (op_adapter/turboquant_torch_ops.h is a symbol
# inventory), so the name is configurable rather than hardcoded.
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
    """Nibble order of the value plane inside one ``(page, head)`` panel.

    ``TOKEN_MAJOR``   -- nibble index ``token * head_size + channel``, the same
    as the key plane. A token's nibbles are then a contiguous run that a
    per-token scatter can write without touching another token's bytes.

    ``CHANNEL_MAJOR`` -- nibble index ``channel * block_size + token``, the
    harness's BF16 signed-int4 strip ABI (``tq_oracle::nibble_index``). It pairs
    token ``2t`` with token ``2t+1`` inside one byte, so a per-token write is a
    read-modify-write and only a whole-page prefill strip can be written without
    one. The PyTorch fallback refuses it for that reason -- the RMW belongs to
    the kernel, which owns the page.
    """

    TOKEN_MAJOR = "token_major"
    CHANNEL_MAJOR = "channel_major"


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
        :class:`TurboQuantKernelStage` values, or ``all``, or ``none`` -- the
        default, because no kernel is registered yet, so everything is emulated.
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
# The adapter: one seam per kernel.
# ---------------------------------------------------------------------------


class TurboQuantOpsAdapter:
    """Per-stage dispatch between the C++ kernels and the PyTorch fallback.

    Every stage of the TurboQuant path is a method here, and every method has
    the same shape: look the kernel up for this stage; if it is switched on and
    registered, call it; otherwise run the emulation. A kernel is therefore
    adopted by flipping one stage in :class:`TurboQuantOpsConfig` -- no backend
    code changes -- and a suspect kernel is bisected by flipping it back.

    The expected kernel signatures below are the staging subtree's C ABIs
    (``op_adapter/turboquant_torch_adpt.h``) lifted to Torch schemas. They are
    provisional until the ``TORCH_LIBRARY_FRAGMENT`` lands, which is what every
    ``TODO: Replace with C++ binding`` marks.
    """

    def __init__(
        self,
        config: TurboQuantOpsConfig | None = None,
    ) -> None:
        self.config: TurboQuantOpsConfig = config if config is not None else TurboQuantOpsConfig.from_env()
        self._emulators: dict[tuple[int, str], TurboQuantEmulator] = {}
        self._missing_kernels: set[str] = set()

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

    def _kernel(self, stage: TurboQuantKernelStage, name: str) -> Callable[..., Any] | None:
        """The registered kernel for ``stage``, or ``None`` to emulate it.

        A stage switched on whose kernel is not registered is a configuration
        error the caller can survive, so it warns once and emulates rather than
        raising: a dry run that dies on a missing kernel tells you nothing you
        did not already know.
        """
        if not self.config.enables(stage):
            return None
        namespace = getattr(torch.ops, self.config.ops_namespace, None)
        kernel = getattr(namespace, name, None) if namespace is not None else None
        if kernel is None:
            if name not in self._missing_kernels:
                self._missing_kernels.add(name)
                logger.warning(
                    "TurboQuant stage %s is switched on but torch.ops.%s.%s is not registered; "
                    "falling back to the PyTorch emulation.",
                    stage.value,
                    self.config.ops_namespace,
                    name,
                )
            return None
        return kernel

    # -- stage: rotation ---------------------------------------------------

    def rotate(self, x: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """Ingress rotation of an activation panel: ``[..., D] -> [..., D]`` fp32."""
        kernel = self._kernel(TurboQuantKernelStage.ROTATE, "npu_turboquant_rotate")
        if kernel is not None:
            # TODO: Replace with C++ binding once the staging subtree registers
            # its Torch schema. Expected:
            #   npu_turboquant_rotate(Tensor x, Tensor pi_signs, Tensor rot_tables,
            #                         Tensor h16, int direction, float scale) -> Tensor
            # direction 0 = Egress (Pi^T), 1 = Ingress (Pi); see
            # rot_oracle::Direction in tests/common/rotate_butterfly_oracle.hpp.
            return kernel(x, emulator.pi_signs, direction=1, scale=1.0)
        return emulator.rotate(x)

    def inverse_rotate(self, y: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """Egress rotation, ``Pi^T y / sqrt(D)``: the un-rotation of a dequantised panel."""
        kernel = self._kernel(TurboQuantKernelStage.ROTATE, "npu_turboquant_rotate")
        if kernel is not None:
            # TODO: Replace with C++ binding (same schema, direction 0). The
            # kernel carries 1/sqrt(D) in its scale argument.
            return kernel(y, emulator.pi_signs, direction=0, scale=emulator.inv_sqrt_len)
        return emulator.inverse_rotate(y)

    def rotate_query(self, query: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        """Rotate the decode query into the cache's basis, fp32.

        Unused by the dequantising dry-run decode, which stays in the model's
        basis. It exists because the fused decode kernel takes a *raw* query and
        rotates it in its prologue, so a staged bring-up wants the two bases
        side by side: ``Pi q . Pi k == D q . k``.
        """
        kernel = self._kernel(TurboQuantKernelStage.ROTATE, "npu_turboquant_rotate_q")
        if kernel is not None:
            # TODO: Replace with C++ binding. Expected:
            #   npu_turboquant_rotate_q(Tensor query, Tensor pi_signs, Tensor rot_tables,
            #                           Tensor h16) -> Tensor   # fp32 out
            return kernel(query, emulator.pi_signs)
        return emulator.rotate(query)

    # -- stage: quantisation ----------------------------------------------

    def quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[..., D]`` rotated fp32 -> ``([..., D/2]`` INT8 bytes, ``[...]`` fp32 scales)."""
        kernel = self._kernel(TurboQuantKernelStage.QUANTIZE, "npu_turboquant_quantize_pack")
        if kernel is not None:
            # TODO: Replace with C++ binding. Expected:
            #   npu_turboquant_quantize_pack(Tensor rotated, float inv_sqrt_len)
            #       -> (Tensor packed_int8, Tensor scales_fp32)
            return kernel(rotated, emulator.inv_sqrt_len)
        codes, scales = emulator.quantize(rotated)
        return emulator.pack(codes), scales

    def unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        """``([..., D/2]`` bytes, ``[...]`` scales) -> ``[..., D]`` fp32, still rotated."""
        kernel = self._kernel(TurboQuantKernelStage.DEQUANTIZE, "npu_turboquant_unpack_dequant")
        if kernel is not None:
            # TODO: Replace with C++ binding. Expected:
            #   npu_turboquant_unpack_dequant(Tensor packed, Tensor scales) -> Tensor
            return kernel(packed, scales)
        return emulator.dequantize(emulator.unpack(packed), scales)

    # -- stage: the cache write -------------------------------------------

    def scatter(
        self,
        packed_key: torch.Tensor,
        key_scales: torch.Tensor,
        packed_value: torch.Tensor,
        value_scales: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter packed nibbles and fp32 scales into the physical block cache.

        ``packed_*`` are ``[T, H, D/2]`` INT8, ``*_scales`` ``[T, H]`` fp32, and
        ``slot_mapping`` ``[T]`` int32 of ``page * block_size + token``. Negative
        slots are padding and are skipped, which is what a graph-capture dummy
        run feeds.
        """
        kernel = self._kernel(TurboQuantKernelStage.SCATTER, "npu_turboquant_reshape_scatter_aiv")
        if kernel is not None:
            # TODO: Replace with C++ binding. The staged C launcher is
            #   launch_turboquant_reshape_scatter_aiv(stream, block_dim, key, value,
            #       key_cache, value_cache, scale_cache, slot_mapping, pi_signs,
            #       cu_seqlens, launch_probe, num_requests, num_kv_heads, head_size,
            #       block_size, num_blocks, item_begin, item_count, inv_sqrt_len)
            # i.e. the kernel rotates and quantises inside the scatter; this seam
            # therefore disappears into reshape_and_cache() once it lands.
            kernel(
                packed_key,
                key_scales,
                packed_value,
                value_scales,
                key_cache,
                value_cache,
                scale_cache,
                slot_mapping,
            )
            return

        if self.config.value_layout is TurboQuantValueLayout.CHANNEL_MAJOR:
            # Checked before the key plane is touched: a half-written step is
            # worse than a refused one.
            raise NotImplementedError(
                "the channel-major value plane pairs tokens 2t and 2t+1 inside one byte, so a per-token "
                "scatter needs a read-modify-write that belongs to the kernel; run the dry run with "
                "TurboQuantValueLayout.TOKEN_MAJOR"
            )

        block_size = key_cache.shape[1]
        slots = slot_mapping.to(torch.int64)
        valid = slots >= 0
        if not bool(valid.all()):
            slots = slots[valid]
            packed_key = packed_key[valid]
            packed_value = packed_value[valid]
            key_scales = key_scales[valid]
            value_scales = value_scales[valid]
        if slots.numel() == 0:
            return

        pages = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots - pages * block_size
        for plane, packed, scales, cache in (
            (TURBOQUANT_KEY_PLANE, packed_key, key_scales, key_cache),
            (TURBOQUANT_VALUE_PLANE, packed_value, value_scales, value_cache),
        ):
            _flat_slot_view(cache)[slots] = packed
            # scale_cache[plane] is (num_blocks, num_kv_heads, block_size), so
            # the pair of index tensors lands one (page, token) column per token.
            scale_cache[plane][pages, :, offsets] = scales.to(scale_cache.dtype)

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        """The fused ingestion: rotate, quantise, pack and scatter ``[T, H, D]`` K/V.

        This is the one entry point the production kernel occupies as a whole;
        the composition below is what it replaces, stage by stage.
        """
        kernel = self._kernel(TurboQuantKernelStage.RESHAPE_AND_CACHE, "npu_turboquant_reshape_and_cache_a8")
        if kernel is not None:
            # TODO: Replace with C++ binding. Expected (the task's named entry,
            # over the staged launch_turboquant_reshape_and_cache_bf16 ABI):
            #   npu_turboquant_reshape_and_cache_a8(Tensor key, Tensor value,
            #       Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor(c!) scale_cache,
            #       Tensor slot_mapping, Tensor pi_signs, Tensor tables) -> ()
            # The staged writer additionally takes tokens_per_core and
            # inv_sqrt_len, and requires slot_mapping[i] == slot_mapping[0] + i
            # with slot_mapping[0] a multiple of block_size: one strip fills one
            # whole physical page.
            kernel(
                key,
                value,
                key_cache,
                value_cache,
                scale_cache,
                slot_mapping,
                emulator.pi_signs,
                _codec_tables(emulator),
            )
            return

        packed_key, key_scales = self.quantize_and_pack(self.rotate(key, emulator), emulator)
        packed_value, value_scales = self.quantize_and_pack(self.rotate(value, emulator), emulator)
        self.scatter(
            packed_key,
            key_scales,
            packed_value,
            value_scales,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
        )

    # -- stage: the cache read --------------------------------------------

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
        """
        kernel = self._kernel(TurboQuantKernelStage.GATHER, "npu_turboquant_gather_pages")
        if kernel is not None:
            # TODO: Replace with C++ binding. Expected:
            #   npu_turboquant_gather_pages(Tensor key_cache, Tensor value_cache,
            #       Tensor scale_cache, Tensor block_tables)
            #       -> (Tensor, Tensor, Tensor, Tensor)
            # A production decode never materialises this -- the fused decode
            # streams pages through UB instead (see fused_decode()). It exists so
            # a gather kernel can be validated against the model in isolation.
            return kernel(key_cache, value_cache, scale_cache, block_tables)

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

    def dequantize_cache(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        emulator: TurboQuantEmulator,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gathered packed pages -> dense activations in ``dtype``.

        Unpack, dequantise, un-rotate: the composition that takes a cache panel
        back to the basis the model's own K and V live in.
        """
        rotated = self.unpack_and_dequantize(packed, scales, emulator)
        return self.inverse_rotate(rotated, emulator).to(dtype)

    # -- stage: the fused decode ------------------------------------------

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
    ) -> torch.Tensor | None:
        """The one-launch decode, or ``None`` when it is not available.

        ``None`` is the dry run's signal to gather, dequantise and call FIA
        instead -- the fallback is a whole attention, not a single operation, so
        it lives in the backend rather than here.
        """
        kernel = self._kernel(TurboQuantKernelStage.DECODE, "npu_turboquant_fused_decode")
        if kernel is None:
            return None
        # TODO: Replace with C++ binding. The staged launcher is
        #   turboquant_mm_fused_decode_opt_raw_query_impl(mode, type, stream, block_dim,
        #       query, pi_signs, rot_tables, h16, gate, query_rot, key_cache, value_cache,
        #       scale_cache, block_tables, context_lens, mode_tables, workspace, output,
        #       tiling, num_tokens, num_heads, num_kv_heads, head_size, block_size,
        #       max_blocks_per_seq, num_splits, heads_per_task, tasks_per_block,
        #       reduce_tasks_per_block, prologue_vectors_per_block,
        #       prologue_cube_chunk_vectors, output_stage, fused_context_limit, scale,
        #       inv_sqrt_len)
        # The host tiling descriptors (include/turboquant_fused_decode.hpp) and the
        # split workspace are the caller's to build and upload; the launcher
        # refuses the launch without them, so this seam grows a tiling helper
        # when the schema lands rather than passing shapes alone.
        kernel(
            query,
            emulator.pi_signs,
            key_cache,
            value_cache,
            scale_cache,
            block_tables,
            context_lens,
            num_heads,
            num_kv_heads,
            scale,
            emulator.inv_sqrt_len,
            output,
        )
        return output


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


def _codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    """The Cube's resident rotation operand, ``2 * (D/32)`` row-major 32x32 INT8 blocks.

    ``tables[s * groups + g][c0][i] = (s ? -1 : +1) * pi_signs[g*32 + c0] * H32[c0][i]``:
    the sign diagonal and the inner Hadamard factor, pre-composed, with the
    outer factor supplied by which block the kernel picks
    (``parity(c1 & g) * groups + g``). Built here so the dry run hands the
    kernel-backed stages exactly the operand ``tq_oracle::build_tables`` audits.
    """
    device = emulator.device
    h_inner = _sylvester_hadamard(emulator.group, device, torch.float32)
    signs = emulator.pi_signs.reshape(emulator.groups, emulator.group)
    # [groups, 32, 32]: signs broadcast over the inner factor's rows.
    positive = signs.unsqueeze(-1) * h_inner.unsqueeze(0)
    tables = torch.stack((positive, -positive), dim=0)
    return tables.reshape(TURBOQUANT_PLANES * emulator.groups, emulator.group, emulator.group).to(torch.int8)


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
        return AscendAttentionMetadataBuilder

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
        self.emulator: TurboQuantEmulator | None = None
        self.scale_cache: torch.Tensor | None = None
        self._use_fia_v2: bool | None = None
        self._warned_scale_carve = False
        # info_once caches on its arguments, so every one of them has to be
        # hashable: a list of stage names raises TypeError before it is logged.
        staged = ", ".join(sorted(stage.value for stage in self.adapter.config.kernel_stages))
        logger.info_once(
            "TurboQuant dry run active: packed INT4 KV cache, kernel stages=%s, ops namespace=%s",
            staged or "none (all emulated)",
            self.adapter.config.ops_namespace,
        )

    # -- static per-layer state -------------------------------------------

    def _ensure_statics(self, layer: AttentionLayer, device: torch.device) -> TurboQuantEmulator:
        """Bind the rotation's constants, once per layer.

        They live on the layer rather than on ``self`` so that a kernel-backed
        stage and its emulation read the same objects, and so that a layer dump
        carries the rotation that produced its cache.
        """
        emulator = self.adapter.emulator_for(self.head_size, device)
        if self.emulator is not emulator:
            self.emulator = emulator
            layer._turboquant_pi_signs = emulator.pi_signs
            layer._turboquant_rotation = emulator.rotation
            layer._turboquant_codec_tables = _codec_tables(emulator)
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
        """Rotate, quantise and scatter this step's K/V into the packed cache.

        The signature is the base class's, including the four-tuple return, so
        the stock ``forward`` drives this without knowing it is quantising.
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

        self.adapter.reshape_and_cache(
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
        """DecodeOnly: gather the packed context, dequantise it, run FIA."""
        emulator = self.emulator
        assert emulator is not None
        batch_size = len(attn_metadata.seq_lens_list)
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )

        fused = self.adapter.fused_decode(
            query=query[:batch_size],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=attn_metadata.block_tables[:batch_size],
            context_lens=attn_metadata.seq_lens[:batch_size],
            emulator=emulator,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            output=output[:batch_size],
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
        """
        if attn_metadata is not None:
            self._ensure_statics(layer, query.device)
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
