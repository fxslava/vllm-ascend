from __future__ import annotations

import importlib
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
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_ascend import envs
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

TURBOQUANT_PI_SEED: Final[int] = 0x5F3759DF

_LCG_MULTIPLIER: Final[int] = 1664525

_LCG_INCREMENT: Final[int] = 1013904223

_LCG_MODULUS_MASK: Final[int] = 0xFFFFFFFF

_LCG_SIGN_BIT: Final[int] = 16

TURBOQUANT_NIBBLE_MIN: Final[int] = -8

TURBOQUANT_NIBBLE_MAX: Final[int] = 7

TURBOQUANT_NIBBLE_MASK: Final[int] = 0xF

TURBOQUANT_NIBBLE_LEVELS: Final[int] = 16

TURBOQUANT_STRIP_TOKENS: Final[int] = 16

TURBOQUANT_NZ_TILE_ROWS: Final[int] = 64

TURBOQUANT_NZ_OPERAND_C0: Final[int] = 32

TURBOQUANT_NZ_FP32_PER_BLOCK: Final[int] = 8

TURBOQUANT_HEADROOM_BITS: Final[int] = 2

TURBOQUANT_LM_CENTROIDS: Final[tuple[float, ...]] = (
    -2.75,
    -2.0,
    -1.625,
    -1.25,
    -0.9375,
    -0.6875,
    -0.375,
    -0.125,
    +0.125,
    +0.375,
    +0.6875,
    +0.9375,
    +1.25,
    +1.625,
    +2.0,
    +2.75,
)

TURBOQUANT_LM_THRESHOLDS: Final[tuple[float, ...]] = (
    -2.375,
    -1.8125,
    -1.4375,
    -1.09375,
    -0.8125,
    -0.53125,
    -0.25,
    0.0,
    +0.25,
    +0.53125,
    +0.8125,
    +1.09375,
    +1.4375,
    +1.8125,
    +2.375,
)

TURBOQUANT_LM_EPS: Final[float] = 1e-20

TURBOQUANT_QUANT_MODE: Final[int] = int(envs.VLLM_ASCEND_TURBOQUANT_QUANT_MODE)

TURBOQUANT_DECODE_UNPACK: Final[int] = 3 if TURBOQUANT_QUANT_MODE == 1 else 0

_FP32_MANTISSA_BITS: Final[int] = 23

_FP32_EXPONENT_MASK: Final[int] = 0xFF

_FP32_EXPONENT_BIAS: Final[int] = 127

_FP32_MAX_BIASED_EXPONENT: Final[int] = 254

TURBOQUANT_STAGES_ENV: Final[str] = "TURBOQUANT_KERNEL_STAGES"

TURBOQUANT_NAMESPACE_ENV: Final[str] = "TURBOQUANT_OPS_NAMESPACE"

TURBOQUANT_DEFAULT_NAMESPACE: Final[str] = "turboquant_ops"


class TurboQuantKernelStage(str, Enum):
    ROTATE = "rotate"
    QUANTIZE = "quantize"
    SCATTER = "scatter"
    GATHER = "gather"
    DEQUANTIZE = "dequantize"
    RESHAPE_AND_CACHE = "reshape_and_cache"
    DECODE = "decode"


class TurboQuantValueLayout(str, Enum):
    TOKEN_MAJOR = "token_major"
    CHANNEL_MAJOR = "channel_major"
    NZ_TILED = "nz_tiled"


class TurboQuantIngestionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class TurboQuantOpsConfig:
    ops_namespace: str = TURBOQUANT_DEFAULT_NAMESPACE
    kernel_stages: frozenset[TurboQuantKernelStage] = frozenset()
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.TOKEN_MAJOR

    @classmethod
    def from_env(cls) -> TurboQuantOpsConfig:
        stage_selection = os.getenv(TURBOQUANT_STAGES_ENV, "none").strip().lower()
        if stage_selection in ("", "none", "0", "off"):
            kernel_stages: frozenset[TurboQuantKernelStage] = frozenset()
        elif stage_selection in ("all", "1", "on"):
            kernel_stages = frozenset(TurboQuantKernelStage)
        else:
            kernel_stages = cls._parse_stages(stage_selection)
        return cls(
            ops_namespace=os.getenv(TURBOQUANT_NAMESPACE_ENV, TURBOQUANT_DEFAULT_NAMESPACE),
            kernel_stages=kernel_stages,
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


def _exp2_fp32(exponent: torch.Tensor) -> torch.Tensor:
    biased = torch.clamp(exponent.to(torch.int32) + _FP32_EXPONENT_BIAS, 1, _FP32_MAX_BIASED_EXPONENT)
    return (biased << _FP32_MANTISSA_BITS).contiguous().view(torch.float32)


def _sylvester_hadamard(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    index = torch.arange(order, device=device, dtype=torch.int64)
    common = index.unsqueeze(1) & index.unsqueeze(0)
    parity = torch.zeros_like(common)
    for bit in range(order.bit_length()):
        parity = parity ^ ((common >> bit) & 1)
    return torch.where(parity.bool(), -1.0, 1.0).to(dtype)


def turboquant_pi_signs(head_size: int, device: torch.device) -> torch.Tensor:
    state = (TURBOQUANT_PI_SEED + head_size) & _LCG_MODULUS_MASK
    bits: list[float] = []
    for _ in range(head_size):
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & _LCG_MODULUS_MASK
        bits.append(1.0 if (state >> _LCG_SIGN_BIT) & 1 else -1.0)
    return torch.tensor(bits, dtype=torch.float32, device=device)


class TurboQuantEmulator:
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
            raise ValueError(f"head_size / group = {groups} must be a power of two: the outer factor is H_{groups}")
        self.head_size: Final[int] = head_size
        self.group: Final[int] = group
        self.groups: Final[int] = groups
        self.device: Final[torch.device] = device
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

    @property
    def rotation(self) -> torch.Tensor:
        if self._rotation is None:
            eye_group = torch.eye(self.group, device=self.device, dtype=torch.float32)
            eye_groups = torch.eye(self.groups, device=self.device, dtype=torch.float32)
            h_inner = _sylvester_hadamard(self.group, self.device, torch.float32)
            h_outer = _sylvester_hadamard(self.groups, self.device, torch.float32)
            inner = torch.kron(eye_groups, h_inner)
            outer = torch.kron(h_outer, eye_group)
            self._rotation = outer @ torch.diag(self.pi_signs) @ inner
        return self._rotation

    @property
    def codec_tables(self) -> torch.Tensor:
        if self._codec_tables is None:
            self._codec_tables = _build_codec_tables(self)
        return self._codec_tables

    @property
    def h16(self) -> torch.Tensor:
        return _h16_operand(self.device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32) @ self.rotation.transpose(0, 1)

    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        return (y.to(torch.float32) @ self.rotation) * self.inv_sqrt_len

    def quantize(self, rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotated_fp32 = rotated.to(torch.float32)
        absmax = rotated_fp32.abs().amax(dim=-1, keepdim=True).contiguous()
        zero_row = absmax == 0.0
        biased_exponent = (absmax.view(torch.int32) >> _FP32_MANTISSA_BITS) & _FP32_EXPONENT_MASK
        exponent = biased_exponent - _FP32_EXPONENT_BIAS
        ones = torch.ones_like(absmax)
        pow_scale = _exp2_fp32(exponent - TURBOQUANT_HEADROOM_BITS)
        inv_scale = _exp2_fp32(TURBOQUANT_HEADROOM_BITS - exponent)
        codes = torch.round(rotated_fp32 * inv_scale)
        codes = torch.clamp(codes, TURBOQUANT_NIBBLE_MIN, TURBOQUANT_NIBBLE_MAX)
        codes = torch.where(zero_row, torch.zeros_like(codes), codes)
        scales = torch.where(zero_row, ones, pow_scale * self.inv_sqrt_len)
        return codes.to(torch.int8), scales.squeeze(-1)

    def dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)

    @staticmethod
    def _lloyd_max_sumsq(row: torch.Tensor, fold: str) -> torch.Tensor:
        """Match the key butterfly or value pairwise reduction's FP32 rounding."""
        squared_values = (row * row).to(torch.float32).clone()
        num_channels = squared_values.numel()

        def sum_pairwise_halves(squared_values: torch.Tensor, stop: int) -> torch.Tensor:
            width = squared_values.numel()
            while width > stop:
                width //= 2
                squared_values = squared_values[:width] + squared_values[width : 2 * width]
            return squared_values

        if fold == "value":
            squared_values = sum_pairwise_halves(squared_values, 1)
        elif fold in ("key", "butterfly"):
            quarters = [
                squared_values[quarter_index * 64 : (quarter_index + 1) * 64]
                for quarter_index in range(num_channels // 64)
            ]
            squared_values = ((quarters[0] + quarters[1]) + quarters[2]) + quarters[3]
            for stride in (1, 2, 4, 8, 16, 32):
                previous_values = squared_values.clone()
                butterfly_indices = torch.arange(64) ^ stride
                squared_values = previous_values + previous_values[butterfly_indices]
        else:
            raise ValueError(f"unknown Lloyd-Max fold {fold!r}")
        return squared_values[0]

    def quantize_lloyd_max(self, rotated: torch.Tensor, fold: str = "key") -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize rotated rows with the selected writer's Lloyd-Max reduction."""
        rotated_fp32 = rotated.to(torch.float32)
        thresholds = torch.tensor(TURBOQUANT_LM_THRESHOLDS, dtype=torch.float32, device=rotated_fp32.device)
        flat = rotated_fp32.reshape(-1, rotated_fp32.shape[-1])
        codes = torch.empty_like(flat, dtype=torch.int8)
        scales = torch.empty(flat.shape[0], dtype=torch.float32, device=rotated_fp32.device)
        for row_index in range(flat.shape[0]):
            row = flat[row_index]
            sum_of_squares = self._lloyd_max_sumsq(row, fold)
            row_rms = torch.sqrt(sum_of_squares + TURBOQUANT_LM_EPS) * self.inv_sqrt_len
            negative_normalized_row = row * (-1.0 / row_rms)

            margins = thresholds.unsqueeze(0) + negative_normalized_row.unsqueeze(-1)
            bins = (margins < 0).sum(dim=-1).to(torch.int32) - 8
            codes[row_index] = bins.to(torch.int8)
            scales[row_index] = row_rms * self.inv_sqrt_len
        return codes.reshape(rotated_fp32.shape), scales

    def dequantize_lloyd_max(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """``centroid[codes + 8] * scales`` in fp32: the codebook reconstruction."""
        centroids = torch.tensor(TURBOQUANT_LM_CENTROIDS, dtype=torch.float32, device=codes.device)
        return centroids[(codes.to(torch.int16) + 8).long()] * scales.to(torch.float32).unsqueeze(-1)

    @staticmethod
    def pack(codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] % TURBOQUANT_PACK_FACTOR:
            raise ValueError(f"cannot pack an odd channel count {codes.shape[-1]}")
        nibbles = codes.to(torch.int16) & TURBOQUANT_NIBBLE_MASK
        pairs = nibbles.reshape(*nibbles.shape[:-1], nibbles.shape[-1] // TURBOQUANT_PACK_FACTOR, 2)
        packed = pairs[..., 0] | (pairs[..., 1] << 4)
        return packed.to(torch.uint8).view(torch.int8)

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        raw = packed.view(torch.uint8).to(torch.int16)
        low = raw & TURBOQUANT_NIBBLE_MASK
        high = (raw >> 4) & TURBOQUANT_NIBBLE_MASK
        pair = torch.stack((low, high), dim=-1)
        signed = torch.where(pair >= 8, pair - TURBOQUANT_NIBBLE_LEVELS, pair)
        return signed.reshape(*packed.shape[:-1], packed.shape[-1] * TURBOQUANT_PACK_FACTOR).to(torch.int8)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes, scales = self.quantize(self.rotate(x))
        return self.pack(codes), scales

    def decode(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self.inverse_rotate(self.dequantize(self.unpack(packed), scales))


TURBOQUANT_STAGE_ROTATED_BASIS: Final[int] = 0

TURBOQUANT_STAGE_UNROTATED: Final[int] = 1

TURBOQUANT_STAGE_GATED: Final[int] = 2

TURBOQUANT_H16_ORDER: Final[int] = 16

TURBOQUANT_TABLE_DTYPE: Final[torch.dtype] = torch.bfloat16

TURBOQUANT_KERNEL_HEAD_SIZE: Final[int] = 256

TURBOQUANT_KERNEL_BLOCK_SIZE: Final[int] = 128

TURBOQUANT_KERNEL_KV_HEADS: Final[int] = 4

TURBOQUANT_DECODE_MAX_KV_HEADS: Final[int] = 4

TURBOQUANT_KERNEL_DTYPE: Final[torch.dtype] = torch.bfloat16

TURBOQUANT_OP_ROTATE: Final[str] = "npu_turboquant_rotate"

TURBOQUANT_OP_QUANTIZE_PACK: Final[str] = "npu_turboquant_quantize_pack"

TURBOQUANT_OP_UNPACK_DEQUANT: Final[str] = "npu_turboquant_unpack_dequant"

TURBOQUANT_OP_RESHAPE_AND_CACHE: Final[str] = "npu_turboquant_reshape_and_cache_a8"

TURBOQUANT_OP_RESHAPE_SCATTER: Final[str] = "npu_turboquant_reshape_scatter_aiv"

TURBOQUANT_OP_GATHER_PAGES: Final[str] = "npu_turboquant_gather_pages"

TURBOQUANT_OP_FUSED_DECODE: Final[str] = "npu_turboquant_fused_decode"

TURBOQUANT_OP_BUILD_DECODE_TILING: Final[str] = "npu_turboquant_build_decode_tiling"

TURBOQUANT_KERNEL_OPS: Final[dict[str, TurboQuantKernelStage]] = {
    TURBOQUANT_OP_ROTATE: TurboQuantKernelStage.ROTATE,
    TURBOQUANT_OP_QUANTIZE_PACK: TurboQuantKernelStage.QUANTIZE,
    TURBOQUANT_OP_UNPACK_DEQUANT: TurboQuantKernelStage.DEQUANTIZE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE: TurboQuantKernelStage.RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER: TurboQuantKernelStage.SCATTER,
    TURBOQUANT_OP_GATHER_PAGES: TurboQuantKernelStage.GATHER,
    TURBOQUANT_OP_FUSED_DECODE: TurboQuantKernelStage.DECODE,
}

TURBOQUANT_DECODE_SPLITS: Final[int] = 8

TURBOQUANT_DECODE_MAX_HEADS_PER_TASK: Final[int] = 16

TURBOQUANT_DECODE_TILING_WORDS: Final[int] = 16

TURBOQUANT_DECODE_DISPATCH_WORDS: Final[int] = 8

TURBOQUANT_DECODE_MAX_GRID: Final[int] = 32

TURBOQUANT_DECODE_MAX_SPLITS: Final[int] = 8

TURBOQUANT_PARTIAL_TAIL: Final[int] = 16

TURBOQUANT_STAGING_SLOTS: Final[int] = 2

TURBOQUANT_FIA_DUMMY_KV_LEN: Final[int] = 1


def turboquant_decode_heads_per_task(num_heads: int, num_kv_heads: int) -> int:
    group_heads = num_heads // num_kv_heads if num_kv_heads else num_heads
    return min(TURBOQUANT_DECODE_MAX_HEADS_PER_TASK, max(group_heads, 1))


def turboquant_decode_tiling_words(tokens: int, num_splits: int, grid: int) -> int:
    return tokens * num_splits * TURBOQUANT_DECODE_TILING_WORDS + grid * TURBOQUANT_DECODE_DISPATCH_WORDS


def turboquant_tiling_upper_bound_words(max_num_reqs: int, max_splits: int) -> int:
    return turboquant_decode_tiling_words(max_num_reqs, max_splits, TURBOQUANT_DECODE_MAX_GRID)


def turboquant_decode_workspace_floats(tokens: int, num_heads: int, head_size: int, num_splits: int) -> int:
    return tokens * num_heads * num_splits * (head_size + TURBOQUANT_PARTIAL_TAIL)


def turboquant_decode_query_rot_floats(tokens: int, num_heads: int, head_size: int) -> int:
    return tokens * num_heads * head_size


TURBOQUANT_REGISTERED_OPS: Final[tuple[str, ...]] = (
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_BUILD_DECODE_TILING,
)

_TURBOQUANT_EXTENSION_LOADED: bool | None = None

_TURBOQUANT_TRACE_ENABLED: Final[bool] = envs.VLLM_ASCEND_TURBOQUANT_TRACE


def _trace_turboquant_dispatch(
    op_name: str,
    uses_npu_kernel: bool,
    **tensors: torch.Tensor | None,
) -> None:
    if not _TURBOQUANT_TRACE_ENABLED:
        return

    tensor_items = [(name, tensor) for name, tensor in tensors.items() if isinstance(tensor, torch.Tensor)]
    devices = sorted({str(tensor.device) for _, tensor in tensor_items})
    device_types = {tensor.device.type for _, tensor in tensor_items}
    if uses_npu_kernel:
        backend = "NPU kernel" if "npu" in device_types else "custom torch op"
    elif device_types == {"cpu"}:
        backend = "CPU (PyTorch)"
    else:
        backend = "PyTorch"
    tensor_layouts = "; ".join(
        f"{name}(shape={tuple(tensor.shape)}, stride={tensor.stride()}, dtype={tensor.dtype}, "
        f"device={tensor.device}, storage_offset={tensor.storage_offset()}, "
        f"contiguous={tensor.is_contiguous()})"
        for name, tensor in tensor_items
    )
    logger.info(
        "TurboQuant dispatch: op=%s backend=%s devices=%s tensors=[%s]",
        op_name,
        backend,
        devices or "none",
        tensor_layouts,
    )


def _is_op_registered(namespace: str, op_name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel(f"{namespace}::{op_name}"))
    except (AttributeError, RuntimeError):
        try:
            getattr(getattr(torch.ops, namespace), op_name)
        except (AttributeError, RuntimeError):
            return False
        return True


def load_turboquant_extension() -> bool:
    global _TURBOQUANT_EXTENSION_LOADED
    if _TURBOQUANT_EXTENSION_LOADED is not None:
        return _TURBOQUANT_EXTENSION_LOADED

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        if not torch.compiler.is_compiling():
            bootstrap_custom_op_env()
    except ImportError as exc:
        logger.debug("TurboQuant: bootstrap_custom_op_env is unavailable (%s); importing anyway.", exc)

    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")

        _TURBOQUANT_EXTENSION_LOADED = True
    except (ImportError, OSError) as exc:
        _TURBOQUANT_EXTENSION_LOADED = False
        logger.debug("TurboQuant: vllm_ascend.vllm_ascend_C did not import (%s).", exc)
    return _TURBOQUANT_EXTENSION_LOADED


class TurboQuantOpsAdapter:
    def __init__(
        self,
        config: TurboQuantOpsConfig | None = None,
    ) -> None:
        self.config: TurboQuantOpsConfig = config if config is not None else TurboQuantOpsConfig.from_env()
        self._emulators: dict[tuple[int, str], TurboQuantEmulator] = {}
        self._missing_kernels: set[str] = set()
        if self.config.kernel_stages:
            load_turboquant_extension()
            self._report_namespace()

    def _report_namespace(self) -> None:
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

    def emulator_for(self, head_size: int, device: torch.device) -> TurboQuantEmulator:
        key = (head_size, str(device))
        emulator = self._emulators.get(key)
        if emulator is None:
            emulator = TurboQuantEmulator(head_size, device)
            self._emulators[key] = emulator
        return emulator

    def _resolve_kernel(self, op_name: str) -> Callable[..., Any] | None:
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
        return self._resolve_kernel(op_name) is not None

    def _require_kernel(self, op_name: str) -> Callable[..., Any]:
        kernel = self._resolve_kernel(op_name)
        if kernel is None:
            raise RuntimeError(f"torch.ops.{self.config.ops_namespace}.{op_name} is not registered")
        return kernel

    def value_layout(self) -> TurboQuantValueLayout:
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE) or self._has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            return TurboQuantValueLayout.NZ_TILED
        return self.config.value_layout

    def rotate(self, x: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        if self._has_kernel(TURBOQUANT_OP_ROTATE):
            _trace_turboquant_dispatch(TURBOQUANT_OP_ROTATE, True, x=x)
            return self._kernel_rotate(
                x=x,
                pi_signs=emulator.pi_signs,
                rot_tables=_codec_tables(emulator),
                h16=_h16_operand(emulator.device),
                output_stage=TURBOQUANT_STAGE_ROTATED_BASIS,
                inv_sqrt_len=1.0,
            )
        _trace_turboquant_dispatch(TURBOQUANT_OP_ROTATE, False, x=x)
        return self._reference_rotate(x, emulator)

    def inverse_rotate(self, y: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        if self._has_kernel(TURBOQUANT_OP_ROTATE):
            _trace_turboquant_dispatch(TURBOQUANT_OP_ROTATE, True, y=y)
            return self._kernel_rotate(
                x=y,
                pi_signs=emulator.pi_signs,
                rot_tables=_codec_tables(emulator),
                h16=_h16_operand(emulator.device),
                output_stage=TURBOQUANT_STAGE_UNROTATED,
                inv_sqrt_len=emulator.inv_sqrt_len,
            )
        _trace_turboquant_dispatch(TURBOQUANT_OP_ROTATE, False, y=y)
        return self._reference_inverse_rotate(y, emulator)

    def _kernel_rotate(
        self,
        x: torch.Tensor,
        pi_signs: torch.Tensor,
        rot_tables: torch.Tensor,
        h16: torch.Tensor,
        output_stage: int,
        inv_sqrt_len: float,
    ) -> torch.Tensor:
        return self._require_kernel(TURBOQUANT_OP_ROTATE)(x, pi_signs, rot_tables, h16, output_stage, inv_sqrt_len)

    def _reference_rotate(self, x: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        return emulator.rotate(x)

    def _reference_inverse_rotate(self, y: torch.Tensor, emulator: TurboQuantEmulator) -> torch.Tensor:
        return emulator.inverse_rotate(y)

    def quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._has_kernel(TURBOQUANT_OP_QUANTIZE_PACK):
            _trace_turboquant_dispatch(TURBOQUANT_OP_QUANTIZE_PACK, True, rotated=rotated)
            return self._kernel_quantize_and_pack(rotated=rotated, inv_sqrt_len=emulator.inv_sqrt_len)
        _trace_turboquant_dispatch(TURBOQUANT_OP_QUANTIZE_PACK, False, rotated=rotated)
        return self._reference_quantize_and_pack(rotated, emulator)

    def unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        if self._has_kernel(TURBOQUANT_OP_UNPACK_DEQUANT):
            _trace_turboquant_dispatch(TURBOQUANT_OP_UNPACK_DEQUANT, True, packed=packed, scales=scales)
            return self._kernel_unpack_and_dequantize(packed=packed, scales=scales)
        _trace_turboquant_dispatch(TURBOQUANT_OP_UNPACK_DEQUANT, False, packed=packed, scales=scales)
        return self._reference_unpack_and_dequantize(packed, scales, emulator)

    def _kernel_quantize_and_pack(
        self, rotated: torch.Tensor, inv_sqrt_len: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._require_kernel(TURBOQUANT_OP_QUANTIZE_PACK)(rotated, inv_sqrt_len)

    def _kernel_unpack_and_dequantize(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self._require_kernel(TURBOQUANT_OP_UNPACK_DEQUANT)(packed, scales)

    def _reference_quantize_and_pack(
        self, rotated: torch.Tensor, emulator: TurboQuantEmulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, scales = emulator.quantize(rotated)
        return emulator.pack(codes), scales

    def _reference_unpack_and_dequantize(
        self, packed: torch.Tensor, scales: torch.Tensor, emulator: TurboQuantEmulator
    ) -> torch.Tensor:
        if TURBOQUANT_QUANT_MODE == 1:
            return emulator.dequantize_lloyd_max(emulator.unpack(packed), scales)
        return emulator.dequantize(emulator.unpack(packed), scales)

    def reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        plan: TurboQuantStripPlan | None = None,
    ) -> None:
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE):
            self._kernel_ingest_prefill(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
                plan=plan,
            )
            return
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_RESHAPE_AND_CACHE,
            False,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
        )
        self._reference_reshape_and_cache_prefill(
            key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator
        )

    def _kernel_ingest_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        plan: TurboQuantStripPlan | None = None,
    ) -> None:
        if plan is None:
            plan = turboquant_plan_strip_ingestion(slot_mapping.tolist(), int(key_cache.shape[1]))
        tables = _codec_tables(emulator)

        if plan.is_whole_batch:
            self._kernel_cache_prefill(
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
            self._kernel_cache_prefill(
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

    def _kernel_cache_prefill(
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
        quant_mode: int = TURBOQUANT_QUANT_MODE,
    ) -> None:
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_RESHAPE_AND_CACHE,
            True,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
            pi_signs=pi_signs,
            tables=tables,
        )
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
            quant_mode,
        )

    def _reference_reshape_and_cache_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        self._write_slots(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            emulator,
            lloyd_max_folds=("key", "value"),
        )

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
        if self._has_kernel(TURBOQUANT_OP_RESHAPE_SCATTER):
            _trace_turboquant_dispatch(
                TURBOQUANT_OP_RESHAPE_SCATTER,
                True,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                pi_signs=emulator.pi_signs,
            )
            self._kernel_cache_decode(
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
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_RESHAPE_SCATTER,
            False,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            slot_mapping=slot_mapping,
        )
        self._reference_reshape_scatter_decode(key, value, key_cache, value_cache, scale_cache, slot_mapping, emulator)

    def _kernel_cache_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        pi_signs: torch.Tensor,
        inv_sqrt_len: float,
        quant_mode: int = TURBOQUANT_QUANT_MODE,
    ) -> None:
        self._require_kernel(TURBOQUANT_OP_RESHAPE_SCATTER)(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            pi_signs,
            inv_sqrt_len,
            quant_mode,
        )

    def _reference_reshape_scatter_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
    ) -> None:
        self._write_slots(
            key,
            value,
            key_cache,
            value_cache,
            scale_cache,
            slot_mapping,
            emulator,
            lloyd_max_folds=("key", "key"),
        )

    def _write_slots(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        emulator: TurboQuantEmulator,
        lloyd_max_folds: tuple[str, str] = ("key", "value"),
    ) -> None:
        if TURBOQUANT_QUANT_MODE == 1:
            codes_key, key_scales = emulator.quantize_lloyd_max(self.rotate(key, emulator), lloyd_max_folds[0])
            codes_value, value_scales = emulator.quantize_lloyd_max(self.rotate(value, emulator), lloyd_max_folds[1])
        else:
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

        num_blocks = key_cache.shape[0]
        slot_capacity = num_blocks * block_size
        max_slot = int(slots.max())
        if max_slot >= slot_capacity:
            raise ValueError(
                f"slot_mapping is out of range for the TurboQuant cache: expected slots in "
                f"[0, {slot_capacity}) ({num_blocks} blocks x {block_size} tokens, from key_cache.shape="
                f"{tuple(key_cache.shape)}), but the largest non-negative slot is {max_slot}."
            )

        pages = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots - pages * block_size

        layout = self.value_layout()
        if layout is TurboQuantValueLayout.CHANNEL_MAJOR:
            self._write_strip_planes(codes_key, codes_value, key_cache, value_cache, pages, offsets)
        elif layout is TurboQuantValueLayout.NZ_TILED:
            self._write_nz_planes(codes_key, codes_value, key_cache, value_cache, slots)
        else:
            for packed, cache in (
                (emulator.pack(codes_key), key_cache),
                (emulator.pack(codes_value), value_cache),
            ):
                _flat_slot_view(cache)[slots] = packed

        if layout is TurboQuantValueLayout.NZ_TILED:
            self._write_nz_scales(key_scales, value_scales, scale_cache, slots)
        else:
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
        num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
        head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
        if block_size % TURBOQUANT_STRIP_TOKENS:
            raise ValueError(
                f"the strip cache is written 16 tokens at a time, so block_size {block_size} must be a "
                f"multiple of {TURBOQUANT_STRIP_TOKENS}"
            )
        strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS

        key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
        key_pages[pages, :, offsets] = TurboQuantEmulator.pack(codes_key)

        strips = torch.div(offsets, TURBOQUANT_STRIP_TOKENS, rounding_mode="floor")
        lanes = offsets - strips * TURBOQUANT_STRIP_TOKENS
        value_pages = _require_contiguous(value_cache, "value").view(
            num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
        )
        touched, inverse = torch.unique(pages, return_inverse=True)
        panel = TurboQuantEmulator.unpack(value_pages[touched])
        panel[inverse, :, strips, :, lanes] = codes_value
        value_pages[touched] = TurboQuantEmulator.pack(panel)

    @staticmethod
    def _nz_byte_offsets(slots: torch.Tensor, num_kv_heads: int, packed_bytes: int) -> torch.Tensor:
        tile_rows = TURBOQUANT_NZ_TILE_ROWS
        c0 = TURBOQUANT_NZ_OPERAND_C0
        groups = packed_bytes // c0
        device = slots.device
        tile_row = slots % tile_rows
        base = (slots - tile_row) * (num_kv_heads * packed_bytes) + tile_row * c0
        heads = torch.arange(num_kv_heads, device=device) * (tile_rows * packed_bytes)
        group = torch.arange(groups, device=device) * (tile_rows * c0)
        byte = torch.arange(c0, device=device)
        return base[..., None, None, None] + heads.view(-1, 1, 1) + group.view(-1, 1) + byte

    def _write_nz_planes(
        self,
        codes_key: torch.Tensor,
        codes_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
        head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
        if block_size % TURBOQUANT_NZ_TILE_ROWS:
            raise ValueError(
                f"the NZ-tiled cache is laid out in {TURBOQUANT_NZ_TILE_ROWS}-row tiles, so block_size "
                f"{block_size} must be a multiple of {TURBOQUANT_NZ_TILE_ROWS}"
            )
        offsets = self._nz_byte_offsets(slots, num_kv_heads, packed_bytes)

        def nz_bytes(codes: torch.Tensor) -> torch.Tensor:
            low, high = codes[..., :packed_bytes], codes[..., packed_bytes:]
            woven = torch.stack((low, high), dim=-1).reshape(*codes.shape[:-1], head_size)
            return TurboQuantEmulator.pack(woven).reshape(*codes.shape[:-1], -1, TURBOQUANT_NZ_OPERAND_C0)

        for codes, cache, name in (
            (codes_key, key_cache, "key"),
            (codes_value, value_cache, "value"),
        ):
            _require_contiguous(cache, name).view(-1)[offsets] = nz_bytes(codes).to(cache.dtype)

    @staticmethod
    def _nz_scale_slot_floats(num_kv_heads: int) -> int:
        lanes = TURBOQUANT_PACK_FACTOR * num_kv_heads
        return -(-lanes // TURBOQUANT_NZ_FP32_PER_BLOCK) * TURBOQUANT_NZ_FP32_PER_BLOCK

    def _write_nz_scales(
        self,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        scale_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        num_kv_heads = scale_cache.shape[2]
        slot_floats = self._nz_scale_slot_floats(num_kv_heads)
        flat = _require_contiguous(scale_cache, "scale").view(-1)
        total_slots = scale_cache.shape[1] * scale_cache.shape[3]
        if flat.numel() < total_slots * slot_floats:
            raise ValueError(
                f"the NZ scale plane needs a whole {slot_floats}-float record per slot "
                f"(NzScaleSlotFloats({num_kv_heads})), {total_slots * slot_floats} floats over the cache; "
                f"the allocated plane holds {flat.numel()}, which is exact only when 2 * num_kv_heads "
                f"is a multiple of {TURBOQUANT_NZ_FP32_PER_BLOCK}"
            )
        lanes = torch.arange(num_kv_heads, device=slots.device)
        records = slots[:, None] * slot_floats + lanes
        flat[records] = key_scales.to(scale_cache.dtype)
        flat[records + num_kv_heads] = value_scales.to(scale_cache.dtype)

    def gather(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_block_tables(block_tables, int(key_cache.shape[0]))
        if self._has_kernel(TURBOQUANT_OP_GATHER_PAGES):
            _trace_turboquant_dispatch(
                TURBOQUANT_OP_GATHER_PAGES,
                True,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
            )
            return self._kernel_gather_pages(
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
            )
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_GATHER_PAGES,
            False,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
        )
        layout = self.value_layout()
        if layout is TurboQuantValueLayout.CHANNEL_MAJOR:
            return self._reference_gather_pages_strip(key_cache, value_cache, scale_cache, block_tables)
        if layout is TurboQuantValueLayout.NZ_TILED:
            return self._reference_gather_pages_nz(key_cache, value_cache, scale_cache, block_tables)
        return self._reference_gather_pages(key_cache, value_cache, scale_cache, block_tables)

    @staticmethod
    def _validate_block_tables(block_tables: torch.Tensor, num_blocks: int) -> None:
        if block_tables.numel() == 0:
            return
        low = block_tables.min()
        high = block_tables.max()
        if bool(low < 0) or bool(high >= num_blocks):
            raise ValueError(
                f"block_tables must lie in [0, {num_blocks}) -- num_blocks is key_cache.shape[0], "
                f"the cache's own page count -- but block_tables has min={int(low)} and "
                f"max={int(high)} (shape {tuple(block_tables.shape)}). A negative page wraps onto "
                "the last page in the golden reader, and a page past the end reads outside the "
                "cache; both are refused before any gather happens."
            )

    def _kernel_gather_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._require_kernel(TURBOQUANT_OP_GATHER_PAGES)(key_cache, value_cache, scale_cache, block_tables)

    def _reference_gather_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = key_cache.shape[1]
        pages = block_tables.to(torch.int64)
        batch_size, max_blocks = pages.shape
        within = torch.arange(block_size, device=pages.device, dtype=torch.int64)
        slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

        packed_key = _flat_slot_view(key_cache)[slots]
        packed_value = _flat_slot_view(value_cache)[slots]
        num_kv_heads = scale_cache.shape[2]

        def gathered_scales(plane: int) -> torch.Tensor:
            return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

        return (
            packed_key,
            gathered_scales(TURBOQUANT_KEY_PLANE),
            packed_value,
            gathered_scales(TURBOQUANT_VALUE_PLANE),
        )

    def _reference_gather_pages_strip(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
        packed_key = key_pages[pages].permute(0, 1, 3, 2, 4).reshape(batch_size, -1, num_kv_heads, packed_bytes)

        value_pages = _require_contiguous(value_cache, "value").view(
            num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
        )
        codes = TurboQuantEmulator.unpack(value_pages[pages])
        token_major = codes.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, -1, num_kv_heads, head_size)
        packed_value = TurboQuantEmulator.pack(token_major.contiguous())

        def gathered_scales(plane: int) -> torch.Tensor:
            return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

        return (
            packed_key,
            gathered_scales(TURBOQUANT_KEY_PLANE),
            packed_value,
            gathered_scales(TURBOQUANT_VALUE_PLANE),
        )

    def _reference_gather_pages_nz(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
        if block_size % TURBOQUANT_NZ_TILE_ROWS:
            raise ValueError(
                f"the NZ-tiled cache is laid out in {TURBOQUANT_NZ_TILE_ROWS}-row tiles, so block_size "
                f"{block_size} must be a multiple of {TURBOQUANT_NZ_TILE_ROWS}"
            )
        pages = block_tables.to(torch.int64)
        batch_size, max_blocks = pages.shape
        within = torch.arange(block_size, device=pages.device, dtype=torch.int64)
        slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

        offsets = self._nz_byte_offsets(slots, num_kv_heads, packed_bytes)

        def read_plane(cache: torch.Tensor, name: str) -> torch.Tensor:
            gathered = _require_contiguous(cache, name).view(-1)[offsets]
            packed = gathered.reshape(batch_size, -1, num_kv_heads, packed_bytes)
            woven = TurboQuantEmulator.unpack(packed)
            codes = torch.cat((woven[..., 0::2], woven[..., 1::2]), dim=-1)
            return TurboQuantEmulator.pack(codes.contiguous())

        slot_floats = self._nz_scale_slot_floats(num_kv_heads)
        flat_scales = _require_contiguous(scale_cache, "scale").view(-1)
        lanes = torch.arange(num_kv_heads, device=slots.device)
        records = slots[..., None] * slot_floats + lanes

        return (
            read_plane(key_cache, "key"),
            flat_scales[records],
            read_plane(value_cache, "value"),
            flat_scales[records + num_kv_heads],
        )

    def dequantize_cache(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        emulator: TurboQuantEmulator,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rotated = self.unpack_and_dequantize(packed, scales, emulator)
        return self.inverse_rotate(rotated, emulator).to(dtype)

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
        if not self._has_kernel(TURBOQUANT_OP_FUSED_DECODE):
            _trace_turboquant_dispatch(
                TURBOQUANT_OP_FUSED_DECODE,
                False,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                scale_cache=scale_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                output=output,
                tiling=tiling,
                workspace=workspace,
                query_rot=query_rot,
            )
            return None
        self._validate_fused_decode_geometry(query, key_cache, scale_cache, output, num_kv_heads)
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_FUSED_DECODE,
            True,
            query=query,
            pi_signs=emulator.pi_signs,
            rot_tables=emulator.codec_tables,
            h16=emulator.h16,
            key_cache=key_cache,
            value_cache=value_cache,
            scale_cache=scale_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            tiling=tiling,
            workspace=workspace,
            query_rot=query_rot,
            output=output,
        )
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

    def _validate_fused_decode_geometry(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        output: torch.Tensor,
        num_kv_heads: int,
    ) -> None:
        if query.dtype != TURBOQUANT_KERNEL_DTYPE:
            raise TypeError(
                f"the fused TurboQuant decode kernel only has {TURBOQUANT_KERNEL_DTYPE} "
                f"instances, but query.dtype is {query.dtype} (query shape "
                f"{tuple(query.shape)}). The golden fallback serves any floating dtype; "
                "this is the kernel launch path."
            )
        if output.dtype != TURBOQUANT_KERNEL_DTYPE:
            raise TypeError(
                f"the fused TurboQuant decode kernel writes {TURBOQUANT_KERNEL_DTYPE} only, but "
                f"output.dtype is {output.dtype} (output shape {tuple(output.shape)}). The "
                "golden fallback serves any floating dtype; this is the kernel launch path."
            )
        if query.data_ptr() == output.data_ptr():
            raise ValueError(
                f"query and output must not alias: both are at data_ptr={query.data_ptr()}. "
                "The fused decode reads the query through its whole prologue while writing "
                "output, so an in-place call would overwrite the query it is still reading."
            )

        num_blocks = int(key_cache.shape[0])
        block_size = int(key_cache.shape[1])
        slot_floats = self._nz_scale_slot_floats(num_kv_heads)
        need = num_blocks * block_size * slot_floats
        if scale_cache.numel() < need:
            raise ValueError(
                f"scale_cache is too small for the fused TurboQuant decode: it holds "
                f"{scale_cache.numel()} floats, but the kernel reads {slot_floats} floats per "
                f"slot (NzScaleSlotFloats({num_kv_heads})) over {num_blocks} blocks x "
                f"{block_size} tokens = {need} (key_cache.shape={tuple(key_cache.shape)}, "
                f"scale_cache.shape={tuple(scale_cache.shape)})."
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
        decode_unpack: int = TURBOQUANT_DECODE_UNPACK,
    ) -> torch.Tensor:
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
            decode_unpack,
        )
        return output


@dataclass(frozen=True)
class TurboQuantStripRun:
    start: int
    stop: int
    pad: int


@dataclass(frozen=True)
class TurboQuantStripPlan:
    runs: tuple[TurboQuantStripRun, ...]
    waves: tuple[tuple[int, ...], ...]
    num_tokens: int

    @property
    def is_whole_batch(self) -> bool:
        return not self.waves and self.runs == (TurboQuantStripRun(0, self.num_tokens, 0),)

    @property
    def num_launches(self) -> int:
        return len(self.runs) + len(self.waves)


def turboquant_plan_strip_ingestion(
    slots: Sequence[int], block_size: int, strip: int = TURBOQUANT_STRIP_TOKENS
) -> TurboQuantStripPlan:
    num_tokens = len(slots)
    runs: list[TurboQuantStripRun] = []
    scatter: list[int] = []
    written: set[int] | None = None

    def run_length(start: int, limit: int) -> int:
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
        if not aligned(start):
            return 0
        tail = run_length(start, strip)
        if not 0 < tail < strip:
            return 0
        pad = strip - tail
        last = slots[start] + tail - 1
        if block_size <= 0 or (last + pad) // block_size != last // block_size:
            return 0
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


def _strip_scatter_waves(slots: Sequence[int], tokens: Sequence[int], strip: int) -> tuple[tuple[int, ...], ...]:
    waves: list[list[int]] = []
    depth: dict[int, int] = {}
    for token in tokens:
        slot = slots[token]
        if slot < 0:
            wave = 0
        else:
            wave = depth.get(slot // strip, 0)
            depth[slot // strip] = wave + 1
        while len(waves) <= wave:
            waves.append([])
        waves[wave].append(token)
    return tuple(tuple(wave) for wave in waves)


def _strip_run_rows(activations: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    window = activations[run.start : run.stop]
    if not run.pad:
        return window
    return torch.cat((window, window[-1:].expand(run.pad, *window.shape[1:]))).contiguous()


def _strip_run_slots(slot_mapping: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    window = slot_mapping[run.start : run.stop]
    if not run.pad:
        return window
    pad = torch.arange(1, run.pad + 1, dtype=window.dtype, device=window.device) + window[-1]
    return torch.cat((window, pad)).contiguous()


def _flat_slot_view(cache: torch.Tensor) -> torch.Tensor:
    if not cache.is_contiguous():
        raise RuntimeError(
            "the TurboQuant packed cache plane must be contiguous to be addressed by slot; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache.view(cache.shape[0] * cache.shape[1], *cache.shape[2:])


def _require_contiguous(cache: torch.Tensor, name: str) -> torch.Tensor:
    if not cache.is_contiguous():
        raise RuntimeError(
            f"the TurboQuant {name} plane must be contiguous to be addressed by page; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache


def _codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    return emulator.codec_tables


def _build_codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    device = emulator.device
    h_inner = _sylvester_hadamard(emulator.group, device, torch.float32)
    signs = emulator.pi_signs.reshape(emulator.groups, emulator.group)
    positive = signs.unsqueeze(-1) * h_inner.unsqueeze(0)
    tables = torch.stack((positive, -positive), dim=0)
    return tables.reshape(TURBOQUANT_PLANES * emulator.groups, emulator.group, emulator.group).to(
        TURBOQUANT_TABLE_DTYPE
    )


_H16_CACHE: dict[str, torch.Tensor] = {}


def _h16_operand(device: torch.device) -> torch.Tensor:
    key = str(device)
    operand = _H16_CACHE.get(key)
    if operand is None:
        operand = _sylvester_hadamard(TURBOQUANT_H16_ORDER, device, torch.float16)
        _H16_CACHE[key] = operand
    return operand


@dataclass
class AscendTurboQuantMetadata(AscendMetadata):
    turboquant_context_lens: torch.Tensor = None
    turboquant_tiling: torch.Tensor = None
    turboquant_workspace: torch.Tensor = None
    turboquant_query_rot: torch.Tensor = None
    turboquant_num_tokens: int = 0


@dataclass(frozen=True)
class TurboQuantQueryLayout:
    request_indices: torch.Tensor
    positions_in_request: torch.Tensor
    query_lengths: torch.Tensor
    query_lengths_list: list[int]
    max_query_len: int
    num_tokens: int


class AscendTurboQuantMetadataBuilder(AscendAttentionMetadataBuilder):
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

        self._block_size = TURBOQUANT_KERNEL_BLOCK_SIZE
        self._num_heads = int(vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config))
        self._num_splits = TURBOQUANT_DECODE_SPLITS
        self._heads_per_task = turboquant_decode_heads_per_task(self._num_heads, self._num_kv_heads)

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
        self._staging_slots = TURBOQUANT_STAGING_SLOTS
        self._staging_step = 0
        staging = [torch.zeros(self._max_num_reqs, dtype=torch.int32, device="cpu") for _ in range(self._staging_slots)]
        try:
            staging = [slot.pin_memory() for slot in staging]
            self._staging_pinned = True
        except (RuntimeError, NotImplementedError):
            self._staging_pinned = False
        self._context_lens_host_ring = staging
        self._context_lens_upload_events: list[Any | None] = [None] * self._staging_slots

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
        assert seq_lens.device.type == "cpu", (
            f"_refresh_context_lens stages `seq_lens` into a host ring, but seq_lens is on "
            f"device {seq_lens.device!r} (device type {seq_lens.device.type!r}). A device "
            "tensor would force a per-step device-to-host synchronisation here and break "
            "ACL graph capture; the CPU mirror is what this method exists to upload."
        )
        num_reqs = int(common_attn_metadata.num_reqs)
        padded = int(seq_lens.shape[0])
        if padded > self._max_num_reqs:
            raise RuntimeError(
                f"the TurboQuant builder was sized for {self._max_num_reqs} requests but this step "
                f"carries {padded}. max_num_seqs is what sizes the launch buffers."
            )
        live = min(num_reqs, padded)
        host = self._staging_slot()
        self._staging_step += 1
        host[:live] = seq_lens[:live].to(torch.int32)
        if padded > live:
            tail = seq_lens[num_reqs:]
            if tail.numel():
                dirty = (tail != 0) & (tail != TURBOQUANT_FIA_DUMMY_KV_LEN)
                if bool(dirty.any()):
                    offending = (num_reqs + torch.nonzero(dirty, as_tuple=False).flatten()).tolist()
                    raise RuntimeError(
                        f"seq_lens[{num_reqs}:] carries a dirty padded tail: expected all 0 "
                        f"(cudagraph padding) or all {TURBOQUANT_FIA_DUMMY_KV_LEN} (the FIA dummy "
                        f"build() appends), but seq_lens has max={int(tail.max())}, "
                        f"min={int(tail.min())} at indices {offending} (shape "
                        f"{tuple(seq_lens.shape)}, num_reqs={num_reqs}). Refusing rather than "
                        "masking a stale length."
                    )
            host[live:padded] = 0
        self._context_lens[:padded].copy_(host[:padded], non_blocking=self._staging_pinned)
        if self._staging_pinned and self._context_lens.device.type != "cpu":
            index = (self._staging_step - 1) % self._staging_slots
            event = self._context_lens_upload_events[index]
            if event is None:
                event = torch.npu.Event()
                self._context_lens_upload_events[index] = event
            event.record()
        return self._context_lens[:padded], host

    def _staging_slot(self) -> torch.Tensor:
        slots = self._staging_slots
        ring = self._context_lens_host_ring
        assert slots == TURBOQUANT_STAGING_SLOTS, (
            f"the TurboQuant host staging ring has {slots} slots, but the asynchronous upload "
            f"contract needs exactly {TURBOQUANT_STAGING_SLOTS} slots."
        )
        assert len(ring) == slots, (
            f"`_context_lens_host_ring` has {len(ring)} buffers but `_staging_slots` says "
            f"{slots}; the ring and the modulo that indexes it must agree."
        )
        assert isinstance(self._staging_step, int) and self._staging_step >= 0, (
            f"`_staging_step` must be a non-negative int, got {self._staging_step!r}; a negative "
            "step would wrap the modulo onto an arbitrary slot."
        )
        index = self._staging_step % slots
        assert 0 <= index < slots, f"staging index {index} is outside the {slots}-slot ring"
        for position, slot in enumerate(ring):
            assert slot.shape == (self._max_num_reqs,), (
                f"staging slot {position} has shape {tuple(slot.shape)}, expected "
                f"({self._max_num_reqs},): every slot must cover the whole device buffer."
            )
            assert slot.dtype == torch.int32, (
                f"staging slot {position} has dtype {slot.dtype}, expected torch.int32: it is "
                "uploaded into the int32 context-lens buffer."
            )
            assert slot.device.type == "cpu", (
                f"staging slot {position} is on device {slot.device!r}, expected a CPU tensor: "
                "a device staging slot defeats the single host-to-device upload."
            )
        events = getattr(self, "_context_lens_upload_events", None)
        event = events[index] if events is not None else None
        if event is not None and not event.query():
            ring[index] = torch.empty_like(ring[index], pin_memory=self._staging_pinned)
        return ring[index]

    def _refresh_tiling(self, host_context_lens: torch.Tensor, tokens: int) -> None:
        namespace = getattr(torch.ops, TURBOQUANT_DEFAULT_NAMESPACE, None)
        build_tiling = getattr(namespace, TURBOQUANT_OP_BUILD_DECODE_TILING, None) if namespace else None
        if build_tiling is None:
            return
        if self._block_size % 64 or self._block_size & (self._block_size - 1):
            raise RuntimeError(
                f"the TurboQuant fused decode's closed-form tile cursor needs block_size to be a "
                f"power-of-two multiple of 64, but this engine's KV cache spec carries "
                f"block_size={self._block_size}. On a hybrid model that value comes from the "
                f"Mamba/kpool page alignment in vllm_ascend.patch.platform.patch_mamba_config; "
                f"otherwise it is the --block-size the engine was started with."
            )
        _trace_turboquant_dispatch(
            TURBOQUANT_OP_BUILD_DECODE_TILING,
            True,
            context_lens=host_context_lens[:tokens],
            tiling=self._tiling,
        )
        build_tiling(
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


@register_backend(AttentionBackendEnum.TURBOQUANT)
class AscendTurboQuantBackend(AscendAttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type[AscendTurboQuantBackendImpl]:
        return AscendTurboQuantBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendTurboQuantMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [TURBOQUANT_KERNEL_BLOCK_SIZE]

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return turboquant_packed_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_scale_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int) -> tuple[int, ...]:
        return turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)

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


TURBOQUANT_PROFILE_ENV: Final[str] = "TURBOQUANT_PROFILE"

_PROFILE_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

TURBOQUANT_PROFILE_TRACE_DIR: Final[str] = "./turboquant_traces"

TURBOQUANT_PROFILE_NOTICE: Final[str] = (
    "TurboQuant profiling active. Ascend CANN does not support in-memory key_averages. "
    "Traces are being dumped to %s for offline analysis in Ascend Insight or chrome://tracing."
)


def turboquant_profile_enabled() -> bool:
    return os.getenv(TURBOQUANT_PROFILE_ENV, "0").strip().lower() in _PROFILE_TRUTHY


def _stream_is_capturing() -> bool:
    with suppress(AttributeError, RuntimeError):
        return bool(torch.npu.is_current_stream_capturing())
    return False


TURBOQUANT_FIA_VERSION_ENV: Final[str] = "TURBOQUANT_FIA_VERSION"


class AscendTurboQuantBackendImpl(AscendAttentionBackendImpl):
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
            raise ValueError(f"TurboQuant needs head_size to be a multiple of {TURBOQUANT_GROUP}, got {head_size}")
        if alibi_slopes is not None:
            raise NotImplementedError("TurboQuant has no ALiBi path: the decode kernel takes no slope operand.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TurboQuant has no sliding-window path: the fused decode reads whole pages from the block table."
            )
        if sinks is not None:
            raise NotImplementedError(
                "learnable attention sinks need the kernel's softmax denominator as an out-tensor; "
                "the dequantising golden path does not expose one."
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
        staged = ", ".join(sorted(stage.value for stage in self.adapter.config.kernel_stages))
        logger.info_once(
            "TurboQuant backend active: packed INT4 KV cache, kernel stages=%s, ops namespace=%s",
            staged or "none (all golden)",
            self.adapter.config.ops_namespace,
        )

    def _check_kernel_geometry(self) -> None:
        config = self.adapter.config
        writers = (
            TurboQuantKernelStage.RESHAPE_AND_CACHE,
            TurboQuantKernelStage.SCATTER,
        )
        enabled_writers = [stage for stage in writers if config.enables(stage)]
        decode_enabled = config.enables(TurboQuantKernelStage.DECODE)
        if not enabled_writers and not decode_enabled:
            return

        enabled_stages = list(enabled_writers)
        if decode_enabled:
            enabled_stages.append(TurboQuantKernelStage.DECODE)
        stage_names = ", ".join(sorted(stage.value for stage in enabled_stages))

        if self.head_size != TURBOQUANT_KERNEL_HEAD_SIZE:
            raise ValueError(
                f"TurboQuant kernel stages {stage_names} are switched on, but all three kernels are "
                f"specialised to head_size {TURBOQUANT_KERNEL_HEAD_SIZE} and this model's is "
                f"{self.head_size}. The golden model has no such constraint: unset "
                f"{TURBOQUANT_STAGES_ENV} to run this shape."
            )

        if enabled_writers and self.num_kv_heads != TURBOQUANT_KERNEL_KV_HEADS:
            raise ValueError(
                f"TurboQuant ingestion stages {stage_names} are switched on, but both cache writers are "
                f"specialised to exactly {TURBOQUANT_KERNEL_KV_HEADS} KV heads and this model has "
                f"{self.num_kv_heads}. The kernel does not check this: outside it the writer reads "
                f"one head's rows and writes another head's page, with every canary intact."
            )

        if decode_enabled and not 0 < self.num_kv_heads <= TURBOQUANT_DECODE_MAX_KV_HEADS:
            raise ValueError(
                f"the TurboQuant fused decode refuses num_kv_heads outside 1.."
                f"{TURBOQUANT_DECODE_MAX_KV_HEADS} (valid_launch), and this model has "
                f"{self.num_kv_heads}. This is the constraint a GQA model is most likely to hit; "
                f"unset {TURBOQUANT_STAGES_ENV} or drop the 'decode' stage to run golden."
            )
        if decode_enabled and self.num_heads % max(self.num_kv_heads, 1):
            raise ValueError(
                f"the TurboQuant fused decode needs num_heads ({self.num_heads}) to be a multiple "
                f"of num_kv_heads ({self.num_kv_heads})."
            )

        if self.model_dtype != TURBOQUANT_KERNEL_DTYPE:
            raise ValueError(
                f"TurboQuant kernel stages {stage_names} are switched on, but every kernel is "
                f"{TURBOQUANT_KERNEL_DTYPE} only and this model runs in {self.model_dtype}. The "
                f"fused decode's dispatch returns WITHOUT LAUNCHING for any other element type, "
                f"which would leave the attention output unwritten and report success, so this is "
                f"refused rather than attempted."
            )

    def _ensure_statics(self, layer: AttentionLayer, device: torch.device) -> TurboQuantEmulator:
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
                "TurboQuant backend is allocating its own fp32 scale plane %s (%.1f MiB per layer): the "
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

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return query, key, value, output
        if kv_cache is None or len(kv_cache) < 2:
            return query, key, value, output

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
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

        phase = self._ingestion_phase(attn_metadata, num_actual_tokens)
        slot_mapping, strip_plan = self._ingestion_inputs(attn_metadata, num_actual_tokens, phase)
        if phase is TurboQuantIngestionPhase.PREFILL:
            self.adapter.reshape_and_cache_prefill(
                key=key[:num_actual_tokens].contiguous(),
                value=value[:num_actual_tokens].contiguous(),
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
                plan=strip_plan,
            )
        else:
            self.adapter.reshape_scatter_decode(
                key=key[:num_actual_tokens].contiguous(),
                value=value[:num_actual_tokens].contiguous(),
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                scale_cache=scale_cache,
                slot_mapping=slot_mapping,
                emulator=emulator,
            )
        notify_kv_cache_written()
        return query, key, value, output

    def _ingestion_inputs(
        self,
        attn_metadata: AscendMetadata,
        num_actual_tokens: int,
        phase: TurboQuantIngestionPhase,
    ) -> tuple[torch.Tensor, TurboQuantStripPlan | None]:
        cached = getattr(attn_metadata, "_turboquant_ingestion_cache", None)
        if cached is not None and cached[0] is attn_metadata.slot_mapping:
            return cached[1], cached[2]
        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens].to(torch.int32)
        strip_plan: TurboQuantStripPlan | None = None
        if phase is TurboQuantIngestionPhase.PREFILL and self.adapter._has_kernel(TURBOQUANT_OP_RESHAPE_AND_CACHE):
            strip_plan = turboquant_plan_strip_ingestion(slot_mapping.tolist(), int(self.key_cache.shape[1]))
        attn_metadata._turboquant_ingestion_cache = (attn_metadata.slot_mapping, slot_mapping, strip_plan)
        return slot_mapping, strip_plan

    @staticmethod
    def _ingestion_phase(attn_metadata: AscendMetadata, num_actual_tokens: int) -> TurboQuantIngestionPhase:
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None)
        num_requests = len(seq_lens_list) if seq_lens_list is not None else 0
        if num_actual_tokens > num_requests:
            return TurboQuantIngestionPhase.PREFILL
        return TurboQuantIngestionPhase.DECODE

    def _gather_dense_kv(
        self, attn_metadata: AscendMetadata, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        seq_lens = getattr(attn_metadata, "turboquant_context_lens", None)
        if seq_lens is None:
            seq_lens = attn_metadata.seq_lens[:batch_size].to(device=dense_key.device, dtype=torch.int32)
        else:
            seq_lens = seq_lens[:batch_size]
        live = (positions.unsqueeze(0) < seq_lens.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        zero = torch.zeros((), dtype=dtype, device=dense_key.device)
        return torch.where(live, dense_key, zero), torch.where(live, dense_value, zero)

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
        emulator = self.emulator
        assert emulator is not None
        batch_size = self._decode_num_tokens(attn_metadata)
        if self.scale_cache is None:
            raise RuntimeError(
                "the TurboQuant decode found no scale plane: nothing has been ingested into this layer yet."
            )

        tiling, workspace, query_rot = self._decode_buffers(attn_metadata, batch_size)
        fused_output = self.adapter.fused_decode(
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
        if fused_output is not None:
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
        tokens = int(getattr(attn_metadata, "turboquant_num_tokens", 0) or 0)
        if tokens:
            return tokens
        return len(attn_metadata.seq_lens_list)

    @staticmethod
    def _decode_context_lens(attn_metadata: AscendMetadata, batch_size: int) -> torch.Tensor:
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
        emulator = self.emulator
        assert emulator is not None

        layout = self._mixed_query_layout(attn_metadata, query.device)
        batch_size = len(layout.query_lengths_list)
        total_tokens = layout.num_tokens
        max_query_len = layout.max_query_len
        request_indices = layout.request_indices
        positions_in_request = layout.positions_in_request

        padded_query = torch.zeros(
            (batch_size, max_query_len, self.num_heads, self.head_size),
            dtype=query.dtype,
            device=query.device,
        )
        padded_query[request_indices, positions_in_request] = query[:total_tokens]

        dense_key, dense_value = self._gather_dense_kv(attn_metadata, batch_size, query.dtype)
        padded_kv_len = dense_key.shape[1]

        seq_lens_device = getattr(attn_metadata, "turboquant_context_lens", None)
        if seq_lens_device is None:
            seq_lens = attn_metadata.seq_lens[:batch_size].to(device=query.device, dtype=torch.int64)
        else:
            seq_lens = seq_lens_device[:batch_size]
        query_lengths = layout.query_lengths
        key_positions = torch.arange(padded_kv_len, device=query.device, dtype=torch.int64)
        query_positions = torch.arange(max_query_len, device=query.device, dtype=torch.int64)
        horizon = (seq_lens - query_lengths).unsqueeze(1) + query_positions.unsqueeze(0)
        atten_mask = (key_positions.view(1, 1, -1) > horizon.unsqueeze(-1)).unsqueeze(1)

        attn_output = self._fused_infer_attention(
            query=padded_query,
            key=dense_key,
            value=dense_value,
            input_layout="BSND",
            actual_seq_lengths=layout.query_lengths_list,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            atten_mask=atten_mask,
            sparse_mode=0,
        )
        attn_output = attn_output.view(batch_size, max_query_len, self.num_heads, self.head_size)
        output[:total_tokens] = attn_output[request_indices, positions_in_request]
        return output

    @staticmethod
    def _mixed_query_layout(attn_metadata: AscendMetadata, device: torch.device) -> TurboQuantQueryLayout:
        cumulative_lengths = tuple(attn_metadata.actual_seq_lengths_q)
        cached = getattr(attn_metadata, "_turboquant_query_layout_cache", None)
        if cached is not None and cached[:2] == (cumulative_lengths, device):
            return cached[2]

        query_lengths = torch.diff(torch.tensor((0, *cumulative_lengths), dtype=torch.int64))
        query_lengths_list = query_lengths.tolist()
        num_tokens = cumulative_lengths[-1]
        request_indices = torch.repeat_interleave(torch.arange(len(query_lengths_list)), query_lengths)
        row_starts = torch.cumsum(query_lengths, dim=0) - query_lengths
        positions_in_request = torch.arange(num_tokens) - row_starts[request_indices]
        host_layout = torch.cat((request_indices, positions_in_request, query_lengths))
        if device.type != "cpu":
            host_layout = host_layout.pin_memory()
        device_layout = host_layout.to(device, non_blocking=host_layout.is_pinned())
        layout = TurboQuantQueryLayout(
            request_indices=device_layout[:num_tokens],
            positions_in_request=device_layout[num_tokens : 2 * num_tokens],
            query_lengths=device_layout[2 * num_tokens :],
            query_lengths_list=query_lengths_list,
            max_query_len=max(query_lengths_list),
            num_tokens=num_tokens,
        )
        attn_metadata._turboquant_query_layout_cache = (cumulative_lengths, device, layout)
        return layout

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
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

    def _profiled_forward(self, call: dict[str, Any]) -> torch.Tensor:
        profiler = self._open_profiler()
        if profiler is None:
            return super().forward(**call)
        logger.info_once(TURBOQUANT_PROFILE_NOTICE, TURBOQUANT_PROFILE_TRACE_DIR)
        with profiler as prof:
            attn_output = super().forward(**call)
            prof.step()
        return attn_output

    def _open_profiler(self) -> Any | None:
        if not getattr(self, "_profile", False) or _stream_is_capturing():
            return None
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
