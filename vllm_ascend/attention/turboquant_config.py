# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and launch geometry for TurboQuant attention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import torch

from vllm_ascend import envs
from vllm_ascend.attention.turboquant_layout import TURBOQUANT_E8_QUANT_MODE as TURBOQUANT_E8_QUANT_MODE

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

# Must match the C++ decode schema default. Explicit overrides must be passed
# to both the descriptor builder and launch; a descriptor alone cannot change it.
TURBOQUANT_FUSED_CONTEXT_LIMIT: Final[int] = 4096

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

TURBOQUANT_LM_REDUCTION_LANES: Final[int] = 64

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
    quant_mode: int = TURBOQUANT_QUANT_MODE

    @property
    def uses_e8(self) -> bool:
        return self.quant_mode == TURBOQUANT_E8_QUANT_MODE

    @classmethod
    def from_env(cls) -> TurboQuantOpsConfig:
        stage_selection = envs.TURBOQUANT_KERNEL_STAGES.strip().lower()
        if stage_selection in ("", "none", "0", "off"):
            kernel_stages: frozenset[TurboQuantKernelStage] = frozenset()
        elif stage_selection in ("all", "1", "on"):
            kernel_stages = frozenset(TurboQuantKernelStage)
        else:
            kernel_stages = cls._parse_stages(stage_selection)
        return cls(
            ops_namespace=envs.TURBOQUANT_OPS_NAMESPACE,
            kernel_stages=kernel_stages,
            quant_mode=turboquant_quant_mode(),
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


def turboquant_quant_mode() -> int:
    mode = envs.VLLM_ASCEND_TURBOQUANT_QUANT_MODE
    if mode not in (0, 1, TURBOQUANT_E8_QUANT_MODE):
        raise ValueError("VLLM_ASCEND_TURBOQUANT_QUANT_MODE must be 0 (uniform), 1 (Lloyd-Max), or 2 (E8)")
    return mode


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

TURBOQUANT_FIA_VERSION_ENV: Final[str] = "TURBOQUANT_FIA_VERSION"
