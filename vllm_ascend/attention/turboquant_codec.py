# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference codec and reusable rotation operands."""

from __future__ import annotations

import math
from typing import Final

import torch

from vllm_ascend.attention.turboquant_config import (
    _FP32_EXPONENT_BIAS,
    _FP32_EXPONENT_MASK,
    _FP32_MANTISSA_BITS,
    _FP32_MAX_BIASED_EXPONENT,
    _LCG_INCREMENT,
    _LCG_MODULUS_MASK,
    _LCG_MULTIPLIER,
    _LCG_SIGN_BIT,
    TURBOQUANT_H16_ORDER,
    TURBOQUANT_HEADROOM_BITS,
    TURBOQUANT_LM_CENTROIDS,
    TURBOQUANT_LM_EPS,
    TURBOQUANT_LM_REDUCTION_LANES,
    TURBOQUANT_LM_THRESHOLDS,
    TURBOQUANT_NIBBLE_LEVELS,
    TURBOQUANT_NIBBLE_MASK,
    TURBOQUANT_NIBBLE_MAX,
    TURBOQUANT_NIBBLE_MIN,
    TURBOQUANT_PI_SEED,
    TURBOQUANT_QUANT_MODE,
    TURBOQUANT_TABLE_DTYPE,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_GROUP,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_PLANES,
)


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
    """Reference codec for the configured uniform or Lloyd-Max cache codebook."""

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
        self._h16: torch.Tensor | None = None

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
        if self._h16 is None:
            self._h16 = _sylvester_hadamard(TURBOQUANT_H16_ORDER, self.device, torch.float16)
        return self._h16

    @staticmethod
    def _hadamard_last_dim(values: torch.Tensor) -> torch.Tensor:
        """Use a fixed FP32 butterfly order, independent of batch/GEMM shape."""
        shape = values.shape
        stride = 1
        while stride < shape[-1]:
            pairs = values.reshape(*shape[:-1], -1, 2, stride)
            low, high = pairs.unbind(-2)
            values = torch.cat((low + high, low - high), dim=-1).reshape(shape)
            stride *= 2
        return values

    def rotate(self, activations: torch.Tensor) -> torch.Tensor:
        # Factor Pi as outer Hadamard @ signs @ inner Hadamard. A dense GEMM
        # can change FP32 accumulation order when a writer splits/pads a batch.
        panel = activations.to(torch.float32).reshape(*activations.shape[:-1], self.groups, self.group)
        panel = self._hadamard_last_dim(panel) * self.pi_signs.reshape(self.groups, self.group)
        return self._hadamard_last_dim(panel.transpose(-1, -2)).transpose(-1, -2).reshape(activations.shape)

    def inverse_rotate(self, rotated: torch.Tensor) -> torch.Tensor:
        panel = rotated.to(torch.float32).reshape(*rotated.shape[:-1], self.groups, self.group)
        panel = self._hadamard_last_dim(panel.transpose(-1, -2)).transpose(-1, -2)
        panel = self._hadamard_last_dim(panel * self.pi_signs.reshape(self.groups, self.group))
        return panel.reshape(rotated.shape) * self.inv_sqrt_len

    def quantize_uniform(self, rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def dequantize_uniform(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)

    @staticmethod
    def _lloyd_max_sumsq(row: torch.Tensor, fold: str) -> torch.Tensor:
        """Match the key butterfly or value pairwise reduction's FP32 rounding."""
        row_fp32 = row.to(torch.float32)
        squared_values = row_fp32 * row_fp32
        num_channels = squared_values.numel()
        if num_channels == 0 or num_channels & (num_channels - 1):
            raise ValueError("Lloyd-Max reduction needs a nonempty power-of-two channel count")

        def sum_pairwise_halves(squared_values: torch.Tensor, stop: int) -> torch.Tensor:
            width = squared_values.numel()
            while width > stop:
                width //= 2
                squared_values = squared_values[:width] + squared_values[width : 2 * width]
            return squared_values

        if fold == "value":
            squared_values = sum_pairwise_halves(squared_values, 1)
        elif fold in ("key", "butterfly"):
            lanes = min(num_channels, TURBOQUANT_LM_REDUCTION_LANES)
            chunks = squared_values.reshape(-1, lanes).unbind(0)
            squared_values = chunks[0]
            # Keep the production D=256 left-to-right quarter fold unchanged.
            for chunk in chunks[1:]:
                squared_values = squared_values + chunk
            indices = torch.arange(lanes, device=row.device)
            stride = 1
            while stride < lanes:
                squared_values = squared_values + squared_values[indices ^ stride]
                stride *= 2
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
        return codes.reshape(rotated_fp32.shape), scales.reshape(rotated_fp32.shape[:-1])

    def dequantize_lloyd_max(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """``centroid[codes + 8] * scales`` in fp32: the codebook reconstruction."""
        centroids = torch.tensor(TURBOQUANT_LM_CENTROIDS, dtype=torch.float32, device=codes.device)
        return centroids[(codes.to(torch.int16) + 8).long()] * scales.to(torch.float32).unsqueeze(-1)

    def quantize(self, rotated: torch.Tensor, fold: str = "key") -> tuple[torch.Tensor, torch.Tensor]:
        """Encode cache rows using the same quantizer as the active writers."""
        if TURBOQUANT_QUANT_MODE == 1:
            return self.quantize_lloyd_max(rotated, fold)
        return self.quantize_uniform(rotated)

    def dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Reconstruct the active cache codebook in fp32."""
        if TURBOQUANT_QUANT_MODE == 1:
            return self.dequantize_lloyd_max(codes, scales)
        return self.dequantize_uniform(codes, scales)

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

    def encode(self, activations: torch.Tensor, *, fold: str = "key") -> tuple[torch.Tensor, torch.Tensor]:
        """Pack rows; batch V uses ``value``, scatter K/V both use ``key``."""
        codes, scales = self.quantize(self.rotate(activations), fold)
        return self.pack(codes), scales

    def decode(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self.inverse_rotate(self.dequantize(self.unpack(packed), scales))


def _build_codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    device = emulator.device
    h_inner = _sylvester_hadamard(emulator.group, device, torch.float32)
    signs = emulator.pi_signs.reshape(emulator.groups, emulator.group)
    positive = signs.unsqueeze(-1) * h_inner.unsqueeze(0)
    tables = torch.stack((positive, -positive), dim=0)
    return tables.reshape(TURBOQUANT_PLANES * emulator.groups, emulator.group, emulator.group).to(
        TURBOQUANT_TABLE_DTYPE
    )
