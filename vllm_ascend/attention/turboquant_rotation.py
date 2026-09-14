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
"""TurboQuant's rotation Pi, and the one weight it is folded into.

    Pi x = D (H (D x)),   D = diag(+-1),   H = normalised Walsh-Hadamard

Pi is symmetric (``Pi^T = Pi``), orthogonal and an involution (``Pi^2 = I``).
K, V and Q are rotated as activations by the Ascend C kernels, which keeps RoPE
in the unrotated basis. The decode's attention output is therefore rotated too:

    O~ = softmax(q k^T) V~ = O Pi         (per head, row-vector convention)

and the inverse is folded into the output projection instead of being applied
by the kernel. With ``W_o`` of shape ``[hidden, H * D]`` acting as
``y = W_o o``, and ``o`` the concatenation of H head vectors,

    y = W_o (I_H (x) Pi) o~ = W_o' o~,     W_o' = W_o (I_H (x) Pi)

so every D-wide input block of every row of ``W_o`` is multiplied by Pi. That is
``Pi^T W_o = Pi W_o`` in the row-vector convention the design notes use. The fold
is block diagonal over heads, so it commutes with tensor-parallel sharding of
``o_proj``'s input dimension, and the bias is untouched.

It is only valid when ``o_proj`` consumes the attention output directly. An
elementwise output gate (Qwen3-Next / Qwen3.5 ``attn_output_gate``) does not
commute with Pi; those layers are never folded and the backend un-rotates their
output on the device instead.

This module imports torch and nothing else, so the offline folding tool
(``scripts/tq_fold_output_rotation.py``) runs on a host without torch_npu.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

# Seed of the +-1 diagonal of Pi.  It is fixed so that every rank, every layer
# and every restart agree on the rotation; the cache is only ever read back by
# the same transform that wrote it, and a folded checkpoint is only valid for
# the Pi it was folded with.
TURBOQUANT_PI_SEED = 0x5F3759DF

# The explicit LCG the sign vector is drawn from. Restated in
# csrc/tests/reference/turbo_quant_cpu.h::cpu_pi_sign_vector.
_LCG_MULTIPLIER = 1664525
_LCG_INCREMENT = 1013904223
_UINT32_MASK = 0xFFFFFFFF
_LCG_SIGN_BIT = 16

# The attribute of the model's HF config that records a folded checkpoint. The
# offline tool writes it into config.json next to the weights it rewrote, so the
# two travel together.
TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY = "turboquant_output_rotation"
TURBOQUANT_OUTPUT_ROTATION_FORMAT_VERSION = 1

# Acceptance bound for a fold: the worst per-sample cosine between the original
# projection of O and the folded projection of O~, with the folded weight in the
# dtype it is stored in.
TURBOQUANT_FOLD_MIN_COSINE = 0.9999

# Rotated outputs a validation draws when the caller does not say.
TURBOQUANT_FOLD_VALIDATION_SAMPLES = 64

_PI_SIGN_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def turboquant_pi_signs(head_size: int, device: torch.device) -> torch.Tensor:
    """Return the deterministic +-1 diagonal of Pi for ``head_size`` channels.

    Generated from an explicit LCG rather than ``torch.Generator`` so that the
    host C++ reference (``csrc/tests/reference/turbo_quant_cpu.h``) produces the
    identical vector without either side depending on the other's RNG.
    """
    key = (head_size, str(device))
    cached = _PI_SIGN_CACHE.get(key)
    if cached is not None:
        return cached
    state = (TURBOQUANT_PI_SEED + head_size) & _UINT32_MASK
    bits = []
    for _ in range(head_size):
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & _UINT32_MASK
        bits.append(1.0 if (state >> _LCG_SIGN_BIT) & 1 else -1.0)
    signs = torch.tensor(bits, dtype=torch.float32, device=device)
    _PI_SIGN_CACHE[key] = signs
    return signs


def walsh_hadamard(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Normalised fast Walsh-Hadamard transform along ``dim``.

    Host reference for the vectorised Ascend C transform; used by the tests, by
    the output-projection fold and by anyone reproducing the cache offline.
    """
    length = x.shape[dim]
    if length & (length - 1):
        raise ValueError(f"Walsh-Hadamard needs a power-of-two length, got {length}")
    out = x.movedim(dim, -1).contiguous()
    lead = out.shape[:-1]
    out = out.reshape(-1, length)
    stride = 1
    while stride < length:
        out = out.view(-1, length // (2 * stride), 2, stride)
        top = out[:, :, 0, :]
        bottom = out[:, :, 1, :]
        out = torch.stack((top + bottom, top - bottom), dim=2)
        out = out.reshape(-1, length)
        stride *= 2
    out = out.reshape(*lead, length) / (length**0.5)
    return out.movedim(-1, dim)


def apply_pi(x: torch.Tensor, pi_signs: torch.Tensor) -> torch.Tensor:
    """Host reference for ``Pi x = D (H (D x))`` over the last axis.

    Pi is its own inverse, so this is both the rotation and the un-rotation.
    """
    return walsh_hadamard(x * pi_signs, dim=-1) * pi_signs


def fold_pi_into_output_projection(weight: torch.Tensor, head_size: int) -> torch.Tensor:
    """Return ``W_o (I_H (x) Pi)`` for an ``o_proj`` weight of shape ``[hidden, H * D]``.

    Row ``r``'s input block for head ``h`` is ``W[r, h*D:(h+1)*D]``, and
    ``(W_h Pi)[r] = Pi W_h[r]`` because Pi is symmetric, so the fold is one
    ``apply_pi`` over the trailing axis of ``W.view(hidden, H, D)``.

    The arithmetic runs in float64 and the result is cast back to the weight's
    own dtype, so the only error a fold introduces is that dtype's rounding of
    the rotated values -- which is what :func:`validate_output_projection_fold`
    measures. The input is not modified.
    """
    if weight.dim() != 2:
        raise ValueError(f"an output projection weight is [hidden, num_heads * head_size], got {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(
            f"the output projection must be folded before it is quantised, got dtype {weight.dtype}; "
            "rotating an integer or scaled weight would need a requantisation this helper does not do"
        )
    in_features = weight.shape[1]
    if head_size <= 0 or head_size & (head_size - 1) or in_features % head_size:
        raise ValueError(
            f"head_size {head_size} must be a power of two dividing the projection's input width {in_features}"
        )
    num_heads = in_features // head_size
    signs = turboquant_pi_signs(head_size, weight.device).to(torch.float64)
    work = weight.detach().to(torch.float64).reshape(weight.shape[0], num_heads, head_size)
    folded = apply_pi(work, signs).reshape(weight.shape).to(weight.dtype)
    if not bool(torch.isfinite(folded).all()):
        raise ValueError(f"folding overflowed {weight.dtype}: a rotated weight does not fit the stored dtype")
    return folded


@dataclass(frozen=True)
class OutputProjectionFoldReport:
    """How closely a folded projection reproduces the original one."""

    # Worst per-sample cosine between W_o (Pi o~) and W_o' o~.
    min_cosine: float
    # Worst per-sample ||W_o' o~ - W_o (Pi o~)|| / ||W_o (Pi o~)||.
    max_relative_error: float
    # ||W_o' - W_o (I_H (x) Pi)||_F / ||W_o||_F, with W_o' as stored.
    weight_relative_error: float
    num_samples: int

    def passed(self, min_cosine: float = TURBOQUANT_FOLD_MIN_COSINE) -> bool:
        return self.min_cosine > min_cosine


def validate_output_projection_fold(
    weight: torch.Tensor,
    folded: torch.Tensor,
    head_size: int,
    num_samples: int = TURBOQUANT_FOLD_VALIDATION_SAMPLES,
    generator: torch.Generator | None = None,
) -> OutputProjectionFoldReport:
    """Check ``folded`` against ``weight`` the way the decode will exercise it.

    Draws rotated attention outputs ``o~`` and compares

        reference   W_o  (Pi o~)     the original weight, the un-rotated output
        candidate   W_o'  o~         the folded weight as stored, the kernel's output

    Both products run in float64 so the comparison sees the stored dtype's
    rounding of ``W_o'`` and nothing of its own. The bias is left out: the fold
    does not touch it and adding it back could only hide a difference.
    """
    if folded.shape != weight.shape:
        raise ValueError(f"folded shape {tuple(folded.shape)} does not match {tuple(weight.shape)}")
    in_features = weight.shape[1]
    num_heads = in_features // head_size
    signs = turboquant_pi_signs(head_size, weight.device).to(torch.float64)
    reference_weight = weight.detach().to(torch.float64)
    candidate_weight = folded.detach().to(torch.float64)

    rotated = torch.randn(
        num_samples, num_heads, head_size, dtype=torch.float64, device=weight.device, generator=generator
    )
    unrotated = apply_pi(rotated, signs)
    reference = unrotated.reshape(num_samples, in_features) @ reference_weight.T
    candidate = rotated.reshape(num_samples, in_features) @ candidate_weight.T

    cosine = torch.nn.functional.cosine_similarity(candidate, reference, dim=-1)
    relative = (candidate - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(torch.finfo(torch.float64).tiny)

    exact_fold = apply_pi(reference_weight.reshape(weight.shape[0], num_heads, head_size), signs).reshape(weight.shape)
    weight_error = (candidate_weight - exact_fold).norm() / reference_weight.norm().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return OutputProjectionFoldReport(
        min_cosine=float(cosine.min()),
        max_relative_error=float(relative.max()),
        weight_relative_error=float(weight_error),
        num_samples=num_samples,
    )


def output_rotation_marker(folded_modules: list[str], head_size: int) -> dict[str, Any]:
    """The config.json record of a folded checkpoint.

    ``folded_modules`` are attention module prefixes as they appear in the
    checkpoint -- ``model.layers.3.self_attn`` for ``model.layers.3.self_attn.o_proj.weight``.
    The seed and head size are recorded because a fold is only valid for the Pi
    it was made with, and nothing else in the checkpoint says which one that was.
    """
    return {
        "format_version": TURBOQUANT_OUTPUT_ROTATION_FORMAT_VERSION,
        "pi_seed": TURBOQUANT_PI_SEED,
        "head_size": head_size,
        "folded_modules": sorted(folded_modules),
    }


def output_rotation_is_folded(hf_config: Any, layer_name: str, head_size: int) -> bool:
    """Whether the attention layer ``layer_name`` feeds a Pi-folded ``o_proj``.

    False when the checkpoint carries no marker: the backend then un-rotates the
    decode output itself, which is correct for any model. When a marker IS
    present every mismatch raises rather than degrading, because the two ways of
    getting it wrong are both silent -- un-rotating into a folded projection, or
    not un-rotating into an unfolded one, each produce plausible-looking text.

    ``layer_name`` is vLLM's attention prefix, ``model.layers.3.self_attn.attn``;
    its parent is the module prefix the marker lists.
    """
    marker = getattr(hf_config, TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY, None)
    if marker is None:
        return False
    if not isinstance(marker, Mapping):
        raise ValueError(f"{TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY} must be a mapping, got {type(marker).__name__}")
    version = marker.get("format_version")
    if version != TURBOQUANT_OUTPUT_ROTATION_FORMAT_VERSION:
        raise ValueError(
            f"{TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY} format_version {version} is not "
            f"{TURBOQUANT_OUTPUT_ROTATION_FORMAT_VERSION}; refold the checkpoint with this vllm-ascend"
        )
    if marker.get("pi_seed") != TURBOQUANT_PI_SEED:
        raise ValueError(
            f"the checkpoint's o_proj was folded with Pi seed {marker.get('pi_seed')!r}, but this build rotates with "
            f"{TURBOQUANT_PI_SEED:#x}; the projection would un-rotate with the wrong transform"
        )
    if marker.get("head_size") != head_size:
        raise ValueError(
            f"the checkpoint's o_proj was folded for head_size {marker.get('head_size')!r}, "
            f"but layer {layer_name} has head_size {head_size}"
        )
    module_prefix = layer_name.rpartition(".")[0]
    if module_prefix in set(marker.get("folded_modules", ())):
        return True
    raise ValueError(
        f"the checkpoint declares folded output projections but {module_prefix!r} (from layer {layer_name!r}) is not "
        "among them. Either vLLM names this module differently from the checkpoint, or the fold skipped it; "
        "running it would feed a rotated output to an unfolded projection or the reverse."
    )
