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
"""CPU kernels for the TurboQuant ``_C_ascend`` operators.

The compiled extension registers these operators for ``PrivateUse1`` alone, so on a host
without an NPU the first call out of :mod:`vllm_ascend.attention.turboquant_v1` fails before
any of the Python orchestration around it has run. :func:`turboquant_cpu_ops` defines the
extension's own schemas through :mod:`torch.library` and gives each operator a CPU kernel that

* raises the ``RuntimeError`` the adapter's ``TORCH_CHECK`` raises
  (``csrc/attention/turboquant/op_adapter/turboquant_torch_adpt.h``), with the same message;
* also refuses what the adapter reads through ``data_ptr`` without checking -- a strided
  ``slot_mapping``, ``block_tables`` or ``context_lens``, a value whose dtype is not the key's,
  a slot or block outside the cache -- because on the device those are silent misreads;
* computes the device's answer in the device's layout: Pi on K and V, Lloyd-Max codes packed
  low nibble first with a -128 bias, the K scales then the V scales in a zero-padded
  burst-aligned slot per token, negative slots skipped, and the decode output left rotated.

A harness running prefill and decode against these can therefore compare numbers rather than
only check that nothing raised. :func:`cpu_fused_infer_attention_score` is the matching
stand-in for the dense prefill.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator, Sequence

import torch
from torch.library import Library

from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_LSE_MAX_LANE,
    TURBOQUANT_LSE_STRIDE,
    TURBOQUANT_LSE_SUM_LANE,
)
from vllm_ascend.attention.turboquant_rotation import apply_pi
from vllm_ascend.attention.turboquant_v1 import TURBOQUANT_LLOYD_MAX_THRESHOLDS

# What the triton stub in tests/ut/conftest.py reports as num_vectorcore.
CPU_VECTOR_CORES = 8

# csrc/attention/turboquant/op_adapter/turboquant_torch_ops.h's definitions, whitespace collapsed.
TURBOQUANT_OP_SCHEMAS = {
    "npu_turboquant_reshape_and_cache": (
        "npu_turboquant_reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, "
        "Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, Tensor codec_tables) -> ()"
    ),
    "npu_turboquant_rotate_q": (
        "npu_turboquant_rotate_q(Tensor query, Tensor pi_signs, Tensor codec_tables, Tensor hadamard16, "
        "Tensor! query_rot) -> ()"
    ),
    "npu_turboquant_paged_attention": (
        "npu_turboquant_paged_attention(Tensor query_rot, Tensor key_cache, Tensor value_cache, "
        "Tensor scale_cache, Tensor block_tables, Tensor context_lens, Tensor codec_tables, Tensor! workspace, "
        "int num_kv_heads, int num_heads, float scale_value, Tensor! out, Tensor(a!)? lse=None) -> ()"
    ),
    "npu_turboquant_vector_core_num": "npu_turboquant_vector_core_num() -> int",
    "npu_turboquant_workspace_size": (
        "npu_turboquant_workspace_size(int num_tokens, int num_heads, int head_size, int max_blocks_per_seq, "
        "int block_size) -> int"
    ),
}

# The kv4fp8 Cube path's definitions, which only an Ascend 950 build registers. They have no CPU
# stand-in here, and leaving them undefined keeps the CPU pass on the AIV decode it reproduces.
TURBOQUANT_CUBE_OP_SCHEMAS = {
    "npu_turboquant_cube_reshape_and_cache": (
        "npu_turboquant_cube_reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, "
        "Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, Tensor codec_tables) -> ()"
    ),
    "npu_turboquant_cube_decode": (
        "npu_turboquant_cube_decode(Tensor query, Tensor? gate, Tensor pi_signs, Tensor codec_tables, "
        "Tensor hadamard16, Tensor key_cache, Tensor value_cache, Tensor scale_cache, Tensor block_tables, "
        "Tensor context_lens, Tensor! workspace, Tensor! query_rot, int num_kv_heads, int num_heads, "
        "float scale_value, int output_stage, Tensor! out, Tensor(a!)? lse=None) -> ()"
    ),
    "npu_turboquant_cube_workspace_size": (
        "npu_turboquant_cube_workspace_size(int num_tokens, int num_heads, int num_kv_heads, int head_size, "
        "int max_blocks_per_seq, int block_size) -> int"
    ),
}

_NAMESPACE = "_C_ascend"
# The two device queries take no tensor, so there is no backend key to dispatch on; the
# extension registers them as catch-all kernels too.
_DEVICE_QUERY_DISPATCH_KEY = "CompositeExplicitAutograd"

# turboquant_adpt
_MAX_SEQUENCE_SPLITS = 8
_FUSED_CONTEXT_LIMIT = 4096
_PARTIAL_TAIL = 16
_FP32_PER_BLOCK = 8
_TILE_ROWS = 16
_CODEC_LEVELS = 16
_ROTATE_TILE = 16
_MIN_HEAD_SIZE = 64
_MAX_HEAD_SIZE = 256

# TurboQuantCodec<4>
_NIBBLE_RADIX = 16
_INT8_BIAS = 128
_EPS = 1e-20

_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)
_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE = 3


def _check(condition: bool, *message: object) -> None:
    """``TORCH_CHECK``, which surfaces in Python as a ``RuntimeError``."""
    if not condition:
        raise RuntimeError("".join(str(part) for part in message))


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def scale_slot_floats(num_kv_heads: int) -> int:
    return _ceil_div(2 * num_kv_heads, _FP32_PER_BLOCK) * _FP32_PER_BLOCK


def codec_table_words(head_size: int, batch_rows: int) -> int:
    return 7 * head_size + 2 * head_size * batch_rows + _CODEC_LEVELS


def paged_attention_workspace_floats(
    num_tokens: int, num_heads: int, head_size: int, max_blocks_per_seq: int, block_size: int, vector_cores: int
) -> int:
    """``PlanPagedAttention(...).workspace_floats``."""
    base_tasks = num_tokens * num_heads
    if base_tasks <= 0:
        return 0
    blocks = max(1, max_blocks_per_seq)
    if blocks * block_size <= _FUSED_CONTEXT_LIMIT:
        return 0
    num_splits = max(_ceil_div(vector_cores, base_tasks), _ceil_div(blocks * block_size, _FUSED_CONTEXT_LIMIT))
    num_splits = min(num_splits, min(_MAX_SEQUENCE_SPLITS, blocks))
    if num_splits <= 1:
        return 0
    return base_tasks * num_splits * (head_size + _PARTIAL_TAIL)


def _check_head_size(head_size: int) -> None:
    _check(
        _MIN_HEAD_SIZE <= head_size <= _MAX_HEAD_SIZE, "TurboQuant requires 64 <= head_size <= 256, got ", head_size
    )
    _check(
        head_size & (head_size - 1) == 0,
        "TurboQuant needs a power-of-two head_size for the Walsh-Hadamard transform, got ",
        head_size,
    )


def _check_codec_tables(tables: torch.Tensor, head_size: int, batch_rows: int) -> None:
    _check(tables.dtype == torch.int32, "codec tables must be int32")
    _check(tables.is_contiguous(), "codec tables must be contiguous")
    words = codec_table_words(head_size, batch_rows)
    _check(
        tables.numel() == words,
        "codec tables must hold ", words, " words for head_size ", head_size, " and batch_rows ", batch_rows,
        ", got ", tables.numel(),
    )  # fmt: skip


def _check_activation_dtype(dtype: torch.dtype) -> None:
    """``ToAscendType``: evaluated at launch, so only once there is something to launch."""
    _check(dtype in _ACTIVATION_DTYPES, "TurboQuant KV cache supports float16 and bfloat16 activations, got ", dtype)


def quantize(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``Quantize4Bit`` over the last axis: RMS scale, strict Lloyd-Max bins, low nibble first."""
    head_size = vectors.shape[-1]
    scale = vectors.pow(2).sum(dim=-1, keepdim=True).sqrt() / head_size**0.5 + _EPS
    thresholds = torch.tensor(TURBOQUANT_LLOYD_MAX_THRESHOLDS, dtype=vectors.dtype)
    bins = ((vectors / scale).unsqueeze(-1) > thresholds).sum(dim=-1)
    packed = bins[..., 0::2] + _NIBBLE_RADIX * bins[..., 1::2] - _INT8_BIAS
    return packed.to(torch.int8), scale.squeeze(-1)


def dequantize(packed: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """``Dequantize4Bit``: one bare level per channel, before the per-vector scale."""
    byte = packed.to(torch.int64) + _INT8_BIAS
    codes = torch.stack((byte % _NIBBLE_RADIX, byte // _NIBBLE_RADIX), dim=-1).flatten(-2)
    return centroids[codes]


def _reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pi_signs: torch.Tensor,
    codec_tables: torch.Tensor,
) -> None:
    _check(key.dim() == 3 and value.dim() == 3, "key/value must be [num_tokens, num_kv_heads, head_size]")
    _check(key.shape == value.shape, "key and value must have the same shape")
    _check(
        key_cache.dim() == 4 and value_cache.dim() == 4,
        "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]",
    )
    _check(key_cache.shape == value_cache.shape, "key and value caches must have the same shape")
    _check(
        key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8, "the packed 4-bit kv cache must be int8"
    )
    _check(scale_cache.dtype == torch.float32, "the kv cache scale plane must be float32")
    _check(scale_cache.dim() == 3, "scale cache must be [num_blocks, block_size, scale_slot]")
    _check(slot_mapping.dtype == torch.int32, "slot_mapping must be int32")
    _check(pi_signs.dtype == torch.float32, "pi_signs must be float32")
    _check(key.is_contiguous() and value.is_contiguous(), "key/value must be contiguous")
    _check(
        key_cache.is_contiguous() and value_cache.is_contiguous() and scale_cache.is_contiguous(),
        "the kv cache and its scale plane must be contiguous",
    )

    num_tokens, num_kv_heads, head_size = key.shape
    num_blocks, block_size = key_cache.shape[:2]
    _check_head_size(head_size)
    _check_codec_tables(codec_tables, head_size, 1)
    _check(key_cache.shape[2] == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.shape[2], " vs ", num_kv_heads)
    _check(key_cache.shape[3] == head_size // 2, "packed cache head dim must be head_size / 2, got ", key_cache.shape[3])
    _check(block_size % _TILE_ROWS == 0, "TurboQuant requires block_size to be a multiple of 16, got ", block_size)
    _check(slot_mapping.numel() == num_tokens, "slot_mapping must hold one slot per token")
    _check(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel")
    _check(
        scale_cache.shape[0] == num_blocks and scale_cache.shape[1] == block_size,
        "the scale plane must be shaped to the same blocks as the cache",
    )
    slot_floats = scale_slot_floats(num_kv_heads)
    _check(
        scale_cache.shape[2] == slot_floats,
        "scale slot must be round_up(2 * num_kv_heads, 8) = ", slot_floats, ", got ", scale_cache.shape[2],
    )  # fmt: skip

    # Read through data_ptr without a check on the device.
    _check(value.dtype == key.dtype, "value dtype ", value.dtype, " is not the key dtype ", key.dtype)
    _check(slot_mapping.is_contiguous(), "slot_mapping must be contiguous")

    if num_tokens == 0:
        return
    _check_activation_dtype(key.dtype)

    slots = slot_mapping.to(torch.int64)
    capacity = num_blocks * block_size
    _check(bool((slots < capacity).all()), "slot_mapping names a slot past the end of the ", capacity, "-slot cache")
    written = slots >= 0
    rows = slots[written]
    planes = torch.cat((key, value), dim=1)[written].to(torch.float32)
    packed, scales = quantize(apply_pi(planes, pi_signs))

    packed_shape = (-1, num_kv_heads, head_size // 2)
    key_cache.view(packed_shape)[rows] = packed[:, :num_kv_heads]
    value_cache.view(packed_shape)[rows] = packed[:, num_kv_heads:]
    token_scales = torch.zeros(rows.numel(), slot_floats, dtype=torch.float32)
    token_scales[:, : 2 * num_kv_heads] = scales
    scale_cache.view(-1, slot_floats)[rows] = token_scales


def _rotate_q(
    query: torch.Tensor,
    pi_signs: torch.Tensor,
    codec_tables: torch.Tensor,
    hadamard16: torch.Tensor,
    query_rot: torch.Tensor,
) -> None:
    _check(query.dim() == 3, "query must be [num_tokens, num_heads, head_size]")
    _check(query.is_contiguous(), "query must be contiguous")
    _check(query.dtype in _ACTIVATION_DTYPES, "query must be float16 or bfloat16, got ", query.dtype)
    _check(query_rot.shape == query.shape, "query_rot must have the same shape as query")
    _check(query_rot.dtype == torch.float32, "query_rot must be float32, got ", query_rot.dtype)
    _check(query_rot.is_contiguous(), "query_rot must be contiguous")
    _check(pi_signs.dtype == torch.float32, "pi_signs must be float32")
    _check(hadamard16.dtype == torch.float16, "hadamard16 must be float16")
    h16_elements = _ROTATE_TILE * _ROTATE_TILE
    _check(
        hadamard16.is_contiguous() and hadamard16.numel() == h16_elements,
        "hadamard16 must be a contiguous ", h16_elements, "-element tensor, got ", hadamard16.numel(),
    )  # fmt: skip
    _check(codec_tables.dtype == torch.int32, "codec tables must be int32")
    _check(codec_tables.is_contiguous(), "codec tables must be contiguous")

    head_size = query.shape[2]
    _check_head_size(head_size)
    _check(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel")
    words = codec_table_words(head_size, 1)
    _check(
        codec_tables.numel() >= words,
        "codec tables must hold at least ", words, " words for head_size ", head_size, ", got ", codec_tables.numel(),
    )  # fmt: skip

    if query.numel() == 0:
        return
    query_rot.copy_(apply_pi(query.to(torch.float32), pi_signs))


def _paged_attention(
    query_rot: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    codec_tables: torch.Tensor,
    workspace: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale_value: float,
    out: torch.Tensor,
    lse: torch.Tensor | None = None,
    *,
    vector_cores: int,
) -> None:
    _check(query_rot.dim() == 3, "query_rot must be [num_tokens, num_heads, head_size]")
    _check(query_rot.dtype == torch.float32, "query_rot must be float32, got ", query_rot.dtype)
    _check(out.shape == query_rot.shape, "out must have the same shape as query_rot")
    _check(out.dtype in _ACTIVATION_DTYPES, "out must be float16 or bfloat16, got ", out.dtype)
    _check(
        key_cache.dim() == 4 and value_cache.dim() == 4,
        "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]",
    )
    _check(
        key_cache.dtype == torch.int8 and value_cache.dtype == torch.int8, "the packed 4-bit kv cache must be int8"
    )
    _check(
        scale_cache.dtype == torch.float32 and scale_cache.dim() == 3,
        "the scale plane must be a float32 [num_blocks, block_size, scale_slot] tensor",
    )
    _check(
        block_tables.dim() == 2 and block_tables.dtype == torch.int32,
        "block_tables must be an int32 [num_tokens, max_blocks_per_seq] tensor",
    )
    _check(context_lens.dtype == torch.int32, "context_lens must be int32")
    _check(query_rot.is_contiguous() and out.is_contiguous(), "query_rot and out must be contiguous")
    _check(
        key_cache.is_contiguous() and value_cache.is_contiguous() and scale_cache.is_contiguous(),
        "the kv cache and its scale plane must be contiguous",
    )

    num_tokens, _, head_size = query_rot.shape
    num_blocks, block_size = key_cache.shape[:2]
    max_blocks_per_seq = block_tables.shape[1]
    _check_head_size(head_size)
    _check_codec_tables(codec_tables, head_size, _TILE_ROWS)
    _check(
        query_rot.shape[1] == num_heads,
        "query head count ", query_rot.shape[1], " does not match num_heads ", num_heads,
    )  # fmt: skip
    _check(
        num_kv_heads > 0 and num_heads % num_kv_heads == 0,
        "num_heads ", num_heads, " must be a multiple of num_kv_heads ", num_kv_heads,
    )  # fmt: skip
    _check(key_cache.shape[2] == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.shape[2], " vs ", num_kv_heads)
    _check(key_cache.shape[3] == head_size // 2, "packed cache head dim must be head_size / 2, got ", key_cache.shape[3])
    _check(
        block_size % _TILE_ROWS == 0,
        "TurboQuant requires block_size to be a multiple of ", _TILE_ROWS, ", got ", block_size,
    )  # fmt: skip
    slot_floats = scale_slot_floats(num_kv_heads)
    _check(
        scale_cache.shape[2] == slot_floats,
        "scale slot must be round_up(2 * num_kv_heads, 8) = ", slot_floats, ", got ", scale_cache.shape[2],
    )  # fmt: skip
    _check(block_tables.shape[0] == num_tokens, "block_tables must hold one row per query token")
    _check(context_lens.numel() == num_tokens, "context_lens must hold one length per query token")

    # Read through data_ptr without a check on the device.
    _check(value_cache.shape == key_cache.shape, "key and value caches must have the same shape")
    _check(
        block_tables.is_contiguous() and context_lens.is_contiguous(),
        "block_tables and context_lens must be contiguous",
    )

    if num_tokens == 0:
        return

    needed = paged_attention_workspace_floats(
        num_tokens, num_heads, head_size, max_blocks_per_seq, block_size, vector_cores
    )
    _check(workspace.dtype == torch.float32, "the decode workspace must be float32, got ", workspace.dtype)
    _check(workspace.is_contiguous(), "the decode workspace must be contiguous")
    _check(
        workspace.numel() >= needed,
        "the decode workspace holds ", workspace.numel(), " float32 words but this launch needs ", needed,
        "; size it with npu_turboquant_workspace_size",
    )  # fmt: skip
    _check_activation_dtype(out.dtype)

    lengths = context_lens.to(torch.int64)
    tables = block_tables.to(torch.int64)
    capacity = max_blocks_per_seq * block_size
    _check(
        bool(((lengths >= 0) & (lengths <= capacity)).all()),
        "context_lens must lie in [0, max_blocks_per_seq * block_size = ", capacity, "]",
    )

    centroid_start = codec_table_words(head_size, _TILE_ROWS) - _CODEC_LEVELS
    centroids = codec_tables[centroid_start:].view(torch.float32)
    packed_shape = (-1, num_kv_heads, head_size // 2)
    keys = key_cache.view(packed_shape)
    values = value_cache.view(packed_shape)
    token_scales = scale_cache.view(-1, slot_floats)
    kv_head_of = torch.arange(num_heads) // (num_heads // num_kv_heads)

    if lse is not None:
        _check(lse.dtype == torch.float32, "lse must be float32, got ", lse.dtype)
        _check(lse.is_contiguous(), "lse must be contiguous")
        _check(
            tuple(lse.shape) == (num_tokens, num_heads, TURBOQUANT_LSE_STRIDE),
            "lse must be [", num_tokens, ", ", num_heads, ", ", TURBOQUANT_LSE_STRIDE, "], got ",
            tuple(lse.shape),
        )  # fmt: skip
        lse.zero_()

    for token in range(num_tokens):
        length = int(lengths[token])
        if length == 0:
            out[token].zero_()
            continue
        positions = torch.arange(length)
        physical = tables[token, positions // block_size]
        _check(
            bool(((physical >= 0) & (physical < num_blocks)).all()),
            "block_tables row ", token, " names a block outside the ", num_blocks, "-block cache",
        )  # fmt: skip
        rows = physical * block_size + positions % block_size
        k_levels = dequantize(keys[rows], centroids)[:, kv_head_of]
        v_levels = dequantize(values[rows], centroids)[:, kv_head_of]
        k_scale = token_scales[rows, :num_kv_heads][:, kv_head_of]
        v_scale = token_scales[rows, num_kv_heads : 2 * num_kv_heads][:, kv_head_of]
        scores = (k_levels * query_rot[token]).sum(dim=-1) * k_scale * scale_value
        weights = torch.softmax(scores, dim=0) * v_scale
        out[token] = torch.einsum("lh,lhd->hd", weights, v_levels).to(out.dtype)
        if lse is not None:
            # The same pair the online softmax leaves in its accumulator lanes: the running
            # max over the whole context and sum_i exp(s_i - max). Computed in one pass here
            # because the whole context is already materialised; the kernel reaches the same
            # numbers tile by tile.
            running_max = scores.amax(dim=0)
            lse[token, :, TURBOQUANT_LSE_MAX_LANE] = running_max
            lse[token, :, TURBOQUANT_LSE_SUM_LANE] = torch.exp(scores - running_max).sum(dim=0)


def _workspace_size(
    num_tokens: int, num_heads: int, head_size: int, max_blocks_per_seq: int, block_size: int, *, vector_cores: int
) -> int:
    _check(
        num_tokens >= 0 and num_heads > 0,
        "workspace sizing needs num_tokens >= 0 and num_heads > 0, got ", num_tokens, " and ", num_heads,
    )  # fmt: skip
    _check(block_size > 0, "workspace sizing needs block_size > 0, got ", block_size)
    _check_head_size(head_size)
    return paged_attention_workspace_floats(
        num_tokens, num_heads, head_size, max_blocks_per_seq, block_size, vector_cores
    )


def cpu_fused_infer_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    atten_mask: torch.Tensor | None,
    input_layout: str,
    actual_seq_lengths: Sequence[int],
    actual_seq_lengths_kv: Sequence[int],
    num_key_value_heads: int,
    num_heads: int,
    scale: float,
    sparse_mode: int,
) -> tuple[torch.Tensor, None]:
    """``torch_npu.npu_fused_infer_attention_score`` for the call ``PrefillNoCache`` makes.

    That call is ``TND`` with cumulative lengths shared by q and kv, and ``sparse_mode=3``, a
    bottom-right causal mask the operator builds itself. Anything else raises rather than being
    approximated, so a change to the call shows up here first.
    """
    _check(input_layout == "TND", "the CPU stand-in implements input_layout TND, got ", input_layout)
    _check(
        sparse_mode == _BOTTOM_RIGHT_CAUSAL_SPARSE_MODE,
        "the CPU stand-in implements sparse_mode ", _BOTTOM_RIGHT_CAUSAL_SPARSE_MODE, ", got ", sparse_mode,
    )  # fmt: skip
    ends = [int(end) for end in actual_seq_lengths]
    _check(ends == [int(end) for end in actual_seq_lengths_kv], "a no-cache prefill has kv lengths equal to q lengths")
    _check(query.dim() == 3 and key.dim() == 3 and value.dim() == 3, "TND inputs are [tokens, heads, head_size]")
    _check(query.shape[0] == key.shape[0] == value.shape[0], "q, k and v must hold the same tokens")
    _check(query.shape[1] == num_heads, "query holds ", query.shape[1], " heads, num_heads is ", num_heads)
    _check(
        key.shape[1] == value.shape[1] == num_key_value_heads and num_heads % num_key_value_heads == 0,
        "key/value hold ", key.shape[1], " heads, num_key_value_heads is ", num_key_value_heads,
    )  # fmt: skip
    _check(
        bool(ends) and ends == sorted(ends) and ends[-1] == query.shape[0],
        "actual_seq_lengths must be cumulative and end at the ", query.shape[0], " query tokens, got ", ends,
    )  # fmt: skip

    group = num_heads // num_key_value_heads
    output = torch.empty(query.shape[0], num_heads, value.shape[2], dtype=query.dtype)
    start = 0
    for end in ends:
        q = query[start:end].to(torch.float32).transpose(0, 1)
        k = key[start:end].to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0)
        v = value[start:end].to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        output[start:end] = attended.transpose(0, 1).to(query.dtype)
        start = end
    return output, None


@contextlib.contextmanager
def turboquant_cpu_ops(vector_cores: int = CPU_VECTOR_CORES) -> Iterator[None]:
    """Serve ``torch.ops._C_ascend.npu_turboquant_*`` from the CPU for the duration of the block.

    A schema is defined only where no loaded extension already defines it, and the two device
    queries get kernels only in that case -- an extension answers them itself. Leaving the block
    removes every definition and kernel it added, so nothing leaks into tests that replace
    ``torch.ops._C_ascend`` with a mock.
    """
    library = Library(_NAMESPACE, "FRAGMENT")
    try:
        namespace = getattr(torch.ops, _NAMESPACE)
        defined = set()
        for name, schema in TURBOQUANT_OP_SCHEMAS.items():
            if not hasattr(namespace, name):
                library.define(schema)
                defined.add(name)
        library.impl("npu_turboquant_reshape_and_cache", _reshape_and_cache, "CPU")
        library.impl("npu_turboquant_rotate_q", _rotate_q, "CPU")
        library.impl(
            "npu_turboquant_paged_attention", functools.partial(_paged_attention, vector_cores=vector_cores), "CPU"
        )
        if "npu_turboquant_vector_core_num" in defined:
            library.impl("npu_turboquant_vector_core_num", lambda: vector_cores, _DEVICE_QUERY_DISPATCH_KEY)
        if "npu_turboquant_workspace_size" in defined:
            library.impl(
                "npu_turboquant_workspace_size",
                functools.partial(_workspace_size, vector_cores=vector_cores),
                _DEVICE_QUERY_DISPATCH_KEY,
            )
        yield
    finally:
        library._destroy()


# What the stand-in for npu_turboquant_cube_workspace_size answers. Sizing is the Cube planner's arithmetic,
# which this module does not reproduce, so a meta pass hands the decode an empty workspace.
META_CUBE_WORKSPACE_FLOATS = 0


def _cube_reshape_and_cache_meta(key, value, key_cache, value_cache, scale_cache, slot_mapping, pi_signs, codec_tables):
    torch._check(key.dim() == 3 and key.shape == value.shape, lambda: "key and value must be [num_tokens, H_KV, D]")
    torch._check(key_cache.dim() == 4 and key_cache.shape == value_cache.shape, lambda: "K and V caches differ")
    torch._check(key_cache.shape[2] == key.shape[1], lambda: "cache num_kv_heads mismatch")
    torch._check(key_cache.shape[3] * 2 == key.shape[2], lambda: "packed cache head dim must be head_size / 2")
    torch._check(scale_cache.shape[:2] == key_cache.shape[:2], lambda: "scale plane must match the cache blocks")
    torch._check(slot_mapping.numel() == key.shape[0], lambda: "slot_mapping must hold one slot per token")
    torch._check(pi_signs.numel() == key.shape[2], lambda: "pi_signs must hold head_size signs")


def _cube_decode_meta(
    query,
    gate,
    pi_signs,
    codec_tables,
    hadamard16,
    key_cache,
    value_cache,
    scale_cache,
    block_tables,
    context_lens,
    workspace,
    query_rot,
    num_kv_heads,
    num_heads,
    scale_value,
    output_stage,
    out,
    lse=None,
):
    torch._check(query.dim() == 3, lambda: "query must be [num_tokens, num_heads, head_size]")
    torch._check(out.shape == query.shape, lambda: "out must have the same shape as query")
    torch._check(query_rot.dtype == torch.float32, lambda: "query_rot must be float32")
    torch._check(query_rot.numel() == query.numel(), lambda: "query_rot must hold one fp32 word per query element")
    torch._check(gate is None or gate.numel() == query.numel(), lambda: "gate must hold one logit per query element")
    torch._check(query.shape[1] == num_heads and num_heads % num_kv_heads == 0, lambda: "head counts mismatch")
    torch._check(key_cache.shape == value_cache.shape and key_cache.shape[2] == num_kv_heads, lambda: "cache heads")
    torch._check(block_tables.dim() == 2 and block_tables.shape[0] == query.shape[0], lambda: "one table row per token")
    torch._check(context_lens.numel() == query.shape[0], lambda: "one context length per query token")
    torch._check(
        lse is None or tuple(lse.shape) == (query.shape[0], num_heads, TURBOQUANT_LSE_STRIDE),
        lambda: "lse must be [num_tokens, num_heads, TURBOQUANT_LSE_STRIDE]",
    )


@contextlib.contextmanager
def turboquant_cube_meta_ops() -> Iterator[None]:
    """Define the Cube operators' pinned schemas with ``Meta`` kernels only, for the block's duration.

    The kernels check the shapes that tie one call together and write nothing: both operators return ``()`` and
    mutate their outputs in place, so a meta pass has no tensor to produce. There is deliberately no CPU kernel;
    what the launch computes is verified on the camodel (``test_sim_950pr_turboquant_fused``), not here.
    """
    library = Library(_NAMESPACE, "FRAGMENT")
    try:
        namespace = getattr(torch.ops, _NAMESPACE)
        defined = set()
        for name, schema in TURBOQUANT_CUBE_OP_SCHEMAS.items():
            if not hasattr(namespace, name):
                library.define(schema)
                defined.add(name)
        if "npu_turboquant_cube_reshape_and_cache" in defined:
            library.impl("npu_turboquant_cube_reshape_and_cache", _cube_reshape_and_cache_meta, "Meta")
        if "npu_turboquant_cube_decode" in defined:
            library.impl("npu_turboquant_cube_decode", _cube_decode_meta, "Meta")
        if "npu_turboquant_cube_workspace_size" in defined:
            library.impl(
                "npu_turboquant_cube_workspace_size",
                lambda *sizes: META_CUBE_WORKSPACE_FLOATS,
                _DEVICE_QUERY_DISPATCH_KEY,
            )
        yield
    finally:
        library._destroy()
