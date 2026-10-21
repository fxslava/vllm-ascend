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
"""Uncompressed attention sinks beside the TurboQuant 4-bit KV cache.

The first few tokens of a sequence carry a disproportionate share of every later
token's softmax mass, and 4-bit Lloyd-Max codes are worst exactly there: the
error on one sink vector is multiplied by the weight the sink attracts.  Keeping
``N`` of them at activation precision and folding their exact contribution back
into the decode is what restores retrieval under 4-bit KV.

Layout -- Option A, a dedicated side-car plane
==============================================

``TurboQuantSinkCache`` holds ``[2, num_blocks, N, num_kv_heads, head_size]`` in the
model's own activation dtype, and is indexed **by the physical block that holds a
sequence's logical block 0**, not by its row in the batch.

That choice is the whole design, so it is worth saying why the two obvious
alternatives were not taken:

* *Indexed by batch row* (``[max_num_seqs, ...]``, which is how the feature is
  usually drawn) is wrong on this engine.  vLLM's ``InputBatch`` condenses rows
  when a request finishes, so a row index is not a stable name for a sequence;
  a sequence that moves would silently read another one's sinks, and nothing
  would raise.
* *Bypassing the quantised writer* (the "slot mapping bypass" option: give a sink
  token ``slot = -1`` so ``reshape_and_cache`` drops it) leaves a zeroed row in
  the middle of the context the decode reads.  The decode's ``context_lens`` is a
  contiguous prefix count, so a dropped row is not skipped -- it dequantises to
  ``0`` and contributes ``exp(0 - m)`` to the softmax.  Excluding a *prefix*
  would mean shifting every later token's slot by ``N``, which is the block
  allocator's arithmetic, not the backend's.

Indexing by the anchor block instead costs no bookkeeping at all: the block
allocator already keeps that block alive exactly as long as the sequence, frees
it exactly when the sequence ends, and shares it under prefix caching.  The
side-car rides ``swap_blocks`` and ``copy_blocks`` unchanged, because both walk
every plane in the cache tuple and gather it in dimension 0 -- which is what
dimension 0 of this plane is.  The price is that rows are reserved for blocks
that never anchor a sequence: ``2 * N * head_size`` activation elements per
block, against the ``block_size * head_size`` packed nibbles a block already
holds, i.e. ``4 * N / block_size`` of the packed cache -- 12.5% at ``N = 4`` and
``block_size = 128``.  A second phase can compact that to ``max_num_seqs`` rows
behind a block-to-row map; the arithmetic below does not care.

So the sinks live in **both** planes: quantised in the paged cache, where the
existing writer put them and the existing decode still reads them, and exact
here.  Nothing on the write path changes.  The fusion below is therefore a
*replacement* rather than an append, which is what lets ``context_lens`` stay
exactly what it was.

Fusing the two streams
======================

Write ``s_i`` for the score the decode gave cached token ``i``, ``m`` for its
running max over the whole context and ``L = sum_i exp(s_i - m)``.  The operator
returns the normalised ``O_all = sum_i exp(s_i - m) v_i / L``.  Replacing the
quantised sinks ``j < N`` with their exact counterparts is then, in one shared
maximum ``m'``:

    a       = exp(m - m')
    num     = a L O_all - sum_j exp(s^q_j - m') v^q_j + sum_j exp(s^d_j - m') v^d_j
    den     = a L       - sum_j exp(s^q_j - m')       + sum_j exp(s^d_j - m')
    O       = num / den

``s^q_j`` and ``v^q_j`` are recomputed here from the ``N`` cached rows, so the
subtraction removes exactly what this module's own arithmetic says the kernel
added; the two are consistent with each other even where they differ from the
device's fp16 internals by a rounding.  The cancellation in ``den`` is real --
sinks are chosen *because* they hold most of the mass -- so it is done in fp32
with a shared maximum, and the residual is clamped at zero rather than allowed
to go negative.

What is missing, and why it is host-side for now
================================================

``L`` is not an output of either decode operator.  It cannot be recovered from
``O_all`` either, at any number of extra launches: the output is a convex
combination of the values, so it is invariant to a rescaling of the weights, and
the total mass is exactly the information that rescaling destroys.

So :func:`quantized_context_softmax_stats` recomputes ``(m, L)`` on the host,
dequantising the context's keys.  That is ``O(context)`` per decode step and is
the reason this is a correctness and ergonomics vehicle rather than a serving
default.  It is also a small change away from being free: the AIV decode already
computes the pair and already writes it to global memory -- ``kPartialMaxLane``
and ``kPartialSumLane`` in the split workspace
(``csrc/attention/turboquant/op_kernel/vector/turboquant_vector_service.h``) --
for the sequences it splits.  Exposing it as an ``lse`` out-tensor on both
decodes, for split and unsplit sequences alike, is new plumbing rather than new
arithmetic, and is what phase two should do first.

Which cache this can read
=========================

**Row-major planes on the Lloyd-Max codebook grid only** -- that is, the AIV decode
and its ``npu_turboquant_reshape_and_cache`` writer.  :func:`_dequantized_rows`
addresses a cache row as ``slot * num_kv_heads * packed_bytes + ...`` and
reconstructs a nibble through :func:`turboquant_dequantize`, and both are properties
of *that* writer rather than of the cache shape, which the two writers share.

The kv4fp8 Cube path is neither, on both counts
(``csrc/attention/turboquant/op_kernel/common/``):

* ``turboquant_mode.h`` sets ``kStoresNzTiles<KV4_FP8>``, so its writer scatters each
  token into an NZ-tiled ``[tile][kv head][C0 column group][tile row]`` image
  (``turboquant_layout.h``, ``NzTiledPackedByte``) that the Cube reads as a fractal
  without permuting;
* its mode config is ``is_affine = true, affine_bias = 7.5``, so a code reconstructs
  as ``(code - 7.5) * scale`` on an fp8 e4m3 operand grid, not through the Lloyd-Max
  centroid table.

Reading a kv4fp8 cache with the functions below therefore returns the wrong bytes
*and* puts them on the wrong grid. Nothing raises: ``(m, L)`` come back plausible and
wrong, the subtraction removes a vector that was never added, and the merged output
is noise -- which is what it did, at a ~4x drop in LongBench score, before the
callers below learned to refuse it. The refusal lives with each caller rather than
here, because this module is handed planes and cannot see which writer filled them;
a kv4fp8 reader is phase-two work and needs the camodel gate, not a host test.

This module imports only ``torch`` and the two TurboQuant modules that do the
same, so a harness with no engine around it can drive the fusion directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_PACK_FACTOR,
    turboquant_dequantize,
    turboquant_scale_slot,
)
from vllm_ascend.attention.turboquant_rotation import apply_pi

# More than this many uncompressed tokens is not an attention sink, it is a second
# cache: the retrieval ablation this follows saturates at 4, and the side-car's cost
# is linear in the count.
TURBOQUANT_MAX_SINK_TOKENS = 8

# How much context :func:`quantized_context_softmax_stats` dequantises at once. The
# reduction is online, so this only bounds peak host memory; at 1024 positions and
# ``num_kv_heads * head_size`` of 1024 it is 4 MiB of fp32 per token in flight.
TURBOQUANT_STATS_CONTEXT_CHUNK = 1024

# Below this, the quantised context's residual mass is taken as fully cancelled by the
# sinks removed from it, and clamped rather than trusted. Relative to the total mass, so
# it is a statement about how many digits fp32 has left after the subtraction.
_RESIDUAL_MASS_EPS = 1e-6


@dataclass(frozen=True)
class TurboQuantSinkConfig:
    """How many leading tokens of a sequence bypass quantisation."""

    num_sink_tokens: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.num_sink_tokens <= TURBOQUANT_MAX_SINK_TOKENS:
            raise ValueError(
                f"[vllm-ascend/turboquant] VLLM_ASCEND_TQ_SINK_TOKENS must lie in "
                f"[0, {TURBOQUANT_MAX_SINK_TOKENS}], got {self.num_sink_tokens}"
            )

    @property
    def enabled(self) -> bool:
        return self.num_sink_tokens > 0

    @classmethod
    def from_env(cls) -> TurboQuantSinkConfig:
        # Imported here rather than at module scope so that a harness with no vLLM around
        # it can still import this module; envs pulls the plugin's package in with it.
        import vllm_ascend.envs as envs_ascend

        return cls(num_sink_tokens=int(envs_ascend.VLLM_ASCEND_TQ_SINK_TOKENS))


class TurboQuantSinkCache:
    """The uncompressed ``[2, num_blocks, N, num_kv_heads, head_size]`` side-car plane.

    Dimension 0 is K then V, mirroring the packed cache's two planes. Dimension 1 is the
    *physical block* that holds the sequence's logical block 0; see the module docstring
    for why that, and not the batch row, is what names a sequence here.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_sink_tokens: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not 0 < num_sink_tokens <= TURBOQUANT_MAX_SINK_TOKENS:
            raise ValueError(f"[vllm-ascend/turboquant] num_sink_tokens must be in [1, 8], got {num_sink_tokens}")
        if num_sink_tokens > block_size:
            # Every sink lives in logical block 0 by construction, and the anchor row
            # arithmetic below assumes it.
            raise ValueError(
                f"[vllm-ascend/turboquant] {num_sink_tokens} sink tokens do not fit a {block_size}-token block"
            )
        self.num_sink_tokens = num_sink_tokens
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.planes = torch.zeros((2, num_blocks, num_sink_tokens, num_kv_heads, head_size), dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.planes.numel() * self.planes.element_size()

    @property
    def bytes_per_sequence(self) -> int:
        """What one sequence's sinks actually occupy, which is what the feature costs a user."""
        return 2 * self.num_sink_tokens * self.num_kv_heads * self.head_size * self.planes.element_size()

    def ingest(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        anchor_blocks: torch.Tensor,
        query_starts: list[int],
        query_lens: list[int],
        context_lens: list[int],
    ) -> int:
        """Copy each request's leading tokens out of this step's K/V into the side-car.

        ``key``/``value`` are ``[num_tokens, num_kv_heads, head_size]`` exactly as
        ``reshape_and_cache`` received them.  ``context_lens`` is how many tokens of each
        sequence were already cached before this step, so a token is a sink iff its
        position in the *sequence* -- not in the batch -- is below ``num_sink_tokens``.
        That makes the ingestion correct for a prompt split across steps as well, even
        though the backend does not currently serve one.

        Returns how many token vectors were stored, which is what a test asserts on.
        """
        rows, sink_indices, blocks = [], [], []
        for request, (start, length, cached) in enumerate(zip(query_starts, query_lens, context_lens)):
            for offset in range(min(length, max(0, self.num_sink_tokens - cached))):
                rows.append(start + offset)
                sink_indices.append(cached + offset)
                blocks.append(request)
        if not rows:
            return 0

        device = key.device
        token_index = torch.tensor(rows, dtype=torch.long, device=device)
        sink_index = torch.tensor(sink_indices, dtype=torch.long, device=device)
        block_index = anchor_blocks.to(torch.long)[torch.tensor(blocks, dtype=torch.long, device=device)]
        self.planes[0][block_index, sink_index] = key[token_index].to(self.planes.dtype)
        self.planes[1][block_index, sink_index] = value[token_index].to(self.planes.dtype)
        return len(rows)

    def ingest_by_slot(self, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor, *, block: int = 0) -> None:
        """Ingest from the cache slots themselves, in one fixed-shape device operation.

        :meth:`ingest` is driven by host-side request boundaries, which is what a ragged
        vLLM batch needs and what makes it a host read.  A caller whose cache is one
        contiguous arena -- ``tools/tq_longbench``, where slot *is* position -- already has
        the answer in ``slots``, and reading it back would be a device-to-host copy per
        layer per chunk in a harness whose whole point is that a step asks the host nothing.

        So the selection is a one-hot matmul instead: ``[num_tokens, N]`` against the
        token's K and V.  The shape depends on nothing but ``N``, so it is capturable, and a
        slot that no token names leaves its row alone -- which is what makes a second pass
        over the same prefix idempotent, and what lets a new sequence starting at slot 0
        overwrite the last one's sinks without any bookkeeping about where the last one
        ended.
        """
        selector = slots.reshape(-1, 1).to(torch.long) == torch.arange(self.num_sink_tokens, device=slots.device)
        written = selector.any(dim=0).reshape(-1, 1, 1)
        weights = selector.to(torch.float32)
        for plane, source in ((self.planes[0], key), (self.planes[1], value)):
            gathered = torch.einsum(
                "tn,thd->nhd", weights, source.reshape(source.shape[0], -1, self.head_size).to(torch.float32)
            )
            plane[block] = torch.where(written, gathered.to(plane.dtype), plane[block])

    def gather(self, anchor_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(key, value)``, each ``[num_tokens, N, num_kv_heads, head_size]``."""
        selected = self.planes[:, anchor_blocks.to(torch.long)]
        return selected[0], selected[1]


def sink_validity_mask(
    seq_lens: torch.Tensor, num_tokens: int, num_sink_tokens: int, device: torch.device
) -> tuple[torch.Tensor, bool]:
    """Which of the ``N`` side-car slots a token's sequence actually filled.

    A sequence shorter than ``N`` has fewer sinks than slots, and a sequence of length
    zero -- what the padding rows of a captured batch look like -- has none.  ``seq_lens``
    is read on the host, where :class:`AscendMetadata` deliberately keeps it, so this
    costs no device synchronisation.

    The second element says whether any slot at all is valid, so a caller can skip the
    whole fusion for a batch of padding rather than launch work that multiplies by zero.

    ``seq_lens`` may describe fewer rows than the batch holds -- a padded capture shape
    runs more query rows than the scheduler filled -- so it is zero-extended to
    ``num_tokens`` rather than trusted for its length.  A row nobody described has no
    sequence, hence no sinks.
    """
    lengths = torch.zeros(num_tokens, dtype=torch.long)
    described = seq_lens.detach().to("cpu", torch.long).clamp_(min=0)[:num_tokens]
    lengths[: described.numel()] = described
    slots = torch.arange(num_sink_tokens, dtype=torch.long)
    mask = slots.unsqueeze(0) < lengths.unsqueeze(1)
    return mask.to(device), bool(mask.any())


def _context_rows(block_tables: torch.Tensor, positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Cache row for each ``(token, context position)``, through the token's block table."""
    physical = torch.gather(block_tables, 1, torch.div(positions, block_size, rounding_mode="floor"))
    return physical * block_size + positions % block_size


def _dequantized_rows(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rows: torch.Tensor,
    num_kv_heads: int,
    *,
    keys_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reconstruct ``Pi k`` and ``Pi v`` for the named cache rows, in fp32.

    ``rows`` is any integer shape; the result carries ``[..., num_kv_heads, head_size]``
    on top of it.  ``keys_only`` skips the value plane, which is what the softmax
    statistics need and is half the work.

    **Row-major planes on the Lloyd-Max grid only.** See the module docstring: a kv4fp8
    Cube cache is NZ-tiled and affine, and this reads it as neither. The caller is what
    knows which writer filled the plane, and what has to refuse.
    """
    head_size = key_cache.shape[-1] * TURBOQUANT_PACK_FACTOR
    packed_shape = (-1, num_kv_heads, head_size // TURBOQUANT_PACK_FACTOR)
    flat_rows = rows.reshape(-1).to(torch.long)
    scale_lanes = scale_cache.view(-1, turboquant_scale_slot(num_kv_heads))[flat_rows]
    tail = rows.shape + (num_kv_heads, head_size)

    keys = turboquant_dequantize(key_cache.view(packed_shape)[flat_rows])
    keys = keys * scale_lanes[:, :num_kv_heads].unsqueeze(-1)
    keys = keys.reshape(tail)
    if keys_only:
        return keys, None
    values = turboquant_dequantize(value_cache.view(packed_shape)[flat_rows])
    values = values * scale_lanes[:, num_kv_heads : 2 * num_kv_heads].unsqueeze(-1)
    return keys, values.reshape(tail)


def _expand_kv_heads(tensor: torch.Tensor, num_heads: int, num_kv_heads: int, dim: int) -> torch.Tensor:
    """GQA/MQA: give every query head of a group its kv head's vector."""
    if num_heads == num_kv_heads:
        return tensor
    return tensor.repeat_interleave(num_heads // num_kv_heads, dim=dim)


def quantized_context_softmax_stats(
    *,
    rotated_query: torch.Tensor,
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale_value: float,
    chunk: int = TURBOQUANT_STATS_CONTEXT_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the decode's own ``(max, sum exp)`` over the quantised context.

    Returns ``(running_max, mass)``, both ``[num_tokens, num_heads]`` fp32, with
    ``mass = sum_i exp(s_i - running_max)`` over the same scores the decode used:
    ``s_i = scale_value * (Pi q) . (Pi k_i)``, the codes dequantised against the Lloyd-Max
    grid and multiplied by the token's K scale.  A token whose context is empty gets
    ``mass = 0``, which is how the operator's zeroed output is spelled here.

    This exists because neither decode operator returns the pair, and it is the
    ``O(context)`` cost the module docstring warns about -- read it before putting this
    anywhere near a serving path.  The reduction is the same online max-and-rescale the
    kernel does, chunked so that a long context does not materialise at once.
    """
    num_tokens = rotated_query.shape[0]
    device = rotated_query.device
    block_size = key_cache.shape[1]
    # Zero-extended rather than trusted for its length, for the reason sink_validity_mask
    # extends it: a padded batch runs more query rows than the scheduler described.
    lengths = torch.zeros(num_tokens, dtype=torch.long)
    described = seq_lens.detach().to("cpu", torch.long)[:num_tokens]
    lengths[: described.numel()] = described

    running_max = torch.full((num_tokens, num_heads), -float("inf"), dtype=torch.float32, device=device)
    mass = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)

    for token in range(num_tokens):
        length = int(lengths[token])
        if length <= 0:
            running_max[token] = 0.0
            continue
        query = rotated_query[token].to(torch.float32)
        table = block_tables[token : token + 1].to(torch.long)
        for start in range(0, length, chunk):
            positions = torch.arange(start, min(start + chunk, length), device=device).unsqueeze(0)
            rows = _context_rows(table, positions, block_size)
            keys, _ = _dequantized_rows(key_cache, value_cache=None, scale_cache=scale_cache, rows=rows,
                                        num_kv_heads=num_kv_heads, keys_only=True)  # fmt: skip
            keys = _expand_kv_heads(keys[0], num_heads, num_kv_heads, dim=1)
            scores = torch.einsum("hd,lhd->lh", query, keys) * scale_value
            chunk_max = scores.amax(dim=0)
            merged = torch.maximum(running_max[token], chunk_max)
            mass[token] = mass[token] * torch.exp(running_max[token] - merged) + torch.exp(scores - merged).sum(dim=0)
            running_max[token] = merged
    return running_max, mass


def fuse_sink_stream(
    *,
    attention_output: torch.Tensor,
    running_max: torch.Tensor,
    mass: torch.Tensor,
    quant_scores: torch.Tensor,
    quant_values: torch.Tensor,
    dense_scores: torch.Tensor,
    dense_values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Replace the quantised sinks' contribution with the uncompressed one.

    Shapes: ``attention_output`` is ``[T, H, D]``, ``running_max``/``mass`` are ``[T, H]``,
    the score tensors are ``[T, N, H]``, the value tensors ``[T, N, H, D]``, and ``valid``
    is a ``[T, N]`` bool.  Every value tensor must already be in *the same basis as*
    ``attention_output`` -- the decode leaves its output rotated, so the caller rotates the
    side-car's V rather than un-rotating the cache's.

    The result is fp32 ``[T, H, D]``; a token with no valid sink, or one whose quantised
    context carried no mass, comes back untouched.
    """
    output = attention_output.to(torch.float32)
    keep = valid.unsqueeze(-1).to(torch.float32)

    # One maximum for both streams, so neither exponential can overflow and the
    # subtraction below happens between numbers of the same size.
    masked_dense = torch.where(valid.unsqueeze(-1), dense_scores, torch.full_like(dense_scores, -float("inf")))
    merged_max = torch.maximum(running_max, masked_dense.amax(dim=1))
    merged_max = torch.where(torch.isfinite(merged_max), merged_max, torch.zeros_like(merged_max))

    rescale = torch.exp(running_max - merged_max)
    total_mass = mass * rescale
    total_numerator = output * total_mass.unsqueeze(-1)

    quant_weights = torch.exp(quant_scores - merged_max.unsqueeze(1)) * keep
    dense_weights = torch.exp(dense_scores - merged_max.unsqueeze(1)) * keep

    # Clamped, not asserted: the sinks are chosen because they hold most of the mass, so
    # what is left after removing them is a small difference of two close fp32 numbers and
    # can land just below zero. Negative residual mass would flip the sign of the answer.
    residual_mass = (total_mass - quant_weights.sum(dim=1)).clamp_(min=0.0)
    residual = total_numerator - torch.einsum("tnh,tnhd->thd", quant_weights, quant_values)

    numerator = residual + torch.einsum("tnh,tnhd->thd", dense_weights, dense_values)
    denominator = residual_mass + dense_weights.sum(dim=1)

    fused = numerator / denominator.clamp(min=torch.finfo(torch.float32).tiny).unsqueeze(-1)
    # A token the fusion cannot speak for keeps what the operator produced: an empty
    # context (mass 0, no sinks), or a batch row that is padding.
    usable = (denominator > _RESIDUAL_MASS_EPS * total_mass.clamp(min=1.0)) & valid.any(dim=1, keepdim=True)
    return torch.where(usable.unsqueeze(-1), fused, output)


def fuse_decode_sinks(
    *,
    attention_output: torch.Tensor,
    query: torch.Tensor,
    rotated_query: torch.Tensor,
    sink_cache: TurboQuantSinkCache,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale_value: float,
    pi_signs: torch.Tensor,
) -> None:
    """Fold the uncompressed sinks into a decode's output, in place.

    ``attention_output`` is the ``[T, H, D]`` activation-dtype view the operator wrote,
    still in the rotated basis -- so this must run *before* any un-rotation or output
    gate.  ``query`` is the raw ``[T, H, D]`` query the layer was handed and
    ``rotated_query`` the ``Pi q`` the decode consumed; the sinks' scores are computed
    against the first, the cache's against the second, and the two agree because ``Pi`` is
    orthogonal.

    Does nothing for a batch in which no sequence has a sink yet.
    """
    num_tokens = attention_output.shape[0]
    device = attention_output.device
    num_sinks = sink_cache.num_sink_tokens

    valid, any_valid = sink_validity_mask(seq_lens, num_tokens, num_sinks, device)
    if not any_valid:
        return

    anchor_blocks = block_tables[:num_tokens, 0]
    sink_rows = anchor_blocks.to(torch.long).unsqueeze(1) * sink_cache.block_size + torch.arange(
        num_sinks, device=device
    ).unsqueeze(0)

    quant_keys, quant_values = _dequantized_rows(key_cache, value_cache, scale_cache, sink_rows, num_kv_heads)
    quant_keys = _expand_kv_heads(quant_keys, num_heads, num_kv_heads, dim=2)
    quant_values = _expand_kv_heads(quant_values, num_heads, num_kv_heads, dim=2)
    quant_scores = torch.einsum("thd,tnhd->tnh", rotated_query.to(torch.float32), quant_keys) * scale_value

    dense_keys, dense_values = sink_cache.gather(anchor_blocks)
    dense_keys = _expand_kv_heads(dense_keys.to(torch.float32), num_heads, num_kv_heads, dim=2)
    # Into the decode's basis: the operator's output is Pi-rotated, and the side-car holds
    # V as the projection produced it. Pi is an involution, so this is the same call that
    # un-rotates -- over N vectors per token rather than the whole context.
    dense_values = apply_pi(_expand_kv_heads(dense_values.to(torch.float32), num_heads, num_kv_heads, dim=2), pi_signs)
    dense_scores = torch.einsum("thd,tnhd->tnh", query.to(torch.float32), dense_keys) * scale_value

    running_max, mass = quantized_context_softmax_stats(
        rotated_query=rotated_query,
        key_cache=key_cache,
        scale_cache=scale_cache,
        block_tables=block_tables[:num_tokens],
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale_value,
    )

    fused = fuse_sink_stream(
        attention_output=attention_output,
        running_max=running_max,
        mass=mass,
        quant_scores=quant_scores,
        quant_values=quant_values,
        dense_scores=dense_scores,
        dense_values=dense_values,
        valid=valid,
    )
    attention_output.copy_(fused.to(attention_output.dtype))
