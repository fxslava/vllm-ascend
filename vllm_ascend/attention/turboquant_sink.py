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
token's softmax mass, and 4-bit codes are worst exactly there: the error on one
sink vector is multiplied by the weight the sink attracts.  Keeping ``N`` of them
at activation precision and folding their exact contribution into the decode is
what restores retrieval under a 4-bit KV cache.

Three moving parts, and the whole design is in how they avoid reading the packed
cache:

1. a **side-car plane** holding the sinks at activation precision, indexed by the
   physical block that anchors the sequence;
2. **neutralised scale rows**, which take the quantised copies of those same
   tokens out of the decode's arithmetic without touching the writer, the slot
   mapping or ``context_lens``;
3. the decode's own **(max, mass) out-tensor**, which turns the merge into an
   append of two streams whose relative weight is known exactly.

Layout -- Option A, a dedicated side-car plane
==============================================

:class:`TurboQuantSinkCache` holds ``[2, num_blocks, N, num_kv_heads, head_size]``
in the model's own activation dtype, indexed **by the physical block that holds a
sequence's logical block 0**, not by its row in the batch.

That choice is the whole layout, so it is worth saying why the two obvious
alternatives were not taken:

* *Indexed by batch row* (``[max_num_seqs, ...]``, which is how the feature is
  usually drawn) is wrong on this engine.  vLLM's ``InputBatch`` condenses rows
  when a request finishes, so a row index is not a stable name for a sequence;
  a sequence that moved would silently read another one's sinks, and nothing
  would raise.
* *Bypassing the quantised writer* (the "slot mapping bypass" option: give a sink
  token ``slot = -1`` so ``reshape_and_cache`` drops it) leaves a zeroed row in
  the middle of the context the decode reads.  ``context_lens`` is a contiguous
  prefix count, so a dropped row is not skipped -- it dequantises to ``0`` and
  contributes ``exp(0 - m)`` to the softmax.  Excluding a *prefix* would mean
  shifting every later token's slot by ``N``, which is the block allocator's
  arithmetic, not the backend's.

Indexing by the anchor block costs no bookkeeping at all: the block allocator
already keeps that block alive exactly as long as the sequence, frees it exactly
when the sequence ends, and shares it under prefix caching.  The side-car rides
``swap_blocks`` and ``copy_blocks`` unchanged, because both walk every plane in
the cache tuple and gather it in dimension 0 -- which is what dimension 0 of this
plane is.  The price is that rows are reserved for blocks that never anchor a
sequence: ``4 * N / block_size`` of the packed cache, 12.5% at ``N = 4`` and
``block_size = 128``.  A later phase can compact that to ``max_num_seqs`` rows
behind a block-to-row map; the arithmetic below does not care.

Taking the quantised copies back out, without reading them
==========================================================

The sink tokens are still *written* to the paged cache -- nothing on the write
path changes -- so the decode still reads them, and their quantised contribution
has to come back out of its answer.  The previous shape of this module did that
by dequantising those rows on the host and subtracting what they contributed.
That is what a kv4fp8 Cube cache cannot support: its planes are NZ-tiled and its
codes are affine about 7.5 on an fp8 operand grid, so a row-major Lloyd-Max
reader returns the wrong bytes off the wrong grid, plausibly and silently.  It
measured a 4x LongBench regression before it was refused.

:meth:`TurboQuantSinkCache.neutralize_scale_rows` removes the need for a reader
entirely.  Every mode reconstructs a cached vector as *code times the token's
scale*, and the two decodes apply that scale on the vector unit after the
product -- ``Mul(scores, scores, keyScale)`` and ``Mul(probs, probs,
valueScale)``.  Zeroing a sink token's K and V scale lanes therefore makes its
key exactly the zero vector and its value contribution exactly zero, on **any**
codebook, in **any** byte order, without unpacking a single nibble.  What the
decode then reads for that row is:

    score  = scale_value * q . 0 = 0     exactly
    value  = 0                           exactly

so the row contributes ``exp(0 - m) = exp(-m)`` of mass and nothing at all to the
numerator.  Over ``n`` sink rows that is ``n`` units of mass in absolute terms --
a *known integer*, not something that has to be measured.  The merge subtracts
it.

Two properties fall out of this that are worth naming:

* the subtraction is well conditioned.  The old scheme subtracted the quantised
  sinks, which by construction hold most of the mass, so the residual was a small
  difference of two large numbers.  Here the subtracted term is ``n`` against a
  total that includes it, so the residual is a sum of non-negative terms;
* ``m >= 0`` whenever a sequence has a sink, because one of the scores the decode
  maximised over is exactly ``0``.  Every exponential the merge takes is therefore
  of a non-positive number, and none of them can overflow.

Fusing the two streams
======================

Write ``m`` and ``L`` for the pair the decode now returns (``lse``, see
``turboquant_layout``), ``O_q`` for its normalised output, ``s_j`` and ``v_j`` for
the exact sinks' logits and values, and ``n`` for how many sink rows the decode
actually read.  In one shared maximum ``m' = max(m, max_j s_j)``:

    w_q  = L exp(m - m')                 the decode's whole mass, rescaled
    w_j  = exp(s_j - m')
    num  = w_q O_q + sum_j w_j v_j
    den  = (w_q - n exp(-m')) + sum_j w_j
    O    = num / den

``num`` keeps the whole of ``w_q``: the neutralised rows added no value, so the
decode's numerator is already the non-sink numerator exactly.  Only ``den`` loses
the ``n exp(-m')`` those rows put there.  The residual is clamped at zero rather
than allowed to go negative, which is a statement about fp32 digits and not about
the arithmetic.

Nothing in the merge reads the packed cache, so it is layout- and
codebook-independent: the same code serves the AIV decode and the kv4fp8 Cube
decode, and neither path can drift away from it.

What this costs, and what it no longer costs
============================================

The merge is ``O(N)`` per decode step, not ``O(context)``: the pair ``(m, L)``
comes out of the kernel rather than being recomputed.  It is all device work on
the caller's stream, and reads nothing back to the host -- the only host tensors
it touches are the ones :class:`AscendMetadata` deliberately keeps there.

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
    turboquant_split_lse,
)
from vllm_ascend.attention.turboquant_rotation import apply_pi

# More than this many uncompressed tokens is not an attention sink, it is a second
# cache: the retrieval ablation this follows saturates at 4, and the side-car's cost
# is linear in the count.
TURBOQUANT_MAX_SINK_TOKENS = 8

# What :class:`TurboQuantSinkCache` aligns its plane's base address to. The kernels
# never read this plane -- it is torch's operand, not an Ascend C one -- but it is
# handed to the same allocator as the tensors that are, and an operand whose address
# is not a whole burst is the single hardest TurboQuant fault to read back (error 264,
# raised from whichever later launch the runtime happened to be working on). 512 is a
# whole DMA line rather than the 32-byte burst floor, so a future in-kernel reader of
# this plane inherits the alignment rather than having to re-establish it.
TURBOQUANT_SINK_ALIGN_BYTES = 512

# The plane's innermost contiguous run -- one token's ``num_kv_heads * head_size``
# elements -- is held to a whole group of this many, so that every row of it starts on
# the same alignment its base has. ``head_size`` is a power of two of at least 64, so
# this always holds; it is checked rather than assumed because it is what makes the
# base alignment mean anything for a row.
TURBOQUANT_SINK_ALIGN_ELEMENTS = 64

# How much context :func:`quantized_context_softmax_stats` dequantises at once. That
# function is a test oracle now, not a runtime path; see its docstring.
TURBOQUANT_STATS_CONTEXT_CHUNK = 1024

# Below this share of the decode's own mass, what is left after the neutralised sink
# rows are subtracted is taken as fully cancelled and the token keeps the operator's
# answer. Relative, so it is a statement about how many digits fp32 has left.
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


def sink_plane_bytes(
    *, num_blocks: int, num_sink_tokens: int, num_kv_heads: int, head_size: int, element_size: int
) -> int:
    """What a side-car plane of this shape will occupy, including its alignment slack.

    A pure function of the shape, so an arena budget can be checked *before* anything is
    allocated -- which is the only point at which a caller can still do something other
    than raise out of memory. :meth:`TurboQuantSinkCache.nbytes` answers the same question
    about a plane that already exists.
    """
    payload = 2 * num_blocks * num_sink_tokens * num_kv_heads * head_size * element_size
    return payload + TURBOQUANT_SINK_ALIGN_BYTES


class TurboQuantSinkCache:
    """The uncompressed ``[2, num_blocks, N, num_kv_heads, head_size]`` side-car plane.

    Dimension 0 is K then V, mirroring the packed cache's two planes. Dimension 1 is the
    *physical block* that holds the sequence's logical block 0; see the module docstring
    for why that, and not the batch row, is what names a sequence here.

    Allocated once, up front, as one contiguous activation-dtype block whose base address
    is a whole :data:`TURBOQUANT_SINK_ALIGN_BYTES`. The allocation carries that much slack
    and the plane is a view into the aligned part of it, so the alignment is a property of
    this object rather than of whatever the allocator happened to return.
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
        if num_blocks <= 0:
            raise ValueError(f"[vllm-ascend/turboquant] the sink plane needs at least one block, got {num_blocks}")
        token_elements = num_kv_heads * head_size
        if token_elements % TURBOQUANT_SINK_ALIGN_ELEMENTS:
            raise ValueError(
                f"[vllm-ascend/turboquant] one sink token is {token_elements} elements "
                f"({num_kv_heads} kv heads x {head_size}), which is not a whole group of "
                f"{TURBOQUANT_SINK_ALIGN_ELEMENTS}; every row of the side-car has to start on the alignment "
                "its base does"
            )

        self.num_sink_tokens = num_sink_tokens
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

        shape = (2, num_blocks, num_sink_tokens, num_kv_heads, head_size)
        element_size = torch.empty((), dtype=dtype).element_size()
        slack = TURBOQUANT_SINK_ALIGN_BYTES // element_size
        if TURBOQUANT_SINK_ALIGN_BYTES % element_size:
            raise ValueError(
                f"[vllm-ascend/turboquant] a {element_size}-byte activation does not divide the "
                f"{TURBOQUANT_SINK_ALIGN_BYTES}-byte sink alignment"
            )
        payload = 1
        for extent in shape:
            payload *= extent
        # One allocation, then a view at the first aligned element of it. Keeping the
        # storage alive is the whole reason it is an attribute and not a local.
        self._storage = torch.zeros(payload + slack, dtype=dtype, device=device)
        misalignment = self._storage.data_ptr() % TURBOQUANT_SINK_ALIGN_BYTES
        offset = 0 if misalignment == 0 else (TURBOQUANT_SINK_ALIGN_BYTES - misalignment) // element_size
        self.planes = self._storage[offset : offset + payload].view(shape)
        if self.planes.data_ptr() % TURBOQUANT_SINK_ALIGN_BYTES:
            raise RuntimeError(
                f"[vllm-ascend/turboquant] the sink plane starts at {self.planes.data_ptr():#x}, which is not a "
                f"whole {TURBOQUANT_SINK_ALIGN_BYTES}-byte line even after {offset} elements of padding; the "
                "allocator did not return an element-aligned block"
            )

    @property
    def nbytes(self) -> int:
        """What the allocation costs, slack included -- which is what the arena paid."""
        return self._storage.numel() * self._storage.element_size()

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
        selection = self._select(query_starts, query_lens, context_lens)
        if selection is None:
            return 0
        token_index, sink_index, request_index = self._selection_tensors(selection, key.device)
        block_index = self._anchor_rows(anchor_blocks, request_index)
        self.planes[0][block_index, sink_index] = key[token_index].to(self.planes.dtype)
        self.planes[1][block_index, sink_index] = value[token_index].to(self.planes.dtype)
        return len(selection[0])

    def neutralize_scale_rows(
        self,
        scale_cache: torch.Tensor,
        *,
        anchor_blocks: torch.Tensor,
        query_starts: list[int],
        query_lens: list[int],
        context_lens: list[int],
    ) -> int:
        """Zero the K and V scale lanes of the cache rows this step's sinks were written to.

        The same selection :meth:`ingest` made, applied to the scale plane instead of the
        K/V.  It runs *after* the packed write, whose own scale store it overwrites, and it
        is what takes the quantised copies of the sinks out of the decode's arithmetic
        without touching the writer, the slot mapping or ``context_lens``.  See the module
        docstring for why a zeroed scale is an exact exclusion on every codebook.

        ``scale_cache`` is the ``[num_blocks, block_size, scale_slot]`` fp32 plane; the
        whole slot is zeroed rather than only its ``2 * num_kv_heads`` live lanes, because
        the padding lanes are not read by anything and one contiguous store is cheaper than
        a strided one.

        Idempotent: a second pass over a prefix already ingested writes the same zeros, so
        a prompt split across steps and a prefix-cache hit both stay correct.  Returns how
        many rows were zeroed.
        """
        selection = self._select(query_starts, query_lens, context_lens)
        if selection is None:
            return 0
        device = scale_cache.device
        _, sink_index, request_index = self._selection_tensors(selection, device)
        block_index = self._anchor_rows(anchor_blocks, request_index, device=device)
        slot_floats = scale_cache.shape[-1]
        rows = block_index * self.block_size + sink_index
        scale_cache.view(-1, slot_floats).index_fill_(0, rows, 0.0)
        return len(selection[0])

    def _select(
        self, query_starts: list[int], query_lens: list[int], context_lens: list[int]
    ) -> tuple[list[int], list[int], list[int]] | None:
        """Which ``(batch row, sink index, request)`` triples this step's K/V holds sinks for.

        Host arithmetic over host lists: the caller's ``query_lens`` and ``context_lens``
        come from metadata vLLM deliberately keeps on the CPU, so this costs no
        synchronisation, and both callers make the same selection from it rather than
        each deriving its own.
        """
        rows: list[int] = []
        sinks: list[int] = []
        requests: list[int] = []
        for request, (start, length, cached) in enumerate(zip(query_starts, query_lens, context_lens)):
            for offset in range(min(length, max(0, self.num_sink_tokens - cached))):
                rows.append(start + offset)
                sinks.append(cached + offset)
                requests.append(request)
        return (rows, sinks, requests) if rows else None

    @staticmethod
    def _selection_tensors(
        selection: tuple[list[int], list[int], list[int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, sinks, requests = selection
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(sinks, dtype=torch.long, device=device),
            torch.tensor(requests, dtype=torch.long, device=device),
        )

    def _anchor_rows(
        self, anchor_blocks: torch.Tensor, request_index: torch.Tensor, *, device: torch.device | None = None
    ) -> torch.Tensor:
        """The physical block each selected token's sequence is anchored on.

        ``anchor_blocks`` is moved to wherever the index is before it is gathered from:
        ``AscendMetadata`` keeps ``block_tables`` on the host on some paths and on the
        device on others, and indexing a tensor with an index on the other device is an
        error rather than a transfer. The move is a handful of int32 words and is a no-op
        when they already agree.
        """
        target = device if device is not None else request_index.device
        return anchor_blocks.to(device=target, dtype=torch.long)[request_index]

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

    def neutralize_scale_rows_by_slot(self, scale_cache: torch.Tensor, *, block: int = 0) -> None:
        """:meth:`neutralize_scale_rows` for the arena :meth:`ingest_by_slot` serves.

        That arena holds one sequence and slot *is* position, so its sinks are always the
        first ``N`` rows of ``block`` -- which makes this a constant slice rather than a
        scatter, with nothing read from the device and nothing to disambiguate when two
        entries would have named the same row.  ``scale_cache`` is the arena's scale plane
        in any shape whose last dimension is the slot.

        Idempotent and unconditional: rows the sequence never reached are zero already, and
        running it after every chunk is what keeps a restart at slot 0 self-overwriting the
        way :meth:`ingest_by_slot` does.
        """
        flat = scale_cache.view(-1, scale_cache.shape[-1])
        base = block * self.block_size
        flat[base : base + self.num_sink_tokens].zero_()

    def gather(self, anchor_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(key, value)``, each ``[num_tokens, N, num_kv_heads, head_size]``."""
        selected = self.planes[:, anchor_blocks.to(device=self.planes.device, dtype=torch.long)]
        return selected[0], selected[1]


def sink_validity_mask(
    seq_lens: torch.Tensor, num_tokens: int, num_sink_tokens: int, device: torch.device
) -> tuple[torch.Tensor, bool | None]:
    """Which of the ``N`` side-car slots a token's sequence actually filled.

    A sequence shorter than ``N`` has fewer sinks than slots, and a sequence of length
    zero -- what the padding rows of a captured batch look like -- has none.  The mask is
    also exactly how many sink rows the decode read with a neutralised scale, which is what
    the merge subtracts from its mass.

    ``seq_lens`` may describe fewer rows than the batch holds -- a padded capture shape
    runs more query rows than the scheduler filled -- so it is zero-extended to
    ``num_tokens`` rather than trusted for its length.  A row nobody described has no
    sequence, hence no sinks.

    **Built where ``seq_lens`` already lives.**  :class:`AscendMetadata` keeps it on the
    host on purpose, and there the whole mask is host arithmetic and the second element can
    also say whether *any* slot is valid, which lets a caller skip the fusion outright for
    a batch of padding.  A caller whose lengths are a device tensor -- the standalone
    harness, whose whole point is that a decode step asks the host nothing -- gets the mask
    built on the device and ``None`` for that answer: reading it would be precisely the
    synchronisation this is avoiding, and the merge is a no-op for an invalid row anyway.
    """
    if seq_lens.device.type == "cpu":
        lengths = torch.zeros(num_tokens, dtype=torch.long)
        described = seq_lens.detach().to("cpu", torch.long).clamp_(min=0)[:num_tokens]
        lengths[: described.numel()] = described
        mask = torch.arange(num_sink_tokens, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.to(device), bool(mask.any())

    lengths = seq_lens.detach().to(device=device, dtype=torch.long).clamp(min=0).reshape(-1)[:num_tokens]
    if lengths.numel() < num_tokens:
        lengths = torch.cat((lengths, lengths.new_zeros(num_tokens - lengths.numel())))
    mask = torch.arange(num_sink_tokens, device=device, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
    return mask, None


def _expand_kv_heads(tensor: torch.Tensor, num_heads: int, num_kv_heads: int, dim: int) -> torch.Tensor:
    """GQA/MQA: give every query head of a group its kv head's vector."""
    if num_heads == num_kv_heads:
        return tensor
    return tensor.repeat_interleave(num_heads // num_kv_heads, dim=dim)


def fuse_sink_stream(
    *,
    attention_output: torch.Tensor,
    running_max: torch.Tensor,
    mass: torch.Tensor,
    neutralized: torch.Tensor,
    dense_scores: torch.Tensor,
    dense_values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Append the uncompressed sinks to a decode's output and take its neutralised rows out.

    Shapes: ``attention_output`` is ``[T, H, D]``, ``running_max``/``mass`` the decode's own
    ``[T, H]`` fp32 pair, ``neutralized`` the ``[T, 1]`` or ``[T, H]`` count of zero-scaled
    sink rows the decode read, ``dense_scores`` ``[T, N, H]``, ``dense_values``
    ``[T, N, H, D]`` and ``valid`` a ``[T, N]`` bool.  Every value must already be in *the
    same basis as* ``attention_output`` -- the decode leaves its output rotated, so the
    caller rotates the side-car's V rather than un-rotating the cache's.

    The result is fp32 ``[T, H, D]``; a token with no valid sink, or one whose decode
    carried no mass at all, comes back untouched.
    """
    output = attention_output.to(torch.float32)
    keep = valid.unsqueeze(-1).to(torch.float32)
    empty = mass <= 0.0

    # One maximum over both streams, so neither exponential can overflow and the
    # subtraction below happens between numbers of the same size. A token whose decode
    # carried no mass has no meaningful max to contribute -- the kernel wrote whatever it
    # initialised the accumulator to -- so it is excluded from the maximum rather than
    # trusted, and excluded from the answer by `usable` below.
    decode_max = torch.where(empty, torch.full_like(running_max, -float("inf")), running_max)
    masked_dense = torch.where(valid.unsqueeze(-1), dense_scores, torch.full_like(dense_scores, -float("inf")))
    merged_max = torch.maximum(decode_max, masked_dense.amax(dim=1))
    merged_max = torch.where(torch.isfinite(merged_max), merged_max, torch.zeros_like(merged_max))

    decode_weight = mass * torch.exp(decode_max - merged_max)
    decode_weight = torch.where(empty, torch.zeros_like(decode_weight), decode_weight)
    dense_weights = torch.exp(dense_scores - merged_max.unsqueeze(1)) * keep

    # The neutralised rows each carried exactly one unit of absolute mass -- their score is
    # exactly zero, so exp(0) is exactly one -- and no value at all. Only the denominator
    # loses them. Clamped, not asserted: the two terms are equal when the whole context is
    # sinks, and fp32 can land that difference just below zero.
    spurious = neutralized.to(torch.float32) * torch.exp(-merged_max)
    residual_mass = (decode_weight - spurious).clamp_(min=0.0)

    numerator = output * decode_weight.unsqueeze(-1) + torch.einsum("tnh,tnhd->thd", dense_weights, dense_values)
    denominator = residual_mass + dense_weights.sum(dim=1)

    fused = numerator / denominator.clamp(min=torch.finfo(torch.float32).tiny).unsqueeze(-1)
    # A token the fusion cannot speak for keeps what the operator produced: an empty
    # context, a batch row that is padding, or a denominator the subtraction consumed.
    usable = (denominator > _RESIDUAL_MASS_EPS * decode_weight.clamp(min=1.0)) & valid.any(dim=1, keepdim=True) & ~empty
    return torch.where(usable.unsqueeze(-1), fused, output)


def fuse_decode_sinks(
    *,
    attention_output: torch.Tensor,
    query: torch.Tensor,
    sink_cache: TurboQuantSinkCache,
    lse: torch.Tensor,
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
    gate, both of which are linear in the output and are applied to the merged answer.
    ``lse`` is the ``[T, H, TURBOQUANT_LSE_STRIDE]`` pair the same launch wrote.

    Reads the side-car and the query and nothing else: no packed plane, no scale plane,
    no host readback.  Does nothing for a batch in which no sequence has a sink yet.
    """
    num_tokens = attention_output.shape[0]
    device = attention_output.device
    num_sinks = sink_cache.num_sink_tokens

    valid, any_valid = sink_validity_mask(seq_lens, num_tokens, num_sinks, device)
    # `None` is "nobody could say without a synchronisation"; only a definite no skips.
    if any_valid is False:
        return

    running_max, mass = turboquant_split_lse(lse[:num_tokens])

    anchor_blocks = block_tables[:num_tokens, 0]
    dense_keys, dense_values = sink_cache.gather(anchor_blocks)
    dense_keys = _expand_kv_heads(dense_keys.to(torch.float32), num_heads, num_kv_heads, dim=2)
    # Into the decode's basis: the operator's output is Pi-rotated, and the side-car holds
    # V as the projection produced it. Pi is an involution, so this is the same call that
    # un-rotates -- over N vectors per token rather than the whole context.
    dense_values = apply_pi(_expand_kv_heads(dense_values.to(torch.float32), num_heads, num_kv_heads, dim=2), pi_signs)
    # Pi is orthogonal, so (Pi q) . (Pi k) is q . k: the raw query and the side-car's raw K
    # give the same logit the decode would have given the exact vector, with no rotation.
    dense_scores = torch.einsum("thd,tnhd->tnh", query.to(torch.float32), dense_keys) * scale_value

    fused = fuse_sink_stream(
        attention_output=attention_output,
        running_max=running_max,
        mass=mass,
        neutralized=valid.sum(dim=1, keepdim=True),
        dense_scores=dense_scores,
        dense_values=dense_values,
        valid=valid,
    )
    attention_output.copy_(fused.to(attention_output.dtype))


# ----------------------------------------------------------------------------------------
# A test oracle, not a runtime path
# ----------------------------------------------------------------------------------------
#
# What follows recomputes on the host what the decode now returns, by dequantising the
# context out of the packed planes. It is kept because it is an *independent* derivation of
# the pair the kernels write, which is what lets a test say the two agree rather than only
# that the merge is self-consistent. Nothing above calls it, and nothing on a serving path
# may: it is O(context) per decode step, and -- the reason the merge no longer uses it -- it
# reads a cache row-major on the Lloyd-Max grid, which is what
# ``npu_turboquant_reshape_and_cache`` writes and what the AIV decode reads back. The kv4fp8
# Cube writer leaves NZ-tiled planes of affine fp8 codes and this reads neither, plausibly
# and without raising.


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

    **Row-major planes on the Lloyd-Max grid only**, for the reason above this section.
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
    """Recompute a decode's own ``(max, sum exp)`` over the quantised context, on the host.

    Returns ``(running_max, mass)``, both ``[num_tokens, num_heads]`` fp32, with
    ``mass = sum_i exp(s_i - running_max)`` over the same scores the decode used:
    ``s_i = scale_value * (Pi q) . (Pi k_i)``, the codes dequantised against the Lloyd-Max
    grid and multiplied by the token's K scale.  A token whose context is empty gets
    ``mass = 0``, which is how the operator's zeroed output is spelled here.

    This is the oracle the decodes' own ``lse`` out-tensor is checked against; see the
    section header. The reduction is the same online max-and-rescale the kernel does,
    chunked so that a long context does not materialise at once.
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
