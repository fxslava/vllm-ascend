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
"""A Qwen3-shaped decoder, written out rather than imported.

``transformers`` supplies the tokenizer and the config; the modules are here so
that the only thing between a token and a kernel launch is arithmetic this file
performs.  There is no attention implementation registry, no cache class, no
generation loop, and nothing that dispatches on a string at runtime.

One layer is:

    RMSNorm -> QKV -> q_norm/k_norm -> RoPE -> cache write -> attention
            -> o_proj -> residual -> RMSNorm -> SwiGLU MLP -> residual

Two details are TurboQuant's rather than Qwen3's:

* **The output gate.** Qwen3.5 projects ``[query | gate]`` out of one matrix and
  multiplies attention by ``sigmoid(gate)`` before ``o_proj``.  When the backend
  reports :attr:`~tq_longbench.ops.AttentionBackend.fuses_output_gate` the gate
  is handed to the decode instead and applied in its epilogue, so this file does
  no elementwise multiply at all -- that is the whole point of the fused path,
  and doing it here as well would square the sigmoid.
* **The folded ``o_proj``.** With ``Pi`` folded into the projection
  (``W_o (I (x) Pi)``) the decode leaves its output rotated and the projection
  un-rotates it for free.  The fold is applied to each checkpoint tensor as it
  is ingested (:func:`fold_output_projection_weight`), or to random weights by
  :func:`fold_output_rotation`; a gated layer cannot use it, because the gate
  sits between attention and ``o_proj`` where the output is still rotated.

GLM-4 fits the same stack: its fused ``query_key_value`` and ``dense_h_to_4h``
are split into these modules at load time (:mod:`tq_longbench.glm4`), and its
half-width interleaved RoPE is a :class:`RotaryEmbedding` variant.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

from tq_longbench._ascend import turboquant_rotation
from tq_longbench.ops import AttentionBackend

#: Where an HF config states the RoPE base, newest spelling first. ``transformers``
#: 5 moved ``rope_theta`` into ``rope_parameters`` (mirrored on ``rope_scaling``)
#: and stopped setting the top-level attribute; 4.x has only the attribute.
_ROPE_PARAMETER_DICTS = ("rope_parameters", "rope_scaling")


def rope_theta_of(config) -> float:
    """The RoPE base this config states, wherever the installed ``transformers`` put it.

    Asked in three places and **refused** if none of them answers, because the
    obvious alternative -- default to 1e6 and carry on -- is how a run gets the
    wrong base without anything going wrong. A base that is off by a factor does
    not raise and does not garble the output: it moves every position, so the
    model is fluent, locally sensible, and worse at reaching back into a long
    context. That is the failure this harness exists to measure, which makes it
    the one failure it cannot be quiet about.

    Qwen2.5 is where this showed up. Its ``config.json`` says
    ``rope_theta: 1000000.0`` and ``transformers`` 5 reports it only as
    ``rope_parameters["rope_theta"]`` -- and 1e6 was the old default, so reading
    it from the wrong place returned the right number for this checkpoint and
    the wrong one for a checkpoint at 500000.
    """
    for name in _ROPE_PARAMETER_DICTS:
        parameters = getattr(config, name, None)
        if isinstance(parameters, dict) and parameters.get("rope_theta") is not None:
            return float(parameters["rope_theta"])
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        raise ValueError(
            f"{getattr(config, 'model_type', 'this config')} states no RoPE base: neither rope_theta nor "
            f"{list(_ROPE_PARAMETER_DICTS)}. Defaulting would put every position at the wrong angle silently."
        )
    return float(theta)


@dataclass(frozen=True)
class ModelShape:
    """The subset of an HF config this harness reads."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attn_output_gate: bool
    #: Qwen3 normalises each head of q and k before RoPE; Qwen2.5 does not.
    #: Neither is declared in the config, so both are read off the checkpoint.
    qk_norm: bool = True
    #: Qwen2 carries q/k/v biases without an ``attention_bias`` key to say so.
    qkv_bias: bool = False
    #: Leading channels of each head that RoPE rotates; ``None`` rotates all of
    #: them. GLM-4 rotates the first half and passes the second half through.
    rotary_dim: int | None = None
    #: Rotate adjacent pairs ``(2i, 2i+1)`` -- ChatGLM / GLM-4 -- rather than the
    #: NeoX halves ``(i, i + d/2)`` Qwen uses. Same frequencies, different
    #: channels: pairing them the wrong way does not raise, it scrambles position.
    rope_interleaved: bool = False

    def __post_init__(self) -> None:
        if self.rotary_dim is not None and not (0 < self.rotary_dim <= self.head_size and self.rotary_dim % 2 == 0):
            raise ValueError(f"rotary_dim {self.rotary_dim} must be even and within head_size {self.head_size}")

    @classmethod
    def from_hf_config(cls, config, attn_output_gate: bool, **detected) -> ModelShape:
        """Read the shape out of an HF config, nested text config and all.

        ``detected`` carries what the config does not state -- ``qk_norm`` and
        ``qkv_bias`` -- because both vary between checkpoints of the same
        ``model_type`` and neither has a key. Guessing either wrong does not
        raise at build time: it raises at load time, on a parameter that has no
        home or no value, which is the outcome worth having.
        """
        # A multimodal checkpoint keeps the language model's shape one level down.
        config = getattr(config, "text_config", config)
        num_heads = config.num_attention_heads
        head_size = getattr(config, "head_dim", None) or config.hidden_size // num_heads
        return cls(
            num_layers=config.num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", num_heads),
            head_size=head_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            rope_theta=rope_theta_of(config),
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            attn_output_gate=attn_output_gate,
            **detected,
        )

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_size)


@dataclass
class ForwardBatch:
    """What one call through the stack needs to know about where it sits.

    ``slots`` and ``prefix_end`` are the tie point: the tokens being written
    occupy ``slots``, and after the write the cache holds exactly ``prefix_end``
    tokens.  Everything downstream derives its context lengths from
    ``prefix_end``, so the two cannot drift apart silently.

    ``workspace`` is the single-token decode's pre-allocated buffers
    (:class:`DecodeWorkspace`). Where it is present the stack writes into it
    rather than allocating, and ``prefix_end`` is **not** to be read: it is a
    host integer, and the whole point of the static path is that the step's
    length lives on the device where a graph replay can see the current value.
    :attr:`context_lens` is that length.
    """

    positions: torch.Tensor
    slots: torch.Tensor
    prefix_end: int
    is_decode: bool
    workspace: DecodeWorkspace | None = None
    #: ``[num_tokens]`` int32 on the device: how much prefix each row attends
    #: over. Only the static decode path sets it; everywhere else the backends
    #: build it from ``prefix_end``.
    context_lens: torch.Tensor | None = None
    #: How many cache slots the decode reads, fixed across a captured graph.
    #: ``None`` lets each backend read the longest context length back to the
    #: host, which is a synchronisation per layer per step and cannot be captured.
    window: int | None = None

    def __post_init__(self) -> None:
        if self.positions.numel() != self.slots.numel():
            raise ValueError(f"{self.positions.numel()} positions but {self.slots.numel()} slots: one slot per token")
        if self.workspace is not None:
            # Reading positions[-1] here would be a device-to-host copy per step,
            # and under graph capture it is also wrong: the value baked in at
            # capture time is not the value a later replay runs on.
            return
        if self.prefix_end != int(self.positions[-1]) + 1:
            raise ValueError(
                f"prefix_end {self.prefix_end} does not follow the last position {int(self.positions[-1])}"
            )


@dataclass
class DecodeWorkspace:
    """Every buffer one single-token decode step writes into, allocated once.

    A BS=1 decode step launches on the order of ten kernels per layer, none of
    which runs for long enough to cover the host work behind it -- so the
    allocator, the autograd bookkeeping and the Python frames are the step. The
    buffers here exist to take the allocator out of that: nothing on the decode
    path calls ``torch.empty`` after this is built.

    They are shared by **every** layer, which is safe because the decode is
    strictly sequential: layer *i* has consumed its ``qkv`` before layer *i+1*
    writes it. ``hidden`` is the residual stream and is updated in place, so it
    is the one buffer whose contents cross a layer boundary.

    The addresses are also the contract a captured graph is recorded against:
    a replay reads whatever is in these buffers *now* and writes its answer back
    into them, so :attr:`token_ids` both receives the step's input and is where
    the graph leaves the token it chose.
    """

    #: The token this step consumes; the graph writes the next one back into it.
    token_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    #: How much prefix to attend over, on the device so a replay sees it change.
    context_lens: torch.Tensor
    #: The residual stream, updated in place across the whole stack.
    hidden: torch.Tensor
    #: ``[1, q_out + 2 * kv_out]``, the fused projection's output.
    qkv: torch.Tensor
    #: ``[1, num_heads, head_size]``, what the attention backend writes.
    attention: torch.Tensor
    #: ``[1, 2 * intermediate_size]``, the fused gate/up projection's output.
    gate_up: torch.Tensor
    #: ``[1, vocab_size]``.
    logits: torch.Tensor
    #: The greedy choice, so the argmax is inside whatever the step is -- a
    #: captured graph included -- and the host reads one integer rather than
    #: launching a reduction over the vocabulary itself.
    next_token: torch.Tensor

    @classmethod
    def build(cls, shape: ModelShape, dtype: torch.dtype, device: torch.device) -> DecodeWorkspace:
        query_rows = shape.num_heads * shape.head_size * (2 if shape.attn_output_gate else 1)
        kv_rows = shape.num_kv_heads * shape.head_size
        index = {"dtype": torch.int64, "device": device}
        return cls(
            token_ids=torch.zeros(1, **index),
            positions=torch.zeros(1, **index),
            slots=torch.zeros(1, dtype=torch.int32, device=device),
            context_lens=torch.ones(1, dtype=torch.int32, device=device),
            hidden=torch.zeros(1, shape.hidden_size, dtype=dtype, device=device),
            qkv=torch.zeros(1, query_rows + 2 * kv_rows, dtype=dtype, device=device),
            attention=torch.zeros(1, shape.num_heads, shape.head_size, dtype=dtype, device=device),
            gate_up=torch.zeros(1, 2 * shape.intermediate_size, dtype=dtype, device=device),
            logits=torch.zeros(1, shape.vocab_size, dtype=dtype, device=device),
            next_token=torch.zeros(1, **index),
        )

    def bytes_allocated(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.hidden, self.qkv, self.attention, self.gate_up, self.logits)
        )

    def seat(self, token: int, position: int, context_len: int) -> None:
        """Point the buffers at one step, without reading anything back.

        Three ``fill_`` launches rather than three host-to-device copies, and no
        allocation: the values are what the next replay (or eager step) will see.
        """
        self.token_ids.fill_(token)
        self.positions.fill_(position)
        self.slots.fill_(position)
        self.context_lens.fill_(context_len)


#: Whether this torch has the fused norm. Looked up once: the decode calls it
#: seventy-two times a token, and ``hasattr`` on a module is not free at that rate.
_HAS_FUSED_RMS_NORM = hasattr(torch.nn.functional, "rms_norm")


def npu_operator(name: str, device: torch.device):
    """``torch_npu``'s operator of that name, if this device is an NPU and it has one.

    Resolved once, when the module that will call it is built, for the same
    reason the norm's ``hasattr`` is: a decode step reaches these on the order of
    a hundred times, and a lookup that answers the same way every time should not
    be repeated at that rate. It also keeps the decision out of the captured
    region, where a branch on a missing attribute would be recorded rather than
    taken.

    ``torch_npu`` is never imported here. It is looked up in ``sys.modules``,
    where :func:`tq_longbench.smoke_glm.resolve_device` put it before torch could
    parse an ``npu`` device at all -- so a run that has an NPU device has already
    imported it, and a run that has not must not be made to.
    """
    if device.type != "npu":
        return None
    module = sys.modules.get("torch_npu")
    return None if module is None else getattr(module, name, None)


class RMSNorm(torch.nn.Module):
    """Root-mean-square norm in fp32, cast back to the activation dtype.

    The accumulation is fp32 whatever the weights are: at 128k context the
    hidden state's variance is summed over thousands of channels, and doing that
    in fp16 loses the low bits of the mean before the reciprocal square root.

    Three routes to that one definition, in the order they are tried:
    ``torch_npu.npu_rms_norm`` on an NPU, which is CANN's own kernel and returns
    ``(normed, rstd)``; ``torch.nn.functional.rms_norm`` where the installed
    torch has it; and the expression at the bottom.

    That expression is what the other two replaced, kept as the fallback and as
    the statement of what they are supposed to compute. Written out it is a
    promote, a square, a mean, an add, a reciprocal square root, two multiplies
    and a cast -- eight launches for one norm, and a decode step does
    seventy-two of them, which at a few microseconds of launch each made it the
    largest non-GEMM cost in the step.

    The torch kernel and the expression agree **bit for bit** on the one-token
    rows a decode norms. Over a 2048-token prefill chunk they differ on about one
    element in forty thousand, by a single float16 ulp, which is a different
    summation order rounding differently and not a different definition. What the
    CANN kernel agrees with has not been measured here, for want of an NPU.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps
        self.normalized_shape = (hidden_size,)
        self._npu_rms_norm = npu_operator("npu_rms_norm", device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._npu_rms_norm is not None:
            # ``(normed, rstd)``. The reciprocal standard deviation is the
            # backward's, and nothing here has a backward -- but the operator
            # returns it either way, so the unpack is not optional.
            return self._npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        if _HAS_FUSED_RMS_NORM:
            return torch.nn.functional.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        promoted = x.to(torch.float32)
        normed = promoted * torch.rsqrt(promoted.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(torch.float32)).to(x.dtype)


class RotaryEmbedding:
    """Precomputed ``cos``/``sin`` for every position the run can reach.

    Built once for ``max_seq_len`` rather than per step: at 128k this is two
    ``max_seq_len x head_size`` tables, and recomputing them inside the decode
    would put a transcendental sweep on the critical path of every token.

    Two variants beyond Qwen's full-width NeoX rotation, both GLM-4's:
    ``rotary_dim`` rotates only the leading channels (the frequencies are those
    of a ``rotary_dim``-wide head, as ChatGLM's ``RotaryEmbedding(kv_channels // 2)``
    and HF ``GlmRotaryEmbedding`` both compute them), and ``interleaved`` pairs
    channel ``2i`` with ``2i+1`` instead of ``i`` with ``i + rotary_dim / 2``.
    """

    def __init__(
        self,
        head_size: int,
        max_seq_len: int,
        theta: float,
        dtype: torch.dtype,
        device: torch.device,
        rotary_dim: int | None = None,
        interleaved: bool = False,
    ) -> None:
        span = head_size if rotary_dim is None else rotary_dim
        if not (0 < span <= head_size and span % 2 == 0):
            raise ValueError(f"rotary_dim {span} must be even and within head_size {head_size}")
        self.rotary_dim = span
        self.interleaved = interleaved
        inv_freq = 1.0 / (theta ** (torch.arange(0, span, 2, dtype=torch.float32, device=device) / span))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)
        emb = angles.repeat_interleave(2, dim=-1) if interleaved else torch.cat((angles, angles), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
        """``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``: the interleaved quarter turn."""
        return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``[num_tokens, num_heads, head_size]`` in place of its own basis."""
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        turn = self._rotate_pairs if self.interleaved else self._rotate_half
        if self.rotary_dim == x.shape[-1]:
            return x * cos + turn(x) * sin
        rotated, passed = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        return torch.cat((rotated * cos + turn(rotated) * sin, passed), dim=-1)

    def apply_(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """:meth:`apply` written back over ``x``, for the decode's fused qkv buffer.

        Same arithmetic, no allocation for the result: q and k are slices of the
        one projection buffer and are rotated where they already sit. The two
        temporaries the expression still needs -- ``turn(x)``, and the gather of
        ``cos``/``sin`` -- are small and, under graph capture, come from the
        graph's own pool and are never re-allocated.

        The pass-through channels a partial rotary leaves alone (GLM-4's second
        half of each head) are already in place, so only the leading
        ``rotary_dim`` are written.
        """
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        turn = self._rotate_pairs if self.interleaved else self._rotate_half
        rotated = x if self.rotary_dim == x.shape[-1] else x[..., : self.rotary_dim]
        turned = turn(rotated)
        rotated.mul_(cos).add_(turned.mul_(sin))
        return x


class Attention(torch.nn.Module):
    """Projections, RoPE, the cache write, and whichever attention the backend is.

    The backend is asked for the attention rather than chosen here, so this
    module is identical for ``native_v5`` and for either TurboQuant decode. What
    it does know is whether the gate is its job or the kernel's.
    """

    def __init__(self, shape: ModelShape, layer_index: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.shape = shape
        self.layer_index = layer_index
        # Qwen3.5 projects [query | gate] out of the query matrix, so the query
        # half of the fused tensor is twice as tall there.
        query_rows = shape.num_heads * shape.head_size * (2 if shape.attn_output_gate else 1)
        kv_rows = shape.num_kv_heads * shape.head_size
        #: How the fused projection's columns divide into query, key and value.
        #: The order is the checkpoints' own -- ``chatglm``'s ``query_key_value``
        #: is concatenated exactly this way -- so a fused checkpoint tensor lands
        #: here unchanged and a split one is re-fused at load time.
        self.qkv_split = (query_rows, kv_rows, kv_rows)
        self.qkv_proj = torch.nn.Linear(
            shape.hidden_size, sum(self.qkv_split), bias=shape.qkv_bias, dtype=dtype, device=device
        )
        #: Whether q and k can be rotated as one tensor: they are adjacent in the
        #: fused projection, so they are -- unless a gate sits between them (the
        #: layout is then ``[q | gate | k | v]``), or ``qk_norm`` has already
        #: copied each out of the buffer to normalise it.
        self._rotates_together = not shape.attn_output_gate and not shape.qk_norm
        #: How many of the fused projection's columns that one rotation covers.
        self._rotary_columns = query_rows + kv_rows
        self.o_proj = torch.nn.Linear(
            shape.num_heads * shape.head_size, shape.hidden_size, bias=False, dtype=dtype, device=device
        )
        # Registered only where the checkpoint has them: an unused RMSNorm would
        # sit at its initial value of 1 and normalise anyway, changing the
        # numbers without ever failing to load.
        if shape.qk_norm:
            self.q_norm = RMSNorm(shape.head_size, shape.rms_norm_eps, dtype, device)
            self.k_norm = RMSNorm(shape.head_size, shape.rms_norm_eps, dtype, device)

    def project(
        self, hidden: torch.Tensor, batch: ForwardBatch, rope: RotaryEmbedding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Everything before the cache write: ``(query, key, value, gate)``, RoPE applied.

        One GEMM, then three views. ``split`` on the output's last axis is free --
        the pieces share the projection's storage -- and reshaping each to
        ``[tokens, heads, D]`` stays a view because only the (contiguous) last
        axis is divided. So the saving over three projections is three launches
        and two kernel tails per layer, and nothing is copied to get it. The
        arithmetic is unchanged: cuBLAS splits the output rows of one GEMM the
        same way it would compute three, and the results agree bit for bit.

        Split out of :meth:`forward` so that a diagnostic can produce exactly the
        tensors the backends are handed -- and compare two backends on the same
        ones -- without reproducing this and drifting from it. See
        ``tools/tq_longbench/diagnose.py``.
        """
        num_tokens = hidden.shape[0]
        shape = self.shape
        workspace = batch.workspace

        if workspace is None:
            fused = self.qkv_proj(hidden)
        else:
            # The decode's one GEMM, into the buffer it always uses.
            fused = workspace.qkv
            weight, bias = self.qkv_proj.weight, self.qkv_proj.bias
            if bias is None:
                torch.mm(hidden, weight.t(), out=fused)
            else:
                torch.addmm(bias, hidden, weight.t(), out=fused)

        query, key, value = fused.split(self.qkv_split, dim=-1)
        gate = None
        if shape.attn_output_gate:
            query, gate = query.chunk(2, dim=-1)
            gate = gate.reshape(num_tokens, shape.num_heads, shape.head_size)
        query = query.view(num_tokens, shape.num_heads, shape.head_size)
        key = key.view(num_tokens, shape.num_kv_heads, shape.head_size)
        value = value.view(num_tokens, shape.num_kv_heads, shape.head_size)

        if shape.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        # RoPE before anything is rotated by Pi: the kernels apply Pi to q and k
        # afterwards, and Pi is orthogonal, so the scores are unchanged -- but
        # only if RoPE has already happened.
        if workspace is None or shape.qk_norm:
            # q_norm/k_norm have already copied q and k out of the buffer, so
            # there is nothing left to rotate in place.
            return rope.apply(query, batch.positions), rope.apply(key, batch.positions), value, gate
        if self._rotates_together:
            # One rotation for both. RoPE is the same map on every head, and the
            # fusion put q's heads and k's immediately before one another in the
            # buffer -- so the two together are one ``[tokens, H + H_kv, D]``
            # view, and rotating it is rotating each of them. Seven launches a
            # layer instead of fourteen, for arithmetic that is unchanged.
            rope.apply_(fused[..., : self._rotary_columns].view(num_tokens, -1, shape.head_size), batch.positions)
            return query, key, value, gate
        return rope.apply_(query, batch.positions), rope.apply_(key, batch.positions), value, gate

    def attend(
        self,
        query: torch.Tensor,
        batch: ForwardBatch,
        backend: AttentionBackend,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ask one backend for this layer's attention, gated where the backend does that itself.

        On the static path the context length is a device tensor the caller owns
        and the window is fixed, so no length is read back to the host: the
        tie-point check is exactly the read the static path exists to remove, and
        the invariant it guards -- that the decode attends over the prefix that
        was written -- holds there by construction, because one buffer is both
        the slot written and the length attended over.
        """
        workspace = batch.workspace
        attention = torch.empty_like(query) if workspace is None else workspace.attention
        kernel_gate = gate if backend.fuses_output_gate else None
        if batch.is_decode:
            if batch.context_lens is not None:
                context_lens = batch.context_lens
            else:
                context_lens = torch.full((query.shape[0],), batch.prefix_end, dtype=torch.int32, device=query.device)
                backend.check_tie_point(context_lens, batch.prefix_end)
            backend.decode(self.layer_index, query, context_lens, attention, kernel_gate, window=batch.window)
        else:
            backend.prefill_chunk(self.layer_index, query, batch.prefix_end, attention, kernel_gate)
        if gate is not None and kernel_gate is None:
            attention = attention * torch.sigmoid(gate)
        return attention

    def forward(
        self,
        hidden: torch.Tensor,
        batch: ForwardBatch,
        rope: RotaryEmbedding,
        write_backends: tuple[AttentionBackend, ...],
        attend_backend: AttentionBackend,
    ) -> torch.Tensor:
        query, key, value, gate = self.project(hidden, batch, rope)
        num_tokens = hidden.shape[0]

        for backend in write_backends:
            backend.write_kv(self.layer_index, key, value, batch.slots)
        attention = self.attend(query, batch, attend_backend, gate)
        return self.o_proj(attention.reshape(num_tokens, -1))


class MLP(torch.nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``, with gate and up projected together."""

    def __init__(self, shape: ModelShape, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        kwargs = {"bias": False, "dtype": dtype, "device": device}
        self.intermediate_size = shape.intermediate_size
        # CANN's SwiGLU: it halves the trailing axis itself and computes
        # ``silu(first) * second``, which is the layout gate_up_proj produces and
        # the one every checkpoint that fuses the two writes. Three launches
        # become one, and the split stops being a tensor operation at all.
        self._npu_swiglu = npu_operator("npu_swiglu", device)
        # Gate and up read the same input and are the same shape, so they are one
        # GEMM whose output is read as two halves. This is also the layout the
        # checkpoints that fuse them already use -- chatglm's ``dense_h_to_4h``
        # and the HF glm export's ``gate_up_proj``, both gate-then-up.
        self.gate_up_proj = torch.nn.Linear(shape.hidden_size, 2 * shape.intermediate_size, **kwargs)
        self.down_proj = torch.nn.Linear(shape.intermediate_size, shape.hidden_size, **kwargs)

    def forward(self, x: torch.Tensor, workspace: DecodeWorkspace | None = None) -> torch.Tensor:
        if workspace is None:
            fused = self.gate_up_proj(x)
        else:
            fused = workspace.gate_up
            torch.mm(x, self.gate_up_proj.weight.t(), out=fused)
        if self._npu_swiglu is not None:
            return self.down_proj(self._npu_swiglu(fused))
        gate, up = fused.split((self.intermediate_size, self.intermediate_size), dim=-1)
        if workspace is None:
            return self.down_proj(torch.nn.functional.silu(gate) * up)
        # silu writes back over the gate half and the product over it again, so
        # the activation costs no allocation at all; ``up`` is read, never written.
        return self.down_proj(torch.nn.functional.silu(gate, inplace=True).mul_(up))


class DecoderLayer(torch.nn.Module):
    def __init__(self, shape: ModelShape, layer_index: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(shape.hidden_size, shape.rms_norm_eps, dtype, device)
        self.self_attn = Attention(shape, layer_index, dtype, device)
        self.post_attention_layernorm = RMSNorm(shape.hidden_size, shape.rms_norm_eps, dtype, device)
        self.mlp = MLP(shape, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        batch: ForwardBatch,
        rope: RotaryEmbedding,
        write_backends: tuple[AttentionBackend, ...],
        attend_backend: AttentionBackend,
    ) -> torch.Tensor:
        workspace = batch.workspace
        if workspace is None:
            hidden = hidden + self.self_attn(self.input_layernorm(hidden), batch, rope, write_backends, attend_backend)
            return hidden + self.mlp(self.post_attention_layernorm(hidden))
        # ``hidden`` is the workspace's residual stream: both residuals are added
        # back into it, so the stack allocates nothing per layer.
        hidden.add_(self.self_attn(self.input_layernorm(hidden), batch, rope, write_backends, attend_backend))
        return hidden.add_(self.mlp(self.post_attention_layernorm(hidden), workspace))


class CausalLM(torch.nn.Module):
    """Embedding, the decoder stack, the final norm and the head."""

    def __init__(self, shape: ModelShape, max_seq_len: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.shape = shape
        self.embed_tokens = torch.nn.Embedding(shape.vocab_size, shape.hidden_size, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(shape, index, dtype, device) for index in range(shape.num_layers)]
        )
        self.norm = RMSNorm(shape.hidden_size, shape.rms_norm_eps, dtype, device)
        self.lm_head = torch.nn.Linear(shape.hidden_size, shape.vocab_size, bias=False, dtype=dtype, device=device)
        if shape.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(
            shape.head_size,
            max_seq_len,
            shape.rope_theta,
            dtype,
            device,
            rotary_dim=shape.rotary_dim,
            interleaved=shape.rope_interleaved,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        batch: ForwardBatch,
        write_backends: tuple[AttentionBackend, ...],
        attend_backend: AttentionBackend,
        last_token_only: bool = True,
    ) -> torch.Tensor:
        workspace = batch.workspace
        if workspace is None:
            hidden = self.embed_tokens(token_ids)
        else:
            # ``index_select`` is ``embed_tokens`` with somewhere to put the row.
            hidden = workspace.hidden
            torch.index_select(self.embed_tokens.weight, 0, token_ids, out=hidden)
        for layer in self.layers:
            hidden = layer(hidden, batch, self.rope, write_backends, attend_backend)
        if last_token_only:
            # Only the final position can produce the next token, and at a 2048
            # token chunk the head is a 2048 x vocab matmul that nothing reads.
            hidden = hidden[-1:]
        if workspace is None:
            return self.lm_head(self.norm(hidden))
        torch.mm(self.norm(hidden), self.lm_head.weight.t(), out=workspace.logits)
        return workspace.logits

    def load_destinations(self) -> dict[str, torch.Tensor]:
        """Where each checkpoint tensor goes, under the name a checkpoint calls it.

        The module tree fuses q/k/v into one projection and gate/up into another,
        but a checkpoint names them separately -- and the loader's contract is
        that an unrecognised name *raises* rather than being skipped. So the
        loader is handed views: ``self_attn.q_proj.weight`` is the first
        ``query_rows`` rows of ``qkv_proj.weight``, and copying into it writes
        the fused tensor in place.

        This is also what keeps the fusion honest about a checkpoint that was
        already fused. ``chatglm``'s ``query_key_value`` is split by
        :func:`~tq_longbench.glm4.glm4_checkpoint_targets` into the three logical
        names and lands back in one tensor in the same order it arrived, so the
        bytes on the device are the bytes in the shard.

        Every parameter that is not part of a fusion maps to itself, so the
        loader's unfilled-parameter check still covers the whole model.
        """
        destinations = {name: tensor for name, tensor in self.named_parameters()}
        for index, layer in enumerate(self.layers):
            prefix = f"layers.{index}."
            attention, mlp = layer.self_attn, layer.mlp
            query_rows, kv_rows, _ = attention.qkv_split
            for kind, fused in (("weight", attention.qkv_proj.weight), ("bias", attention.qkv_proj.bias)):
                if fused is None:
                    continue
                del destinations[f"{prefix}self_attn.qkv_proj.{kind}"]
                pieces = fused.split((query_rows, kv_rows, kv_rows), dim=0)
                for name, piece in zip(("q_proj", "k_proj", "v_proj"), pieces):
                    destinations[f"{prefix}self_attn.{name}.{kind}"] = piece
            del destinations[f"{prefix}mlp.gate_up_proj.weight"]
            gate, up = mlp.gate_up_proj.weight.split((mlp.intermediate_size, mlp.intermediate_size), dim=0)
            destinations[f"{prefix}mlp.gate_proj.weight"] = gate
            destinations[f"{prefix}mlp.up_proj.weight"] = up
        return destinations


def require_foldable(shape: ModelShape) -> None:
    """Refuse a fold the model cannot take, before any weight is touched.

    A gated layer is refused rather than skipped: silently leaving it unfolded
    while the decode was told the layer is folded would return the rotated basis
    to a projection that does not undo it.
    """
    if shape.attn_output_gate:
        raise ValueError(
            "a layer with an attn_output_gate cannot fold Pi into o_proj: the gate sits between attention and "
            "o_proj, where the output is still rotated"
        )


def fold_output_projection_weight(weight: torch.Tensor, head_size: int) -> torch.Tensor:
    """``W_o (I (x) Pi)`` for one checkpoint tensor, folded where it was loaded.

    This is the ingestion-time fold: the shard tensor is still on the host in
    the checkpoint's own dtype, so the arithmetic runs in float64 there and the
    result is rounded once, into float32, before the copy onto the device casts
    it to the run's dtype. Folding after the load instead would round bf16 to
    fp16 first and then round the rotated values again -- and would ask the NPU
    for float64, which it does not do well.
    """
    fold = turboquant_rotation().fold_pi_into_output_projection
    return fold(weight.to(device="cpu", dtype=torch.float32), head_size)


#: Where the ``o_proj`` fold runs. ``host`` is the shipped float64 transform on
#: the CPU; ``device`` is the same map as a matmul, wherever the weight lives;
#: ``auto`` picks ``device`` on an accelerator and ``host`` on a CPU, which is
#: the right way round -- see :func:`fold_on_device`.
FOLD_SITES = ("auto", "host", "device")


def fold_precision(device: torch.device) -> torch.dtype:
    """What the fold's matmul should accumulate in on ``device``.

    float64 on a CPU, where it is both available and fast enough to be the
    obvious choice -- a float64 matmul reproduces the shipped float64 transform
    to the last bit while running an order of magnitude quicker than it, because
    a BLAS ``dgemm`` is a better-optimised thing than seven strided passes.
    float32 anywhere else: an NPU has no fast float64, and float32 leaves 13 bits
    of headroom over the float16 the result is stored in. That headroom is the
    whole argument, and :meth:`~tq_longbench.engine.StandaloneModelRunner.load_weights`
    checks it on the first layer rather than assuming it.
    """
    return torch.float64 if device.type == "cpu" else torch.float32


def pi_matrix(head_size: int, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pi as a ``[D, D]`` matrix, so the fold can be a matmul instead of a transform.

    Built by rotating the identity with the *shipped* ``apply_pi`` rather than
    assembled from a Hadamard matrix and the sign vector here. The two would be
    the same matrix right up until one of them drifted, and a Pi that disagreed
    with the kernels' would not raise -- it would decode the cache against the
    wrong basis and return plausible wrong numbers. Building it from the shipped
    transform means there is only ever one definition.

    Pi is symmetric and its own inverse, which is why one matrix serves both the
    fold and the un-rotation, and why ``W_h Pi`` can be written as a right
    multiply on rows: ``(W_h Pi)[r] = Pi W_h[r]``.
    """
    rotation = turboquant_rotation()
    signs = rotation.turboquant_pi_signs(head_size, torch.device("cpu")).to(torch.float64)
    identity = torch.eye(head_size, dtype=torch.float64)
    resolved = fold_precision(device) if dtype is None else dtype
    return rotation.apply_pi(identity, signs).to(device=device, dtype=resolved)


def fold_on_device(site: str, device: torch.device) -> bool:
    """Whether ``site`` means folding where the weight already is.

    ``auto`` folds on an accelerator and leaves a CPU run on the shipped
    transform. Not because the transform is faster -- it is not, the matmul beats
    it about thirteen times over even on a CPU, despite ``O(D^2)`` against
    ``O(D log D)``, because a BLAS ``gemm`` is a better-optimised thing than
    seven strided float64 passes -- but because on a CPU the shipped transform is
    the harness's own reference, and a reference is worth more exact than quick.
    ``--fold-site device`` takes the speedup there too, at float64, and
    reproduces the transform to the last bit on everything measured so far.

    On an accelerator there is no such trade: the fold happens where the weight
    is already going, on the Cube, and what is avoided is 40 layers of
    single-threaded host float64 in front of every run.
    """
    if site not in FOLD_SITES:
        raise ValueError(f"unknown fold site {site!r}; choose one of {list(FOLD_SITES)}")
    if site == "auto":
        return device.type != "cpu"
    return site == "device"


def fold_output_projection_in_place(weight: torch.Tensor, head_size: int, pi: torch.Tensor) -> None:
    """Fold ``W_o`` to ``W_o (I (x) Pi)`` where it already lives, as one batched matmul.

    ``weight`` is ``[hidden, H * D]`` and is rewritten in place. Viewed as
    ``[hidden, H, D]`` the fold is a right multiply by ``Pi`` on the trailing
    axis, the same map for every head and every row, which ``matmul`` broadcasts
    over the leading two axes -- one launch, and on an NPU one that lands on the
    Cube rather than on the host.

    The product accumulates in ``pi``'s dtype -- :func:`fold_precision` picks it --
    and is rounded once, on the way back into the weight's own. That is the same
    contract the shipped float64 transform offers: the only error a fold
    introduces is the stored dtype's rounding of the rotated values. At float64
    the two routes agree bit for bit; at float32 they agree to within one float16
    ulp, which :meth:`~tq_longbench.engine.StandaloneModelRunner.load_weights`
    does not take on trust but checks on the first layer it folds.
    """
    hidden, in_features = weight.shape
    if in_features % head_size:
        raise ValueError(f"an o_proj of width {in_features} does not divide into {head_size}-wide heads")
    if pi.shape != (head_size, head_size):
        raise ValueError(f"Pi is {tuple(pi.shape)}, not ({head_size}, {head_size})")
    with torch.no_grad():
        rotated = weight.view(hidden, in_features // head_size, head_size).to(pi.dtype) @ pi
        weight.copy_(rotated.reshape(weight.shape).to(weight.dtype))


def fold_output_rotation(model: CausalLM) -> int:
    """Rewrite every ``o_proj`` to ``W_o (I (x) Pi)`` and report how many were folded.

    With the fold in place the decode's ``ROTATED_BASIS`` output stage is what
    ``o_proj`` wants, so nothing un-rotates at runtime.  A checkpoint run folds
    at ingestion instead (:func:`fold_output_projection_weight`); this is for
    weights that never came from one.
    """
    require_foldable(model.shape)
    fold = turboquant_rotation().fold_pi_into_output_projection
    folded = 0
    for layer in model.layers:
        weight = layer.self_attn.o_proj.weight
        with torch.no_grad():
            # fold_pi_into_output_projection promotes to float64 and casts back
            # once, so the weight goes in at its own dtype rather than via fp32.
            weight.copy_(fold(weight, model.shape.head_size))
        folded += 1
    return folded
