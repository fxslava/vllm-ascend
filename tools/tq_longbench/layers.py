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
from dataclasses import dataclass

import torch

from tq_longbench._ascend import turboquant_rotation
from tq_longbench.ops import AttentionBackend


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
            rope_theta=float(getattr(config, "rope_theta", 1.0e6)),
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
    """

    positions: torch.Tensor
    slots: torch.Tensor
    prefix_end: int
    is_decode: bool

    def __post_init__(self) -> None:
        if self.positions.numel() != self.slots.numel():
            raise ValueError(f"{self.positions.numel()} positions but {self.slots.numel()} slots: one slot per token")
        if self.prefix_end != int(self.positions[-1]) + 1:
            raise ValueError(
                f"prefix_end {self.prefix_end} does not follow the last position {int(self.positions[-1])}"
            )


class RMSNorm(torch.nn.Module):
    """Root-mean-square norm in fp32, cast back to the activation dtype.

    The accumulation is fp32 whatever the weights are: at 128k context the
    hidden state's variance is summed over thousands of channels, and doing that
    in fp16 loses the low bits of the mean before the reciprocal square root.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        q_out = shape.num_heads * shape.head_size
        kv_out = shape.num_kv_heads * shape.head_size
        bias = shape.qkv_bias
        # Qwen3.5 projects [query | gate] out of one matrix.
        self.q_proj = torch.nn.Linear(
            shape.hidden_size, q_out * (2 if shape.attn_output_gate else 1), bias=bias, dtype=dtype, device=device
        )
        self.k_proj = torch.nn.Linear(shape.hidden_size, kv_out, bias=bias, dtype=dtype, device=device)
        self.v_proj = torch.nn.Linear(shape.hidden_size, kv_out, bias=bias, dtype=dtype, device=device)
        self.o_proj = torch.nn.Linear(q_out, shape.hidden_size, bias=False, dtype=dtype, device=device)
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

        Split out of :meth:`forward` so that a diagnostic can produce exactly the
        tensors the backends are handed -- and compare two backends on the same
        ones -- without reproducing this and drifting from it. See
        ``tools/tq_longbench/diagnose.py``.
        """
        num_tokens = hidden.shape[0]
        shape = self.shape

        query = self.q_proj(hidden)
        gate = None
        if shape.attn_output_gate:
            query, gate = query.chunk(2, dim=-1)
            gate = gate.view(num_tokens, shape.num_heads, shape.head_size)
        query = query.view(num_tokens, shape.num_heads, shape.head_size)
        key = self.k_proj(hidden).view(num_tokens, shape.num_kv_heads, shape.head_size)
        value = self.v_proj(hidden).view(num_tokens, shape.num_kv_heads, shape.head_size)

        if shape.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        # RoPE before anything is rotated by Pi: the kernels apply Pi to q and k
        # afterwards, and Pi is orthogonal, so the scores are unchanged -- but
        # only if RoPE has already happened.
        return rope.apply(query, batch.positions), rope.apply(key, batch.positions), value, gate

    def attend(
        self,
        query: torch.Tensor,
        batch: ForwardBatch,
        backend: AttentionBackend,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ask one backend for this layer's attention, gated where the backend does that itself."""
        attention = torch.empty_like(query)
        kernel_gate = gate if backend.fuses_output_gate else None
        if batch.is_decode:
            context_lens = torch.full((query.shape[0],), batch.prefix_end, dtype=torch.int32, device=query.device)
            backend.check_tie_point(context_lens, batch.prefix_end)
            backend.decode(self.layer_index, query, context_lens, attention, kernel_gate)
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
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, shape: ModelShape, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        kwargs = {"bias": False, "dtype": dtype, "device": device}
        self.gate_proj = torch.nn.Linear(shape.hidden_size, shape.intermediate_size, **kwargs)
        self.up_proj = torch.nn.Linear(shape.hidden_size, shape.intermediate_size, **kwargs)
        self.down_proj = torch.nn.Linear(shape.intermediate_size, shape.hidden_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


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
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), batch, rope, write_backends, attend_backend)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


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
        hidden = self.embed_tokens(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, batch, self.rope, write_backends, attend_backend)
        if last_token_only:
            # Only the final position can produce the next token, and at a 2048
            # token chunk the head is a 2048 x vocab matmul that nothing reads.
            hidden = hidden[-1:]
        return self.lm_head(self.norm(hidden))


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
