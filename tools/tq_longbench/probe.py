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
"""Where in the stack a long-context answer stopped being about the prompt.

A run that retrieves the needle and then recites the haystack, or loops, has
gone wrong *somewhere*, and the output says only that it did. This reads four
numbers off every layer for the last token of the prefill -- the token whose
logits choose the first word of the answer -- so the question becomes which
layer rather than whether.

* ``needle_mass`` -- how much of that token's attention lands on the needle's
  own positions. The direct measure of retrieval: a layer that stops looking at
  the needle is a layer that stopped answering the question.
* ``attention_entropy`` -- the mean entropy of the attention distribution over
  heads, in nats. A collapse toward zero is a head pinned to one position; a
  rise toward ``log(context)`` is attention that has become uniform, which at
  64k is indistinguishable from not attending at all.
* ``target_rank`` -- the logit lens: the layer's hidden state pushed through the
  *final* norm and the unembedding, and the rank the expected token comes out
  at. The layer where a correct answer becomes the top prediction, or stops
  being one.
* ``hidden_cosine`` -- the same layer's output against an fp16 ``cann_dense``
  run of the same prompt. This is the only one that needs two runs, and it is
  the one that separates "the model does this anyway" from "the quantised cache
  did this".

**Cost when it is off is zero**, not small: the hooks are registered by
:class:`LayerProbe`'s context manager and removed on the way out, so a run
without ``--profile-layers`` has an empty hook dict and torch's own fast path
skips the machinery entirely. Nothing in :mod:`tq_longbench.layers` knows this
module exists.

**Cost when it is on** is one extra qkv projection and one attention row per
layer per chunk, in float32. That is a diagnostic's price, and it is why this is
a flag rather than a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from tq_longbench._ascend import turboquant_layout, turboquant_rotation
from tq_longbench.kv_cache import DenseKVCache, TurboQuantKVCache

#: Below this, a layer's hidden state has stopped tracking the dense baseline.
#: Quantisation noise at 4 bits sits around 0.99 through a whole stack; a layer
#: under 0.95 has diverged rather than been rounded.
DIVERGENCE_COSINE = 0.95

#: Reported when the needle's own positions are not known -- a task that is not
#: needle-in-a-haystack still gets the other three columns.
_UNKNOWN = None


@dataclass(frozen=True)
class LayerReading:
    """The four numbers for one layer, on the last token of the prefill."""

    layer: int
    needle_mass: float | None = None
    attention_entropy: float | None = None
    target_rank: int | None = None
    hidden_cosine: float | None = None

    def diverged(self) -> bool:
        return self.hidden_cosine is not None and self.hidden_cosine < DIVERGENCE_COSINE


@dataclass
class ProbeReport:
    """Every layer's reading, and the layer where the run stopped making sense."""

    readings: list[LayerReading] = field(default_factory=list)
    #: Context length the readings were taken over, for the entropy ceiling.
    context: int = 0

    def breakdown_layer(self) -> int | None:
        """``L*``: the first layer that diverged, else the steepest fall in needle mass.

        Two questions in one column, in the order they are worth asking. If a
        dense baseline was run, the first layer whose hidden state left it is the
        answer and nothing else needs weighing. Without one there is no reference
        to leave, so the fallback is where the most attention stopped reaching
        the needle between one layer and the next -- which is a weaker claim, and
        :meth:`describe` says which of the two it made.
        """
        for reading in self.readings:
            if reading.diverged():
                return reading.layer
        masses = [(r.layer, r.needle_mass) for r in self.readings if r.needle_mass is not None]
        if len(masses) < 2:
            return None
        drops = [(before - after, layer) for (_, before), (layer, after) in zip(masses, masses[1:])]
        worst, layer = max(drops)
        return layer if worst > 0 else None

    def describe(self, limit: int = 0) -> str:
        """A compact ASCII table, and one line saying what it points at.

        ``limit`` prints only the most interesting rows: the first, the last, and
        the ones around ``L*``. Zero prints every layer, which is what a 40-layer
        run wants when it is being read rather than skimmed.
        """
        star = self.breakdown_layer()
        ceiling = math.log(self.context) if self.context > 1 else 0.0
        header = f"{'layer':>5} {'needle_mass':>12} {'entropy':>9} {'target_rank':>12} {'hidden_cos':>11}"
        lines = [header, "-" * len(header)]
        for reading in self._rows(limit, star):
            if reading is None:
                lines.append(f"{'...':>5}")
                continue
            mark = " <- L*" if reading.layer == star else ""
            lines.append(
                f"{reading.layer:>5} {_number(reading.needle_mass, '.4f'):>12} "
                f"{_number(reading.attention_entropy, '.3f'):>9} "
                f"{_number(reading.target_rank, 'd'):>12} {_number(reading.hidden_cosine, '.6f'):>11}{mark}"
            )
        lines.append("")
        if ceiling:
            lines.append(f"entropy ceiling at this context is ln({self.context}) = {ceiling:.3f} nats (uniform)")
        lines.append(self._verdict(star))
        return "\n".join(lines)

    def _rows(self, limit: int, star: int | None):
        if limit <= 0 or len(self.readings) <= limit:
            return list(self.readings)
        keep = {0, len(self.readings) - 1}
        if star is not None:
            keep |= {index for index in range(star - 1, star + 2) if 0 <= index < len(self.readings)}
        rows, last = [], None
        for index, reading in enumerate(self.readings):
            if index in keep:
                if last is not None and index - last > 1:
                    rows.append(None)
                rows.append(reading)
                last = index
        return rows

    @property
    def compared(self) -> bool:
        """Whether a dense baseline was run, which is what makes divergence measurable."""
        return any(reading.hidden_cosine is not None for reading in self.readings)

    def _verdict(self, star: int | None) -> str:
        if star is None:
            return "no breakdown layer: nothing diverged and needle mass never fell."
        reading = self.readings[star]
        if reading.diverged():
            return (
                f"L* = {star}: the first layer whose hidden state left the dense baseline "
                f"(cos {reading.hidden_cosine:.6f} < {DIVERGENCE_COSINE})."
            )
        if self.compared:
            worst = min(r.hidden_cosine for r in self.readings if r.hidden_cosine is not None)
            return (
                f"L* = {star}: the steepest fall in needle mass. Every layer tracked the dense baseline "
                f"(worst cos {worst:.6f} >= {DIVERGENCE_COSINE}), so attention moved off the needle without "
                "the cache being what moved it."
            )
        return (
            f"L* = {star}: the steepest fall in needle mass. No dense baseline was run, so this is where "
            "attention moved off the needle, not where the run provably diverged."
        )


def _number(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def cached_keys(backend, layer: int, length: int) -> torch.Tensor | None:
    """The first ``length`` cached keys as ``[length, H_kv, D]``, in the model's own basis.

    Read out of whichever pool the run is using, rather than accumulated here: a
    64k prefix of fp16 keys for 40 layers is gigabytes, and the cache already
    holds exactly what the decode read.

    A TurboQuant pool holds ``Pi k`` quantised, so it is dequantised and rotated
    once more -- Pi is an involution -- which puts it back in the same basis a
    dense pool would have stored. That makes the query the probe scores against
    the raw one either way, and makes the two pools' numbers comparable, which is
    the entire point of reading them side by side.

    ``None`` where the cache is neither, so a new backend degrades to the two
    metrics that need no keys instead of reporting a wrong one.
    """
    cache = getattr(backend, "cache", None)
    if isinstance(cache, DenseKVCache):
        packed = (-1, backend.shape.num_kv_heads, backend.shape.head_size)
        return cache.key_planes[layer].view(packed)[:length].to(torch.float32)
    if isinstance(cache, TurboQuantKVCache):
        return _dequantized_keys(backend, cache, layer, length)
    return None


def _dequantized_keys(backend, cache: TurboQuantKVCache, layer: int, length: int) -> torch.Tensor:
    """``Pi`` applied to the dequantised 4-bit keys, which returns them to the model's basis."""
    from tq_longbench.reference import centroid_table, dequantize

    layout = turboquant_layout()
    heads, head_size = backend.shape.num_kv_heads, backend.shape.head_size
    key_plane, _, scale_plane = cache.planes(layer)
    packed = key_plane.view(-1, heads, head_size // layout.TURBOQUANT_PACK_FACTOR)[:length]
    scales = scale_plane.view(-1, backend.geometry.scale_slot)[:length, :heads]
    keys = dequantize(packed, centroid_table(packed.device)) * scales.unsqueeze(-1).to(torch.float32)
    rotation = turboquant_rotation()
    signs = rotation.turboquant_pi_signs(head_size, keys.device)
    return rotation.apply_pi(keys.to(torch.float32), signs)


class LayerProbe:
    """Forward hooks that read the four metrics off every layer, for one prompt.

    Used as a context manager, so the hooks exist only while it is open::

        with LayerProbe(runner, needle_slots=slots, target_token=token) as probe:
            runner.generate(ids, ...)
        print(probe.report().describe())

    ``reference`` is a previous probe's hidden states -- run the same prompt
    through ``cann_dense`` first and pass its :attr:`hidden` in -- and is what
    turns ``hidden_cosine`` from empty into the column that matters.
    """

    def __init__(
        self,
        runner,
        needle_slots: torch.Tensor | None = None,
        target_token: int | None = None,
        reference: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self.runner = runner
        self.model = runner.model
        self.needle_slots = needle_slots
        self.target_token = target_token
        self.reference = reference or {}
        #: Per-layer hidden state of the last prefill token, for a later run to compare against.
        self.hidden: dict[int, torch.Tensor] = {}
        self._attention: dict[int, tuple[float | None, float | None]] = {}
        self._ranks: dict[int, int] = {}
        self._cosines: dict[int, float] = {}
        self._handles: list = []
        self._context = 0

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> LayerProbe:
        for index, layer in enumerate(self.model.layers):
            self._handles.append(layer.register_forward_hook(self._make_layer_hook(index)))
            self._handles.append(layer.self_attn.register_forward_hook(self._make_attention_hook(index)))
        return self

    def __exit__(self, *_) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ----------------------------------------------------------------- hooks

    def _make_layer_hook(self, index: int):
        def hook(_module, args, output):
            batch = args[1]
            if batch.is_decode:
                # The last *prefill* token is the one whose logits choose the
                # answer's first word; a decode step is a different question.
                return
            last = output[-1:].detach().to(torch.float32)
            self.hidden[index] = last
            if index in self.reference:
                self._cosines[index] = cosine(last, self.reference[index])
            if self.target_token is not None:
                self._ranks[index] = self._logit_lens_rank(last)

        return hook

    def _make_attention_hook(self, index: int):
        def hook(module, args, _output):
            hidden, batch, rope = args[0], args[1], args[2]
            if batch.is_decode:
                return
            backend = args[4]
            keys = cached_keys(backend, index, batch.prefix_end)
            if keys is None:
                return
            with torch.no_grad():
                query = module.project(hidden, batch, rope)[0][-1:].to(torch.float32)
            self._attention[index] = _attention_metrics(
                query, keys, module.shape.scale, self.needle_slots, batch.prefix_end
            )
            self._context = batch.prefix_end

        return hook

    def _logit_lens_rank(self, hidden: torch.Tensor) -> int:
        """Where the expected token sits if this layer's state were the last one.

        The *final* norm and unembedding, deliberately: the lens asks what the
        head would say now, not what this layer would say to the next one.
        """
        with torch.no_grad():
            logits = self.model.lm_head(self.model.norm(hidden.to(self.model.norm.weight.dtype)))[-1]
            return int((logits > logits[self.target_token]).sum())

    # ---------------------------------------------------------------- report

    def report(self) -> ProbeReport:
        readings = []
        for index in range(len(self.model.layers)):
            mass, entropy = self._attention.get(index, (_UNKNOWN, _UNKNOWN))
            readings.append(
                LayerReading(
                    layer=index,
                    needle_mass=mass,
                    attention_entropy=entropy,
                    target_rank=self._ranks.get(index),
                    hidden_cosine=self._cosines.get(index),
                )
            )
        return ProbeReport(readings, self._context)


def _attention_metrics(
    query: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    needle_slots: torch.Tensor | None,
    context: int,
) -> tuple[float | None, float]:
    """``(needle_mass, entropy)`` for one query row over ``keys``, in float32.

    The softmax is the one the decode forms and then throws away, recomputed for
    a single row -- which is why this is affordable at 64k and why it is the
    attention the run actually paid for rather than a model of it.
    """
    heads = query.shape[1]
    group = heads // keys.shape[1]
    expanded = keys.repeat_interleave(group, dim=1)[:context]
    scores = torch.einsum("qhd,khd->hk", query, expanded) * scale
    weights = torch.softmax(scores.to(torch.float32), dim=-1)
    entropy = float(-(weights * weights.clamp_min(1e-30).log()).sum(-1).mean())
    if needle_slots is None or needle_slots.numel() == 0:
        return _UNKNOWN, entropy
    inside = needle_slots[needle_slots < context].to(weights.device)
    mass = float(weights[:, inside].sum(-1).mean()) if inside.numel() else 0.0
    return mass, entropy


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Cosine of two hidden states, in float64 on the host.

    Public because ``--eval-fidelity`` compares a quantised run's output
    embedding against a dense one's with the same measure the layer table uses,
    and the two numbers are only comparable if they are the same function.
    """
    left = actual.detach().to("cpu", torch.float64).flatten()
    right = expected.detach().to("cpu", torch.float64).flatten()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-30))


def needle_slot_positions(prompt_ids: torch.Tensor, needle_ids: torch.Tensor) -> torch.Tensor:
    """Where ``needle_ids`` occurs in ``prompt_ids``, as cache slots.

    The paging is the identity -- a slot is a position -- so a match at index
    ``i`` is the slot the needle's first token was written to. Returns every
    position of every occurrence, because a haystack built by repetition can
    contain the answer's digits by accident and a probe that assumed one match
    would silently measure the wrong one.
    """
    if needle_ids.numel() == 0 or needle_ids.numel() > prompt_ids.numel():
        return torch.empty(0, dtype=torch.int64)
    windows = prompt_ids.unfold(0, needle_ids.numel(), 1)
    starts = (windows == needle_ids.to(windows.device)).all(dim=1).nonzero(as_tuple=True)[0]
    if starts.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    offsets = torch.arange(needle_ids.numel(), dtype=torch.int64)
    return (starts.unsqueeze(1).cpu() + offsets).flatten().unique()
