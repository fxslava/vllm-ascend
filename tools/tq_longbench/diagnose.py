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
"""Per-layer quantisation loss on a real checkpoint: which layers the 4-bit cache hurts.

    python tools/tq_longbench/diagnose.py --model-path F:/AI/Qwen2.5-3B-Instruct --context 8192
    python tools/tq_longbench/diagnose.py --model-path /models/glm-4-9b-chat-1m --device npu:0 \
        --backend turboquant_cube --context 8192

When a long-context answer degrades, the first question is whether the decode is
wrong or the cache is simply lossy, and an end-to-end score cannot tell the two
apart.  This can: it runs the prefill once, then at the decode step hands **the
same query** to the dense backend and to the quantised one and reports the
cosine between their outputs, layer by layer.

Isolated, deliberately.  The model continues on the dense backend's answer, so
each layer's number is that layer's own loss rather than the accumulation of
every layer before it.  A uniform band near the 4-bit floor means the decode is
right and the cache is lossy; a single layer far below its neighbours is a
finding about the model, and one layer near zero while the rest are fine is a
bug in the backend.

Measured on Qwen2.5-3B-Instruct at context 6925 (2026-09-18), which is the
baseline a regression would show against:

    median 0.9933, max 0.9968, and two outliers -- layer 0 at 0.8953 and
    layer 20 at 0.8608.

Layer 0 is the expected one: its attention is dominated by the sink token, whose
key and value are far outside the distribution the per-vector RMS scale is
chosen for, so 4 bits cost the most there.  ``sink_mass`` is that sentence as a
number, so the claim can be checked on the model in front of you rather than
recalled: it is how much of the layer's attention lands on the first few
positions, and a layer whose ``attn_cos`` is low while its ``sink_mass`` is high
is the expected failure rather than a new one.

The other columns say where in the stack the damage lands:

* ``attn_cos`` -- the two backends' attention outputs for the same query. The
  original column, and the one the baseline above quotes.
* ``hidden_cos`` -- the two *layer outputs* those attentions produce, o_proj,
  MLP and both residuals included. Always the higher of the two: the residual
  stream dilutes whatever attention got wrong, and how much it dilutes it is the
  thing worth knowing.
* ``needle_mass`` -- how much of the decode's attention lands on the needle's own
  positions, read off the pool under test.
* ``entropy`` -- the mean entropy of that attention over heads, in nats.
* ``target_rank`` -- the logit lens: where the answer's first token sits if this
  layer's hidden state were the last one.

All five are read at the decode step, over the quantised pool's own keys
(dequantised back to the model's basis), because that is the step this tool
exists to explain.  :mod:`tq_longbench.probe` reads the same kind of numbers at
the last prefill token instead, through hooks, and needs two runs to compare
anything; this one has both backends open at once and needs one.

The quantised side of the comparison is ``--backend`` -- the Cube decode where
there is an NPU to run it, its torch reference where there is not -- and the
dense side is the ``dense_staging`` prefill pool.  Which backend that pool runs
through is not assumed on a device: :func:`negotiate_dense_staging` probes one
synthetic layer before any weight loads, because an Ascend 950 refuses
op-plugin's fused attention outright (``EZ9903``) and stages through
``cann_dense`` instead.  Without it a 950 run reads as a device error forty
layers into a prefill rather than as a named refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """Appended, so the standard library keeps precedence over ``tools/bisect``."""
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)


_bootstrap_path()

import argparse  # noqa: E402  (after the path bootstrap above)
from dataclasses import dataclass  # noqa: E402

import torch  # noqa: E402

from tq_longbench.engine import (  # noqa: E402
    RunnerConfig,
    StandaloneModelRunner,
    read_checkpoint_shape,
)
from tq_longbench.families import family_for  # noqa: E402
from tq_longbench.glm4 import read_config  # noqa: E402
from tq_longbench.layers import ForwardBatch  # noqa: E402
from tq_longbench.ops import (  # noqa: E402
    DENSE_ATTENTION_APIS,
    DENSE_BACKENDS,
    LayerShape,
    reference_equivalent,
)
from tq_longbench.preflight import select_dense_backend  # noqa: E402
from tq_longbench.probe import cached_keys, cosine, needle_slot_positions  # noqa: E402
from tq_longbench.smoke_glm import (  # noqa: E402
    encode,
    load_tokenizer,
    quantised_backends,
    resolve_device,
)
from tq_longbench.tasks import needle_in_a_haystack  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

#: Below this a layer is called out. Well under the ~0.99 band 4-bit sits in, so
#: it flags an outlier rather than the floor itself.
OUTLIER_COSINE = 0.95

#: How many leading positions are counted as the attention sink when the mass
#: over them is measured. Not ``RunnerConfig.sink_tokens``, which is a different
#: number with a similar name: that one *changes* the run by keeping leading
#: tokens out of the 4-bit cache, and this one only ever measures it.
SINK_WINDOW = 4

#: A logit peak closer to zero than this makes a ratio against it meaningless.
_NEGLIGIBLE_LOGIT = 1e-6

#: What can sit immediately before the needle, for widening an answer back to
#: the sentence it was spliced in as. Four rather than one because the splice
#: point moves with ``--context`` and the depth: mid-document the filler ends in
#: ``". "``, at depth 0 the needle follows ``"Document: "``, and a prompt whose
#: header ends in a newline offers neither.
_SENTENCE_BREAKS = (". ", ".\n", "\n", ": ")


@dataclass(frozen=True)
class LayerRow:
    """One layer's reading at the decode step.

    Every field but ``attn_cosine`` is optional, and for the same reason: a
    column that could not be measured here -- no answer in the prompt, a
    tokenizer that cannot place the needle, a backend whose cache this cannot
    read -- prints as ``-`` rather than as a number that means something else.
    """

    layer: int
    attn_cosine: float
    hidden_cosine: float | None = None
    needle_mass: float | None = None
    sink_mass: float | None = None
    entropy: float | None = None
    target_rank: int | None = None
    #: The peak pre-softmax score over the sink positions against the peak over
    #: the rest of the context. Not in the table -- it is a magnitude, and the
    #: softmax reads differences -- but it is what says whether the sink
    #: positions are scored like the rest of the context or like another kind of
    #: token, so it is carried and reported under it.
    sink_logit_ratio: float | None = None

    def outlier(self) -> bool:
        return self.attn_cosine < OUTLIER_COSINE


def default_device() -> str:
    """The best accelerator present, preferring the one the kernels are written for.

    ``torch_npu`` has to be imported before ``torch.npu`` exists, which is why
    the probe is an import rather than a ``hasattr``: on an Ascend host that has
    not imported it yet ``hasattr(torch, "npu")`` is False and the NPU goes
    unnoticed. Where it is not installed the ``ImportError`` is the answer, so a
    developer host reads ``cpu`` here and nothing below asks it for a device.
    """
    try:
        import torch_npu  # noqa: F401  (registers torch.npu)

        if torch.npu.is_available():
            return "npu:0"
    except (ImportError, AttributeError):
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def default_backend(device: str) -> str:
    """The Cube decode where there is an NPU to run it, its torch reference where there is not."""
    return "turboquant_cube" if device.startswith("npu") else reference_equivalent("turboquant_cube")


def build_parser() -> argparse.ArgumentParser:
    device = default_device()
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/diagnose.py",
        description="Per-layer cosine between the quantised and unquantised decode, on real weights.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context", type=int, default=8192, help="approximate prompt tokens")
    parser.add_argument("--device", default=device, help="npu, npu:N, cuda:N or cpu (default: the best one present)")
    parser.add_argument(
        "--backend",
        default=default_backend(device),
        choices=quantised_backends(),
        help="the quantised decode under test; the dense side is always the dense_staging pool "
        "(default: the Cube decode on an NPU, its torch reference elsewhere)",
    )
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--attn-output-gate", action="store_true")
    parser.add_argument(
        "--fold-output-rotation",
        action="store_true",
        help="fold Pi into o_proj at ingestion, so the decode returns its output in the rotated basis",
    )
    parser.add_argument(
        "--dense-staging-backend",
        default=None,
        choices=sorted(DENSE_BACKENDS),
        help="pin the unquantised prefill pool rather than letting the pre-flight negotiate it",
    )
    parser.add_argument(
        "--dense-attention-api",
        default=None,
        choices=sorted(DENSE_ATTENTION_APIS),
        help="pin native_v5's torch_npu entry point rather than letting the pre-flight negotiate it",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the synthetic one-layer probe of the unquantised prefill path",
    )
    parser.add_argument(
        "--sink-window",
        type=int,
        default=SINK_WINDOW,
        help="how many leading positions sink_mass sums over; this only measures the run, it does "
        f"not change it the way the TurboQuant --sink-tokens sweep does (default: {SINK_WINDOW})",
    )
    return parser


def describe(rows: list[LayerRow]) -> str:
    """The layer table, then the two or three lines that say what it points at."""
    header = (
        f"{'layer':>5} {'attn_cos':>9} {'hidden_cos':>11} {'needle_mass':>12} "
        f"{'sink_mass':>10} {'entropy':>8} {'target_rank':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.layer:>5} {row.attn_cosine:>9.5f} {_number(row.hidden_cosine, '.6f'):>11} "
            f"{_number(row.needle_mass, '.4f'):>12} {_number(row.sink_mass, '.4f'):>10} "
            f"{_number(row.entropy, '.3f'):>8} {_number(row.target_rank, 'd'):>12}"
            f"{'  <-- outlier' if row.outlier() else ''}"
        )
    lines.append("")
    lines.extend(_summary(rows))
    return "\n".join(lines)


def _number(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _summary(rows: list[LayerRow]) -> list[str]:
    """The attention cosine's band and its outliers, then whatever else was measured."""
    values = sorted(row.attn_cosine for row in rows)
    outliers = [row.layer for row in rows if row.outlier()]
    lines = [
        f"  attn_cos: min {values[0]:.6f}   median {values[len(values) // 2]:.6f}   max {values[-1]:.6f}",
        f"  below {OUTLIER_COSINE}: {outliers if outliers else 'none'}",
    ]
    hidden = sorted(row.hidden_cosine for row in rows if row.hidden_cosine is not None)
    if hidden:
        lines.append(f"  hidden_cos: min {hidden[0]:.6f}   median {hidden[len(hidden) // 2]:.6f}")
    lines.extend(_peak_line("sink_mass", rows, lambda row: row.sink_mass, ratio=True))
    lines.extend(_peak_line("needle_mass", rows, lambda row: row.needle_mass))
    return lines


def _peak_line(name: str, rows: list[LayerRow], read, ratio: bool = False) -> list[str]:
    """``median`` and the layer holding the maximum, or nothing when nothing was measured.

    ``ratio`` appends that layer's sink logit ratio, which describes the sink
    positions and only them -- it is the column's companion, not a property of
    whichever layer happened to peak.
    """
    measured = [(read(row), row) for row in rows if read(row) is not None]
    if not measured:
        return []
    values = sorted(value for value, _ in measured)
    value, row = max(measured, key=lambda pair: pair[0])
    trailer = f", logit ratio {row.sink_logit_ratio:.2f}" if ratio and row.sink_logit_ratio is not None else ""
    return [f"  {name}: median {values[len(values) // 2]:.4f}, max {value:.4f} at layer {row.layer}{trailer}"]


def prompt_tokens(model_path: str, prompt: str) -> tuple[object, torch.Tensor]:
    """``prompt`` through the checkpoint's own tokenizer, GLM-4's remote-code one included.

    GLM-4's tokenizer is remote code, which a plain ``from_pretrained`` refuses,
    and on ``transformers`` 4.28 it needs the special tokens registering that
    :func:`~tq_longbench.smoke_glm.load_tokenizer` registers. The fallback is for
    a checkpoint that loader has no answer for: the same ``AutoTokenizer`` call,
    with the trust those configs ask for.

    ``use_chat_template=False`` keeps the bare document this script has always
    measured -- the numbers in the module docstring are that prompt's, and turn
    markers would move every token id underneath them.

    The tokenizer comes back with the ids because the needle columns need it
    again afterwards, and loading a second one is how the two stop agreeing.
    """
    family = family_for(read_config(model_path))
    try:
        tokenizer = load_tokenizer(model_path, family)
    except Exception as error:
        print(f"  {family.name} tokenizer loader failed ({error}); falling back to AutoTokenizer", file=sys.stderr)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, encode(tokenizer, prompt, use_chat_template=False, family=family)


def needle_span(item) -> tuple[int, int] | None:
    """The needle's own characters inside ``item.prompt``.

    ``EvalItem`` carries no needle field, so the sentence is recovered rather
    than read: from the task module's own template where the item is a
    needle-in-a-haystack one, and otherwise by widening the answer to the
    sentence it sits in. An item whose answer is not in its prompt -- a
    summarisation task -- has no span at all, and the needle column stays empty
    rather than pointing at whatever the search happened to hit.

    The widening takes the *nearest* of :data:`_SENTENCE_BREAKS`, not the first
    one found: any single break drags the span back through the prompt's header
    at depth 0, where the needle opens the document and no filler sentence
    precedes it. It is still only a sentence, and the splice lands on a word
    boundary rather than a sentence one -- so at a middle depth it also takes
    the filler words the needle displaced, which is why the template is tried
    first.
    """
    stated = getattr(item, "needle", None) or item.extra.get("needle") or _niah_needle(item)
    if stated:
        start = item.prompt.find(stated)
        return (start, start + len(stated)) if start >= 0 else None
    if not item.answers:
        return None
    start = item.prompt.find(item.answers[0])
    if start < 0:
        return None
    breaks = ((item.prompt.rfind(mark, 0, start), mark) for mark in _SENTENCE_BREAKS)
    left = max((at + len(mark) for at, mark in breaks if at >= 0), default=0)
    right = item.prompt.find(".", start + len(item.answers[0]))
    return (left, len(item.prompt) if right < 0 else right + 1)


def _niah_needle(item) -> str | None:
    """The needle sentence as :func:`~tq_longbench.tasks.needle_in_a_haystack` spliced it in.

    Formatted from the task module's own template, so a change to the wording
    moves both together and there is no second copy of the sentence to drift.
    The import is local and guarded because the template is private to that
    module: a release that renames or re-fields it leaves this returning
    ``None``, and the span search takes over, rather than failing the run over a
    diagnostic column.
    """
    city = item.extra.get("city") if hasattr(item, "extra") else None
    if not city or not item.answers:
        return None
    try:
        from tq_longbench.tasks import _NIAH_NEEDLE

        needle = _NIAH_NEEDLE.format(city=city, code=item.answers[0])
    except (ImportError, KeyError, IndexError):
        return None
    return needle if needle in item.prompt else None


def needle_slots(tokenizer, prompt: str, token_ids: torch.Tensor, span: tuple[int, int] | None) -> torch.Tensor:
    """The cache slots the needle's characters were tokenised into.

    Paging is the identity here -- a slot is a position -- so these are token
    indices into the prefill. Two routes, because the obvious one does not
    survive every tokenizer: the offset mapping where there is one, and a search
    for the needle's own ids where there is not.

    Tokenising the needle alone and searching for it is not enough on its own,
    which is what left this column empty: the splice puts filler immediately
    before and after the needle, and a tokenizer merges across both boundaries
    -- the leading space joins the previous word, the trailing period the next --
    so the needle's first and last ids need not occur in the prompt at all.
    :func:`_aligned_slots` drops them and searches for what is left.
    """
    if span is None:
        return torch.empty(0, dtype=torch.int64)
    start, end = span
    offsets = _offset_mapping(tokenizer, prompt, token_ids.numel())
    if offsets is None:
        return _aligned_slots(tokenizer, prompt[start:end], token_ids)
    inside = [index for index, (left, right) in enumerate(offsets) if right > left and left < end and right > start]
    return torch.tensor(inside, dtype=torch.int64)


def _offset_mapping(tokenizer, prompt: str, expected: int) -> list[tuple[int, int]] | None:
    """``(start, end)`` characters per token, or ``None`` where this tokenizer cannot say.

    Only a fast (Rust-backed) tokenizer answers at all; GLM-4's is remote Python
    code and raises, which is why there is a second route rather than an
    assertion. The length check is not belt-and-braces: a mapping is only usable
    if it describes the tokenisation the run prefilled, and a call that added a
    different set of special tokens would be silently off by the difference.
    """
    try:
        mapping = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    except Exception:
        return None
    if mapping is None or len(mapping) != expected:
        return None
    return [(int(left), int(right)) for left, right in mapping]


def _aligned_slots(tokenizer, needle: str, token_ids: torch.Tensor) -> torch.Tensor:
    """Search the prompt's ids for the needle's, relaxing one token at each end.

    Four attempts per spelling, cheapest first, and the leading-space spelling as
    well: which of the two a tokenizer produces for the same sentence is exactly
    the boundary this is working around. The first search that finds anything
    wins -- a later, shorter one would only find the same positions minus their
    ends.
    """
    for text in (needle, f" {needle}"):
        ids = _encoded(tokenizer, text, add_special_tokens=False)
        if ids is None:
            continue
        for head, tail in ((0, 0), (1, 0), (0, 1), (1, 1)):
            core = ids[head : ids.numel() - tail]
            if core.numel() == 0:
                continue
            slots = needle_slot_positions(token_ids.cpu(), core)
            if slots.numel():
                return slots
    return torch.empty(0, dtype=torch.int64)


def _encoded(tokenizer, text: str, add_special_tokens: bool = True) -> torch.Tensor | None:
    """``text`` as ids, or ``None`` where the tokenizer refused the call.

    Refusals are real rather than defensive: a remote-code tokenizer need not
    accept every keyword ``transformers`` documents, and a diagnostic column is
    worth giving up where a run is not.
    """
    try:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")
    except Exception:
        return None
    return encoded["input_ids"][0].to(torch.int64)


def target_token(tokenizer, item) -> int | None:
    """The token the reference answer opens with, for the logit lens to rank.

    Encoded without special tokens so the first id is the answer's own; where a
    tokenizer refuses that keyword the last id of the ordinary encoding is the
    nearest thing that is certainly not a prefix marker -- GLM-4 opens every
    encoding with ``[gMASK]<sop>``, and a lens ranking a control token would
    report a flat column and mean nothing by it.
    """
    if not item.answers:
        return None
    ids = _encoded(tokenizer, item.answers[0], add_special_tokens=False)
    if ids is not None:
        return int(ids[0]) if ids.numel() else None
    ids = _encoded(tokenizer, item.answers[0])
    return int(ids[-1]) if ids is not None and ids.numel() else None


def negotiate_dense_staging(args, device: torch.device) -> tuple[str | None, str | None]:
    """Which unquantised backend and entry point the dense_staging prefill runs through here.

    Probed on one synthetic layer at the checkpoint's own head counts, before a
    weight loads, and only on an NPU: elsewhere there is a single candidate
    (``dense_reference``) and nothing to negotiate. A pinned ``--dense-*`` flag is
    the answer and is not second-guessed, and a pre-flight that finds no path is
    not fatal -- the runner still has its own default, and the error it raises
    then is the one worth reading.
    """
    pinned = args.dense_staging_backend is not None or args.dense_attention_api is not None
    if device.type != "npu" or args.skip_preflight or pinned:
        return args.dense_staging_backend, args.dense_attention_api

    shape = read_checkpoint_shape(args.model_path, args.attn_output_gate)
    selection = select_dense_backend(
        device,
        LayerShape(shape.num_heads, shape.num_kv_heads, shape.head_size, shape.scale),
        _DTYPES[args.dtype],
        args.block_size,
        role="prefill",
        output_rotation_folded=args.fold_output_rotation,
    )
    if not selection.passed:
        for result in selection.results:
            print(f"  {result.describe()}", file=sys.stderr)
        print("  no unquantised prefill path passed the pre-flight; leaving the choice to the runner", file=sys.stderr)
        return None, args.dense_attention_api
    return selection.backend, selection.api


def layer_output(layer, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """What :meth:`~tq_longbench.layers.DecoderLayer.forward` returns for an attention it is handed.

    Split out so the dense and the quantised trajectory are the same arithmetic
    over the same inbound state, differing in one tensor. That is what makes
    their cosine this layer's own number rather than a running total.
    """
    hidden = hidden + layer.self_attn.o_proj(attention.reshape(hidden.shape[0], -1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def logit_lens_rank(model, hidden: torch.Tensor, target: int | None) -> int | None:
    """Where ``target`` would sit if this layer's state were the last one.

    The *final* norm and unembedding, deliberately, as in
    :mod:`tq_longbench.probe`: the lens asks what the head would say now, not
    what this layer would say to the next one.
    """
    if target is None:
        return None
    logits = model.lm_head(model.norm(hidden.to(model.norm.weight.dtype)))[-1]
    return int((logits > logits[target]).sum())


def attention_scores(
    query: torch.Tensor,
    backend,
    layer: int,
    scale: float,
    context: int,
    needle: torch.Tensor | None,
    sink_window: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """``(needle_mass, sink_mass, entropy, sink_logit_ratio)`` for the decode's query row.

    The softmax the decode forms and throws away, recomputed once per layer in
    float32 over the keys the pool under test actually holds --
    :func:`~tq_longbench.probe.cached_keys` dequantises and un-rotates them, so
    the query they are scored against is the raw one and a 4-bit pool's numbers
    are comparable with a dense one's.

    All four are ``None`` for a backend whose cache this cannot read, rather
    than a fabricated zero. Each one that is not costs a device-to-host read per
    layer, which is a diagnostic's price and never a decode's.
    """
    keys = cached_keys(backend, layer, context)
    if keys is None:
        return None, None, None, None
    group = query.shape[1] // keys.shape[1]
    expanded = keys.repeat_interleave(group, dim=1)[:context]
    scores = torch.einsum("qhd,khd->hk", query[-1:].to(torch.float32), expanded) * scale
    weights = torch.softmax(scores, dim=-1)
    entropy = float(-(weights * weights.clamp_min(1e-30).log()).sum(-1).mean())

    sink = min(max(sink_window, 0), context)
    sink_mass = float(weights[:, :sink].sum(-1).mean()) if sink else None
    mass = None
    if needle is not None and needle.numel():
        inside = needle[needle < context].to(weights.device)
        mass = float(weights[:, inside].sum(-1).mean()) if inside.numel() else 0.0
    return mass, sink_mass, entropy, _sink_logit_ratio(scores, sink)


def _sink_logit_ratio(scores: torch.Tensor, sink: int) -> float | None:
    """The sink positions' peak score against the peak over the rest of the context.

    A magnitude comparison rather than the quantity the softmax reads -- that is
    the difference, and ``sink_mass`` already reports what the difference came
    to. This says whether the sink positions are being scored like the rest of
    the context at all. ``None`` where there is no rest, or where its peak sits
    too close to zero for a ratio against it to mean anything.
    """
    rest = scores[:, sink:]
    if not sink or rest.numel() == 0:
        return None
    peak = float(rest.max())
    if abs(peak) < _NEGLIGIBLE_LOGIT:
        return None
    return float(scores[:, :sink].max()) / peak


@torch.inference_mode()
def layer_profile(
    runner: StandaloneModelRunner,
    token_ids: torch.Tensor,
    needle: torch.Tensor | None = None,
    target: int | None = None,
    sink_window: int = SINK_WINDOW,
) -> list[LayerRow]:
    """Prefill, then compare the two backends on one decode step, layer by layer.

    The layer's own ``project`` produces the query, so what the two backends are
    asked is exactly what :meth:`~tq_longbench.layers.Attention.forward` would
    ask them -- no second copy of the projection to drift.

    Both trajectories are carried through the whole layer, so ``hidden_cos`` is
    measured where the residual stream has already diluted the attention's
    error; the stack then continues on the dense one, which is what keeps each
    layer's numbers its own.
    """
    dense, quantised = runner.prefill_backend, runner.decode_backend
    if dense is quantised:
        raise ValueError(
            f"the diagnostic needs two backends to compare, and --backend {runner.config.backend} decodes out "
            "of the same pool the dense_staging prefill filled. Choose a quantised --backend."
        )

    from tq_longbench.engine import RunMetrics

    logits = runner.prefill(token_ids, RunMetrics())
    position = token_ids.numel()
    step = torch.tensor([int(torch.argmax(logits[-1]))], dtype=torch.int64, device=runner.device)

    batch = ForwardBatch(
        positions=torch.arange(position, position + 1, dtype=torch.int64, device=runner.device),
        slots=quantised.cache.slot_mapping(position, 1),
        prefix_end=position + 1,
        is_decode=True,
    )

    rows: list[LayerRow] = []
    hidden = runner.model.embed_tokens(step)
    for layer in runner.model.layers:
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        query, key, value, gate = attn.project(normed, batch, runner.model.rope)
        for backend in runner.write_backends:
            backend.write_kv(attn.layer_index, key, value, batch.slots)
        reference = attn.attend(query, batch, dense, gate)
        measured = attn.attend(query, batch, quantised, gate)

        # The same inbound state through both, so the only difference downstream
        # of here is the attention itself.
        forward = layer_output(layer, hidden, reference)
        diverged = layer_output(layer, hidden, measured)
        mass, sink_mass, entropy, ratio = attention_scores(
            query, quantised, attn.layer_index, attn.shape.scale, batch.prefix_end, needle, sink_window
        )
        rows.append(
            LayerRow(
                layer=attn.layer_index,
                attn_cosine=cosine(measured, reference),
                hidden_cosine=cosine(diverged, forward),
                needle_mass=mass,
                sink_mass=sink_mass,
                entropy=entropy,
                target_rank=logit_lens_rank(runner.model, forward, target),
                sink_logit_ratio=ratio,
            )
        )
        # Continue on the reference trajectory, so each layer's number is its own.
        hidden = forward
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    item = needle_in_a_haystack(args.context, depth=0.0, seed=0)
    tokenizer, token_ids = prompt_tokens(args.model_path, item.prompt)
    needle = needle_slots(tokenizer, item.prompt, token_ids, needle_span(item))
    target = target_token(tokenizer, item)

    dense_staging_backend, dense_attention_api = negotiate_dense_staging(args, device)

    runner = StandaloneModelRunner(
        RunnerConfig(
            model_path=args.model_path,
            backend=args.backend,
            prefill_mode="dense_staging",
            max_seq_len=token_ids.numel() + 64,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            dtype=_DTYPES[args.dtype],
            device=str(device),
            attn_output_gate=args.attn_output_gate,
            fold_output_rotation=args.fold_output_rotation,
            dense_staging_backend=dense_staging_backend,
            dense_attention_api=dense_attention_api,
        )
    )
    print(
        f"{args.model_path}: {runner.shape.num_layers} layers, {runner.shape.num_heads} heads / "
        f"{runner.shape.num_kv_heads} kv, head_size {runner.shape.head_size}"
    )
    print(f"dense {runner.prefill_backend.name} against quantised {runner.decode_backend.name}")
    print(f"prefill {token_ids.numel()} tokens on {device}, then one decode step")
    print(_needle_line(needle, token_ids.numel(), target), end="\n\n")

    rows = layer_profile(runner, token_ids.to(runner.device), needle, target, args.sink_window)
    print(describe(rows))
    return 0


def _needle_line(needle: torch.Tensor, total: int, target: int | None) -> str:
    """What the needle columns will be able to say, before the run pays for them."""
    if needle.numel() == 0:
        return "needle: not located in the prompt, so needle_mass stays empty"
    return f"needle: {needle.numel()} of {total} tokens, first at {int(needle[0])}, target token {target}"


if __name__ == "__main__":
    raise SystemExit(main())
