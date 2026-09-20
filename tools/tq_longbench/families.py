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
"""Which family a ``config.json`` is, and the three things that do not follow from it.

The *shape* needs no family. Head counts, head size, the RoPE base and how much
of each head it rotates, whether q and k are normed, whether qkv carries a bias:
:func:`tq_longbench.engine.read_checkpoint_shape` reads all of it from the
checkpoint, and GLM-4 needs a reader of its own only because ``chatglm`` spells
every field differently (:mod:`tq_longbench.glm4`). Qwen2.5 needs nothing: its
config is an ordinary HF one, its RoPE is the NeoX halves over the whole head
that :class:`~tq_longbench.layers.RotaryEmbedding` does by default, and the two
things its config leaves out -- ``qk_norm`` and ``qkv_bias`` -- are read off the
checkpoint's tensor names rather than guessed from ``model_type``.

Three things do not come out of the config, and a run is wrong without them:

* **Where a turn ends.** A missing stop id does not raise. Generation runs its
  whole budget and the continuation reads as a model with more to say, which is
  indistinguishable from the real thing. ``glm-4-9b-chat-1m`` declares no
  ``eos_token_id`` at all; Qwen2.5 declares one of the two it uses --
  ``config.json`` names ``<|im_end|>`` and is silent about ``<|endoftext|>``,
  which its own ``generation_config.json`` lists beside it. So a family writes
  its stop tokens down **by name**, and :meth:`ModelFamily.stop_ids` resolves
  each through the tokenizer before falling back to the published id.
* **How a turn is assembled**, for a ``transformers`` that cannot apply the
  checkpoint's own chat template. That fallback is never a preference -- the
  checkpoint's template is the authority, and :meth:`ModelFamily.chat_prompt_ids`
  is only reached where there is none -- but falling through to bare text is not
  neutral either: a chat model handed a document with no turn markers continues
  the document, which over a needle-in-a-haystack prompt means continuing the
  haystack. :attr:`ModelFamily.turn` is the transcription.
* **Which prefill mode** a context length asks for, which is a question about
  what the unquantised staging pool costs per token.

A ``model_type`` no family claims gets :data:`GENERIC`: its tokenizer's chat
template and its config's ``eos_token_id`` are then the only authorities, which
for anything exported in the last few years is enough. What :data:`GENERIC`
never does is lend it another model's ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from tq_longbench.glm4 import GLM4_BATCHED_DECODE_MIN_TOKENS, GLM4_MODEL_TYPES, is_glm4_config
from tq_longbench.ops import DENSE_BACKENDS


class TurnBody:
    """Where the user's own text sits in a :attr:`ModelFamily.turn`."""

    def __repr__(self) -> str:
        return "BODY"


#: The one :class:`TurnBody`; a turn names it exactly once.
BODY = TurnBody()

#: What a turn is made of: a special token's name, a literal string, or :data:`BODY`.
TurnPart = str | TurnBody

#: Where ``dense_staging`` stops being affordable, for a family whose KV footprint
#: per token is known. GLM-4's own threshold
#: (:data:`~tq_longbench.glm4.GLM4_BATCHED_DECODE_MIN_TOKENS`), which is the same
#: number as :data:`~tq_longbench.engine.DENSE_STAGING_RECOMMENDED_MAX_SEQ`: above
#: it the fp16 pool held beside the quantised one is what exhausts device memory.
DENSE_STAGING_MAX_TOKENS = GLM4_BATCHED_DECODE_MIN_TOKENS


def special_token_id(tokenizer, name: str, published: int) -> int:
    """The id this tokenizer gives ``name``, or the published one if it does not know it.

    Three places to look, because which of them answers depends on the
    ``transformers`` release: the added-token table
    :func:`tq_longbench.smoke_glm.register_declared_special_tokens` fills on
    4.28, the tokenizer's own conversion on anything newer, and the constant as
    the last word. A lookup that returns the unknown-token id is treated as not
    knowing, since ``<unk>`` is not a stop token and would quietly truncate every
    continuation at the first unknown word.
    """
    table = getattr(tokenizer, "added_tokens_encoder", None) or {}
    if name in table:
        return int(table[name])
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        try:
            resolved = convert(name)
        except Exception:  # a tokenizer that raises for an unknown token has answered
            resolved = None
        unknown = getattr(tokenizer, "unk_token_id", None)
        if isinstance(resolved, int) and resolved != unknown:
            return resolved
    return published


def text_ids(tokenizer, text: str) -> torch.Tensor:
    """The text's own tokens, with whatever prefix the tokenizer adds stripped off.

    ChatGLM4Tokenizer prepends ``[gMASK]<sop>`` to everything it encodes, and a
    turn that has already put those in by id must not carry them twice. Asking
    for ``add_special_tokens=False`` is the direct way to decline; where the
    signature does not take it, the prefix is *measured* -- by encoding the empty
    string -- rather than assumed to be two tokens long.
    """
    try:
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    except TypeError:  # a tokenizer whose __call__ does not take the flag
        prefix = tokenizer("", return_tensors="pt").input_ids[0].numel()
        return tokenizer(text, return_tensors="pt").input_ids[0][prefix:]


@dataclass(frozen=True)
class ModelFamily:
    """One checkpoint family's turn markers, stop tokens and staging threshold."""

    name: str
    #: Every ``config.json`` ``model_type`` this family claims.
    model_types: frozenset[str]
    #: Token name -> published id, for every token that should end an assistant
    #: turn. The ids are a fallback, never the lookup: :meth:`stop_ids` asks the
    #: tokenizer first, so a checkpoint that numbers them differently still stops.
    stop_tokens: Mapping[str, int] = field(default_factory=dict)
    #: Token name -> published id, for the special tokens :attr:`turn` names.
    template_tokens: Mapping[str, int] = field(default_factory=dict)
    #: One user turn with the generation prompt after it, as special-token names,
    #: literal text and :data:`BODY`. Empty where no transcription exists.
    turn: tuple[TurnPart, ...] = ()
    #: At and above this many context tokens the default prefill is
    #: ``batched_decode``: ``dense_staging`` holds a full fp16 KV pool beside the
    #: quantised one, and that pool is what runs the device's memory out first.
    #: Zero means never stage -- what to assume about a checkpoint whose KV
    #: footprint per token nothing here has been sized against.
    batched_decode_min_tokens: int = DENSE_STAGING_MAX_TOKENS

    def __post_init__(self) -> None:
        """Every way a transcription can be wrong that would not raise at run time.

        A marker with no id is the one worth catching: ``chat_prompt_ids`` would
        put its six characters through the tokenizer as ordinary text, which
        encodes to something, so the prompt would be subtly not the checkpoint's
        format and nothing would say so.
        """
        if not self.turn:
            return
        if sum(isinstance(part, TurnBody) for part in self.turn) != 1:
            raise ValueError(f"{self.name}'s turn must name BODY exactly once")
        literals = [part for part in self.turn if isinstance(part, str)]
        unnamed = [part for part in literals if part.startswith("<|") and part not in self.template_tokens]
        if unnamed:
            raise ValueError(f"{self.name}'s turn uses {unnamed}, which template_tokens gives no id")
        if not any(part in self.template_tokens for part in literals):
            raise ValueError(f"{self.name}'s turn has no special token from template_tokens; it is not a turn")

    # ------------------------------------------------------------ stop tokens

    def stop_ids(self, tokenizer) -> set[int]:
        """Every id that should end an assistant turn, resolved through ``tokenizer``."""
        if tokenizer is None:
            return set(self.stop_tokens.values())
        return {special_token_id(tokenizer, name, published) for name, published in self.stop_tokens.items()}

    # -------------------------------------------------------------- the turn

    def has_turn(self) -> bool:
        """Whether this family has a transcribed turn to fall back on."""
        return bool(self.turn)

    def chat_prompt_ids(self, tokenizer, text: str) -> torch.Tensor:
        """:attr:`turn` as ids, assembled rather than parsed.

        The special tokens are put in by id and everything else is tokenised, so
        nothing depends on the tokenizer recognising ``[gMASK]`` or
        ``<|im_start|>`` as a token when it meets those characters in a string --
        which it does only once the added-token table knows them, and which would
        fail as ordinary text rather than raise.

        Adjacent literal parts are tokenised **together with the body**, in one
        call, because a BPE tokenizer merges across the join: encoding a newline
        and the prompt separately is not the token sequence encoding their
        concatenation gives, and the turn is a transcription of a string.
        """
        if not self.turn:
            raise ValueError(f"{self.name} has no transcribed turn to assemble")
        ids: list[int] = []
        run: list[str] = []

        def flush() -> None:
            if run:
                ids.extend(int(token) for token in text_ids(tokenizer, "".join(run)))
                run.clear()

        for part in self.turn:
            if isinstance(part, TurnBody):
                run.append(text)
            elif part in self.template_tokens:
                flush()
                ids.append(special_token_id(tokenizer, part, self.template_tokens[part]))
            else:
                run.append(part)
        flush()
        return torch.tensor(ids, dtype=torch.int64)

    # ----------------------------------------------------------- the prefill

    def prefill_mode(self, context_tokens: int, backend: str) -> str:
        """``batched_decode`` from :attr:`batched_decode_min_tokens` up.

        A dense baseline decodes out of the unquantised pool, so it needs
        ``dense_staging`` at any length.
        """
        if backend in DENSE_BACKENDS:
            return "dense_staging"
        return "batched_decode" if context_tokens >= self.batched_decode_min_tokens else "dense_staging"


#: GLM-4's stop tokens, and the ids ``glm-4-9b-chat`` and ``-1m`` give them.
#: All three end an assistant turn: ``<|endoftext|>`` ends the document,
#: ``<|user|>`` starts the next turn, and ``<|observation|>`` hands control to a
#: tool. Generation that ignores the last two runs straight on into a turn it
#: then has to invent both halves of, which is what rambling into the haystack
#: looks like from the inside.
GLM4_STOP_TOKENS = {"<|endoftext|>": 151329, "<|user|>": 151336, "<|observation|>": 151338}

#: The tokens GLM-4's chat template is built from, with the same published ids.
#: ``[gMASK]<sop>`` is the document prefix the tokenizer normally prepends itself;
#: ``<|user|>`` and ``<|assistant|>`` are the turn markers.
GLM4_TEMPLATE_TOKENS = {"[gMASK]": 151331, "<sop>": 151333, "<|user|>": 151336, "<|assistant|>": 151337}

#: ``[gMASK]<sop><|user|>\n{prompt}\n<|assistant|>``.
GLM4 = ModelFamily(
    name="GLM-4",
    model_types=GLM4_MODEL_TYPES,
    stop_tokens=GLM4_STOP_TOKENS,
    template_tokens=GLM4_TEMPLATE_TOKENS,
    turn=("[gMASK]", "<sop>", "<|user|>", "\n", BODY, "\n", "<|assistant|>"),
)

#: Qwen2.5's stop tokens. ``config.json`` declares ``<|im_end|>`` and stops
#: there; ``generation_config.json`` lists ``<|endoftext|>`` beside it, and that
#: is the one a continuation which drops out of the chat format ends on. Neither
#: is inferred from the other.
QWEN2_STOP_TOKENS = {"<|im_end|>": 151645, "<|endoftext|>": 151643}

#: ChatML's two markers, at the ids Qwen2.5 gives them. The roles -- ``user``,
#: ``assistant`` -- are ordinary text after the marker, not tokens of their own.
QWEN2_TEMPLATE_TOKENS = {"<|im_start|>": 151644, "<|im_end|>": 151645}

#: ``<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n``.
QWEN2 = ModelFamily(
    name="Qwen2.5",
    model_types=frozenset({"qwen2"}),
    stop_tokens=QWEN2_STOP_TOKENS,
    template_tokens=QWEN2_TEMPLATE_TOKENS,
    turn=("<|im_start|>", "user\n", BODY, "<|im_end|>", "\n", "<|im_start|>", "assistant\n"),
)

#: A checkpoint no family claims: no borrowed stop ids, no transcribed turn, and
#: no unquantised staging pool at any length, since how many bytes per token one
#: would cost is exactly what is not known about it.
GENERIC = ModelFamily(name="generic", model_types=frozenset(), batched_decode_min_tokens=0)

#: Every family with a transcription, in the order :func:`family_for` asks them.
FAMILIES = (GLM4, QWEN2)


def family_for(config: Mapping) -> ModelFamily:
    """The family that claims ``config``'s ``model_type``, or :data:`GENERIC`.

    GLM-4 is asked through :func:`tq_longbench.glm4.is_glm4_config` rather than
    by set membership, because that is where the GLM variants this harness does
    **not** build -- GLM-4-0414's sandwich norms -- are refused by name. Matching
    them as a family would load a model missing two norms per layer.
    """
    if is_glm4_config(dict(config)):
        return GLM4
    model_type = config.get("model_type")
    for family in FAMILIES:
        if model_type in family.model_types:
            return family
    return GENERIC
