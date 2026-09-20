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
"""Smoke-test one checkpoint through the standalone runner, on whatever device it has.

    python tools/tq_longbench/smoke_glm.py --model-path /models/glm-4-9b-chat-1m \\
        --device npu --backend turboquant_cube --tokens 131072

    python tools/tq_longbench/smoke_glm.py --model-path F:/AI/models/Qwen2.5-3B-Instruct \\
        --device cuda --backend cann_dense --tokens 256 --max-new-tokens 32

Nothing below is GLM-4's by construction. The shape comes off ``config.json``
through :func:`~tq_longbench.engine.read_checkpoint_shape` -- head counts, head
size, the RoPE base and how much of each head it rotates -- and what the config
cannot state (which tokens end a turn, how a turn is assembled) comes from the
family :func:`~tq_longbench.families.family_for` reads out of ``model_type``.
``chatglm`` is GLM-4's half-width *interleaved* RoPE over 32 query heads and 4 kv,
with three stop tokens its config declares none of; ``qwen2`` is Qwen2.5's
full-width NeoX RoPE at ``theta = 1e6``, 16 query heads over 2 kv, and ChatML.
Both run the same four stages, and neither is hardcoded anywhere in this file.

0. **Pre-flight**, before a byte of the checkpoint is read: every attention path
   the run will take -- the decode backend, the dense_staging prefill, the
   reference backend -- runs one synthetic layer with *this checkpoint's* head
   counts and is checked against exact attention (:mod:`tq_longbench.preflight`).
   An operator the SoC refuses fails here in well under the time ingestion takes.
   When the dense prefill is what fails and ``--prefill-mode`` was left to
   default, the run switches to ``batched_decode`` rather than stopping; the
   torch_npu entry point that did pass is the one the run then uses.
   ``--skip-preflight`` skips it.
1. **The contract.** The shape read off ``config.json``, and the decode's output
   stage: ungated, with ``W_o' = W_o (I (x) Pi)`` folded at ingestion, the Cube
   decode must run ``ROTATED_BASIS`` -- the kernel's ``kRotatedBasis``, stage 0,
   where its output passes straight to ``o_proj`` with nothing un-rotated
   in-kernel. Anything else is refused before a token is generated.
2. **A short prompt**, greedy, with the decode latency it cost.
3. **Needle in a haystack** at ``--tokens``, one prefill per depth.

The prefill mode follows the context unless ``--prefill-mode`` says otherwise:
``batched_decode`` from 32768 tokens up (no fp16 pool beside the quantised one),
``dense_staging`` below it. Nothing here is compared with eager Hugging Face --
a 9B chatglm checkpoint's remote modeling code on the NPU is its own project --
so ``--reference-backend native_v5`` is the way to A/B the short prompt against
the unquantised cache.

``--dummy-prompt`` skips the tokenizer altogether -- a 64-token prompt of ones,
the short prompt only -- for a kernel sanity check on a host whose
``transformers`` cannot load the checkpoint's tokenizer. Without it,
:func:`load_tokenizer` carries GLM-4's through ``transformers`` 4.28 (see there).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Keep ``transformers`` from importing TensorFlow when it probes for a backend.
#: Nothing here is a TF model, and on a host whose TensorFlow predates NumPy 2 the
#: probe is not a slow no-op but an ``AttributeError`` out of ``np.object``/``np.bool``
#: during ``import transformers`` -- a failure with nothing to do with the run.
#: ``TF_ENABLE_ONEDNN_OPTS`` silences the banner a TF that does load prints.
TENSORFLOW_OFF = {"USE_TF": "0", "TF_ENABLE_ONEDNN_OPTS": "0"}


def _bootstrap_path() -> None:
    """Appended, so the standard library keeps precedence over ``tools/bisect``."""
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)


def quiet_tensorflow() -> None:
    """Apply :data:`TENSORFLOW_OFF`, unless the environment already said otherwise.

    Import-time, not inside ``main``: ``transformers`` reads these once, when it
    is first imported, and a caller that imports this module has already decided
    to run the harness. An explicit ``USE_TF`` in the environment is left alone.
    """
    for name, value in TENSORFLOW_OFF.items():
        os.environ.setdefault(name, value)


_bootstrap_path()
quiet_tensorflow()

import argparse  # noqa: E402  (after the path bootstrap above)
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402

from tq_longbench._ascend import assert_no_vllm_imported, turboquant_layout  # noqa: E402
from tq_longbench.engine import (  # noqa: E402
    DECODE_GRAPH_MODES,
    DEFAULT_CHUNK_SIZE,
    PREFILL_MODES,
    RunnerConfig,
    StandaloneModelRunner,
    read_checkpoint_shape,
)
from tq_longbench.families import (  # noqa: E402
    DENSE_STAGING_MAX_TOKENS,
    GLM4,
    ModelFamily,
    family_for,
)
from tq_longbench.glm4 import read_config  # noqa: E402
from tq_longbench.layers import FOLD_SITES  # noqa: E402
from tq_longbench.ops import (  # noqa: E402
    DENSE_BACKENDS,
    LayerShape,
    TurboQuantCubeBackend,
    backend_names,
    dense_backend_for,
)
from tq_longbench.preflight import (  # noqa: E402
    ProbeResult,
    probe_backend,
    select_dense_api,
    select_dense_backend,
)
from tq_longbench.tasks import needle_in_a_haystack, needle_report  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

DEFAULT_PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"

DEFAULT_DEPTHS = "0.0,0.5,1.0"

#: Cache room beyond the prompt and the generated tokens, so a tokenised prompt
#: a little over its estimate still fits.
CONTEXT_HEADROOM_TOKENS = 512

#: ``--dummy-prompt``: ``[batch, tokens]`` of token id 1, no tokenizer involved.
DUMMY_PROMPT_SHAPE = (1, 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/smoke_glm.py",
        description="Run a checkpoint through the standalone runner with the folded-o_proj contract.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=model_directory,
        help="directory holding config.json and safetensors shards; a Windows path in either slash is fine",
    )
    parser.add_argument("--device", default="npu", help="npu, npu:N, cuda:N or cpu (default: npu)")
    parser.add_argument("--backend", default="turboquant_cube", choices=backend_names())
    parser.add_argument(
        "--tokens",
        type=int,
        default=8192,
        help="needle-in-a-haystack context length; 0 runs the short prompt only",
    )
    parser.add_argument(
        "--prefill-mode",
        default=None,
        choices=list(PREFILL_MODES),
        help="default: the family's policy -- for GLM-4 and Qwen2.5, batched_decode from "
        f"{DENSE_STAGING_MAX_TOKENS} context tokens up and dense_staging below",
    )
    parser.add_argument("--depths", default=DEFAULT_DEPTHS, help="comma-separated needle depths in [0, 1]")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument(
        "--no-fold-output-rotation",
        action="store_true",
        help="leave o_proj unfolded so the decode un-rotates in-kernel (UNROTATED stage), for an A/B",
    )
    parser.add_argument(
        "--reference-backend",
        default=None,
        choices=backend_names(),
        help="also run the short prompt on this backend (e.g. native_v5) and count agreeing tokens",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="skip the tokenizer's chat template")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the synthetic one-layer probe of every attention path before the weights load",
    )
    parser.add_argument(
        "--dummy-prompt",
        action="store_true",
        help="no tokenizer at all: run a 64-token prompt of ones through the short-prompt stage only "
        "(kernel sanity check; skips needle in a haystack)",
    )
    parser.add_argument(
        "--fold-site",
        default="auto",
        choices=list(FOLD_SITES),
        help="where the o_proj fold runs: auto (the device when there is one), host, or device",
    )
    parser.add_argument(
        "--decode-graph",
        default="auto",
        choices=list(DECODE_GRAPH_MODES),
        help="capture the single-token decode as a device graph: auto (where the device and backend allow it), "
        "on (refuse a run that cannot), off (launch it a kernel at a time -- the baseline to compare against)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def model_directory(text: str) -> str:
    """``--model-path`` written the way this host spells paths, whatever was typed.

    Three things a checkpoint directory brings that are not the caller's problem,
    and all three are Windows': the separator, since ``F:/AI/models/X`` and
    ``F:\\AI\\models\\X`` are one directory and :class:`~pathlib.Path` already
    knows it; a trailing separator tab-completion leaves behind; and ``~`` or a
    ``%VAR%`` the shell did not expand. Normalised once here so that every later
    ``Path(model_path) / ...`` and every message quoting it agree on one spelling.

    Nothing is opened: an ``argparse`` type runs while the parser is still being
    built, and whether the directory exists is :func:`require_checkpoint`'s
    question, asked once in :func:`main` where the answer can be a sentence.
    """
    return str(Path(os.path.expandvars(text)).expanduser())


def require_checkpoint(model_path: str) -> None:
    """Refuse a ``--model-path`` that is not a checkpoint directory, before anything reads it.

    Because of what the first read would say instead: a directory that does not
    exist surfaces as ``FileNotFoundError: no safetensors shards under ...`` from
    inside the weight load, three stages and a pre-flight later, and a config
    that is not there reads as ``model_type None``. A typo in a drive letter
    deserves to be one sentence at the top of the run.
    """
    path = Path(model_path)
    if not path.is_dir():
        raise SystemExit(f"--model-path {model_path} is not a directory")
    if not (path / "config.json").is_file():
        raise SystemExit(f"--model-path {model_path} holds no config.json; it wants the checkpoint directory")


def resolve_device(name: str) -> torch.device:
    """``npu`` means ``npu:0``, and needs ``torch_npu`` imported before torch can parse it."""
    if name.startswith("npu"):
        import torch_npu  # noqa: F401  (registers the npu device type)

        return torch.device("npu:0" if name == "npu" else name)
    return torch.device(name)


def parse_depths(text: str) -> tuple[float, ...]:
    depths = tuple(float(part) for part in text.split(",") if part.strip())
    if not depths or any(not 0.0 <= depth <= 1.0 for depth in depths):
        raise ValueError(f"--depths must be comma-separated values in [0, 1], got {text!r}")
    return depths


def choose_prefill_mode(args: argparse.Namespace, family: ModelFamily) -> str:
    """``--prefill-mode`` if given, else the family's policy -- and a warning where they differ.

    The warning is the flag overriding the policy, not a length: it fires exactly
    when ``--prefill-mode dense_staging`` was asked for somewhere the family
    would have prefilled through the decode instead. A dense baseline decodes out
    of the unquantised pool, so for it the policy *is* ``dense_staging`` and there
    is nothing to warn about.
    """
    policy = family.prefill_mode(args.tokens, args.backend)
    mode = args.prefill_mode or policy
    if mode == "dense_staging" and policy == "batched_decode":
        print(
            f"  warning: dense_staging at {args.tokens} tokens allocates a full fp16 KV pool beside the quantised "
            "one; batched_decode is the default here because that pool is what exhausts HBM",
            file=sys.stderr,
        )
    return mode


def prompt_route(tokenizer, use_chat_template: bool, family: ModelFamily) -> str:
    """Which of :func:`encode`'s three routes this tokenizer will take.

    Printed before a run rather than inferred from its output: the difference
    between a chat prompt and a bare document is the difference between a model
    that answers and one that continues the haystack, and until now it was
    decided silently by which ``transformers`` happened to be installed.
    """
    if not use_chat_template:
        return "raw (--raw-prompt): no turn markers"
    if _has_chat_template(tokenizer):
        return "the tokenizer's own chat template"
    if family.has_turn():
        return f"{family.name} turn markers, assembled here (this transformers has no apply_chat_template)"
    return f"raw: this tokenizer has no chat template and no turn is transcribed for {family.name}"


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def encode(tokenizer, text: str, use_chat_template: bool, family: ModelFamily) -> torch.Tensor:
    """Token ids for one user turn, as a chat turn wherever that is possible.

    Three routes, in order of authority (:func:`prompt_route` names the one
    taken): the checkpoint's own chat template; the family's turn markers
    assembled here (:meth:`~tq_longbench.families.ModelFamily.chat_prompt_ids`),
    for a ``transformers`` that has no ``apply_chat_template``; and the bare text.

    The middle route is the one this function exists for. Falling through to the
    bare text is not a neutral default -- a chat model handed a document with no
    turn markers around it continues the document, which over a
    needle-in-a-haystack prompt means continuing the haystack. That failure reads
    as a model problem and is a prompt problem, so it is worth the transcription.
    """
    if use_chat_template and _has_chat_template(tokenizer):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"][0].to(torch.int64)
    if use_chat_template and family.has_turn():
        return family.chat_prompt_ids(tokenizer, text)
    return tokenizer(text, return_tensors="pt").input_ids[0].to(torch.int64)


def truncate_middle(token_ids: torch.Tensor, keep: int) -> torch.Tensor:
    """Drop from the middle, as LongBench does: the head has the instructions, the tail the question."""
    if token_ids.numel() <= keep:
        return token_ids
    half = keep // 2
    return torch.cat((token_ids[:half], token_ids[-(keep - half) :]))


def describe_decode(runner, summary: dict) -> str:
    """Whether the decode ran as a graph, and what the captures cost.

    Printed because the eager and captured paths differ by roughly a factor of
    three and by nothing at all in the text. A run that fell back quietly would
    read as the kernels being slow, which is the one conclusion this harness
    should never lead someone to by accident.
    """
    if not summary["decode_graph"]:
        return f"decode: eager, a kernel at a time ({runner.decode_graph_refusal})"
    captured = f"{summary['captures']} window(s) captured in {summary['capture_ms']:.0f} ms"
    return f"decode: replayed from a captured graph; {captured}, which is not counted in the percentiles"


def describe_stop(metrics, tokenizer, budget: int) -> str:
    """How generation ended: on which token, or against the budget.

    Printed because the two look identical from the text. A continuation that
    ended on ``<|im_end|>`` and one that was still going when the budget ran out
    both come back as a plausible sentence, and only the second is a finding --
    it means the stop set did not contain what this checkpoint actually emits,
    and every latency percentile above it averages over tokens the model never
    meant to send.
    """
    token = metrics.stopped_on_token
    if token is None:
        return f"ran the whole {budget}-token budget -- nothing in the stop set was emitted"
    name = None
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            name = tokenizer.convert_ids_to_tokens(token)
        except Exception:  # a tokenizer that cannot name an id has answered
            name = None
    return f"stopped on {name!r} ({token})" if name else f"stopped on token {token}"


def stop_at_eos(produced: list[int], eos_ids: set[int]) -> list[int]:
    for index, token in enumerate(produced):
        if token in eos_ids:
            return produced[:index]
    return produced


def eos_ids_for(config: dict, tokenizer, family: ModelFamily | None = None) -> set[int]:
    """Every id that should end a continuation, from all three places one can be declared.

    ``config.json``, the tokenizer, and the family's own stop tokens. The third
    is not belt and braces in either direction. ``glm-4-9b-chat-1m``'s
    ``config.json`` declares no ``eos_token_id``, and on the ``transformers``
    4.28 path :func:`load_tokenizer` builds for, ``tokenizer.eos_token_id`` is
    ``None`` as well: without it the stop set is **empty**, nothing ever halts
    generation, and every continuation runs its whole budget. Qwen2.5 is the
    milder version of the same thing -- its config names ``<|im_end|>`` and its
    ``generation_config.json`` names ``<|endoftext|>`` beside it, and only the
    first is read here. Either way the difference is between an answer and an
    answer followed by whatever the model says next.

    The family defaults to whichever one ``config`` names, so nothing borrows
    another model's ids by being asked the wrong question.
    """
    configured = config.get("eos_token_id")
    ids = set(configured if isinstance(configured, list) else [configured] if configured is not None else [])
    if tokenizer is None:
        return ids
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(tokenizer.eos_token_id)
    return ids | (family or family_for(config)).stop_ids(tokenizer)


def dummy_prompt_ids() -> torch.Tensor:
    """The ``--dummy-prompt`` input, flattened to the one sequence the runner takes."""
    return torch.ones(DUMMY_PROMPT_SHAPE, dtype=torch.long)[0]


def load_tokenizer(model_path: str, family: ModelFamily):
    """The checkpoint's tokenizer; GLM-4's remote-code one on ``transformers`` 4.28 as well.

    Only GLM-4 takes the long way round, and only because its tokenizer needs it:
    ``ChatGLM4Tokenizer._convert_token_to_id`` looks tokens up in its tiktoken
    ``mergeable_ranks``, which hold no special tokens. Newer ``transformers``
    registers ``<|endoftext|>``, ``[gMASK]``, ``<sop>`` and the rest from
    ``tokenizer_config.json``'s ``added_tokens_decoder`` before anything looks
    them up; 4.28 does not read that field, so two things break:

    * ``from_pretrained`` ends in ``sanitize_special_tokens()``, which looks the
      special tokens up -- ``KeyError: '<|endoftext|>'``. It is suppressed for
      that one call and restored after. Letting it run would not help anyway:
      it would append the tokens at fresh ids past the vocabulary.
    * Every later lookup -- the ``[gMASK]<sop>`` prefix on each encode, the
      special ids ``decode`` skips, ``eos_token_id`` -- hits the same
      ``KeyError``. So the tokens the config declares are registered at the ids
      it declares, which is what newer releases do on their own.

    On a ``transformers`` that already registered them, both steps are no-ops.
    Every other family is an ordinary ``from_pretrained``: Qwen2.5's tokenizer is
    a plain ``Qwen2Tokenizer`` whose added-token table every supported
    ``transformers`` reads for itself, and shadowing a base-class method to load
    it would be a workaround for a problem it does not have.
    """
    from transformers import AutoTokenizer

    if family is not GLM4:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    # Wherever the method is defined, the class attribute shadows it for the
    # one call; afterwards either the class's own method goes back or the
    # shadow is removed, so nothing outlives the load.
    patched = hasattr(PreTrainedTokenizerBase, "sanitize_special_tokens")
    own = PreTrainedTokenizerBase.__dict__.get("sanitize_special_tokens")
    if patched:
        PreTrainedTokenizerBase.sanitize_special_tokens = lambda self: 0
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    finally:
        if own is not None:
            PreTrainedTokenizerBase.sanitize_special_tokens = own
        elif patched:
            del PreTrainedTokenizerBase.sanitize_special_tokens
    register_declared_special_tokens(tokenizer, model_path)
    return tokenizer


def register_declared_special_tokens(tokenizer, model_path: str) -> list[str]:
    """Give the tokenizer the special tokens ``tokenizer_config.json`` declares, at their ids.

    Returns the tokens that were missing. Only a ``transformers`` that ignores
    ``added_tokens_decoder`` (4.28) is missing any; on it ``added_tokens_encoder``
    and ``added_tokens_decoder`` are plain dicts, and ``unique_no_split_tokens``
    keeps ``tokenize`` from splitting a special token that appears in the text.
    """
    path = Path(model_path) / "tokenizer_config.json"
    if not path.is_file():
        return []
    declared = json.loads(path.read_text(encoding="utf-8")).get("added_tokens_decoder", {})
    missing = {
        int(index): entry["content"]
        for index, entry in declared.items()
        if entry["content"] not in tokenizer.added_tokens_encoder
    }
    if not missing:
        return []
    for index, content in missing.items():
        tokenizer.added_tokens_encoder[content] = index
        tokenizer.added_tokens_decoder[index] = content
    tokenizer.unique_no_split_tokens = sorted(set(tokenizer.unique_no_split_tokens) | set(missing.values()))
    tokenizer._create_trie(tokenizer.unique_no_split_tokens)
    return sorted(missing.values())


def render(tokenizer, token_ids: list[int]) -> str:
    """Text when there is a tokenizer, the raw ids under ``--dummy-prompt``.

    A tokenizer that cannot decode what the model produced falls back to the ids
    rather than ending the run. GLM-4's remote-code tokenizer raises for an id
    outside its tiktoken ranks -- which a wrong-basis decode emits, and which is
    exactly the run whose output is worth seeing.
    """
    if tokenizer is None:
        return f"ids {token_ids}"
    text, failure = decode_text(tokenizer, token_ids)
    return repr(text) if failure is None else f"<undecodable: {failure}> ids {token_ids}"


def decode_text(tokenizer, token_ids: list[int]) -> tuple[str, str | None]:
    """The continuation as text, and why it is empty when it is.

    A decode that raises leaves the scoring to fail honestly -- an empty string
    matches no needle -- instead of taking the run down with it. Which is worth
    distinguishing from a model that answered wrongly, hence the second value.
    """
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True), None
    except Exception as error:  # a decode failure is a finding, not the end of the run
        return "", f"{type(error).__name__}: {' '.join(str(error).split())[:80]}"


def build_runner(
    args,
    backend: str,
    prefill_mode: str,
    max_seq_len: int,
    dense_attention_api: str | None = None,
    dense_staging_backend: str | None = None,
) -> StandaloneModelRunner:
    runner = StandaloneModelRunner(
        RunnerConfig(
            model_path=args.model_path,
            backend=backend,
            prefill_mode=prefill_mode,
            max_seq_len=max_seq_len,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            dtype=_DTYPES[args.dtype],
            device=str(args.device),
            max_new_tokens=args.max_new_tokens,
            # A dense baseline's decode never rotates anything, so there is nothing to fold for.
            fold_output_rotation=not args.no_fold_output_rotation and backend not in DENSE_BACKENDS,
            seed=args.seed,
            fold_site=args.fold_site,
            dense_attention_api=dense_attention_api,
            dense_staging_backend=dense_staging_backend,
            decode_graph=args.decode_graph,
        )
    )
    assert_no_vllm_imported()
    return runner


class PreflightFailed(SystemExit):
    """Raised before any weight loads, with every probe's verdict in the message."""


@dataclass
class AttentionPlan:
    """What the pre-flight settled: the prefill mode, the staging backend, and each dense path's entry point."""

    prefill_mode: str
    #: ``native_v5``'s entry point when it is the decode (or reference) backend, and
    #: ``"staging"``'s when the dense_staging pool only prefills. Absent: torch_npu's default.
    dense_apis: dict[str, str | None] = field(default_factory=dict)
    #: Which unquantised backend the dense_staging pool passed its probe as.
    #: ``None`` where there is no staging pool, or where it failed everywhere.
    staging_backend: str | None = None
    results: list[ProbeResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def dense_api_for(self, backend: str) -> str | None:
        """The entry point a runner decoding with ``backend`` hands its dense pool."""
        return self.dense_apis.get(backend if backend in DENSE_BACKENDS else "staging")

    def report(self) -> str:
        lines = [f"  {result.describe()}" for result in self.results]
        lines += [f"  {note}" for note in self.notes]
        return "\n".join(lines)


def plan_attention(args, shape, prefill_mode: str) -> AttentionPlan:
    """Probe every attention path the run will take, on one synthetic layer, before any weight loads.

    ``shape`` is the checkpoint's own, so the probe runs the head counts the run
    will run rather than a stand-in: a GQA ratio is what decides how a fused
    kernel groups its rows, and an entry point that is refused at 16/2 and
    accepted at 32/4 would pass a probe that guessed.

    Raises :class:`PreflightFailed` when a path the run cannot do without fails.
    The one it can do without is the dense_staging prefill: when that fails and
    ``--prefill-mode`` was not given, the run prefills through the decode backend
    (``batched_decode``) instead, which the decode backend's probe has just run.
    """
    plan = AttentionPlan(prefill_mode)
    layer = LayerShape(shape.num_heads, shape.num_kv_heads, shape.head_size, shape.scale)
    probe = {"device": args.device, "dtype": _DTYPES[args.dtype], "block_size": args.block_size}
    folded = not args.no_fold_output_rotation

    def need(passed: bool, what: str, hint: str) -> None:
        if not passed:
            raise PreflightFailed(
                f"pre-flight failed before loading any weights: {what}\n{plan.report()}\n  hint: {hint}"
            )

    def dense_decode(backend: str) -> None:
        selection = select_dense_api(backend, layer, **probe, role="decode")
        plan.results.extend(selection.results)
        need(
            selection.passed,
            f"no attention entry point runs {backend} here",
            "Ascend 950 refuses aclnnFusedInferAttentionScore V1-V4 (EZ9903), which is as far as op-plugin's "
            "entry points reach; --backend cann_dense is the unquantised baseline that does not call it.",
        )
        plan.dense_apis[backend] = selection.api

    if args.backend in DENSE_BACKENDS:
        dense_decode(args.backend)
    else:
        # prefill_chunk as well as decode: that is batched_decode's prefill path.
        decode = probe_backend(args.backend, layer, **probe, output_rotation_folded=folded)
        plan.results.append(decode)
        need(decode.passed, f"the {args.backend} decode does not run here", "see the error on its line above")
        if prefill_mode == "dense_staging":
            staging = dense_backend_for(args.device)
            selection = select_dense_backend(
                args.device,
                layer,
                role="prefill",
                output_rotation_folded=folded,
                dtype=probe["dtype"],
                block_size=probe["block_size"],
            )
            plan.results.extend(selection.results)
            if selection.passed:
                plan.dense_apis["staging"] = selection.api
                plan.staging_backend = selection.backend
                if selection.backend != staging:
                    plan.notes.append(
                        f"the dense_staging prefill runs through {selection.backend}, not {staging}: no fused "
                        "attention entry point is accepted here (see the FAILED lines above)"
                    )
            elif args.prefill_mode is None:
                plan.prefill_mode = "batched_decode"
                plan.notes.append(
                    f"the dense_staging prefill ({staging}) does not run here; prefilling through the "
                    f"{args.backend} decode instead (batched_decode), which passed above"
                )
            else:
                need(
                    False,
                    f"--prefill-mode dense_staging, but its unquantised prefill ({staging}) does not run here",
                    f"--prefill-mode batched_decode prefills through the {args.backend} decode, which passed",
                )

    reference = args.reference_backend
    if reference and reference != args.backend:
        if reference in DENSE_BACKENDS:
            dense_decode(reference)
        else:
            result = probe_backend(reference, layer, **probe, output_rotation_folded=folded)
            plan.results.append(result)
            need(result.passed, f"the reference backend {reference} does not run here", "drop --reference-backend")
    return plan


def check_contract(runner: StandaloneModelRunner) -> str:
    """Refuse a run that is not the ungated folded-o_proj contract; return its output stage's name.

    Neither GLM-4 nor Qwen2.5 gates its attention output, so ``o_proj`` consumes
    the decode's output directly and Pi folds into it offline. A gate would sit
    between the two, where the output is still rotated, and the fold would be
    applied to the wrong side of it.
    """
    if runner.shape.attn_output_gate:
        raise RuntimeError("this contract is the ungated one, but the runner built an output gate")
    backend = runner.decode_backend
    folded = backend.output_rotation_folded
    if backend.name in DENSE_BACKENDS:
        return "dense, o_proj folded" if folded else "dense (no rotation)"
    if isinstance(backend, TurboQuantCubeBackend):
        stage = backend._output_stage(None)
        stages = turboquant_layout().TurboQuantOutputStage
        expected = stages.ROTATED_BASIS if folded else stages.UNROTATED
        if stage != expected:
            raise RuntimeError(f"the Cube decode selected {stage.name}, not {expected.name}")
        return f"{stage.name} (stage {int(stage)})"
    return "folded o_proj, no un-rotation" if folded else "unfolded, un-rotated after the decode"


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_checkpoint(args.model_path)
    args.device = resolve_device(args.device)
    depths = parse_depths(args.depths)
    config = read_config(args.model_path)
    # Raises for a GLM the harness does not build (GLM-4-0414's sandwich norms);
    # anything else it does not recognise gets families.GENERIC, whose tokenizer
    # and config are then the only authorities on the prompt and the stop set.
    family = family_for(config)
    # Reads config.json and the shard headers, no tensor data. The pre-flight
    # probes this shape and the runner builds it from the same reader, so the
    # layer that is probed is the layer that will run.
    checkpoint_shape = read_checkpoint_shape(args.model_path)
    if args.dummy_prompt and args.tokens:
        # Nothing to build a haystack out of without a tokenizer.
        print(f"  --dummy-prompt: skipping the needle-in-a-haystack stage (--tokens {args.tokens})", file=sys.stderr)
        args.tokens = 0
    prefill_mode = choose_prefill_mode(args, family)

    if args.skip_preflight:
        plan = AttentionPlan(prefill_mode)
    else:
        print("=== 0. pre-flight: one synthetic layer through every attention path, no weights ===")
        started = time.perf_counter()
        plan = plan_attention(args, checkpoint_shape, prefill_mode)
        print(plan.report())
        print(f"  {time.perf_counter() - started:.2f} s; prefill {plan.prefill_mode}")
        print()
    prefill_mode = plan.prefill_mode

    use_chat_template = not args.raw_prompt
    if args.dummy_prompt:
        tokenizer = None
        short_ids = dummy_prompt_ids()
    else:
        tokenizer = load_tokenizer(args.model_path, family)
        short_ids = encode(tokenizer, DEFAULT_PROMPT, use_chat_template, family)
    eos_ids = eos_ids_for(config, tokenizer, family)
    max_seq_len = max(args.tokens, short_ids.numel()) + args.max_new_tokens + CONTEXT_HEADROOM_TOKENS

    print("=== 1. the contract ===")
    runner = build_runner(
        args, args.backend, prefill_mode, max_seq_len, plan.dense_api_for(args.backend), plan.staging_backend
    )
    shape = runner.shape
    print(
        f"  {family.name} ({config.get('model_type')!r}): {shape.num_layers} layers, {shape.num_heads} heads / "
        f"{shape.num_kv_heads} kv, D={shape.head_size}, ffn {shape.intermediate_size}, vocab {shape.vocab_size}"
    )
    print(
        f"  rope: theta {shape.rope_theta:.6g}, rotary_dim {shape.rotary_dim} of {shape.head_size}, "
        f"interleaved={shape.rope_interleaved}; qkv_bias={shape.qkv_bias} gate={shape.attn_output_gate}"
    )
    # Both of these used to be decided silently, by which transformers happened
    # to be installed. An empty stop set in particular is invisible from the
    # output: it looks exactly like a model with more to say.
    print(f"  {runner.fold_report.describe()}")
    print(f"  prompt: {prompt_route(tokenizer, use_chat_template, family)}")
    print(f"  stop ids: {sorted(eos_ids) if eos_ids else 'NONE -- generation will run its whole budget'}")
    memory = runner.memory_report()
    print(
        f"  backend {args.backend} on {args.device}, prefill {prefill_mode}, max_seq_len {max_seq_len}, "
        f"output stage {check_contract(runner)}"
    )
    print(
        f"  kv {memory['kv_cache_mb']:.1f} MB held (dense equivalent {memory['dense_equivalent_mb']:.1f} MB), "
        f"{memory['bytes_per_token_per_layer']:.0f} B/token/layer"
    )

    print("\n=== 2. short prompt ===")
    produced, metrics = runner.generate(
        short_ids.to(runner.device), max_new_tokens=args.max_new_tokens, stop_ids=eos_ids
    )
    produced = stop_at_eos(produced, eos_ids)
    summary = metrics.summary()
    print(f"  {args.backend:20s} {render(tokenizer, produced)}")
    print(
        f"  {'':20s} ttft {summary['ttft_ms']:.0f} ms, decode p50 {summary['decode_p50_us'] / 1e3:.2f} ms "
        f"({summary['decode_tokens_per_second']:.1f} tok/s), p90 {summary['decode_p90_us'] / 1e3:.2f}, "
        f"p99 {summary['decode_p99_us'] / 1e3:.2f}, peak {summary['device_peak_mb']:.0f} MB"
    )
    print(f"  {'':20s} {describe_decode(runner, summary)}")
    print(f"  {'':20s} {describe_stop(metrics, tokenizer, args.max_new_tokens)}")

    if args.reference_backend:
        reference_mode = "dense_staging" if args.reference_backend in DENSE_BACKENDS else prefill_mode
        del runner
        _free(args.device)
        reference = build_runner(
            args,
            args.reference_backend,
            reference_mode,
            short_ids.numel() + args.max_new_tokens + 64,
            plan.dense_api_for(args.reference_backend),
        )
        expected, _ = reference.generate(
            short_ids.to(reference.device), max_new_tokens=args.max_new_tokens, stop_ids=eos_ids
        )
        expected = stop_at_eos(expected, eos_ids)
        agreed = sum(mine == theirs for mine, theirs in zip(produced, expected))
        print(f"  {args.reference_backend:20s} {render(tokenizer, expected)}  ({agreed}/{len(expected)} tokens agree)")
        del reference
        _free(args.device)
        runner = (
            build_runner(
                args, args.backend, prefill_mode, max_seq_len, plan.dense_api_for(args.backend), plan.staging_backend
            )
            if args.tokens
            else None
        )

    if args.tokens:
        print(f"\n=== 3. needle in a haystack at ~{args.tokens} tokens ===")
        keep = max_seq_len - CONTEXT_HEADROOM_TOKENS - args.max_new_tokens
        hits = clean = 0
        for depth in depths:
            item = needle_in_a_haystack(args.tokens, depth=depth, seed=int(depth * 100) + args.seed)
            prompt_ids = truncate_middle(encode(tokenizer, item.prompt, use_chat_template, family), keep)
            produced, metrics = runner.generate(
                prompt_ids.to(runner.device), max_new_tokens=args.max_new_tokens, stop_ids=eos_ids
            )
            answer, failure = decode_text(tokenizer, stop_at_eos(produced, eos_ids))
            report = needle_report(answer, item.answers[0], item.extra.get("city"))
            hits += report.found
            clean += report.clean
            summary = metrics.summary()
            print(
                f"    depth {depth:<5} tokens={prompt_ids.numel():>7} {report.describe():<11s} "
                f"ttft {summary['ttft_ms']:.0f} ms decode p50 {summary['decode_p50_us']:.0f} us "
                f"{answer.strip()[:40]!r}" + (f" <undecodable: {failure}>" if failure else "")
            )
        # Both counts, always: a run where they differ retrieved the needle and
        # then kept going, which "3/3 retrieved" on its own reads as a clean pass.
        print(f"  {args.backend}: {hits}/{len(depths)} retrieved, {clean}/{len(depths)} clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
