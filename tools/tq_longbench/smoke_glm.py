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
"""Smoke-test GLM-4 (``THUDM/glm-4-9b-chat-1m``) through the standalone runner.

    python tools/tq_longbench/smoke_glm.py --model-path /models/glm-4-9b-chat-1m \\
        --device npu --backend turboquant_cube --tokens 131072

Four stages, in order:

0. **Pre-flight**, before a byte of the checkpoint is read: every attention path
   the run will take -- the decode backend, the dense_staging prefill, the
   reference backend -- runs one synthetic layer with GLM-4's head counts and is
   checked against exact attention (:mod:`tq_longbench.preflight`). An operator
   the SoC refuses fails here in well under the time ingestion takes. When the
   dense prefill is what fails and ``--prefill-mode`` was left to default, the
   run switches to ``batched_decode`` rather than stopping; the torch_npu entry
   point that did pass is the one the run then uses. ``--skip-preflight`` skips it.
1. **The contract.** The shape read off ``config.json`` (head counts, the
   half-width interleaved RoPE, the qkv bias), and the decode's output stage:
   ungated, with ``W_o' = W_o (I (x) Pi)`` folded at ingestion, the Cube decode
   must run ``ROTATED_BASIS`` -- the kernel's ``kRotatedBasis``, stage 0, where
   its output passes straight to ``o_proj`` with nothing un-rotated in-kernel.
   Anything else is refused before a token is generated.
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
``transformers`` cannot load GLM-4's tokenizer. Without it, :func:`load_tokenizer`
carries the tokenizer through ``transformers`` 4.28 (see there).
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
from tq_longbench.engine import DEFAULT_CHUNK_SIZE, PREFILL_MODES, RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.glm4 import (  # noqa: E402
    GLM4_BATCHED_DECODE_MIN_TOKENS,
    glm4_model_shape,
    glm4_prefill_mode,
    is_glm4_config,
    read_config,
)
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
        description="Run GLM-4 (glm-4-9b-chat-1m) through the standalone runner with the folded-o_proj contract.",
    )
    parser.add_argument("--model-path", required=True, help="directory holding config.json and safetensors shards")
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
        help=f"default: batched_decode from {GLM4_BATCHED_DECODE_MIN_TOKENS} context tokens up, dense_staging below",
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
    parser.add_argument("--seed", type=int, default=0)
    return parser


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


def choose_prefill_mode(args: argparse.Namespace) -> str:
    mode = args.prefill_mode or glm4_prefill_mode(args.tokens, args.backend)
    if mode == "dense_staging" and args.tokens >= GLM4_BATCHED_DECODE_MIN_TOKENS and args.backend not in DENSE_BACKENDS:
        print(
            f"  warning: dense_staging at {args.tokens} tokens allocates a full fp16 KV pool beside the quantised "
            "one; batched_decode is the default here because that pool is what exhausts HBM",
            file=sys.stderr,
        )
    return mode


def encode(tokenizer, text: str, use_chat_template: bool) -> torch.Tensor:
    """Token ids for one user turn, through the chat template when the tokenizer has one."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
    else:
        ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[0].to(torch.int64)


def truncate_middle(token_ids: torch.Tensor, keep: int) -> torch.Tensor:
    """Drop from the middle, as LongBench does: the head has the instructions, the tail the question."""
    if token_ids.numel() <= keep:
        return token_ids
    half = keep // 2
    return torch.cat((token_ids[:half], token_ids[-(keep - half) :]))


def stop_at_eos(produced: list[int], eos_ids: set[int]) -> list[int]:
    for index, token in enumerate(produced):
        if token in eos_ids:
            return produced[:index]
    return produced


def eos_ids_for(config: dict, tokenizer) -> set[int]:
    configured = config.get("eos_token_id")
    ids = set(configured if isinstance(configured, list) else [configured] if configured is not None else [])
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def dummy_prompt_ids() -> torch.Tensor:
    """The ``--dummy-prompt`` input, flattened to the one sequence the runner takes."""
    return torch.ones(DUMMY_PROMPT_SHAPE, dtype=torch.long)[0]


def load_tokenizer(model_path: str):
    """GLM-4's remote-code tokenizer, loadable on ``transformers`` 4.28 as well.

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
    """
    from transformers import AutoTokenizer
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
            dense_attention_api=dense_attention_api,
            dense_staging_backend=dense_staging_backend,
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


def plan_attention(args, config: dict, prefill_mode: str) -> AttentionPlan:
    """Probe every attention path the run will take, on one synthetic layer, before any weight loads.

    Raises :class:`PreflightFailed` when a path the run cannot do without fails.
    The one it can do without is the dense_staging prefill: when that fails and
    ``--prefill-mode`` was not given, the run prefills through the decode backend
    (``batched_decode``) instead, which the decode backend's probe has just run.
    """
    plan = AttentionPlan(prefill_mode)
    shape = glm4_model_shape(config)
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
    """Refuse a run that is not GLM-4's attention contract; return its output stage's name."""
    if runner.shape.attn_output_gate:
        raise RuntimeError("GLM-4 has no output gate, but the runner built one")
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
    args.device = resolve_device(args.device)
    depths = parse_depths(args.depths)
    config = read_config(args.model_path)
    if not is_glm4_config(config):
        raise SystemExit(f"{args.model_path} is model_type {config.get('model_type')!r}, not a GLM-4 checkpoint")
    if args.dummy_prompt and args.tokens:
        # Nothing to build a haystack out of without a tokenizer.
        print(f"  --dummy-prompt: skipping the needle-in-a-haystack stage (--tokens {args.tokens})", file=sys.stderr)
        args.tokens = 0
    prefill_mode = choose_prefill_mode(args)

    if args.skip_preflight:
        plan = AttentionPlan(prefill_mode)
    else:
        print("=== 0. pre-flight: one synthetic layer through every attention path, no weights ===")
        started = time.perf_counter()
        plan = plan_attention(args, config, prefill_mode)
        print(plan.report())
        print(f"  {time.perf_counter() - started:.2f} s; prefill {plan.prefill_mode}")
        print()
    prefill_mode = plan.prefill_mode

    use_chat_template = not args.raw_prompt
    if args.dummy_prompt:
        tokenizer = None
        short_ids = dummy_prompt_ids()
    else:
        tokenizer = load_tokenizer(args.model_path)
        short_ids = encode(tokenizer, DEFAULT_PROMPT, use_chat_template)
    eos_ids = eos_ids_for(config, tokenizer)
    max_seq_len = max(args.tokens, short_ids.numel()) + args.max_new_tokens + CONTEXT_HEADROOM_TOKENS

    print("=== 1. the contract ===")
    runner = build_runner(
        args, args.backend, prefill_mode, max_seq_len, plan.dense_api_for(args.backend), plan.staging_backend
    )
    shape = runner.shape
    print(
        f"  {shape.num_layers} layers, {shape.num_heads} heads / {shape.num_kv_heads} kv, D={shape.head_size}, "
        f"ffn {shape.intermediate_size}, vocab {shape.vocab_size}"
    )
    print(
        f"  rope: theta {shape.rope_theta:.6g}, rotary_dim {shape.rotary_dim} of {shape.head_size}, "
        f"interleaved={shape.rope_interleaved}; qkv_bias={shape.qkv_bias} gate={shape.attn_output_gate}"
    )
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
    produced, metrics = runner.generate(short_ids.to(runner.device), max_new_tokens=args.max_new_tokens)
    produced = stop_at_eos(produced, eos_ids)
    summary = metrics.summary()
    print(f"  {args.backend:20s} {render(tokenizer, produced)}")
    print(
        f"  {'':20s} ttft {summary['ttft_ms']:.0f} ms, decode p50 {summary['decode_p50_us']:.0f} us, "
        f"p99 {summary['decode_p99_us']:.0f} us, peak {summary['device_peak_mb']:.0f} MB"
    )

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
        expected, _ = reference.generate(short_ids.to(reference.device), max_new_tokens=args.max_new_tokens)
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
            prompt_ids = truncate_middle(encode(tokenizer, item.prompt, use_chat_template), keep)
            produced, metrics = runner.generate(prompt_ids.to(runner.device), max_new_tokens=args.max_new_tokens)
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
