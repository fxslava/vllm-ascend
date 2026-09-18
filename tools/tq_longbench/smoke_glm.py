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

Three stages, in order:

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
import gc  # noqa: E402

import torch  # noqa: E402

from tq_longbench._ascend import assert_no_vllm_imported, turboquant_layout  # noqa: E402
from tq_longbench.engine import DEFAULT_CHUNK_SIZE, PREFILL_MODES, RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.glm4 import (  # noqa: E402
    GLM4_BATCHED_DECODE_MIN_TOKENS,
    glm4_prefill_mode,
    is_glm4_config,
    read_config,
)
from tq_longbench.ops import DENSE_BACKENDS, TurboQuantCubeBackend, backend_names  # noqa: E402
from tq_longbench.tasks import needle_in_a_haystack  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

DEFAULT_PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"

DEFAULT_DEPTHS = "0.0,0.5,1.0"

#: Cache room beyond the prompt and the generated tokens, so a tokenised prompt
#: a little over its estimate still fits.
CONTEXT_HEADROOM_TOKENS = 512


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


def _eos_ids(config: dict, tokenizer) -> set[int]:
    configured = config.get("eos_token_id")
    ids = set(configured if isinstance(configured, list) else [configured] if configured is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def build_runner(args, backend: str, prefill_mode: str, max_seq_len: int) -> StandaloneModelRunner:
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
        )
    )
    assert_no_vllm_imported()
    return runner


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
    prefill_mode = choose_prefill_mode(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_ids = _eos_ids(config, tokenizer)
    use_chat_template = not args.raw_prompt
    short_ids = encode(tokenizer, DEFAULT_PROMPT, use_chat_template)
    max_seq_len = max(args.tokens, short_ids.numel()) + args.max_new_tokens + CONTEXT_HEADROOM_TOKENS

    print("=== 1. the contract ===")
    runner = build_runner(args, args.backend, prefill_mode, max_seq_len)
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
    print(f"  {args.backend:20s} {tokenizer.decode(produced, skip_special_tokens=True)!r}")
    print(f"  {'':20s} decode p50 {summary['decode_p50_us']:.0f} us, p99 {summary['decode_p99_us']:.0f} us")

    if args.reference_backend:
        reference_mode = "dense_staging" if args.reference_backend in DENSE_BACKENDS else prefill_mode
        del runner
        _free(args.device)
        reference = build_runner(
            args, args.reference_backend, reference_mode, short_ids.numel() + args.max_new_tokens + 64
        )
        expected, _ = reference.generate(short_ids.to(reference.device), max_new_tokens=args.max_new_tokens)
        expected = stop_at_eos(expected, eos_ids)
        agreed = sum(mine == theirs for mine, theirs in zip(produced, expected))
        print(
            f"  {args.reference_backend:20s} {tokenizer.decode(expected, skip_special_tokens=True)!r}  "
            f"({agreed}/{len(expected)} tokens agree)"
        )
        del reference
        _free(args.device)
        runner = build_runner(args, args.backend, prefill_mode, max_seq_len) if args.tokens else None

    if args.tokens:
        print(f"\n=== 3. needle in a haystack at ~{args.tokens} tokens ===")
        keep = max_seq_len - CONTEXT_HEADROOM_TOKENS - args.max_new_tokens
        hits = 0
        for depth in depths:
            item = needle_in_a_haystack(args.tokens, depth=depth, seed=int(depth * 100) + args.seed)
            prompt_ids = truncate_middle(encode(tokenizer, item.prompt, use_chat_template), keep)
            produced, metrics = runner.generate(prompt_ids.to(runner.device), max_new_tokens=args.max_new_tokens)
            answer = tokenizer.decode(stop_at_eos(produced, eos_ids), skip_special_tokens=True)
            found = item.answers[0] in answer
            hits += found
            summary = metrics.summary()
            print(
                f"    depth {depth:<5} tokens={prompt_ids.numel():>7} found={str(found):5s} "
                f"prefill {summary['prefill_seconds']:.1f}s decode p50 {summary['decode_p50_us']:.0f} us "
                f"{answer.strip()[:40]!r}"
            )
        print(f"  {args.backend}: {hits}/{len(depths)} retrieved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
