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
"""Retrieval and latency across a context ladder, one backend against another.

    python tools/tq_longbench/run_benchmark.py --model-path /models/glm-4-9b-chat-1m \\
        --backends turboquant_cube,cann_dense --contexts 4096,8192,16384,32768,65536 \\
        --out-file bench.jsonl

``smoke_glm.py`` answers "does this backend work at all". This answers "what does
it cost, and does it still retrieve, as the context grows" -- which is the same
needle at every rung of a ladder, with the clock and the allocator read at each.

**One runner per rung.** A runner's KV pool is static and sized by ``max_seq_len``,
so a 4k rung measured inside a 64k runner would report 64k's memory. Each
(backend, context) therefore builds its own runner and tears it down, which costs
a weight load per rung -- minutes on a 9B checkpoint. ``--reuse-runner`` trades
that away: one runner per backend, sized for the longest rung, and the memory
columns then describe that size at every rung rather than the rung's own.

**What is reported.** Per (backend, context): retrieval over the depths, TTFT in
milliseconds (the prompt in, the first token chosen -- prefill plus the argmax
over its logits), decode P50 and P99 in microseconds per token, the KV megabytes
held against the dense equivalent, and the allocator's own peak. The first decode
step of a sequence carries first-touch costs that no later step does, so it lands
in P99 and not in P50; both are reported rather than either alone.

The generation budget defaults to :data:`DEFAULT_MAX_NEW_TOKENS` because the
needle is a six-digit passcode behind whatever preamble the chat template invites:
a budget that ends mid-number scores as a miss that was not one. It is a *budget*,
not a length -- generation still stops at EOS.

A record per item is written as it completes (JSONL, ``--out-file``), so a ladder
that dies at 64k still leaves 32k and below on disk.
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
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402

# smoke_glm sets USE_TF/TF_ENABLE_ONEDNN_OPTS as it is imported, before anything
# here can reach transformers; quiet_tensorflow() is called again below so that
# this file states the dependency rather than relying on an import side effect.
from tq_longbench.smoke_glm import (  # noqa: E402
    CONTEXT_HEADROOM_TOKENS,
    AttentionPlan,
    PreflightFailed,
    decode_text,
    encode,
    eos_ids_for,
    load_tokenizer,
    model_directory,
    plan_attention,
    prompt_route,
    quiet_tensorflow,
    resolve_device,
    stop_at_eos,
    truncate_middle,
)

quiet_tensorflow()

from tq_longbench._ascend import assert_no_vllm_imported  # noqa: E402
from tq_longbench.engine import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    PREFILL_MODES,
    RunnerConfig,
    StandaloneModelRunner,
    read_checkpoint_shape,
)
from tq_longbench.families import ModelFamily, family_for  # noqa: E402
from tq_longbench.glm4 import read_config  # noqa: E402
from tq_longbench.layers import FOLD_SITES  # noqa: E402
from tq_longbench.ops import DENSE_BACKENDS, backend_names  # noqa: E402
from tq_longbench.tasks import needle_in_a_haystack, needle_report  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

#: The ladder, when ``--contexts`` is left alone. Doubling rungs, because what is
#: being looked for is the shape of the curve rather than a number at one length.
DEFAULT_CONTEXTS = "4096,8192,16384,32768,65536"

#: Enough of a continuation to carry the whole passcode out from behind a
#: preamble. The needle's own budget (16, in :mod:`tq_longbench.tasks`) is sized
#: for a bare completion; a chat template's "The secret passcode for Lisbon is"
#: spends most of that before reaching a digit.
DEFAULT_MAX_NEW_TOKENS = 48

#: Where the needle is buried. The ends are in because they are what a
#: context-length bug takes first: a decode short by a block loses the tail, and
#: a prefill that starts mid-prompt loses the head.
DEFAULT_DEPTHS = "0.0,0.25,0.5,0.75,1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/run_benchmark.py",
        description="Needle-in-a-haystack retrieval, TTFT, decode percentiles and HBM across a context ladder.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=model_directory,
        help="directory holding config.json and safetensors shards; a Windows path in either slash is fine",
    )
    parser.add_argument("--device", default="npu", help="npu, npu:N, cuda:N or cpu (default: npu)")
    parser.add_argument(
        "--backends",
        default="turboquant_cube",
        help=f"comma-separated, run in order; one of {backend_names()}",
    )
    parser.add_argument(
        "--contexts",
        default=DEFAULT_CONTEXTS,
        help=f"comma-separated approximate prompt lengths in tokens (default: {DEFAULT_CONTEXTS})",
    )
    parser.add_argument("--depths", default=DEFAULT_DEPTHS, help="comma-separated needle depths in [0, 1]")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"generation budget per item (default: {DEFAULT_MAX_NEW_TOKENS}); note this is tokens *out*, "
        "where smoke_glm.py's --tokens is context in",
    )
    parser.add_argument(
        "--prefill-mode",
        default=None,
        choices=list(PREFILL_MODES),
        help="default: the checkpoint's own policy at each rung's length",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument(
        "--no-fold-output-rotation",
        action="store_true",
        help="leave o_proj unfolded so the decode un-rotates in-kernel (UNROTATED stage), for an A/B",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="skip the tokenizer's chat template")
    parser.add_argument(
        "--reuse-runner",
        action="store_true",
        help="one runner per backend sized for the longest rung, instead of one per rung; faster, and the "
        "memory columns then report that size at every rung",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the synthetic one-layer probe of every attention path before the weights load",
    )
    parser.add_argument(
        "--fold-site",
        default="auto",
        choices=list(FOLD_SITES),
        help="where the o_proj fold runs: auto (the device when there is one), host, or device",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default=None, help="JSONL results; stdout if omitted")
    return parser


def parse_int_list(text: str, what: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"--{what} must be comma-separated positive integers, got {text!r}")
    return values


def parse_depths(text: str) -> tuple[float, ...]:
    depths = tuple(float(part) for part in text.split(",") if part.strip())
    if not depths or any(not 0.0 <= depth <= 1.0 for depth in depths):
        raise ValueError(f"--depths must be comma-separated values in [0, 1], got {text!r}")
    return depths


def parse_backends(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in backend_names()]
    if not names or unknown:
        raise ValueError(f"--backends: {unknown or 'nothing given'}; choose from {backend_names()}")
    return names


def prefill_mode_for(args: argparse.Namespace, backend: str, context: int, family: ModelFamily) -> str:
    """What this rung prefills through: the flag if given, else the family's policy.

    Per rung rather than once, because the policy is a function of the length --
    GLM-4 stages a 16k prefill through the unquantised pool and refuses to at 64k,
    where that pool is what would exhaust HBM. A family nothing has sized stages
    at no length at all (``batched_decode_min_tokens`` 0), which is the same
    conservative answer this gave before any family was named.
    """
    if args.prefill_mode:
        return args.prefill_mode
    return family.prefill_mode(context, backend)


def plan_namespace(args: argparse.Namespace, backend: str) -> argparse.Namespace:
    """The fields :func:`tq_longbench.smoke_glm.plan_attention` reads, for one backend.

    Spelled out rather than passed through, so that a new flag on either CLI
    cannot silently change what the other one probes.
    """
    return argparse.Namespace(
        backend=backend,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        no_fold_output_rotation=args.no_fold_output_rotation,
        prefill_mode=args.prefill_mode,
        reference_backend=None,
    )


def build_runner(
    args: argparse.Namespace, backend: str, prefill_mode: str, max_seq_len: int, plan: AttentionPlan
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
            # Not every caller offers the flag; those that do not keep the
            # engine's own 'auto', which is what this module has always used.
            decode_graph=getattr(args, "decode_graph", "auto"),
            dense_attention_api=plan.dense_api_for(backend),
            dense_staging_backend=plan.staging_backend,
        )
    )
    assert_no_vllm_imported()
    return runner


def free_device(device: torch.device) -> None:
    """Reclaim what the last rung dropped, before the next one allocates.

    The caller drops its own reference *first* -- rebinding its ``runner`` to
    ``None`` -- because a runner still named anywhere is a runner whose pools are
    still held, and the point of tearing one down per rung is that the next one's
    memory figures describe the next one.
    """
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class Rung:
    """One (backend, context) pair: every depth run through one runner."""

    backend: str
    context: int
    prefill_mode: str
    records: list[dict] = field(default_factory=list)

    @property
    def hits(self) -> int:
        """Continuations that carried the passcode, wherever in them it turned up."""
        return sum(record["found"] for record in self.records)

    @property
    def clean(self) -> int:
        """Of those, the ones that led with it and did not fall into a loop.

        Reported next to :attr:`hits` rather than instead of it, because the gap
        between the two is what a retrieval column on its own cannot show: a
        model that answers and then recites the haystack has retrieved, and is
        still worth looking at.
        """
        return sum(record["clean"] for record in self.records)

    def median(self, key: str) -> float:
        values = [record[key] for record in self.records]
        return statistics.median(values) if values else 0.0

    def peak(self, key: str) -> float:
        return max((record[key] for record in self.records), default=0.0)


def run_rung(
    args: argparse.Namespace,
    runner: StandaloneModelRunner,
    tokenizer,
    eos_ids: set[int],
    backend: str,
    context: int,
    prefill_mode: str,
    depths: tuple[float, ...],
    keep: int,
    sink,
    family: ModelFamily,
) -> Rung:
    """One needle per depth at this context, each written out as it finishes."""
    rung = Rung(backend, context, prefill_mode)
    for depth in depths:
        item = needle_in_a_haystack(context, depth=depth, seed=int(depth * 100) + args.seed)
        prompt_ids = truncate_middle(encode(tokenizer, item.prompt, not args.raw_prompt, family), keep)
        started = time.perf_counter()
        produced, metrics = runner.generate(
            prompt_ids.to(runner.device), max_new_tokens=args.max_new_tokens, stop_ids=eos_ids
        )
        answer, failure = decode_text(tokenizer, stop_at_eos(produced, eos_ids))
        report = needle_report(answer, item.answers[0], item.extra.get("city"))
        record = {
            "backend": backend,
            "prefill_mode": prefill_mode,
            "context_requested": context,
            "depth": depth,
            "found": report.found,
            "clean": report.clean,
            "verdict": report.describe(),
            "first_number": report.first_number,
            "echoed_prompt": report.echoed_prompt,
            "degenerate": report.degenerate,
            "prediction": answer,
            "answers": item.answers,
            "decode_failure": failure,
            "wall_seconds": round(time.perf_counter() - started, 3),
            **metrics.summary(),
        }
        rung.records.append(record)
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        print(
            f"    depth {depth:<5} tokens={record['prompt_tokens']:>7} {record['verdict']:<11s} "
            f"ttft {record['ttft_ms']:>8.0f} ms  p50 {record['decode_p50_us']:>8.0f} us  "
            f"p99 {record['decode_p99_us']:>8.0f} us  peak {record['device_peak_mb']:>7.0f} MB  "
            f"{answer.strip()[:40]!r}" + (f"  <undecodable: {failure}>" if failure else ""),
            file=sys.stderr,
        )
    return rung


def summarise(rungs: list[Rung], depths: tuple[float, ...]) -> str:
    """The table the run exists to produce, one line per rung."""
    header = (
        f"{'backend':22s} {'context':>8s} {'prefill':>14s} {'retrieved':>10s} {'clean':>10s} {'ttft ms':>10s} "
        f"{'p50 us':>9s} {'p99 us':>9s} {'kv MB':>9s} {'dense MB':>9s} {'peak MB':>9s}"
    )
    lines = [header, "-" * len(header)]
    for rung in rungs:
        lines.append(
            f"{rung.backend:22s} {rung.context:>8d} {rung.prefill_mode:>14s} "
            f"{f'{rung.hits}/{len(depths)}':>10s} {f'{rung.clean}/{len(depths)}':>10s} "
            f"{rung.median('ttft_ms'):>10.0f} "
            f"{rung.median('decode_p50_us'):>9.0f} {rung.median('decode_p99_us'):>9.0f} "
            f"{rung.median('kv_cache_mb'):>9.1f} {rung.median('dense_equivalent_mb'):>9.1f} "
            f"{rung.peak('device_peak_mb'):>9.0f}"
        )
    lines.append("")
    lines.append("retrieved: the passcode appears in the continuation as a whole token, wherever in it.")
    lines.append("clean: it also led with it and did not loop. retrieved > clean means the model rambled after")
    lines.append("answering -- read the per-item verdicts (found+late, found+loop) and the predictions.")
    lines.append("ttft/p50/p99 are medians over the depths; peak MB is the allocator's high-water mark over them.")
    lines.append("The first decode step of a sequence pays first-touch costs no later step does: it lands in p99.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device = resolve_device(args.device)
    backends = parse_backends(args.backends)
    contexts = parse_int_list(args.contexts, "contexts")
    depths = parse_depths(args.depths)
    config = read_config(args.model_path)
    family = family_for(config)

    tokenizer = load_tokenizer(args.model_path, family)
    eos_ids = eos_ids_for(config, tokenizer, family)
    longest = max(contexts)
    print(f"prompt: {prompt_route(tokenizer, not args.raw_prompt, family)}", file=sys.stderr)
    print(
        f"stop ids: {sorted(eos_ids) if eos_ids else 'NONE -- every continuation will run its whole budget'}",
        file=sys.stderr,
    )

    sink = args.out_file.open("w", encoding="utf-8") if args.out_file else sys.stdout
    rungs: list[Rung] = []
    try:
        for backend in backends:
            print(f"\n=== {backend} ===", file=sys.stderr)
            # The plan is per backend, and is settled at the longest rung: a
            # prefill path refused there is refused everywhere below it too.
            plan = plan_for_backend(args, backend, contexts, family)
            if plan.results or plan.notes:
                print(plan.report(), file=sys.stderr)
            runner = None
            shared_mode = rung_prefill_mode(args, plan, backend, longest, family)
            if args.reuse_runner:
                runner = build_runner(args, backend, shared_mode, cache_tokens(args, longest), plan)
            for context in contexts:
                mode = shared_mode if args.reuse_runner else rung_prefill_mode(args, plan, backend, context, family)
                max_seq_len = cache_tokens(args, longest if args.reuse_runner else context)
                print(f"  context {context} ({mode}, cache {max_seq_len})", file=sys.stderr)
                if not args.reuse_runner:
                    runner = None
                    free_device(args.device)
                    runner = build_runner(args, backend, mode, max_seq_len, plan)
                keep = max_seq_len - CONTEXT_HEADROOM_TOKENS - args.max_new_tokens
                rungs.append(
                    run_rung(args, runner, tokenizer, eos_ids, backend, context, mode, depths, keep, sink, family)
                )
            runner = None
            free_device(args.device)
    finally:
        if args.out_file:
            sink.close()

    print()
    print(summarise(rungs, depths))
    return 0


def cache_tokens(args: argparse.Namespace, context: int) -> int:
    """The static cache a rung needs: the prompt, its continuation and room to over-tokenise into."""
    return context + args.max_new_tokens + CONTEXT_HEADROOM_TOKENS


def plan_for_backend(
    args: argparse.Namespace, backend: str, contexts: tuple[int, ...], family: ModelFamily
) -> AttentionPlan:
    """Pre-flight this backend once, covering every prefill path the ladder will take.

    Probed as ``dense_staging`` when *any* rung would stage, not when the longest
    one would: GLM-4 stages at 16k and refuses to at 64k, and a plan settled at 64k
    alone would never learn whether the unquantised pool runs here at all.
    """
    modes = {prefill_mode_for(args, backend, context, family) for context in contexts}
    mode = "dense_staging" if "dense_staging" in modes else "batched_decode"
    if args.skip_preflight:
        return AttentionPlan(mode)
    try:
        return plan_attention(plan_namespace(args, backend), read_checkpoint_shape(args.model_path), mode)
    except PreflightFailed as failure:
        raise SystemExit(f"{backend}: {failure}") from failure


def rung_prefill_mode(
    args: argparse.Namespace, plan: AttentionPlan, backend: str, context: int, family: ModelFamily
) -> str:
    """This rung's own policy, unless the pre-flight found no unquantised pool runs here.

    The pre-flight's fallback is the run's, not one rung's: a staging pool that
    could not be built on the synthetic layer cannot be built at 4k either.
    """
    mode = prefill_mode_for(args, backend, context, family)
    if mode == "dense_staging" and backend not in DENSE_BACKENDS and plan.prefill_mode == "batched_decode":
        return "batched_decode"
    return mode


if __name__ == "__main__":
    raise SystemExit(main())
