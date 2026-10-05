"""Persistent-pool cold/warm generation with per-token offload accounting."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

from ..inference.dsv2_lite import V2LiteDecoder


class CacheTrace:
    """Measure synchronous fill service and record layer-qualified expert keys."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.fill = decoder.source.fill_slot_params
        self.acquire = decoder.pool.acquire_for_step
        self.requests = []
        self.fill_seconds = 0.0
        self.acquisition_seconds = 0.0
        self.read_request_bytes = 0
        self.read_seconds = 0.0

    def counted_read(self, original):
        def read(chunk, offset, length):
            started = time.perf_counter()
            original(chunk, offset, length)
            self.read_seconds += time.perf_counter() - started
            self.read_request_bytes += length

        return read

    def timed_fill(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.fill(*args, **kwargs)
        self.fill_seconds += time.perf_counter() - started
        return result

    def traced_acquire(self, layer, experts, provider):
        self.requests.extend((layer, int(expert)) for expert in experts)
        self.decoder.runtime.synchronize_device()
        started = time.perf_counter()
        result = self.acquire(layer, experts, provider)
        self.decoder.runtime.synchronize_device()
        self.acquisition_seconds += time.perf_counter() - started
        return result

    def io_bytes(self):
        return sum(loader.bytes_read for loader in self.decoder.source._loaders.values())

    def step(self, token):
        decoder = self.decoder
        stats = decoder.pool.stats
        before = (
            stats.loads,
            stats.hits,
            stats.evictions,
            decoder.source.bytes_streamed,
            self.io_bytes(),
            self.read_request_bytes,
        )
        self.requests = []
        read_seconds_before = self.read_seconds
        self.fill_seconds = self.acquisition_seconds = 0.0
        decoder.runtime.synchronize_device()
        started = time.perf_counter()
        logits = decoder.step(token)
        decoder.runtime.synchronize_device()
        elapsed = time.perf_counter() - started
        after = (
            stats.loads,
            stats.hits,
            stats.evictions,
            decoder.source.bytes_streamed,
            self.io_bytes(),
            self.read_request_bytes,
        )
        delta = [right - left for left, right in zip(before, after)]
        row = dict(
            zip(("misses", "hits", "evictions", "payload_bytes", "file_read_bytes", "read_request_bytes"), delta)
        )
        row.update(
            token_id=token,
            seconds=elapsed,
            load_seconds=self.fill_seconds,
            acquisition_seconds=self.acquisition_seconds,
            pool_overhead_seconds=self.acquisition_seconds - self.fill_seconds,
            execution_and_host_seconds=elapsed - self.acquisition_seconds,
            hit_rate=delta[1] / (delta[0] + delta[1]),
            read_service_seconds=self.read_seconds - read_seconds_before,
            experts=[list(key) for key in self.requests],
            resident_count=decoder.pool.resident_expert_count,
            allocated_bytes=decoder.runtime.memory_allocated(),
        )
        return logits, row


def summarize(rows):
    decode = [row for row in rows if row["phase"] == "decode"]
    return {
        "seconds": sum(row["seconds"] for row in rows),
        "mean_decode_seconds": statistics.mean(row["seconds"] for row in decode),
        "load_seconds": sum(row["load_seconds"] for row in rows),
        "pool_overhead_seconds": sum(row["pool_overhead_seconds"] for row in rows),
        "execution_and_host_seconds": sum(row["execution_and_host_seconds"] for row in rows),
        "misses": sum(row["misses"] for row in rows),
        "hits": sum(row["hits"] for row in rows),
        "evictions": sum(row["evictions"] for row in rows),
        "payload_bytes": sum(row["payload_bytes"] for row in rows),
        "file_read_bytes": sum(row["file_read_bytes"] for row in rows),
        "read_request_bytes": sum(row["read_request_bytes"] for row in rows),
        "read_service_seconds": sum(row["read_service_seconds"] for row in rows),
    }


def benchmark(directory, slots, prompt, tokenizer, new_tokens):
    decoder = V2LiteDecoder(directory, "cuda:0", slots, len(prompt) + new_tokens, storage="disk")
    trace = CacheTrace(decoder)
    cold_allocated = decoder.runtime.memory_allocated()
    runs, frequency = [], Counter()
    union = set()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(decoder.source, "fill_slot_params", trace.timed_fill))
            stack.enter_context(patch.object(decoder.pool, "acquire_for_step", trace.traced_acquire))
            for loader in decoder.source._loaders.values():
                reader = loader._reader
                for name in ("read_into", "read_tail_into"):
                    stack.enter_context(patch.object(reader, name, trace.counted_read(getattr(reader, name))))
            for run in range(3):
                decoder.reset_sequence()
                rows, generated = [], []
                active = set()
                for token in prompt:
                    logits, row = trace.step(token)
                    row["phase"] = "prefill"
                    rows.append(row)
                for position in range(new_tokens):
                    token = int(logits.argmax(-1).item())
                    generated.append(token)
                    if token == decoder.config["eos_token_id"] or position == new_tokens - 1:
                        break
                    logits, row = trace.step(token)
                    row["phase"] = "decode"
                    rows.append(row)
                for row in rows:
                    keys = [tuple(key) for key in row["experts"]]
                    frequency.update(keys)
                    active.update(keys)
                    union.update(keys)
                    row["cumulative_unique_experts"] = len(active)
                result = {
                    "run": run + 1,
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(generated),
                    "unique_experts": len(active),
                    "per_layer_unique": dict(sorted(Counter(layer for layer, _ in active).items())),
                    "summary": summarize(rows),
                    "tokens": rows,
                }
                runs.append(result)
                print(f"slots={slots} run={run + 1} unique={len(active)} {result['summary']}", flush=True)
        return {
            "slots": slots,
            "runs": runs,
            "union_unique_experts": len(union),
            "cold_allocated_bytes": cold_allocated,
            "populated_allocated_bytes": decoder.runtime.memory_allocated(),
            "arena_bytes": decoder.pool.slot_arena.numel(),
            "populated_weight_bytes": decoder.pool.resident_expert_count * decoder.pool.layout.slot_num_bytes,
            "final_resident_count": decoder.pool.resident_expert_count,
            "arena_pointers_stable": decoder.pointers() == decoder.fingerprint,
            "io_backends": decoder.source.backends,
            "expert_frequencies": [[layer, expert, count] for (layer, expert), count in frequency.most_common()],
        }
    finally:
        decoder.close()
        runtime = decoder.runtime
        del decoder, trace
        gc.collect()
        runtime.release_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, nargs="+", default=[64, 156, 384])
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--raw-prompt", action="store_true", help="tokenize plain text without the chat template")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least two")
    tokenizer = AutoTokenizer.from_pretrained(args.weights_dir, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True
    )
    if hasattr(prompt, "keys"):
        prompt = prompt["input_ids"]
    if args.raw_prompt:
        prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt:
        parser.error("prompt must contain tokens")
    report = {
        "prompt_ids": prompt,
        "device": "RTX 5070 / cuda:0",
        "configurations": [],
        "timing": "synchronized wall time; load=file read+H2D; execution includes host launch overhead",
        "cold": "empty routed VRAM pool; shared/backbone resident; filesystem cache not flushed",
    }
    for slots in args.slots:
        report["configurations"].append(benchmark(args.weights_dir, slots, prompt, tokenizer, args.new_tokens))
        args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
