"""Three generations from the full RAM/VRAM partition, with disk guards."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

from ..inference.dsv2_lite import V2LiteDecoder


def benchmark(directory, slots, tokens, tokenizer, new_tokens):
    started = time.perf_counter()
    decoder = V2LiteDecoder(directory, "cuda:0", slots, len(tokens) + new_tokens, storage="exclusive")
    startup_seconds = time.perf_counter() - started
    provider = decoder.source
    disk = provider._disk_source
    read_bytes_at_boot = sum(loader.bytes_read for loader in disk._loaders.values())
    guard_checks = 0
    # Deliberate shutdown probes must fail before any reader reopens a file.
    for action in (
        lambda: disk.read_param(1, 0, "w1"),
        lambda: next(iter(disk._loaders.values()))._reader.read_tail_into(provider.host_scratch, 0, 4),
    ):
        try:
            action()
        except RuntimeError:
            guard_checks += 1
        else:
            raise RuntimeError("shutdown disk guard failed")
    attempts_at_boot = disk.closed_read_attempts
    counters = ("swap_seconds", "h2d_seconds", "d2h_seconds", "duplex_seconds", "h2d_bytes", "d2h_bytes", "swaps")
    rows_by_run = []
    reader_requests = 0

    def forbid_read(*args, **kwargs):
        nonlocal reader_requests
        reader_requests += 1
        raise RuntimeError("generation attempted a forbidden disk request")

    def step(token):
        before = [getattr(provider, name) for name in counters]
        hits, misses = decoder.pool.stats.hits, decoder.pool.stats.loads
        decoder.runtime.synchronize_device()
        began = time.perf_counter()
        logits = decoder.step(token)
        decoder.runtime.synchronize_device()
        elapsed = time.perf_counter() - began
        provider.validate_residency(decoder.pool)
        delta = [getattr(provider, name) - value for name, value in zip(counters, before)]
        row = dict(zip(counters, delta))
        row.update(
            seconds=elapsed,
            token=token,
            hits=decoder.pool.stats.hits - hits,
            misses=decoder.pool.stats.loads - misses,
            disk_read_bytes=sum(loader.bytes_read for loader in disk._loaders.values()) - read_bytes_at_boot,
        )
        return logits, row

    try:
        with ExitStack() as stack:
            for loader in disk._loaders.values():
                for name in ("read_into", "read_tail_into", "mmap_view"):
                    stack.enter_context(patch.object(loader._reader, name, forbid_read))
            for run in range(3):
                decoder.reset_sequence()
                rows, generated = [], []
                for token in tokens:
                    logits, row = step(token)
                    row["phase"] = "prefill"
                    rows.append(row)
                for position in range(new_tokens):
                    token = int(logits.argmax(-1).item())
                    generated.append(token)
                    if token == decoder.config["eos_token_id"] or position == new_tokens - 1:
                        break
                    logits, row = step(token)
                    row["phase"] = "decode"
                    rows.append(row)
                decoded = [row for row in rows if row["phase"] == "decode"]
                summary = {name: sum(row[name] for row in rows) for name in (*counters, "seconds", "hits", "misses")}
                summary["mean_decode_seconds"] = statistics.mean(row["seconds"] for row in decoded)
                summary["mean_decode_swap_seconds"] = statistics.mean(row["swap_seconds"] for row in decoded)
                summary["mean_decode_h2d_seconds"] = statistics.mean(row["h2d_seconds"] for row in decoded)
                summary["mean_decode_d2h_seconds"] = statistics.mean(row["d2h_seconds"] for row in decoded)
                summary["mean_decode_duplex_seconds"] = statistics.mean(row["duplex_seconds"] for row in decoded)
                result = {
                    "run": run + 1,
                    "summary": summary,
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(generated),
                    "tokens": rows,
                }
                rows_by_run.append(result)
                print(f"slots={slots} run={run + 1}: {summary}", flush=True)
        descriptors_closed = all(
            all(
                getattr(loader._reader, field) is None
                for field in ("_fd", "_win_handle", "_tail_fd", "_tail_handle", "_mmap_fd")
            )
            for loader in disk._loaders.values()
        )
        result = {
            "slots": slots,
            "host_slots": provider.host_slots,
            "host_bytes": provider.host_bytes,
            "host_pinned": all(block.is_pinned() for block in provider.host_arenas),
            "startup_seconds": startup_seconds,
            "startup_routed_bytes": provider.startup_bytes,
            "disk_read_bytes": sum(loader.bytes_read for loader in disk._loaders.values()) - read_bytes_at_boot,
            "runtime_reader_requests": reader_requests,
            "runtime_source_read_attempts": disk.closed_read_attempts - attempts_at_boot,
            "disk_guard_probes_passed": guard_checks,
            "file_descriptors_closed": descriptors_closed,
            "partition_disjoint": not decoder.pool.resident_keys() & provider.host_residents.keys(),
            "partition_total_experts": len(decoder.pool.resident_keys() | provider.host_residents.keys()),
            "arena_pointers_stable": decoder.pointers() == decoder.fingerprint,
            "device_allocated_bytes": decoder.runtime.memory_allocated(),
            "runs": rows_by_run,
        }
        if not (
            result["disk_read_bytes"]
            == result["runtime_reader_requests"]
            == result["runtime_source_read_attempts"]
            == 0
            and result["partition_disjoint"]
            and descriptors_closed
            and result["host_pinned"]
            and result["arena_pointers_stable"]
        ):
            raise RuntimeError("exclusive hierarchy validation failed")
        return result
    finally:
        decoder.close()
        runtime = decoder.runtime
        decoder = provider = disk = None
        gc.collect()
        runtime.release_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("F:/AI/models/DeepSeek-V2-Lite-Chat"))
    parser.add_argument("--slots", type=int, nargs="+", default=[64, 156])
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least two")
    tokenizer = AutoTokenizer.from_pretrained(args.weights_dir, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True
    )
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    report = {
        "configurations": [],
        "prompt_ids": tokens,
        "timing": "CUDA event durations for PCIe directions; synchronized wall time for swap service",
    }
    for slots in args.slots:
        print(f"Bootstrapping all 1664 experts with {slots} VRAM slots...", flush=True)
        report["configurations"].append(benchmark(args.weights_dir, slots, tokens, tokenizer, args.new_tokens))
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
