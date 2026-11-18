# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TurboQuant tiling build/upload latency with and without a stream wait.

Run on an ascend950 build:
    python benchmarks/scripts/benchmark_turboquant_tiling.py --batch-sizes 1 16 128
    python benchmarks/scripts/benchmark_turboquant_tiling.py --queued-matmuls 4
"""

import argparse
import json
import statistics
import time

import torch

from vllm_ascend.attention.turboquant_v1 import load_turboquant_extension, turboquant_tiling_upper_bound_words


def measure(tiling_op, batch_size, iterations, queued_matmuls, asynchronous):
    words = turboquant_tiling_upper_bound_words(batch_size, 8)
    context_lengths = torch.full((batch_size,), 4096, dtype=torch.int32)
    host_tiling = torch.empty(words, dtype=torch.int32)
    device_tiling = torch.empty(words, dtype=torch.int32, device="npu")
    matrix = torch.ones((1024, 1024), dtype=torch.bfloat16, device="npu")
    product = torch.empty_like(matrix)

    def launch():
        if asynchronous:
            tiling_op(context_lengths, device_tiling, 24, 4, 256, 128, 64, 8, 6)
        else:
            tiling_op(context_lengths, host_tiling, 24, 4, 256, 128, 64, 8, 6)
            device_tiling.copy_(host_tiling, non_blocking=False)

    for _ in range(10):
        launch()
    torch.npu.synchronize()
    submission_times = []
    started = time.perf_counter_ns()
    for _ in range(iterations):
        for _ in range(queued_matmuls):
            torch.mm(matrix, matrix, out=product)
        submitted = time.perf_counter_ns()
        launch()
        submission_times.append((time.perf_counter_ns() - submitted) / 1000)
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter_ns() - started) / 1000
    return {
        "mode": "pinned_async" if asynchronous else "pageable_blocking",
        "batch_size": batch_size,
        "buffer_bytes_upper_bound": words * host_tiling.element_size(),
        "queued_matmuls_per_step": queued_matmuls,
        "median_host_submission_us": statistics.median(submission_times),
        "mean_step_us_including_drain": elapsed_us / iterations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--queued-matmuls", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1 or args.queued_matmuls < 0 or min(args.batch_sizes) < 1:
        parser.error("batch sizes and iterations must be positive; queued matmuls must be nonnegative")
    load_turboquant_extension()
    tiling_op = torch.ops.turboquant_ops.npu_turboquant_build_decode_tiling
    for batch_size in args.batch_sizes:
        for asynchronous in (False, True):
            print(json.dumps(measure(tiling_op, batch_size, args.iterations, args.queued_matmuls, asynchronous)))


if __name__ == "__main__":
    main()
