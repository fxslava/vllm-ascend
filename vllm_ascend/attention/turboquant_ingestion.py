# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan strip writes and conflict-free scatter launches on the host."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_STRIP_TOKENS,
)


@dataclass(frozen=True)
class TurboQuantStripRun:
    start: int
    stop: int
    pad: int


@dataclass(frozen=True)
class TurboQuantStripPlan:
    runs: tuple[TurboQuantStripRun, ...]
    waves: tuple[tuple[int, ...], ...]
    num_tokens: int

    @property
    def is_whole_batch(self) -> bool:
        return not self.waves and self.runs == (TurboQuantStripRun(0, self.num_tokens, 0),)

    @property
    def num_launches(self) -> int:
        return len(self.runs) + len(self.waves)


def turboquant_plan_strip_ingestion(
    slots: Sequence[int], block_size: int, strip: int = TURBOQUANT_STRIP_TOKENS
) -> TurboQuantStripPlan:
    num_tokens = len(slots)
    runs: list[TurboQuantStripRun] = []
    scatter: list[int] = []
    written: set[int] | None = None

    def run_length(start: int, limit: int) -> int:
        if start >= num_tokens or slots[start] < 0:
            return 0
        length = 1
        while length < limit and start + length < num_tokens and slots[start + length] == slots[start] + length:
            length += 1
        return length

    def aligned(start: int) -> bool:
        return start < num_tokens and slots[start] >= 0 and slots[start] % strip == 0

    def whole_strip(start: int) -> bool:
        return aligned(start) and run_length(start, strip) == strip

    def pad_for_partial_strip(start: int) -> int:
        if not aligned(start):
            return 0
        tail = run_length(start, strip)
        if not 0 < tail < strip:
            return 0
        pad = strip - tail
        last = slots[start] + tail - 1
        if block_size <= 0 or (last + pad) // block_size != last // block_size:
            return 0
        nonlocal written
        if written is None:
            written = {slot for slot in slots if slot >= 0}
        if any(slots[start] + offset in written for offset in range(tail, strip)):
            return 0
        return pad

    index = 0
    while index < num_tokens:
        if whole_strip(index):
            stop = index + strip
            while whole_strip(stop):
                stop += strip
            runs.append(TurboQuantStripRun(index, stop, 0))
            index = stop
            continue
        pad = pad_for_partial_strip(index)
        if pad:
            stop = index + (strip - pad)
            runs.append(TurboQuantStripRun(index, stop, pad))
            index = stop
            continue
        scatter.append(index)
        index += 1

    return TurboQuantStripPlan(
        runs=tuple(runs),
        waves=_strip_scatter_waves(slots, scatter, strip),
        num_tokens=num_tokens,
    )


def _strip_scatter_waves(slots: Sequence[int], tokens: Sequence[int], strip: int) -> tuple[tuple[int, ...], ...]:
    waves: list[list[int]] = []
    depth: dict[int, int] = {}
    for token in tokens:
        slot = slots[token]
        if slot < 0:
            wave = 0
        else:
            wave = depth.get(slot // strip, 0)
            depth[slot // strip] = wave + 1
        while len(waves) <= wave:
            waves.append([])
        waves[wave].append(token)
    return tuple(tuple(wave) for wave in waves)


def _strip_run_rows(activations: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    window = activations[run.start : run.stop]
    if not run.pad:
        return window
    return torch.cat((window, window[-1:].expand(run.pad, *window.shape[1:]))).contiguous()


def _strip_run_slots(slot_mapping: torch.Tensor, run: TurboQuantStripRun) -> torch.Tensor:
    window = slot_mapping[run.start : run.stop]
    if not run.pad:
        return window
    pad = torch.arange(1, run.pad + 1, dtype=window.dtype, device=window.device) + window[-1]
    return torch.cat((window, pad)).contiguous()
