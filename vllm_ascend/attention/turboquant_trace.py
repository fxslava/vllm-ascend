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
"""Direct-to-disk capture of what the TurboQuant operators are actually handed.

An Ascend launch is asynchronous, so a kernel that traps does not fail its own
launch: the runtime raises ``507035`` / ``264`` from some *later* API call, names
whichever operator the queue was working on, and takes the engine down with it.
By the time that happens the arguments that caused it are gone, and in a
multi-process engine the worker's ``logger`` and ``print`` output is redirected
somewhere that may never be flushed.  Both of those are why this module exists
and why it does not use ``vllm.logger``:

* every record is appended to its own file, opened line-buffered, so the last
  record before an abort is already in the file rather than in a buffer the
  abort discards;
* ``TURBOQUANT_DRY_RUN=1`` stops the kernels being launched at all, so a run
  walks through prefill and every decode step of every layer and records the
  real production shapes instead of stopping at the first faulting launch.

The tracer is a process-wide singleton because there is nowhere to thread one
through: ``AttentionImpl.forward`` is an upstream signature, and a worker process
must not open one file handle per layer.  It holds a file handle and a sequence
counter and nothing else, and every entry point is guarded so a diagnostic can
never become the failure it exists to describe -- a file that cannot be opened
falls back to stderr, and a tensor that cannot be described is recorded as the
error rather than raised.

This module deliberately imports neither vLLM nor ``torch_npu``: the same records
are what ``tools/tq_longbench`` needs when it drives these operators with no
engine around them.  See :mod:`vllm_ascend.attention.turboquant_v1` for the call
sites.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from typing import Any

import torch

import vllm_ascend.envs as envs_ascend

# The alignments worth knowing about when an operand faults. 32 is the global-memory
# burst every copy these kernels make is cut into (TURBOQUANT_GM_BURST_BYTES); 64 and
# 512 are the wider boundaries the DMA engine and the L2 cache line prefer, and an
# operand that clears 32 but not 512 is the signature of a slice taken part-way into a
# larger allocation.
TURBOQUANT_TRACE_ALIGNMENTS = (32, 64, 512)

# How many leading slot / block ids each record carries. Enough to recognise the
# batch's shape without turning one record into a page of numbers.
TURBOQUANT_TRACE_PREVIEW = 10

TURBOQUANT_TRACE_DEFAULT_PATH = "/workspace/turboquant_trace.log"


def describe_tensor(tensor: Any) -> dict[str, Any] | None:
    """Everything about ``tensor`` an operand fault can be explained by.

    The pointer and its remainders are the point: ``.contiguous()`` returns a
    contiguous view unchanged, storage offset and all, so an operand can arrive
    contiguous and still start part-way through a burst -- which the AI core
    reports as "the address for scalar to access GM is invalid" rather than as a
    misalignment.  ``None`` is a value the operators accept (an absent output
    gate), so it is recorded as ``None`` rather than skipped.
    """
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return {"not_a_tensor": repr(tensor)[:256]}
    try:
        pointer = tensor.data_ptr()
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": bool(tensor.is_contiguous()),
            "numel": int(tensor.numel()),
            "data_ptr": hex(pointer),
            "align": {str(boundary): pointer % boundary for boundary in TURBOQUANT_TRACE_ALIGNMENTS},
        }
    except Exception as error:  # noqa: BLE001 - a description must never raise
        return {"describe_failed": repr(error)}


def describe_tensors(**named: Any) -> dict[str, Any]:
    """:func:`describe_tensor` over a whole operand list, keyed by parameter name."""
    return {name: describe_tensor(tensor) for name, tensor in named.items()}


def describe_indices(indices: Any, *, capacity: int | None = None) -> dict[str, Any] | None:
    """Summarise an index tensor -- ``slot_mapping``, ``block_tables``, ``seq_lens``.

    Costs one device-to-host copy of the tensor, which is a synchronisation, and
    is why nothing here runs unless ``TURBOQUANT_CAPTURE_SIGNATURES=1``.  One
    copy rather than a handful of ``.item()`` calls: the reductions run on the
    host side of the single transfer.

    ``capacity`` is the number of rows the kernel can address -- ``num_blocks *
    block_size`` for a slot mapping, ``num_blocks`` for a block table.  Both ends
    matter: the writers silently drop a negative slot (vLLM's padding marker) and
    one past the last row, so a count of each is the difference between "those
    tokens are in the cache" and "those tokens are not".
    """
    if indices is None:
        return None
    if not isinstance(indices, torch.Tensor):
        return {"not_a_tensor": repr(indices)[:256]}
    try:
        flat = indices.detach().reshape(-1)
        summary: dict[str, Any] = {"count": int(flat.numel()), "capacity": capacity}
        if flat.numel() == 0:
            return summary
        host = flat.to("cpu", torch.int64)
        summary["min"] = int(host.min())
        summary["max"] = int(host.max())
        summary["negative"] = int((host < 0).sum())
        summary["first"] = host[:TURBOQUANT_TRACE_PREVIEW].tolist()
        if capacity is not None:
            summary["past_capacity"] = int((host >= capacity).sum())
        return summary
    except Exception as error:  # noqa: BLE001 - a description must never raise
        return {"describe_failed": repr(error)}


class TurboQuantTracer:
    """Appends one JSON record per line to the trace file.

    ``capture`` and ``dry_run`` are independent: capture without dry-run records
    the launches a live run makes, dry-run without capture walks the engine past
    the kernels without writing anything.  :attr:`active` is what the call sites
    branch on, so a disabled tracer costs one attribute read.
    """

    def __init__(self, *, capture: bool, dry_run: bool, path: str) -> None:
        self.capture = capture
        self.dry_run = dry_run
        self.path = path
        self._file: Any = None
        self._owns_file = False
        self._sequence = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> TurboQuantTracer:
        return cls(
            capture=envs_ascend.TURBOQUANT_CAPTURE_SIGNATURES,
            dry_run=envs_ascend.TURBOQUANT_DRY_RUN,
            path=envs_ascend.TURBOQUANT_TRACE_PATH or TURBOQUANT_TRACE_DEFAULT_PATH,
        )

    @property
    def active(self) -> bool:
        """Whether any call site has work to do -- record it, bypass it, or both."""
        return self.capture or self.dry_run

    def record(self, event: str, **fields: Any) -> None:
        """Append one record. A no-op unless ``TURBOQUANT_CAPTURE_SIGNATURES=1``."""
        if not self.capture:
            return
        try:
            with self._lock:
                self._sequence += 1
                payload: dict[str, Any] = {
                    "seq": self._sequence,
                    "time": round(time.time(), 6),
                    "pid": os.getpid(),
                    "event": event,
                }
                payload.update(fields)
                line = json.dumps(payload, default=repr)
                handle = self._handle()
                if handle is not None:
                    handle.write(line + "\n")
        except Exception as error:  # noqa: BLE001 - the trace must never fail the run
            print(f"[vllm-ascend/turboquant] trace record dropped: {error!r}", file=sys.stderr, flush=True)

    def close(self) -> None:
        """Release the file handle. Tests and a clean worker shutdown; not the hot path."""
        with self._lock:
            if self._file is not None and self._owns_file:
                with contextlib.suppress(OSError):
                    self._file.close()
            self._file = None
            self._owns_file = False

    def _handle(self) -> Any:
        """Open the trace file once, line-buffered, and keep it open.

        ``buffering=1`` is the whole point: a record reaches the OS on its own
        newline, so the arguments of the launch that aborted the process are on
        disk even though nothing ever flushed or closed the file.  A path that
        cannot be opened -- a read-only container, a missing directory -- falls
        back to stderr rather than losing the capture.
        """
        if self._file is not None:
            return self._file
        try:
            self._file = open(self.path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
            self._owns_file = True
        except OSError as error:
            print(
                f"[vllm-ascend/turboquant] cannot open {self.path} ({error!r}); tracing to stderr instead",
                file=sys.stderr,
                flush=True,
            )
            self._file = sys.stderr
            self._owns_file = False
        return self._file


_TRACER: TurboQuantTracer | None = None
_TRACER_LOCK = threading.Lock()


def turboquant_tracer() -> TurboQuantTracer:
    """The process's tracer, built from the environment on first use.

    Built once rather than per call so the hot path never re-reads the
    environment, and so every layer of every step appends to one file in one
    order.
    """
    global _TRACER
    if _TRACER is None:
        with _TRACER_LOCK:
            if _TRACER is None:
                _TRACER = TurboQuantTracer.from_env()
    return _TRACER


def reset_turboquant_tracer() -> None:
    """Drop the cached tracer so the next call re-reads the environment.

    For tests, which need to toggle the variables inside one process.
    """
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None:
            _TRACER.close()
        _TRACER = None
