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

import atexit
import contextlib
import json
import os
import sys
import threading
import time
from collections import deque
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

# The flight recorder's default depth. A decode step touches one operator per layer, so 128
# covers roughly the last three steps of a 40-layer model -- enough to see what the shapes were
# doing on the way into a failure without holding the whole run.
TURBOQUANT_RING_CAPACITY = 128

# Written beside the trace log, so one directory holds everything a failing run produced.
TURBOQUANT_CRASH_FILENAME = "turboquant_last_crash.jsonl"
TURBOQUANT_CASES_FILENAME = "turboquant_unique_cases.jsonl"


def snapshot_tensor(tensor: Any) -> tuple | None:
    """What the ring actually stores: five attribute reads, no formatting, no allocation.

    ``shape`` and ``stride()`` are already tuples and ``dtype``/``device`` are interned
    objects, so holding them costs nothing and pins nothing -- the tensor itself is
    deliberately *not* retained, or the ring would keep 128 dispatches of activations alive.

    Formatting is where the time actually goes: building the strings and lists that
    :func:`render_tensor` produces costs more than the rest of the dispatch path put
    together, so it happens once, when the ring is written out, rather than 128 times over
    for entries that are about to be overwritten.
    """
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return (repr(tensor)[:128],)
    try:
        return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device, tensor.data_ptr())
    except Exception as error:  # noqa: BLE001 - a description must never raise
        return (repr(error),)


def render_tensor(snap: tuple | None) -> dict[str, Any] | None:
    """Turn a :func:`snapshot_tensor` tuple into the JSON a dump or a case file carries."""
    if snap is None:
        return None
    if len(snap) == 1:
        return {"not_a_tensor": snap[0]}
    shape, stride, dtype, device, pointer = snap
    name = str(dtype)
    return {
        "shape": list(shape),
        "stride": list(stride),
        "dtype": name[6:] if name.startswith("torch.") else name,
        "device": str(device),
        "a32": pointer % 32,
        "a512": pointer % 512,
    }


def render_entry(entry: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A ring entry as JSON: the snapshots rendered, everything else passed through."""
    out = {k: v for k, v in entry.items() if k != "_snaps"}
    out["t"] = round(out.get("t", 0.0), 6)
    out["tensors"] = {name: render_tensor(snap) for name, snap in entry.get("_snaps", {}).items()}
    out.update(extra)
    return out


def compact_tensor(tensor: Any) -> dict[str, Any] | None:
    """:func:`snapshot_tensor` then :func:`render_tensor`, for a caller that wants both."""
    return render_tensor(snapshot_tensor(tensor))


def host_seq_lengths(seq_lens: Any) -> list[int] | None:
    """The sequence lengths, but only when reading them is free.

    ``AscendMetadata.seq_lens`` reaches the backend on the *host* (see
    ``AscendTurboQuantAttentionBackendImpl._device_index``), so for the path that matters
    these are an ordinary memory read. If a caller hands over a device tensor instead, the
    values are skipped rather than paid for: a synchronisation per dispatch is exactly what
    the ring exists not to cost.
    """
    if not isinstance(seq_lens, torch.Tensor) or seq_lens.device.type != "cpu":
        return None
    try:
        return seq_lens.detach().reshape(-1).tolist()
    except Exception:  # noqa: BLE001 - a description must never raise
        return None


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


def cache_planes(kv_cache: Any) -> tuple:
    """The KV cache as a tuple of planes, whatever the caller actually passed.

    A served step hands the backend a tuple of planes, but the profile run that
    sizes the cache (``determine_available_memory`` -> ``profile_run`` ->
    ``_dummy_run``) hands it a bare empty ``torch.Tensor`` instead.  ``bool()`` on
    that raises "Boolean value of Tensor with no values is ambiguous" and
    ``len()`` on a 0-d one raises as well, so neither ``kv_cache or ()`` nor
    ``len(kv_cache)`` is a question a tensor can answer.  Asking what it is first
    is: a tensor is one plane, anything iterable is its own planes, and anything
    else is described as the single object it is rather than refused.
    """
    if kv_cache is None:
        return ()
    if isinstance(kv_cache, torch.Tensor):
        return (kv_cache,)
    try:
        return tuple(kv_cache)
    except TypeError:
        return (kv_cache,)


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

    def __init__(
        self,
        *,
        capture: bool,
        dry_run: bool,
        path: str,
        ring_capacity: int = 0,
        record_cases: bool = False,
    ) -> None:
        self.capture = capture
        self.dry_run = dry_run
        self.path = path
        self.record_cases = record_cases
        self._file: Any = None
        self._owns_file = False
        self._sequence = 0
        self._lock = threading.Lock()
        # The flight recorder. A deque with maxlen drops from the far end on append, so the
        # ring costs one append per dispatch and never grows.
        self._ring: deque | None = deque(maxlen=ring_capacity) if ring_capacity > 0 else None
        self._seen_cases: set = set()
        self._cases_file: Any = None
        self._hooks_installed = False
        self._dumped = False
        self._own_excepthook: Any = None
        self._previous_excepthook: Any = None

    @classmethod
    def from_env(cls) -> TurboQuantTracer:
        ring = envs_ascend.TURBOQUANT_FLIGHT_RECORDER
        # "1" means on at the default depth; any larger number is the depth itself.
        capacity = TURBOQUANT_RING_CAPACITY if ring == 1 else max(ring, 0)
        return cls(
            capture=envs_ascend.TURBOQUANT_CAPTURE_SIGNATURES,
            dry_run=envs_ascend.TURBOQUANT_DRY_RUN,
            path=envs_ascend.TURBOQUANT_TRACE_PATH or TURBOQUANT_TRACE_DEFAULT_PATH,
            ring_capacity=capacity,
            record_cases=envs_ascend.TURBOQUANT_RECORD_CASES,
        )

    @property
    def active(self) -> bool:
        """Whether any call site has work to do -- observe it, record it, bypass it.

        One attribute read is what a disabled tracer costs a dispatch, which is why every
        mode hangs off this single object rather than being checked separately.
        """
        return self.capture or self.dry_run or self._ring is not None or self.record_cases

    @property
    def recording(self) -> bool:
        return self._ring is not None

    def sibling_path(self, filename: str) -> str:
        """A file beside the trace log, so one directory holds a failing run's whole output."""
        return os.path.join(os.path.dirname(self.path) or ".", filename)

    def observe(
        self,
        op: str,
        *,
        tensors: dict[str, Any],
        block_size: int | None = None,
        num_tokens: int | None = None,
        seq_lens: Any = None,
        **fields: Any,
    ) -> None:
        """One dispatch, through whichever of the three recorders are enabled.

        The ring first and unconditionally-but-cheaply, then the case harvester (which only
        writes when the configuration is one it has not seen), and the full JSON record last
        because it is the only one that touches the disk on every call.
        """
        if not (self._ring is not None or self.record_cases):
            return
        try:
            lengths = host_seq_lengths(seq_lens)
            longest = max(lengths) if lengths else None
            # The ragged tail: the quantity that decides whether the decode's last tile is a
            # whole one. A kernel that indexes unified buffer by it faults on anything that is
            # not a multiple of 8 -- which is what AI core error 340 turned out to be.
            tail = None if longest is None or not block_size else longest % block_size
            snaps = {name: snapshot_tensor(t) for name, t in tensors.items()}
            entry = {
                "op": op,
                "t": time.time(),
                "num_tokens": num_tokens,
                "block_size": block_size,
                "seq_len": longest,
                "seq_len_mod_block": tail,
                "seq_lens": lengths[:TURBOQUANT_TRACE_PREVIEW] if lengths else None,
                "_snaps": snaps,
            }
            entry.update(fields)
            if self._ring is not None:
                self._ring.append(entry)
                self._install_hooks()
            if self.record_cases:
                self._harvest(op, entry, snaps, tail, num_tokens, fields)
        except Exception as error:  # noqa: BLE001 - the recorder must never fail the run
            print(f"[vllm-ascend/turboquant] flight record dropped: {error!r}", file=sys.stderr, flush=True)

    def _harvest(
        self,
        op: str,
        entry: dict[str, Any],
        shapes: dict[str, Any],
        tail: int | None,
        num_tokens: int | None,
        fields: dict[str, Any],
    ) -> None:
        """Append the configuration if it is one this process has not launched before.

        Keyed by the ragged tail rather than the sequence length, because the length itself
        takes a new value on every decode step while the tail is what the kernel's tiling
        actually turns on -- key on the length and the file grows without bound.
        """
        signature = (
            op,
            tail,
            num_tokens,
            fields.get("num_heads"),
            fields.get("num_kv_heads"),
            fields.get("head_dim"),
            tuple(tuple(s[0]) if s and len(s) == 5 else None for s in shapes.values()),
            tuple(tuple(s[1]) if s and len(s) == 5 else None for s in shapes.values()),
        )
        with self._lock:
            if signature in self._seen_cases:
                return
            self._seen_cases.add(signature)
            handle = self._cases_handle()
            if handle is None:
                return
            # One write of one line: O_APPEND makes a write of this size atomic, so a second
            # worker appending to the same file cannot interleave a partial record.
            handle.write(json.dumps(render_entry(entry, case=len(self._seen_cases)), default=repr) + "\n")

    def _cases_handle(self) -> Any:
        if self._cases_file is not None:
            return self._cases_file
        path = self.sibling_path(TURBOQUANT_CASES_FILENAME)
        try:
            self._cases_file = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        except OSError as error:
            print(f"[vllm-ascend/turboquant] cannot open {path} ({error!r})", file=sys.stderr, flush=True)
            self.record_cases = False
            return None
        return self._cases_file

    def _install_hooks(self) -> None:
        """Arrange for the ring to reach disk if the process leaves through Python.

        Deliberately not at import time: a process that never dispatches has nothing to dump,
        and a worker that never enables the recorder should not have its excepthook rewritten.

        The limitation is worth stating plainly -- an AI core fault aborts the process, and
        neither of these runs then. For that failure the line-buffered trace log
        (``TURBOQUANT_CAPTURE_SIGNATURES``) is what survives; the ring is for the Python-level
        exception, which is the case where the last few dispatches are the interesting part
        and nothing has been written down.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        atexit.register(self._dump_on_exit)
        previous = sys.excepthook
        self._previous_excepthook = previous

        def hook(exc_type, exc_value, exc_tb):
            self.dump_ring(reason=f"{getattr(exc_type, '__name__', exc_type)}: {exc_value}")
            previous(exc_type, exc_value, exc_tb)

        self._own_excepthook = hook
        sys.excepthook = hook

    def _uninstall_hooks(self) -> None:
        """Give the interpreter back what it had.

        Without this a tracer that is replaced leaves its atexit hook behind, and every
        replacement adds another: at shutdown they all fire, each trying to write a dump for a
        run that is long over. Harmless in a worker, which builds one tracer and keeps it, but
        it is the tests that build a new one per case -- and a diagnostic that litters the
        shutdown of the process it was meant to explain is not one worth keeping.
        """
        if not self._hooks_installed:
            return
        self._hooks_installed = False
        with contextlib.suppress(Exception):
            atexit.unregister(self._dump_on_exit)
        # Only if nothing chained after ours; otherwise the later hook owns the chain.
        if self._own_excepthook is not None and sys.excepthook is self._own_excepthook:
            sys.excepthook = self._previous_excepthook
        self._own_excepthook = None
        self._previous_excepthook = None

    def _dump_on_exit(self) -> None:
        self.dump_ring(reason="process exit")

    def dump_ring(self, *, reason: str, path: str | None = None) -> str | None:
        """Write the ring out as JSON lines. Returns the path, or None if there was nothing."""
        if self._ring is None:
            return None
        with self._lock:
            entries = list(self._ring)
            if not entries or self._dumped:
                return None
            self._dumped = True
        target = path or self.sibling_path(TURBOQUANT_CRASH_FILENAME)
        try:
            with open(target, "w", encoding="utf-8") as handle:
                header = {"reason": reason, "pid": os.getpid(), "entries": len(entries), "t": round(time.time(), 6)}
                handle.write(json.dumps(header, default=repr) + "\n")
                for entry in entries:
                    handle.write(json.dumps(render_entry(entry), default=repr) + "\n")
            print(
                f"[vllm-ascend/turboquant] wrote the last {len(entries)} dispatches to {target} ({reason})",
                file=sys.stderr,
                flush=True,
            )
            return target
        except OSError as error:
            print(f"[vllm-ascend/turboquant] cannot write {target} ({error!r})", file=sys.stderr, flush=True)
            return None

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
        """Release the file handles and the interpreter hooks. Not the hot path."""
        self._uninstall_hooks()
        with self._lock:
            if self._file is not None and self._owns_file:
                with contextlib.suppress(OSError):
                    self._file.close()
            self._file = None
            self._owns_file = False
            if self._cases_file is not None:
                with contextlib.suppress(OSError):
                    self._cases_file.close()
            self._cases_file = None

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
