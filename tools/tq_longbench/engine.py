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
"""One sequence, one process, one loop.

:class:`StandaloneModelRunner` owns the weights, the cache pools and the
generation loop.  What it deliberately does not own is a scheduler, a request
queue, a block allocator or a worker: a LongBench item is one prompt and one
greedy continuation, and everything an engine adds to that is overhead the
measurement would have to subtract again.

**The two prefill modes** differ in what the prefix the decode reads is made of:

* ``dense_staging`` -- the unquantised cache is allocated too, prefill attends
  over it with ``npu_fused_infer_attention_score``, and each chunk is written to
  *both* pools.  Prefill is then exactly the reference model's prefill, and the
  only quantisation in the run is the one being measured.  It costs a full fp16
  KV cache, which is what caps it near 32k.
* ``batched_decode`` -- only the quantised pool exists.  A chunk attends through
  the TurboQuant decode with one context length per token, which is bottom-right
  causal written out row by row.  No fp16 cache, so 128k fits; the prefix the
  chunk attends over is itself quantised, which is the difference from
  ``dense_staging`` and the reason the NIAH alignment check uses the latter.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tq_longbench import glm4
from tq_longbench._ascend import assert_no_vllm_imported, turboquant_rotation
from tq_longbench.kv_cache import CacheGeometry, dense_equivalent_bytes
from tq_longbench.layers import (
    FOLD_SITES,
    CausalLM,
    DecodeWorkspace,
    ForwardBatch,
    ModelShape,
    fold_on_device,
    fold_output_projection_in_place,
    fold_output_projection_weight,
    fold_output_rotation,
    pi_matrix,
    require_foldable,
)
from tq_longbench.ops import (
    DENSE_BACKENDS,
    AttentionBackend,
    LayerShape,
    build_backend,
    dense_backend_for,
)

PREFILL_MODES = ("dense_staging", "batched_decode")

DEFAULT_CHUNK_SIZE = 2048

#: Above this the fp16 staging pool stops being affordable; ``run_eval`` says so
#: rather than letting the allocation fail halfway through a sweep.
DENSE_STAGING_RECOMMENDED_MAX_SEQ = 32768

#: ``RunnerConfig.decode_graph``.
DECODE_GRAPH_MODES = ("auto", "on", "off")

_MEGABYTE = 1024 * 1024

_O_PROJ_WEIGHT_SUFFIX = "self_attn.o_proj.weight"

#: ``(checkpoint name, tensor) -> [(harness parameter, tensor), ...]``; empty to skip.
CheckpointMapper = Callable[[str, torch.Tensor], list[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class FoldReport:
    """How many ``o_proj`` weights were folded, where, and what it cost."""

    folded: int
    site: str
    seconds: float

    def describe(self) -> str:
        return f"{self.folded} o_proj folded on the {self.site} in {self.seconds:.2f} s"


#: ``npu_format_cast``'s ACL_FORMAT_FRACTAL_NZ -- the Cube's own tiling, as
#: ``vllm_ascend/utils.py`` numbers it. A weight already in it needs no layout
#: conversion on the way into the Cube.
ACL_FORMAT_FRACTAL_NZ = 29

#: The SoC versions ``torch_npu.npu.get_soc_version()`` reports for a 310P, as
#: ``vllm_ascend``'s ``check_ascend_device_type`` reads them.
SOC_VERSIONS_310P = range(200, 206)

#: ``RunnerConfig.weight_nz``. ``auto`` is the plugin's own policy, below.
WEIGHT_NZ_MODES = ("auto", "on", "off")

#: A cast that reorders a weight's bytes must not change the product it forms.
#: Held to a cosine rather than an exact match because the Cube may accumulate a
#: reordered sum differently; anything that is actually the wrong layout lands
#: nowhere near this.
NZ_PROBE_MIN_COSINE = 0.9999


@dataclass(frozen=True)
class WeightLayoutReport:
    """Whether the fused projections were cast to FRACTAL_NZ, and why not if not."""

    cast: int
    seconds: float
    reason: str = ""
    #: What ``torch_npu.get_npu_format`` said about the first weight afterwards,
    #: so "cast" means the bytes moved rather than that the call returned.
    confirmed_format: int | None = None
    #: The probe product's agreement with the same product before the cast.
    cosine: float | None = None

    def describe(self) -> str:
        if not self.cast:
            return f"weights in ND: {self.reason}"
        confirmed = "" if self.confirmed_format is None else f", format {self.confirmed_format}"
        agreement = "" if self.cosine is None else f", probe cos {self.cosine:.6f}"
        return f"{self.cast} fused projections cast to FRACTAL_NZ in {self.seconds:.2f} s{confirmed}{agreement}"


def should_cast_to_nz(mode: str, dtype: torch.dtype, device: torch.device) -> tuple[bool, str]:
    """Whether to put a fused projection in FRACTAL_NZ, and the reason either way.

    ``auto`` is ``vllm_ascend``'s own policy for a weight of this dtype, which is
    worth following rather than improving on:

    * float32 never converts -- NZ has no float32 tiling.
    * A 310P always converts; its Cube reads nothing else well.
    * Anywhere else, ``_should_trans_nz`` converts a float16 or bfloat16 weight
      only at ``weight_nz_mode == 2``, and the shipped default
      (``VLLM_ASCEND_ENABLE_NZ=1``) is "quantised weights only". So on a 910/950
      the plugin leaves dense fp16 weights in ND, and ``auto`` does too.

    ``on`` is that opt-in: it casts wherever the operator exists, which is the
    thing to measure on silicon before making it the default here. Being unable
    to measure it is exactly why ``auto`` defers to the plugin instead of
    guessing that NZ must be faster because it is more native.
    """
    if mode == "off":
        return False, "weight_nz='off'"
    if device.type != "npu":
        return False, f"{device.type} has no fractal layout"
    if dtype == torch.float32:
        return False, "float32 has no NZ tiling"
    if mode == "on":
        return True, "weight_nz='on'"
    soc = _soc_version()
    if soc is None:
        return False, "torch_npu does not report a SoC version, so the 310P rule cannot be applied"
    if soc in SOC_VERSIONS_310P:
        return True, f"SoC {soc} is a 310P, which always converts"
    return False, (
        f"SoC {soc} is not a 310P and the dtype is {dtype}; vllm_ascend converts these only at "
        "weight_nz_mode 2, so --weight-nz on is how to ask for it"
    )


def _soc_version() -> int | None:
    npu = getattr(sys.modules.get("torch_npu"), "npu", None)
    reader = getattr(npu, "get_soc_version", None)
    if reader is None:
        return None
    try:
        return int(reader())
    except Exception:  # a torch_npu that cannot answer has answered
        return None


@dataclass
class RunnerConfig:
    model_path: str
    backend: str = "turboquant_cube"
    prefill_mode: str = "dense_staging"
    max_seq_len: int = 32768
    chunk_size: int = DEFAULT_CHUNK_SIZE
    block_size: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "npu:0"
    max_new_tokens: int = 64
    attn_output_gate: bool = False
    fold_output_rotation: bool = False
    #: Skip the weight files and initialise randomly. For shape and plumbing
    #: checks where a checkpoint is neither available nor relevant.
    random_weights: bool = False
    seed: int = 0
    #: NativeV5Backend's torch_npu entry point (ops.DENSE_ATTENTION_APIS); ``None``
    #: takes the most preferred one torch_npu has. ``smoke_glm``'s pre-flight
    #: fills it with the one that actually ran on this SoC.
    dense_attention_api: str | None = None
    #: Which unquantised backend ``dense_staging`` allocates its pool through;
    #: ``None`` takes :func:`ops.dense_backend_for`. The pre-flight fills it when
    #: the preferred one is refused here -- an Ascend 950 stages through
    #: ``cann_dense``, because op-plugin's fused attention is refused on it.
    dense_staging_backend: str | None = None
    #: Where the ``o_proj`` fold runs (``layers.FOLD_SITES``). ``auto`` folds on
    #: the device when there is one: the shipped float64 transform is 40 layers
    #: of single-threaded host work, and the same map is a matmul the Cube does
    #: while the next shard is still being read.
    fold_site: str = "auto"
    #: Whether the decode runs as a captured graph (``DECODE_GRAPH_MODES``).
    #: ``auto`` captures wherever the device and the backend allow it and falls
    #: back quietly where they do not; ``on`` makes a run that cannot capture an
    #: error rather than a slow success, which is what a benchmark wants; ``off``
    #: is the eager path, and the one to compare against.
    decode_graph: str = "auto"
    #: Whether the fused projections are cast to Ascend's fractal NZ layout
    #: (``WEIGHT_NZ_MODES``). ``auto`` follows vllm_ascend's own policy for the
    #: dtype and SoC; see :func:`should_cast_to_nz`.
    weight_nz: str = "auto"
    #: How many leading tokens of the sequence bypass 4-bit quantisation and are
    #: attended to at activation precision (``VLLM_ASCEND_TQ_SINK_TOKENS``). 0 is
    #: off, and every TurboQuant path is then the one that ships. Above 0 each
    #: decode step also recomputes the quantised context's softmax denominator on
    #: the host, because neither decode operator returns it, so a step costs
    #: ``O(context)`` torch work on top of its launch and can be neither static
    #: nor captured. Only the TurboQuant backends read it.
    sink_tokens: int = 0

    def __post_init__(self) -> None:
        if self.decode_graph not in DECODE_GRAPH_MODES:
            raise ValueError(f"unknown decode_graph {self.decode_graph!r}; choose one of {list(DECODE_GRAPH_MODES)}")
        if self.weight_nz not in WEIGHT_NZ_MODES:
            raise ValueError(f"unknown weight_nz {self.weight_nz!r}; choose one of {list(WEIGHT_NZ_MODES)}")
        if self.prefill_mode not in PREFILL_MODES:
            raise ValueError(f"unknown prefill mode {self.prefill_mode!r}; choose one of {list(PREFILL_MODES)}")
        if self.backend in DENSE_BACKENDS and self.prefill_mode != "dense_staging":
            raise ValueError(
                f"the {self.backend} baseline decodes out of the unquantised cache, so it needs --prefill-mode "
                "dense_staging; batched_decode never allocates one."
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens cannot be negative, got {self.sink_tokens}")
        if self.sink_tokens and self.prefill_mode == "batched_decode":
            # batched_decode presents a whole chunk as a batch of per-position decodes, so
            # every prefill row would pay the merge's O(context) recomputation and the
            # prefill would be O(context^2) on the host. It is also the mode with no
            # counterpart in the plugin: vllm-ascend's TurboQuant backend refuses chunked
            # prefill outright and prefills densely, where the sinks are exact anyway.
            raise ValueError(
                "sink_tokens needs --prefill-mode dense_staging: the sink merge recomputes the softmax "
                "denominator per attending row, so a batched_decode prefill would pay it once per prompt "
                "token. Prefill densely (which is what the plugin's backend does, and where the sinks are "
                "already exact), or run with --sink-tokens 0."
            )
        if self.sink_tokens and self.backend == "turboquant_cube":
            # See TurboQuantCubeBackend: the merge reads the packed cache row-major on the
            # Lloyd-Max grid, and the kv4fp8 writer leaves NZ-tiled affine fp8 planes.
            # Refused here as well so a run fails before its weights load.
            raise ValueError(
                "sink_tokens cannot be combined with turboquant_cube: the uncompressed-sink merge reads the "
                "packed cache row-major on the Lloyd-Max codebook grid, and the kv4fp8 Cube writer leaves "
                "NZ-tiled planes of codes that are affine on an fp8 grid -- it would return plausible wrong "
                "numbers rather than raise. Use the turboquant_aiv backend, or sink_tokens 0."
            )
        if self.sink_tokens and self.backend in DENSE_BACKENDS:
            raise ValueError(
                f"{self.backend} decodes out of the unquantised cache, so there is nothing for sink_tokens "
                "to keep uncompressed; it is a TurboQuant option."
            )
        if self.fold_site not in FOLD_SITES:
            raise ValueError(f"unknown fold_site {self.fold_site!r}; choose one of {list(FOLD_SITES)}")
        if self.dense_staging_backend is not None and self.dense_staging_backend not in DENSE_BACKENDS:
            raise ValueError(
                f"dense_staging_backend {self.dense_staging_backend!r} does not decode out of an unquantised "
                f"cache; choose one of {sorted(DENSE_BACKENDS)}"
            )


#: How many real decode passes run before a capture records one.
#:
#: Three, because the first is not representative of the ones after it and the
#: capture must record the steady state: cuBLAS picks and caches a kernel for
#: each new GEMM shape on first call, the caching allocator learns the step's
#: block sizes, and lazy module state (the column index the dense backend grows,
#: a backend's scratch) is built. Capturing any of that would bake a one-off into
#: every replay -- or, for an allocation that happens once, capture a pointer the
#: replay then reuses for something else.
GRAPH_WARMUP_STEPS = 3

#: Graph capture is per *window* -- how many cache slots the decode reads -- and
#: the windows are powers of two, so a run captures ``log2`` of its context
#: rather than one graph per position. A step takes the smallest window that
#: covers it; the slots between its context length and the window are masked off
#: by ``context_lens``, exactly as an over-long window always was, so the only
#: cost of rounding up is reading cache that contributes nothing.
GRAPH_WINDOW_GROWTH = 2


def graph_windows(max_seq_len: int, block_size: int) -> tuple[int, ...]:
    """The window sizes a run may capture, smallest first.

    Powers of two from one block up to the cache, so a 128k run holds ten graphs
    rather than 128k of them and no step reads more than twice the cache it
    needs. The cache's own length is always the last, because a step at the very
    end must have a window that covers it.
    """
    windows: list[int] = []
    window = block_size
    while window < max_seq_len:
        windows.append(window)
        window *= GRAPH_WINDOW_GROWTH
    windows.append(max_seq_len)
    return tuple(windows)


class GraphCaptureUnavailable(RuntimeError):
    """Why this run cannot capture a decode graph. Never raised past ``decode_graph='on'``."""


def _graph_api(device: torch.device):
    """``(new_graph, capture_context)`` for this device, or ``None`` where there is none.

    The two vendors spell the same thing the same way one level down:
    ``torch.cuda.CUDAGraph`` / ``torch.cuda.graph``, and ``NPUGraph`` / ``graph``
    on the NPU. Only the module differs, so only the module is dispatched on.

    ``torch_npu`` is never imported here. It is reached through
    ``sys.modules`` if something else has already imported it -- which
    :func:`tq_longbench.smoke_glm.resolve_device` does before torch can parse an
    ``npu`` device at all -- and through ``torch.npu``, the alias it installs,
    otherwise. A CUDA run must not import it and a CPU run has nothing to import.
    """
    if device.type == "cuda":
        return torch.cuda.CUDAGraph, torch.cuda.graph
    if device.type != "npu":
        return None
    for module in (getattr(sys.modules.get("torch_npu"), "npu", None), getattr(torch, "npu", None)):
        graph_type = getattr(module, "NPUGraph", None)
        capture = getattr(module, "graph", None)
        if graph_type is not None and capture is not None:
            return graph_type, capture
    return None


class StaticDecodeGraph:
    """The whole single-token decode -- every layer, the head and the argmax -- recorded once.

    A BS=1 decode step of a 3B model is on the order of four hundred kernel
    launches, none of which runs for as long as it takes the host to submit the
    next one. Measured on an RTX 5070 before any of this: 42 ms a token, of which
    **42 ms was host time** -- the device was idle, waiting for Python. Fusing
    the projections took a millisecond off, because the problem was never the
    number of GEMMs, it was that every launch crosses the interpreter.

    A captured graph is the answer to exactly that: the launches are recorded
    once into a structure the driver replays as a unit, so a step costs one
    submission and whatever the kernels genuinely take.

    Three things have to be true of a step before it can be recorded, and the
    rest of this harness's decode path was changed to make them true:

    * **Fixed addresses.** A replay reads and writes the buffers the capture saw.
      That is :class:`~tq_longbench.layers.DecodeWorkspace`, and it is why the
      step's inputs are written into tensors rather than passed as arguments.
    * **Fixed shapes.** The context grows by a token a step, so the attention's
      shape would too. Hence the window: the decode reads a fixed number of cache
      slots and masks off the ones past its own length, and the captures are
      bucketed by window (:func:`graph_windows`).
    * **No synchronisation.** Every host read of a device tensor is illegal
      during capture -- and was a stall before it. The tie-point check and each
      backend's ``int(context_lens.max())`` were the two, and both are gone from
      this path: the length lives in ``context_lens`` on the device, which is
      also what makes a replay see a length the capture never knew.

    What a replay cannot do is change how long the model is or which layer runs;
    it re-executes the recorded work over whatever the buffers now hold. So the
    graph is invalidated by nothing here -- the weights, the pools and the
    workspace all outlive it -- and the only per-step host work left is seating
    three integers and reading one back.
    """

    def __init__(
        self,
        model: CausalLM,
        workspace: DecodeWorkspace,
        write_backends: tuple[AttentionBackend, ...],
        decode_backend: AttentionBackend,
        windows: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.write_backends = write_backends
        self.decode_backend = decode_backend
        self.windows = windows
        self.device = device
        api = _graph_api(device)
        if api is None:
            raise GraphCaptureUnavailable(f"{device.type} has no graph capture in this torch build")
        if not decode_backend.supports_static_decode:
            raise GraphCaptureUnavailable(
                f"the {decode_backend.name} decode reads its context lengths back to the host on every layer, "
                "which cannot be captured; its lengths would be frozen at the length the capture saw"
            )
        if not decode_backend.supports_graph_capture:
            raise GraphCaptureUnavailable(
                f"the {decode_backend.name} decode path has a shape that depends on a tensor's values, which is "
                "a host read wherever it appears; see its supports_graph_capture"
            )
        self._new_graph, self._capture = api
        self._graphs: dict[int, object] = {}
        self._pool = None
        #: Every window that was captured, in the order it was first needed.
        self.captured: list[int] = []

    # ------------------------------------------------------------- the step

    def window_for(self, context_len: int) -> int:
        """The smallest captured window that covers ``context_len`` slots."""
        for window in self.windows:
            if window >= context_len:
                return window
        raise ValueError(f"a context of {context_len} exceeds the {self.windows[-1]}-slot cache")

    def _batch(self, window: int) -> ForwardBatch:
        workspace = self.workspace
        return ForwardBatch(
            positions=workspace.positions,
            slots=workspace.slots,
            # Never read on this path: the length that matters is context_lens,
            # on the device, where a replay can see it change.
            prefix_end=0,
            is_decode=True,
            workspace=workspace,
            context_lens=workspace.context_lens,
            window=window,
        )

    def _run(self, window: int) -> None:
        """One decode step into the workspace, ending with the token it chose.

        The argmax is part of the step rather than the caller's: inside a capture
        it costs nothing extra, and it turns the host's job from launching a
        reduction over 152k logits into reading one integer back.
        """
        workspace = self.workspace
        logits = self.model(
            workspace.token_ids,
            self._batch(window),
            self.write_backends,
            self.decode_backend,
        )
        torch.argmax(logits[-1], dim=-1, keepdim=True, out=workspace.next_token)

    def capture(self, window: int) -> None:
        """Warm up and record one window, if it is not recorded already.

        The warmup passes are real: they execute, and they write this step's K/V
        into this step's slot. That is safe because writing the same token at the
        same position is idempotent -- the pass is the step, run more than once --
        and it is *necessary*, because a capture that had not seen the steady
        state would record a kernel choice or an allocation that only the first
        call makes.

        Recording itself executes nothing, so the caller still replays afterwards
        to get the step it asked for.
        """
        if window in self._graphs:
            return
        stream_api = getattr(torch, self.device.type)
        side = stream_api.Stream()
        side.wait_stream(stream_api.current_stream())
        with stream_api.stream(side):
            for _ in range(GRAPH_WARMUP_STEPS):
                self._run(window)
        stream_api.current_stream().wait_stream(side)
        stream_api.synchronize()

        graph = self._new_graph()
        with self._capture_context(graph):
            self._run(window)
        if self._pool is None:
            self._pool = getattr(graph, "pool", lambda: None)()
        self._graphs[window] = graph
        self.captured.append(window)

    def _capture_context(self, graph):
        """The vendor's capture context, sharing one memory pool across the windows.

        Sharing is safe here, and it is worth saying why rather than assuming it:
        a shared pool lets a later graph's allocations land on an earlier
        graph's, so it is only sound when nothing an earlier replay wrote into
        the pool has to survive. Nothing does. Everything that crosses a step
        boundary -- the workspace buffers and both KV pools -- was allocated in
        the runner's constructor, long before any capture, so it is outside the
        pool entirely. What is inside is the per-step temporaries: the gather of
        ``cos``/``sin``, the softmax scratch, each projection's output. All of
        them are dead by the time the step ends, whichever window ran it.

        So one pool costs the largest window's intermediates rather than the sum
        of every window's. A capture API that does not take a pool is used
        without one -- a bigger footprint, not a wrong answer -- because refusing
        to capture over it would be the worse trade.
        """
        if self._pool is None:
            return self._capture(graph)
        try:
            return self._capture(graph, pool=self._pool)
        except TypeError:
            return self._capture(graph)

    def step(self, token: int, position: int) -> int:
        """Run one decode step for ``token`` at ``position``; return the token it chose.

        The three integers are written into the workspace with ``fill_``, which
        is a launch rather than a copy across the bus, and the answer comes back
        as one host read. Everything between them is the replay.
        """
        context_len = position + 1
        window = self.window_for(context_len)
        self.workspace.seat(token, position, context_len)
        self.capture(window)
        self._graphs[window].replay()
        return int(self.workspace.next_token)

    def describe(self) -> str:
        if not self.captured:
            return "decode graph: ready, nothing captured yet"
        return f"decode graph: {len(self.captured)} captured at windows {sorted(self.captured)}"


@dataclass
class RunMetrics:
    """What one sequence cost. Latencies in microseconds, memory in MB."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    #: Time to first token: the prompt in, the first token chosen. Prefill plus
    #: the argmax that reads its logits, which is the only work between them.
    ttft_seconds: float = 0.0
    decode_step_us: list[float] = field(default_factory=list)
    kv_cache_mb: float = 0.0
    dense_equivalent_mb: float = 0.0
    #: The uncompressed attention-sink side-car, on top of :attr:`kv_cache_mb` rather
    #: than inside it: the compression claim is about the packed pool, and this is what
    #: keeping a handful of tokens out of it costs. Zero with sinks off.
    sink_cache_mb: float = 0.0
    #: Device memory at the end of the run, from the allocator rather than the
    #: pools: what the caches hold is budgeted, what the run peaked at is measured.
    #: Zero on a device with no allocator to ask (CPU).
    device_allocated_mb: float = 0.0
    device_peak_mb: float = 0.0
    device_reserved_mb: float = 0.0
    #: The stop token the loop ended on, or ``None`` where it ran out of budget.
    #: A whole run of ``None`` means nothing is stopping generation -- which is
    #: what an empty stop set looks like from the outside, and is otherwise
    #: indistinguishable from a model that simply had more to say.
    stopped_on_token: int | None = None
    #: Whether the decode ran as a captured graph. On the record because the two
    #: paths differ by an order of magnitude and by nothing visible in the text.
    decode_graph: bool = False
    #: The steps that also captured a window, kept apart from
    #: :attr:`decode_step_us` rather than dropped: they are real time the run
    #: spent, they are not what a steady-state token costs, and averaging them in
    #: would make a long run look better than a short one for no reason.
    capture_us: list[float] = field(default_factory=list)

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def kv_saved_mb(self) -> float:
        return self.dense_equivalent_mb - self.kv_cache_mb

    @property
    def decode_tokens_per_second(self) -> float:
        """Steady-state decode throughput, from the median step rather than the mean.

        The median because a capture is not a decode and an allocator's first
        touch is not either: both land in the tail, and a mean over sixteen steps
        lets one of them halve the number the run is judged on.
        """
        median = self.percentile(0.50)
        return 1e6 / median if median else 0.0

    def percentile(self, fraction: float) -> float:
        if not self.decode_step_us:
            return 0.0
        ordered = sorted(self.decode_step_us)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_seconds": round(self.prefill_seconds, 4),
            "prefill_tokens_per_second": round(self.prefill_tokens_per_second, 2),
            "ttft_ms": round(self.ttft_seconds * 1e3, 2),
            "decode_p50_us": round(self.percentile(0.50), 2),
            "decode_p90_us": round(self.percentile(0.90), 2),
            "decode_p99_us": round(self.percentile(0.99), 2),
            "kv_cache_mb": round(self.kv_cache_mb, 2),
            "sink_cache_mb": round(self.sink_cache_mb, 3),
            "dense_equivalent_mb": round(self.dense_equivalent_mb, 2),
            "kv_saved_mb": round(self.kv_saved_mb, 2),
            "device_allocated_mb": round(self.device_allocated_mb, 2),
            "device_peak_mb": round(self.device_peak_mb, 2),
            "device_reserved_mb": round(self.device_reserved_mb, 2),
            "stopped_on_token": self.stopped_on_token,
            "hit_token_budget": self.stopped_on_token is None,
            "decode_graph": self.decode_graph,
            "captures": len(self.capture_us),
            "capture_ms": round(sum(self.capture_us) / 1e3, 2),
            "decode_tokens_per_second": round(self.decode_tokens_per_second, 2),
        }


class StandaloneModelRunner:
    """Weights, caches and a greedy loop, with nothing between them and the kernels."""

    def __init__(self, config: RunnerConfig, model_shape: ModelShape | None = None) -> None:
        assert_no_vllm_imported()
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.shape = model_shape or self._load_shape()
        self.geometry = CacheGeometry(
            num_layers=self.shape.num_layers,
            num_kv_heads=self.shape.num_kv_heads,
            head_size=self.shape.head_size,
            block_size=config.block_size,
            max_seq_len=config.max_seq_len,
        )
        layer_shape = LayerShape(
            num_heads=self.shape.num_heads,
            num_kv_heads=self.shape.num_kv_heads,
            head_size=self.shape.head_size,
            scale=self.shape.scale,
        )

        self.decode_backend: AttentionBackend = build_backend(
            config.backend,
            self.geometry,
            layer_shape,
            self.device,
            config.dtype,
            config.fold_output_rotation,
            config.dense_attention_api,
            config.sink_tokens,
        )
        # dense_staging needs the unquantised pool as well, unless the decode
        # backend already is it. Which unquantised backend that is follows the
        # device, not --backend: a CUDA run stages through torch.
        if config.prefill_mode == "dense_staging" and config.backend not in DENSE_BACKENDS:
            # Folded like the decode: its output feeds the same o_proj.
            self.prefill_backend: AttentionBackend = build_backend(
                config.dense_staging_backend or dense_backend_for(self.device),
                self.geometry,
                layer_shape,
                self.device,
                config.dtype,
                config.fold_output_rotation,
                config.dense_attention_api,
            )
        else:
            self.prefill_backend = self.decode_backend

        # Deduplicated by identity: when the two are the same object a chunk
        # must be written once, not twice -- a second write would requantise
        # what the first wrote and is not idempotent.
        self.write_backends: tuple[AttentionBackend, ...] = (
            (self.decode_backend,)
            if self.prefill_backend is self.decode_backend
            else (self.prefill_backend, self.decode_backend)
        )

        self.model = CausalLM(self.shape, config.max_seq_len, config.dtype, self.device)
        self.model.eval()
        #: Filled by :meth:`load_weights`; a run with random weights folds nothing.
        self.fold_report = FoldReport(0, "host", 0.0)
        if config.fold_output_rotation:
            require_foldable(self.shape)
        if not config.random_weights:
            # Folds each o_proj as its shard tensor is ingested.
            self.load_weights()
        elif config.fold_output_rotation:
            fold_output_rotation(self.model)

        #: Whether the fused projections ended up in FRACTAL_NZ, and why not.
        #: Before the workspace and the graph: a capture records the matmuls the
        #: weights are in *now*, so the layout has to be settled first.
        self.layout_report = self._cast_fused_projections()

        #: The single-token decode's buffers, and the graph recorded over them.
        #: Both are built here rather than on the first step, so what a run costs
        #: to set up is paid before anything is timed.
        self.workspace = DecodeWorkspace.build(self.shape, config.dtype, self.device)
        #: Bytes the backends hold in buffers they would otherwise have grown.
        #: Before the graph, and before the first prefill: a replay writes to the
        #: addresses its capture saw, so a buffer replaced afterwards leaves every
        #: graph recorded over it writing into memory that has been freed.
        self.reserved_launch_bytes = self._reserve_launch_buffers()
        self.decode_graph, self.decode_graph_refusal = self._build_decode_graph()

    def _reserve_launch_buffers(self) -> int:
        """Pin each backend's on-demand buffers at this run's ceiling.

        The widest launch a backend sees is a decode step -- one token -- unless
        it prefills as well, which is ``batched_decode``, where a chunk is
        ``chunk_size`` tokens through the same ``decode`` call. ``dense_staging``
        prefills through a different backend and only ever writes the quantised
        pool through this one, and a write sizes nothing.

        The two backends are asked separately rather than through
        ``write_backends``, because what has to be covered is the widest *launch*
        each one makes, and those differ between them.
        """
        prefills = self.prefill_backend is self.decode_backend
        reserved = self.decode_backend.reserve_launch_buffers(self.config.chunk_size if prefills else 1)
        if not prefills:
            reserved += self.prefill_backend.reserve_launch_buffers(self.config.chunk_size)
        return reserved

    def _cast_fused_projections(self) -> WeightLayoutReport:
        """Put ``qkv_proj`` and ``gate_up_proj`` in the Cube's own tiling, where that is the policy.

        The two fused projections and not ``o_proj`` or ``down_proj``: those are
        the ones a decode step reads twice a layer, and ``o_proj`` has Pi folded
        into it, which is arithmetic on the ND values that must happen first.

        The cast is **checked**, on the first weight, against the product the ND
        weight formed a moment earlier. A layout the operator reorders wrongly --
        or one a later matmul silently converts back -- does not raise: it
        returns numbers, and a model whose first projection is transposed is
        fluent nonsense. Checking costs one matmul per run and is the only way to
        learn the answer without an NPU to try it on.
        """
        wanted, reason = should_cast_to_nz(self.config.weight_nz, self.config.dtype, self.device)
        if not wanted:
            return WeightLayoutReport(0, 0.0, reason)
        torch_npu = sys.modules.get("torch_npu")
        cast = getattr(torch_npu, "npu_format_cast", None)
        if cast is None:
            if self.config.weight_nz == "on":
                raise RuntimeError("weight_nz='on' but this torch_npu has no npu_format_cast")
            return WeightLayoutReport(0, 0.0, "this torch_npu has no npu_format_cast")

        weights = [
            weight
            for layer in self.model.layers
            for weight in (layer.self_attn.qkv_proj.weight, layer.mlp.gate_up_proj.weight)
        ]
        probe = torch.randn(1, self.shape.hidden_size, dtype=self.config.dtype, device=self.device)
        before = torch.mm(probe, weights[0].t()).to(torch.float32)

        started = time.perf_counter()
        for weight in weights:
            weight.data = cast(weight.data, ACL_FORMAT_FRACTAL_NZ)
        seconds = time.perf_counter() - started

        cosine = _cosine(torch.mm(probe, weights[0].t()).to(torch.float32), before)
        reader = getattr(torch_npu, "get_npu_format", None)
        confirmed = None if reader is None else int(reader(weights[0].data))
        if cosine <= NZ_PROBE_MIN_COSINE:
            raise RuntimeError(
                f"the FRACTAL_NZ cast changed what the first fused projection computes: cosine {cosine:.6f} "
                f"against the same product in ND, floor {NZ_PROBE_MIN_COSINE}. The weights are reordered but "
                "the matmul is not reading them that way; run with --weight-nz off."
            )
        return WeightLayoutReport(len(weights), seconds, reason, confirmed, cosine)

    def _build_decode_graph(self) -> tuple[StaticDecodeGraph | None, str | None]:
        """The decode graph, or ``None`` and the reason there is not one.

        ``decode_graph='on'`` turns the reason into an error. That is the mode a
        benchmark should use: a run that quietly fell back to eager and reported
        75 ms a token would read as the kernels being slow, which is the one
        conclusion this harness exists to make impossible to reach by accident.
        """
        if self.config.decode_graph == "off":
            return None, "decode_graph='off'"
        try:
            graph = StaticDecodeGraph(
                self.model,
                self.workspace,
                self.write_backends,
                self.decode_backend,
                graph_windows(self.config.max_seq_len, self.config.block_size),
                self.device,
            )
        except GraphCaptureUnavailable as refusal:
            if self.config.decode_graph == "on":
                raise
            return None, str(refusal)
        return graph, None

    # ---------------------------------------------------------------- config

    def _raw_config(self) -> dict:
        return glm4.read_config(self.config.model_path)

    def _is_glm4(self) -> bool:
        return not self.config.random_weights and glm4.is_glm4_config(self._raw_config())

    def _load_shape(self) -> ModelShape:
        return read_checkpoint_shape(
            self.config.model_path,
            self.config.attn_output_gate,
            weights_on_disk=not self.config.random_weights,
        )

    def _checkpoint_mapper(self) -> CheckpointMapper:
        """How this checkpoint's tensor names land on the harness's modules."""
        if self._is_glm4():
            shape = self.shape
            return lambda name, tensor: glm4.glm4_checkpoint_targets(name, tensor, shape)

        def one_to_one(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
            target = _map_checkpoint_name(name)
            return [] if target is None else [(target, tensor)]

        return one_to_one

    # --------------------------------------------------------------- weights

    def load_weights(self) -> int:
        """Load safetensors shards straight into the modules, and report the count.

        Direct rather than through ``from_pretrained``: the point of the harness
        is that no framework sits between the checkpoint and the kernels, and a
        loader that instantiates an HF model first would allocate the weights
        twice on a device that has no room for it.

        With ``fold_output_rotation`` every ``o_proj`` is folded to
        ``W_o (I (x) Pi)`` here, on the host tensor, before it reaches the
        device -- see :func:`~tq_longbench.layers.fold_output_projection_weight`.
        """
        from safetensors.torch import load_file

        root = Path(self.config.model_path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            shards = sorted(
                {root / name for name in json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()}
            )
        else:
            shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors shards under {root}")

        # Views, not parameters: q/k/v and gate/up are each one fused tensor in
        # the module tree, and a checkpoint's separate names write into slices of
        # it. See CausalLM.load_destinations.
        destinations = self.model.load_destinations()
        mapper = self._checkpoint_mapper()
        fold = self.config.fold_output_rotation
        on_device = fold and fold_on_device(self.config.fold_site, self.device)
        pi = pi_matrix(self.shape.head_size, self.device) if on_device else None
        filled: set[str] = set()
        folded = 0
        checked_the_device_fold = False
        fold_seconds = 0.0
        for shard in shards:
            # One shard resident at a time: a 128k-context run has no headroom
            # for the whole checkpoint in host memory alongside the pools.
            for name, tensor in load_file(str(shard)).items():
                for target, piece in mapper(name, tensor):
                    parameter = destinations.get(target)
                    if parameter is None:
                        raise KeyError(f"{name} maps to {target}, which this harness's module tree does not have")
                    if tuple(parameter.shape) != tuple(piece.shape):
                        raise ValueError(
                            f"{name} is {tuple(piece.shape)} in the checkpoint but {tuple(parameter.shape)} in the "
                            f"harness. If this is a gated model, pass --attn-output-gate."
                        )
                    is_o_proj = fold and target.endswith(_O_PROJ_WEIGHT_SUFFIX)
                    started = time.perf_counter()
                    if is_o_proj and not on_device:
                        piece = fold_output_projection_weight(piece, self.shape.head_size)
                    fold_seconds += time.perf_counter() - started if is_o_proj else 0.0
                    with torch.no_grad():
                        parameter.copy_(piece.to(device=self.device, dtype=parameter.dtype))
                    if is_o_proj and on_device:
                        started = time.perf_counter()
                        fold_output_projection_in_place(parameter, self.shape.head_size, pi)
                        if not checked_the_device_fold:
                            # Once, on the first one: the matmul accumulates in
                            # float32, and a device that quietly gives a matmul
                            # less precision than that would fold every layer
                            # slightly wrong and raise nowhere. Checked against
                            # the weight it came from, so it is the fold that is
                            # under test rather than the route.
                            self._check_device_fold(piece, parameter)
                            checked_the_device_fold = True
                        fold_seconds += time.perf_counter() - started
                    folded += is_o_proj
                    filled.add(target)

        unfilled = sorted(set(destinations) - filled)
        if self.shape.tie_word_embeddings:
            unfilled = [name for name in unfilled if name != "lm_head.weight"]
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameters were never filled from the checkpoint, starting with {unfilled[:4]}. "
                "Running with them at their initial values would produce fluent nonsense rather than an error."
            )
        if fold and folded != self.shape.num_layers:
            # A layer the decode treats as folded but whose o_proj was not would
            # hand the rotated basis to a projection that does not undo it.
            raise RuntimeError(f"folded {folded} o_proj weights for {self.shape.num_layers} layers")
        self.fold_report = FoldReport(folded, "device" if on_device else "host", fold_seconds)
        return len(filled)

    def _check_device_fold(self, original: torch.Tensor, folded: torch.Tensor) -> None:
        """Hold one device-folded projection to the float64 host route it replaces.

        ``validate_output_projection_fold`` draws rotated attention outputs and
        compares ``o~ W_folded`` against ``o W_original`` -- which is the product
        the decode actually forms, rather than the weights side by side. A fold
        that is wrong in the way a reduced-precision matmul would make it wrong
        shows up there and nowhere else: the run would otherwise be fluent.
        """
        rotation = turboquant_rotation()
        report = rotation.validate_output_projection_fold(
            original.to(device="cpu", dtype=torch.float32),
            folded.to(device="cpu", dtype=torch.float32),
            self.shape.head_size,
        )
        if not report.passed():
            raise RuntimeError(
                f"the o_proj fold on {self.device} does not reproduce the host fold: cosine "
                f"{report.min_cosine:.6f} over {report.num_samples} samples, relative error "
                f"{report.max_relative_error:.3e}. This device's float32 matmul is not float32 enough to "
                "fold through; re-run with fold_site='host' (--fold-site host)."
            )

    # ---------------------------------------------------------------- memory

    def _sink_cache_bytes(self) -> int:
        """The decode backend's uncompressed sink plane, where it has one."""
        reader = getattr(self.decode_backend, "sink_cache_bytes", None)
        return reader() if reader is not None else 0

    def memory_report(self) -> dict:
        """What the pools hold, and what the same context would have cost dense."""
        dense_bytes = dense_equivalent_bytes(self.geometry, self.config.dtype)
        held = self.decode_backend.cache.bytes_allocated()
        if self.prefill_backend is not self.decode_backend:
            held += self.prefill_backend.cache.bytes_allocated()
        return {
            "kv_cache_mb": held / _MEGABYTE,
            "dense_equivalent_mb": dense_bytes / _MEGABYTE,
            "decode_cache_mb": self.decode_backend.cache.bytes_allocated() / _MEGABYTE,
            "bytes_per_token_per_layer": self.decode_backend.cache.bytes_per_token_per_layer(),
            # Budgeted like the pools rather than left to the allocator: small
            # next to a KV cache, and the whole of what a decode step writes into.
            "decode_workspace_mb": self.workspace.bytes_allocated() / _MEGABYTE,
            # Zero with sinks off, and counted separately from kv_cache_mb either way:
            # the compression claim is about the packed pool, and this is what keeping a
            # handful of tokens out of it costs on top.
            "sink_cache_mb": self._sink_cache_bytes() / _MEGABYTE,
        }

    # -------------------------------------------------------------- the loop

    def _batch(self, start: int, count: int, is_decode: bool) -> ForwardBatch:
        positions = torch.arange(start, start + count, dtype=torch.int64, device=self.device)
        return ForwardBatch(
            positions=positions,
            slots=self.decode_backend.cache.slot_mapping(start, count),
            prefix_end=start + count,
            is_decode=is_decode,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor, metrics: RunMetrics) -> torch.Tensor:
        """Run the prompt through in chunks and return the logits of its last token."""
        total = token_ids.numel()
        if total > self.config.max_seq_len:
            raise ValueError(f"a {total}-token prompt does not fit the {self.config.max_seq_len}-token cache")
        _synchronize(self.device)
        started = time.perf_counter()
        logits = None
        for start in range(0, total, self.config.chunk_size):
            count = min(self.config.chunk_size, total - start)
            batch = self._batch(start, count, is_decode=False)
            logits = self.model(token_ids[start : start + count], batch, self.write_backends, self.prefill_backend)
        _synchronize(self.device)
        metrics.prefill_seconds = time.perf_counter() - started
        metrics.prompt_tokens = total
        assert logits is not None
        return logits

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        stop_ids: Collection[int] | None = None,
    ) -> tuple[list[int], RunMetrics]:
        """Greedy continuation of ``token_ids``; returns the new tokens and what they cost.

        ``stop_ids`` ends the loop, rather than only ending the text afterwards.
        The difference is not cosmetic: a run that generates its whole budget and
        is trimmed on the way out has already paid for every step, so its decode
        percentiles average over tokens the model never meant to emit, and the
        continuation on record is a trimmed version of something longer. The stop
        token itself is kept in the returned ids -- it is what happened -- and
        :func:`tq_longbench.smoke_glm.stop_at_eos` is still what turns them into
        text, so the two agree about where the answer ends.
        """
        metrics = RunMetrics(**self._memory_metrics())
        _reset_peak_memory(self.device)
        started = time.perf_counter()
        logits = self.prefill(token_ids, metrics)
        produced: list[int] = []
        position = token_ids.numel()
        limit = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        metrics.decode_graph = self.decode_graph is not None
        next_token = int(torch.argmax(logits[-1]))
        # argmax syncs on its own (the id crosses to the host), so this boundary
        # is the token existing, not the launch being queued.
        metrics.ttft_seconds = time.perf_counter() - started

        for _ in range(limit):
            produced.append(next_token)
            if stop_ids and next_token in stop_ids:
                metrics.stopped_on_token = next_token
                break
            if position + 1 > self.config.max_seq_len:
                break
            # A capture the first time a window is needed, and the step it was
            # captured for runs as the replay right after, so nothing is
            # generated twice and nothing is timed that was not a decode. The
            # capture itself lands in that step's latency, which is why it is
            # reported (``captures``) rather than smoothed away.
            captured_before = len(self.decode_graph.captured) if self.decode_graph else 0
            _synchronize(self.device)
            step_started = time.perf_counter()
            if self.decode_graph is not None:
                next_token = self._graph_step(next_token, position)
            else:
                next_token = self._eager_step(next_token, position)
            _synchronize(self.device)
            elapsed_us = (time.perf_counter() - step_started) * 1e6
            metrics.decode_graph = self.decode_graph is not None
            if self.decode_graph and len(self.decode_graph.captured) != captured_before:
                metrics.capture_us.append(elapsed_us)
            else:
                metrics.decode_step_us.append(elapsed_us)
            position += 1
        metrics.generated_tokens = len(produced)
        for name, value in device_memory_mb(self.device).items():
            setattr(metrics, name, value)
        return produced, metrics

    def _graph_step(self, token: int, position: int) -> int:
        """One step through the graph, falling back to eager if a capture is refused.

        Only under ``decode_graph='auto'``, and only once: a device whose graph
        API exists but will not record this step -- which no machine here can
        rule out for an NPU -- finishes its run eagerly instead of failing it,
        and says so. ``on`` lets the failure through, because a run that asked
        for a graph and silently did not get one is a benchmark reporting the
        wrong number.

        The step is retried eagerly rather than lost: the capture executed
        nothing, so the token it was for has not been generated yet.
        """
        try:
            return self.decode_graph.step(token, position)
        except Exception as failure:
            if self.config.decode_graph == "on":
                raise
            self.decode_graph = None
            self.decode_graph_refusal = f"capture failed and the run continued eagerly: {failure}"
            return self._eager_step(token, position)

    def _eager_step(self, token: int, position: int) -> int:
        """One decode step through the workspace, launch by launch.

        The same arithmetic the graph records, over the same buffers -- so the
        two paths differ in how the work is submitted and in nothing else, and
        ``--decode-graph off`` is a fair thing to compare against rather than a
        different model. The window is the graph's too: it is what removes the
        per-layer read of ``context_lens`` back to the host, and that is worth
        having whether or not a graph is captured over it.
        """
        workspace = self.workspace
        context_len = position + 1
        workspace.seat(token, position, context_len)
        batch = ForwardBatch(
            positions=workspace.positions,
            slots=workspace.slots,
            prefix_end=0,
            is_decode=True,
            workspace=workspace,
            context_lens=workspace.context_lens,
            window=self._eager_window(context_len),
        )
        logits = self.model(workspace.token_ids, batch, self.write_backends, self.decode_backend)
        torch.argmax(logits[-1], dim=-1, keepdim=True, out=workspace.next_token)
        return int(workspace.next_token)

    def _eager_window(self, context_len: int) -> int | None:
        """How many slots the eager decode reads, or ``None`` to let the backend ask.

        A backend that cannot run window-free is one that reads its lengths back
        anyway, so there is nothing to save; everywhere else the window is the
        context itself, because without a graph there is no reason to round up
        to a bucket and read cache that contributes nothing.
        """
        return context_len if self.decode_backend.supports_static_decode else None

    def _memory_metrics(self) -> dict:
        report = self.memory_report()
        return {
            "kv_cache_mb": report["kv_cache_mb"],
            "dense_equivalent_mb": report["dense_equivalent_mb"],
            "sink_cache_mb": report["sink_cache_mb"],
        }


_ATTENTION_SUFFIXES = {
    "q_proj.weight",
    "q_proj.bias",
    "k_proj.weight",
    "k_proj.bias",
    "v_proj.weight",
    "v_proj.bias",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
}


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.detach().to("cpu", torch.float64).flatten()
    right = expected.detach().to("cpu", torch.float64).flatten()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-30))


def checkpoint_tensor_names(model_path: str | Path) -> list[str]:
    """Every tensor name in the checkpoint, read from the index or the shard headers.

    Names only -- ``safe_open`` gives them without reading a byte of tensor data,
    so this costs nothing next to the load that follows.
    """
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])

    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names.extend(handle.keys())
    return names


def checkpoint_features(model_path: str | Path) -> dict:
    """What the config does not say: whether q/k are normed, and whether qkv has a bias.

    Read off the tensor names rather than the config, because neither is
    reliably declared. Qwen2.5 carries q/k/v biases with no ``attention_bias``
    key, and ``model_type`` alone does not separate a Qwen3 checkpoint (which
    norms each head) from a Qwen2 one (which does not). Getting either wrong
    builds a model whose parameters and the checkpoint's do not correspond,
    which :meth:`StandaloneModelRunner.load_weights` then refuses -- loudly,
    rather than by normalising with a weight of 1 that was never loaded.
    """
    names = checkpoint_tensor_names(model_path)
    return {
        "qk_norm": any(name.endswith("self_attn.q_norm.weight") for name in names),
        "qkv_bias": any(name.endswith("self_attn.q_proj.bias") for name in names),
    }


def read_checkpoint_shape(
    model_path: str | Path,
    attn_output_gate: bool = False,
    *,
    weights_on_disk: bool = True,
) -> ModelShape:
    """The harness's shape for the checkpoint at ``model_path``, without loading a weight.

    Two readers, chosen by ``model_type``: GLM-4's, because ``chatglm`` spells
    every field its own way, and the HF one for everything else -- Qwen2.5
    included, whose config states its head counts, its ``rope_theta`` and its
    full-width NeoX RoPE in the ordinary keys. Both are read here rather than
    once in the runner and again in the pre-flight: the synthetic layer the
    pre-flight probes has to be the layer the run will build, and a second
    reader is a second thing that can drift.

    ``weights_on_disk=False`` is a random-weights run: ``config.json`` is all
    there is, so the GLM-4 reader (which the fused tensor names go with) is not
    used and :func:`checkpoint_features` has nothing to read.
    """
    config = glm4.read_config(model_path)
    if weights_on_disk and glm4.is_glm4_config(config):
        # Read off config.json directly: the chatglm config class is remote code,
        # and nothing it computes is needed that the JSON does not say.
        return glm4.glm4_model_shape(config, attn_output_gate)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    detected = checkpoint_features(model_path) if weights_on_disk else {}
    return ModelShape.from_hf_config(hf_config, attn_output_gate, **detected)


def _map_checkpoint_name(name: str) -> str | None:
    """Translate a Hugging Face parameter name to this harness's module tree.

    Returns ``None`` for anything the harness has no home for -- rotary inverse
    frequencies, which are recomputed, and any buffer a checkpoint carries that
    is not a weight.  A name that *should* map but does not is caught by the
    unfilled-parameter check in :meth:`StandaloneModelRunner.load_weights`,
    because a silently skipped projection is fluent nonsense rather than a crash.
    """
    if name.endswith("rotary_emb.inv_freq"):
        return None
    if name == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"
    if not name.startswith("model.layers."):
        return None
    _, _, index, rest = name.split(".", 3)
    if rest.startswith("self_attn.") and rest[len("self_attn.") :] in _ATTENTION_SUFFIXES:
        return f"layers.{index}.self_attn.{rest[len('self_attn.') :]}"
    if rest.startswith("mlp.") or rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return f"layers.{index}.{rest}"
    return None


def device_memory_mb(device: torch.device) -> dict:
    """What the allocator holds on ``device``, in megabytes; zeros where there is no allocator.

    The pools' own arithmetic (:meth:`StandaloneModelRunner.memory_report`) says
    what the caches were budgeted. This says what the process actually took --
    weights, pools, and whatever a prefill chunk's score matrix peaked at, which
    on ``cann_dense`` is the largest transient in the run and is invisible to a
    budget.
    """
    module = getattr(torch, device.type, None)
    if module is None or not all(hasattr(module, name) for name in ("memory_allocated", "max_memory_allocated")):
        return {"device_allocated_mb": 0.0, "device_peak_mb": 0.0, "device_reserved_mb": 0.0}
    reserved = getattr(module, "memory_reserved", None)
    return {
        "device_allocated_mb": module.memory_allocated(device) / _MEGABYTE,
        "device_peak_mb": module.max_memory_allocated(device) / _MEGABYTE,
        "device_reserved_mb": (reserved(device) if reserved else 0) / _MEGABYTE,
    }


def _reset_peak_memory(device: torch.device) -> None:
    """Start a sequence's peak from here, so one run's transients are not the next one's."""
    module = getattr(torch, device.type, None)
    reset = getattr(module, "reset_peak_memory_stats", None)
    if reset is not None:
        reset(device)


def _synchronize(device: torch.device) -> None:
    """Drain the device queue so a timing boundary means what it says.

    Every launch in this harness is asynchronous; without this the prefill
    "duration" is the time to enqueue it and the decode percentiles measure
    dispatch rather than the kernels.
    """
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
