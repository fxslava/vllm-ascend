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
"""Pre-flight: every attention path a run will take, on one synthetic layer, before any weight loads.

A 9B checkpoint takes minutes to ingest and fold. An operator the SoC refuses --
``aclnnFusedInferAttentionScore`` V1-V4 on Ascend 950 (EZ9903), a standalone
library that does not load, a kernel symbol that is missing -- fails at the
first prefill chunk, after all of that. :func:`probe_backend` reaches the same
failure in the time one layer's launches take: it builds the backend over a
one-layer cache with the model's real head counts, writes :data:`PROBE_TOKENS`
synthetic tokens, runs a causal prefill chunk and a decode step, and compares
both against exact attention computed in float32 on the host.

The comparison is the point, not just the absence of an exception. An entry
point that runs but reads the causal mask the wrong way returns numbers too; a
dense path is held to :data:`DENSE_PROBE_MIN_COSINE` and a 4-bit one to
:data:`QUANTIZED_PROBE_MIN_COSINE`, which is quantisation loss away from any of
the failure modes (a wrong mask or basis lands far below it).

:func:`select_dense_api` does this once per torch_npu entry point
(:data:`tq_longbench.ops.DENSE_ATTENTION_APIS`) and keeps the first that passes,
so a run uses what this SoC and CANN actually accept rather than what the
torch_npu build merely exports. :func:`select_dense_backend` widens that to the
unquantised backends themselves: where no fused entry point is accepted at all,
which is every Ascend 950, it settles on ``cann_dense`` -- matmul, softmax,
matmul -- rather than leaving the run with no unquantised path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from tq_longbench._ascend import turboquant_rotation
from tq_longbench.kv_cache import CacheGeometry
from tq_longbench.ops import (
    DENSE_BACKENDS,
    LayerShape,
    NativeV5Backend,
    available_dense_attention_apis,
    build_backend,
    dense_backend_candidates,
)

#: Synthetic context length: a full prefill chunk and a decode over it, one block.
PROBE_TOKENS = 16

#: Unquantised attention against float32 exact attention: storage rounding only.
DENSE_PROBE_MIN_COSINE = 0.999

#: The 4-bit cache against exact attention over the unquantised K/V. Gaussian
#: synthetic data measured ~0.99 through the CPU stand-ins; a wrong mask, a
#: wrong basis or garbage output all land far below.
QUANTIZED_PROBE_MIN_COSINE = 0.95

_ERROR_DETAIL_CHARS = 300


@dataclass(frozen=True)
class ProbeResult:
    """One backend (and, for the native one, one torch_npu entry point) on the synthetic layer."""

    backend: str
    api: str | None
    passed: bool
    seconds: float
    detail: str

    def describe(self) -> str:
        name = self.backend if self.api is None else f"{self.backend}[{self.api}]"
        verdict = "ok" if self.passed else "FAILED"
        return f"{name:32s} {verdict:6s} {self.seconds * 1e3:7.0f} ms  {self.detail}"


def exact_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Bottom-right causal GQA attention in float32: query row ``i`` sees keys ``0 .. len_kv - len_q + i``."""
    query, key, value = (tensor.to(torch.float32) for tensor in (query, key, value))
    group = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(group, dim=1)
    value = value.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale
    offset = key.shape[0] - query.shape[0]
    rows = torch.arange(query.shape[0]).view(-1, 1)
    columns = torch.arange(key.shape[0]).view(1, -1)
    scores = scores.masked_fill(columns > rows + offset, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), value)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.detach().to("cpu", torch.float64).flatten()
    right = expected.to(torch.float64).flatten()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-30))


def _first_line(error: BaseException) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return f"{type(error).__name__}: {text[:_ERROR_DETAIL_CHARS]}"


def probe_backend(
    name: str,
    shape: LayerShape,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    *,
    output_rotation_folded: bool = False,
    dense_attention_api: str | None = None,
    prefill: bool = True,
    decode: bool = True,
    seed: int = 0,
) -> ProbeResult:
    """Write, prefill and decode :data:`PROBE_TOKENS` synthetic tokens through one backend.

    Never raises for what the backend does: construction failures (a library
    that does not load), launch failures (an aclnn interface the SoC refuses) and
    wrong numbers all come back as a failed result with the reason.
    """
    started = time.perf_counter()
    reference_rotation = None
    try:
        geometry = CacheGeometry(
            num_layers=1,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=block_size,
            max_seq_len=max(block_size, PROBE_TOKENS),
        )
        backend = build_backend(
            name, geometry, shape, device, dtype, output_rotation_folded, dense_attention_api=dense_attention_api
        )
        generator = torch.Generator().manual_seed(seed)
        heads, kv_heads, head_size = shape.num_heads, shape.num_kv_heads, shape.head_size
        query = torch.randn(PROBE_TOKENS, heads, head_size, generator=generator).to(dtype)
        key = torch.randn(PROBE_TOKENS, kv_heads, head_size, generator=generator).to(dtype)
        value = torch.randn(PROBE_TOKENS, kv_heads, head_size, generator=generator).to(dtype)
        backend.write_kv(0, key.to(device), value.to(device), backend.cache.slot_mapping(0, PROBE_TOKENS))

        if output_rotation_folded:
            # A folded o_proj expects O Pi, so that is what the backend returns.
            rotation = turboquant_rotation()
            signs = rotation.turboquant_pi_signs(head_size, torch.device("cpu"))

            def reference_rotation(tensor):
                return rotation.apply_pi(tensor, signs)

        cosines = []
        if prefill:
            out = torch.empty(PROBE_TOKENS, heads, head_size, dtype=dtype, device=device)
            backend.prefill_chunk(0, query.to(device), PROBE_TOKENS, out)
            expected = exact_attention(query, key, value, shape.scale)
            cosines.append(("prefill", out, expected))
        if decode:
            last = query[-1:]
            out = torch.empty(1, heads, head_size, dtype=dtype, device=device)
            lengths = torch.full((1,), PROBE_TOKENS, dtype=torch.int32, device=device)
            backend.decode(0, last.to(device), lengths, out)
            cosines.append(("decode", out, exact_attention(last, key, value, shape.scale)))
        # .cpu() in _cosine waits for the device, so an asynchronous kernel fault surfaces here too.
        measured = [
            (step, _cosine(out, expected if reference_rotation is None else reference_rotation(expected)))
            for step, out, expected in cosines
        ]
    except Exception as error:  # the probe exists to turn any failure into a verdict
        return ProbeResult(name, dense_attention_api, False, time.perf_counter() - started, _first_line(error))

    floor = DENSE_PROBE_MIN_COSINE if name in DENSE_BACKENDS else QUANTIZED_PROBE_MIN_COSINE
    passed = all(value > floor for _, value in measured)
    detail = ", ".join(f"{step} cos {value:.5f}" for step, value in measured) + f" (floor {floor})"
    return ProbeResult(name, dense_attention_api, passed, time.perf_counter() - started, detail)


@dataclass(frozen=True)
class DenseSelection:
    """Whether an unquantised path works here, through which backend and entry point, and every probe run."""

    passed: bool
    api: str | None
    results: list[ProbeResult]
    #: The backend that passed. ``None`` until :func:`select_dense_backend`
    #: chooses between them; :func:`select_dense_api` was asked about one.
    backend: str | None = None


def select_dense_api(
    name: str,
    shape: LayerShape,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    *,
    role: str,
    output_rotation_folded: bool = False,
) -> DenseSelection:
    """The first torch_npu entry point that passes the probe.

    ``role`` is ``"prefill"`` (the dense_staging pool, which only prefills) or
    ``"decode"`` (a native_v5 run, which does both). A backend other than
    ``native_v5`` has no entry point to choose and is probed once, with ``api``
    left ``None``.
    """
    decode = role == "decode"
    if name != NativeV5Backend.name:
        result = probe_backend(
            name, shape, device, dtype, block_size, output_rotation_folded=output_rotation_folded, decode=decode
        )
        return DenseSelection(result.passed, None, [result], name if result.passed else None)
    try:
        candidates = available_dense_attention_apis(role)
    except ImportError as error:
        return DenseSelection(False, None, [ProbeResult(name, None, False, 0.0, _first_line(error))])
    if not candidates:
        missing = ProbeResult(name, None, False, 0.0, "this torch_npu exports none of the attention entry points")
        return DenseSelection(False, None, [missing])
    results = []
    for api in candidates:
        result = probe_backend(
            name,
            shape,
            device,
            dtype,
            block_size,
            output_rotation_folded=output_rotation_folded,
            dense_attention_api=api,
            decode=decode,
        )
        results.append(result)
        if result.passed:
            return DenseSelection(True, api, results, name)
    return DenseSelection(False, None, results)


def select_dense_backend(
    device: torch.device,
    shape: LayerShape,
    dtype: torch.dtype,
    block_size: int,
    *,
    role: str,
    output_rotation_folded: bool = False,
) -> DenseSelection:
    """The first unquantised backend that passes the probe, with its entry point.

    :func:`tq_longbench.ops.dense_backend_candidates` sets the order: the fused
    kernel first because it is the one worth measuring, ``cann_dense`` behind it
    because matmul and softmax are accepted where ``aclnnFusedInferAttentionScore``
    V1-V4 is not. On Ascend 950 that is the whole of op-plugin's FIA (``EZ9903``),
    so on a 950 this is what keeps an unquantised baseline in the run at all --
    every probe above it fails, and the results carry each refusal's reason.
    """
    results: list[ProbeResult] = []
    for name in dense_backend_candidates(device):
        selection = select_dense_api(
            name,
            shape,
            device,
            dtype,
            block_size,
            role=role,
            output_rotation_folded=output_rotation_folded,
        )
        results.extend(selection.results)
        if selection.passed:
            return DenseSelection(True, selection.api, results, name)
    return DenseSelection(False, None, results)
