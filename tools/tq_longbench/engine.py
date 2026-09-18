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
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tq_longbench import glm4
from tq_longbench._ascend import assert_no_vllm_imported
from tq_longbench.kv_cache import CacheGeometry, dense_equivalent_bytes
from tq_longbench.layers import (
    CausalLM,
    ForwardBatch,
    ModelShape,
    fold_output_projection_weight,
    fold_output_rotation,
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

_MEGABYTE = 1024 * 1024

_O_PROJ_WEIGHT_SUFFIX = "self_attn.o_proj.weight"

#: ``(checkpoint name, tensor) -> [(harness parameter, tensor), ...]``; empty to skip.
CheckpointMapper = Callable[[str, torch.Tensor], list[tuple[str, torch.Tensor]]]


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

    def __post_init__(self) -> None:
        if self.prefill_mode not in PREFILL_MODES:
            raise ValueError(f"unknown prefill mode {self.prefill_mode!r}; choose one of {list(PREFILL_MODES)}")
        if self.backend in DENSE_BACKENDS and self.prefill_mode != "dense_staging":
            raise ValueError(
                f"the {self.backend} baseline decodes out of the unquantised cache, so it needs --prefill-mode "
                "dense_staging; batched_decode never allocates one."
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.dense_staging_backend is not None and self.dense_staging_backend not in DENSE_BACKENDS:
            raise ValueError(
                f"dense_staging_backend {self.dense_staging_backend!r} does not decode out of an unquantised "
                f"cache; choose one of {sorted(DENSE_BACKENDS)}"
            )


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

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def kv_saved_mb(self) -> float:
        return self.dense_equivalent_mb - self.kv_cache_mb

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
            "dense_equivalent_mb": round(self.dense_equivalent_mb, 2),
            "kv_saved_mb": round(self.kv_saved_mb, 2),
            "device_allocated_mb": round(self.device_allocated_mb, 2),
            "device_peak_mb": round(self.device_peak_mb, 2),
            "device_reserved_mb": round(self.device_reserved_mb, 2),
            "stopped_on_token": self.stopped_on_token,
            "hit_token_budget": self.stopped_on_token is None,
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
        if config.fold_output_rotation:
            require_foldable(self.shape)
        if not config.random_weights:
            # Folds each o_proj as its shard tensor is ingested.
            self.load_weights()
        elif config.fold_output_rotation:
            fold_output_rotation(self.model)

    # ---------------------------------------------------------------- config

    def _config_path(self) -> Path:
        return Path(self.config.model_path) / "config.json"

    def _hf_config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(self.config.model_path, trust_remote_code=True)

    def _raw_config(self) -> dict:
        return glm4.read_config(self.config.model_path)

    def _is_glm4(self) -> bool:
        return not self.config.random_weights and glm4.is_glm4_config(self._raw_config())

    def _load_shape(self) -> ModelShape:
        if self._is_glm4():
            # Read off config.json directly: the chatglm config class is remote
            # code, and nothing it computes is needed that the JSON does not say.
            return glm4.glm4_model_shape(self._raw_config(), self.config.attn_output_gate)
        return ModelShape.from_hf_config(self._hf_config(), self.config.attn_output_gate, **self._checkpoint_features())

    def _checkpoint_mapper(self) -> CheckpointMapper:
        """How this checkpoint's tensor names land on the harness's modules."""
        if self._is_glm4():
            shape = self.shape
            return lambda name, tensor: glm4.glm4_checkpoint_targets(name, tensor, shape)

        def one_to_one(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
            target = _map_checkpoint_name(name)
            return [] if target is None else [(target, tensor)]

        return one_to_one

    def _tensor_names(self) -> list[str]:
        """Every tensor name in the checkpoint, read from the index or the shard headers.

        Names only -- ``safe_open`` gives them without reading a byte of tensor
        data, so this costs nothing next to the load that follows.
        """
        root = Path(self.config.model_path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            return list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])

        from safetensors import safe_open

        names: list[str] = []
        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                names.extend(handle.keys())
        return names

    def _checkpoint_features(self) -> dict:
        """What the config does not say: whether q/k are normed, and whether qkv has a bias.

        Read off the tensor names rather than the config, because neither is
        reliably declared. Qwen2.5 carries q/k/v biases with no ``attention_bias``
        key, and ``model_type`` alone does not separate a Qwen3 checkpoint (which
        norms each head) from a Qwen2 one (which does not). Getting either wrong
        builds a model whose parameters and the checkpoint's do not correspond,
        which :meth:`load_weights` then refuses -- loudly, rather than by
        normalising with a weight of 1 that was never loaded.
        """
        if self.config.random_weights:
            return {}
        names = self._tensor_names()
        return {
            "qk_norm": any(name.endswith("self_attn.q_norm.weight") for name in names),
            "qkv_bias": any(name.endswith("self_attn.q_proj.bias") for name in names),
        }

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

        destinations = dict(self.model.named_parameters())
        mapper = self._checkpoint_mapper()
        fold = self.config.fold_output_rotation
        filled: set[str] = set()
        folded = 0
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
                    if fold and target.endswith(_O_PROJ_WEIGHT_SUFFIX):
                        piece = fold_output_projection_weight(piece, self.shape.head_size)
                        folded += 1
                    with torch.no_grad():
                        parameter.copy_(piece.to(device=self.device, dtype=parameter.dtype))
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
        return len(filled)

    # ---------------------------------------------------------------- memory

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

        for _ in range(limit):
            next_token = int(torch.argmax(logits[-1]))
            if not produced:
                # argmax syncs on its own (the id crosses to the host), so this
                # boundary is the token existing, not the launch being queued.
                metrics.ttft_seconds = time.perf_counter() - started
            produced.append(next_token)
            if stop_ids and next_token in stop_ids:
                metrics.stopped_on_token = next_token
                break
            if position + 1 > self.config.max_seq_len:
                break
            step = torch.tensor([next_token], dtype=torch.int64, device=self.device)
            batch = self._batch(position, 1, is_decode=True)
            _synchronize(self.device)
            step_started = time.perf_counter()
            logits = self.model(step, batch, self.write_backends, self.decode_backend)
            _synchronize(self.device)
            metrics.decode_step_us.append((time.perf_counter() - step_started) * 1e6)
            position += 1
        metrics.generated_tokens = len(produced)
        for name, value in device_memory_mb(self.device).items():
            setattr(metrics, name, value)
        return produced, metrics

    def _memory_metrics(self) -> dict:
        report = self.memory_report()
        return {"kv_cache_mb": report["kv_cache_mb"], "dense_equivalent_mb": report["dense_equivalent_mb"]}


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
