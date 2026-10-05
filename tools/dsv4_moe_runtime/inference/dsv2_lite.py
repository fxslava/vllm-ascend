"""Real-weight, single-sequence V2-Lite validation over the static expert pool.

Weights and expanded KV cache have fixed addresses. Attention is an eager
reference path and allocates temporaries; this is not a zero-allocation decoder.
The V4 FP4/clamped custom kernel is deliberately incompatible with V2 BF16.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from ..core.config import DSV2_LITE_GEOMETRY
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.runtime import make_runtime
from ..hardware.sharded_safetensors import SafetensorsShardIndex, ShardedSafetensorsExpertSource
from .layer_dispatch import MoELayerScratch, execute_moe_layer


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Complete checkpoint schema, including every routed expert."""
    required = {
        "hidden_size": 2048,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "intermediate_size": 10944,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "kv_lora_rank": 512,
        "q_lora_rank": None,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 102400,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "scoring_func": "softmax",
        "topk_method": "greedy",
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
        "n_group": 1,
        "topk_group": 1,
        "hidden_act": "silu",
        "attention_bias": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported V2-Lite configuration: {key}={config.get(key)!r}, expected {value!r}")
    if config["rope_scaling"]["type"] != "yarn":
        raise ValueError("V2-Lite validation requires YaRN")
    shapes = {
        "model.embed_tokens.weight": (102400, 2048),
        "lm_head.weight": (102400, 2048),
        "model.norm.weight": (2048,),
    }
    for layer in range(27):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + name + ".weight"] = (2048,)
        for name, shape in {
            "q_proj": (3072, 2048),
            "kv_a_proj_with_mqa": (576, 2048),
            "kv_a_layernorm": (512,),
            "kv_b_proj": (4096, 512),
            "o_proj": (2048, 2048),
        }.items():
            shapes[prefix + "self_attn." + name + ".weight"] = shape
        modules = (
            [("mlp", 10944)]
            if layer == 0
            else [("mlp.shared_experts", 2816)] + [(f"mlp.experts.{expert}", 1408) for expert in range(64)]
        )
        if layer:
            shapes[prefix + "mlp.gate.weight"] = (64, 2048)
        for module, width in modules:
            for name, shape in {
                "gate_proj": (width, 2048),
                "up_proj": (width, 2048),
                "down_proj": (2048, width),
            }.items():
                shapes[prefix + module + "." + name + ".weight"] = shape
    return shapes


def audit_checkpoint(directory: Path) -> tuple[dict, SafetensorsShardIndex, dict]:
    config = json.loads((directory / "config.json").read_text())
    shapes = expected_shapes(config)
    index = SafetensorsShardIndex.from_directory(directory)
    if set(index.tensor_names) != set(shapes):
        raise ValueError("checkpoint keys differ from the complete V2-Lite schema")
    for name, shape in shapes.items():
        span = index.span(name)
        if span.shape != shape or span.dtype != "BF16" or span.num_bytes != math.prod(shape) * 2:
            raise ValueError(f"invalid tensor {name}: {span}")
    routed = sum(index.span(name).num_bytes for name in shapes if ".experts." in name)
    shared = sum(index.span(name).num_bytes for name in shapes if ".shared_experts." in name)
    return (
        config,
        index,
        {
            "tensors": len(shapes),
            "parameters": index.total_bytes // 2,
            "checkpoint_bytes": index.total_bytes,
            "routed_bytes": routed,
            "shared_bytes": shared,
            "backbone_bytes": index.total_bytes - routed - shared,
        },
    )


def yarn_tables(config: dict, capacity: int, device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Checkpoint's original YaRN formula and interleaved RoPE convention."""
    dim, base = config["qk_rope_head_dim"], config["rope_theta"]
    rope = config["rope_scaling"]
    factor = rope["factor"]

    def correction(rotations: float) -> float:
        return (
            dim * math.log(rope["original_max_position_embeddings"] / (rotations * 2 * math.pi)) / (2 * math.log(base))
        )

    low = max(math.floor(correction(rope["beta_fast"])), 0)
    high = min(math.ceil(correction(rope["beta_slow"])), dim - 1)
    ramp = ((torch.arange(dim // 2, device=device) - low) / (high - low or 0.001)).clamp(0, 1)
    extra = base ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    inverse = extra / factor * ramp + extra * (1 - ramp)
    phase = torch.outer(torch.arange(capacity, device=device).float(), inverse).repeat(1, 2)

    def mscale(value: float) -> float:
        return 1 + 0.1 * value * math.log(factor) if factor > 1 else 1

    amplitude = mscale(rope["mscale"]) / mscale(rope["mscale_all_dim"])
    scale = (config["qk_nope_head_dim"] + dim) ** -0.5 * mscale(rope["mscale_all_dim"]) ** 2
    return (phase.cos() * amplitude).bfloat16(), (phase.sin() * amplitude).bfloat16(), scale


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).transpose(-1, -2).reshape_as(x)
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class V2LiteDecoder:
    """Fixed weight arenas with bounded expert offload and static expanded KV."""

    @torch.inference_mode()
    def __init__(self, directory: Path, device: str = "cuda:0", slots: int = 64, capacity: int = 128):
        self.config, index, self.audit = audit_checkpoint(directory)
        if capacity < 1 or slots < DSV2_LITE_GEOMETRY.top_k:
            raise ValueError("capacity must be positive and slots must hold top-k")
        self.runtime = make_runtime(device)
        self.capacity = capacity
        self.position = 0
        self.completed_tokens = 0
        layout = ExpertTensorLayout.for_dense_bf16(DSV2_LITE_GEOMETRY)
        shared = ExpertTensorLayout.for_shared_expert(DSV2_LITE_GEOMETRY, 2)
        kv_bytes = 27 * 16 * capacity * (192 + 128) * 2
        needed = self.audit["backbone_bytes"] + self.audit["shared_bytes"] + slots * layout.slot_num_bytes + kv_bytes
        free = self.runtime.device_free_memory()
        reserve = 512 * 1024**2
        if free is not None and needed + reserve > free:
            raise MemoryError(f"static plan {needed} bytes + {reserve} reserve exceeds free {free}")
        self.pool = StaticExpertSlotPool(
            DSV2_LITE_GEOMETRY, slots, layout, device, shared_layout=shared, shared_layers=range(1, 27)
        )
        self.source = ShardedSafetensorsExpertSource(
            self.runtime, index, layout, range(1, 27), range(64), shared_layout=shared
        )
        self.pool.fill_shared_experts(self.source)
        names = [name for name in index.tensor_names if ".experts." not in name and ".shared_experts." not in name]
        self.backbone = torch.empty(self.audit["backbone_bytes"] // 2, dtype=torch.bfloat16, device=device)
        self.weights = {}
        offset = 0
        for name in names:
            span = index.span(name)
            count = math.prod(span.shape)
            target = self.backbone.narrow(0, offset, count).view(span.shape)
            with safe_open(str(span.shard), framework="pt", device="cpu") as shard:
                target.copy_(shard.get_tensor(name))
            self.weights[name] = target
            offset += count
        self.scratch = MoELayerScratch(DSV2_LITE_GEOMETRY, layout, shared, device)
        self.keys = torch.empty(27, 16, capacity, 192, device=device, dtype=torch.bfloat16)
        self.values = torch.empty(27, 16, capacity, 128, device=device, dtype=torch.bfloat16)
        self.cos, self.sin, self.scale = yarn_tables(self.config, capacity, device)
        self.runtime.synchronize_device()
        self.fingerprint = self.pointers()
        self.static_bytes = needed

    def pointers(self) -> tuple[int, ...]:
        return (
            self.backbone.data_ptr(),
            self.pool.slot_arena.data_ptr(),
            self.keys.data_ptr(),
            self.values.data_ptr(),
            *self.scratch.fingerprint(),
        )

    def norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        weight = self.weights[name]
        if self.runtime.device.startswith("npu"):
            # torch_npu owns the aclnn RMSNorm binding and current stream.
            import torch_npu

            return torch_npu.npu_rms_norm(x, weight, epsilon=self.config["rms_norm_eps"])[0]
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.config["rms_norm_eps"])).to(
            x.dtype
        ) * weight

    @torch.inference_mode()
    def step(self, token: int) -> torch.Tensor:
        if self.position >= self.capacity:
            raise ValueError("static KV capacity exhausted")
        if not 0 <= token < self.config["vocab_size"]:
            raise ValueError("token outside vocabulary")
        x = self.weights["model.embed_tokens.weight"][token : token + 1]
        position = self.position
        for layer in range(27):
            prefix = f"model.layers.{layer}."
            normalized = self.norm(x, prefix + "input_layernorm.weight")
            attn = prefix + "self_attn."
            q = F.linear(normalized, self.weights[attn + "q_proj.weight"]).view(16, 1, 192)
            compressed = F.linear(normalized, self.weights[attn + "kv_a_proj_with_mqa.weight"])
            kv = F.linear(
                self.norm(compressed[:, :512], attn + "kv_a_layernorm.weight"), self.weights[attn + "kv_b_proj.weight"]
            ).view(16, 1, 256)
            query = torch.cat((q[..., :128], rotate(q[..., 128:], self.cos[position], self.sin[position])), -1)
            key_rope = rotate(compressed[:, 512:].view(1, 1, 64), self.cos[position], self.sin[position])
            self.keys[layer, :, position : position + 1, :128].copy_(kv[..., :128])
            self.keys[layer, :, position : position + 1, 128:].copy_(key_rope.expand(16, 1, 64))
            self.values[layer, :, position : position + 1].copy_(kv[..., 128:])
            scores = torch.matmul(query, self.keys[layer, :, : position + 1].transpose(-1, -2)) * self.scale
            probs = scores.softmax(-1, dtype=torch.float32).to(x.dtype)
            attended = torch.matmul(probs, self.values[layer, :, : position + 1]).reshape(1, 2048)
            x = x + F.linear(attended, self.weights[attn + "o_proj.weight"])
            normalized = self.norm(x, prefix + "post_attention_layernorm.weight")
            if layer == 0:
                gate = F.linear(normalized, self.weights[prefix + "mlp.gate_proj.weight"])
                up = F.linear(normalized, self.weights[prefix + "mlp.up_proj.weight"])
                output = F.linear(F.silu(gate) * up, self.weights[prefix + "mlp.down_proj.weight"])
            else:
                output = execute_moe_layer(
                    normalized, layer, self.weights[prefix + "mlp.gate.weight"], self.pool, self.source, self.scratch
                ).output
            x = x + output
        logits = F.linear(self.norm(x, "model.norm.weight"), self.weights["lm_head.weight"])
        self.pool.advance_generation(self.completed_tokens)
        self.completed_tokens += 1
        self.position += 1
        if self.pointers() != self.fingerprint:
            raise RuntimeError("static arena pointers changed")
        return logits

    def reset_sequence(self) -> None:
        """Reset KV visibility while preserving expert residency and policy age."""
        self.runtime.synchronize_device()
        self.position = 0

    def close(self) -> None:
        self.runtime.synchronize_device()
        self.source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        _, _, audit = audit_checkpoint(args.weights_dir)
        args.report.write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2))
        return
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.weights_dir, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True
    )
    if hasattr(prompt, "keys"):
        prompt = prompt["input_ids"]
    started = time.perf_counter()
    decoder = V2LiteDecoder(args.weights_dir, args.device, args.slots, len(prompt) + args.max_new_tokens)
    load_seconds = time.perf_counter() - started
    try:
        latencies, generated = [], []
        started = time.perf_counter()
        for token in prompt:
            logits = decoder.step(token)
        decoder.runtime.synchronize_device()
        prefill_seconds = time.perf_counter() - started
        finite = bool(torch.isfinite(logits).all().item())
        for index in range(args.max_new_tokens):
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == decoder.config["eos_token_id"] or index == args.max_new_tokens - 1:
                break
            started = time.perf_counter()
            logits = decoder.step(token)
            decoder.runtime.synchronize_device()
            latencies.append(time.perf_counter() - started)
            finite = finite and bool(torch.isfinite(logits).all().item())
        report = {
            **decoder.audit,
            "device": args.device,
            "load_seconds": load_seconds,
            "prompt_ids": prompt,
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": latencies,
            "static_weight_and_kv_bytes": decoder.static_bytes,
            "expert_arena_bytes": decoder.pool.slot_arena.numel(),
            "allocated_bytes": decoder.runtime.memory_allocated(),
            "reserved_bytes": decoder.runtime.memory_reserved(),
            "finite_logits": finite,
            "arena_pointers_stable": decoder.pointers() == decoder.fingerprint,
            "expert_loads": decoder.pool.stats.loads,
            "expert_hits": decoder.pool.stats.hits,
            "bytes_streamed": decoder.source.bytes_streamed,
            "execution": "eager attention + static BF16 expert dispatch; custom V4 FP4 kernel disabled",
        }
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not finite:
            raise RuntimeError("non-finite generation logits")
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
