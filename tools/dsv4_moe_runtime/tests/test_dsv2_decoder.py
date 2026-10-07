"""Schema, rotary convention and aclnn stream regression checks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ..draft_inference.backends import AclnnV5Backend
from ..inference.dsv2_lite import V2LiteDecoder, expected_shapes, rotate, yarn_tables

CHECKPOINT = Path("F:/AI/models/DeepSeek-V2-Lite-Chat")


def test_sequence_reset_preserves_cache_and_monotonic_policy_clock():
    decoder = object.__new__(V2LiteDecoder)
    decoder.position = 12
    decoder.completed_tokens = 42
    decoder.pool = object()
    pool = decoder.pool
    decoder.runtime = SimpleNamespace(synchronize_device=lambda: None)
    decoder.reset_sequence()
    assert decoder.position == 0
    assert decoder.completed_tokens == 42
    assert decoder.pool is pool


def test_aclnn_launch_uses_producer_stream():
    stream = SimpleNamespace(npu_stream=12345)
    backend = object.__new__(AclnnV5Backend)
    backend._runtime = SimpleNamespace(current_stream=lambda: stream)
    assert backend._stream() == 12345


def test_interleaved_rotation():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    assert torch.equal(rotate(x, torch.zeros(4), torch.ones(4)), torch.tensor([[[-2.0, -4.0, 1.0, 3.0]]]))


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="local checkpoint unavailable")
def test_schema_and_yarn_against_checkpoint_reference():
    # Extract only the independent rotary implementation, without importing
    # the checkpoint's obsolete transformers model dependencies.
    import ast
    import math

    config = json.loads((CHECKPOINT / "config.json").read_text())
    shapes = expected_shapes(config)
    assert len(shapes) == 5291
    assert shapes["model.layers.26.self_attn.kv_b_proj.weight"] == (4096, 512)
    assert shapes["model.layers.26.mlp.shared_experts.gate_proj.weight"] == (2816, 2048)
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        expected_shapes({**config, "num_experts_per_tok": 5})
    names = {
        "DeepseekV2RotaryEmbedding",
        "DeepseekV2YarnRotaryEmbedding",
        "yarn_find_correction_dim",
        "yarn_find_correction_range",
        "yarn_get_mscale",
        "yarn_linear_ramp_mask",
    }
    parsed = ast.parse((CHECKPOINT / "modeling_deepseek.py").read_text(encoding="utf-8"))
    subset = ast.Module(body=[node for node in parsed.body if getattr(node, "name", "") in names], type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(subset, "checkpoint_rotary_reference", "exec"), namespace)
    rope = config["rope_scaling"]
    reference = namespace["DeepseekV2YarnRotaryEmbedding"](
        64,
        max_position_embeddings=32,
        base=10000,
        scaling_factor=rope["factor"],
        **{
            name: rope[name]
            for name in ("original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
        },
    )
    cos, sin, scale = yarn_tables(config, 32, "cpu")
    torch.testing.assert_close(cos, reference.cos_cached.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(sin, reference.sin_cached.bfloat16(), rtol=0, atol=0)
    assert scale == pytest.approx(192**-0.5 * namespace["yarn_get_mscale"](40, 0.707) ** 2)
