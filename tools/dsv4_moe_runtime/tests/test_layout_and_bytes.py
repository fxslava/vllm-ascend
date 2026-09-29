"""Exact slot byte-accounting verification against the real checkpoint metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.config import EXPERT_PARAM_NAMES, DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout

DEEPSEEK_V4_MODEL_DIR = Path(r"C:\DeepSeekV4")
HF_CONFIG_FILENAME = "config.json"
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"

EXPECTED_SLOT_NUM_BYTES = 13_369_344  # 12.75 MiB = 3 x (17 * 2**18)
EXPECTED_PACKED_PROJECTION_BYTES = 4_194_304  # [2048,2048] or [4096,1024] packed fp4
EXPECTED_SCALE_PROJECTION_BYTES = 262_144  # [2048,128] or [4096,64] e8m0
EXPECTED_INSTANCE_COUNT = 11_264  # (43 decoder blocks + 1 MTP block) x 256 experts
EXPECTED_EXPERT_TENSOR_COUNT = 67_584  # instances x {w1,w2,w3} x {weight,scale}
EXPECTED_TOTAL_EXPERT_BYTES = 150_592_290_816  # instances x slot bytes
EXPECTED_NON_EXPERT_BYTES = 9_017_195_080  # index total minus routed experts
EXPECTED_INDEX_TOTAL_SIZE = 159_609_485_896
EXPECTED_HASH_LAYERS = (0, 1, 2)
EXPERT_INDEX_TENSOR_PATTERN = re.compile(
    r"(?:layers\.(\d+)|mtp\.(\d+))\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)"
)


def test_expert_byte_size(full_layout: ExpertTensorLayout) -> None:
    assert full_layout.slot_num_bytes == EXPECTED_SLOT_NUM_BYTES
    assert full_layout.slot_num_bytes == 12.75 * 1024 * 1024  # 12.75 MiB, exact

    for name in EXPERT_PARAM_NAMES:
        packed = full_layout.spec_for(name)
        scale = full_layout.spec_for(name + "_scale")
        assert packed.num_bytes == EXPECTED_PACKED_PROJECTION_BYTES, name
        assert scale.num_bytes == EXPECTED_SCALE_PROJECTION_BYTES, name
        assert packed.num_bytes + scale.num_bytes == 4_456_448, name
    per_projection_total = EXPECTED_PACKED_PROJECTION_BYTES + EXPECTED_SCALE_PROJECTION_BYTES
    assert per_projection_total * 3 == EXPECTED_SLOT_NUM_BYTES

    # Regions are 128-byte aligned with no hidden padding beyond alignment.
    for spec in full_layout.specs:
        assert spec.offset_bytes % 128 == 0
        assert spec.offset_bytes + spec.num_bytes <= full_layout.slot_num_bytes
    assert full_layout.specs[-1].offset_bytes + full_layout.specs[-1].num_bytes == full_layout.slot_num_bytes

    # --- Cross-check against the real checkpoint metadata, when available. ---
    config_path = DEEPSEEK_V4_MODEL_DIR / HF_CONFIG_FILENAME
    if config_path.exists():
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert hf_config["hidden_size"] == DeepSeekV4MoEConfig().hidden_size
        assert hf_config["moe_intermediate_size"] == DeepSeekV4MoEConfig().moe_intermediate_size
        assert hf_config["n_routed_experts"] == DeepSeekV4MoEConfig().num_routed_experts
        assert hf_config["num_experts_per_tok"] == DeepSeekV4MoEConfig().top_k
        assert hf_config["num_hidden_layers"] == DeepSeekV4MoEConfig().num_layers
        assert hf_config["num_hash_layers"] == DeepSeekV4MoEConfig().num_hash_layers
        assert hf_config["expert_dtype"] == "fp4"

    index_path = DEEPSEEK_V4_MODEL_DIR / SAFETENSORS_INDEX_FILENAME
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        expert_tensors = [name for name in weight_map if EXPERT_INDEX_TENSOR_PATTERN.fullmatch(name)]
        assert len(expert_tensors) == EXPECTED_EXPERT_TENSOR_COUNT
        instances = {name[: -len(".w1.weight")] for name in expert_tensors if name.endswith(".w1.weight")}
        assert len(instances) == EXPECTED_INSTANCE_COUNT
        total_expert_bytes = len(instances) * full_layout.slot_num_bytes
        assert total_expert_bytes == EXPECTED_TOTAL_EXPERT_BYTES
        total_size = index["metadata"]["total_size"]
        assert total_size == EXPECTED_INDEX_TOTAL_SIZE
        assert total_size - total_expert_bytes == EXPECTED_NON_EXPERT_BYTES
        hash_layers = {
            int(match.group(1))
            for name in weight_map
            if (match := re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.tid2eid", name))
        }
        assert tuple(sorted(hash_layers)) == EXPECTED_HASH_LAYERS
        bias_layers = {
            int(match.group(1))
            for name in weight_map
            if (match := re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias", name))
        }
        assert tuple(sorted(bias_layers)) == tuple(
            range(max(EXPECTED_HASH_LAYERS) + 1, DeepSeekV4MoEConfig().num_layers)
        )
