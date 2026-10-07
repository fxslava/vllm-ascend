"""FP8 serialization, padded slot geometry and native CUDA expert execution."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..core.config import DSV2_LITE_GEOMETRY
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.sharded_safetensors import SafetensorsShardIndex, bind_sharded_expert_spans
from ..inference.layer_dispatch import MoELayerScratch, fp8_swiglu_into, quantize_activation_into
from ..quantize_fp8 import convert, quantize_tensor


def test_quantization_zero_and_nonfinite():
    quantized, scale = quantize_tensor(torch.zeros(16, 16, dtype=torch.bfloat16))
    assert torch.equal(quantized.float(), torch.zeros(16, 16))
    assert scale.item() == 1
    with pytest.raises(ValueError, match="non-finite"):
        quantize_tensor(torch.tensor([float("nan")]))


def test_slot_alignment_and_shared_padding():
    layout = ExpertTensorLayout.for_fp8(DSV2_LITE_GEOMETRY)
    assert layout.slot_num_bytes == 8_651_264
    assert layout.slot_num_bytes % 512 == 0
    assert all(spec.offset_bytes % 128 == 0 for spec in layout.specs)
    shared = ExpertTensorLayout.for_shared_expert(DSV2_LITE_GEOMETRY, 2)
    pool = StaticExpertSlotPool(DSV2_LITE_GEOMETRY, 6, layout, "cpu", shared_layout=shared, shared_layers=[1, 2])
    assert pool.slots_per_shared_expert == 4
    assert pool.shared_region(1).data_ptr() + shared.slot_num_bytes <= pool.shared_region(2).data_ptr()
    for views in (pool.param_views(0), pool.shared_param_views(1)):
        assert all(view.data_ptr() % 128 == 0 for view in views.values())


def test_streamed_conversion_and_resume(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    metadata = b'{"test": "unchanged"}\n'
    (source / "config.json").write_bytes(metadata)
    tensors = {"model.norm.weight": torch.ones(32, dtype=torch.bfloat16)}
    layout = ExpertTensorLayout.for_fp8(DSV2_LITE_GEOMETRY)
    for projection, shape in [("gate_proj", (1408, 2048)), ("up_proj", (1408, 2048)), ("down_proj", (2048, 1408))]:
        tensors[f"model.layers.1.mlp.experts.0.{projection}.weight"] = torch.ones(shape, dtype=torch.bfloat16)
    save_file(tensors, str(source / "model.safetensors"))
    report = convert(source, output, "cpu")
    assert report["quantized_tensors"] == 3
    assert (output / "config.json").read_bytes() == metadata
    index = SafetensorsShardIndex.from_directory(output)
    bound = bind_sharded_expert_spans(index, layout, [1], [0])
    assert len(bound[(1, 0)]) == 6
    assert index.total_bytes == report["tensor_bytes"]
    with safe_open(str(output / "model.safetensors"), framework="pt") as shard:
        assert torch.equal(shard.get_tensor("model.norm.weight"), tensors["model.norm.weight"])
    assert convert(source, output, "cpu", resume=True) == report
    with pytest.raises(FileExistsError):
        convert(source, output, "cpu")
    manifest = json.loads((output / "runtime_fp8.json").read_text())
    assert manifest["format"] == "routed_e4m3fn_tensorwise"
    span = index.span("model.layers.1.mlp.experts.0.down_proj.weight")
    with span.shard.open("r+b") as damaged:
        damaged.seek(span.begin)
        damaged.write(bytes(16))
    with pytest.raises(ValueError, match="bytes mismatch"):
        convert(source, output, "cpu", resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_fp8_matches_dequantized_operands_and_preserves_buffers():
    layout = ExpertTensorLayout.for_fp8(DSV2_LITE_GEOMETRY)
    scratch = MoELayerScratch(DSV2_LITE_GEOMETRY, layout, None, "cuda:0")
    arena = torch.zeros(layout.slot_num_bytes, dtype=torch.uint8, device="cuda:0")
    views = layout.slice_region_views(arena, 0)
    generator = torch.Generator().manual_seed(7)
    for name in ("w1", "w2", "w3"):
        weight = torch.randn(layout.spec_for(name).logical_shape, generator=generator) * 0.01
        quantized, scale = quantize_tensor(weight)
        views[name].copy_(quantized.view(torch.uint8))
        views[name + "_scale"].copy_(scale.view(torch.uint8))
    x = torch.randn(1, 2048, device="cuda:0", dtype=torch.bfloat16)
    before = scratch.fingerprint()
    weight_bytes_before = arena.clone()
    result = fp8_swiglu_into(x, views, scratch).clone()
    assert torch.equal(arena, weight_bytes_before)
    assert torch.isfinite(result).all()
    quantize_activation_into(x, scratch.fp8_input, scratch.fp8_work, scratch.fp8_input_scale)
    activation = scratch.fp8_input.float() * scratch.fp8_input_scale
    weights = {
        name: views[name].view(torch.float8_e4m3fn).float() * views[name + "_scale"].view(torch.float32)
        for name in ("w1", "w2", "w3")
    }
    gate = (activation @ weights["w1"].t()).bfloat16()
    up = (activation @ weights["w3"].t()).bfloat16()
    hidden = torch.nn.functional.silu(gate) * up
    quantize_activation_into(hidden, scratch.fp8_down_input, scratch.fp8_down_work, scratch.fp8_down_scale)
    expected = ((scratch.fp8_down_input.float() * scratch.fp8_down_scale) @ weights["w2"].t()).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.002)
    assert before == scratch.fingerprint()
