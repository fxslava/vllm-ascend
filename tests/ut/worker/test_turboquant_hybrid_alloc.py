# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant three-plane KV cache in the model runner.

The case under test is the one that used to fail: a hybrid model (full
attention plus Gated DeltaNet, e.g. Qwen3.5) where the TurboQuant layers must
get their own three planes while the recurrent layers keep the Mamba page and
the Mamba allocator untouched.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_PLANES,
    turboquant_packed_plane_bytes,
    turboquant_page_size_bytes,
    turboquant_scale_cache_shape,
    turboquant_scale_plane_bytes,
)
from vllm_ascend.core.kv_cache_interface import AscendTurboQuantAttentionSpec, turboquant_cache_spec
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

ATTN = "model.layers.1.self_attn.attn"
MAMBA = "model.layers.0.linear_attn"
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64
NUM_BLOCKS = 3


def _dense_spec(**overrides) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        **overrides,
    )


def _specs(padded: bool = True) -> dict[str, object]:
    dense = _dense_spec()
    common_page = dense.page_size_bytes
    attn = turboquant_cache_spec(dense, page_size_padded=common_page if padded else None)
    # The hybrid "align" contract patch_mamba_config sets up: the Mamba page is
    # padded to the attention page so every KV cache group shares one page size.
    mamba = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((2, 2), (1, 2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=common_page,
    )
    assert mamba.page_size_bytes == common_page == attn.page_size_bytes
    return {ATTN: attn, MAMBA: mamba}


def _make_config(specs) -> KVCacheConfig:
    layer_stride = NUM_BLOCKS * specs[ATTN].page_size_bytes
    backing_size = 2 * layer_stride
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_size,
                layers=[ATTN],
                offset=0,
                layer_stride=layer_stride,
                block_stride=specs[ATTN].page_size_bytes,
            ),
            KVCacheTensor(
                size=backing_size,
                layers=[MAMBA],
                offset=layer_stride,
                layer_stride=layer_stride,
                block_stride=specs[MAMBA].page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[ATTN], kv_cache_spec=specs[ATTN]),
            KVCacheGroupSpec(layer_names=[MAMBA], kv_cache_spec=specs[MAMBA]),
        ],
    )


def _make_runner(specs, is_hybrid: bool = True) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_layer_compact=True, is_block_compact=True),
        ),
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(is_hybrid=is_hybrid),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        quant_config=None,
    )
    runner.model_config = SimpleNamespace(is_hybrid=is_hybrid, hf_text_config=SimpleNamespace(model_type="qwen3_5"))
    runner.compilation_config = runner.vllm_config.compilation_config
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1),
        xlite_graph_config=SimpleNamespace(enabled=False),
    )
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    # use_dcp is a read-only property over dcp_size.
    runner.dcp_size = 1
    runner.use_sparse = False
    runner.use_compress = False
    runner.use_hybrid_blocks = True
    runner.sparse_kv_offload_enabled = False
    runner.tp_rank = 0
    runner.kernel_block_sizes = [BLOCK_SIZE]
    runner._get_layer_kv_cache_specs = lambda _config: dict(specs)
    runner._get_attention_kv_cache_dims = lambda *_args: (HEAD_SIZE, HEAD_SIZE)
    # Only the attention group reaches the reshape loop; Mamba state is viewed
    # by its own path.
    groups = [
        SimpleNamespace(
            backend=None,
            kv_cache_spec=specs[ATTN],
            layer_names=[ATTN],
            kv_cache_group_id=0,
        )
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
    runner._allocate_int8_cache_tensor = lambda size, _alignment: torch.zeros(size, dtype=torch.int8)
    return runner


def test_spec_budgets_the_three_planes_and_shrinks_the_page():
    dense = _dense_spec()
    attn = turboquant_cache_spec(dense)

    assert isinstance(attn, AscendTurboQuantAttentionSpec)
    assert (attn.pack_factor, attn.num_planes) == (TURBOQUANT_PACK_FACTOR, TURBOQUANT_PLANES)
    assert attn.packed_bytes == HEAD_SIZE // TURBOQUANT_PACK_FACTOR
    assert attn.payload_bytes == TURBOQUANT_PLANES * turboquant_packed_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert attn.scale_plane_bytes == turboquant_scale_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS)
    assert attn.real_page_size_bytes == turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert attn.page_size_bytes == attn.real_page_size_bytes
    # The whole point: the packed page is a fraction of the dense one, and the
    # scale plane is inside it rather than beside it.
    assert attn.page_size_bytes * 3 < dense.page_size_bytes
    # Every scheduler-visible attribute is carried over untouched.
    assert (attn.block_size, attn.num_kv_heads, attn.head_size, attn.dtype) == (
        dense.block_size,
        dense.num_kv_heads,
        dense.head_size,
        dense.dtype,
    )


def test_spec_pads_the_page_to_a_common_size():
    dense = _dense_spec()
    attn = turboquant_cache_spec(dense, page_size_padded=dense.page_size_bytes)

    assert attn.page_size_bytes == dense.page_size_bytes
    assert attn.real_page_size_bytes == turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="smaller than its"):
        turboquant_cache_spec(dense, page_size_padded=64)
    with pytest.raises(ValueError, match="GM bursts"):
        turboquant_cache_spec(dense, page_size_padded=dense.page_size_bytes + 1)


def test_spec_merge_keeps_the_packing():
    attn = turboquant_cache_spec(_dense_spec())
    merged = AscendTurboQuantAttentionSpec.merge([attn, attn])
    assert isinstance(merged, AscendTurboQuantAttentionSpec)
    assert (merged.pack_factor, merged.num_planes) == (attn.pack_factor, attn.num_planes)
    assert merged.page_size_bytes == attn.page_size_bytes


@patch.dict(os.environ, {"ENABLE_TURBOQUANT": "1"})
def test_conversion_pads_only_hybrid_models():
    dense = _dense_spec()

    hybrid = _make_runner(_specs(), is_hybrid=True)._as_turboquant_spec(ATTN, dense)
    assert isinstance(hybrid, AscendTurboQuantAttentionSpec)
    assert hybrid.page_size_bytes == dense.page_size_bytes

    plain = _make_runner(_specs(), is_hybrid=False)._as_turboquant_spec(ATTN, dense)
    assert isinstance(plain, AscendTurboQuantAttentionSpec)
    assert plain.page_size_bytes == plain.real_page_size_bytes


@patch.dict(os.environ, {"ENABLE_TURBOQUANT": "1"})
def test_conversion_leaves_other_layers_alone():
    runner = _make_runner(_specs())
    sliding = _dense_spec(sliding_window=8)
    assert runner._as_turboquant_spec(ATTN, sliding) is sliding
    mamba = _specs()[MAMBA]
    assert runner._as_turboquant_spec(MAMBA, mamba) is mamba


@patch.dict(os.environ, {"ENABLE_TURBOQUANT": "0"})
def test_conversion_is_inert_without_the_environment_switch():
    runner = _make_runner(_specs())
    dense = _dense_spec()
    assert runner._as_turboquant_spec(ATTN, dense) is dense


def test_hybrid_allocation_isolates_turboquant_from_the_mamba_page():
    specs = _specs()
    config = _make_config(specs)
    runner = _make_runner(specs)

    raw = runner._allocate_kv_cache_tensors(config)

    assert set(raw) == {ATTN, MAMBA}
    planes = raw[ATTN]
    assert isinstance(planes, tuple) and len(planes) == 3
    packed = NUM_BLOCKS * turboquant_packed_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    scales = NUM_BLOCKS * turboquant_scale_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS)
    assert [plane.numel() for plane in planes] == [packed, packed, scales]

    # The Mamba layer is allocated exactly as it was before TurboQuant: one
    # flat buffer of its own (padded) page budget.
    mamba_cache = raw[MAMBA]
    assert isinstance(mamba_cache, torch.Tensor)
    assert mamba_cache.numel() == NUM_BLOCKS * specs[MAMBA].page_size_bytes

    # Nothing overlaps: the planes live in the attention layer's own region of
    # the shared backing, and the padded tail of that region stays unused.
    layer_stride = NUM_BLOCKS * specs[ATTN].page_size_bytes
    region_start = planes[0].data_ptr()
    assert mamba_cache.data_ptr() == region_start + layer_stride
    assert planes[-1].data_ptr() + planes[-1].numel() <= region_start + layer_stride
    assert sum(plane.numel() for plane in planes) == NUM_BLOCKS * specs[ATTN].real_page_size_bytes

    for plane in planes:
        plane.fill_(7)
    assert int(mamba_cache.abs().sum()) == 0


def test_hybrid_reshape_views_the_three_planes():
    specs = _specs()
    config = _make_config(specs)
    runner = _make_runner(specs)

    raw = runner._allocate_kv_cache_tensors(config)
    kv_caches = runner._reshape_kv_cache_tensors(config, raw)

    key, value, scale = kv_caches[ATTN]
    packed_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
    assert tuple(key.shape) == tuple(value.shape) == packed_shape
    assert (key.dtype, value.dtype, scale.dtype) == (torch.int8, torch.int8, torch.float32)
    # Exactly what AscendTurboQuantBackendImpl._ensure_scale_cache demands
    # of the allocator's third plane.
    assert tuple(scale.shape) == turboquant_scale_cache_shape(key.shape[0], key.shape[1], key.shape[2])

    # The views address the carved bytes, not a copy.
    key[0, 0, 0, 0] = 9
    assert int(raw[ATTN][0][0]) == 9


def test_block_major_views_split_the_scale_plane_for_the_block_copier():
    specs = _specs()
    config = _make_config(specs)
    runner = _make_runner(specs)

    raw = runner._allocate_kv_cache_tensors(config)
    kv_caches = runner._reshape_kv_cache_tensors(config, raw)
    runner.kv_caches = [kv_caches[ATTN]]

    normalized = runner._block_major_kv_caches()
    assert len(normalized) == 1
    key, value, key_scale, value_scale = normalized[0]
    assert tuple(key_scale.shape) == tuple(value_scale.shape) == (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)
    # Every tensor the copier sees now indexes blocks on dimension 0.
    for tensor in (key, value, key_scale, value_scale):
        assert tensor.shape[0] == NUM_BLOCKS
    # And they are views, not copies.
    key_scale[1, 0, 0] = 1.5
    assert float(kv_caches[ATTN][2][0, 1, 0, 0]) == 1.5


def test_reshape_rejects_a_two_plane_allocation():
    specs = _specs()
    config = _make_config(specs)
    runner = _make_runner(specs)

    raw = runner._allocate_kv_cache_tensors(config)
    raw[ATTN] = raw[ATTN][:2]
    with pytest.raises(ValueError, match="three raw planes"):
        runner._reshape_kv_cache_tensors(config, raw)


def test_block_movers_move_the_scale_plane_with_the_codes():
    from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantBackend

    def _layer():
        key = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2, dtype=torch.int8)
        value = torch.zeros_like(key)
        scale = torch.zeros(
            turboquant_scale_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS),
            dtype=torch.float32,
        )
        key[0] = 1
        value[0] = 2
        scale[:, 0] = 0.25
        return key, value, scale

    mapping = torch.tensor([[0, 2]])

    key, value, scale = _layer()
    AscendTurboQuantBackend.copy_blocks([(key, value, scale)], mapping)
    assert int(key[2].min()) == 1
    assert int(value[2].min()) == 2
    # The scale plane is plane-major, so its block dimension is 1: both the K
    # and the V scales of page 0 must now be on page 2.
    assert float(scale[0, 2].min()) == 0.25
    assert float(scale[1, 2].min()) == 0.25

    src = _layer()
    dst = tuple(torch.zeros_like(tensor) for tensor in src)
    AscendTurboQuantBackend.swap_blocks(list(src), list(dst), mapping)
    assert int(dst[0][2].min()) == 1
    assert int(dst[1][2].min()) == 2
    assert float(dst[2][0, 2].min()) == 0.25
    assert float(dst[2][1, 2].min()) == 0.25
