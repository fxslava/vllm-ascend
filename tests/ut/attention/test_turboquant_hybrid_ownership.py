# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for recurrent writes into the packed hybrid cache.

Execute the actual runner/worker methods with minimal engine descriptors so
this reproducer also runs without vLLM, torch_npu or CANN. Only method loading
and engine setup are stubbed; allocation, carving, page views and budgeting
execute the checkout's Python code. The companion integration tests using
real vLLM specs live in tests/ut/worker/test_turboquant_hybrid_alloc.py.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.turboquant_codec import TurboQuantEmulator
from vllm_ascend.attention.turboquant_config import TurboQuantOpsConfig, TurboQuantValueLayout
from vllm_ascend.attention.turboquant_layout import turboquant_carve_planes, turboquant_plane_views
from vllm_ascend.attention.turboquant_metadata import AscendTurboQuantMetadataBuilder
from vllm_ascend.attention.turboquant_ops import TurboQuantOpsAdapter

NUM_BLOCKS = 4
NUM_KV_HEADS = 4
HEAD_SIZE = 256
KERNEL_BLOCK_SIZE = 128
MAMBA_BLOCK_SIZE = 65536
STATE_FLOATS = 8
ATTENTION_LAYER = "model.layers.3.self_attn.attn"
RECURRENT_LAYERS = tuple(f"model.layers.{index}.linear_attn" for index in range(3))
REPO = Path(__file__).resolve().parents[3]


class _AttentionSpec(SimpleNamespace):
    pass


class _FullAttentionSpec(_AttentionSpec):
    pass


class _TurboQuantSpec(_FullAttentionSpec):
    pass


class _MambaSpec(SimpleNamespace):
    pass


class _UniformTypeSpecs(SimpleNamespace):
    pass


class _CpuGpuBuffer:
    def __init__(self, *shape, dtype, device, pin_memory):
        self.cpu = torch.zeros(shape, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(shape, dtype=dtype)

    def copy_to_gpu(self, count):
        self.gpu[:count].copy_(self.cpu[:count])


def _load_class(filename, class_name, method_names, namespace):
    """Avoid importing engine/device dependencies unrelated to these methods."""
    source = REPO / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    assert {method.name for method in methods} == set(method_names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return type(class_name, (), {name: namespace[name] for name in method_names})


def _load_functions(filename, function_names, namespace):
    """Load production layout helpers without importing device dependencies."""
    source = REPO / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    assert {function.name for function in functions} == set(function_names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)


@pytest.fixture(params=[896, 1024, 2048])
def hybrid_pool(request):
    block_size = request.param
    dense_page_bytes = 2 * block_size * NUM_KV_HEADS * HEAD_SIZE * torch.bfloat16.itemsize
    specs = {
        ATTENTION_LAYER: _TurboQuantSpec(
            block_size=block_size,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            page_size_bytes=dense_page_bytes,
        ),
        **{
            name: _MambaSpec(block_size=MAMBA_BLOCK_SIZE, page_size_bytes=dense_page_bytes) for name in RECURRENT_LAYERS
        },
    }
    groups = [SimpleNamespace(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()]
    # All four groups overlap at descriptor offset zero, as the standardized
    # shared pool permits. Their token block sizes match the live server log.
    config = SimpleNamespace(
        num_blocks=NUM_BLOCKS,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=NUM_BLOCKS * dense_page_bytes,
                layers=[name],
                offset=0,
                layer_stride=NUM_BLOCKS * dense_page_bytes,
                block_stride=dense_page_bytes,
            )
            for name in specs
        ],
    )
    namespace = {
        "torch": torch,
        "logger": logging.getLogger(__name__),
        "AttentionSpec": _AttentionSpec,
        "FullAttentionSpec": _FullAttentionSpec,
        "AscendTurboQuantAttentionSpec": _TurboQuantSpec,
        "MambaSpec": _MambaSpec,
        "AscendSFAIndexerCacheSpec": type("IndexerSpec", (), {}),
        "AttentionLayerBase": object,
        "UniformTypeKVCacheSpecs": _UniformTypeSpecs,
        "turboquant_carve_planes": turboquant_carve_planes,
        "is_deepseek_v41_cache": lambda _: False,
        "requires_padded_page_layout": lambda _: False,
        "get_layers_from_vllm_config": lambda *_: {},
        "get_layerwise_reuse_config": lambda _: None,
        "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        "get_dtype_size": lambda dtype: dtype.itemsize,
        "get_kv_cache_groups": lambda *_: groups,
        "maybe_apply_startup_plan": lambda _: None,
        "GiB_bytes": 1024**3,
    }
    _load_functions("vllm_ascend/worker/utils.py", ("row_major_strides", "make_page_strided_cache_view"), namespace)
    runner_cls = _load_class(
        "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        ("_allocate_kv_cache_tensors", "_carve_turboquant_planes", "_adjust_kv_layout"),
        namespace,
    )
    runner = runner_cls()
    runner.device = torch.device("cpu")
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1), xlite_graph_config=SimpleNamespace(enabled=False)
    )
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5"))
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_layer_compact=True, is_block_compact=True),
        ),
        kv_transfer_config=None,
    )
    runner.use_dcp = runner.use_sparse = runner.use_compress = runner.sparse_kv_offload_enabled = False
    runner.runner_only_attn_layers = set()
    runner._get_layer_kv_cache_specs = lambda _: specs
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(())
    runner._uses_page_strided_kv_layout = lambda _: False
    runner._allocate_int8_cache_tensor = lambda size, _: torch.zeros(size, dtype=torch.int8)
    runner.supports_standardized_shared_kv_backing = True
    worker_cls = _load_class(
        "vllm_ascend/worker/worker.py",
        "NPUWorker",
        ("_scale_kv_cache_memory_for_multi_group", "determine_available_memory", "_apply_kvpp_memory_budget"),
        namespace,
    )
    worker = worker_cls()
    worker.vllm_config = runner.vllm_config
    worker.model_config = runner.model_config
    worker.model_runner = runner
    worker.get_kv_cache_spec = lambda: specs
    return SimpleNamespace(runner=runner, worker=worker, config=config, specs=specs, groups=groups)


def _storage_bytes(raw):
    storages = {}
    for cache in raw.values():
        planes = cache if isinstance(cache, tuple) else (cache,)
        for plane in planes:
            storage = plane.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return storages


def test_recurrent_writes_and_attention_zeroing_preserve_other_groups(hybrid_pool):
    pool = hybrid_pool
    raw = pool.runner._allocate_kv_cache_tensors(pool.config)
    packed = turboquant_plane_views(raw[ATTENTION_LAYER], KERNEL_BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ratio = pool.specs[ATTENTION_LAYER].block_size // KERNEL_BLOCK_SIZE
    assert packed[0].shape[:2] == (NUM_BLOCKS * ratio, KERNEL_BLOCK_SIZE)
    for index, plane in enumerate(raw[ATTENTION_LAYER]):
        plane.fill_(index + 1)
    before = tuple(plane.clone() for plane in raw[ATTENTION_LAYER])
    states = []
    for name in RECURRENT_LAYERS:
        state = pool.runner._adjust_kv_layout(
            raw[name], [(NUM_BLOCKS, STATE_FLOATS)], [torch.float32], pool.specs[name].page_size_bytes
        )[0]
        for block_id in range(NUM_BLOCKS):
            # At four physical blocks, recurrent page 1 used to land exactly
            # on attention page 0's scale records, despite distinct block IDs.
            state[block_id].fill_(block_id + 1.25)
        state[-1].copy_(state[1])
        state[0].zero_()
        states.append(state)
    for actual, expected in zip(raw[ATTENTION_LAYER], before):
        assert torch.equal(actual, expected), "recycled recurrent blocks overwrote attention cache bytes"

    states_before = [state.clone() for state in states]
    for plane in raw[ATTENTION_LAYER]:
        plane.zero_()
    for state, expected in zip(states, states_before):
        assert torch.equal(state, expected), "initializing attention blocks overwrote recurrent state"
    assert len(_storage_bytes(raw)) == len(pool.specs)


@pytest.mark.parametrize("uniform_wrapper", [False, True])
@pytest.mark.parametrize("extra_attention_layer", [False, True])
def test_private_allocation_fits_the_advertised_memory_budget(hybrid_pool, uniform_wrapper, extra_attention_layer):
    pool = hybrid_pool
    if extra_attention_layer:
        # A descriptor may contain several layers. They each need their own
        # private region even though the other groups overlay this descriptor.
        extra_name = "model.layers.7.self_attn.attn"
        pool.specs[extra_name] = _TurboQuantSpec(**vars(pool.specs[ATTENTION_LAYER]))
        pool.groups[0].layer_names.append(extra_name)
        pool.config.kv_cache_tensors[0].layers.append(extra_name)
        for descriptor in pool.config.kv_cache_tensors:
            descriptor.size *= 2
    if uniform_wrapper:
        pool.groups[0].kv_cache_spec = _UniformTypeSpecs(
            kv_cache_specs={name: pool.specs[name] for name in pool.groups[0].layer_names}
        )
    sum_pages = sum(spec.page_size_bytes for spec in pool.specs.values())
    available = NUM_BLOCKS * sum_pages
    largest_group_page = max(
        sum(pool.specs[name].page_size_bytes for name in group.layer_names) for group in pool.groups
    )
    advertised = pool.worker._scale_kv_cache_memory_for_multi_group(available)
    assert advertised == available * largest_group_page // sum_pages
    assert advertised // largest_group_page == pool.config.num_blocks

    raw = pool.runner._allocate_kv_cache_tensors(pool.config)
    assert sum(_storage_bytes(raw).values()) == available


def test_explicit_memory_budget_also_covers_private_layers(hybrid_pool):
    pool = hybrid_pool
    available = NUM_BLOCKS * sum(spec.page_size_bytes for spec in pool.specs.values())
    worker = pool.worker
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=available)
    worker.init_snapshot = SimpleNamespace(free_memory=2 * available)
    worker._kvpp_cache_allocation_plan = None
    worker._apply_kv_offload_decode_memory_constraints = lambda memory: memory
    worker.model_runner.profile_run = lambda: None
    assert worker.determine_available_memory() == available // len(pool.groups)


def test_dense_hybrid_retains_shared_pool_capacity(hybrid_pool):
    pool = hybrid_pool
    dense_spec = _FullAttentionSpec(**vars(pool.specs[ATTENTION_LAYER]))
    pool.specs[ATTENTION_LAYER] = dense_spec
    pool.groups[0].kv_cache_spec = dense_spec
    available = NUM_BLOCKS * dense_spec.page_size_bytes
    assert pool.worker._scale_kv_cache_memory_for_multi_group(available) == available


def test_private_pages_expand_and_write_across_the_third_page(hybrid_pool):
    """Run the real table constructor/expansion/draft slots against private KV."""
    pool = hybrid_pool
    block_size = pool.specs[ATTENTION_LAYER].block_size
    namespace = {
        "torch": torch,
        "np": np,
        "get_dcp_group": lambda: SimpleNamespace(world_size=1, rank_in_group=0),
        "CpuGpuBuffer": _CpuGpuBuffer,
    }
    table_cls = _load_class(
        "vllm_ascend/worker/block_table.py",
        "BlockTable",
        (
            "__init__",
            "_make_buffer",
            "_convert_physical_to_logical_blocks",
            "append_row",
            "add_row",
            "compute_slot_mapping_draft",
        ),
        namespace,
    )
    table = table_cls(block_size, 1, 3, 8, False, torch.device("cpu"), kernel_sizes=[KERNEL_BLOCK_SIZE])
    page_order = [3, 1, 2]
    table.add_row(page_order, 0)
    table.block_table.copy_to_gpu(1)
    ratio = block_size // KERNEL_BLOCK_SIZE
    assert table.blocks_per_phys_block == ratio
    assert table.block_table.gpu[0].tolist() == [page * ratio + sub for page in page_order for sub in range(ratio)]
    positions = np.array([block_size - 1, block_size, 2 * block_size - 1, 2 * block_size, 2 * block_size + 1])
    table.compute_slot_mapping_draft(np.zeros_like(positions), positions)
    expected_slots = torch.tensor(
        [page_order[position // block_size] * block_size + position % block_size for position in positions],
        dtype=torch.int32,
    )
    assert torch.equal(table.slot_mapping.gpu[: len(positions)], expected_slots)

    raw = pool.runner._allocate_kv_cache_tensors(pool.config)
    physical = turboquant_plane_views(raw[ATTENTION_LAYER], block_size, NUM_KV_HEADS, HEAD_SIZE)
    kernel = turboquant_plane_views(raw[ATTENTION_LAYER], KERNEL_BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    expected_cache = tuple(torch.zeros_like(plane) for plane in physical)
    emulator = TurboQuantEmulator(HEAD_SIZE, torch.device("cpu"))
    adapter = TurboQuantOpsAdapter(TurboQuantOpsConfig(value_layout=TurboQuantValueLayout.NZ_TILED))
    generator = torch.Generator().manual_seed(2049)
    key, value = [torch.randn(len(positions), NUM_KV_HEADS, HEAD_SIZE, generator=generator) for _ in range(2)]
    adapter.reshape_scatter_decode(key, value, *kernel, table.slot_mapping.gpu[: len(positions)], emulator)
    adapter.reshape_scatter_decode(key, value, *expected_cache, expected_slots, emulator)
    for actual, expected in zip(physical, expected_cache):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dense_physical = adapter.gather(*expected_cache, torch.tensor([page_order], dtype=torch.int32))
    dense_kernel = adapter.gather(*kernel, table.block_table.gpu)
    for actual, expected in zip(dense_kernel, dense_physical):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    validator = object.__new__(AscendTurboQuantMetadataBuilder)
    validator._block_size = KERNEL_BLOCK_SIZE
    for position in (2 * block_size - 1, 2 * block_size, 2 * block_size + 1):
        table.compute_slot_mapping_draft(np.array([0]), np.array([position]))
        validator._validate_slot_mapping(
            SimpleNamespace(num_reqs=1, num_actual_tokens=1, slot_mapping=table.slot_mapping.gpu[:1]),
            table.block_table.gpu,
            torch.ones(1, dtype=torch.int32),
            torch.tensor([position + 1], dtype=torch.int32),
        )
    for name in RECURRENT_LAYERS:
        assert not torch.count_nonzero(raw[name])
