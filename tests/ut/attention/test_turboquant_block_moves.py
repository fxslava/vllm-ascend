# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.turboquant_codec import TurboQuantEmulator
from vllm_ascend.attention.turboquant_config import TurboQuantValueLayout
from vllm_ascend.attention.turboquant_layout import turboquant_scale_block_views
from vllm_ascend.attention.turboquant_metadata import AscendTurboQuantMetadataBuilder
from vllm_ascend.attention.turboquant_ops import TurboQuantOpsAdapter
from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantBackend, AscendTurboQuantBackendImpl

NUM_BLOCKS = 4
BLOCK_SIZE = 128
NUM_KV_HEADS = 4
HEAD_SIZE = 64


def _cache():
    packed_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2)
    key = torch.arange(NUM_BLOCKS, dtype=torch.int8).view(-1, 1, 1, 1).expand(packed_shape).clone()
    value = key + 17
    # Distinct scales in every lane expose both incomplete copies and writes
    # to unrelated pages. The shape alone does not describe their byte layout.
    scales = torch.arange(2 * NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE, dtype=torch.float32)
    return key, value, scales.view(2, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)


@pytest.mark.parametrize("layout", list(TurboQuantValueLayout))
@pytest.mark.parametrize("operation", ["copy", "swap"])
def test_block_moves_preserve_the_physical_scale_records(monkeypatch, layout, operation):
    monkeypatch.setattr(TurboQuantOpsAdapter, "value_layout", lambda self: layout)
    source = _cache()
    destination = source if operation == "copy" else tuple(torch.full_like(plane, -7) for plane in source)
    before = tuple(plane.clone() for plane in destination)
    source_before = tuple(plane.clone() for plane in source)
    mapping = torch.tensor([[0, 2], [2, 0]], dtype=torch.int64)
    src, dst = mapping.unbind(1)

    if operation == "copy":
        AscendTurboQuantBackend.copy_blocks([destination], mapping)
    else:
        AscendTurboQuantBackend.swap_blocks(list(source), list(destination), mapping)

    for index in (0, 1):
        before[index][dst] = source_before[index][src]
        assert torch.equal(destination[index], before[index])
    if layout is TurboQuantValueLayout.NZ_TILED:
        before[2].view(NUM_BLOCKS, -1)[dst] = source_before[2].view(NUM_BLOCKS, -1)[src]
    else:
        before[2][:, dst] = source_before[2][:, src]
    assert torch.equal(destination[2], before[2]), "scale records were copied using the wrong page axis"
    if operation == "swap":
        for original, unchanged in zip(source, source_before):
            assert torch.equal(original, unchanged)


@pytest.mark.parametrize("layout", list(TurboQuantValueLayout))
def test_impl_block_views_zero_only_the_selected_scale_page(layout):
    cache = _cache()
    original = cache[2].clone()
    impl = object.__new__(AscendTurboQuantBackendImpl)
    impl.adapter = TurboQuantOpsAdapter()
    impl.adapter.value_layout = lambda: layout
    views = impl.get_kv_cache_block_views(cache)
    for view in views[2:]:
        assert view.shape[0] == NUM_BLOCKS
        view[2].zero_()
    if layout is TurboQuantValueLayout.NZ_TILED:
        original.view(NUM_BLOCKS, -1)[2].zero_()
    else:
        original[:, 2].zero_()
    assert torch.equal(cache[2], original)


def test_nz_scale_views_refuse_records_without_the_required_padding():
    scales = torch.zeros((2, NUM_BLOCKS, 2, BLOCK_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="aligned bursts"):
        turboquant_scale_block_views(scales, slot_major=True)


def _slot_validation_inputs():
    builder = object.__new__(AscendTurboQuantMetadataBuilder)
    builder._block_size = BLOCK_SIZE
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        slot_mapping=torch.tensor([383, 1152, 641, 642, 643, -1], dtype=torch.int32),
    )
    # Unused table columns and the padded slot are deliberately dirty.
    table = torch.tensor([[2, 9, -1], [5, -1, -1]], dtype=torch.int32)
    queries = torch.tensor([2, 3, 7], dtype=torch.int32)
    contexts = torch.tensor([129, 4, 1], dtype=torch.int32)
    return builder, metadata, table, queries, contexts


@pytest.mark.parametrize("validate_slots", [False, True])
def test_dummy_metadata_invalidates_previous_request_slots_before_upload(monkeypatch, validate_slots):
    from vllm_ascend.attention import turboquant_v1

    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    metadata.offload_dummy = True
    previous_slots = metadata.slot_mapping
    monkeypatch.setattr(turboquant_v1.envs, "VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS", validate_slots)

    def upload(*args):
        assert previous_slots.tolist() == [-1] * previous_slots.numel()
        return contexts, contexts

    builder._refresh_context_lens = upload
    builder._refresh_tiling = upload
    builder._tiling = builder._workspace = builder._query_rot = object()
    builder._validate_slot_mapping = lambda *args: pytest.fail("dummy requests do not own physical slots")
    builder._build_backend_metadata(
        metadata,
        block_table=table,
        query_lens=queries,
        seq_lens=contexts,
        num_decodes=0,
        num_prefills=2,
    )
    assert metadata.slot_mapping is previous_slots

    cache = _cache()
    before = tuple(plane.clone() for plane in cache)
    emulator = TurboQuantEmulator(HEAD_SIZE, torch.device("cpu"))
    activations = torch.ones(metadata.num_actual_tokens, NUM_KV_HEADS, HEAD_SIZE)
    TurboQuantOpsAdapter()._write_slots(
        activations, activations, *cache, previous_slots[: metadata.num_actual_tokens], emulator
    )
    for actual, original in zip(cache, before):
        assert torch.equal(actual, original), "dummy execution overwrote the previous request's cache"


def test_slot_validation_accepts_only_live_request_rows():
    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    builder._validate_slot_mapping(metadata, table, queries, contexts)


def test_slot_validation_ignores_a_trailing_fia_padding_request():
    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    metadata.num_reqs = 3
    table = torch.cat((table, torch.zeros_like(table[:1])))
    builder._validate_slot_mapping(metadata, table, queries, contexts)


def test_slot_validation_refreshes_when_the_same_buffer_is_reused():
    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    builder._validate_slot_mapping(metadata, table, queries, contexts)
    metadata.slot_mapping[2] = 769  # A valid slot, but in another physical page.
    with pytest.raises(ValueError, match=r"request=1, token_row=2, position=1, actual_slot=769, expected_slot=641"):
        builder._validate_slot_mapping(metadata, table, queries, contexts)


@pytest.mark.parametrize("actual_slot", [-1, 0, 1153])
def test_slot_validation_rejects_skipped_or_misaddressed_live_tokens(actual_slot):
    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    metadata.slot_mapping[1] = actual_slot
    with pytest.raises(ValueError, match=r"token_row=1.*expected_slot=1152"):
        builder._validate_slot_mapping(metadata, table, queries, contexts)


def test_slot_validation_runs_before_context_and_tiling_uploads(monkeypatch):
    from vllm_ascend.attention import turboquant_v1

    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    metadata.slot_mapping[2] = 769
    monkeypatch.setattr(turboquant_v1.envs, "VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS", True)
    builder._refresh_context_lens = lambda *args: pytest.fail("invalid slots reached the context upload")
    builder._refresh_tiling = lambda *args: pytest.fail("invalid slots reached the tiling upload")
    with pytest.raises(ValueError, match="slot_mapping mismatch"):
        builder._build_backend_metadata(
            metadata,
            block_table=table,
            query_lens=queries,
            seq_lens=contexts,
            num_decodes=0,
            num_prefills=2,
        )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("lengths", "live query lengths"),
        ("slots", "shorter than the current batch"),
        ("context", "query_len=2, context_len=1"),
        ("row_capacity", "exceeds its block-table row"),
        ("unmapped", "unmapped live block-table entry"),
    ],
)
def test_slot_validation_refuses_incomplete_or_inconsistent_metadata(invalid, message):
    builder, metadata, table, queries, contexts = _slot_validation_inputs()
    if invalid == "lengths":
        metadata.num_actual_tokens += 1
    elif invalid == "slots":
        metadata.slot_mapping = metadata.slot_mapping[:4]
    elif invalid == "context":
        contexts[0] = 1
    elif invalid == "row_capacity":
        contexts[0] = 4 * BLOCK_SIZE
    else:
        table[0, 1] = -1
    with pytest.raises(ValueError, match=message):
        builder._validate_slot_mapping(metadata, table, queries, contexts)


def test_slot_validation_refuses_two_live_requests_writing_the_same_slot():
    builder, metadata, _, _, _ = _slot_validation_inputs()
    metadata.num_actual_tokens = 2
    metadata.slot_mapping = torch.tensor([256, 256], dtype=torch.int32)
    table = torch.tensor([[2], [2]], dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    with pytest.raises(ValueError, match=r"alias physical slot 256 at token rows \[0, 1\]"):
        builder._validate_slot_mapping(metadata, table, lengths, lengths)
