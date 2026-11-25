# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.turboquant import make_metadata_builder, register_ops
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_config import TURBOQUANT_STAGING_SLOTS
from vllm_ascend.attention.turboquant_metadata import AscendTurboQuantMetadataBuilder
from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantBackendImpl


@pytest.mark.parametrize("upload_complete", [False, True])
def test_context_staging_never_overwrites_an_inflight_source(upload_complete):
    builder = object.__new__(AscendTurboQuantMetadataBuilder)
    builder._max_num_reqs = 4
    builder._staging_slots = TURBOQUANT_STAGING_SLOTS
    builder._staging_step = TURBOQUANT_STAGING_SLOTS
    builder._staging_pinned = False
    builder._context_lens_host_ring = [torch.full((4,), index, dtype=torch.int32) for index in range(2)]
    builder._context_lens_upload_events = [SimpleNamespace(query=lambda: upload_complete), None]
    previous_source = builder._context_lens_host_ring[0]

    next_source = builder._staging_slot()
    next_source.fill_(123)

    assert (next_source is previous_source) is upload_complete
    if not upload_complete:
        assert previous_source.tolist() == [0] * 4


@pytest.mark.parametrize("cumulative_lengths", [[1, 4, 6], [4], [1, 2, 3]])
def test_mixed_query_layout_reuses_one_upload_across_layers(cumulative_lengths, monkeypatch):
    metadata = SimpleNamespace(actual_seq_lengths_q=cumulative_lengths)
    device = torch.device("cpu")
    layout = AscendTurboQuantBackendImpl._mixed_query_layout(metadata, device)

    expected_requests = []
    expected_positions = []
    query_lengths = []
    start = 0
    for request, end in enumerate(cumulative_lengths):
        query_lengths.append(end - start)
        expected_requests.extend([request] * (end - start))
        expected_positions.extend(range(end - start))
        start = end
    assert layout.request_indices.tolist() == expected_requests
    assert layout.positions_in_request.tolist() == expected_positions
    assert layout.query_lengths.tolist() == query_lengths
    assert layout.query_lengths_list == query_lengths
    assert layout.num_tokens == cumulative_lengths[-1]
    assert layout.max_query_len == max(query_lengths)

    def unexpected_upload(*args, **kwargs):
        pytest.fail("query layout was uploaded again for another layer")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_upload)
    assert AscendTurboQuantBackendImpl._mixed_query_layout(metadata, device) is layout


def test_mixed_query_layout_refreshes_reused_metadata():
    metadata = SimpleNamespace(actual_seq_lengths_q=[1, 4])
    device = torch.device("cpu")
    first = AscendTurboQuantBackendImpl._mixed_query_layout(metadata, device)
    metadata.actual_seq_lengths_q[:] = [2, 4]
    second = AscendTurboQuantBackendImpl._mixed_query_layout(metadata, device)

    assert second is not first
    assert second.request_indices.tolist() == [0, 0, 1, 1]
    assert second.positions_in_request.tolist() == [0, 1, 0, 1]
    assert second.query_lengths_list == [2, 2]


def test_mixed_forward_preserves_query_order_and_causal_mask():
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[1, 4, 6],
        turboquant_context_lens=torch.tensor([5, 7, 8], dtype=torch.int32),
        seq_lens_list=[5, 7, 8],
    )
    impl = object.__new__(AscendTurboQuantBackendImpl)
    impl.emulator = object()
    impl.num_heads = 2
    impl.head_size = 4
    query = torch.arange(48, dtype=torch.float32).reshape(6, 2, 4)
    dense_key = torch.zeros((3, 8, 1, 4))
    impl._gather_dense_kv = lambda *args: (dense_key, dense_key)
    calls = []

    def attend(**kwargs):
        calls.append(kwargs)
        padded = kwargs["query"]
        assert padded.shape == (3, 3, 2, 4)
        torch.testing.assert_close(padded[0, :1], query[:1])
        torch.testing.assert_close(padded[1], query[1:4])
        torch.testing.assert_close(padded[2, :2], query[4:6])
        horizons = torch.tensor([[4, 5, 6], [4, 5, 6], [6, 7, 8]])
        expected_mask = (torch.arange(8).view(1, 1, -1) > horizons.unsqueeze(-1)).unsqueeze(1)
        assert torch.equal(kwargs["atten_mask"], expected_mask)
        assert kwargs["actual_seq_lengths"] == [1, 3, 2]
        return padded + 100

    impl._run_dense_attention = attend
    for _ in range(2):
        output = torch.empty_like(query)
        assert impl._forward_mixed(query, metadata, output) is output
        torch.testing.assert_close(output, query + 100)
    assert len(calls) == 2


def test_decode_metadata_refreshes_at_the_third_physical_page(monkeypatch):
    """A single captured width must see new lengths, slots and third-page IDs."""
    contexts = (2047, 2048, 2049, 2177)
    seen = []

    def build_tiling(lengths, tiling, *args):
        seen.append(lengths.tolist())
        tiling[0] = lengths[0]

    register_ops(monkeypatch, npu_turboquant_build_decode_tiling=build_tiling)
    builder = make_metadata_builder(1, 24, 4, 256, 128, 24)
    slots = torch.full((1,), -1, dtype=torch.int64)
    table = torch.zeros((1, 24), dtype=torch.int32)
    names = ("_context_lens", "_tiling", "_workspace", "_query_rot", "_decode_slot_mapping")
    addresses = {name: getattr(builder, name).data_ptr() for name in names}
    impl = object.__new__(AscendTurboQuantBackendImpl)
    for context in contexts:
        physical_pages = torch.tensor([3, 1, 2], dtype=torch.int32)
        if context > 2048:
            physical_pages[2] = 7  # Reassigned page ID, same table storage.
        table.copy_((physical_pages[:, None] * 8 + torch.arange(8)).reshape(1, -1))
        position = context - 1
        expected_slot = int(physical_pages[position // 1024]) * 1024 + position % 1024
        slots.fill_(expected_slot)
        common = SimpleNamespace(
            num_reqs=1, num_actual_tokens=1, attn_state=AscendAttentionState.DecodeOnly, slot_mapping=slots
        )
        extra = builder._build_backend_metadata(
            common,
            block_table=table,
            query_lens=torch.ones(1, dtype=torch.int32),
            seq_lens=torch.tensor([context], dtype=torch.int32),
            num_decodes=1,
            num_prefills=0,
        )
        metadata = SimpleNamespace(attn_state=common.attn_state, slot_mapping=slots, **extra)
        decode_slots, plan = impl._ingestion_inputs(metadata, 1, impl._ingestion_phase(metadata, 1))
        assert decode_slots.tolist() == [expected_slot] and plan is None
        assert extra["turboquant_context_lens"].tolist() == [context]
        assert int(extra["turboquant_tiling"][0]) == context
        assert {name: getattr(builder, name).data_ptr() for name in names} == addresses
    assert seen == [[context] for context in contexts]
