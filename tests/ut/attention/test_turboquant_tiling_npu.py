# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.turboquant import make_metadata_builder

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture
def tiling_op():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from vllm_ascend.attention.turboquant_ops import load_turboquant_extension

    load_turboquant_extension()
    namespace = torch.ops.turboquant_ops
    if not hasattr(namespace, "npu_turboquant_build_decode_tiling"):
        pytest.skip("requires an ascend950 TurboQuant build")
    return namespace.npu_turboquant_build_decode_tiling


@pytest.mark.parametrize("batch_size", [1, 17])
def test_queued_tiling_uploads_preserve_sources_and_stream_order(tiling_op, batch_size):
    from vllm_ascend.attention.turboquant_config import turboquant_tiling_upper_bound_words

    words = turboquant_tiling_upper_bound_words(batch_size, 8)
    device_tiling = torch.full((words,), -1, dtype=torch.int32, device="npu")
    expected = []
    snapshots = []
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for step in range(64):
            context_lengths = (torch.arange(batch_size, dtype=torch.int32) * 257 + step) % 4097
            host_tiling = torch.full((words,), -1, dtype=torch.int32)
            tiling_op(context_lengths, host_tiling, 24, 4, 256, 128, 64, 8, 6)
            expected.append(host_tiling)
            tiling_op(context_lengths, device_tiling, 24, 4, 256, 128, 64, 8, 6)
            snapshots.append(device_tiling.clone())
            torch.empty(words, dtype=torch.int32, pin_memory=True).fill_(0x5A5A)
    stream.synchronize()
    for actual, reference in zip(snapshots, expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


def test_invalid_lengths_do_not_modify_device_tiling(tiling_op):
    context_lengths = torch.tensor([-1], dtype=torch.int32)
    device_tiling = torch.full((512,), -1, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="context_lens"):
        tiling_op(context_lengths, device_tiling, 24, 4, 256, 128, 64, 8, 6)
    assert torch.all(device_tiling == -1)


def test_context_upload_ring_preserves_queued_steps(tiling_op):
    builder = make_metadata_builder(4, 24, 4, 256, 128, 64, device="npu")
    expected = []
    snapshots = []
    for step in range(64):
        lengths = torch.tensor([step, 128 + step, 1, 1], dtype=torch.int32)
        device_lengths, _ = builder._refresh_context_lens(SimpleNamespace(num_reqs=2), lengths)
        snapshots.append(device_lengths.clone())
        expected.append(torch.tensor([step, 128 + step, 0, 0], dtype=torch.int32))
    torch.npu.synchronize()
    for actual, reference in zip(snapshots, expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
