# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared TurboQuant test setup; numerical and layout oracles stay in the tests."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.attention.turboquant_layout import turboquant_packed_bytes, turboquant_scale_cache_shape
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_DECODE_MAX_SPLITS,
    TURBOQUANT_DECODE_SPLITS,
    TURBOQUANT_DEFAULT_NAMESPACE,
    AscendTurboQuantMetadataBuilder,
    TurboQuantKernelStage,
    TurboQuantOpsAdapter,
    TurboQuantOpsConfig,
    TurboQuantValueLayout,
    turboquant_decode_query_rot_floats,
    turboquant_decode_workspace_floats,
    turboquant_tiling_upper_bound_words,
)


class OpRecorder:
    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result

    @property
    def args(self) -> tuple[Any, ...]:
        assert len(self.calls) == 1, f"expected exactly one call, got {len(self.calls)}"
        return self.calls[0][0]


def register_ops(monkeypatch, *, namespace: str = TURBOQUANT_DEFAULT_NAMESPACE, **entries: OpRecorder) -> None:
    monkeypatch.setattr(torch, "ops", SimpleNamespace(**{namespace: SimpleNamespace(**entries)}))


def make_adapter(
    *stages: TurboQuantKernelStage,
    value_layout: TurboQuantValueLayout = TurboQuantValueLayout.TOKEN_MAJOR,
) -> TurboQuantOpsAdapter:
    return TurboQuantOpsAdapter(TurboQuantOpsConfig(kernel_stages=frozenset(stages), value_layout=value_layout))


def make_cache(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, *, device="cpu", decode: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (num_blocks, block_size, num_kv_heads, turboquant_packed_bytes(head_size))
    scales = (
        (num_blocks * block_size * TurboQuantOpsAdapter._nz_scale_slot_floats(num_kv_heads),)
        if decode
        else turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)
    )
    return (
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(packed, dtype=torch.int8, device=device),
        torch.zeros(scales, dtype=torch.float32, device=device),
    )


def make_activations(
    num_tokens: int, num_kv_heads: int, head_size: int, *, seed: int, device="cpu", dtype=torch.float32
):
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, num_kv_heads, head_size)
    return tuple(torch.randn(shape, generator=generator).to(device=device, dtype=dtype) for _ in range(2))


def make_decode_buffers(tokens: int, num_heads: int, head_size: int, *, device="cpu") -> dict[str, torch.Tensor]:
    return {
        "tiling": torch.zeros(
            turboquant_tiling_upper_bound_words(tokens, TURBOQUANT_DECODE_MAX_SPLITS), dtype=torch.int32, device=device
        ),
        "workspace": torch.zeros(
            turboquant_decode_workspace_floats(tokens, num_heads, head_size, TURBOQUANT_DECODE_SPLITS),
            dtype=torch.float32,
            device=device,
        ),
        "query_rot": torch.zeros(
            turboquant_decode_query_rot_floats(tokens, num_heads, head_size), dtype=torch.float32, device=device
        ),
    }


def make_metadata_builder(
    max_num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    max_blocks: int,
    *,
    device="cpu",
) -> AscendTurboQuantMetadataBuilder:
    """Run the real buffer allocation, replacing only engine-specific base setup."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
        model_config=SimpleNamespace(get_num_attention_heads=lambda _: num_heads),
        parallel_config=SimpleNamespace(),
    )
    spec = SimpleNamespace(num_kv_heads=num_kv_heads, head_size=head_size, block_size=block_size)

    def init_base(builder, *args, **kwargs):
        builder.max_num_blocks_per_req = max_blocks

    with patch.object(AscendAttentionMetadataBuilder, "__init__", init_base):
        builder = AscendTurboQuantMetadataBuilder(spec, ["attention"], config, torch.device(device))
    builder._block_size = block_size
    return builder
