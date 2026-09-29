"""Shared fixtures: sanity geometries, mock weight providers, allocation guards."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence

import pytest
import torch

from ..core.config import SANITY_GEOMETRY, DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout

FORBIDDEN_ALLOCATOR_NAMES = (
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    "tensor",
    "as_tensor",
    "cat",
    "stack",
    "rand",
    "randn",
    "randint",
)


@contextlib.contextmanager
def forbid_torch_allocations() -> Iterator[None]:
    """Booby-trap every torch allocator for the duration of a decode loop."""
    originals = {name: getattr(torch, name) for name in FORBIDDEN_ALLOCATOR_NAMES}
    original_clone = torch.Tensor.clone

    def make_guard(name: str):
        def guard(*_args, **_kwargs):
            raise AssertionError(f"decode-step path attempted allocation via torch.{name}")

        return guard

    def guard_clone(*_args, **_kwargs):
        raise AssertionError("decode-step path attempted allocation via Tensor.clone")

    try:
        for name in originals:
            setattr(torch, name, make_guard(name))
        torch.Tensor.clone = guard_clone  # type: ignore[assignment]
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch, name, fn)
        torch.Tensor.clone = original_clone  # type: ignore[assignment]


class PinnedMemoryWeightProvider:
    """Production-shaped blocking provider over pre-staged host buffers."""

    def __init__(self, buffers: Mapping[tuple[int, int, str], torch.Tensor]):
        self._buffers: dict[tuple[int, int, str], torch.Tensor] = {}
        for key, tensor in buffers.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"host provider buffer {key} must live on CPU, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"host provider buffer {key} must be contiguous")
            self._buffers[key] = tensor

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        try:
            return self._buffers[(layer_idx, expert_id, param_key)]
        except KeyError:
            raise KeyError(
                f"no host buffer staged for (layer={layer_idx}, expert={expert_id}, param={param_key})"
            ) from None


class MockWeightProvider(PinnedMemoryWeightProvider):
    """Pre-allocates deterministic synthetic buffers for every expert parameter.

    Each buffer starts with a 4-byte big-endian tag uniquely encoding
    (layer, expert, param), followed by a position-dependent pattern -- so
    ``torch.equal`` against a slot view proves the *right* expert's bytes
    landed in the *right* slot. All allocation happens at construction.
    """

    def __init__(self, layout: ExpertTensorLayout, layer_ids: Sequence[int], num_experts: int):
        buffers: dict[tuple[int, int, str], torch.Tensor] = {}
        num_params = len(layout.specs)
        for layer_idx in layer_ids:
            for expert_id in range(num_experts):
                for param_index, spec in enumerate(layout.specs):
                    tag = (layer_idx * 256 + expert_id) * num_params + param_index
                    buffers[(layer_idx, expert_id, spec.param_key)] = _synthetic_bytes(
                        tag, spec.view_shape[0], spec.view_shape[1]
                    )
        super().__init__(buffers)


def _synthetic_bytes(tag: int, rows: int, cols: int) -> torch.Tensor:
    row_index = torch.arange(rows).unsqueeze(1)
    col_index = torch.arange(cols).unsqueeze(0)
    buffer = ((tag + 31 * row_index + col_index) % 256).to(torch.uint8)
    flat = buffer.view(-1)
    for byte_pos in range(4):
        flat[byte_pos] = (tag >> (8 * (3 - byte_pos))) & 0xFF
    return buffer


@pytest.fixture
def sanity_config() -> DeepSeekV4MoEConfig:
    return SANITY_GEOMETRY


@pytest.fixture
def sanity_layout() -> ExpertTensorLayout:
    return ExpertTensorLayout.for_deepseek_v4_flash(SANITY_GEOMETRY)


@pytest.fixture
def full_layout() -> ExpertTensorLayout:
    return ExpertTensorLayout.for_deepseek_v4_flash(DeepSeekV4MoEConfig())


@pytest.fixture
def mock_provider_factory() -> type[MockWeightProvider]:
    return MockWeightProvider


@pytest.fixture
def allocation_guard() -> type[forbid_torch_allocations]:  # type: ignore[misc]
    return forbid_torch_allocations  # type: ignore[return-value]
