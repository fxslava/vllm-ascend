# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant dry-run codec and its backend descriptor.

Two properties are pinned here, both of which an innocent-looking edit can
break without failing anything else:

* the quantiser reads the fp32 exponent field with integer ops only. ``frexp``
  and ``ldexp`` say the same thing, but neither has an NPU kernel: they fall
  back to the CPU, a fallback is a device-to-host copy, and that copy aborts an
  ACL graph capture (``aclrtMemcpy`` 107030). So the test both compares against
  the ``frexp``/``ldexp`` spelling and forbids calling it.
* the backend's ``get_name`` is an ``AttentionBackendEnum`` member name that
  resolves back to this class. vLLM looks a backend up both ways.
"""

from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_HEADROOM_BITS,
    TURBOQUANT_NIBBLE_MAX,
    TURBOQUANT_NIBBLE_MIN,
    AscendTurboQuantDryRunBackend,
    TurboQuantEmulator,
    _exp2_fp32,
)

HEAD_SIZE = 64
CPU = torch.device("cpu")


@pytest.fixture
def emulator() -> TurboQuantEmulator:
    return TurboQuantEmulator(head_size=HEAD_SIZE, device=CPU)


def _reference_quantize(emulator: TurboQuantEmulator, rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The previous spelling of :meth:`TurboQuantEmulator.quantize`, verbatim."""
    rotated_fp32 = rotated.to(torch.float32)
    absmax = rotated_fp32.abs().amax(dim=-1, keepdim=True)
    zero_row = absmax == 0.0
    _, frexp_exponent = torch.frexp(absmax)
    exponent = frexp_exponent.to(torch.int32) - 1
    ones = torch.ones_like(absmax)
    pow_scale = torch.ldexp(ones, exponent - TURBOQUANT_HEADROOM_BITS)
    inv_scale = torch.ldexp(ones, TURBOQUANT_HEADROOM_BITS - exponent)
    codes = torch.round(rotated_fp32 * inv_scale)
    codes = torch.clamp(codes, TURBOQUANT_NIBBLE_MIN, TURBOQUANT_NIBBLE_MAX)
    codes = torch.where(zero_row, torch.zeros_like(codes), codes)
    scales = torch.where(zero_row, ones, pow_scale * emulator.inv_sqrt_len)
    return codes.to(torch.int8), scales.squeeze(-1)


def _panels() -> dict[str, torch.Tensor]:
    """Rows spanning the normal exponent range, plus every documented branch."""
    torch.manual_seed(0)
    return {
        "randn": torch.randn(97, HEAD_SIZE),
        "zero_rows": torch.zeros(3, HEAD_SIZE),
        "powers_of_two": torch.stack([torch.full((HEAD_SIZE,), float(2.0**k)) for k in range(-100, 101, 7)]),
        "mixed_magnitudes": (torch.randn(64, HEAD_SIZE) * torch.tensor([2.0**k for k in range(-32, 32)]).unsqueeze(1)),
        "exact_halves": torch.full((5, HEAD_SIZE), 0.5),
        "smallest_normal": torch.full((2, HEAD_SIZE), float(2.0**-126)),
        "one_zero_row_among_others": torch.cat([torch.randn(4, HEAD_SIZE), torch.zeros(1, HEAD_SIZE)]),
    }


@pytest.mark.parametrize("name", sorted(_panels()))
def test_quantize_is_bit_identical_to_the_frexp_spelling(emulator, name):
    rotated = emulator.rotate(_panels()[name])
    codes, scales = emulator.quantize(rotated)
    expected_codes, expected_scales = _reference_quantize(emulator, rotated)
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)
    assert torch.isfinite(scales).all()


def test_quantize_calls_neither_frexp_nor_ldexp(emulator):
    """The capture-safety property, not a style preference."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the quantiser must not use an op that falls back to the host")

    rotated = emulator.rotate(torch.randn(32, HEAD_SIZE))
    with patch.object(torch, "frexp", refuse), patch.object(torch, "ldexp", refuse):
        codes, scales = emulator.quantize(rotated)
    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float32


def test_quantize_survives_a_rotation_that_overflows_fp32(emulator):
    """Garbage in: the scale plane must still hold finite numbers."""
    overflowed = emulator.rotate(torch.full((2, HEAD_SIZE), float(2.0**125)))
    assert not torch.isfinite(overflowed).all(), "this case is about the overflow"
    codes, scales = emulator.quantize(overflowed)
    assert torch.isfinite(scales).all()
    assert bool((codes.to(torch.int32).abs() <= -TURBOQUANT_NIBBLE_MIN).all())


def test_quantize_of_a_denormal_row_saturates_instead_of_underflowing(emulator):
    """A row that small quantises to zero codes; its scale must stay normal."""
    codes, scales = emulator.quantize(torch.full((2, HEAD_SIZE), float(2.0**-140)))
    assert int(codes.abs().sum()) == 0
    assert torch.isfinite(scales).all()
    assert bool((scales > 0).all())


@pytest.mark.parametrize("exponent", [-126, -64, -1, 0, 1, 23, 64, 127])
def test_exp2_fp32_is_exact(exponent):
    built = _exp2_fp32(torch.tensor([exponent], dtype=torch.int32))
    assert built.dtype == torch.float32
    assert float(built) == 2.0**exponent


def test_exp2_fp32_clamps_to_the_normal_range():
    """Out of range is clamped, never an infinity, a NaN or a denormal."""
    built = _exp2_fp32(torch.tensor([-400, -130, 130, 400], dtype=torch.int32))
    assert torch.isfinite(built).all()
    assert bool((built > 0).all())
    assert float(built[0]) == float(built[1]) == 2.0**-126
    assert float(built[2]) == float(built[3]) == 2.0**127


def test_round_trip_reconstructs_the_activation():
    """The 4-bit grid's own error, not a bug: ~0.99 cosine at D=256."""
    emulator = TurboQuantEmulator(head_size=256, device=CPU)
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    x_hat = emulator.decode(*emulator.encode(x))
    cosine = torch.nn.functional.cosine_similarity(x.reshape(1, -1), x_hat.reshape(1, -1))
    assert float(cosine) > 0.98


def test_backend_name_is_the_enum_member_it_registers_as():
    name = AscendTurboQuantDryRunBackend.get_name()
    assert name == "TURBOQUANT"
    # vLLM indexes the enum with whatever get_name returns; a name that is not a
    # member raises there, which is why this backend does not carry its own.
    assert AttentionBackendEnum[name] is AttentionBackendEnum.TURBOQUANT
    # ... and the member resolves back to this class rather than to vLLM's own
    # unrelated TurboQuant backend, because the module registered an override.
    assert AttentionBackendEnum.TURBOQUANT.get_class() is AscendTurboQuantDryRunBackend
