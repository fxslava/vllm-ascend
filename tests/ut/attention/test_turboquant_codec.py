# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant codec and its backend descriptor.

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

import struct
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.attention import turboquant_codec
from vllm_ascend.attention.turboquant_codec import TurboQuantEmulator, _exp2_fp32
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_HEADROOM_BITS,
    TURBOQUANT_NIBBLE_MAX,
    TURBOQUANT_NIBBLE_MIN,
)
from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantBackend

HEAD_SIZE = 64
CPU = torch.device("cpu")


def test_rotation_operands_are_cached_on_the_codec_instance():
    emulator = TurboQuantEmulator(head_size=64, device=CPU)
    assert emulator.h16 is emulator.h16
    assert emulator.codec_tables is emulator.codec_tables
    assert emulator.rotation is emulator.rotation
    other = TurboQuantEmulator(head_size=64, device=CPU)
    assert emulator.h16 is not other.h16
    assert torch.equal(emulator.h16, other.h16)


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
    codes, scales = emulator.quantize_uniform(rotated)
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
        codes, scales = emulator.quantize_uniform(rotated)
    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float32


def test_quantize_survives_a_rotation_that_overflows_fp32(emulator):
    """Garbage in: the scale plane must still hold finite numbers."""
    overflowed = emulator.rotate(torch.full((2, HEAD_SIZE), float(2.0**125)))
    assert not torch.isfinite(overflowed).all(), "this case is about the overflow"
    codes, scales = emulator.quantize_uniform(overflowed)
    assert torch.isfinite(scales).all()
    assert bool((codes.to(torch.int32).abs() <= -TURBOQUANT_NIBBLE_MIN).all())


def test_quantize_of_a_denormal_row_saturates_instead_of_underflowing(emulator):
    """A row that small quantises to zero codes; its scale must stay normal."""
    codes, scales = emulator.quantize_uniform(torch.full((2, HEAD_SIZE), float(2.0**-140)))
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


@pytest.mark.parametrize("head_size", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("fold", ["key", "butterfly", "value"])
def test_lloyd_max_preserves_all_leading_scale_dimensions(head_size, fold):
    emulator = TurboQuantEmulator(head_size=head_size, device=CPU)
    rotated = torch.ones(2, 3, 4, head_size)
    codes, scales = emulator.quantize_lloyd_max(rotated, fold)
    assert codes.shape == rotated.shape
    assert scales.shape == rotated.shape[:-1]
    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float32
    expected = torch.sqrt(torch.tensor(float(head_size)) + turboquant_codec.TURBOQUANT_LM_EPS)
    expected = (expected * emulator.inv_sqrt_len) * emulator.inv_sqrt_len
    assert torch.equal(scales, expected.expand_as(scales))


@pytest.mark.parametrize("fold", ["key", "butterfly", "value"])
def test_lloyd_max_keeps_the_production_256_channel_fp32_reduction_order(fold):
    """Scalar FP32 oracle for the two kernels' distinct reduction trees."""
    row = torch.randn(256, generator=torch.Generator().manual_seed(17))
    row *= torch.logspace(-3, 3, 256)

    def fp32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    squares = [fp32(float(value) ** 2) for value in row]
    if fold == "value":
        while len(squares) > 1:
            half = len(squares) // 2
            squares = [fp32(squares[i] + squares[i + half]) for i in range(half)]
    else:
        squares = [
            fp32(fp32(fp32(squares[i] + squares[i + 64]) + squares[i + 128]) + squares[i + 192]) for i in range(64)
        ]
        for stride in (1, 2, 4, 8, 16, 32):
            squares = [fp32(squares[i] + squares[i ^ stride]) for i in range(64)]
    assert float(TurboQuantEmulator._lloyd_max_sumsq(row, fold)) == squares[0]


@pytest.mark.parametrize("width", [0, 3, 63, 192])
def test_lloyd_max_refuses_invalid_reduction_widths(width):
    with pytest.raises(ValueError, match="nonempty power-of-two"):
        TurboQuantEmulator._lloyd_max_sumsq(torch.ones(width), "key")


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("fold", ["key", "value"])
def test_codec_helpers_select_the_active_writer_codebook(monkeypatch, mode, fold):
    monkeypatch.setattr(turboquant_codec, "TURBOQUANT_QUANT_MODE", mode)
    emulator = TurboQuantEmulator(head_size=256, device=CPU)
    x = torch.randn(3, 4, 256, generator=torch.Generator().manual_seed(18))
    rotated = emulator.rotate(x)
    if mode == 1:
        codes, scales = emulator.quantize_lloyd_max(rotated, fold)
        reconstructed = emulator.dequantize_lloyd_max(codes, scales)
    else:
        codes, scales = emulator.quantize_uniform(rotated)
        reconstructed = emulator.dequantize_uniform(codes, scales)
    packed, actual_scales = emulator.encode(x, fold=fold)
    assert torch.equal(packed, emulator.pack(codes))
    assert torch.equal(actual_scales, scales)
    assert torch.equal(emulator.dequantize(codes, scales), reconstructed)
    assert torch.equal(emulator.decode(packed, actual_scales), emulator.inverse_rotate(reconstructed))


@pytest.mark.parametrize("head_size", [32, 64, 128, 256, 512])
def test_rotation_matches_its_matrix_and_is_independent_of_ingestion_batch(head_size):
    emulator = TurboQuantEmulator(head_size=head_size, device=CPU)
    x = torch.randn(17, 4, head_size, generator=torch.Generator().manual_seed(19))
    rotated = emulator.rotate(x)
    torch.testing.assert_close(rotated, x @ emulator.rotation.T, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        emulator.inverse_rotate(rotated),
        rotated @ emulator.rotation * emulator.inv_sqrt_len,
        atol=1e-4,
        rtol=1e-4,
    )
    split = torch.cat([emulator.rotate(x[:1]), emulator.rotate(x[1:16]), emulator.rotate(x[16:])])
    padded = emulator.rotate(torch.cat([x, x[-1:].expand(15, -1, -1)]))
    assert torch.equal(rotated, split)
    assert torch.equal(rotated, padded[: len(x)])


def test_backend_name_is_the_enum_member_it_registers_as():
    name = AscendTurboQuantBackend.get_name()
    assert name == "TURBOQUANT"
    # vLLM indexes the enum with whatever get_name returns; a name that is not a
    # member raises there, which is why this backend does not carry its own.
    assert AttentionBackendEnum[name] is AttentionBackendEnum.TURBOQUANT
    # ... and the member resolves back to this class rather than to vLLM's own
    # unrelated TurboQuant backend, because the module registered an override.
    assert AttentionBackendEnum.TURBOQUANT.get_class() is AscendTurboQuantBackend
