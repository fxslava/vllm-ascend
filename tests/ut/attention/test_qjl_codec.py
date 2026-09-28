#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""The QJL "3 + 1" codec: that it moves no bytes, and what it does and does not fix.

Two independent claims are pinned here, and they pull in opposite directions.

*The layout is free, and it is also not a new quantizer.* Every packed byte still holds two
4-bit codes, the plane still packs to ``head_size // 2``, and the bytes are identical to the
ones the shipped writers emit -- checked against their own arithmetic and against
``turboquant_dequantize``, the host's only reader of the packed format. The price of that
freedom is that the reconstruction is identical too: reading the word as a sign plus a
magnitude cannot change a number the cache already decodes.

*The unbiased estimator is not free, and it is not about the layout.* Deterministic rounding
has a fixed per-key bias of about 14% of the dot product, and relabelling its bits does not
move it. Stochastic rounding of the level index removes it exactly, at a cost of sqrt(2) on
the single-draw error. These tests measure all three numbers rather than asserting the
direction of the effect.

Mirrors ``csrc/attention/turboquant/op_kernel/common/qjl_codec.h``, whose static_asserts prove
the same bit identities at compile time.
"""

import math
import unittest

import torch

from vllm_ascend.attention.qjl_reference import (
    QJL_AFFINE_BIAS,
    QJL_CODE_LEVELS,
    QJL_MAGNITUDE_LEVELS,
    QJL_NIBBLES_PER_BYTE,
    CachePlane,
    ScaleMode,
    SignPolarity,
    decode_asymmetric_uniform,
    decode_levels,
    encode_asymmetric_uniform,
    encode_deterministic,
    encode_stochastic,
    expected_reconstruction,
    inner_product_bias,
    is_negative,
    join_sign_magnitude,
    magnitude_index,
    pack_plane,
    reconstruct,
    roundtrip_through_cache,
    sign_bit,
    unpack_plane,
)
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_PACK_FACTOR,
    turboquant_dequantize,
)

HEAD_SIZE = 128
NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 2

# How far a per-key bias has to sit from zero before the test calls it a bias. The
# deterministic codecs measure about 12-14% of the mean |<q, k>| and the unbiased estimator
# measures 7e-7, so the gate is nowhere near either.
BIASED_FRACTION_OF_SIGNAL = 0.05
UNBIASED_FRACTION_OF_SIGNAL = 1e-4

# Monte-Carlo bias falls as 1 / sqrt(trials), so quadrupling the trials should halve it.
MC_TRIALS = 64
MC_TRIALS_RATIO = 4

# The long-tail fixture: Student-t degrees of freedom, and the outlier channels laid over it.
STUDENT_T_DF = 3
OUTLIER_CHANNELS = 3
OUTLIER_SCALE = 16.0

# An RMS scale's clipping residual has to beat the absmax scale's fp32 round-off by this much
# before the test believes it is clipping rather than noise.
CLIPPING_RESIDUAL_RATIO = 100.0


class TestQjlBitLayout(unittest.TestCase):
    """The packing: 1 sign bit + 3 magnitude bits, in the bytes the cache already holds."""

    def test_every_code_is_one_sign_bit_and_three_magnitude_bits(self):
        """All 16 words split into (sign, magnitude) and rejoin to themselves, both polarities."""
        nibbles = torch.arange(QJL_CODE_LEVELS, dtype=torch.int32)
        for polarity in SignPolarity:
            magnitude = magnitude_index(nibbles, polarity)
            negative = is_negative(nibbles, polarity)
            # The magnitude really is 3 bits wide and the sign really is 1.
            self.assertTrue(bool((magnitude >= 0).all() and (magnitude < QJL_MAGNITUDE_LEVELS).all()))
            self.assertTrue(bool(((sign_bit(nibbles) == 0) | (sign_bit(nibbles) == 1)).all()))
            # Together they carry the whole word: 8 positives and 8 negatives, each magnitude
            # used exactly twice. A split that lost a bit could not be onto.
            self.assertEqual(int(negative.sum()), QJL_CODE_LEVELS // 2)
            self.assertEqual(sorted(magnitude.tolist()), sorted(list(range(QJL_MAGNITUDE_LEVELS)) * 2))
            self.assertTrue(torch.equal(join_sign_magnitude(negative, magnitude, polarity), nibbles))

    def test_packed_bytes_are_bit_identical_to_the_shipped_writers(self):
        """The plane is byte-for-byte what the AIV and Cube encoders emit today."""
        generator = torch.Generator().manual_seed(7)
        bins = torch.randint(0, QJL_CODE_LEVELS, (32, HEAD_SIZE), generator=generator, dtype=torch.int32)
        low, high = bins[..., 0::QJL_NIBBLES_PER_BYTE], bins[..., 1::QJL_NIBBLES_PER_BYTE]

        # TurboQuantCodec<4>::StagePackedBytes stores `low + 16 * high - 128` as int8.
        aiv = (low + QJL_CODE_LEVELS * high - 128).to(torch.int8)
        self.assertTrue(torch.equal(pack_plane(bins, CachePlane.AIV_LLOYD_MAX), aiv))

        # TurboQuantModeCodec<KV4_FP8>::StageAffinePlane stages `bin - 8` through int4b_t,
        # whose two's-complement nibble has the bits of `bin ^ 8` in both halves.
        raw = (low ^ QJL_MAGNITUDE_LEVELS) | ((high ^ QJL_MAGNITUDE_LEVELS) << 4)
        cube = torch.where(raw >= 128, raw - 256, raw).to(torch.int8)
        self.assertTrue(torch.equal(pack_plane(bins, CachePlane.CUBE_KV4_FP8), cube))

    def test_roundtrip_is_bit_exact_over_the_whole_code_space(self):
        """pack -> unpack is the identity on every code, and the decode hits the shipped grids."""
        bins = torch.arange(QJL_CODE_LEVELS, dtype=torch.int32).repeat(HEAD_SIZE // QJL_CODE_LEVELS).unsqueeze(0)
        for plane in CachePlane:
            packed = pack_plane(bins, plane)
            self.assertEqual(packed.dtype, torch.int8)
            self.assertTrue(torch.equal(unpack_plane(packed, plane), bins))

        # KV4_FP8's mid-rise grid: TurboQuantModeTraits<KV4_FP8>::kCentroids[b] == b - 7.5.
        cube = decode_levels(bins, CachePlane.CUBE_KV4_FP8)
        self.assertTrue(torch.allclose(cube, bins.to(torch.float32) - QJL_AFFINE_BIAS))
        # No exact zero level, which is why the stored sign bit survives into the dot product.
        self.assertTrue(bool((cube != 0.0).all()))

        # The AIV word decodes to the Lloyd-Max table, read through the shipped reader.
        aiv_packed = pack_plane(bins, CachePlane.AIV_LLOYD_MAX)
        self.assertTrue(torch.equal(turboquant_dequantize(aiv_packed), decode_levels(bins, CachePlane.AIV_LLOYD_MAX)))

    def test_cache_footprint_is_unchanged(self):
        """A full cache plane still packs to (blocks, block_size, kv_heads, head_size // 2) int8."""
        generator = torch.Generator().manual_seed(11)
        vectors = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, generator=generator)
        recon, packed = roundtrip_through_cache(vectors, encode_deterministic)

        self.assertEqual(packed.dtype, torch.int8)
        expected_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        self.assertEqual(tuple(packed.shape), expected_shape)
        self.assertEqual(packed.numel(), vectors.numel() // TURBOQUANT_PACK_FACTOR)
        self.assertEqual(tuple(recon.shape), tuple(vectors.shape))

    def test_the_layout_is_a_relabelling_not_a_new_quantizer(self):
        """The 3+1 decode reproduces the shipped mid-rise reconstruction exactly.

        This is the 2026-09-22 verdict as a regression test: a sign bit plus a 3-bit magnitude
        spans the same 16-point grid a signed int4 word does, so the packing cannot change a
        reconstructed value. If this ever fails, the sign/magnitude split stopped agreeing with
        the grid the kernels decode -- not that a better codec was found.
        """
        generator = torch.Generator().manual_seed(13)
        vectors = torch.randn(256, HEAD_SIZE, generator=generator)
        bins, scale = encode_deterministic(vectors, ScaleMode.ABSMAX)
        shipped = (bins.to(torch.float32) - QJL_AFFINE_BIAS) * scale
        self.assertEqual(float((reconstruct(bins, scale) - shipped).abs().max()), 0.0)


class TestQjlInnerProductBias(unittest.TestCase):
    """The estimator: which encoders are biased on <q, k>, and by how much."""

    @staticmethod
    def _keys(kind, rows, generator):
        if kind == "gaussian":
            return torch.randn(rows, HEAD_SIZE, generator=generator)
        # Long-tail: Student-t with STUDENT_T_DF degrees of freedom (infinite kurtosis), built
        # from the seeded generator as z / sqrt(chi2_df / df) rather than drawn from
        # torch.distributions, which has no generator argument and would leave the fixture at
        # the mercy of global RNG state. A few channels are then scaled up, which is the regime
        # where a per-vector scale is set by an outlier and the bulk is crushed.
        normal = torch.randn(rows, HEAD_SIZE, generator=generator)
        chi_square = torch.randn(rows, HEAD_SIZE, STUDENT_T_DF, generator=generator).pow(2).mean(dim=-1)
        heavy = normal / chi_square.sqrt()
        heavy[:, :: HEAD_SIZE // OUTLIER_CHANNELS] *= OUTLIER_SCALE
        return heavy

    def _bias_rms(self, query, keys, recon):
        return float(inner_product_bias(query, keys, recon).pow(2).mean().sqrt())

    def test_deterministic_codecs_carry_a_per_key_bias_that_relabelling_does_not_remove(self):
        """Both 4-bit deterministic codecs are biased, on Gaussian and long-tail keys alike.

        The comparison is per key and deliberately not pooled over keys: on a symmetric key
        distribution the deterministic error averages towards zero *across* keys, which makes a
        pooled mean look almost unbiased while every individual logit is still off. E[k_hat | k]
        is what the softmax sees.
        """
        for kind in ("gaussian", "long_tail"):
            generator = torch.Generator().manual_seed(3)
            query = torch.randn(HEAD_SIZE, generator=generator)
            keys = self._keys(kind, 512, generator)
            signal = float(keys.matmul(query).abs().mean())

            bins, scale = encode_deterministic(keys)
            qjl_rms = self._bias_rms(query, keys, reconstruct(bins, scale))
            codes, asym_scale, zero_point = encode_asymmetric_uniform(keys)
            asym_rms = self._bias_rms(query, keys, decode_asymmetric_uniform(codes, asym_scale, zero_point))

            self.assertGreater(qjl_rms, BIASED_FRACTION_OF_SIGNAL * signal, f"{kind}: 3+1 deterministic")
            self.assertGreater(asym_rms, BIASED_FRACTION_OF_SIGNAL * signal, f"{kind}: asymmetric uniform")

    def test_stochastic_rounding_is_exactly_unbiased(self):
        """E[k_hat | k] = k in closed form, so E[<q, k_hat>] = <q, k> to fp32 round-off.

        Asserted on the expectation rather than on a sample mean: with an absmax scale rounded
        up through fp16 nothing clips, E[bin] = t identically, and the residual is only the
        round-off of dividing by the scale and multiplying it back.
        """
        for kind in ("gaussian", "long_tail"):
            generator = torch.Generator().manual_seed(5)
            query = torch.randn(HEAD_SIZE, generator=generator)
            keys = self._keys(kind, 512, generator)
            signal = float(keys.matmul(query).abs().mean())
            bias = inner_product_bias(query, keys, expected_reconstruction(keys, ScaleMode.ABSMAX))
            self.assertLess(float(bias.abs().max()), UNBIASED_FRACTION_OF_SIGNAL * signal, kind)

        # The guarantee is conditional on nothing clipping, and that is why the default scale is
        # ABSMAX. An RMS scale clips its tail, and a clipped coordinate keeps a bias no amount of
        # dithering removes; pinned here so the default cannot drift without this failing.
        generator = torch.Generator().manual_seed(5)
        keys = self._keys("long_tail", 512, generator)
        absmax_residual = float((expected_reconstruction(keys, ScaleMode.ABSMAX) - keys).abs().max())
        rms_residual = float((expected_reconstruction(keys, ScaleMode.RMS) - keys).abs().max())
        self.assertGreater(rms_residual, CLIPPING_RESIDUAL_RATIO * absmax_residual)

    def test_monte_carlo_bias_decays_as_one_over_sqrt_trials(self):
        """Sampling the dithered encoder converges on the exact dot product at the sqrt rate."""
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(HEAD_SIZE, generator=generator)
        keys = self._keys("gaussian", 512, generator)
        exact = keys.matmul(query)

        def mc_bias(trials):
            total = torch.zeros_like(exact)
            for _ in range(trials):
                bins, scale = encode_stochastic(keys, generator=generator)
                total += reconstruct(bins, scale).matmul(query)
            return float((total / trials - exact).pow(2).mean().sqrt())

        few = mc_bias(MC_TRIALS)
        many = mc_bias(MC_TRIALS * MC_TRIALS_RATIO)
        self.assertLess(many, few)
        # Halving per quadrupling, with room for sampling noise on 512 keys.
        self.assertAlmostEqual(few / many, math.sqrt(MC_TRIALS_RATIO), delta=0.75)

    def test_the_dither_trades_bias_for_variance(self):
        """The cost of unbiasedness: about sqrt(2) on the single-draw error.

        Recorded because it is the reason stochastic rounding is a Phase-2 measurement and not
        an automatic win -- a single decode is noisier than the deterministic one, and only
        averaging over many keys recovers the accuracy.
        """
        generator = torch.Generator().manual_seed(23)
        query = torch.randn(HEAD_SIZE, generator=generator)
        keys = self._keys("gaussian", 512, generator)

        bins, scale = encode_deterministic(keys)
        deterministic = self._bias_rms(query, keys, reconstruct(bins, scale))
        bins, scale = encode_stochastic(keys, generator=generator)
        stochastic = self._bias_rms(query, keys, reconstruct(bins, scale))

        self.assertGreater(stochastic, deterministic)
        self.assertAlmostEqual(stochastic / deterministic, math.sqrt(2.0), delta=0.3)


if __name__ == "__main__":
    unittest.main()
