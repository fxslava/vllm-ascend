Executing now — overwriting the script with the Leech/E8/B4/S3 roster, pasting the provided Golay/Leech decoder (with only the parity-fix loop vectorized, same math, since all Golay words have even weight and only the ±2 correction is reachable), and launching the benchmark immediately after.

Now running the ablation exactly as commanded:

Shape bug in the codes concatenation — fixing to reshape both parts to [N, ·] first:

The benchmark ran, but the Leech rows are demonstrably broken — 10.5 dB is far below even the plain integer lattice Z²⁴, so this is a decoder bug, not a lattice property, and I won't report those numbers as Λ₂₄. The output itself exposes two concrete bugs:

1. The pasted `B` matrix does not generate G₂₄: the printed weight distribution [0,6,8,10,12,14,16,18,24] is impossible for the [24,12,8] Golay code (weights must be only {0,8,12,16,24}).
2. A √2 scale bug: leech_nearest(x) returns y/√2 where y is nearest to x in Z²⁴-scale, so it quantizes x/√2, shrinking the reconstruction by √2 — the observed MSE ≈ (1−1/√2)² confirms it exactly. The codec must call it at sub·λ·√2.

Fixing both in one pass — selecting the standard bordered QR-circulant B by its fingerprint (self-orthogonality + weight distribution), verifying the corrected decoder's measured NSM, then immediately rerunning:

B4 = border + (C + I) (the QR-residue circulant with diagonal flipped) is the true G₂₄ generator — self-orthogonal, rank 12, weight distribution exactly {0,8,12,16,24}. The pasted matrix fails all three fingerprints. Patching the script: QR-generated B, the √2 scale fix, and measured NSM in provenance:

Now the √2 scale fix in the Leech codec (call the decoder at λ·√2 so the returned y/√2 lives in u·λ scale):

Now add measured-NSM validation to build_codecs and provenance (replacing the purely nominal values):

Quick functional check (1 seed) to confirm the fixed decoder's measured NSM, then the full run:

The fixed decoder is confirmed functional: measured Leech NSM 0.06682 vs the optimal 0.06577 (within 1.6% — the two-coset projection is essentially an optimal decoder), E8 0.07163 vs 0.07166, and single-seed SNR 23.51 dB for Leech vs 23.20 for E8. The NaN t-stats are just the 1-seed smoke artifact. Running the full benchmark now:

Full run complete with consistent results across all κ. One final quick computation the report needs — the direct Leech-vs-E8 paired t (the tables only test against B4):

The benchmark is complete. The strict six-codec lattice roster ran with seeds=8, seq_len=512, κ ∈ {1, 8, 64}, and the results are written to [mla_vq_results.json](/D:/Projects/vllm-ascend/mla_vq_results.json). One thing you need to know about the pasted decoder: it could not be used as-is — its first benchmark came out at 10.5 dB (worse than plain scalar 3-bit), and the run's own provenance line exposed why: the hand-typed B rows generated weights {0,6,8,10,12,14,16,18,24}, which is not the [24,12,8] Golay code (no weight-6 word exists in G₂₄), and leech_nearest(x) returns y/√2 with y nearest to x in Z²⁴-scale, so calling it at u·λ silently shrinks the reconstruction by √2. I replaced B with the bordered QR-residue circulant of 11 (fingerprint-verified at import: self-orthogonal, rank 12, weights exactly {0,8,12,16,24}) and call the decoder at λ·√2. The corrected decoder's measured NSM is 0.06682 vs the Leech optimum 0.06577 (within 1.6%), so the two-coset projection is now functionally the Leech lattice.

### MLA D=512, S=512, κ=1, 8 seeds (κ=8 and κ=64 order identically)

| codec | rate b/c | QK^T NMSE | softmax KL | KL p99 | cos (O) | rel L2 (O) | KV SNR dB | t(KL) vs B4 | t(O) vs B4 |
|---|---|---|---|---|---|---|---|---|---|
| B4 scalar LM16 (4b) | 4.031 | 4.73e-3 | 3.685e-5 | 4.51e-5 | 0.99521 | 0.09754 | 20.25 | — | — |
| L1_E8 (4b) | 4.031 | 2.395e-3 | 1.867e-5 | 2.273e-5 | 0.99762 | 0.06877 | 23.20 | −205 | −19.7 |
| L2_Leech24 (4b) | 4.031 | 2.235e-3 | 1.742e-5 | 2.129e-5 | 0.99779 | 0.06634 | 23.51 | −239 | −22.1 |
| S3 scalar LM8 (3b) | 3.031 | 1.722e-2 | 1.342e-4 | 1.639e-4 | 0.98358 | 0.17957 | 14.64 | +593 | +34.9 |
| L1_E8 (3b) | 3.031 | 9.585e-3 | 7.471e-5 | 9.127e-5 | 0.99073 | 0.13667 | 17.19 | +337 | +28.8 |
| L2_Leech24 (3b) | 3.031 | 8.940e-3 | 6.968e-5 | 8.552e-5 | 0.99106 | 0.13454 | 17.49 | +399 | +23.6 |

### Findings