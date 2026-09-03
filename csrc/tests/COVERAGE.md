# Validation coverage: C++ kernel suite vs. the Python test suite

Audit of what `csrc/tests/kernels/test_*.cpp` checks against what the Python
suite checks, for the five operator families the Ascend 310P3 fp16 decode path
uses. Written against the tree at `d6a7d0636`.

**Headline:** the two suites do not overlap. For all five operators the Python
unit tests replace the NPU operator with a mock and assert only that it was
called; no Python test in this repository checks the numerics of
`npu_rms_norm`, `npu_swiglu`, `npu_apply_rotary_pos_emb`, `aclnnMatmul` or the
310P 5-D NZ KV cache write on hardware. The C++ suite is the only numerical
check of any of them. The Python suite is nevertheless stronger on three axes
the C++ suite does not touch at all: dtype breadth, ragged shapes, and sentinel
handling in the slot mapping.

---

## 1. What each suite actually runs

### C++ (`csrc/tests/kernels/`)

| Operator | Binary | Device parity cases | Host-only cases | Shape matrix |
| --- | --- | ---: | ---: | --- |
| MatMul (`aclnnMatmul`) | `test_matmul_310p` | 18 | 4 | M=1; K∈{2048, 4096}; N∈{2048, 4096, 11008} |
| RMSNorm (`aclnnRmsNorm`) | `test_rmsnorm_310p` | 48 | 3 | tokens∈{1, 7, 32, 128}; hidden∈{1536, 2048, 4096, 8192} |
| SwiGLU (`aclnnSwiGlu`) | `test_activation_swiglu_310p` | 48 | 4 | tokens∈{1, 7, 32, 128}; intermediate∈{4096, 8960, 11008, 14336} |
| RoPE (`aclnnApplyRotaryPosEmbV2`) | `test_rotary_embedding_310p` | 27 | 4 | tokens∈{1, 16, 32, 128}; head_dim∈{64, 128}; half + interleave; θ∈{1e4, 1e6} |
| KV cache write (`aclnnScatterPaKvCache`) | `test_paged_attention_310p` | 7 | 5 | 7 GQA × block-size × context-length combinations |
| Paged attention decode | `test_paged_attention_310p` | 14, **all skipped** | — | — |

Every device case compares against a naive fp32 CPU reference in
`common/cpu_reference.cpp`, on inputs pre-rounded to fp16 so the device and the
reference start from bit-identical values.

The 14 paged-attention decode cases are unconditional `GTEST_SKIP`s:
`aclnnPagedAttention` does not exist on CANN 9.1.0 (`torch_npu._npu_paged_attention`
is `atb::PagedAttentionOperation` in `libatb.so`, a C++ object API the aclnn
launch path cannot drive), and `aclnnIncreFlashAttentionV4` expects a different
paged KV layout from the 310P 5-D NZ cache. **Decode attention numerics are
unverified in both suites.** This is the single largest hole in the coverage.

### Python

**Unit tests — `tests/ut/`. Numerics: none.**

| Path | What it does |
| --- | --- |
| `tests/ut/ops/test_layernorm.py` | Patches `torch_npu.npu_rms_norm` with `lambda x, w, eps: (x + 1, None)`. Asserts the mock was called and that `out == x + 1`. The non-310P variant is additionally `@pytest.mark.skip`-ped outright; the 310P variant is `skipif(not is_310p_hw())`. |
| `tests/ut/ops/test_activation.py` | Patches `torch_npu.npu_swiglu` with `lambda x: x + 1`. Asserts dispatch and argument identity. |
| `tests/ut/ops/test_rotary_embedding.py` | Patches `torch.ops.vllm.npu_rotary_embedding`. Asserts delegation and `__init__` signature compatibility with upstream. |
| `tests/ut/_310p/ops/test_rotary_embedding_310.py` | mrope cos/sin slice buffer reuse (`data_ptr()` identity) and a drafting flag. No arithmetic. |
| `tests/ut/_310p/attention/test_attention_v1_310.py` | Patches `torch_npu._npu_paged_attention` and `_npu_reshape_and_cache`. Asserts dispatch, and that `get_kv_cache_shape(10, 20, 30, 40) == (2, 10, 75, 20, 16)`. |
| MatMul | No unit test of the operator exists. `tests/ut/ops/test_linear.py` covers TP wiring and `npu_format_cast`, not arithmetic. |

**Nightly hardware tests — `tests/e2e/nightly/.../singlecard_ops/`. Real numerics, different operators.**

| Test | Operator it actually exercises | Overlap with the five |
| --- | --- | --- |
| `test_batch_matmul_transpose.py` | `torch.ops._C_ascend.batch_matmul_transpose`, a custom csrc kernel | Not `aclnnMatmul` / `F.linear` |
| `test_add_rms_norm_bias.py` | `torch.ops._C_ascend.npu_add_rms_norm_bias`, the add+norm+bias fusion | Not `npu_rms_norm` |
| `test_pa_kv_cache_ops.py` | `torch_npu.npu_scatter_pa_kv_cache` on the **generic 4-D** cache `[num_blocks, block_size, num_heads, head_dim]` | Same operator, **different layout** — never the 310P 5-D NZ shape |
| `triton/test_rope.py` | `vllm_ascend.ops.triton.rope`, a Triton kernel | Not `npu_apply_rotary_pos_emb` |
| `test_dequant_swiglu_quant.py` | `npu_dequant_swiglu_quant`, the int8 path | Not fp16 `npu_swiglu` |

`tests/e2e/nightly/310p/` contains only gated-delta-rule operators. **No Python
test in the repository is both 310P-gated and numerical for any of the five.**

---

## 2. How output tensors are verified

| | C++ suite | Python suite |
| --- | --- | --- |
| Reference | Naive fp32 CPU reference, written to match the v200 vector unit's fp32 accumulation | Mock (all UTs); NPU-computed golden (`batch_matmul_transpose`, `triton/test_rope`); CPU golden (`add_rms_norm_bias` only) |
| Predicate | Elementwise `\|a−e\| ≤ atol + rtol·\|e\|`, the `torch.allclose` predicate, in `tensor_compare.hpp` | `torch.testing.assert_close` (elementwise), with two exceptions below |
| Diagnostics on failure | Element count, mismatch count, non-finite count, max abs error + index, max rel error + index, first mismatch with both values, and the recorded *rationale* for the tolerance | pytest's default mismatch summary |
| Non-finite handling | Explicit: NaN==NaN and same-signed Inf agree, everything else is a mismatch, and the count is reported separately | `assert_close` default (`equal_nan=False` unless set) |
| Input conditioning | Inputs pre-rounded to fp16 so both sides start bit-identical; the reference output is re-rounded to fp16 before comparison | `torch.randn` in the target dtype; golden usually computed in the same dtype |

Two Python tests use a **different and weaker criterion** than elementwise
`atol`/`rtol`, which is worth knowing before treating their tolerances as
comparable:

- `test_batch_matmul_transpose.py` reduces to a global max-norm:
  `max_diff ≤ atol` and `max_diff / max|expected| ≤ rtol`. That is an L∞
  *relative-to-the-largest-element* check. A small-magnitude output element can
  be arbitrarily wrong in relative terms and still pass, as long as its absolute
  error stays under `atol`. The apparently very tight `1e-5` is only reachable
  at all because the golden is `torch.bmm` **on the same NPU**, so any
  systematic hardware rounding cancels on both sides.
- `test_add_rms_norm_bias.py` splits by magnitude — `assert_close(y*(y>1), …, atol=atol, rtol=100)`
  and `assert_close(y*(y≤1), …, rtol=rtol, atol=100)` — applying `atol` to large
  values and `rtol` to small ones, which is the opposite of the usual pairing.
  The masked-out entries are zero on both sides and pass trivially.

The C++ tolerances, by contrast, are named constants carrying a written
rationale (`kFp16DefaultTolerance` = 1e-3/1e-3, "≈1 fp16 ULP near unit scale";
`kPagedAttentionTolerance` = 4e-3, "re-tune on first hardware run"), and the KV
cache write uses `{0.0, 0.0, "cache write is a pure copy of fp16-exact values"}`
— a bit-exactness requirement, matching `test_pa_kv_cache_ops.py`'s `atol=0, rtol=0`.

---

## 3. Numerical edge cases

### Covered by C++, not by Python

- **Zero-magnitude rows through RMSNorm.** An all-zero row makes ε the only term
  under the square root; with ε=1e-6 the reciprocal is 1e3, large enough that a
  kernel computing `rstd` in fp16 rather than fp32 would saturate. Checked for
  finiteness on both `y` and `rstd`.
- **All-zero weights through MatMul.** Must give exactly zero with no NaN
  leaking from an uninitialised accumulator or a padded K tile. Needs no
  reference, so it isolates the kernel from the host arithmetic.
- **silu saturation.** Gates at ±30 with a unit up-projection: the positive tail
  must converge to the identity and the negative tail to zero, with the bound
  above one fp16 ULP at magnitude 30 (0.03125) so a correctly-rounded result
  cannot fail.
- **Algebraic invariants that need no reference at all** — RMSNorm invariance
  under row scaling by 4 (exact in fp16), MatMul linearity under scaling by 2,
  RoPE pair-norm preservation, RoPE identity at position 0, SwiGLU zeroing when
  the up half is zero. These catch a kernel that folds a scale or splits an
  input at the wrong offset even when a random-input comparison stays inside
  tolerance.
- **Cache offset uniqueness.** Every `(block, offset, kv_head, dim)` in the 5-D
  NZ layout maps to a distinct in-bounds index, verified exhaustively. A
  collision would alias two heads and surface downstream only as "accuracy is
  slightly worse".
- **Scattered access patterns.** Block tables are drawn from a Fisher-Yates
  shuffle so logical order never matches physical order; RoPE positions follow
  `(i·37 + 11) mod 4096` so a kernel that ignores the per-token cos/sin row
  cannot pass by accident.

### Covered by Python, not by C++

- **`slot_mapping = -1` as a skip sentinel.** `test_pa_kv_cache_ops.py` checks
  both a mixed `[0, -1, 3]` mapping and an all-`-1` mapping, asserting the cache
  is left bit-identical where the sentinel appears. The C++ suite only ever
  passes valid slots. This is a real behavioural contract — the model runner
  emits `-1` for padded slots — and it is untested on the 310P layout.
- **bf16 and fp32.** `add_rms_norm_bias` runs all three dtypes with per-dtype
  tolerances (fp16 1.0986e-3, bf16 7.9346e-3, fp32 2.4414e-4); `triton/test_rope`
  runs bf16 and fp16. The C++ suite is fp16-only.
- **Degenerate shapes.** `batch_matmul_transpose` covers `(1,1,1,1)`, `K=1`,
  `N=1`, `(2,3,4,5)`. The C++ MatMul suite has no degenerate case.
- **Partial rotary dim.** `triton/test_rope` covers `rotary_dim=32` inside
  `head_size=64`, i.e. the pass-through tail. `reference::ApplyRotaryPosEmb`
  supports `rotary_dim < head_dim`, but every C++ device case passes
  `rotary_dim == head_dim`, so the pass-through path is exercised on the host
  reference only.

### Covered by neither

- Decode attention numerics (see above).
- Denormal inputs, and inputs that overflow fp16 to Inf.
- ε values other than 1e-6.
- The quantised W8A8 / int8-KV paths, chunked prefill, and the splitfuse
  attention variants.

**On the fp16-only choice:** this is defensible rather than an oversight. The
310P (DaVinci v200) path forces fp16 throughout — `_310p/attention/attention_mask.py`
builds fp16 masks, `_310p/ops/fla/chunk_gated_delta_rule.py` raises on any other
dtype, `_310p/fused_moe/grouped_topk_router.py` casts logits to fp16. What is
missing is that **no test asserts the restriction**, so nothing fails if a bf16
tensor reaches a 310P kernel.

---

## 4. Tiling and shape variation against real inference dimensions

### Non-power-of-two and ragged shapes

| | C++ | Python |
| --- | --- | --- |
| Token counts | 1, 7, 32, 128 (7 is the only ragged value) | 1, 16, 64, **77**, 128, **255**, **1000** (`add_rms_norm_bias`); 1, 4, 8, 16, **1024** (`triton/test_rope`) |
| Hidden / column widths | 1536, 2048, 4096, 8192 — all 16-aligned by construction | 8, 16, 128, **3000**, 7168, **15000** (`add_rms_norm_bias`) |
| Context lengths | 1, 63, 64, 200, 300, 512 — straddling both 64 and 128 block boundaries | n/a |

Python is clearly stronger here. `3000` and `15000` columns and `77`/`255`/`1000`
rows exercise tail handling that the C++ matrix, whose widths are all multiples
of 16 by design, never reaches. The C++ suite makes that deliberate and says so
— `RmsNormShapes.QwenHiddenSizesAreBurstAligned` asserts every width is a whole
number of 32-byte MTE bursts — but "the shapes Qwen uses are aligned" is a
weaker claim than "the kernel handles a ragged tail", and only the latter
protects against a model with an unusual hidden size.

The C++ suite is stronger on **paged** shape variation: context lengths that
land exactly on and just short of a block boundary, at both supported block
sizes, with partially-filled trailing blocks and per-sequence length variation.
Python has no equivalent.

### Coverage of the actual projection dimensions

The parity suite's MatMul matrix is the cartesian product of two hidden-size
lists, which means it reaches the attention projections and `gate_up`, but
structurally **cannot** reach two shapes every forward pass runs:

| Projection | Shape | In the parity matrix? |
| --- | --- | --- |
| `q/k/v` separate, `o_proj` | K, N ∈ {2048, 4096} | Yes |
| `gate_up` | K ∈ {2048, 4096}, N = 11008 | Yes |
| **fused QKV** | N = (n_q + 2·n_kv)·d — 6144 for 32h/8kv/d128, 4608 for 28h/4kv/d128 | **No** — N is neither a hidden size nor an MLP width |
| **`down_proj`** | K = 11008, N ∈ {2048, 4096} | **No** — `LinearInputSizes()` is {2048, 4096}, so K never exceeds 4096 |

`down_proj` is the deepest reduction in the model and the only projection whose
K exceeds its N; a K-tiling bug that only appears past 4096 would pass the whole
suite. Both are now covered by `bench_matmul_310p` (`kExtraProjectionShapes`),
which measures them but does not check them — adding `11008` to
`shapes::LinearInputSizes()` would close it for the parity suite as a one-line
change, at the cost of six more device cases.

### M > 1

`test_matmul_310p.cpp` documents this itself: every parity case is M=1, which is
a GEMV and does not exercise the cube unit's M tiling at all. `bench_matmul_310p`
now sweeps M ∈ {1, 32, 128, 512}, so the prefill regime is measured; it is still
not *verified*.

---

## 5. Where each suite wins

**C++ is stronger on:**

1. Numerics for these five operators — it is the only coverage that exists.
2. The 310P-specific layout contracts: the 5-D NZ cache shape, exhaustive offset
   uniqueness, the 64/128 block sizes, the `block_size · head_size ≤ 128·128`
   limit, the 32-element SwiGLU gate, the head_dim ∈ {64, 128} RoPE gate.
3. Reference-free algebraic invariants, which catch a class of bug that
   comparison against a reference does not.
4. Failure diagnostics, and tolerances that carry a written justification.
5. Paged-access realism: shuffled block tables, ragged context lengths, both
   block sizes.
6. Determinism and portability: `DeterministicRandom` hand-rolls Box-Muller and
   the uniform mapping because libstdc++ and libc++ disagree on the distribution
   classes, so a failure reproduces bit-for-bit anywhere.

**Python is stronger on:**

1. dtype breadth (bf16, fp32) with per-dtype tolerances.
2. Ragged and degenerate shapes.
3. Sentinel semantics in `slot_mapping`.
4. Partial `rotary_dim`.
5. Reach — it tests the whole dispatch path including the plugin's Python
   wrappers, which the C++ suite bypasses entirely. A regression in
   `AscendRMSNorm310.forward_oot`'s argument order is caught by the mock test
   and invisible to the C++ suite.

The two are complementary rather than redundant: Python checks *that the right
operator is called with the right arguments*, C++ checks *that the operator
computes the right answer*. Neither substitutes for the other.

---

## 6. Recommended next steps, in priority order

1. **Decode attention.** Wire `aclnnIncreFlashAttentionV4` (the verified
   prototype is in `common/aclnn_ops.hpp`) or link `libatb.so` for
   `atb::PagedAttentionOperation`. 14 skipped cases and the most arithmetically
   complex kernel in the path.
2. **`slot_mapping = -1`.** One additional C++ case, mirroring
   `test_pa_kv_cache_ops.py`, on the 5-D NZ layout.
3. **`down_proj` K.** Add `11008` to `shapes::LinearInputSizes()`.
4. **A ragged width per operator.** One non-16-aligned hidden size and one
   non-16-aligned token count, asserting either correct tail handling or a clean
   rejection.
5. **Assert the fp16 restriction.** A test that a bf16 tensor reaching a 310P
   kernel fails loudly rather than silently.
6. **Partial `rotary_dim`.** A device case with `rotary_dim < head_dim`,
   checking the pass-through tail is untouched.
