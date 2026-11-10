# TurboQuant × vLLM-Ascend × Qwen3.5-27B — Integration Map

**Scope.** Architectural entry-point audit for integrating TurboQuant (rotation `Pi = D H D` applied
at runtime → 4-bit packed KV planes + burst-aligned fp32 scale plane) into the **Qwen3.5-27B**
hybrid model (GQA full attention + Gated DeltaNet), on branch `feat/tq-main-clean`.
No operator implementation code here — signatures, call trees, layout contracts, fusion
candidates, hardcodes, and a modification checklist.

**Headline finding.** TurboQuant is *already substantially wired into this branch* by four commits
(`63142af58` kernels, `0eae6aeeb` backend+cache-spec, `0f8b5814a` selection/validation,
`e3b7ae306` Qwen3.5 gated-layer routing). The remaining work for Qwen3.5-27B is **not** the
attention path — it is:

1. the **hybrid allocator conflict**: the TurboQuant three-plane carve only exists on the
   `not use_mamba` allocation path, while Qwen3.5's full-attention layers share one flat physical
   buffer with the GDN layers (`hybrid_with_attn_and_mamba`), so a TurboQuant run on Qwen3.5
   dies at the `len(raw_planes) == 3` assertion;
2. **build gating**: `CMakeLists.txt` compiles the entire TurboQuant kernel library (AIV *and*
   Cube) only for `SOC_VERSION ~ ascend950` — there are no kernels in a 910B build despite the
   backend docstring naming 910B;
3. the ingestion op accepts **fp16/bf16 activations**, not the int8-input ("a8") pipeline named in
   the task — `npu_turboquant_reshape_and_cache` is the existing entry and an int8-input variant
   would be a new schema.

---

## 1. End-to-End Call Graph

### 1.1 Full-attention layer (the 16 layers TurboQuant quantizes)

```
Qwen3_5ForConditionalGeneration.forward                     (upstream vllm, qwen3_5.py)
  └─ Qwen3_5Model.forward → per layer
       AscendQwen3_5DecoderLayer.forward                    vllm_ascend/patch/worker/patch_qwen3_5.py:121
         ├─ layer_type == "linear_attention"  → self.linear_attn(hidden_states)      ──► §1.3 (bypasses paged KV)
         └─ layer_type == "full_attention"    → AscendQwen3NextAttention.forward     patch_qwen3_5.py:66 (bound to
                                                                                       Qwen3NextAttention.forward:234)
              ├─ qkv, _ = self.qkv_proj(hidden_states)
              ├─ [mrope]  torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(qkv, q_w, k_w, cos_sin,
              │            num_q_heads, num_kv_heads, head_size, eps, mrope_section, ...)   :76   (q,k,v,gate out)
              ├─ [text]   qkv.split → q_norm/k_norm (Qwen3NextRMSNorm) → rotary_emb(positions,q,k)  :92-104
              │            ── RoPE stays in the UNROTATED basis; Pi is applied later, in-kernel
              ├─ if attn_output_gate and turboquant_fuses_output_gate(self.attn):     :106
              │     turboquant_gated_attention_forward(attn, q, k, v, gate)
              │       └─ torch.ops.vllm.turboquant_gated_attention(q,k,v,gate,out,layer_name)
              │            (custom op, graph-splitting; registered ops/turboquant_attention.py:94)
              │            └─ attn_layer.impl.forward(..., output_gate=gate)         turboquant_attention.py:71
              └─ else self.attn(q,k,v) → vllm Attention.forward
                     └─ torch.ops.vllm.unified_attention_with_output(...)  (graph-splitting op)
                          └─ impl.forward == AscendTurboQuantAttentionBackendImpl.forward
                                                                                       turboquant_v1.py:1529
                               ├─ reshape_and_cache(q,k,v,kv_cache,attn_metadata,out)  :784  ── INGESTION
                               │    ├─ _ensure_scale_cache(kv_cache)                  :432   (binds kv_cache[2])
                               │    ├─ _ensure_sink_cache(...)                       :517   (opt-in sinks)
                               │    ├─ key/value/slots → .contiguous(), slots→int32  :813-815
                               │    ├─ [AIV]  torch.ops._C_ascend.npu_turboquant_reshape_and_cache
                               │    │         (key, value, key_cache, value_cache, scale_cache,
                               │    │          slot_mapping, pi_signs, codec_tables)          :872-885
                               │    └─ [Cube] torch.ops._C_ascend.npu_turboquant_cube_reshape_and_cache
                               │              (..., mode_tables, int(mode))                    (KV3 only extras)
                               ├─ forward_impl(...)                                    :1501
                               │    ├─ DecodeOnly + cube → forward_cube_decode         :1332
                               │    │     torch.ops._C_ascend.npu_turboquant_cube_decode(query, gate,
                               │    │       pi_signs, codec_tables, hadamard16, key_cache, value_cache,
                               │    │       scale_cache, block_tables, seq_lens, workspace, query_rot,
                               │    │       num_kv_heads, num_heads, scale, output_stage, out[, mode_tables, mode])
                               │    ├─ DecodeOnly       → forward_paged_attention     :1159
                               │    │     npu_turboquant_rotate_q(query,pi,codec,h16 → query_rot fp32)
                               │    │     npu_turboquant_paged_attention(query_rot, key_cache, value_cache,
                               │    │         scale_cache, block_tables, seq_lens, codec_tables16, workspace,
                               │    │         num_kv_heads, num_heads, scale, out)
                               │    │     [sinks merge] [npu_turboquant_rotate_q un-rotate if not folded]
                               │    └─ PrefillNoCache   → _forward_prefill_no_cache   :1460
                               │          torch_npu.npu_fused_infer_attention_score(TND, dense, full precision;
                               │          V pre-rotated via npu_turboquant_rotate_q only if o_proj is Pi-folded)
                               └─ output[:n] = attn_output

C++ dispatch (identical for every op above):
  torch.ops._C_ascend.npu_turboquant_*                     TORCH_LIBRARY_FRAGMENT
    ├─ op_adapter/turboquant_torch_adpt.h  npu_turboquant_*()   TORCH_CHECK contracts → launcher
    ├─ op_kernel/*.cpp  turboquant_*_impl(AscendType, aclrtStream, blockDim, void* operands,
    │                     uint32 num_tokens/num_kv_heads/head_size/block_size/num_blocks, ...)
    │                     host tiling from op_host/turboquant_tiling.{h,cpp} (Plan* functions)
    └─ <<<>>> ASC launch on c10_npu::getCurrentNPUStream().stream()   (Davinci v3 / AIV + Cube)
```

### 1.2 Stock path (for contrast — what TurboQuant replaces)

```
impl = AscendAttentionBackendImpl                       vllm_ascend/attention/attention_v1.py:508
  reshape_and_cache(...)                                :1656
    └─ DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
         └─ torch_npu.npu_scatter_pa_kv_cache(cache_mode="Norm")   device/device_op.py:40-50
  forward_paged_attention(...)                          :1589
    └─ torch_npu._npu_paged_attention(query, key_cache, value_cache, num_kv_heads, num_heads,
                                      scale_value, block_table, context_lens, out)      :1597-1607
```
Both stock ops come from **torch_npu (CANN aclnn)**; they are not in this repo. There is no
`reshape_and_cache_flash` and no `paged_attention_v1/v2` wrapper anywhere in the plugin.

### 1.3 GDN / hybrid layer boundary (the 48 layers that bypass the paged KV cache)

```
AscendQwen3_5DecoderLayer.forward → self.linear_attn = QwenGatedDeltaNetAttention (upstream class,
                                   patched at import time — patch_qwen3_5.py:235-247)
  AscendGatedDeltaNetAttention.forward(hidden_states, output=None)     vllm_ascend/ops/gdn.py:181
    ├─ torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, b, a, core_attn_out, prefix, False)  :239
    └─ _forward_core(...)                                              gdn.py:263
         ├─ conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]                :296-297
         ├─ torch.ops._C_ascend.npu_causal_conv1d_custom(..., conv_state, ...)        :324,371,394,415
         ├─ DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)                     :437 (Triton)
         ├─ decode/spec: torch.ops._C_ascend.npu_recurrent_gated_delta_rule(q,k,v,g,beta,
         │                state=ssm_state, scale, cu_seqlens, ssm_state_indices, ...)  :469,493,577
         └─ prefill:  torch_npu.npu_chunk_gated_delta_rule(...)  (fused CANN)          :142
                       or Triton chunk_gated_delta_rule(...)                            :550
```

**GDN never touches `reshape_and_cache` or any paged-attention op.** Its state slot is
`block_table[:, 0]` of the request's row (`ops/gdn_attn_builder.py:527,583`). The states live in
`kv_cache[0]` (conv) / `kv_cache[1]` (ssm) views carved out of the **same physical int8 buffer**
as the attention K/V when the hybrid group shares a `KVCacheTensor` ("align" mamba cache mode) —
see §4.3. TurboQuant must never quantize these: the ssm state is fp32 recurrent state
(`patch/__init__.py:845-855` exists precisely because the AscendC ops must support that fp32
state); quantizing it would break the recurrence itself.

---

## 2. Exact Method Signatures

### 2.1 Python — backend (`vllm_ascend/attention/turboquant_v1.py`)

```python
class AscendTurboQuantAttentionBackend(AscendAttentionBackend):          # :240 descriptor
    @staticmethod get_name() -> "ASCEND_TURBOQUANT"
    @staticmethod get_impl_cls() -> type[AscendTurboQuantAttentionBackendImpl]
    @classmethod  supports_pcp() -> False
    @staticmethod get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                                     head_size: int, cache_dtype_str: str = "") -> tuple[int, ...]
        # KV4_FP8: (2, num_blocks, block_size, num_kv_heads, head_size // 2)
        # KV3_FP4: (2, num_blocks, block_size, 1, turboquant_packed_plane_bytes(head_size, num_kv_heads, mode))
    @staticmethod get_supported_kernel_block_sizes() -> [128]
    @staticmethod swap_blocks(src_kv_cache: list[Tensor], dst_kv_cache: list[Tensor],
                              src_to_dst: Tensor) -> None                # moves ALL planes incl. scale
    @staticmethod copy_blocks(kv_caches: list[Tensor], src_to_dists: Tensor) -> None

class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):  # :314
    def __init__(self, *args, **kwargs)         # inherits AscendAttentionBackendImpl.__init__
        # (num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
        #  kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name,
        #  sinks, **kwargs)  — attention_v1.py:509-523
    @property fuses_output_gate -> bool          # cube_decode and not output_rotation_folded  :371
    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None                  :381
    def reshape_and_cache(self, query, key, value, kv_cache: tuple[Tensor, ...],
                          attn_metadata: AscendMetadata, output: Tensor)                     :784
    def forward_paged_attention(self, query, attn_metadata, output: Tensor | None) -> Tensor # AIV decode :1159
    def forward_cube_decode(self, query, attn_metadata, output: Tensor,
                            output_gate: Tensor | None = None) -> Tensor                     # Cube     :1332
    def _forward_prefill_no_cache(self, query, key, value, attn_metadata, output) -> Tensor  :1460
    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output,
                     output_gate: Tensor | None = None)                                      :1501
    def forward(self, layer: AttentionLayer, query, key, value, kv_cache,
                attn_metadata, output=None, output_scale=None,
                output_block_scale=None, output_gate=None) -> Tensor                         :1529

def activate_turboquant_backend(layer: torch.nn.Module) -> None                             # :1607
    # swaps impl.__class__ in place (same mechanism as the C8 scheme);
    # sets layer.kv_cache_torch_dtype = torch.int8
def turboquant_cube_decode_available() -> bool                                              # :158
def turboquant_cube_decode_selected(model_dtype) -> bool                                    # :174
```

### 2.2 Python — gated-attention op (`vllm_ascend/ops/turboquant_attention.py`)

```python
def turboquant_fuses_output_gate(attn: torch.nn.Module) -> bool                             # :31
def turboquant_gated_attention_forward(attn, query, key, value, gate) -> Tensor             # :36
def turboquant_gated_attention(query, key, value, gate, output, layer_name) -> None         # :59 custom-op body
# registered: vllm::turboquant_gated_attention, mutates_args=["output"], dispatch_key="PrivateUse1"
```

### 2.3 Python — quantization registry entry (`vllm_ascend/quantization/methods/turboquant_v1.py`)

```python
@register_scheme("TurboQuant", "attention")
class AscendTurboQuantKVCacheAttentionMethod(AscendAttentionScheme):        # :47
    backend = AscendTurboQuantAttentionBackend
    def create_weights(self, layer) -> None            # layer.kv_cache_torch_dtype=int8; activate_turboquant_backend(layer)
    def process_weights_after_loading(self, layer) -> None
    def apply(self, layer, query, key, value, kv_cache, attn_metadata, attn_type,
              scale, output) -> Tensor                 # raises — handled by the backend, never called
```

### 2.4 Torch-library op schemas (`csrc/attention/turboquant/op_adapter/turboquant_torch_ops.h`)

Registered into `_C_ascend` via `TORCH_LIBRARY_FRAGMENT` (included exactly once from
`csrc/torch_binding.cpp:2713`); PrivateUse1 kernel dispatch, not pybind11 `MODULE`:

```
npu_turboquant_reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache,
                                 Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs,
                                 Tensor codec_tables) -> ()
npu_turboquant_rotate_q(Tensor query, Tensor pi_signs, Tensor codec_tables, Tensor hadamard16,
                        Tensor! query_rot) -> ()
npu_turboquant_paged_attention(Tensor query_rot, Tensor key_cache, Tensor value_cache,
                               Tensor scale_cache, Tensor block_tables, Tensor context_lens,
                               Tensor codec_tables, Tensor! workspace, int num_kv_heads,
                               int num_heads, float scale_value, Tensor! out) -> ()
npu_turboquant_vector_core_num() -> int
npu_turboquant_workspace_size(int num_tokens, int num_heads, int head_size,
                              int max_blocks_per_seq, int block_size) -> int
# under VLLM_ENABLE_TURBOQUANT_CUBE (Ascend 950 builds only):
npu_turboquant_cube_reshape_and_cache(... , Tensor? mode_tables=None, int mode=4) -> ()
npu_turboquant_cube_decode(Tensor query, Tensor? gate, Tensor pi_signs, Tensor codec_tables,
                           Tensor hadamard16, Tensor key_cache, Tensor value_cache,
                           Tensor scale_cache, Tensor block_tables, Tensor context_lens,
                           Tensor! workspace, Tensor! query_rot, int num_kv_heads, int num_heads,
                           float scale_value, int output_stage, Tensor! out,
                           Tensor? mode_tables=None, int mode=4) -> ()
npu_turboquant_cube_workspace_size(int num_tokens, int num_heads, int num_kv_heads,
                                   int head_size, int max_blocks_per_seq, int block_size) -> int
```

### 2.5 ATen adapter + device launcher (`op_adapter/turboquant_torch_adpt.h`, `op_kernel/*.cpp`)

```cpp
inline void npu_turboquant_reshape_and_cache(at::Tensor &key, at::Tensor &value,
    at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &scale_cache,
    at::Tensor &slot_mapping, at::Tensor &pi_signs, at::Tensor &codec_tables);        // :200
inline void npu_turboquant_rotate_q(at::Tensor &query, at::Tensor &pi_signs,
    at::Tensor &codec_tables, at::Tensor &hadamard16, at::Tensor &query_rot);          // :274
inline void npu_turboquant_paged_attention(at::Tensor &query_rot, at::Tensor &key_cache,
    at::Tensor &value_cache, at::Tensor &scale_cache, at::Tensor &block_tables,
    at::Tensor &context_lens, at::Tensor &codec_tables, at::Tensor &workspace,
    int64_t num_kv_heads, int64_t num_heads, double scale_value, at::Tensor &out);     // :331
inline void npu_turboquant_cube_reshape_and_cache(..., const c10::optional<at::Tensor> &mode_tables,
    int64_t mode);                                                                     // :507
inline void npu_turboquant_cube_decode(at::Tensor &query, const c10::optional<at::Tensor> &gate,
    ..., int64_t output_stage, at::Tensor &out, const c10::optional<at::Tensor> &mode_tables,
    int64_t mode);                                                                     // :578

// host launchers (all void*, stream from c10_npu::getCurrentNPUStream(); ASC launch <<<>>>):
turboquant_reshape_and_cache_impl(AscendType, aclrtStream, blockDim,
    key*, value*, key_cache*, value_cache*, scale_cache*, slot_mapping*, pi_signs*,
    codec_tables*, num_tokens, num_kv_heads, head_size, block_size, num_blocks,
    tokens_per_core, inv_sqrt_len)                       // in turboquant_paged_attention.cpp:757
turboquant_rotate_q_impl(...)                            // turboquant_rotate_q.cpp
turboquant_paged_attention_impl(...)                     // turboquant_paged_attention.cpp
turboquant_mm_reshape_and_cache_impl(...)                // cube writer
turboquant_mm_fused_decode_impl / _raw_query_impl(...)   // cube decode (kv3 / kv4)
turboquant_mm_fused_decode_opt_impl / _opt_raw_query_impl(...)  // built+exported, NOTHING calls them yet
```
FakeTensor/meta registrations: `csrc/torch_binding_meta.cpp:2063-2073` +
`vllm_ascend/meta_registration.py:83-103` (no-op meta kernels so export/FakeTensor works).

### 2.6 Cache-spec & rotation helpers

```python
# vllm_ascend/core/kv_cache_interface.py:234
@dataclass(frozen=True, kw_only=True)
class AscendTurboQuantAttentionSpec(FullAttentionSpec):
    pack_factor: int = 2; mode: int = KV4_FP8; burst_floats: int = 8
    scale_slot_floats -> int        # cdiv(2*num_kv_heads, 8) * 8
    packed_bytes -> int             # head_size//2 (KV4) | 3*head_size//8 (KV3)
    payload_bytes -> int            # 2 * block_size * turboquant_packed_plane_bytes(...)
    scale_plane_bytes -> int        # block_size * scale_slot_floats * 4
    real_page_size_bytes -> int     # payload + scale
    @classmethod merge(cls, specs) -> Self
# registered → FullAttentionManager  (kv_cache_interface.py:335-339)

# vllm_ascend/attention/turboquant_layout.py (no vLLM imports; host mirror of C++ layout)
turboquant_packed_bytes(head_size, mode) -> int                       # :112
turboquant_packed_plane_bytes(head_size, num_kv_heads, mode) -> int   # :125 (KV3 pads row to 32 B)
turboquant_scale_slot(num_kv_heads) -> int                            # :226
turboquant_codec_tables(head_size, batch_rows, device) -> Tensor      # :240
turboquant_mode_tables(head_size, batch_rows, device) -> Tensor       # :136 (KV3 only; 1 or 8 rows)
turboquant_hadamard16(device) -> Tensor                               # :200 (16×16 fp16 Sylvester)
turboquant_dequantize(packed, dtype=fp32) -> Tensor                   # :298 (host-only reader)

# vllm_ascend/attention/turboquant_rotation.py
turboquant_pi_signs(head_size, device) -> Tensor                      # :72  LCG-seeded ±1 diagonal
apply_pi(x, pi_signs) -> Tensor; walsh_hadamard(x, dim) -> Tensor     # :117/:93 host reference
fold_pi_into_output_projection(weight, head_size) -> Tensor           # :125 offline o_proj fold
output_rotation_is_folded(hf_config, layer_name, head_size) -> bool   # :235 reads config marker
```

### 2.7 Stock ingestion wrapper (replaced by §2.1 on TQ layers)

```python
# vllm_ascend/device/device_op.py:40
class BaseDeviceAdaptor:
    @classmethod def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping)
        → torch_npu.npu_scatter_pa_kv_cache(key=..., value=..., key_cache=..., value_cache=...,
                                            slot_mapping=..., cache_mode="Norm")
```

---

## 3. Qwen3.5-27B Specific Audit

### 3.1 Architecture facts (from `Qwen/Qwen3.5-27B` `config.json`, fetched 2026-10-26)

| Item | Value | Consequence for TurboQuant |
| --- | --- | --- |
| layers | 64 = 48 `linear_attention` (GDN) + 16 `full_attention`, `full_attention_interval: 4` (L,L,L,F repeating) | TQ applies to 16 layers only; GDN layers pass through untouched |
| full-attn heads | `num_attention_heads: 24`, `num_key_value_heads: 4` (GQA 6:1) | `num_heads % num_kv_heads == 0` holds (adapter check `adpt :376`) |
| `head_dim` | **256** | `CheckHeadSize` accepts 64–256 powers of two ✔; KV4 packed = 128 B/head; KV3 requires exactly 256 ✔ (`platform.py:484-485`) |
| `attn_output_gate` | **true** (every full-attn layer) | Pi-into-`o_proj` fold **ruled out** (gate sits between attention and o_proj and does not commute with Pi — `turboquant_rotation.py:38-41`); decode must un-rotate on device (AIV) or gate inside the launch (Cube `output_stage=GATED`) |
| default dtype | bfloat16 | Cube decode requires fp16 (`turboquant_cube_decode_selected`, `turboquant_v1.py:174-180`) → **default 27B runs take the AIV decode**; gate then applied in torch (`forward_impl:1523-1526`), not fused |
| `max_position_embeddings` | 262144 | long-context decode splits; workspace growth rules in `_decode_workspace` apply |
| model_type / arch | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | multimodal → mrope path in the patched attention (`_uses_multimodal_rope`, `patch_qwen3_5.py:61-63`); `is_hybrid == True` |
| GDN | `linear_num_key_heads: 16`, `linear_num_value_heads: 48`, key/value head dim 128, `linear_conv_kernel_dim: 4` | state shapes from `MambaStateShapeCalculator.gated_delta_net_state_shape` (patched `get_state_shape`, gdn.py:159) |

Per-token byte math for one full-attn layer at `block_size=128`, KV4 (AIV):
payload 2 × 4 × 128 B = 1024 B + scale `round_up(8,8)=8` × 4 B = 32 B → **1056 B/token vs 4096 B bf16
(3.88×)**. `num_kv_heads=4` is a multiple of 4, so upstream's `int4_per_token_head` page arithmetic
matches exactly (`utils.py:1595-1598`). Whole-model reduction is diluted: only 16 of 64 layers are
full-attention, and the GDN ssm state (`48×128×128` fp32 ≈ 3.1 MB/seq/layer) dominates the hybrid
cache budget and is untouched by TQ.

### 3.2 Full-attention path contracts

- K/V reaching `reshape_and_cache` are **post-QKV-split, post-q/k-RMSNorm, post-RoPE**, shape
  `[num_tokens, 4, 256]`, activation dtype (bf16 default), and **must be made contiguous**
  (`turboquant_v1.py:813-815`) because a fused QKV projection hands over column slices.
- Order of transforms is fixed and correct: RoPE in the unrotated basis → Pi rotation inside the
  writer kernel. Never move Pi before RoPE.
- Decode is `DecodeOnly`-only: the paged operator takes one block-table row + one context length
  **per query token**; no mask/query-extent argument exists. Prefill is dense full-precision
  (`PrefillNoCache`), reads nothing from the 4-bit cache.
- `block_tables` `[num_tokens, max_blocks_per_seq]` int32, `context_lens` int32 (one per token),
  `slot_mapping` int32 with `-1` padding — all staged/validated via `_device_index`
  (`TURBOQUANT_INDEX_DTYPE = torch.int32`, `turboquant_v1.py:150-155`, `_device_index:1014`);
  `seq_lens_device` (runner's persistent int32 buffer) preferred over host `seq_lens`.
- Every GM operand must be **contiguous, 32-byte-burst aligned, on-device** (`TURBOQUANT_GM_BURST_BYTES=32`;
  violation surfaces as CANN error 264, not an alignment report). `slot_mapping` is read scalar-wise.
- Graph mode: TQ runs are forced to PIECEWISE (or NONE) — `_apply_turboquant_graph_mode`,
  `platform.py:556-614`; both attention entry ops are graph-splitting ops, TQ kernels run eagerly
  between captured pieces.

### 3.3 GDN / recurrence state isolation — findings

1. **Functional isolation is complete.** GDN forward uses `npu_causal_conv1d_custom`,
   `npu_recurrent_gated_delta_rule`, `npu_chunk_gated_delta_rule`, Triton
   `chunk_gated_delta_rule` — never `reshape_and_cache`/paged attention. Its metadata comes from
   `AscendGDNAttentionMetadataBuilder.build(...)` (`ops/gdn_attn_builder.py:513`), separate from
   the TQ backend's builders; state indices are `mamba block_table[:, 0]`.
2. **Physical co-residency depends on grouping.** When a `KVCacheTensor.shared_by` mixes MambaSpec
   and AttentionSpec layers, `_allocate_kv_cache_tensors` (`model_runner_v1.py:4232-4286`) hands
   every layer of the tensor **one flat int8 buffer**; `_reshape_kv_cache_tensors` slices
   `conv, ssm` from offset 0 and attention K/V from the tail after `conv_block_padding`
   (`:4788-4797, :4864-4884`; layout comment `:4963-4969`). One block-id space and one block table
   serve the whole group; `kernel_block_sizes=[0]` disables token-level slot mapping for mamba
   groups (`may_reinitialize_input_batch:4985-5030`).
3. **The alignment that creates co-residency is forced** by `patch/platform/patch_mamba_config.py`
   (`attn_single_token_k_page_size * attn_block_size == ssm_block_page_size`,
   `mamba_page_size_padded = attn_page_size + conv_block_page_size`, `mamba_cache_mode="align"`).
4. **TurboQuant does not fit this contract today**: the three-plane carve
   (`model_runner_v1.py:4349-4379`) is guarded by `not use_mamba` (`:4340`), and the reshape
   branch **asserts `len(raw_planes) == 3`** (`:4581`) — a hybrid shared tensor is a single flat
   tensor, so the run fails at reshape time (i.e., "crash under load" style, at engine start with
   cache init, not at config validation).
5. Additional fragility: `self.hybrid_with_attn_and_mamba` (`:4232-4240`) is a **global** flag set
   while iterating tensors — once any tensor mixes mamba+attn, later pure-attn tensors also take
   the flat single-tensor path, ordering-dependently.

### 3.4 Identified hardcodes / obstacles (current plugin)

| # | Location | Hardcode / guard | Impact on Qwen3.5-27B TQ |
| --- | --- | --- | --- |
| H1 | `model_runner_v1.py:4340` + `:4581` | TQ three-plane alloc only on `not use_mamba`; reshape asserts 3 planes | **Blocker.** Hybrid shared tensor cannot back a TQ layer |
| H2 | `CMakeLists.txt:118-135` | `vllm_ascend_turboquant` built only for `SOC_VERSION ~ ascend950` (AIV *and* Cube) | **Blocker on 910B.** Backend docstring claims 910B support; ops won't exist |
| H3 | `platform.py:448-553` `_validate_turboquant_config` | rejects chunked prefill, prefix caching, speculative decoding, `compress_ratios` models, `block_size % 16`, presets ≠ `int4_per_token_head`; KV3 adds fp16 + cube env + head 256 + `block % 64` + v1 runner | No qwen3_5-specific gate, but hybrid is **not rejected either** — H1 hits later at runtime. Spec-decode rejection matters for qwen3_5 MTP deployments |
| H4 | `platform.py:556-614` | full graph → PIECEWISE/NONE downgrade | Serving constraint, verified behavior statement |
| H5 | `attention_v1.py:92-110` `get_impl_cls` | `turboquant_enabled(...)` → TQ impl; refuses DCP combination | Selection is config/env-driven, **not model-type-driven** — no qwen3_5 trap here |
| H6 | `attention_v1.py:127-147` `get_kv_cache_shape` | packed last dim `head_size//2` (or KV3 plane bytes) | Token-major layout `[num_blocks, block_size, num_kv_heads, head_size/2]` — **not** head-major |
| H7 | `utils.py:1214` | `is_hybrid` skips generic block-size logic | Qwen3.5 block sizing routed through mamba-align path (interacts with H1) |
| H8 | `model_runner_v1.py:4925-4933` | `k_cache_dtype = spec.dtype` (fa-quant / sparse-c8 overrides only) | For TQ the view path is the special case at `:4572-4611`; the generic `_view_raw_kv_cache` ENABLE_TURBOQUANT branch (`:4536-4542`) still over-allocates from an unpacked spec — only the `AscendTurboQuantAttentionSpec` route is exact |
| H9 | `patch_qwen3_5.py:61-63` | `"qwen3_5" in model_type and hasattr(rotary_emb, "mrope_section")` | mrope branch changes where q/k/v/gate come from (fused triton op) — gate extraction differs but TQ branch is agnostic |
| H10 | `ec_manager/score_ec_manager.py:165-170` | `mt == "qwen3_5"`: vision-layer count 27, distinct β formula | Irrelevant to TQ; noted as a qwen3_5 hardcode inventory item |
| H11 | `quantization/configs/modelslim_config.py:99-139` | `packed_modules_model_mapping` for `qwen3_5*` (weight-name mapping only) | Not a KV gate; TurboQuant registry key is `"TurboQuant"` per attention layer |
| H12 | `utils.py:1593-1603` | `TURBOQUANT_BLOCK_SIZE_MULTIPLE=16`; dtype strings `int4_per_token_head`, `turboquant_*`, `kv3fp4` | Activation routes: env `ENABLE_TURBOQUANT`, `--kv-cache-dtype`, or per-layer scheme (+cache_dtype, see `turboquant_enabled`) |
| H13 | cube decode constraints | fp16 activations, `block % 64`, NZ tiles, UB pre-flight `FusedDecodeUbBytes` ≤ 256 KiB | 27B at bf16 → AIV; `VLLM_ASCEND_TURBOQUANT_CUBE_DECODE=1` is default-on but inert without fp16 |
| H14 | sinks | `VLLM_ASCEND_TQ_SINK_TOKENS` refused on kv4fp8 Cube cache (host merge reads row-major Lloyd-Max grid only) | Diagnostic feature; keep off |
| H15 | KV-transfer connectors | hybrid connectors (ascend_store / mooncake hybrid/layerwise) assume the documented hybrid layout (conv/ssm head, K/V tail) | A TQ hybrid layout change (H1 fix) must guard these |

Environment variables (all in `vllm_ascend/envs.py`): `ENABLE_TURBOQUANT` (:80, 0),
`VLLM_ASCEND_TURBOQUANT_CUBE_DECODE` (:88, 1), `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS` (:96, 0),
`TURBOQUANT_CAPTURE_SIGNATURES` (:106), `TURBOQUANT_DRY_RUN` (:113), `TURBOQUANT_TRACE_PATH`
(:117), `TURBOQUANT_FLIGHT_RECORDER` (:126), `TURBOQUANT_RECORD_CASES` (:132),
`VLLM_ASCEND_TQ_SINK_TOKENS` (:151). Build-only: `VLLM_ENABLE_TURBOQUANT`,
`VLLM_ENABLE_TURBOQUANT_CUBE` (CMake).

---

## 4. Memory Layout & Allocation Contracts

### 4.1 Page geometry (per full-attn layer, KV4)

```
logical shape (get_kv_cache_shape): (2, num_blocks, block_size, num_kv_heads, head_size // 2)
physical allocation: THREE separate contiguous planes carved from ONE page budget
  raw_k     : int8  [num_blocks, block_size, num_kv_heads, head_size//2]   (packed K codes)
  raw_v     : int8  [num_blocks, block_size, num_kv_heads, head_size//2]   (packed V codes)
  raw_scale : fp32 [num_blocks, block_size, scale_slot]   scale_slot = cdiv(2*num_kv_heads, 8)*8
  carve: model_runner_v1.py:4349-4379 (integer byte arithmetic; planes sum to the allocation)
  view:   model_runner_v1.py:4572-4611 (kernel-shaped views; asserts whole pages)

KV3_FP4 (Cube-only): plane bytes per token = pad32(num_kv_heads * 3 * head_size/8);
  logical (2, num_blocks, block_size, 1, plane_bytes); NZ-tiled by the Cube writer/reader.
```

Stride order is **token-major inside the block**: `[num_blocks, block_size(token), num_kv_heads,
head_size/2]` — neither of the two head-major variants (`[num_blocks, heads, head/2, block]` /
`[num_blocks, heads, block, head/2]`) is used. Codes: 4-bit Lloyd-Max (16 levels) on the AIV path
(`turboquant_codec_tables`, writer stores `low + 16*high − 128` in an int8 byte); affine-fp8-grid
codes on the kv4fp8 Cube path. Scales are per-token per-head fp32, K lanes then V lanes, padded to
one 32-byte burst — the padding is the whole point of `AscendTurboQuantAttentionSpec`
(`kv_cache_interface.py:244-258`).

### 4.2 Sub-byte strides and block sizes

- No 4-bit torch dtype is ever allocated. Everything is **raw int8 bytes** viewed as
  int8/fp32 planes; the paged allocator accounts bytes via `real_page_size_bytes`
  (payload + scale). Sub-byte packing is therefore invisible to allocation — the contract is
  "byte budget per page", not element width. No 16-bit-minimum alignment constraint exists beyond
  the 32-byte GM burst rule on operand base addresses.
- Kernel block sizes: `[128]` (both stock and TQ backends). Config floor: `block_size % 16`
  (codec tile `TURBOQUANT_TILE_ROWS=16`); Cube decode reads 64-row tiles → `block % 64`;
  `ascend_config` warns when ≠ 128. Upstream's 16/32/64 are representable but off the verified
  path.

### 4.3 Hybrid (Qwen3.5) physical layout — the conflict

```
today, one KVCacheTensor shared by {linear_attn…, full_attention…} layers ("align" mode):
  [ conv_state | ssm_state | kv_padding | K (unpacked dtype) | V (unpacked dtype) ]
    ↑ offset 0, mamba slices                ↑ tail, attn slices after conv_block_padding
  (model_runner_v1.py:4963-4969 comment; carve at :4788-4797, :4864-4884)

TQ requires: [ … mamba … | padding | K_packed | V_packed | scale ]   ← does not exist
```
Additionally the hybrid attn carve computes `attn_tensor_page_size = prod(shape[1:]) *
get_dtype_size(spec.dtype)` (`:4864-4884`) — a formula that only knows one dtype and two planes.

### 4.4 Scale-cache co-location (question 3.3 of the brief)

The fp32 scale plane is **not** packed into page headers and is **not** an independent global
side tensor (the pre-spec design, still the CPU-harness fallback in `_ensure_scale_cache`
`turboquant_v1.py:465-483`, was exactly that and is now the error path). It is the **third plane
carved from the same page budget**, indexed by the same `(block, token)` as the packed planes,
visible to the block allocator through `AscendTurboQuantAttentionSpec`. Header packing is
deliberately declined: the writer scatters a token's scales as **one aligned 32-byte burst**
(`turboquant_scale_slot`), and header-interleaving would break both that alignment and the
decode's contiguous plane reads. `swap_blocks`/`copy_blocks` move all planes together — a copy
that leaves scales behind is documented as silent corruption, not an error.

---

## 5. Kernel Fusion Matrix

Ranked by (latency win × feasibility) ÷ complexity. "Status" reflects this branch.

| Rank | Fusion | Status | Latency vs complexity |
| --- | --- | --- | --- |
| F1 | **Writer one-launch**: Pi rotation + per-token-head quant + scale-burst scatter + packed store | **Shipped** (`npu_turboquant_reshape_and_cache`) | Baseline; replaces `npu_scatter_pa_kv_cache` 1:1, rotation free in UB |
| F2 | **Cube decode one-launch** (kv4fp8): raw-query rotation (butterfly→Mmad w/ `hadamard16`) + paged decode + online softmax + fp32 accum → un-rotate → `sigmoid(gate)` → fp16 cast | **Shipped** (`npu_turboquant_cube_decode`, `output_stage` 0/1/2) | Best decode latency (one launch/layer/step); requires fp16 + 950 + `block%64`; **this is what `patch_qwen3_5.py:106-113` wires the gate into** |
| F3 | **Prologue**: QKV split + q/k RMSNorm + RoPE(mrope) + K/V pack fused into the writer | **Candidate** | Saves the `.contiguous()` materialization (`turboquant_v1.py:813-815`) + 1–2 elementwise launches per layer per step; writer would take the fused-QKV layout + norm weights + cos/sin tables. Medium-high complexity (kernel learns RoPE tables); biggest win on prefill/large batches |
| F4 | **AIV decode**: fold `rotate_q` into the paged-decode prologue (mirror of the Cube raw-query entry) | **Candidate** | Saves 1 AIV launch/layer/step (the default bf16 27B path!); rotate is currently a separate op (`forward_paged_attention:1201`). Medium complexity; vector-unit rotation stays, launch overhead goes |
| F5 | **o_proj un-rotation fold** (`scripts/tq_fold_output_rotation.py`) | **Declined for Qwen3.5** | `attn_output_gate=true` on all 16 full-attn layers; elementwise gate ≠ commute with Pi (`turboquant_rotation.py:38-41`). Device un-rotation (AIV extra launch / Cube in-launch) is mandatory |
| F6 | Scale plane in page headers | **Declined by design** | Breaks burst-aligned scatter + contiguous decode reads; current third-plane co-location (§4.4) achieves allocator visibility without a header format |
| F7 | Uncompressed-sink side-car + host softmax merge (`VLLM_ASCEND_TQ_SINK_TOKENS`) | Shipped, default off | O(context) host cost per step; refused on kv4fp8 Cube (layout mismatch); diagnostic/accuracy tool, not a serving fusion |
| F8 | Modulo-scheduled decode (`turboquant_mm_fused_decode_opt_impl`, `_opt_raw_query_impl`) | **Built, unreachable** | Exported launchers with no adapter schema and no Python call site (`csrc/.../README.md:76-79`); future decode headroom behind a new dispatch |
| F9 | Paged prefill over the 4-bit cache | **Missing (strategic)** | Would lift the chunked-prefill / prefix-caching / spec-decode rejections (H3) that qwen3.5 deployments commonly want; largest single kernel-effort item |

Decode-side INT4 consumption requirements (for any new consumer such as `turboquant_paged_attention`
variants): rotated **fp32** query `[num_tokens, num_heads, head_size]`; int8 packed planes §4.1;
fp32 scale plane; int32 `block_tables[num_tokens, max_blocks]`; int32 `context_lens[num_tokens]`;
codec tables at `TURBOQUANT_TILE_ROWS=16` batch; fp32 workspace sized by
`npu_turboquant_workspace_size`; fp16/bf16 `out`; exactly one query token per sequence.

---

## 6. Concrete Action Plan

Naming note: the task's `turboquant_reshape_and_cache_a8` has no counterpart in-tree — the
existing entry is `npu_turboquant_reshape_and_cache`, and its K/V inputs are **fp16/bf16**, not
int8. The checklist below therefore covers (a) making the existing path actually run on
Qwen3.5-27B, and (b) adding the a8 (int8-activation-input) ingestion variant.

### Phase 0 — decide the hybrid layout (prerequisite for everything)

- **Option A (recommended): keep TQ full-attn layers out of mamba-shared tensors.** In
  `vllm_ascend/patch/platform/patch_kv_cache_utils.py` / `patch_mamba_config.py`, exclude layers
  whose spec is `AscendTurboQuantAttentionSpec` from the uniform-page-size/align merge so their
  `KVCacheTensor` is attn-only. Cost: those 16 layers lose prefix-sharing with GDN blocks (they
  don't have it meaningfully anyway — different allocators), and `hybrid_with_attn_and_mamba`
  must then be computed per-tensor (fix the global-flag ordering at
  `model_runner_v1.py:4232-4240` regardless). The existing `:4349-4379` carve and `:4572-4611`
  view then work unmodified.
- **Option B: extend the hybrid carve to five regions** (`conv | ssm | padding | K_pack | V_pack |
  scale`). Touches `_allocate_kv_cache_tensors`, `_reshape_kv_cache_tensors`
  (`model_runner_v1.py:4208-4490, 4545-5030`), `AscendTurboQuantAttentionSpec` page math under
  `page_size_padded`, `swap_blocks`/`copy_blocks`, and every hybrid KV-transfer consumer (H15).
  Only worth it if GDN+TQ block sharing is a hard requirement.

### Phase 1 — correctness guards (small, do first)

1. `vllm_ascend/platform.py::_validate_turboquant_config` — add an explicit statement for
   `model_config.is_hybrid` (support declaration under the chosen Phase-0 option, or a clean
   rejection naming H1). Today the engine passes validation and fails at cache reshape.
2. `vllm_ascend/worker/model_runner_v1.py:4232-4240` — make `hybrid_with_attn_and_mamba`
   per-tensor (pre-pass) so allocation is order-independent.
3. `get_kv_cache_spec` (`:5188-5345`): verify the `Attention` isinstance branch converts qwen3_5
   full-attn layers via `_as_turboquant_spec` — it does (`:5224-5226`) — and that the
   `use_compress` branch precedence (`:310-311`, `:5220`) can't shadow it (qwen3_5 has no
   `compress_ratios`; guard already exists at H3).

### Phase 2 — ingestion wiring for the a8 variant (int8 K/V in)

4. `csrc/attention/turboquant/op_adapter/turboquant_torch_ops.h` — new schema
   `npu_turboquant_reshape_and_cache_a8(Tensor key_i8, Tensor key_scale, Tensor value_i8,
   Tensor value_scale, Tensor! key_cache, Tensor! value_cache, Tensor! scale_cache,
   Tensor slot_mapping, Tensor codec_tables) -> ()` (exact activation-quant contract TBD: per-channel
   static vs per-token dynamic input scales).
5. `csrc/attention/turboquant/op_adapter/turboquant_torch_adpt.h` — validating launcher
   `npu_turboquant_reshape_and_cache_a8(...)` (mirror `:200-272` checks; dequantize-to-fp16 in UB
   before `Pi`, or pre-fold `Pi` into int8 via a sign-flip permutation since `D` commutes with
   per-channel scales and `H` needs fp math).
6. `csrc/attention/turboquant/op_kernel/turboquant_paged_attention.cpp` — launcher
   `turboquant_reshape_and_cache_a8_impl(...)` + kernel variant; register the writer in the
   dispatch comment table (`README.md`).
7. `csrc/torch_binding_meta.cpp` + `vllm_ascend/meta_registration.py` — no-op meta kernels for the
   new op (FakeTensor/export).
8. `CMakeLists.txt` — add the new TU to `VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCES` (`:122`); if 910B
   support is real, relax the `SOC_VERSION ~ ascend950` gate (`:118`) for the AIV-only sources
   (H2) — the Cube/ASC-toolchain parts stay 950-only.
9. `vllm_ascend/envs.py` — selection flag per AGENTS.md env review (e.g.
   `VLLM_ASCEND_TURBOQUANT_A8_INGEST`, documented in the `env_variables` dict).
10. `vllm_ascend/attention/turboquant_v1.py::reshape_and_cache` — pick the a8 writer when the
    layer hands over quantized activations (writer tuple at `:872-885`); keep burst-alignment
    validation via `_validate_cache_write`.
11. Producer side (optional, end-to-end a8): `vllm_ascend/quantization/methods/turboquant_v1.py`
    `create_weights` — attach per-channel k/v scales à la `AscendC8KVCacheAttentionMethod`
    (`methods/kv_cache/kv_c8.py:108-130`) so `k_proj`/`v_proj` emit int8; model patch stays in
    `patch/worker/patch_qwen3_5.py` (no model-file additions — AGENTS.md patching rule).

### Phase 3 — tests (none exist on this branch; AGENTS.md requires them)

12. Port `tests/ut/attention/test_turboquant_page_budget.py` (referenced by
    `kv_cache_interface.py:271` but absent here) plus layout/codec UTs from the dev branch;
    new UT for the hybrid allocation under Phase-0 Option A/B.
13. E2E smoke: qwen3_5 full-attn layer cosine vs bf16 oracle on NPU
    (`tests/e2e/pull_request/one_card/…` pattern); LongBench-style accuracy gate with
    `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS=1` + `TURBOQUANT_CAPTURE_SIGNATURES=1` tracing for the
    first bring-up runs.

### Phase 4 — later headroom

14. F4 (fold rotate into AIV decode), F9 (paged prefill over 4-bit cache — unlocks chunked
    prefill/prefix caching/spec-decode for qwen3_5 MTP), F8 dispatch for the opt decode,
    full-graph verification (drop the H4 downgrade).

### Verification anchors already in-tree

- Determinism: `turboquant_pi_signs` LCG (seed `0x5F3759DF`) must match the C++ host reference.
- Layout twins that must stay bit-identical: `turboquant_codec_tables` ↔
  `TurboQuantCodec<4>::ConstTableWords`; `turboquant_mode_tables` ↔ `ModeTables(KV3_FP4)`
  (a drifted copy returns plausible wrong numbers — cosine ≈ 0.33, never an error).
- `TURBOQUANT_GM_BURST_BYTES = 32` alignment on every operand; `TURBOQUANT_INDEX_DTYPE = int32`.
