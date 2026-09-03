# Bare-metal kernel tests (Ascend 310P3)

A standalone GTest suite for the four operator families a **Qwen3.5** forward
pass needs, driven straight through the CANN runtime. No Python, no PyTorch, no
`torch_npu`, no Torch C++ ABI.

| Kernel | Test binary | Stands in for |
| --- | --- | --- |
| RMSNorm | `test_rmsnorm_310p` | `torch_npu.npu_rms_norm` — `AscendRMSNorm310.forward_oot` |
| Rotary embedding | `test_rotary_embedding_310p` | `torch_npu.npu_apply_rotary_pos_emb` — `_rope_forward_oot` |
| SiluAndMul (SwiGLU) | `test_activation_swiglu_310p` | `torch_npu.npu_swiglu` — `AscendSiluAndMul310.forward` |
| Paged attention + KV cache | `test_paged_attention_310p` | `torch_npu._npu_paged_attention`, `torch_npu._npu_reshape_and_cache` |

The Python unit tests for these paths (`tests/ut/ops/test_layernorm.py`,
`test_activation.py`, `test_rotary_embedding.py`) mock `torch_npu` out entirely
and assert only on dispatch. This suite is where the numerics are actually
checked, against fp32 CPU references, at the shapes Qwen3.5 uses.

---

## Layout

```
csrc/tests/
├── CMakeLists.txt                     standalone project, not included from the repo root
├── common/
│   ├── acl_check.hpp                  ACL_CHECK / ASSERT_ACL_OK, with aclGetRecentErrMsg attached
│   ├── aclnn_ops.hpp / .cpp           the version-sensitive aclnn prototypes — read this first
│   ├── aclnn_runtime.hpp / .cpp       dlopen/dlsym loader, aclTensor RAII, two-phase launch
│   ├── cpu_reference.hpp / .cpp       naive fp32 references for all four kernels
│   ├── device_buffer.hpp              32-byte aligned RAII device allocation
│   ├── device_tensor.hpp              device buffer + aclTensor descriptor, with host conversions
│   ├── fp16.hpp                       IEEE-754 binary16 conversion, round-to-nearest-even
│   ├── main.cpp                       entry point, prints the operator inventory
│   ├── qwen_shapes.hpp                Qwen3.5 shapes and the 310P alignment rules
│   ├── random_data.hpp                deterministic, platform-independent test data
│   ├── tensor_compare.hpp             allclose with a diagnostic report
│   └── test_harness.hpp / .cpp        AscendTestEnvironment: aclInit, device, context, stream
└── kernels/
    ├── test_rmsnorm_310p.cpp
    ├── test_rotary_embedding_310p.cpp
    ├── test_activation_swiglu_310p.cpp
    └── test_paged_attention_310p.cpp
```

Each test file has two layers. Tests named `*Reference` and `*Shapes` are
host-only: they check the CPU reference and the layout arithmetic and run
anywhere, including on a build machine with no NPU. The rest need a device and
skip with an explanatory message when one is not attached.

---

## Prerequisites

- CANN toolkit (headers at `$ASCEND_HOME_PATH/include/acl/acl.h`)
- CMake >= 3.16 (3.18+ enables per-test `ctest` granularity)
- A C++17 compiler
- An Ascend 310P3 device for the parity tests; the host-only tests need none

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Configure and build

```bash
cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build/csrc-tests -j "$(nproc)"
```

On a host without internet access, either point FetchContent at an unpacked
googletest tree:

```bash
cmake -S csrc/tests -B build/csrc-tests -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/opt/googletest
```

or use a system install:

```bash
cmake -S csrc/tests -B build/csrc-tests -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF
```

If CANN is not on the default path, pass it explicitly:

```bash
cmake -S csrc/tests -B build/csrc-tests -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Run

```bash
ctest --test-dir build/csrc-tests --output-on-failure
```

One suite at a time:

```bash
ctest --test-dir build/csrc-tests -R rmsnorm --output-on-failure
```

Or run a binary directly, which is the better option while debugging because it
prints the operator inventory first:

```bash
./build/csrc-tests/test_paged_attention_310p --gtest_filter='*seq4_h32_kv8*'
```

Select a device other than 0:

```bash
ASCEND_TEST_DEVICE_ID=3 ./build/csrc-tests/test_rmsnorm_310p
```

Useful GTest flags: `--gtest_list_tests`, `--gtest_repeat=10`,
`--gtest_shuffle`, `--gtest_output=xml:results.xml`.

---

## First run on a new CANN release

The suite resolves every aclnn operator with `dlopen`/`dlsym` rather than
linking `libopapi.so` directly, the same way
`csrc/aclnn_torch_adapter/op_api_common.h` does. A renamed or absent operator
becomes a skip naming the symbol instead of a link error that takes out the
whole binary.

The trade-off is that the argument lists in `common/aclnn_ops.hpp` are declared
by hand and are **not** checked by the compiler. Before trusting the first run
on a CANN version this suite has not seen, confirm each prototype:

```bash
grep -rA24 'GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/aclnn_rms_norm.h $ASCEND_HOME_PATH/include/aclnnop/aclnn_swi_glu.h
```

```bash
grep -rlA24 'ApplyRotaryPosEmb\|ReshapeAndCache\|PagedAttention' $ASCEND_HOME_PATH/include/aclnnop/
```

A mismatch shows up as a non-zero status from the planning call with the CANN
diagnostic attached, not as silent corruption. `aclnnPagedAttention` carries the
most optional arguments of the five and is the most likely to differ.

The inventory printed at start-up tells you what resolved:

```
[ascend-test] aclnn operator inventory
[ascend-test]   aclnnRmsNorm                 found
[ascend-test]   aclnnSwiGlu                  found
[ascend-test]   aclnnApplyRotaryPosEmbV2     found
...
```

---

## Profiling with msprof

Collect a full trace of one suite:

```bash
msprof --application="./build/csrc-tests/test_rmsnorm_310p --gtest_filter=*tokens128_hidden8192*" --output=./prof/rmsnorm --aic-metrics=PipeUtilization --ai-core=on
```

Narrow the filter to a single parameterised case. Without it the trace covers
every shape in the suite and the per-kernel timings are hard to attribute.

Task-level timeline only, which is much cheaper:

```bash
msprof --application="./build/csrc-tests/test_paged_attention_310p --gtest_filter=*seq4_h32_kv8_d128_b64_ctx512*" --output=./prof/pa --task-time=on --aicpu=on
```

Export the collected data to CSV:

```bash
msprof --export=on --output=./prof/rmsnorm
```

Then read `./prof/rmsnorm/**/mindstudio_profiler_output/op_summary_*.csv` for
per-operator duration, and `op_statistic_*.csv` for the aggregate.

Two things to keep in mind when reading the numbers:

- Every `RunAclnn` call synchronises the stream, so the timings include launch
  overhead and are not a throughput measurement. Use `--gtest_repeat` to get a
  stable per-call figure.
- The workspace is allocated and freed per call. In the plugin it comes from the
  caching allocator, so a profile taken here overstates allocation cost.

For a memory-access view instead of a pipe-utilisation view:

```bash
msprof --application="./build/csrc-tests/test_activation_swiglu_310p" --output=./prof/swiglu --aic-metrics=MemoryUB
```

---

## 310P specifics encoded in the tests

These come from the plugin source rather than from general Ascend documentation,
and several differ from the GPU defaults:

- **KV cache is 5-D NZ**, not the generic `[2, num_blocks, block_size,
  num_kv_heads, head_size]`. `AscendAttentionBackend310.get_kv_cache_shape`
  returns `(2, num_blocks, num_kv_heads * head_size / 16, block_size, 16)` and
  the runner allocates each half with `acl_format=ACL_FORMAT_FRACTAL_NZ`. This
  requires `num_kv_heads * head_size % 16 == 0`.
- **Block size is 64 or 128**, not 16 or 32.
  `get_supported_kernel_block_sizes()` returns `[128, 64]`, and the runner
  additionally requires `block_size * head_size <= 128 * 128`
  (`_ATTENTION_BLOCK_SIZE_LIMIT`).
- **SwiGLU needs `x.shape[-1] % 32 == 0`**. `AscendSiluAndMul310.forward` falls
  back to eager torch otherwise, so a width that fails the gate never reaches
  the kernel.
- **Rotary embedding supports head dims 64 and 128 only**, per the
  `self.rotary_dim in (64, 128)` gate in the 310P rotary module.
- **Allocations are padded to 32 bytes.** MTE2/MTE3 move 32-byte bursts, so an
  unpadded tail lets the last burst run past the end of the buffer.

## Scope

Covered: numerical parity against fp32 CPU references, layout and alignment
contracts, GQA head mapping, both rotary layouts, block-table paging across
multiple blocks, and context-length bounds.

Not covered: the quantised (W8A8 / int8 KV cache) paths, chunked prefill and the
splitfuse attention variants, multi-device or graph-capture execution, and
performance regression thresholds. The four kernels here are the fp16 decode
path only, which is what the brief scoped.

## Adding a kernel

1. Declare the operator in `common/aclnn_ops.hpp` with a `mirrors:` comment
   naming the `torch_npu` entry point and the vllm-ascend call site, and add it
   to `ProbeAllOperators`.
2. Add a reference to `common/cpu_reference.{hpp,cpp}`.
3. Add `kernels/test_<name>_310p.cpp` with host-only reference tests plus the
   device parity tests.
4. Append the binary name to `VLLM_ASCEND_KERNEL_TESTS` in `CMakeLists.txt`.
