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
"""GLM-4 (``THUDM/glm-4-9b-chat-1m``) on the standalone runner.

GLM-4 is dense GQA with no ``q_norm``/``k_norm`` and no output gate, so it runs
on :mod:`tq_longbench.layers` unchanged once three things are translated:

* **The config.** The ``chatglm`` checkpoint names everything its own way
  (``num_layers``, ``ffn_hidden_size``, ``kv_channels``, ``multi_query_group_num``,
  ``padded_vocab_size``, ``layernorm_epsilon``) and states RoPE as a ratio:
  the base is ``10000 * rope_ratio``. :func:`glm4_model_shape` reads it -- and
  the HF-native ``glm`` export's keys -- straight out of ``config.json``, so the
  checkpoint's ``trust_remote_code`` config class is never executed.
* **The tensor names.** ``chatglm`` fuses q, k and v into one
  ``query_key_value`` (with a bias) and gate and up into one ``dense_h_to_4h``;
  the ``glm`` export splits q/k/v but still fuses ``gate_up_proj``.
  :func:`glm4_checkpoint_targets` splits both at load time into the harness's
  separate projections, so :class:`~tq_longbench.layers.Attention` and
  :class:`~tq_longbench.layers.MLP` need no fused variant.
* **RoPE.** Both formats rotate only the first half of each head
  (``kv_channels // 2``, i.e. ``partial_rotary_factor = 0.5``) and pair
  *adjacent* channels, ChatGLM's ``reshape(..., rot_dim // 2, 2)``. The shape
  carries ``rotary_dim`` and ``rope_interleaved`` for
  :class:`~tq_longbench.layers.RotaryEmbedding`.

**The attention contract.** No gate means ``o_proj`` consumes the attention
output directly, so Pi folds into it offline, ``W_o' = W_o (I (x) Pi)``, and the
Cube decode runs its ``kRotatedBasis`` output stage (``TurboQuantOutputStage``
0): the kernel's output passes through with no in-kernel un-rotation. The head
counts are read from the checkpoint, not assumed -- ``glm-4-9b-chat-1m`` has
``multi_query_group_num = 4`` (32 query heads over 4 KV heads, ``D = 128``),
where ``glm-4-9b-chat`` has 2.

``glm4`` (GLM-4-0414) is a different layer -- it adds post-attention and
post-MLP norms this harness does not model -- and is refused by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tq_longbench.layers import ModelShape
from tq_longbench.ops import DENSE_BACKENDS

CHATGLM_MODEL_TYPE = "chatglm"
HF_GLM_MODEL_TYPE = "glm"
GLM4_MODEL_TYPES = frozenset({CHATGLM_MODEL_TYPE, HF_GLM_MODEL_TYPE})

#: GLM-4-0414's layer carries sandwich norms; loading it here would leave them out.
_UNSUPPORTED_GLM_MODEL_TYPES = frozenset({"glm4", "glm4_moe"})

#: ChatGLM's ``RotaryEmbedding`` base before ``rope_ratio`` scales it.
CHATGLM_ROPE_BASE = 10000.0

#: Both formats rotate half of each head: ChatGLM builds ``RotaryEmbedding(kv_channels // 2)``,
#: HF ``glm`` defaults ``partial_rotary_factor`` to 0.5.
GLM4_PARTIAL_ROTARY_FACTOR = 0.5

#: At and above this many context tokens the default prefill is ``batched_decode``:
#: ``dense_staging`` would hold a full fp16 KV pool beside the quantised one, and
#: at GLM-4's 1M-token reach that pool is what runs HBM out first.
GLM4_BATCHED_DECODE_MIN_TOKENS = 32768

_CHATGLM_LAYER_PREFIX = "transformer.encoder.layers."
_HF_LAYER_PREFIX = "model.layers."

#: Whole-model tensors, both formats, to the harness's module tree.
_GLOBAL_NAMES = {
    "transformer.embedding.word_embeddings.weight": "embed_tokens.weight",
    "transformer.encoder.final_layernorm.weight": "norm.weight",
    "transformer.output_layer.weight": "lm_head.weight",
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

#: Per-layer tensors that map one to one. Fused tensors are handled separately.
_LAYER_NAMES = {
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    # chatglm
    "self_attention.dense.weight": "self_attn.o_proj.weight",
    "mlp.dense_4h_to_h.weight": "mlp.down_proj.weight",
    # glm
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.q_proj.bias": "self_attn.q_proj.bias",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.k_proj.bias": "self_attn.k_proj.bias",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.v_proj.bias": "self_attn.v_proj.bias",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}

_FUSED_QKV = "self_attention.query_key_value."
_FUSED_GATE_UP = ("mlp.dense_h_to_4h.weight", "mlp.gate_up_proj.weight")


def read_config(model_path: str | Path) -> dict:
    """``config.json`` as a plain dict, or an empty one when the path has none."""
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_glm4_config(config: dict) -> bool:
    """Whether ``config`` is a GLM-4 checkpoint this adapter serves; raise on a GLM it cannot."""
    model_type = config.get("model_type")
    if model_type in _UNSUPPORTED_GLM_MODEL_TYPES:
        raise ValueError(
            f"model_type {model_type!r} (GLM-4-0414) adds post-attention and post-MLP norms this harness does not "
            f"model; only {sorted(GLM4_MODEL_TYPES)} checkpoints such as THUDM/glm-4-9b-chat-1m are supported"
        )
    return model_type in GLM4_MODEL_TYPES


def is_glm4_checkpoint(model_path: str | Path) -> bool:
    return is_glm4_config(read_config(model_path))


def glm4_model_shape(config: dict, attn_output_gate: bool = False) -> ModelShape:
    """The harness's shape for a ``chatglm`` or ``glm`` ``config.json``.

    Every layer variant the harness does not build is refused here rather than
    approximated: a LayerNorm read as an RMSNorm, or a dropped ``dense`` bias,
    loads without complaint and generates plausible text.
    """
    if attn_output_gate:
        raise ValueError("GLM-4 has no attention output gate; drop --attn-output-gate")
    if config.get("model_type") == CHATGLM_MODEL_TYPE:
        return _chatglm_shape(config)
    if config.get("model_type") == HF_GLM_MODEL_TYPE:
        return _hf_glm_shape(config)
    raise ValueError(f"not a GLM-4 config: model_type {config.get('model_type')!r}")


def _chatglm_shape(config: dict) -> ModelShape:
    unsupported = {
        "rmsnorm": (config.get("rmsnorm", True), True),
        "post_layer_norm": (config.get("post_layer_norm", True), True),
        "apply_residual_connection_post_layernorm": (
            config.get("apply_residual_connection_post_layernorm", False),
            False,
        ),
        "add_bias_linear": (config.get("add_bias_linear", False), False),
        "original_rope": (config.get("original_rope", True), True),
    }
    for key, (value, required) in unsupported.items():
        if bool(value) != required:
            raise ValueError(f"chatglm {key}={value} is not the GLM-4 layer this harness builds (needs {required})")

    num_heads = config["num_attention_heads"]
    head_size = config.get("kv_channels") or config["hidden_size"] // num_heads
    num_kv_heads = config["multi_query_group_num"] if config.get("multi_query_attention", False) else num_heads
    return ModelShape(
        num_layers=config["num_layers"],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        hidden_size=config["hidden_size"],
        intermediate_size=config["ffn_hidden_size"],
        vocab_size=config["padded_vocab_size"],
        rms_norm_eps=float(config["layernorm_epsilon"]),
        rope_theta=CHATGLM_ROPE_BASE * float(config.get("rope_ratio", 1)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        attn_output_gate=False,
        qk_norm=False,
        qkv_bias=bool(config.get("add_qkv_bias", False)),
        # ChatGLM's RotaryEmbedding(kv_channels // 2): half the head, from the front.
        rotary_dim=int(head_size * GLM4_PARTIAL_ROTARY_FACTOR),
        rope_interleaved=True,
    )


def _hf_glm_shape(config: dict) -> ModelShape:
    num_heads = config["num_attention_heads"]
    head_size = config.get("head_dim") or config["hidden_size"] // num_heads
    return ModelShape(
        num_layers=config["num_hidden_layers"],
        num_heads=num_heads,
        num_kv_heads=config.get("num_key_value_heads", num_heads),
        head_size=head_size,
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)),
        rope_theta=float(config.get("rope_theta", CHATGLM_ROPE_BASE)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        attn_output_gate=False,
        qk_norm=False,
        qkv_bias=bool(config.get("attention_bias", True)),
        rotary_dim=int(head_size * float(config.get("partial_rotary_factor", GLM4_PARTIAL_ROTARY_FACTOR))),
        rope_interleaved=True,
    )


def glm4_checkpoint_targets(name: str, tensor: torch.Tensor, shape: ModelShape) -> list[tuple[str, torch.Tensor]]:
    """Where one checkpoint tensor goes in the harness, split if the checkpoint fused it.

    Returns an empty list for what is recomputed (RoPE's ``inv_freq``). Anything
    else unrecognised **raises**: a GLM tensor with no home here is a layer
    variant the harness does not build, and skipping it would run the model
    without it.
    """
    if name.endswith("rotary_pos_emb.inv_freq") or name.endswith("rotary_emb.inv_freq"):
        return []
    if name in _GLOBAL_NAMES:
        return [(_GLOBAL_NAMES[name], tensor)]

    for prefix in (_CHATGLM_LAYER_PREFIX, _HF_LAYER_PREFIX):
        if name.startswith(prefix):
            index, _, rest = name[len(prefix) :].partition(".")
            break
    else:
        raise KeyError(f"{name} is not a GLM-4 tensor this harness knows")
    layer = f"layers.{index}."

    if rest in _LAYER_NAMES:
        return [(layer + _LAYER_NAMES[rest], tensor)]
    if rest.startswith(_FUSED_QKV):
        kind = rest[len(_FUSED_QKV) :]
        q_rows = shape.num_heads * shape.head_size
        kv_rows = shape.num_kv_heads * shape.head_size
        if tensor.shape[0] != q_rows + 2 * kv_rows:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows but {shape.num_heads} query and {shape.num_kv_heads} kv heads of "
                f"{shape.head_size} need {q_rows + 2 * kv_rows}; the config's multi_query_group_num disagrees with "
                "the checkpoint"
            )
        query, key, value = tensor.split((q_rows, kv_rows, kv_rows), dim=0)
        return [
            (f"{layer}self_attn.q_proj.{kind}", query),
            (f"{layer}self_attn.k_proj.{kind}", key),
            (f"{layer}self_attn.v_proj.{kind}", value),
        ]
    if rest in _FUSED_GATE_UP:
        if tensor.shape[0] != 2 * shape.intermediate_size:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, not 2 x intermediate_size {shape.intermediate_size}")
        # ChatGLM's swiglu is silu(chunk[0]) * chunk[1]: gate first, then up.
        gate, up = tensor.chunk(2, dim=0)
        return [(f"{layer}mlp.gate_proj.weight", gate), (f"{layer}mlp.up_proj.weight", up)]
    raise KeyError(f"{name} is a GLM-4 layer tensor this harness does not model")


def glm4_prefill_mode(context_tokens: int, backend: str) -> str:
    """``batched_decode`` from :data:`GLM4_BATCHED_DECODE_MIN_TOKENS` context tokens up.

    A dense baseline decodes out of the unquantised pool, so it needs
    ``dense_staging`` at any length.
    """
    if backend in DENSE_BACKENDS:
        return "dense_staging"
    return "batched_decode" if context_tokens >= GLM4_BATCHED_DECODE_MIN_TOKENS else "dense_staging"
