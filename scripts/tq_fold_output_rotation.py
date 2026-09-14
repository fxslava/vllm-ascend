#!/usr/bin/env python3
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
"""Fold TurboQuant's output de-rotation into a checkpoint's ``o_proj`` weights.

The TurboQuant decode writes its attention output in the rotated basis,
``O~ = softmax(q k^T) V~``, and no kernel takes it back out. This tool rewrites

    W_o  ->  W_o (I_H (x) Pi)          for every  <prefix>.self_attn.o_proj.weight

so the projection does, and records the fold in ``config.json`` under
``turboquant_output_rotation``. The backend reads that record to decide, per
layer, whether to leave the output rotated (folded) or un-rotate it on the
device (not folded). Every other tensor and file is copied unchanged.

The fold is refused, not attempted, when it would be wrong:

* an output gate -- ``attn_output_gate`` truthy, or a model type whose
  attention gates by default (Qwen3-Next, Qwen3.5). ``o_proj(O * g)`` is not
  ``o_proj'(O~ * g)`` for an elementwise ``g``; run those models unfolded.
* a model type this tool cannot vouch for, unless ``--assume-no-output-gate``
  says the attention output reaches ``o_proj`` with nothing in between.
* a quantised ``o_proj``: a non-float weight, or companion tensors such as
  ``weight_scale``. Fold before quantising.

Each folded weight is validated before anything is written: the worst cosine
between ``W_o (Pi o~)`` and ``W_o' o~`` over random rotated outputs, with
``W_o'`` in its stored dtype, must exceed ``--min-cosine`` (0.9999).

Runs on torch and safetensors alone -- no torch_npu, no vLLM:

    python scripts/tq_fold_output_rotation.py --src /models/Qwen3-8B --dst /models/Qwen3-8B-tq-folded
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ROTATION_MODULE_PATH = REPO_ROOT / "vllm_ascend" / "attention" / "turboquant_rotation.py"

CONFIG_NAME = "config.json"
SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
SAFETENSORS_SUFFIX = ".safetensors"

# <module prefix>.o_proj.weight, where the prefix ends in self_attn. The prefix
# is what the marker records and what the backend matches vLLM's layer name to.
O_PROJ_WEIGHT_PATTERN = re.compile(r"^(?P<prefix>.*\bself_attn)\.o_proj\.weight$")
O_PROJ_ALLOWED_SUFFIXES = ("weight", "bias")

# Model types whose attention output reaches o_proj directly. Not a list of
# what TurboQuant supports -- a list of what this tool has checked has no
# operation between the attention and the projection.
UNGATED_MODEL_TYPES = frozenset({"llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe"})
# Model types whose attention applies an elementwise output gate when the
# config does not say otherwise (vLLM reads attn_output_gate with default True).
GATED_BY_DEFAULT_MODEL_TYPES = frozenset({"qwen3_next", "qwen3_5", "qwen3_5_moe"})


class FoldRefused(RuntimeError):
    """The checkpoint cannot be folded correctly; nothing was written."""


def load_rotation_module() -> ModuleType:
    """Import turboquant_rotation.py by path.

    ``import vllm_ascend.attention...`` would run the package __init__, which
    imports vLLM; the rotation module itself needs only torch.
    """
    spec = importlib.util.spec_from_file_location("turboquant_rotation", ROTATION_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise FoldRefused(f"cannot load {ROTATION_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text_config(config: dict) -> dict:
    """The language model's config: nested under text_config for VL models."""
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def attention_geometry(config: dict) -> tuple[int, int]:
    """(num_attention_heads, head_size) of the language model."""
    text = text_config(config)
    num_heads = text.get("num_attention_heads")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise FoldRefused("config has no positive num_attention_heads")
    head_size = text.get("head_dim")
    if head_size is None:
        hidden = text.get("hidden_size")
        if not isinstance(hidden, int) or hidden % num_heads:
            raise FoldRefused("config has neither head_dim nor a hidden_size divisible by num_attention_heads")
        head_size = hidden // num_heads
    return num_heads, int(head_size)


def check_no_output_gate(config: dict, assume_no_output_gate: bool) -> None:
    text = text_config(config)
    model_types = {str(config.get("model_type", "")), str(text.get("model_type", ""))} - {""}
    gate = text.get("attn_output_gate", config.get("attn_output_gate"))
    if gate or (gate is None and model_types & GATED_BY_DEFAULT_MODEL_TYPES):
        raise FoldRefused(
            f"model type {sorted(model_types)} gates its attention output elementwise before o_proj "
            f"(attn_output_gate={gate!r}); Pi does not commute with that gate, so the fold would change the model. "
            "Serve this checkpoint unfolded -- the backend un-rotates the decode output on the device."
        )
    if not model_types & UNGATED_MODEL_TYPES and not assume_no_output_gate:
        raise FoldRefused(
            f"model type {sorted(model_types)} is not one this tool has checked for an operation between the "
            "attention output and o_proj. If there is none, pass --assume-no-output-gate."
        )


def checkpoint_shards(src: Path) -> list[Path]:
    index = src / SAFETENSORS_INDEX_NAME
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted({src / name for name in weight_map.values()})
    shards = sorted(src.glob(f"*{SAFETENSORS_SUFFIX}"))
    if not shards:
        raise FoldRefused(f"no {SAFETENSORS_SUFFIX} files in {src}")
    return shards


def check_unquantised(names: set[str], prefixes: set[str]) -> None:
    for name in names:
        head, _, suffix = name.rpartition(".")
        if head.endswith(".o_proj") and head[: -len(".o_proj")] in prefixes and suffix not in O_PROJ_ALLOWED_SUFFIXES:
            raise FoldRefused(
                f"{name} accompanies an output projection: it looks quantised, and a rotated weight would need "
                "requantising. Fold the float checkpoint, then quantise."
            )


def fold_checkpoint(args: argparse.Namespace, rotation: ModuleType) -> int:
    src, dst = Path(args.src), Path(args.dst)
    config = json.loads((src / CONFIG_NAME).read_text(encoding="utf-8"))
    if rotation.TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY in config:
        raise FoldRefused(f"{src} is already folded ({rotation.TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY} is set)")
    check_no_output_gate(config, args.assume_no_output_gate)
    num_heads, head_size = attention_geometry(config)

    shards = checkpoint_shards(src)
    names: set[str] = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            names.update(handle.keys())
    prefixes = {m.group("prefix") for m in map(O_PROJ_WEIGHT_PATTERN.match, names) if m}
    if not prefixes:
        raise FoldRefused("found no <prefix>.self_attn.o_proj.weight tensors to fold")
    check_unquantised(names, prefixes)

    generator = torch.Generator().manual_seed(args.seed)
    folded_shards: dict[Path, tuple[dict[str, torch.Tensor], dict[str, str] | None]] = {}
    worst_cosine = 1.0
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        touched = False
        with safe_open(str(shard), framework="pt") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                match = O_PROJ_WEIGHT_PATTERN.match(name)
                if match:
                    if tensor.dim() != 2 or tensor.shape[1] != num_heads * head_size:
                        raise FoldRefused(
                            f"{name} is {tuple(tensor.shape)}, not [hidden, {num_heads} * {head_size}]; a sharded or "
                            "reshaped projection cannot be folded head by head"
                        )
                    folded = rotation.fold_pi_into_output_projection(tensor, head_size)
                    report = rotation.validate_output_projection_fold(
                        tensor, folded, head_size, num_samples=args.num_samples, generator=generator
                    )
                    worst_cosine = min(worst_cosine, report.min_cosine)
                    print(
                        f"  {name}: {str(tensor.dtype).removeprefix('torch.')} {tuple(tensor.shape)} "
                        f"cos_min={report.min_cosine:.9f} out_relerr={report.max_relative_error:.2e} "
                        f"weight_relerr={report.weight_relative_error:.2e}"
                    )
                    if not report.passed(args.min_cosine):
                        raise FoldRefused(f"{name}: cos_min {report.min_cosine:.9f} <= {args.min_cosine}")
                    tensor = folded
                    touched = True
                tensors[name] = tensor
        if touched:
            folded_shards[shard] = (tensors, metadata)

    print(f"folded {len(prefixes)} output projections, head_size {head_size}, worst cos {worst_cosine:.9f}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    dst.mkdir(parents=True, exist_ok=False)
    for entry in src.iterdir():
        if entry.is_file() and entry.name != CONFIG_NAME and entry not in folded_shards:
            shutil.copy2(entry, dst / entry.name)
        elif entry.is_dir():
            shutil.copytree(entry, dst / entry.name)
    for shard, (tensors, metadata) in folded_shards.items():
        save_file(tensors, str(dst / shard.name), metadata=metadata)
    config[rotation.TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY] = rotation.output_rotation_marker(
        sorted(prefixes), head_size
    )
    (dst / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    return 0


def parse_args(argv: list[str], rotation: ModuleType) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Hugging Face checkpoint directory (safetensors)")
    parser.add_argument("--dst", required=True, help="output directory; must not exist")
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=rotation.TURBOQUANT_FOLD_MIN_COSINE,
        help="per-projection acceptance bound",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=rotation.TURBOQUANT_FOLD_VALIDATION_SAMPLES,
        help="rotated outputs drawn per projection",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the validation draws")
    parser.add_argument(
        "--assume-no-output-gate",
        action="store_true",
        help="fold a model type this tool has not checked; you vouch that o_proj consumes the attention output",
    )
    parser.add_argument("--dry-run", action="store_true", help="fold and validate in memory, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    rotation = load_rotation_module()
    try:
        return fold_checkpoint(parse_args(argv, rotation), rotation)
    except (FoldRefused, ValueError) as refusal:
        # ValueError is the fold helper refusing a weight it cannot rotate
        # exactly (a non-float dtype, a width that is not whole heads).
        print(f"refused: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
