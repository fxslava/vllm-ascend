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
"""Write a run's quantised KV cache out for someone who has only torch.

The consumer of this is another process, on another machine, with no
``vllm_ascend``, no ``torch_npu`` and no kernels -- so the dump has to carry not
just the cache but everything needed to *read* it. Three things are easy to
forget and none of them fails loudly:

* **The cache does not hold K.** It holds ``Pi k``, quantised: a Hadamard-and-signs
  rotation is applied before the codec so the 4-bit levels see a flatter
  distribution. Reconstructing without undoing it gives tensors of the right
  shape and dtype whose cosine against the true K is **-0.005**. So Pi travels
  with the dump, as a plain ``[D, D]`` matrix rather than as a function -- it is
  an involution, so one matmul undoes it.
* **The codes are two per byte, low nibble first.** Reading them high-first is
  also silent, and also meaningless: cosine **0.000**.
* **The scales are not derivable.** Packed codes without them decode to
  plausible wrong magnitudes, and they are stored in a padded, token-major slot
  that the shapes alone do not explain.

So :func:`dump_quantised_cache` writes three files: the tensors, a
``metadata.json`` describing the run that produced them, and a ``README.md``
whose reconstruction snippet is the one this module's own tests execute against
the real cache. That last part is the point -- a guide nobody runs is a guide
that drifts.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tq_longbench._ascend import turboquant_layout
from tq_longbench.kv_cache import TurboQuantKVCache
from tq_longbench.layers import pi_matrix

#: The file the tensors go in. One file rather than a directory of them, because
#: the planes only mean anything together.
TENSOR_FILE = "kv_cache_tensors.pt"
METADATA_FILE = "metadata.json"
README_FILE = "README.md"

#: ``torch.save`` protocol that ``weights_only=True`` can read back. Everything
#: in the dictionary is a tensor or a Python scalar, so nothing needs a class to
#: be unpickled -- which is what lets the consumer load it without this package.
_TENSOR_KEYS_DOC = "see README.md"


def normalise_dump_dir(text: str) -> Path:
    """``--dump-kv-dir`` as a path this host can open, whichever way it was typed.

    The same normalisation ``--model-path`` gets, for the same reasons:
    ``F:/AI/kvcache`` and ``F:\\AI\\kvcache`` are one directory, a trailing
    separator is noise, and ``~`` or an unexpanded ``%VAR%`` is the shell's
    business rather than this program's. Resolved to an absolute path, so the
    directory a run reports is the directory it wrote to even if something later
    changes the working directory.
    """
    return Path(os.path.expandvars(text)).expanduser().resolve()


def run_directory(root: Path, run_name: str) -> Path:
    """``root/run_name``, created if it is not there, parents and all."""
    directory = root / run_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def default_run_name(model_path: str, backend: str, context_tokens: int) -> str:
    """A directory name that says what the run was, and does not collide with the last one.

    The checkpoint's own directory name, the backend, the context and a
    timestamp: enough to tell two dumps apart on a listing without opening
    either one's metadata.
    """
    model = re.sub(r"[^0-9A-Za-z._-]+", "-", Path(model_path).name).strip("-").lower()
    return f"{model or 'model'}_{backend}_{context_tokens}_{time.strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class DumpReport:
    """Where a dump went and how big it was, for the line the run prints."""

    directory: Path
    tokens: int
    blocks: int
    layers: int
    bytes_written: int

    def describe(self) -> str:
        megabytes = self.bytes_written / (1024 * 1024)
        return (
            f"kv dump: {self.layers} layers x {self.blocks} blocks ({self.tokens} tokens), "
            f"{megabytes:.1f} MB -> {self.directory}"
        )


def blocks_for(tokens: int, block_size: int) -> int:
    """How many whole blocks hold ``tokens``; at least one, so a dump is never empty."""
    return max(1, -(-tokens // block_size))


def collect_tensors(runner, cached_tokens: int) -> dict[str, torch.Tensor]:
    """Every tensor the consumer needs, on the CPU, detached from the run.

    Trimmed to the blocks the context actually reached: the pool is sized for
    ``max_seq_len`` and a run that filled a tenth of it should not write nine
    tenths of zeros. The paging survives the trim -- a slot is still a position,
    and ``block_table`` still indexes it -- because whole blocks are kept.

    The planes are stacked over layers rather than written as one key per layer:
    thirty-six keys that must be reassembled in order is an invitation to
    reassemble them out of order, and the stack makes ``[layer]`` the first axis
    everywhere.
    """
    cache = runner.decode_backend.cache
    if not isinstance(cache, TurboQuantKVCache):
        raise TypeError(
            f"the {runner.decode_backend.name} backend has no quantised cache to dump; "
            "use --backend turboquant_cube, turboquant_aiv or turboquant_reference"
        )
    geometry = runner.geometry
    blocks = blocks_for(cached_tokens, geometry.block_size)

    def stacked(planes: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([plane[:blocks].clone().detach().cpu() for plane in planes])

    layout = turboquant_layout()
    head_size = geometry.head_size
    return {
        # [layers, blocks, block_size, kv_heads, head_size // 2] int8
        "key_planes": stacked(cache.key_planes),
        "value_planes": stacked(cache.value_planes),
        # [layers, blocks, block_size, scale_slot] float32
        "scale_planes": stacked(cache.scale_planes),
        # The codec, so the codes mean something without this package.
        "centroids": torch.tensor(layout.TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32),
        "thresholds": torch.tensor(layout.TURBOQUANT_LLOYD_MAX_THRESHOLDS, dtype=torch.float32),
        # Pi as a matrix rather than as the transform it is: the consumer has no
        # apply_pi to call, and a [D, D] involution needs no library.
        "pi": pi_matrix(head_size, torch.device("cpu"), dtype=torch.float32),
        # The indexing. Identity paging, written down rather than assumed.
        "block_table": cache.block_table(cached_tokens).clone().detach().cpu(),
        "slot_mapping": cache.slot_mapping(0, cached_tokens).clone().detach().cpu(),
        "context_lens": torch.tensor([cached_tokens], dtype=torch.int32),
    }


def build_metadata(
    runner,
    *,
    model_path: str,
    family: str,
    cached_tokens: int,
    context_tokens: int,
    prompt: str,
    needle: dict,
) -> dict:
    """What the tensors cannot say: which model, which run, and whether it worked.

    ``retrieved`` in particular. A cache from a run that lost the needle is still
    worth analysing -- more so, arguably -- but analysing it as though the model
    had found it would be reading the wrong thing, and the tensors carry no hint
    either way.
    """
    shape = runner.shape
    geometry = runner.geometry
    blocks = blocks_for(cached_tokens, geometry.block_size)
    return {
        "model_name": Path(model_path).name,
        "model_family": family,
        "model_path": str(model_path),
        "num_layers": shape.num_layers,
        "hidden_size": shape.hidden_size,
        "num_attention_heads": shape.num_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_dim": shape.head_size,
        "rope_theta": shape.rope_theta,
        "backend": runner.config.backend,
        "prefill_mode": runner.config.prefill_mode,
        "dtype": str(runner.config.dtype).replace("torch.", ""),
        "context_tokens": context_tokens,
        "cached_tokens": cached_tokens,
        "allocated_blocks": geometry.num_blocks,
        "dumped_blocks": blocks,
        "block_size": geometry.block_size,
        "scale_slot": geometry.scale_slot,
        "pack_factor": turboquant_layout().TURBOQUANT_PACK_FACTOR,
        "max_seq_len": geometry.max_seq_len,
        "niah": needle,
        "prompt": prompt,
        "tensor_file": TENSOR_FILE,
        "tensor_keys": _TENSOR_KEYS_DOC,
    }


def render_readme(metadata: dict, tensors: dict[str, torch.Tensor]) -> str:
    """The guide, with this dump's own shapes in it.

    Generated rather than shipped as a static file so the shapes are the ones in
    the box beside it, and so a model with a different head count does not get a
    README describing someone else's.
    """
    shapes = "\n".join(
        f"| `{name}` | `{tuple(tensor.shape)}` | `{str(tensor.dtype).replace('torch.', '')}` | {_PURPOSE[name]} |"
        for name, tensor in tensors.items()
    )
    heads = metadata["num_kv_heads"]
    return _README_TEMPLATE.format(
        model_name=metadata["model_name"],
        family=metadata["model_family"],
        backend=metadata["backend"],
        layers=metadata["num_layers"],
        kv_heads=heads,
        head_dim=metadata["head_dim"],
        block_size=metadata["block_size"],
        cached_tokens=metadata["cached_tokens"],
        scale_slot=metadata["scale_slot"],
        shapes=shapes,
        tensor_file=TENSOR_FILE,
        metadata_file=METADATA_FILE,
        kv_heads_x2=2 * heads,
    )


#: One line per tensor for the README's table. Keyed by the same names
#: :func:`collect_tensors` uses, so a tensor added there without a purpose here
#: raises rather than appearing in the table unexplained.
_PURPOSE = {
    "key_planes": "packed 4-bit **Pi k**, two codes per byte, low nibble first",
    "value_planes": "packed 4-bit **Pi v**, same packing",
    "scale_planes": "per-vector RMS scales, token-major; see the slot layout below",
    "centroids": "the 16 Lloyd-Max reconstruction levels a code indexes",
    "thresholds": "the 15 bin edges, for re-quantising experiments (not needed to read)",
    "pi": "the rotation as a matrix; an involution, so one matmul undoes it",
    "block_table": "block ids covering the cached context (paging is the identity here)",
    "slot_mapping": "the slot each cached position was written to (also the identity)",
    "context_lens": "how many positions are live; everything past this is zeros",
}


_README_TEMPLATE = """# Quantised KV cache: {model_name}

A TurboQuant 4-bit KV cache taken from a `{backend}` run of {model_name}
({family}), {layers} layers, {kv_heads} kv heads of {head_dim} channels,
{cached_tokens} cached positions in blocks of {block_size}.

Everything needed to read it is in `{tensor_file}`. Nothing here needs
`vllm_ascend`, `torch_npu` or any compiled kernel -- plain torch on a CPU is
enough. `{metadata_file}` carries the run's parameters and what the model did
with the prompt.

```python
import torch

blob = torch.load("{tensor_file}", map_location="cpu", weights_only=True)
```

## What is in the file

| key | shape | dtype | what it is |
| --- | --- | --- | --- |
{shapes}

## The three things that fail silently

**The cache does not hold K.** It holds `Pi @ k`: a Hadamard-and-signs rotation
applied before the codec, so the 4-bit levels see a flatter distribution.
Reconstructing without undoing it gives a tensor of the right shape and dtype
whose cosine against the true K is **-0.005**. `pi` is that rotation as a
`[{head_dim}, {head_dim}]` matrix and it is its own inverse, so `rotated @ pi`
undoes it.

**Two codes share a byte, low nibble first.** Reading them high-first is also
silent, and also meaningless: cosine **0.000**.

**The scales are not derivable from the codes.** `scale_planes[..., :{kv_heads}]`
are the K scales, one per (token, kv head); `[..., {kv_heads}:{kv_heads_x2}]` are
the V scales. The slot is {scale_slot} wide rather than {kv_heads_x2} because the
kernel pads it to a whole aligned burst -- the rest is zeros and means nothing.

## Reconstructing K and V

```python
import torch

blob = torch.load("{tensor_file}", map_location="cpu", weights_only=True)
centroids, pi = blob["centroids"], blob["pi"]
tokens = int(blob["context_lens"][0])
kv_heads, head_dim = {kv_heads}, {head_dim}


def rebuild(planes, scales, layer):
    "K or V for one layer, as [tokens, kv_heads, head_dim] float32."
    packed = planes[layer].reshape(-1, kv_heads, head_dim // 2)[:tokens]
    byte = packed.to(torch.int64) + 128           # int8 is stored biased
    codes = torch.stack((byte % 16, byte // 16), dim=-1).flatten(-2)   # low nibble first
    rotated = centroids[codes] * scales.unsqueeze(-1)                  # this is Pi @ k
    return rotated @ pi                                                # Pi is an involution


scales = blob["scale_planes"].reshape(blob["scale_planes"].shape[0], -1, blob["scale_planes"].shape[-1])
keys = [rebuild(blob["key_planes"], scales[i, :tokens, :kv_heads], i) for i in range(blob["key_planes"].shape[0])]
values = [
    rebuild(blob["value_planes"], scales[i, :tokens, kv_heads : 2 * kv_heads], i)
    for i in range(blob["value_planes"].shape[0])
]

print(keys[0].shape)   # torch.Size([{cached_tokens}, {kv_heads}, {head_dim}])
```

Against the unquantised K these reconstruct to a cosine of about **0.995** --
that gap is the 4 bits, and it is the thing worth measuring.

## Positions

Paging is the identity: slot *i* holds position *i*, and `block_table` is
`arange`. So `planes[layer].reshape(-1, kv_heads, head_dim // 2)[p]` is
position *p* directly, and `slot_mapping` is written down only so that an
analysis which assumes otherwise has something to check against. Rows at or past
`context_lens` were never written and are zeros.
"""


def dump_quantised_cache(
    runner,
    directory: Path,
    *,
    model_path: str,
    family: str,
    cached_tokens: int,
    context_tokens: int,
    prompt: str,
    needle: dict,
) -> DumpReport:
    """Write the tensors, the metadata and the guide into ``directory``.

    ``cached_tokens`` is how much of the pool is live, which is the prompt plus
    whatever the run generated on top of it -- not the prompt alone. The two are
    both in the metadata, because a compression study that assumed the cache
    ended where the prompt did would be reading a few dozen positions of
    generated context as though they were haystack.
    """
    tensors = collect_tensors(runner, cached_tokens)
    metadata = build_metadata(
        runner,
        model_path=model_path,
        family=family,
        cached_tokens=cached_tokens,
        context_tokens=context_tokens,
        prompt=prompt,
        needle=needle,
    )
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / TENSOR_FILE
    torch.save(tensors, tensor_path)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / README_FILE).write_text(render_readme(metadata, tensors), encoding="utf-8")
    return DumpReport(
        directory=directory,
        tokens=cached_tokens,
        blocks=metadata["dumped_blocks"],
        layers=metadata["num_layers"],
        bytes_written=tensor_path.stat().st_size,
    )
