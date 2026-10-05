"""Offline routed-expert E4M3FN conversion; leaves all other weights unchanged."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
ROUTED_WEIGHT = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def quantize_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar decoding scale; zero tensors use unity to avoid division by zero."""
    value = weight.float()
    if not torch.isfinite(value).all():
        raise ValueError("source weight contains non-finite values")
    maximum = value.abs().amax()
    scale = torch.where(maximum > 0, maximum / FP8_MAX, torch.ones_like(maximum)).reshape(1, 1)
    quantized = (value / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if not torch.isfinite(quantized.float()).all():
        raise ValueError("FP8 conversion produced non-finite values")
    return quantized.cpu(), scale.cpu()


def convert(source: Path, destination: Path, device: str = "cuda:0", resume: bool = False) -> dict:
    """Write each shard atomically and publish the index only after validation."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and output must differ")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError("output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, destination / path.name)
    weight_map = {}
    total_bytes = 0
    quantized_count = 0
    for path in sorted(source.glob("*.safetensors")):
        header = {}
        cursor = 0
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            metadata = shard.metadata()
            names = shard.keys()
            for name in names:
                shape = shard.get_slice(name).get_shape()
                count = 1
                for dimension in shape:
                    count *= dimension
                if ROUTED_WEIGHT.fullmatch(name):
                    header[name] = {"dtype": "F8_E4M3", "shape": shape, "data_offsets": [cursor, cursor + count]}
                    cursor += count
                    header[name + "_scale_inv"] = {
                        "dtype": "F32",
                        "shape": [1, 1],
                        "data_offsets": [cursor, cursor + 4],
                    }
                    cursor += 4
                    quantized_count += 1
                else:
                    if shard.get_slice(name).get_dtype() != "BF16":
                        raise ValueError(f"non-BF16 source: {name}")
                    header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [cursor, cursor + count * 2]}
                    cursor += count * 2
            if metadata:
                header["__metadata__"] = metadata
            temporary = destination / (path.name + ".tmp")
            existing = destination / path.name
            # Stream one tensor at a time: safetensors save_file serializes a
            # full shard and can temporarily duplicate several GiB of RAM.
            if not (resume and existing.exists()):
                encoded = json.dumps(header).encode("utf-8")
                encoded += b" " * (-len(encoded) % 8)
                with temporary.open("wb") as output:
                    output.write(struct.pack("<Q", len(encoded)))
                    output.write(encoded)
                    for name in names:
                        weight = shard.get_tensor(name)
                        tensors = quantize_tensor(weight.to(device)) if ROUTED_WEIGHT.fullmatch(name) else (weight,)
                        for tensor in tensors:
                            output.write(tensor.view(torch.uint8).numpy().tobytes())
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(existing)
            with safe_open(str(existing), framework="pt", device="cpu") as check:
                if set(check.keys()) != set(header) - {"__metadata__"}:
                    raise ValueError("output shard key mismatch")
                for name in names:
                    weight = shard.get_tensor(name)
                    expected = quantize_tensor(weight.to(device)) if ROUTED_WEIGHT.fullmatch(name) else (weight,)
                    keys = (name, name + "_scale_inv") if len(expected) == 2 else (name,)
                    for key, tensor in zip(keys, expected):
                        restored = check.get_tensor(key)
                        if restored.shape != tensor.shape or restored.dtype != tensor.dtype:
                            raise ValueError(f"output tensor schema mismatch: {key}")
                        if not torch.isfinite(restored.float()).all():
                            raise ValueError(f"output tensor contains non-finite values: {key}")
                        if not torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8)):
                            raise ValueError(f"output tensor bytes mismatch: {key}")
                        weight_map[key] = path.name
                        total_bytes += tensor.numel() * tensor.element_size()
        print(f"Validated {path.name}: {len(header) - bool(metadata)} tensors", flush=True)
    if not weight_map or not quantized_count:
        raise ValueError("no routed expert weights found")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n"
    )
    report = {"format": "routed_e4m3fn_tensorwise", "quantized_tensors": quantized_count, "tensor_bytes": total_bytes}
    (destination / "runtime_fp8.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume", action="store_true", help="verify completed shards and continue interrupted conversion"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(convert(args.source, args.output, args.device, args.resume))


if __name__ == "__main__":
    main()
