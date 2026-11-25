# SPDX-License-Identifier: Apache-2.0
import inspect
import re
import struct
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention import turboquant_e8 as e8


def test_e8_packet_bytes_and_strided_views():
    labels = torch.arange(3 * 64 * 32).reshape(3, 64, 32).to(torch.uint16)
    scales = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64) * 0.125
    packed = e8.pack_packets(labels, scales)
    assert packed.shape == (3, 4352)
    assert packed.stride() == (4352, 1)
    got_labels, got_scales = e8.unpack_packets(packed)
    assert got_labels.stride() == (2176, 32, 1)
    assert got_scales.stride() == (1088, 1)
    torch.testing.assert_close(got_labels, labels)
    torch.testing.assert_close(got_scales, scales)
    # Byte checks independently pin the UINT16 little-endian alphabet and scale offset.
    assert packed[0, :4].tolist() == [0, 0, 1, 0]
    assert packed[1, 4096:4100].tolist() == list(struct.pack("<f", 8.0))
    assert got_labels.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert got_scales.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()


@pytest.mark.parametrize("count", [4, 6, 8, 254, 256])
def test_e8_offset_table_has_full_dma_burst(count):
    offsets = e8.packet_byte_offsets([1, 0] * (count // 2), 2)
    assert offsets.dtype == torch.int64
    assert offsets.numel() * 8 % 32 == 0
    assert offsets[:count].tolist() == [4352, 0] * (count // 2)
    assert offsets[count:].tolist() == [0] * (-count % 4)


@pytest.mark.parametrize("ids,total", [([0] * 3, 1), ([0] * 258, 1), ([0, 1, 2, 3], 3), ([-1] * 4, 2), ([0.5] * 4, 2)])
def test_e8_offset_table_refuses_invalid_metadata(ids, total):
    with pytest.raises(ValueError):
        e8.packet_byte_offsets(ids, total)


@pytest.mark.parametrize(
    "fn,args",
    [
        (e8.pack_packets, (torch.zeros(1, 64, 32, dtype=torch.int8), torch.ones(1, 64))),
        (e8.unpack_packets, (torch.zeros(1, 4352, dtype=torch.int8),)),
    ],
)
def test_e8_refuses_native_nibble_buffers(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_e8_schema_call_order_mutation_and_meta_registration():
    root = Path(e8.__file__).resolve().parents[2]
    text = (root / "csrc/attention/turboquant/e8_torch_bindings.cpp").read_text()
    literals = r'"(?:[^"\\]|\\.)*"'
    schemas = {}
    for match in re.finditer(r"ops\.def\(\s*((?:" + literals + r"\s*)+)\)\s*;", text):
        schema = torch._C.parse_schema("".join(s[1:-1] for s in re.findall(literals, match[1])))
        schemas[schema.name] = schema
    assert set(schemas) == {"reader", "dequant"}
    for name, schema in schemas.items():
        fn = getattr(e8, name)
        assert [a.name for a in schema.arguments] == list(inspect.signature(fn).parameters)
        assert not schema.returns
        assert f'ops.impl("{name}", c10::kMeta,' in text
        assert f'ops.impl("{name}", c10::kPrivateUse1,' in text
        args = [object() for _ in schema.arguments]
        op = Mock()
        with patch.object(torch.ops.turboquant_e8_ops, name, op, create=True):
            fn(*args)
        assert list(op.call_args.args) == args
    writes = {a.name for a in schemas["reader"].arguments if a.alias_info and a.alias_info.is_write}
    assert writes == {"output", "key_witness", "value_witness", "score_witness"}
    writes = {a.name for a in schemas["dequant"].arguments if a.alias_info and a.alias_info.is_write}
    assert writes == {"bf16", "int8"}


def test_e8_runtime_wrappers_do_not_allocate_or_copy():
    for fn in (e8.reader, e8.dequant):
        source = inspect.getsource(fn)
        assert not re.search(r"\.(empty|zeros|tensor|to|cpu|clone|contiguous)\(", source)
