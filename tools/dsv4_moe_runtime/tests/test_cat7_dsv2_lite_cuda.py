"""CAT 7 -- DeepSeek-V2-Lite on CUDA: layout profile, budget guard, real shards.

Covers the second architecture target of the static MoE runtime and its
workstation validation device:

* the ``dsv2-lite`` profile against the upstream ``config.json`` (27 layers,
  64 routed experts, top-6, 2 shared experts, dense BF16 slots of 16.50 MiB)
  and the unchanged ``dsv4-flash`` slot contract next to it;
* the 12 GiB VRAM budget guard (pool + resident backbone + workspace reserve);
* sharded safetensors ingestion with Hugging Face naming, 8-byte span alignment
  and BF16 dtype validation -- parsed from an index, streamed through pinned
  host chunks;
* on CUDA hardware: allocator-accounted zero allocation over a full trace,
  pinned-window H2D DMA and byte-exact read-back of device slots.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from ..benchmarks.offload_stress import OffloadStressHarness
from ..benchmarks.stress_config import BenchConfig, parse_config
from ..benchmarks.synthetic_source import SyntheticExpertSource
from ..benchmarks.telemetry import full_fingerprints
from ..benchmarks.trace_simulator import RouterTraceSimulator
from ..core.config import MoEGeometry
from ..core.layout import (
    DENSE_BF16_KINDS,
    SLOT_REGION_ALIGN_BYTES,
    ExpertTensorLayout,
)
from ..core.profiles import (
    DSV2_LITE_PROFILE,
    DSV4_FLASH_PROFILE,
    LAYOUT_PROFILES,
    NAMING_HF_DEEPSEEK,
    BackboneProfile,
    MoELayoutProfile,
    profile_for,
)
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.exchange_buffer import TransitExchangeBuffer
from ..hardware.exclusive_staging import ExclusiveStagingProvider
from ..hardware.runtime import CpuRuntime, CudaRuntime, make_runtime
from ..hardware.safetensors_provider import (
    HF_DEEPSEEK_NAMING,
    WeightLayoutMismatchError,
    naming_scheme,
)
from ..hardware.sharded_safetensors import (
    SafetensorsShardIndex,
    ShardedSafetensorsExpertSource,
    bind_sharded_expert_spans,
)
from ..hardware.vram_budget import (
    DEVICE_WORKSPACE_RESERVE_BYTES,
    MemoryBudgetError,
    assert_budget_fits,
    plan_budget,
)
from .conftest import forbid_torch_allocations

GIB = 1024**3
DSV2_LITE_SLOT_BYTES = 17_301_504  # 3 x 1408 x 2048 x 2 bytes = 16.50 MiB
DSV4_FLASH_SLOT_BYTES = 13_369_344  # the untouched FP4 + E8M0 contract
RTX_5070_TOTAL_BYTES = 12 * GIB

# Reduced V2-Lite-shaped geometry for on-disk fixtures: identical structure
# (dense BF16 gate/up/down, a dense prefix layer, HF naming) at 48 KiB/slot.
CAT7_GEOMETRY = MoEGeometry(
    hidden_size=128,
    moe_intermediate_size=64,
    num_routed_experts=8,
    top_k=2,
    num_layers=3,
    num_hash_layers=0,
    vocab_size=1024,
)
CAT7_LAYOUT = ExpertTensorLayout.for_expert_geometry(
    CAT7_GEOMETRY.hidden_size, CAT7_GEOMETRY.moe_intermediate_size, DENSE_BF16_KINDS
)
CAT7_LAYER_IDS = (1, 2)  # layer 0 is the dense MLP prefix, as in V2-Lite
CAT7_NUM_SHARDS = 2
CAT7_BACKBONE = BackboneProfile(
    hidden_size=128,
    vocab_size=1024,
    num_layers=3,
    num_attention_heads=2,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    kv_lora_rank=32,
    q_lora_rank=None,
    dense_intermediate_size=256,
    first_k_dense_layers=1,  # the dense prefix the trace must skip
    num_shared_experts=2,
    moe_intermediate_size=64,
)
#: Registered into ``LAYOUT_PROFILES`` by the end-to-end fixture test: a real
#: 26.8 GiB DeepSeek-V2-Lite checkpoint cannot live in a unit test, so the
#: profile is the same structure at 48 KiB/slot.
CAT7_PROFILE = MoELayoutProfile(
    name="cat7-tiny",
    description="CAT7 fixture: V2-Lite-shaped dense BF16 at 48 KiB/slot",
    geometry=CAT7_GEOMETRY,
    expert_kinds=DENSE_BF16_KINDS,
    default_pool_slots=4,
    default_transit_slots=4,
    default_hot_experts=2,
    checkpoint_naming=NAMING_HF_DEEPSEEK,
    backbone=CAT7_BACKBONE,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
HARNESS_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


# --------------------------------------------------------------- fixtures


def _expert_payload(layer_idx: int, expert_id: int, param_index: int, num_bytes: int) -> bytes:
    """Deterministic per-(layer, expert, param) bytes; first 3 bytes are the tag."""
    tag = bytes((layer_idx & 0xFF, expert_id & 0xFF, param_index & 0xFF))
    body = bytes(((layer_idx * 97 + expert_id * 31 + param_index * 7 + i) & 0xFF) for i in range(num_bytes - 3))
    return tag + body


def _write_shard(path: Path, tensors: Sequence[tuple[str, bytes, str, tuple[int, ...]]]) -> None:
    """Write one safetensors shard, 8-byte aligned exactly like HF exporters."""
    header: dict[str, dict[str, object]] = {}
    cursor = 0
    for name, payload, dtype, shape in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + len(payload)]}
        cursor += len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Reproduce upstream alignment exactly: safetensors only pads the header to
    # 8 bytes, so the data section (and every span in it) is 8-byte aligned and
    # generally *not* 128-byte aligned the way this runtime's own packer is.
    while (8 + len(header_bytes)) % SLOT_REGION_ALIGN_BYTES != 8:
        header_bytes += b" "
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for _name, payload, _dtype, _shape in tensors:
            handle.write(payload)


def _write_dsv2_lite_checkpoint(
    directory: Path,
    layout: ExpertTensorLayout = CAT7_LAYOUT,
    layer_ids: Sequence[int] = CAT7_LAYER_IDS,
    num_experts: int = CAT7_GEOMETRY.num_routed_experts,
    num_shards: int = CAT7_NUM_SHARDS,
    dtype: str = "BF16",
) -> dict[str, bytes]:
    """Write a sharded HF-style checkpoint; returns every tensor's exact bytes."""
    scheme = naming_scheme(HF_DEEPSEEK_NAMING)
    per_shard: list[list[tuple[str, bytes, str, tuple[int, ...]]]] = [[] for _ in range(num_shards)]
    payloads: dict[str, bytes] = {}
    experts_per_shard = -(-num_experts // num_shards)
    for layer_idx in layer_ids:
        for expert_id in range(num_experts):
            for param_index, spec in enumerate(layout.specs):
                name = scheme.tensor_name(layer_idx, expert_id, spec)
                payload = _expert_payload(layer_idx, expert_id, param_index, spec.num_bytes)
                payloads[name] = payload
                per_shard[expert_id // experts_per_shard].append((name, payload, dtype, spec.logical_shape))
    weight_map: dict[str, str] = {}
    for shard_index, tensors in enumerate(per_shard):
        shard_name = f"model-{shard_index + 1:05d}-of-{num_shards:05d}.safetensors"
        _write_shard(directory / shard_name, tensors)
        for name, _payload, _dtype, _shape in tensors:
            weight_map[name] = shard_name
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(len(p) for p in payloads.values())}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return payloads


@pytest.fixture
def dsv2_lite_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    payloads = _write_dsv2_lite_checkpoint(tmp_path)
    return tmp_path, payloads


def _expected_param_bytes(payloads: dict[str, bytes], layer_idx: int, expert_id: int, param_key: str) -> bytes:
    spec = CAT7_LAYOUT.spec_for(param_key)
    return payloads[naming_scheme(HF_DEEPSEEK_NAMING).tensor_name(layer_idx, expert_id, spec)]


class _FakeCapacityRuntime:
    """Capacity-probe stub: a 12 GiB device with a configurable free pool."""

    def __init__(self, free_bytes: int, total_bytes: int = RTX_5070_TOTAL_BYTES):
        self._free = free_bytes
        self._total = total_bytes

    def device_total_memory(self) -> int | None:
        return self._total

    def device_free_memory(self) -> int | None:
        return self._free


# ------------------------------------------------------- layout & profile


def test_dsv2_lite_profile_matches_upstream_config() -> None:
    profile = profile_for(DSV2_LITE_PROFILE)
    geometry = profile.geometry
    assert (geometry.num_layers, geometry.num_routed_experts, geometry.top_k) == (27, 64, 6)
    assert (geometry.hidden_size, geometry.moe_intermediate_size, geometry.vocab_size) == (2048, 1408, 102400)
    assert geometry.num_hash_layers == 0  # V2-Lite has no tid2eid hash layers
    assert profile.num_shared_experts == 2
    assert (profile.first_moe_layer, profile.num_moe_layers) == (1, 26)
    assert list(profile.moe_layer_ids) == list(range(1, 27))


def test_dsv2_lite_slot_is_dense_bf16_without_scales() -> None:
    layout = profile_for(DSV2_LITE_PROFILE).expert_layout
    assert [spec.param_key for spec in layout.specs] == ["w1", "w2", "w3"]
    assert not any(spec.is_scale for spec in layout.specs)
    assert {spec.num_bytes for spec in layout.specs} == {1408 * 2048 * 2}
    assert layout.slot_num_bytes == DSV2_LITE_SLOT_BYTES == int(16.5 * 1024 * 1024)
    assert all(spec.offset_bytes % SLOT_REGION_ALIGN_BYTES == 0 for spec in layout.specs)
    assert layout.spec_for("w1").logical_shape == (1408, 2048)
    assert layout.spec_for("w2").logical_shape == (2048, 1408)
    assert layout.spec_for("w3").logical_shape == (1408, 2048)
    with pytest.raises(KeyError):
        layout.spec_for("w1_scale")


def test_dsv4_flash_slot_contract_is_unchanged() -> None:
    layout = profile_for(DSV4_FLASH_PROFILE).expert_layout
    assert layout.slot_num_bytes == DSV4_FLASH_SLOT_BYTES
    assert [spec.param_key for spec in layout.specs] == ["w1", "w1_scale", "w2", "w2_scale", "w3", "w3_scale"]
    assert sum(spec.is_scale for spec in layout.specs) == 3


def test_dense_bf16_slot_views_reinterpret_as_bfloat16() -> None:
    pool = StaticExpertSlotPool(CAT7_GEOMETRY, num_slots=2, layout=CAT7_LAYOUT)
    views = pool.param_views(0)
    assert set(views) == {"w1", "w2", "w3"}
    for param_key, byte_view in views.items():
        spec = CAT7_LAYOUT.spec_for(param_key)
        assert byte_view.dtype == torch.uint8 and tuple(byte_view.shape) == spec.view_shape
        elements = byte_view.view(torch.bfloat16)
        assert tuple(elements.shape) == spec.logical_shape
        elements.fill_(torch.finfo(torch.bfloat16).max)
        assert torch.equal(byte_view, pool.param_views(0)[param_key])  # same storage, no copy
    assert pool.scale_views(0) == {}


def test_backbone_reservation_accounts_shared_experts() -> None:
    backbone = profile_for(DSV2_LITE_PROFILE).backbone
    assert backbone is not None
    breakdown = backbone.breakdown()
    # 26 MoE layers x gate/up/down x hidden x (2 shared experts x 1408) x 2 B
    assert breakdown["shared_experts"] == 26 * 3 * 2048 * (2 * 1408) * 2
    assert breakdown["embeddings"] == 2 * 102400 * 2048 * 2  # embed_tokens + lm_head
    assert breakdown["dense_mlp"] == 1 * 3 * 2048 * 10944 * 2  # the single dense prefix layer
    assert breakdown["total"] == sum(value for key, value in breakdown.items() if key != "total")
    assert 2.0 * GIB < breakdown["total"] < 3.5 * GIB  # the ~3.5 GiB backbone target


# ------------------------------------------------------------ CLI wiring


def test_cli_applies_dsv2_lite_profile_defaults() -> None:
    config = parse_config(["--layout", DSV2_LITE_PROFILE, "--device", "cuda:0"])
    assert config.device == "cuda:0"
    assert (config.pool_slots, config.transit_slots, config.hot_experts) == (384, 256, 16)
    assert config.small_geometry is False
    assert config.layout.slot_num_bytes == DSV2_LITE_SLOT_BYTES
    assert config.first_moe_layer == 1
    assert config.trace_plan() == (26, 64, 0)
    assert config.weights_dir is None and config.count_backbone is True


def test_cli_keeps_dsv4_flash_defaults() -> None:
    config = parse_config([])
    assert config.device == "npu:0"
    assert (config.pool_slots, config.transit_slots, config.hot_experts) == (3200, 2048, 64)
    assert config.layout.slot_num_bytes == DSV4_FLASH_SLOT_BYTES
    assert config.first_moe_layer == 0 and config.trace_plan() == (43, 256, 3)


def test_cli_dry_run_accepts_an_explicit_layout() -> None:
    config = parse_config(["--dry-run", "--layout", DSV2_LITE_PROFILE])
    assert config.device == "cpu" and config.steps == 50
    assert config.pool_slots == 16 and config.transit_slots == 64  # dry-run preset wins
    assert config.layout.slot_num_bytes == DSV2_LITE_SLOT_BYTES


def test_cli_rejects_conflicting_and_unknown_layouts() -> None:
    with pytest.raises(SystemExit):
        parse_config(["--small-geometry", "--layout", DSV2_LITE_PROFILE])
    with pytest.raises(SystemExit):
        parse_config(["--layout", "dsv3-moe"])
    with pytest.raises(SystemExit):  # 2048 x 16.50 MiB = 33 GiB > the 32 GiB host cap
        parse_config(["--layout", DSV2_LITE_PROFILE, "--transit-slots", "2048"])
    with pytest.raises(SystemExit):  # 64 hot experts would leave no cold tail
        parse_config(["--layout", DSV2_LITE_PROFILE, "--hot-experts", "64"])
    with pytest.raises(SystemExit):
        parse_config(["--layout", DSV2_LITE_PROFILE, "--weights-dir", "does-not-exist"])


# --------------------------------------------------------- budget guard


@pytest.mark.parametrize(
    ("pool_slots", "expected_fit"),
    [(384, True), (486, True), (512, False), (2048, False)],
)
def test_budget_guard_verdicts_on_a_12gib_card(pool_slots: int, expected_fit: bool) -> None:
    profile = profile_for(DSV2_LITE_PROFILE)
    free = 10.78 * GIB  # what the RTX 5070 reports free with a desktop attached
    budget = plan_budget(
        profile, pool_slots, transit_slots=256, device="cuda:0", runtime=_FakeCapacityRuntime(int(free))
    )
    assert budget.fits is expected_fit
    assert budget.pool_bytes == pool_slots * DSV2_LITE_SLOT_BYTES
    assert budget.backbone_modeled and budget.backbone_bytes == profile.backbone_num_bytes
    assert budget.reserve_bytes == DEVICE_WORKSPACE_RESERVE_BYTES
    assert budget.transit_host_bytes == 256 * DSV2_LITE_SLOT_BYTES  # host DDR, not VRAM
    if expected_fit:
        assert_budget_fits(budget)
    else:
        with pytest.raises(MemoryBudgetError, match="reduce --pool-slots"):
            assert_budget_fits(budget)


def test_budget_guard_can_exclude_the_backbone_reservation() -> None:
    profile = profile_for(DSV2_LITE_PROFILE)
    runtime = _FakeCapacityRuntime(int(9.0 * GIB))
    with_backbone = plan_budget(profile, 480, 256, "cuda:0", runtime)
    harness_only = plan_budget(profile, 480, 256, "cuda:0", runtime, count_backbone=False)
    assert with_backbone.fits is False and harness_only.fits is True
    assert harness_only.backbone_bytes == 0 and harness_only.backbone_modeled is False
    assert harness_only.deployment_device_bytes == harness_only.pool_bytes


def test_budget_is_unknown_without_a_capacity_probe() -> None:
    budget = plan_budget(profile_for(DSV2_LITE_PROFILE), 384, 256, "cuda:0", runtime=None)
    assert budget.fits is None and budget.headroom_bytes is None and budget.max_pool_slots is None
    assert_budget_fits(budget)  # unknown never blocks a plan
    assert "UNKNOWN" in budget.render()


# ------------------------------------------------- dense-prefix trace offset


def test_trace_skips_the_dense_prefix_layer() -> None:
    profile = profile_for(DSV2_LITE_PROFILE)
    simulator = RouterTraceSimulator(
        profile.geometry,
        num_experts_available=profile.geometry.num_routed_experts,
        hot_experts=16,
        hot_ratio=0.35,
        zipf_exponent=1.2,
        seed=7,
        num_layers=profile.num_moe_layers,
        num_hash_layers=0,
        first_layer_idx=profile.first_moe_layer,
    )
    trace = simulator.generate_trace(200)
    layers = {step.layer_idx for step in trace}
    assert layers == set(range(1, 27)) and all(step.routed_by == "score" for step in trace)
    assert all(len(set(step.expert_ids)) == 6 for step in trace)


def test_synthetic_source_follows_the_layer_offset() -> None:
    source = SyntheticExpertSource(CAT7_LAYOUT, num_layers=2, num_experts=8, seed=3, first_layer=1)
    assert not source.contains(0, 0) and source.contains(1, 0) and source.contains(2, 7)
    assert not source.contains(3, 0)
    region = torch.empty(CAT7_LAYOUT.slot_num_bytes, dtype=torch.uint8)
    source.fill_slot(region, 2, 5)
    assert torch.equal(region[: CAT7_LAYOUT.spec_for("w1").num_bytes].view(-1), source.read_param(2, 5, "w1").view(-1))


# ----------------------------------------------- sharded safetensors (host)


def test_shard_index_parses_every_shard(dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]]) -> None:
    directory, payloads = dsv2_lite_checkpoint
    index = SafetensorsShardIndex.from_directory(directory)
    assert len(index.shard_paths) == CAT7_NUM_SHARDS
    assert set(index.tensor_names) == set(payloads)
    assert index.total_bytes == sum(len(payload) for payload in payloads.values())
    span = index.span(naming_scheme(HF_DEEPSEEK_NAMING).tensor_name(1, 0, CAT7_LAYOUT.specs[0]))
    assert span.dtype == "BF16" and span.shape == CAT7_LAYOUT.specs[0].logical_shape
    assert span.shard.read_bytes()[span.begin : span.begin + span.num_bytes] == _expected_param_bytes(
        payloads, 1, 0, "w1"
    )


def test_sharded_binding_enforces_the_slot_contract(dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]]) -> None:
    directory, _payloads = dsv2_lite_checkpoint
    index = SafetensorsShardIndex.from_directory(directory)
    bound = bind_sharded_expert_spans(index, CAT7_LAYOUT, CAT7_LAYER_IDS, range(8), naming=HF_DEEPSEEK_NAMING)
    assert len(bound) == len(CAT7_LAYER_IDS) * 8
    for spans in bound.values():
        assert sum(span.num_bytes for span in spans.values()) == CAT7_LAYOUT.slot_num_bytes
    # HF exporters align spans to 8 bytes, so strict 128-byte mode must reject.
    with pytest.raises(WeightLayoutMismatchError, match="not 128-byte aligned"):
        bind_sharded_expert_spans(
            index, CAT7_LAYOUT, CAT7_LAYER_IDS, range(8), naming=HF_DEEPSEEK_NAMING, strict_alignment=True
        )
    with pytest.raises(WeightLayoutMismatchError, match="missing tensor"):
        bind_sharded_expert_spans(index, CAT7_LAYOUT, (0,), range(8), naming=HF_DEEPSEEK_NAMING)


def test_sharded_binding_rejects_a_non_bf16_checkpoint(tmp_path: Path) -> None:
    _write_dsv2_lite_checkpoint(tmp_path, dtype="F8_E4M3")
    index = SafetensorsShardIndex.from_directory(tmp_path)
    with pytest.raises(WeightLayoutMismatchError, match="not BF16"):
        bind_sharded_expert_spans(index, CAT7_LAYOUT, CAT7_LAYER_IDS, range(8), naming=HF_DEEPSEEK_NAMING)


def test_hf_naming_maps_gate_up_down() -> None:
    scheme = naming_scheme(HF_DEEPSEEK_NAMING)
    names = [scheme.tensor_name(3, 12, spec) for spec in CAT7_LAYOUT.specs]
    assert names == [
        "model.layers.3.mlp.experts.12.gate_proj.weight",
        "model.layers.3.mlp.experts.12.down_proj.weight",
        "model.layers.3.mlp.experts.12.up_proj.weight",
    ]


def test_sharded_source_streams_into_a_pinned_host_region(
    dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]],
) -> None:
    directory, payloads = dsv2_lite_checkpoint
    runtime = CpuRuntime()
    source = ShardedSafetensorsExpertSource(
        runtime,
        SafetensorsShardIndex.from_directory(directory),
        CAT7_LAYOUT,
        layer_ids=CAT7_LAYER_IDS,
        expert_ids=range(8),
    )
    try:
        assert source.chunks_are_io_aligned()  # unbuffered IO needs 4096-aligned chunks
        assert set(source.backends.values()) <= {"unbuffered", "mmap"}
        assert source.contains(2, 7) and not source.contains(0, 0)
        region = torch.empty(CAT7_LAYOUT.slot_num_bytes, dtype=torch.uint8)
        with forbid_torch_allocations():
            source.fill_slot(region, 2, 7)
        for spec in CAT7_LAYOUT.specs:
            staged = region[spec.offset_bytes : spec.offset_bytes + spec.num_bytes]
            assert bytes(staged.tolist()) == _expected_param_bytes(payloads, 2, 7, spec.param_key)
            assert bytes(source.read_param(2, 7, spec.param_key).flatten().tolist()) == bytes(staged.tolist())
        assert source.bytes_streamed == CAT7_LAYOUT.slot_num_bytes and source.slot_fills == 1
    finally:
        source.close()


# ------------------------------------------------------------ CUDA device


@requires_cuda
def test_cuda_runtime_exposes_accounting_streams_and_capacity() -> None:
    runtime = make_runtime("cuda:0")
    assert isinstance(runtime, CudaRuntime)
    assert runtime.device == "cuda:0" and runtime.has_allocator_accounting
    assert runtime.supports_pinned_host_memory
    total, free = runtime.device_total_memory(), runtime.device_free_memory()
    assert total is not None and free is not None and 0 < free <= total
    assert isinstance(runtime.make_stream(), torch.cuda.Stream)

    baseline = runtime.memory_allocated()
    block = torch.empty(4 * 1024**2, dtype=torch.uint8, device=runtime.device)
    assert runtime.memory_allocated() - baseline >= block.numel()
    del block
    runtime.release_cache()
    assert runtime.memory_allocated() == baseline


@requires_cuda
def test_transit_window_is_pinned_host_memory_on_cuda() -> None:
    runtime = make_runtime("cuda:0")
    buffer = TransitExchangeBuffer(runtime, CAT7_LAYOUT, host_slots=4, overflow="drop_oldest")
    assert buffer.pinned
    assert buffer.host_region(0).is_pinned()
    assert buffer.capacity_bytes == 4 * CAT7_LAYOUT.slot_num_bytes
    fingerprint = buffer.fingerprint()
    device_region = torch.zeros(CAT7_LAYOUT.slot_num_bytes, dtype=torch.uint8, device=runtime.device)
    with forbid_torch_allocations():
        buffer.stage_eviction(device_region, key=(1, 0))  # HBM -> pinned DDR
        buffer.synchronize()
    assert buffer.lookup((1, 0)) is not None
    assert buffer.fingerprint() == fingerprint


@requires_cuda
def test_sharded_shards_stream_into_cuda_slots_without_allocating(
    dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]],
) -> None:
    """DirectStorage path: disk -> pinned chunks -> non-blocking H2D into slots."""
    directory, payloads = dsv2_lite_checkpoint
    runtime = make_runtime("cuda:0")
    source = ShardedSafetensorsExpertSource(
        runtime,
        SafetensorsShardIndex.from_directory(directory),
        CAT7_LAYOUT,
        layer_ids=CAT7_LAYER_IDS,
        expert_ids=range(8),
    )
    pool = StaticExpertSlotPool(CAT7_GEOMETRY, num_slots=4, layout=CAT7_LAYOUT, device=runtime.device)
    try:
        assert source.pinned_chunks
        fingerprint = full_fingerprints(pool)
        runtime.synchronize_device()
        baseline = runtime.memory_allocated()
        with forbid_torch_allocations():
            for layer_idx in CAT7_LAYER_IDS:
                for expert_id in (0, 5):  # one expert per shard
                    reservation = pool.acquire_for_step(layer_idx, [expert_id, expert_id + 1], source)
                    try:
                        source.synchronize()
                        runtime.synchronize_device()
                        for expert, slot in zip(reservation.expert_ids, reservation.slot_ids):
                            for param_key, view in pool.param_views(slot).items():
                                assert bytes(view.flatten().cpu().tolist()) == _expected_param_bytes(
                                    payloads, layer_idx, expert, param_key
                                )
                    finally:
                        pool.release_step(reservation)
        assert runtime.memory_allocated() == baseline  # delta == 0 bytes
        assert full_fingerprints(pool) == fingerprint
        assert source.experts_served == 8
    finally:
        source.close()


@requires_cuda
def test_exclusive_staging_promotes_real_shards_into_cuda_slots(
    dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]],
) -> None:
    """Window path: disk -> pinned transit window -> H2D DMA -> device slot."""
    directory, payloads = dsv2_lite_checkpoint
    runtime = make_runtime("cuda:0")
    source = ShardedSafetensorsExpertSource(
        runtime,
        SafetensorsShardIndex.from_directory(directory),
        CAT7_LAYOUT,
        layer_ids=CAT7_LAYER_IDS,
        expert_ids=range(8),
    )
    window = TransitExchangeBuffer(runtime, CAT7_LAYOUT, host_slots=4, overflow="drop_oldest")
    provider = ExclusiveStagingProvider(runtime, CAT7_LAYOUT, window, source)
    pool = StaticExpertSlotPool(
        CAT7_GEOMETRY, num_slots=2, layout=CAT7_LAYOUT, device=runtime.device, exchange_buffer=window
    )
    try:
        baseline = runtime.memory_allocated()
        for step, (layer_idx, experts) in enumerate([(1, [0, 1]), (2, [6, 7]), (1, [0, 1])]):
            reservation = pool.acquire_for_step(layer_idx, experts, provider)
            provider.synchronize()
            runtime.synchronize_device()
            for expert, slot in zip(experts, reservation.slot_ids):
                for param_key, view in pool.param_views(slot).items():
                    assert bytes(view.flatten().cpu().tolist()) == _expected_param_bytes(
                        payloads, layer_idx, expert, param_key
                    ), (step, expert, param_key)
            pool.release_step(reservation)
        assert runtime.memory_allocated() == baseline
        assert provider.window_source_fills >= 4 and pool.stats.evictions > 0
    finally:
        source.close()


@pytest.mark.parametrize("device", HARNESS_DEVICES)
def test_harness_streams_real_shards_end_to_end(
    dsv2_lite_checkpoint: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """``--weights-dir``: disk shards -> transit window -> slots, verified."""
    directory, _payloads = dsv2_lite_checkpoint
    monkeypatch.setitem(LAYOUT_PROFILES, CAT7_PROFILE.name, CAT7_PROFILE)
    config = BenchConfig(
        device=device,
        steps=8,
        pool_slots=4,
        hot_experts=2,
        hot_ratio=0.5,
        zipf_exponent=1.2,
        seed=11,
        small_geometry=False,
        transit_slots=4,
        verify_samples=4,
        buckets=2,
        report_json=None,
        check_only=False,
        layout_name=CAT7_PROFILE.name,
        weights_dir=directory,
    )
    report = OffloadStressHarness(config).run()
    assert "safetensors shards" in report.weight_source and "unbuffered" in report.weight_source
    assert report.slot_num_bytes == CAT7_LAYOUT.slot_num_bytes
    assert report.verification is not None and report.verification[0] and report.verification[1] > 0
    assert report.fingerprints_stable and report.invariants_ok()
    if device != "cpu":
        assert report.baseline_allocated == report.final_allocated


@requires_cuda
def test_dsv2_lite_harness_run_on_cuda_holds_zero_allocation() -> None:
    """The shipped CLI path on cuda:0: 52 steps (2 tokens) of the real profile."""
    config = parse_config(
        [
            "--device",
            "cuda:0",
            "--layout",
            DSV2_LITE_PROFILE,
            "--steps",
            "52",
            "--pool-slots",
            "16",
            "--transit-slots",
            "12",
            "--verify-samples",
            "4",
            "--buckets",
            "4",
        ]
    )
    report = OffloadStressHarness(config).run()
    assert report.layout_name == DSV2_LITE_PROFILE
    assert report.slot_num_bytes == DSV2_LITE_SLOT_BYTES
    assert report.allocator_invariant_ok is True and not report.allocation_violation_steps
    assert report.baseline_allocated == report.final_allocated  # delta == 0 bytes
    assert report.fingerprints_stable and report.verification is not None and report.verification[0]
    assert report.budget.fits is True and report.device_budget_ok
    assert report.host_window_bytes == 12 * DSV2_LITE_SLOT_BYTES
    assert report.invariants_ok()
