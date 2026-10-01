"""Shared-expert paired-slot ingestion, and one MoE layer actually computed.

Two things get verified here that the ingestion tests cannot:

* **Paired reservation.** A shared expert is twice a routed slot, so it occupies
  a consecutive pair carved from the same arena. The pair must be invisible to
  the eviction policy -- a shared expert is used by every token at every MoE
  layer, so it can never be the right victim -- and the routed free list must
  not be able to reach it.
* **Semantics, not just bytes.** Byte-exact ingestion says the right bytes
  arrived; it says nothing about whether the slot view is the right matrix. A
  transposed projection or a BF16 reinterpretation off by a row passes every
  ingestion test and produces garbage. So the layer is computed and held
  against an independent reference built from the same bytes.

Most cases run on a generated 48 KiB/slot checkpoint shaped like V2-Lite, so
they work in CI. The ones that need the 29 GiB DeepSeek-V2-Lite-Chat checkpoint
skip when it is absent and are the only place the real geometry is exercised.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ..core.config import DSV2_LITE_GEOMETRY, DeepSeekV4MoEConfig
from ..core.layout import DENSE_BF16_KINDS, SLOT_REGION_ALIGN_BYTES, ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.runtime import make_runtime
from ..hardware.safetensors_provider import (
    HF_DEEPSEEK_NAMING,
    WeightLayoutMismatchError,
    naming_scheme,
)
from ..hardware.sharded_safetensors import SafetensorsShardIndex, ShardedSafetensorsExpertSource
from ..inference.layer_dispatch import MoELayerScratch, execute_moe_layer, swiglu_into
from .test_cat7_dsv2_lite_cuda import (
    CAT7_GEOMETRY,
    CAT7_LAYER_IDS,
    CAT7_LAYOUT,
    _write_dsv2_lite_checkpoint,
)

#: The fixture backbone declares two shared experts, as V2-Lite does.
CAT7_SHARED_EXPERTS = 2
CAT7_SHARED_LAYOUT = ExpertTensorLayout.for_shared_expert(CAT7_GEOMETRY, CAT7_SHARED_EXPERTS, DENSE_BF16_KINDS)

REAL_CHECKPOINT = Path(os.environ.get("DSV2_LITE_CHECKPOINT", r"F:\AI\models\DeepSeek-V2-Lite-Chat"))

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_real_checkpoint = pytest.mark.skipif(
    not (REAL_CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"no DeepSeek-V2-Lite checkpoint at {REAL_CHECKPOINT}",
)

#: BF16 GEMM accumulation against an FP32 reference lands here; far enough from
#: 1.0 to tolerate accumulation order, far enough from a wrong orientation
#: (which lands near zero) to be a real gate.
DISPATCH_COSINE = 0.9999


def _shared_checkpoint(directory: Path) -> dict[str, bytes]:
    """The routed fixture plus a shared-expert module per layer, in one shard set."""
    payloads = _write_dsv2_lite_checkpoint(directory)
    # Appended as its own shard so the index spans more than one file, which is
    # how the real checkpoint places experts and shared experts.
    from .test_cat7_dsv2_lite_cuda import _write_shard

    scheme = naming_scheme(HF_DEEPSEEK_NAMING)
    tensors = []
    for layer_idx in CAT7_LAYER_IDS:
        for param_index, spec in enumerate(CAT7_SHARED_LAYOUT.specs):
            name = scheme.shared_tensor_name(layer_idx, spec)
            payload = bytes((layer_idx * 7 + param_index * 13 + index) % 251 for index in range(spec.num_bytes))
            payloads[name] = payload
            tensors.append((name, payload, "BF16", spec.logical_shape))
    shard_name = "model-00003-of-00003.safetensors"
    _write_shard(directory / shard_name, tensors)
    index_path = directory / "model.safetensors.index.json"
    import json

    document = json.loads(index_path.read_text(encoding="utf-8"))
    for name, _payload, _dtype, _shape in tensors:
        document["weight_map"][name] = shard_name
    index_path.write_text(json.dumps(document), encoding="utf-8")
    return payloads


@pytest.fixture
def shared_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    return tmp_path, _shared_checkpoint(tmp_path)


def _source(directory: Path, device: str) -> ShardedSafetensorsExpertSource:
    return ShardedSafetensorsExpertSource(
        make_runtime(device),
        SafetensorsShardIndex.from_directory(directory),
        CAT7_LAYOUT,
        layer_ids=CAT7_LAYER_IDS,
        expert_ids=range(CAT7_GEOMETRY.num_routed_experts),
        naming=HF_DEEPSEEK_NAMING,
        strict_alignment=False,
        shared_layout=CAT7_SHARED_LAYOUT,
    )


def _pool(device: str, num_slots: int = 4, shared_layers=CAT7_LAYER_IDS) -> StaticExpertSlotPool:
    return StaticExpertSlotPool(
        CAT7_GEOMETRY,
        num_slots=num_slots,
        layout=CAT7_LAYOUT,
        device=device,
        shared_layout=CAT7_SHARED_LAYOUT,
        shared_layers=shared_layers,
    )


# --------------------------------------------------------------- the layout


class TestSharedExpertLayout:
    def test_a_shared_expert_is_a_whole_number_of_routed_slots(self) -> None:
        """Two shared experts of ``moe_intermediate`` width tile exactly two slots."""
        assert CAT7_SHARED_LAYOUT.slots_per_region(CAT7_LAYOUT) == CAT7_SHARED_EXPERTS
        assert CAT7_SHARED_LAYOUT.slot_num_bytes == CAT7_SHARED_EXPERTS * CAT7_LAYOUT.slot_num_bytes

    def test_the_real_geometry_is_exactly_two_slots_with_no_padding(self) -> None:
        """33.00 MiB over 16.50 MiB, which is what makes pairing viable at all."""
        routed = ExpertTensorLayout.for_dense_bf16(DSV2_LITE_GEOMETRY)
        shared = ExpertTensorLayout.for_shared_expert(DSV2_LITE_GEOMETRY, 2)
        assert routed.slot_num_bytes == 17_301_504
        assert shared.slot_num_bytes == 34_603_008
        assert shared.slots_per_region(routed) == 2
        # No padding: a partial trailing slot would belong to neither region.
        assert shared.slot_num_bytes == 2 * routed.slot_num_bytes

    def test_every_shared_region_is_128_byte_aligned(self) -> None:
        for shared in (CAT7_SHARED_LAYOUT, ExpertTensorLayout.for_shared_expert(DSV2_LITE_GEOMETRY, 2)):
            for spec in shared.specs:
                assert spec.offset_bytes % SLOT_REGION_ALIGN_BYTES == 0, spec.name
                assert spec.num_bytes % SLOT_REGION_ALIGN_BYTES == 0, spec.name
            assert shared.slot_num_bytes % SLOT_REGION_ALIGN_BYTES == 0

    def test_a_region_that_does_not_tile_the_slot_is_refused(self) -> None:
        """Rounding up would leave a tail reachable from both the free list and the views."""
        odd = DeepSeekV4MoEConfig(
            hidden_size=128,
            moe_intermediate_size=64,
            num_routed_experts=8,
            top_k=2,
            num_layers=3,
            num_hash_layers=0,
            vocab_size=1024,
        )
        routed = ExpertTensorLayout.for_expert_geometry(128, 64, DENSE_BF16_KINDS)
        # Three halves of a slot: a legal layout, not a legal pairing.
        awkward = ExpertTensorLayout.for_expert_geometry(odd.hidden_size, 96, DENSE_BF16_KINDS)
        with pytest.raises(ValueError, match="whole-number ratio"):
            awkward.slots_per_region(routed)

    def test_an_unaligned_region_offset_is_refused(self) -> None:
        arena = torch.zeros(CAT7_SHARED_LAYOUT.slot_num_bytes + 256, dtype=torch.uint8)
        with pytest.raises(ValueError, match="aligned"):
            CAT7_SHARED_LAYOUT.slice_region_views(arena, 8)


# ----------------------------------------------------------- the reservation


class TestPairedSlotReservation:
    def test_the_pair_sits_above_the_routed_slots(self) -> None:
        pool = _pool("cpu", num_slots=4)
        assert pool.num_slots == 4
        assert pool.reserved_slot_count == len(CAT7_LAYER_IDS) * CAT7_SHARED_EXPERTS
        assert pool.total_slot_count == 4 + pool.reserved_slot_count
        assert pool.shared_slot_ids(CAT7_LAYER_IDS[0]) == (4, 5)
        assert pool.shared_slot_ids(CAT7_LAYER_IDS[1]) == (6, 7)
        expected = pool.total_slot_count * CAT7_LAYOUT.slot_num_bytes
        assert pool.slot_arena.numel() == expected

    def test_the_policy_never_sees_a_reserved_slot(self) -> None:
        """Routed admission must not be able to land on the shared region.

        The shared expert is read every layer, so an eviction policy that could
        choose it would evict and re-admit it on every step -- and the bytes it
        overwrote would be read as a routed expert.
        """
        pool = _pool("cpu", num_slots=CAT7_GEOMETRY.top_k)
        reserved = {slot for layer in CAT7_LAYER_IDS for slot in pool.shared_slot_ids(layer)}
        assert reserved
        # Every slot the policy can hand out comes from the routed range.
        for _ in range(20):
            decision = pool.policy.plan_admissions(
                [(CAT7_LAYER_IDS[0], expert) for expert in range(CAT7_GEOMETRY.top_k)],
                {},
                list(range(pool.num_slots)),
                locked_slots=[],
            )
            handed_out = {slot for _key, slot in decision.admissions} | {slot for _key, slot in decision.hits}
            assert handed_out.isdisjoint(reserved)
            assert all(slot < pool.num_slots for slot in handed_out)

    def test_the_views_cover_the_pair_exactly(self) -> None:
        pool = _pool("cpu")
        layer = CAT7_LAYER_IDS[0]
        region = pool.shared_region(layer)
        assert region.numel() == CAT7_SHARED_LAYOUT.slot_num_bytes
        views = pool.shared_param_views(layer)
        assert sum(view.numel() for view in views.values()) == CAT7_SHARED_LAYOUT.slot_num_bytes
        # The views must alias the pair's bytes, not a copy of them.
        region.fill_(0)
        views["w1"].fill_(7)
        assert int(region[0]) == 7

    def test_shared_layers_without_a_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="without a shared_layout"):
            StaticExpertSlotPool(CAT7_GEOMETRY, num_slots=4, layout=CAT7_LAYOUT, device="cpu", shared_layers=(1,))

    def test_a_pool_without_a_reservation_is_unchanged(self) -> None:
        """The default path must not grow an arena or gain reserved slots."""
        pool = StaticExpertSlotPool(CAT7_GEOMETRY, num_slots=4, layout=CAT7_LAYOUT, device="cpu")
        assert pool.reserved_slot_count == 0
        assert pool.total_slot_count == pool.num_slots
        assert pool.slot_arena.numel() == 4 * CAT7_LAYOUT.slot_num_bytes
        with pytest.raises(KeyError, match="no shared-expert reservation"):
            pool.shared_slot_ids(1)


# ------------------------------------------------------------- the ingestion


class TestSharedExpertIngestion:
    def test_the_bytes_are_the_checkpoint_s(self, shared_checkpoint) -> None:
        directory, payloads = shared_checkpoint
        source = _source(directory, "cpu")
        pool = _pool("cpu")
        try:
            staged = pool.fill_shared_experts(source)
            assert staged == len(CAT7_LAYER_IDS) * CAT7_SHARED_LAYOUT.slot_num_bytes
            scheme = naming_scheme(HF_DEEPSEEK_NAMING)
            for layer_idx in CAT7_LAYER_IDS:
                assert pool.shared_expert_is_resident(layer_idx)
                views = pool.shared_param_views(layer_idx)
                for spec in CAT7_SHARED_LAYOUT.specs:
                    expected = payloads[scheme.shared_tensor_name(layer_idx, spec)]
                    assert bytes(views[spec.param_key].reshape(-1).numpy()) == expected, spec.param_key
        finally:
            source.close()

    def test_binding_against_the_routed_layout_is_refused(self, shared_checkpoint) -> None:
        """A shared expert bound as a routed one would load a fraction of each tensor.

        Caught on the span size, before the shape check even runs: the routed
        layout asks for half of each shared projection, which is a legal read of
        the right shape at the right offset and completely wrong.
        """
        directory, _payloads = shared_checkpoint
        source = _source(directory, "cpu")
        try:
            with pytest.raises(WeightLayoutMismatchError, match="layout requires"):
                source.bind_shared_expert(CAT7_LAYER_IDS[0], CAT7_LAYOUT)
        finally:
            source.close()

    def test_a_source_without_a_shared_layout_says_so(self, shared_checkpoint) -> None:
        directory, _payloads = shared_checkpoint
        source = ShardedSafetensorsExpertSource(
            make_runtime("cpu"),
            SafetensorsShardIndex.from_directory(directory),
            CAT7_LAYOUT,
            layer_ids=CAT7_LAYER_IDS,
            expert_ids=range(CAT7_GEOMETRY.num_routed_experts),
            naming=HF_DEEPSEEK_NAMING,
            strict_alignment=False,
        )
        pool = _pool("cpu")
        try:
            with pytest.raises(WeightLayoutMismatchError, match="without a shared_layout"):
                pool.fill_shared_experts(source)
        finally:
            source.close()

    def test_a_naming_scheme_without_shared_experts_says_so(self) -> None:
        from ..hardware.safetensors_provider import DSV4_FLAT_NAMING

        with pytest.raises(WeightLayoutMismatchError, match="does not name shared experts"):
            DSV4_FLAT_NAMING.shared_tensor_name(1, CAT7_SHARED_LAYOUT.specs[0])


# -------------------------------------------------------------- the dispatch


def _reference_swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


class TestLayerDispatch:
    def test_swiglu_into_matches_the_obvious_expression(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(1, 128, dtype=torch.float32)
        gate, up = torch.randn(64, 128), torch.randn(64, 128)
        down = torch.randn(128, 64)
        out = torch.zeros(1, 128)
        swiglu_into(x, gate, up, down, torch.zeros(1, 64), torch.zeros(1, 64), out)
        assert torch.allclose(out, _reference_swiglu(x, gate, up, down), atol=1e-4)

    def test_a_sequence_longer_than_the_scratchpad_is_refused(self, shared_checkpoint) -> None:
        """A step is one token's routing; batching is refused, not approximated."""
        directory, _payloads = shared_checkpoint
        source = _source(directory, "cpu")
        pool = _pool("cpu")
        scratch = MoELayerScratch(CAT7_GEOMETRY, CAT7_LAYOUT, CAT7_SHARED_LAYOUT, device="cpu")
        router = torch.zeros(CAT7_GEOMETRY.num_routed_experts, CAT7_GEOMETRY.hidden_size)
        try:
            pool.fill_shared_experts(source)
            with pytest.raises(NotImplementedError, match="one token"):
                execute_moe_layer(
                    torch.zeros(4, CAT7_GEOMETRY.hidden_size), CAT7_LAYER_IDS[0], router, pool, source, scratch
                )
        finally:
            source.close()

    def test_computing_against_an_unfilled_shared_region_is_refused(self, shared_checkpoint) -> None:
        """Zeroed BF16 is a wrong answer, not an error, so the dispatcher checks."""
        directory, _payloads = shared_checkpoint
        source = _source(directory, "cpu")
        pool = _pool("cpu")
        scratch = MoELayerScratch(CAT7_GEOMETRY, CAT7_LAYOUT, CAT7_SHARED_LAYOUT, device="cpu")
        router = torch.zeros(CAT7_GEOMETRY.num_routed_experts, CAT7_GEOMETRY.hidden_size)
        try:
            with pytest.raises(RuntimeError, match="never filled"):
                execute_moe_layer(
                    torch.zeros(1, CAT7_GEOMETRY.hidden_size), CAT7_LAYER_IDS[0], router, pool, source, scratch
                )
        finally:
            source.close()

    def test_the_step_allocates_nothing_and_moves_nothing(self, shared_checkpoint) -> None:
        directory, _payloads = shared_checkpoint
        source = _source(directory, "cpu")
        pool = _pool("cpu", num_slots=CAT7_GEOMETRY.top_k * 2)
        scratch = MoELayerScratch(CAT7_GEOMETRY, CAT7_LAYOUT, CAT7_SHARED_LAYOUT, device="cpu")
        torch.manual_seed(1)
        router = torch.randn(CAT7_GEOMETRY.num_routed_experts, CAT7_GEOMETRY.hidden_size, dtype=torch.bfloat16)
        x = torch.randn(1, CAT7_GEOMETRY.hidden_size, dtype=torch.bfloat16)
        try:
            pool.fill_shared_experts(source)
            execute_moe_layer(x, CAT7_LAYER_IDS[0], router, pool, source, scratch)  # warm
            fingerprint = scratch.fingerprint()
            arena_ptr = pool.slot_arena.data_ptr()
            for step in range(8):
                pool.advance_generation(step)
                x.add_(1e-3)
                result = execute_moe_layer(x, CAT7_LAYER_IDS[0], router, pool, source, scratch)
                assert len(result.expert_ids) == CAT7_GEOMETRY.top_k
                assert len(set(result.expert_ids)) == CAT7_GEOMETRY.top_k
                assert scratch.fingerprint() == fingerprint, f"step {step}: a scratch buffer moved"
                assert pool.slot_arena.data_ptr() == arena_ptr
                assert result.output.data_ptr() == scratch.y.data_ptr()
        finally:
            source.close()


# ----------------------------------------------- the real 29 GiB checkpoint


@requires_real_checkpoint
class TestRealCheckpoint:
    LAYER = 1

    def _build(self, device: str):
        routed = ExpertTensorLayout.for_dense_bf16(DSV2_LITE_GEOMETRY)
        shared = ExpertTensorLayout.for_shared_expert(DSV2_LITE_GEOMETRY, 2)
        index = SafetensorsShardIndex.from_directory(REAL_CHECKPOINT)
        source = ShardedSafetensorsExpertSource(
            make_runtime(device),
            index,
            routed,
            layer_ids=[self.LAYER],
            expert_ids=range(DSV2_LITE_GEOMETRY.num_routed_experts),
            naming=HF_DEEPSEEK_NAMING,
            strict_alignment=False,
            shared_layout=shared,
        )
        pool = StaticExpertSlotPool(
            DSV2_LITE_GEOMETRY,
            num_slots=16,
            layout=routed,
            device=device,
            shared_layout=shared,
            shared_layers=[self.LAYER],
        )
        return index, source, pool, routed, shared

    def test_the_shared_expert_arrives_byte_exact(self) -> None:
        index, source, pool, _routed, shared = self._build("cpu")
        try:
            staged = pool.fill_shared_experts(source)
            assert staged == shared.slot_num_bytes == 34_603_008
            host = bytes(pool.shared_region(self.LAYER).reshape(-1).numpy())
            for spec in shared.specs:
                span = source.bind_shared_expert(self.LAYER, shared)[spec.param_key]
                with open(span.shard, "rb") as handle:  # independent of the loader
                    handle.seek(span.begin)
                    expected = handle.read(span.num_bytes)
                got = host[spec.offset_bytes : spec.offset_bytes + spec.num_bytes]
                assert got == expected, spec.param_key
        finally:
            source.close()

    @requires_cuda
    def test_one_token_through_layer_one_matches_an_fp32_reference(self) -> None:
        """The whole point: the staged bytes, multiplied, against the same bytes in fp32.

        Byte equality cannot catch a transposed projection; this can.
        """
        index, source, pool, routed, shared = self._build("cuda:0")
        try:
            pool.fill_shared_experts(source)
            prefix = f"model.layers.{self.LAYER}.mlp"
            gate_span = index.span(f"{prefix}.gate.weight")
            router_bytes = torch.empty(gate_span.num_bytes, dtype=torch.uint8, device="cuda:0")
            source._loader_for(gate_span).stream_into(router_bytes.view(-1), gate_span.begin, gate_span.num_bytes)
            router = router_bytes.view(torch.bfloat16).view(gate_span.shape)

            scratch = MoELayerScratch(DSV2_LITE_GEOMETRY, routed, shared, device="cuda:0")
            torch.manual_seed(7)
            x = torch.randn(1, DSV2_LITE_GEOMETRY.hidden_size, device="cuda:0", dtype=torch.bfloat16) * 0.02
            result = execute_moe_layer(x, self.LAYER, router, pool, source, scratch)
            torch.cuda.synchronize()

            def host_weight(name: str, shape: tuple[int, int]) -> torch.Tensor:
                span = index.span(name)
                with open(span.shard, "rb") as handle:
                    handle.seek(span.begin)
                    raw = handle.read(span.num_bytes)
                return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(shape).float()

            host_x = x.float().cpu()
            probabilities = torch.softmax(host_x @ host_weight(f"{prefix}.gate.weight", (64, 2048)).t(), dim=-1)
            values, indices = torch.topk(probabilities, DSV2_LITE_GEOMETRY.top_k, dim=-1)
            assert [int(index_value) for index_value in indices[0]] == list(result.expert_ids)

            reference = _reference_swiglu(
                host_x,
                host_weight(f"{prefix}.shared_experts.gate_proj.weight", (2816, 2048)),
                host_weight(f"{prefix}.shared_experts.up_proj.weight", (2816, 2048)),
                host_weight(f"{prefix}.shared_experts.down_proj.weight", (2048, 2816)),
            )
            for position in range(DSV2_LITE_GEOMETRY.top_k):
                expert = int(indices[0, position])
                reference = reference + values[0, position] * _reference_swiglu(
                    host_x,
                    host_weight(f"{prefix}.experts.{expert}.gate_proj.weight", (1408, 2048)),
                    host_weight(f"{prefix}.experts.{expert}.up_proj.weight", (1408, 2048)),
                    host_weight(f"{prefix}.experts.{expert}.down_proj.weight", (2048, 1408)),
                )

            got = result.output.float().cpu().flatten().double()
            want = reference.flatten().double()
            cosine = float(torch.dot(got, want) / (got.norm() * want.norm()))
            assert cosine > DISPATCH_COSINE, f"cos {cosine} -- the slot views are not the matrices they should be"
        finally:
            source.close()
