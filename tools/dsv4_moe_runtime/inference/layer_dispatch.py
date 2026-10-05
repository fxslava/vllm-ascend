"""One MoE layer of a decode step: route, stage, compute, accumulate.

This closes the loop the rest of the runtime was built for. Until now the
offload path was exercised by a synthetic trace that staged expert bytes and
checked them; here the staged bytes are actually multiplied by something, so
the slot views have to be right in a way byte equality alone does not prove --
a transposed projection or a BF16 reinterpretation off by a row passes every
ingestion test and produces garbage here.

The step is:

1. **Route.** ``softmax`` over all routed experts, then top-k. DeepSeek-V2-Lite
   is ``scoring_func: softmax``, ``topk_method: greedy``,
   ``norm_topk_prob: false``, ``routed_scaling_factor: 1.0``, so the top-k
   probabilities are used *unnormalised* and unscaled -- the weights do not sum
   to one, and renormalising them (as several MoE implementations do) changes
   the layer's output.
2. **Stage.** ``pool.acquire_for_step`` hands the policy the top-k and gets back
   resident slots, admitting what is missing through the provider.
3. **Shared expert.** Unconditional for every token, so it is read from its
   reserved slot pair -- never admitted, never evicted.
4. **Routed experts.** SwiGLU from the slot views, scaled by the router weight.
5. **Accumulate.** ``y = shared + sum_k w_k * expert_k(x)``.

**Zero allocation.** Every intermediate lives in :class:`MoELayerScratch`,
allocated once, and every operation writes through ``out=`` or an in-place
variant. That includes the softmax, which is spelled out as
max/sub/exp/sum/div because ``torch.softmax`` has no ``out=``.

**One token per call.** Not a simplification of this module but of the
runtime's shape: ``acquire_for_step`` takes exactly one top-k expert list, so a
step *is* one token's routing. Batching would need per-expert token gathering
with counts unknown until routing runs, which cannot be allocation-free without
a worst-case buffer per expert; :func:`execute_moe_layer` refuses a longer
sequence rather than pretending.

**One host sync per layer, unavoidably.** The host has to learn which experts to
stage before it can stage them, so the top-k indices come back across the bus.
That is inherent to weight offload, not an oversight -- see
``AGENTS.md`` on ``tensor.item()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import FP8_KIND, ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..protocols.provider import WeightProviderProtocol

#: The activation dtype a dense BF16 checkpoint's slots reinterpret to.
COMPUTE_DTYPE = torch.bfloat16
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

__all__ = ["MoELayerScratch", "MoELayerResult", "execute_moe_layer", "swiglu_into"]


@dataclass
class MoELayerResult:
    """What one layer's dispatch did, for the benchmark to report."""

    layer_idx: int
    expert_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    admissions: int
    hits: int
    output: torch.Tensor  # aliases scratch.y; valid until the next call


class MoELayerScratch:
    """Every intermediate one layer's dispatch needs, allocated once.

    Sized from the config and the two layouts rather than from a traced shape,
    so the buffers are right before the first token rather than after it.
    """

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        routed_layout: ExpertTensorLayout,
        shared_layout: ExpertTensorLayout | None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = COMPUTE_DTYPE,
        num_tokens: int = 1,
    ):
        self._device = torch.device(device)
        self.dtype = dtype
        self.num_tokens = num_tokens
        hidden = config.hidden_size
        experts = config.num_routed_experts
        # The intermediate widths come from the layouts, not the config, because
        # the shared module is wider than moe_intermediate_size by exactly the
        # shared-expert count and only the layout knows that.
        routed_inter = routed_layout.spec_for("w1").logical_shape[0]
        shared_inter = None if shared_layout is None else shared_layout.spec_for("w1").logical_shape[0]

        def buffer(*shape: int, buffer_dtype: torch.dtype = dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=buffer_dtype, device=self._device)

        # Routing. Logits and the softmax are computed in fp32: a 64-way
        # softmax in BF16 has ~3 decimal digits, and the top-k ordering of
        # near-tied experts decides which weights get staged.
        self.x_fp32 = buffer(num_tokens, hidden, buffer_dtype=torch.float32)
        # The router weight is cast into here when the caller hands over a BF16
        # one. ``Tensor.to(torch.float32)`` would allocate, and it would do so
        # on the hot path once per layer per token -- which is exactly the kind
        # of per-step allocation the rest of this runtime is built to avoid.
        self.router_fp32 = buffer(experts, hidden, buffer_dtype=torch.float32)
        self.logits = buffer(num_tokens, experts, buffer_dtype=torch.float32)
        self.row_max = buffer(num_tokens, 1, buffer_dtype=torch.float32)
        self.exponentials = buffer(num_tokens, experts, buffer_dtype=torch.float32)
        self.denominator = buffer(num_tokens, 1, buffer_dtype=torch.float32)
        self.probabilities = buffer(num_tokens, experts, buffer_dtype=torch.float32)
        self.topk_values = buffer(num_tokens, config.top_k, buffer_dtype=torch.float32)
        self.topk_indices = torch.zeros(num_tokens, config.top_k, dtype=torch.int64, device=self._device)
        self.route_weight = buffer(num_tokens, 1, buffer_dtype=torch.float32)
        self.weighted_routed = buffer(num_tokens, hidden, buffer_dtype=torch.float32)
        self.routed_sum = buffer(num_tokens, hidden, buffer_dtype=torch.float32)

        # Compute.
        self.routed_gate = buffer(num_tokens, routed_inter)
        self.routed_up = buffer(num_tokens, routed_inter)
        self.routed_out = buffer(num_tokens, hidden)
        self.shared_gate = None if shared_inter is None else buffer(num_tokens, shared_inter)
        self.shared_up = None if shared_inter is None else buffer(num_tokens, shared_inter)
        self.shared_out = buffer(num_tokens, hidden)
        self.y = buffer(num_tokens, hidden)
        self.fp8_enabled = routed_layout.spec_for("w1").kind == FP8_KIND
        if self.fp8_enabled:
            if self._device.type != "cuda":
                raise ValueError("native FP8 scratch requires CUDA")
            self.fp8_input = buffer(num_tokens, hidden, buffer_dtype=torch.float8_e4m3fn)
            self.fp8_down_input = buffer(num_tokens, routed_inter, buffer_dtype=torch.float8_e4m3fn)
            self.fp8_input_scale = buffer(1, 1, buffer_dtype=torch.float32)
            self.fp8_down_scale = buffer(1, 1, buffer_dtype=torch.float32)
            self.fp8_work = buffer(num_tokens, hidden, buffer_dtype=torch.float32)
            self.fp8_down_work = buffer(num_tokens, routed_inter, buffer_dtype=torch.float32)
            self.fp8_gate_out = buffer(num_tokens, routed_inter, buffer_dtype=torch.float32)
            self.fp8_up_out = buffer(num_tokens, routed_inter, buffer_dtype=torch.float32)
            self.fp8_down_out = buffer(num_tokens, hidden, buffer_dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return self._device

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every buffer, for a pointer-stability fingerprint."""
        found = [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
        return tuple(found)

    def fingerprint(self) -> tuple[int, ...]:
        return tuple(tensor.data_ptr() for tensor in self.tensors())


def _as_weight(view: torch.Tensor, logical_shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    """Reinterpret a ``uint8`` slot view as the projection it stores.

    The slot is raw bytes by design -- the pool knows sizes, not semantics -- so
    the compute layer is where a byte window becomes a matrix. ``view(dtype)``
    reinterprets the trailing dimension, which is why the byte view is
    ``(rows, cols * itemsize)`` and comes back as ``(rows, cols)``.
    """
    typed = view.view(dtype)
    if tuple(typed.shape) != logical_shape:
        raise ValueError(f"slot view reinterpreted to {tuple(typed.shape)}, layout says {logical_shape}")
    return typed


def swiglu_into(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_buffer: torch.Tensor,
    up_buffer: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``down(silu(gate(x)) * up(x))`` with no allocation.

    The projections are stored ``[out_features, in_features]`` as the checkpoint
    has them, so each GEMM transposes the weight. ``Tensor.t()`` is a view, so
    the transpose costs nothing and ``torch.mm`` handles the strides.
    """
    torch.mm(x, gate_weight.t(), out=gate_buffer)
    torch.mm(x, up_weight.t(), out=up_buffer)
    torch.nn.functional.silu(gate_buffer, inplace=True)
    gate_buffer.mul_(up_buffer)
    return torch.mm(gate_buffer, down_weight.t(), out=out)


def quantize_activation_into(x, output, work, scale):
    """Dynamic tensorwise FP8 activation conversion without a host sync."""
    work.copy_(x)
    torch.abs(work, out=work)
    torch.amax(work, dim=(0, 1), keepdim=True, out=scale)
    scale.div_(FP8_MAX).clamp_(min=torch.finfo(torch.float32).tiny)
    work.copy_(x).div_(scale).clamp_(-FP8_MAX, FP8_MAX)
    output.copy_(work)


def fp8_swiglu_into(x, views, scratch):
    """Native FP8 Tensor Core GEMMs with FP32 accumulation and BF16 SwiGLU."""
    quantize_activation_into(x, scratch.fp8_input, scratch.fp8_work, scratch.fp8_input_scale)

    def gemm(activation, scale, name, output):
        torch.ops.aten._scaled_mm.out(
            activation,
            views[name].view(torch.float8_e4m3fn).t(),
            scale,
            views[name + "_scale"].view(torch.float32),
            out_dtype=torch.float32,
            use_fast_accum=False,
            out=output,
        )

    gemm(scratch.fp8_input, scratch.fp8_input_scale, "w1", scratch.fp8_gate_out)
    gemm(scratch.fp8_input, scratch.fp8_input_scale, "w3", scratch.fp8_up_out)
    scratch.routed_gate.copy_(scratch.fp8_gate_out)
    scratch.routed_up.copy_(scratch.fp8_up_out)
    torch.nn.functional.silu(scratch.routed_gate, inplace=True)
    scratch.routed_gate.mul_(scratch.routed_up)
    quantize_activation_into(scratch.routed_gate, scratch.fp8_down_input, scratch.fp8_down_work, scratch.fp8_down_scale)
    gemm(scratch.fp8_down_input, scratch.fp8_down_scale, "w2", scratch.fp8_down_out)
    scratch.routed_out.copy_(scratch.fp8_down_out)
    return scratch.routed_out


def execute_moe_layer(
    x: torch.Tensor,
    layer_idx: int,
    router_weight: torch.Tensor,
    pool: StaticExpertSlotPool,
    provider: WeightProviderProtocol,
    scratch: MoELayerScratch,
) -> MoELayerResult:
    """Route, stage and compute one MoE layer for one token.

    ``x`` is ``[num_tokens, hidden_size]`` (or ``[1, num_tokens, hidden_size]``,
    squeezed) and ``router_weight`` is the layer's ``mlp.gate.weight``,
    ``[num_routed_experts, hidden_size]``. The returned ``output`` aliases
    ``scratch.y`` and is overwritten by the next call.
    """
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise NotImplementedError(f"one sequence at a time, got a batch of {x.shape[0]}")
        x = x[0]
    if x.dim() != 2:
        raise ValueError(f"x must be [num_tokens, hidden] or [1, num_tokens, hidden], got {tuple(x.shape)}")
    if x.shape[0] != scratch.num_tokens:
        raise NotImplementedError(
            f"the scratchpad is sized for {scratch.num_tokens} token(s) but x holds {x.shape[0]}. A step is one "
            "token's routing (see acquire_for_step); batching needs per-expert gathering this prototype does not do."
        )
    routed_layout = pool.layout

    # --- 1. Route. fp32 softmax, spelled out because torch.softmax has no out=.
    # Both operands are copied into fixed fp32 buffers rather than cast with
    # ``.to()``, which would allocate per call. An fp32 router weight is used
    # as given, so a caller that casts it once at load time pays nothing here.
    scratch.x_fp32.copy_(x)
    if router_weight.dtype == torch.float32:
        router = router_weight
    else:
        scratch.router_fp32.copy_(router_weight)
        router = scratch.router_fp32
    torch.mm(scratch.x_fp32, router.t(), out=scratch.logits)
    torch.amax(scratch.logits, dim=-1, keepdim=True, out=scratch.row_max)
    torch.sub(scratch.logits, scratch.row_max, out=scratch.exponentials)
    torch.exp(scratch.exponentials, out=scratch.exponentials)
    torch.sum(scratch.exponentials, dim=-1, keepdim=True, out=scratch.denominator)
    torch.div(scratch.exponentials, scratch.denominator, out=scratch.probabilities)
    torch.topk(
        scratch.probabilities,
        scratch.topk_indices.shape[1],
        dim=-1,
        out=(scratch.topk_values, scratch.topk_indices),
    )

    # The one host sync: the host cannot stage weights it has not been told to.
    expert_ids = tuple(int(index) for index in scratch.topk_indices[0].tolist())

    # --- 2. Stage. The policy decides residency; the pool applies it.
    before = (pool.stats.loads, pool.stats.hits)
    with pool.step(layer_idx, expert_ids, provider) as reservation:
        admissions = pool.stats.loads - before[0]
        hits = pool.stats.hits - before[1]

        # --- 3. Shared expert, from its reserved pair.
        if pool.shared_layout is not None and layer_idx in pool.shared_layers:
            if not pool.shared_expert_is_resident(layer_idx):
                raise RuntimeError(
                    f"layer {layer_idx}'s shared-expert region was never filled; call "
                    "pool.fill_shared_experts(provider) at warm-up. Computing against it would read zeros, "
                    "which is a wrong answer rather than an error."
                )
            shared = pool.shared_param_views(layer_idx)
            shared_layout = pool.shared_layout
            assert scratch.shared_gate is not None and scratch.shared_up is not None
            swiglu_into(
                x,
                _as_weight(shared["w1"], shared_layout.spec_for("w1").logical_shape, scratch.dtype),
                _as_weight(shared["w3"], shared_layout.spec_for("w3").logical_shape, scratch.dtype),
                _as_weight(shared["w2"], shared_layout.spec_for("w2").logical_shape, scratch.dtype),
                scratch.shared_gate,
                scratch.shared_up,
                scratch.shared_out,
            )
        else:
            scratch.shared_out.zero_()

        scratch.routed_sum.zero_()

        # --- 4. Routed experts, scaled by their router weights.
        for position, slot_id in enumerate(reservation.slot_ids):
            views = pool.param_views(slot_id)
            if scratch.fp8_enabled:
                fp8_swiglu_into(x, views, scratch)
            else:
                swiglu_into(
                    x,
                    _as_weight(views["w1"], routed_layout.spec_for("w1").logical_shape, scratch.dtype),
                    _as_weight(views["w3"], routed_layout.spec_for("w3").logical_shape, scratch.dtype),
                    _as_weight(views["w2"], routed_layout.spec_for("w2").logical_shape, scratch.dtype),
                    scratch.routed_gate,
                    scratch.routed_up,
                    scratch.routed_out,
                )
            # Kept on device: reading the weight to the host would add a sync
            # per expert on top of the one the routing already costs.
            # norm_topk_prob is false for this architecture, so the raw
            # softmax probability is the weight -- no renormalisation.
            scratch.route_weight.copy_(scratch.topk_values[:, position : position + 1])
            # Match the checkpoint: weight and reduce routed outputs in fp32,
            # cast once to BF16, then add the shared expert in BF16.
            scratch.weighted_routed.copy_(scratch.routed_out)
            scratch.weighted_routed.mul_(scratch.route_weight)
            scratch.routed_sum.add_(scratch.weighted_routed)

        scratch.y.copy_(scratch.routed_sum)
        scratch.y.add_(scratch.shared_out)

        return MoELayerResult(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
            slot_ids=reservation.slot_ids,
            admissions=admissions,
            hits=hits,
            output=scratch.y,
        )
