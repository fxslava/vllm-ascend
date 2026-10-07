"""DeepSeek-V4 Flash MoE stack on aclnn V5: gating, dispatch, FP4 grouped GEMM.

Single-layer chain on Ascend 950PR (V5 interfaces exclusively):

    router logits [T, 256] fp32 --aclnnMoeGatingTopKV2--> expertIdx [T, 6] int32
    expertIdx --aclnnMoeInitRoutingV4--> expandedX [T*6, 4096] fp8 + the device
        expertTokensCountOrCumsum int64 [256] that feeds the GMM groupList
    (expandedX, groupList) --aclnnGroupedMatmulV5--> [T*6, 2048] bf16

Shapes and quant follow the DeepSeek-V4 Flash execution mapping
(:data:`DSV4_MOE_PROFILE`): hidden 4096, moe_intermediate 2048, 256 routed + 1
shared expert, top-6, routed scaling 1.5, swiglu limit 10.0. Routed GEMM
weights are FP4 E2M1 (aclDataType 40) with UE8M0 block-32 scales (37); the
routed GEMM activations are FP8 E4M3 (36) with per-token UE8M0 block-32
scales. The clamped SwiGLU (limit 10.0) runs downstream of GEMM1 through
``aclnnSwigluMxQuant`` (clampLimit), so the grouped GEMM itself keeps
``actType NONE``.

The pure ``*_plan_args`` builders are shared by the torch-backed operator
classes and by the torch-free ctypes bring-up runner
(``benchmarks/camodel_moe_v5_bringup.py``); both append the two trailing
protocol arguments (workspace size, executor) at call time.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from ..hardware.aclnn_binding import (
    ACL_DT_FLOAT32,
    ACL_DT_FLOAT8_E4M3FN,
    ACL_DT_FLOAT8_E8M0,
    ACL_DT_FP4X2_E2M1,
)
from ..hardware.v5_ops import _V5OpBase

# aclnnGroupedMatmulV5GetWorkspaceSize (22 protocol args): 12 tensor(-list)
# pointers, 4 scalars, the tuning-config int array, 3 output lists, then the
# trailing workspace-size/executor pair appended by the two-call protocol.
_GROUPED_MATMUL_V5_PLAN_ARGTYPES: list[object] = (
    [ctypes.c_void_p] * 12
    + [ctypes.c_int64] * 4
    + [ctypes.c_void_p]
    + [ctypes.c_void_p] * 3
    + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
)

# aclnnMoeGatingTopKV2GetWorkspaceSize (18 args): 4 tensors, 6 int64, bool,
# 2 doubles, 3 output tensors, then the trailing pair.
_MOE_GATING_TOP_K_V2_PLAN_ARGTYPES: list[object] = (
    [ctypes.c_void_p] * 4
    + [ctypes.c_int64] * 6
    + [ctypes.c_bool, ctypes.c_double, ctypes.c_double]
    + [ctypes.c_void_p] * 3
    + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
)

# aclnnMoeInitRoutingV4GetWorkspaceSize (21 args): 6 optional-capable input
# tensors, 4 int64 scalars, the emit-counts bool, quant mode int64,
# active-expert-range array, row-index type int64, 5 output tensors, then the
# trailing workspace-size/executor pair appended by the two-call protocol.
_MOE_INIT_ROUTING_V4_PLAN_ARGTYPES: list[object] = (
    [ctypes.c_void_p] * 6
    + [ctypes.c_int64] * 4
    + [ctypes.c_bool]
    + [ctypes.c_int64]
    + [ctypes.c_void_p]
    + [ctypes.c_int64]
    + [ctypes.c_void_p] * 5
    + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
)


@dataclass(frozen=True)
class Dsv4MoeProfile:
    """DeepSeek-V4 Flash execution mapping the 950PR port is sized for."""

    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_layers: int = 43
    num_routed_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    # MLA decode (aclnnFusedInferAttentionScoreV5): 64 heads over the q_lora
    # projection; index_head_dim is the DSV4 indexer head width. FIA itself
    # receives the projected tensors -- q_lora/index dims ride on shapes.
    num_attention_heads: int = 64
    q_lora_rank: int = 1024
    index_head_dim: int = 128
    # Precision contract: FP8 E4M3 (36) dense/MLA, FP4 E2M1 (40) routed expert
    # weights, UE8M0 (37) block scales, block size 32 on the routed GEMM.
    fp8_dtype_code: int = ACL_DT_FLOAT8_E4M3FN
    fp4_dtype_code: int = ACL_DT_FP4X2_E2M1
    e8m0_dtype_code: int = ACL_DT_FLOAT8_E8M0
    fp_scale_block: int = 32


DSV4_MOE_PROFILE = Dsv4MoeProfile()


_MOE_GATING_NORM_TYPE_PRE_NORMALIZED = -1

# aclnnSoftplusGetWorkspaceSize (5 protocol args): self, beta (host aclScalar),
# threshold (host aclScalar), out, then the trailing pair.
_SOFTPLUS_PLAN_ARGTYPES: list[object] = [ctypes.c_void_p] * 3 + [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_void_p),
]

# aclnnSqrtGetWorkspaceSize (4 protocol args): self, out, then the trailing pair.
_SQRT_PLAN_ARGTYPES: list[object] = [ctypes.c_void_p] * 2 + [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_void_p),
]


@dataclass(frozen=True)
class SoftplusConfig:
    """aclnnSoftplus scalars: softplus(x) = ln(1 + exp(beta*x)) below threshold."""

    beta: float = 1.0
    threshold: float = 20.0


@dataclass(frozen=True)
class GroupedMatmulV5Config:
    """aclnnGroupedMatmulV5 scalars for the M-grouped expert GEMM."""

    split_item: int = 3  # single output tensor spanning all groups
    group_type: int = 0  # groups on the M axis (tokens per expert)
    group_list_type: int = 0  # groupList holds per-expert token cumsum
    act_type: int = 0  # NONE: clamped SwiGLU is a separate aclnnSwigluMxQuant
    expected_tokens_per_expert: int = 0  # >0 enables the tiling hint


@dataclass(frozen=True)
class Dsv4GatingConfig:
    """aclnnMoeGatingTopKV2 scalars for noaux_tc routing on pre-normalized scores.

    The DSV4 scoring function (sqrt(softplus(logits))) is NOT expressible in
    the stock operator -- its normType only offers softmax(0)/sigmoid(1) -- so
    the scores arrive pre-normalized from the decomposed Softplus+Sqrt stages
    (see :class:`SqrtSoftplusRouter`). The gating stage performs bias-shifted
    selection (noaux_tc: bias shifts *selection only*; weights keep their raw
    scores, cf. ``routing.score_router.ScoreRouteResolver``), L1-renormalizes
    the selected top-k scores (renorm=1, guarded by eps) and applies the
    routed scaling factor.

    norm_type is the pre-normalized/identity mode: stock docs enumerate only
    softmax(0)/sigmoid(1), so the bypass value must be confirmed against the
    deployed kernel at device bring-up (fallback: L1 renorm outside the op).
    """

    k: int = 6
    k_group: int = 1
    group_count: int = 1  # noaux_tc: selection is not group-constrained
    group_select_mode: int = 0
    renorm: int = 1  # top-k weights renormalized to sum 1.0 before scaling
    norm_type: int = _MOE_GATING_NORM_TYPE_PRE_NORMALIZED
    out_flag: bool = False
    routed_scaling_factor: float = 1.5
    eps: float = 1e-20


def _pointer(value: object) -> object:
    """Normalize handles / raw pointers / None into a ctypes-friendly value."""
    if value is None:
        return None
    pointer = getattr(value, "pointer", None)
    return pointer if pointer is not None else value


def grouped_matmul_v5_plan_args(
    x_list: object,
    weight_list: object,
    group_list: object,
    out_list: object,
    *,
    scale_list: object = None,
    per_token_scale_list: object = None,
    tuning_config: object = None,
    config: GroupedMatmulV5Config = GroupedMatmulV5Config(),
) -> tuple[object, ...]:
    """20 leading arguments of ``aclnnGroupedMatmulV5GetWorkspaceSize``.

    DSV4 routed GEMM (per expert e): out[m_e, 2048] = x[m_e, 4096] fp8 x
    weight[e] fp4 with UE8M0 block-32 scales on both sides. ``group_list`` is
    the device int64 [256] cumsum emitted by ``aclnnMoeInitRoutingV4``.
    """
    null = None
    return (
        _pointer(x_list),
        _pointer(weight_list),
        null,  # bias
        _pointer(scale_list),
        null,  # offset
        null,  # antiquant scale/offset (MX path uses scale, not antiquant)
        null,
        _pointer(per_token_scale_list),
        _pointer(group_list),
        null,  # activation input/quant scale/offset (actType NONE)
        null,
        null,
        config.split_item,
        config.group_type,
        config.group_list_type,
        config.act_type,
        _pointer(tuning_config),
        _pointer(out_list),
        null,  # activation feature out
        null,  # dynamic quant scale out
    )


def moe_gating_top_k_v2_plan_args(
    x: object,
    y_out: object,
    expert_idx_out: object,
    *,
    bias: object = None,
    input_ids: object = None,
    tid2eid: object = None,
    out_out: object = None,
    config: Dsv4GatingConfig = Dsv4GatingConfig(),
) -> tuple[object, ...]:
    """16 leading arguments of ``aclnnMoeGatingTopKV2GetWorkspaceSize``."""
    return (
        _pointer(x),
        _pointer(bias),
        _pointer(input_ids),
        _pointer(tid2eid),
        config.k,
        config.k_group,
        config.group_count,
        config.group_select_mode,
        config.renorm,
        config.norm_type,
        config.out_flag,
        config.routed_scaling_factor,
        config.eps,
        _pointer(y_out),
        _pointer(expert_idx_out),
        _pointer(out_out),
    )


def moe_init_routing_v4_plan_args(
    x: object,
    expert_idx: object,
    expanded_x_out: object,
    expanded_row_idx_out: object,
    expert_tokens_out: object,
    *,
    expert_capacity: int = 0,
    expert_num: int = DSV4_MOE_PROFILE.num_routed_experts,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = True,
    quant_mode: int = 0,
    active_expert_range: object = None,
    row_idx_type: int = 0,
    scale: object = None,
    offset: object = None,
    active_num: object = None,
    topk_weight: object = None,
    expanded_scale_out: object = None,
    expanded_topk_weight_out: object = None,
) -> tuple[object, ...]:
    """19 leading arguments of ``aclnnMoeInitRoutingV4GetWorkspaceSize``.

    Dropless DSV4 dispatch: ``expert_capacity=0``/``drop_pad_mode=0``, cumsum
    token counts (type 0) requested because they *are* the GMM groupList.
    """
    return (
        _pointer(x),
        _pointer(expert_idx),
        _pointer(scale),
        _pointer(offset),
        _pointer(active_num),
        _pointer(topk_weight),
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        _pointer(active_expert_range),
        row_idx_type,
        _pointer(expanded_x_out),
        _pointer(expanded_row_idx_out),
        _pointer(expert_tokens_out),
        _pointer(expanded_scale_out),
        _pointer(expanded_topk_weight_out),
    )


class GroupedMatmulV5Op(_V5OpBase):
    """M-grouped FP4/UE8M0 expert GEMM (256 experts, DSV4 geometry)."""

    plan_symbol = "aclnnGroupedMatmulV5GetWorkspaceSize"
    launch_candidates = ("aclnnGroupedMatmulV5",)
    plan_argtypes = _GROUPED_MATMUL_V5_PLAN_ARGTYPES

    def __init__(self, library, device, workspace_bytes: int = 0, config=GroupedMatmulV5Config()):
        super().__init__(library, device, workspace_bytes)
        self._config = config
        self._tuning = (
            library.create_int_array([config.expected_tokens_per_expert])
            if config.expected_tokens_per_expert > 0
            else None
        )

    def _args(self, x_list, weight_list, group_list, out_list, scale_list, per_token_scale_list):
        return grouped_matmul_v5_plan_args(
            x_list,
            weight_list,
            group_list,
            out_list,
            scale_list=scale_list,
            per_token_scale_list=per_token_scale_list,
            tuning_config=self._tuning,
            config=self._config,
        )

    def plan_static(self, x_list, weight_list, group_list, out_list, scale_list=None, per_token_scale_list=None):
        result = self._plan(*self._args(x_list, weight_list, group_list, out_list, scale_list, per_token_scale_list))
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(self, x_list, weight_list, group_list, out_list, stream_pointer, scale_list=None, per_token_scale_list=None):
        result = self._plan(*self._args(x_list, weight_list, group_list, out_list, scale_list, per_token_scale_list))
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class MoeGatingTopKV2Op(_V5OpBase):
    """noaux_tc router (sqrtsoftplus scores, top-6, routed scaling 1.5)."""

    plan_symbol = "aclnnMoeGatingTopKV2GetWorkspaceSize"
    launch_candidates = ("aclnnMoeGatingTopKV2",)
    plan_argtypes = _MOE_GATING_TOP_K_V2_PLAN_ARGTYPES

    def __init__(self, library, device, workspace_bytes: int = 0, config=Dsv4GatingConfig()):
        super().__init__(library, device, workspace_bytes)
        self._config = config

    def _args(self, x, y_out, expert_idx_out, bias, input_ids, tid2eid, out_out):
        return moe_gating_top_k_v2_plan_args(
            x, y_out, expert_idx_out,
            bias=bias, input_ids=input_ids, tid2eid=tid2eid, out_out=out_out, config=self._config,
        )

    def plan_static(self, x, y_out, expert_idx_out, bias=None, input_ids=None, tid2eid=None, out_out=None):
        result = self._plan(*self._args(x, y_out, expert_idx_out, bias, input_ids, tid2eid, out_out))
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(self, x, y_out, expert_idx_out, stream_pointer, bias=None, input_ids=None, tid2eid=None, out_out=None):
        result = self._plan(*self._args(x, y_out, expert_idx_out, bias, input_ids, tid2eid, out_out))
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class MoeInitRoutingV4Op(_V5OpBase):
    """Dropless dispatch; its device cumsum feeds the GMM groupList directly."""

    plan_symbol = "aclnnMoeInitRoutingV4GetWorkspaceSize"
    launch_candidates = ("aclnnMoeInitRoutingV4",)
    plan_argtypes = _MOE_INIT_ROUTING_V4_PLAN_ARGTYPES

    def __init__(self, library, device, workspace_bytes: int = 0, expert_num: int = DSV4_MOE_PROFILE.num_routed_experts):
        super().__init__(library, device, workspace_bytes)
        self._expert_num = expert_num

    def _args(self, x, expert_idx, expanded_x, expanded_row_idx, expert_tokens, expanded_scale):
        return moe_init_routing_v4_plan_args(
            x, expert_idx, expanded_x, expanded_row_idx, expert_tokens,
            expert_num=self._expert_num, expanded_scale_out=expanded_scale,
        )

    def plan_static(self, x, expert_idx, expanded_x, expanded_row_idx, expert_tokens, expanded_scale):
        result = self._plan(*self._args(x, expert_idx, expanded_x, expanded_row_idx, expert_tokens, expanded_scale))
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(self, x, expert_idx, expanded_x, expanded_row_idx, expert_tokens, expanded_scale, stream_pointer):
        result = self._plan(*self._args(x, expert_idx, expanded_x, expanded_row_idx, expert_tokens, expanded_scale))
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class SoftplusOp(_V5OpBase):
    """aclnnSoftplus: the first stage of the decomposed sqrtsoftplus scoring."""

    plan_symbol = "aclnnSoftplusGetWorkspaceSize"
    launch_candidates = ("aclnnSoftplus",)
    plan_argtypes = _SOFTPLUS_PLAN_ARGTYPES

    def __init__(self, library, device, workspace_bytes: int = 0, config=SoftplusConfig()):
        super().__init__(library, device, workspace_bytes)
        self._config = config
        self._beta = library.create_scalar(config.beta, ACL_DT_FLOAT32)
        self._threshold = library.create_scalar(config.threshold, ACL_DT_FLOAT32)

    def _args(self, x, out):
        return (x.pointer, self._beta.pointer, self._threshold.pointer, out.pointer)

    def plan_static(self, x, out):
        result = self._plan(*self._args(x, out))
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(self, x, out, stream_pointer):
        result = self._plan(*self._args(x, out))
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class SqrtOp(_V5OpBase):
    """aclnnSqrt: the second stage of the decomposed sqrtsoftplus scoring."""

    plan_symbol = "aclnnSqrtGetWorkspaceSize"
    launch_candidates = ("aclnnSqrt",)
    plan_argtypes = _SQRT_PLAN_ARGTYPES

    def plan_static(self, x, out):
        result = self._plan(x.pointer, out.pointer)
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(self, x, out, stream_pointer):
        result = self._plan(x.pointer, out.pointer)
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class SqrtSoftplusRouter:
    """Zero-alloc DSV4 router: aclnnSoftplus -> aclnnSqrt -> aclnnMoeGatingTopKV2.

    The stock gating operator cannot express sqrt(softplus(logits)) (normType
    is softmax/sigmoid only), so the scoring runs as an explicit elementwise
    sequence on the caller's stream and the gating stage receives
    pre-normalized scores (renorm=1, eps=1e-20, routed scaling 1.5). Selection
    bias (noaux_tc) stays inside the gating op via its bias input. The two
    intermediate buffers are allocated once here (AOT); everything the caller
    owns is written in place, and expertIdxOut/yOut feed aclnnMoeInitRoutingV4
    -> aclnnGroupedMatmulV5 on the same stream without host round-trips.
    """

    def __init__(self, library, device, num_tokens: int, num_experts: int, workspace_bytes: int = 0):
        import torch  # lazy: device intermediates need torch tensors

        self._library = library
        self._softplus_buffer = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=device)
        self._scores_buffer = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=device)
        self._softplus = SoftplusOp(library, device, workspace_bytes)
        self._sqrt = SqrtOp(library, device, workspace_bytes)
        self._gating = MoeGatingTopKV2Op(library, device, workspace_bytes)

    def _handles(self, logits, y_out, expert_idx_out, bias):
        lib = self._library
        handles = [lib.create_tensor(t) for t in (logits, self._softplus_buffer, self._scores_buffer, y_out, expert_idx_out)]
        if bias is not None:
            handles.append(lib.create_tensor(bias))
        return handles

    def plan_static(self, logits, y_out, expert_idx_out, bias=None):
        handles = self._handles(logits, y_out, expert_idx_out, bias)
        try:
            for op, x, out in (
                (self._softplus, handles[0], handles[1]),
                (self._sqrt, handles[1], handles[2]),
            ):
                op._reserve_workspace(op._plan(x.pointer, *op._args(x, out)[1:]).workspace_size)
            result = self._gating.plan_static(handles[2], handles[3], handles[4], bias=bias and handles[5])
            return result
        finally:
            for handle in handles:
                self._library.destroy_tensor(handle)

    def execute(self, logits, y_out, expert_idx_out, stream_pointer, bias=None):
        handles = self._handles(logits, y_out, expert_idx_out, bias)
        try:
            self._softplus.execute(handles[0], handles[1], stream_pointer)
            self._sqrt.execute(handles[1], handles[2], stream_pointer)
            self._gating.execute(handles[2], handles[3], handles[4], stream_pointer, bias=bias and handles[5])
        finally:
            for handle in handles:
                self._library.destroy_tensor(handle)


def sqrt_softplus_routing(
    logits,
    *,
    top_k: int = DSV4_MOE_PROFILE.num_experts_per_tok,
    routed_scaling_factor: float = DSV4_MOE_PROFILE.routed_scaling_factor,
    beta: float = 1.0,
    threshold: float = 20.0,
    eps: float = 1e-20,
):
    """CPU reference of the decomposed DSV4 router (torch, fp32 in/out).

    scores = sqrt(softplus(logits)); top-k selection; L1 renorm of the
    selected scores to sum 1.0 (eps-guarded denominator); routed scaling.
    Returns ``(expert_idx int32 [T, k], weights float32 [T, k])`` with
    ``weights.sum(-1) == routed_scaling_factor``.
    """
    import torch  # lazy: CPU reference only (bring-up hosts may lack torch)

    linear = beta * logits.to(torch.float32)
    softplus = torch.where(
        linear > threshold,
        linear / beta,
        (torch.log1p(torch.exp(linear.to(torch.float64))) / beta).to(torch.float32),
    )
    scores = softplus.sqrt()
    top_scores, top_idx = torch.topk(scores, top_k, dim=-1)
    denom = top_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    weights = top_scores / denom * routed_scaling_factor
    return top_idx.to(torch.int32), weights.to(torch.float32)


def cumsum_group_list(topk_indices: "object", num_experts: int) -> "object":
    """Host reference for the device-born groupList (MoeInitRoutingV4 output).

    Returns the int64 [num_experts] inclusive cumsum of per-expert token
    counts: ``group_list[e]`` = tokens routed to experts ``<= e``, and the
    last entry equals ``topk_indices.numel()``. Device bring-up never builds
    this on host -- it is the CPU-side contract oracle.
    """
    import torch  # lazy: CPU contract oracle only (bring-up hosts may lack torch)

    flat = topk_indices.reshape(-1).to(torch.int64)
    if int(flat.min()) < 0 or int(flat.max()) >= num_experts:
        raise ValueError("top-k expert indices out of the expert range")
    counts = torch.bincount(flat, minlength=num_experts)
    return torch.cumsum(counts, dim=0).to(torch.int64)
