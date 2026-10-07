"""Contract tests for the DeepSeek-V4 Flash MoE V5 bindings (torch-free).

These validate what can be proven without a CANN runtime: the aclDataType
codes fixed by the toolkit ABI, the DSV4 execution-mapping profile, the
plan-argument arity/positioning against the transcribed CANN prototypes, and
the host-side cumsum oracle for the device-born GMM groupList. Runtime
behavior (plan/launch against libopapi) is covered by
``benchmarks/camodel_moe_v5_bringup.py`` on the 950PR host.
"""

from __future__ import annotations

import ctypes

import pytest
import torch

from ..hardware.aclnn_binding import (
    ACL_DT_FLOAT8_E4M3FN,
    ACL_DT_FLOAT8_E5M2,
    ACL_DT_FLOAT8_E8M0,
    ACL_DT_FP4X2_E2M1,
)
from ..hardware.v5_ops_moe import (
    DSV4_MOE_PROFILE,
    Dsv4GatingConfig,
    GroupedMatmulV5Config,
    MoeGatingTopKV2Op,
    MoeInitRoutingV4Op,
    GroupedMatmulV5Op,
    cumsum_group_list,
    grouped_matmul_v5_plan_args,
    moe_gating_top_k_v2_plan_args,
    moe_init_routing_v4_plan_args,
)


def test_acl_dtype_codes_match_toolkit_abi():
    """9.2.0 acl_base_rt.h: E5M2=35, E4M3FN=36, UE8M0=37, E2M1=40, E1M2=41."""
    assert ACL_DT_FLOAT8_E5M2 == 35
    assert ACL_DT_FLOAT8_E4M3FN == 36
    assert ACL_DT_FLOAT8_E8M0 == 37
    assert ACL_DT_FP4X2_E2M1 == 40


def test_dsv4_profile_matches_execution_mapping():
    profile = DSV4_MOE_PROFILE
    assert profile.hidden_size == 4096
    assert profile.moe_intermediate_size == 2048
    assert profile.num_layers == 43
    assert profile.num_routed_experts == 256
    assert profile.num_shared_experts == 1
    assert profile.num_experts_per_tok == 6
    assert profile.routed_scaling_factor == 1.5
    assert profile.swiglu_limit == 10.0
    assert profile.num_attention_heads == 64
    assert profile.q_lora_rank == 1024
    assert profile.index_head_dim == 128
    # Routed GEMM: FP4 weights + UE8M0 block-32; dense/MLA: FP8 E4M3.
    assert profile.fp8_dtype_code == ACL_DT_FLOAT8_E4M3FN
    assert profile.fp4_dtype_code == 40
    assert profile.e8m0_dtype_code == ACL_DT_FLOAT8_E8M0
    assert profile.fp_scale_block == 32


def test_grouped_matmul_config_defaults():
    config = GroupedMatmulV5Config()
    assert config.split_item == 3  # single output over all groups
    assert config.group_type == 0  # M-axis grouping (tokens per expert)
    assert config.group_list_type == 0  # groupList = per-expert cumsum
    assert config.act_type == 0  # clamped SwiGLU runs in aclnnSwigluMxQuant


def test_gating_config_defaults():
    config = Dsv4GatingConfig()
    assert config.k == DSV4_MOE_PROFILE.num_experts_per_tok
    assert config.routed_scaling_factor == DSV4_MOE_PROFILE.routed_scaling_factor
    assert config.group_count == 1  # noaux_tc: no group-constrained selection
    assert config.renorm == 0  # weights keep their raw scores
    assert config.out_flag is False


def _arg_slots(argtypes: list[object]) -> list[object]:
    """Drop the two trailing protocol args (workspace size, executor)."""
    assert argtypes[-2:] == [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    return argtypes[:-2]


def test_grouped_matmul_plan_args_match_prototype():
    argtypes = GroupedMatmulV5Op.plan_argtypes
    args = grouped_matmul_v5_plan_args(1, 2, 3, 4, scale_list=5, per_token_scale_list=6)
    assert len(args) == len(_arg_slots(argtypes))
    # groupList is the 9th pointer argument (device cumsum, never host data);
    # tuningConfig sits right after the four scalars.
    assert args[8] == 3
    assert args[16] is None
    assert args[0] == 1 and args[1] == 2 and args[3] == 5 and args[7] == 6


def test_gating_plan_args_match_prototype():
    argtypes = MoeGatingTopKV2Op.plan_argtypes
    args = moe_gating_top_k_v2_plan_args(10, 11, 12)
    assert len(args) == len(_arg_slots(argtypes))
    config = Dsv4GatingConfig()
    assert args[4:11] == (
        config.k,
        config.k_group,
        config.group_count,
        config.group_select_mode,
        config.renorm,
        config.norm_type,
        config.out_flag,
    )
    assert args[11] == pytest.approx(config.routed_scaling_factor)
    assert args[12] == pytest.approx(config.eps)


def test_init_routing_plan_args_match_prototype():
    argtypes = MoeInitRoutingV4Op.plan_argtypes
    args = moe_init_routing_v4_plan_args(20, 21, 22, 23, 24)
    assert len(args) == len(_arg_slots(argtypes))
    assert args[7] == DSV4_MOE_PROFILE.num_routed_experts  # expert_num
    assert args[8] == 0  # drop_pad_mode: dropless
    assert args[9] == 0  # token counts as cumsum -> the GMM groupList
    assert args[10] is True  # counts are actually emitted
    assert args[13] == 0  # gather row indices


def test_cumsum_group_list_reference_contract():
    generator = torch.Generator().manual_seed(704)
    num_tokens, top_k, num_experts = 32, DSV4_MOE_PROFILE.num_experts_per_tok, 256
    topk = torch.randint(0, num_experts, (num_tokens, top_k), generator=generator)
    group_list = cumsum_group_list(topk, num_experts)
    assert group_list.dtype == torch.int64
    assert group_list.shape == (num_experts,)
    assert int(group_list[-1]) == num_tokens * top_k
    assert bool((group_list[1:] >= group_list[:-1]).all())
    counts = torch.bincount(topk.reshape(-1), minlength=num_experts)
    assert bool((group_list == torch.cumsum(counts, 0)).all())


def test_cumsum_group_list_rejects_out_of_range_experts():
    bad = torch.tensor([[0, 256]])
    with pytest.raises(ValueError, match="expert range"):
        cumsum_group_list(bad, 256)
