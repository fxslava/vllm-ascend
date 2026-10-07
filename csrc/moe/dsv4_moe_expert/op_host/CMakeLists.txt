# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and can be redistributed under the terms
# and conditions of the CANN Open Software License Agreement Version 2.0.
# See the License for details.
# ----------------------------------------------------------------------------

# The kernel carries an AICore config for the dav-c310 regbase core
# (ascend950) and the dav-c220 vector core (ascend910b, i.e. Ascend910B1..B4).
# Any other compute unit has no code path in op_kernel and is skipped here
# rather than failing the kernel compile on an unsupported core generation.
if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT AND NOT "ascend910b" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        dsv4_moe_expert_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Dsv4MoeExpert
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE dsv4_moe_expert ACLNNTYPE aclnn)
endif()
