"""DeepSeek-V4 Flash MoE static runtime: zero-allocation expert slot pool,
pinned-DDR offload transports and NPU memory stress tooling.

Public API (backwards-compatible import surface)::

    from tools.dsv4_moe_runtime import StaticExpertSlotPool, ExpertTensorLayout
"""

from .benchmarks.offload_stress import (
    BenchConfig,
    OffloadStressHarness,
    parse_config,
)
from .benchmarks.synthetic_source import SyntheticExpertSource
from .benchmarks.trace_simulator import RouterTraceSimulator, TraceStep
from .core.config import (
    DSV2_LITE_GEOMETRY,
    EXPERT_PARAM_NAMES,
    SANITY_GEOMETRY,
    SCALE_PARAM_SUFFIX,
    DeepSeekV4MoEConfig,
    MoEGeometry,
)
from .core.generational_policy import GenerationalRadixPolicy
from .core.layout import (
    BF16_NUM_BYTES,
    DENSE_BF16_KIND,
    DENSE_BF16_KINDS,
    E8M0_SCALE_KIND,
    FP4_BLOCK32_KINDS,
    FP4_BLOCK_SIZE,
    FP4_ELEMS_PER_BYTE,
    PACKED_FP4_KIND,
    SLOT_REGION_ALIGN_BYTES,
    ExpertTensorLayout,
    ExpertTensorSpec,
)
from .core.ledger import SlotExhaustionError
from .core.legacy_lru_policy import LegacyLruPolicy
from .core.profiles import (
    DSV2_LITE_PROFILE,
    DSV4_FLASH_PROFILE,
    LAYOUT_PROFILES,
    NAMING_DSV4_FLAT,
    NAMING_HF_DEEPSEEK,
    PROFILE_NAMES,
    SANITY_PROFILE,
    BackboneProfile,
    MoELayoutProfile,
    profile_for,
)
from .core.slot_pool import (
    UNRESIDENT_SLOT_ID,
    SlotPoolStats,
    StaticExpertSlotPool,
    StepReservation,
)
from .hardware.dummy_kernel import (
    DEFAULT_EXPERT_LATENCY_US,
    DIGEST_MASK,
    DummyExpertKernelRunner,
    ExpertKernelRunner,
)
from .hardware.exchange_buffer import TransitExchangeBuffer
from .hardware.exclusive_staging import ExclusiveStagingProvider
from .hardware.lifecycle import RuntimeLifecycleManager, TeardownReport, TeardownStep
from .hardware.pinned_storage import AscendPinnedHostStorage
from .hardware.runtime import CpuRuntime, CudaRuntime, DeviceRuntime, NpuRuntime, make_runtime
from .hardware.safetensors_provider import (
    HF_DEEPSEEK_NAMING,
    ExpertNamingScheme,
    SafetensorsExpertProvider,
    WeightLayoutMismatchError,
)
from .hardware.sharded_safetensors import (
    SafetensorsShardIndex,
    ShardedSafetensorsExpertSource,
)
from .hardware.vram_budget import (
    DEVICE_WORKSPACE_RESERVE_BYTES,
    DeviceMemoryBudget,
    MemoryBudgetError,
    assert_budget_fits,
    plan_budget,
)
from .hardware.weight_loader import StreamingWeightLoader
from .protocols.provider import SlotFillProviderProtocol, WeightByteSource, WeightProviderProtocol
from .protocols.residency_policy import AdmissionDecision, EvictionPolicyProtocol
from .protocols.router import RouteResolverProtocol
from .routing.hash_router import HashRouteResolver
from .routing.score_router import ScoreRouteResolver

__all__ = [
    "AdmissionDecision",
    "AscendPinnedHostStorage",
    "assert_budget_fits",
    "BackboneProfile",
    "BenchConfig",
    "BF16_NUM_BYTES",
    "CpuRuntime",
    "CudaRuntime",
    "DeepSeekV4MoEConfig",
    "DEFAULT_EXPERT_LATENCY_US",
    "DENSE_BF16_KIND",
    "DENSE_BF16_KINDS",
    "DEVICE_WORKSPACE_RESERVE_BYTES",
    "DeviceMemoryBudget",
    "DeviceRuntime",
    "DIGEST_MASK",
    "DSV2_LITE_GEOMETRY",
    "DSV2_LITE_PROFILE",
    "DSV4_FLASH_PROFILE",
    "DummyExpertKernelRunner",
    "E8M0_SCALE_KIND",
    "EvictionPolicyProtocol",
    "ExclusiveStagingProvider",
    "EXPERT_PARAM_NAMES",
    "ExpertKernelRunner",
    "ExpertNamingScheme",
    "ExpertTensorLayout",
    "ExpertTensorSpec",
    "FP4_BLOCK32_KINDS",
    "FP4_BLOCK_SIZE",
    "FP4_ELEMS_PER_BYTE",
    "GenerationalRadixPolicy",
    "HashRouteResolver",
    "HF_DEEPSEEK_NAMING",
    "LAYOUT_PROFILES",
    "LegacyLruPolicy",
    "make_runtime",
    "MemoryBudgetError",
    "MoEGeometry",
    "MoELayoutProfile",
    "NAMING_DSV4_FLAT",
    "NAMING_HF_DEEPSEEK",
    "NpuRuntime",
    "OffloadStressHarness",
    "PACKED_FP4_KIND",
    "parse_config",
    "plan_budget",
    "profile_for",
    "PROFILE_NAMES",
    "RouteResolverProtocol",
    "RouterTraceSimulator",
    "RuntimeLifecycleManager",
    "SafetensorsExpertProvider",
    "SafetensorsShardIndex",
    "SANITY_GEOMETRY",
    "SANITY_PROFILE",
    "SCALE_PARAM_SUFFIX",
    "ScoreRouteResolver",
    "ShardedSafetensorsExpertSource",
    "SLOT_REGION_ALIGN_BYTES",
    "SlotExhaustionError",
    "SlotFillProviderProtocol",
    "SlotPoolStats",
    "StaticExpertSlotPool",
    "StepReservation",
    "StreamingWeightLoader",
    "SyntheticExpertSource",
    "TeardownReport",
    "TeardownStep",
    "TraceStep",
    "TransitExchangeBuffer",
    "UNRESIDENT_SLOT_ID",
    "WeightByteSource",
    "WeightLayoutMismatchError",
    "WeightProviderProtocol",
]
