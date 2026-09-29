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
from .benchmarks.trace_simulator import RouterTraceSimulator, TraceStep
from .core.config import (
    EXPERT_PARAM_NAMES,
    SANITY_GEOMETRY,
    SCALE_PARAM_SUFFIX,
    DeepSeekV4MoEConfig,
)
from .core.layout import (
    FP4_BLOCK_SIZE,
    FP4_ELEMS_PER_BYTE,
    SLOT_REGION_ALIGN_BYTES,
    ExpertTensorLayout,
    ExpertTensorSpec,
)
from .core.ledger import SlotExhaustionError
from .core.slot_pool import (
    UNRESIDENT_SLOT_ID,
    SlotPoolStats,
    StaticExpertSlotPool,
    StepReservation,
)
from .hardware.pinned_storage import AscendPinnedHostStorage
from .hardware.runtime import CpuRuntime, DeviceRuntime, NpuRuntime, make_runtime
from .protocols.provider import SlotFillProviderProtocol, WeightProviderProtocol
from .protocols.router import RouteResolverProtocol
from .routing.hash_router import HashRouteResolver
from .routing.score_router import ScoreRouteResolver

__all__ = [
    "EXPERT_PARAM_NAMES",
    "FP4_BLOCK_SIZE",
    "FP4_ELEMS_PER_BYTE",
    "SANITY_GEOMETRY",
    "SLOT_REGION_ALIGN_BYTES",
    "SCALE_PARAM_SUFFIX",
    "AscendPinnedHostStorage",
    "BenchConfig",
    "CpuRuntime",
    "DeepSeekV4MoEConfig",
    "DeviceRuntime",
    "ExpertTensorLayout",
    "ExpertTensorSpec",
    "HashRouteResolver",
    "NpuRuntime",
    "OffloadStressHarness",
    "RouteResolverProtocol",
    "RouterTraceSimulator",
    "ScoreRouteResolver",
    "SlotExhaustionError",
    "SlotFillProviderProtocol",
    "SlotPoolStats",
    "StaticExpertSlotPool",
    "StepReservation",
    "TraceStep",
    "UNRESIDENT_SLOT_ID",
    "WeightProviderProtocol",
    "make_runtime",
    "parse_config",
]
