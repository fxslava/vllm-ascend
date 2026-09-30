"""Backward-compatibility shim: the legacy ledger moved to ``legacy_lru_policy``.

The flat-LRU state machine was migrated behind the pluggable
``EvictionPolicyProtocol`` (see
:mod:`tools.dsv4_moe_runtime.core.legacy_lru_policy`); this module keeps the
historical import surface alive.
"""

from __future__ import annotations

from ..protocols.residency_policy import SlotExhaustionError
from .legacy_lru_policy import _SlotResidencyLedger

__all__ = ["SlotExhaustionError", "_SlotResidencyLedger"]
