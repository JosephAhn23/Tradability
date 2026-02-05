"""
Global hazard system: when uncertainty is extreme, exposure must go to zero.

Triggers: telemetry blackout, subsystem failure, correlation meltdown,
liquidity shock, integrity failure, stale inputs.
hazard_level >= 4 -> EMERGENCY_GROSS (near 0).
"""

from dataclasses import dataclass, field
from typing import List, Optional

# When hazard_level >= 4, gross exposure must not exceed this (e.g. 5% or 0)
EMERGENCY_GROSS = 0.05
MAX_GROSS = 1.0


@dataclass
class HazardContext:
    """
    Global hazard state. Fail-closed: missing critical data => assume worst.
    """

    hazard_level: int  # 0..5
    reason_codes: List[str] = field(default_factory=list)
    telemetry_blackout: bool = False
    subsystem_failure: bool = False
    correlation_meltdown: bool = False
    liquidity_shock: bool = False
    integrity_failure: bool = False
    stale_input_count: int = 0
    # For audit
    exposure_before: float = 0.0
    exposure_after: float = 0.0
    number_halted: int = 0
    correlation_assumption: str = "normal"  # normal | inflated | worst_case
    telemetry_integrity: str = "ok"  # ok | fail
    subsystem_failures_count: int = 0

    @property
    def is_emergency(self) -> bool:
        return self.hazard_level >= 4

    @property
    def max_gross_allowed(self) -> float:
        return EMERGENCY_GROSS if self.is_emergency else MAX_GROSS


def compute_hazard_level(
    telemetry_blackout: bool = False,
    subsystem_failure: bool = False,
    correlation_meltdown: bool = False,
    liquidity_shock: bool = False,
    integrity_failure: bool = False,
    stale_input_count: int = 0,
) -> HazardContext:
    """
    Compute hazard level 0..5 from triggers.
    Level 4+ => emergency gross (near 0).
    """
    reasons = []
    level = 0
    if telemetry_blackout:
        level = max(level, 5)
        reasons.append("telemetry_blackout")
    if integrity_failure:
        level = max(level, 5)
        reasons.append("integrity_failure")
    if subsystem_failure:
        level = max(level, 4)
        reasons.append("subsystem_failure")
    if correlation_meltdown:
        level = max(level, 4)
        reasons.append("correlation_meltdown")
    if liquidity_shock:
        level = max(level, 4)
        reasons.append("liquidity_shock")
    if stale_input_count > 0:
        level = max(level, min(4, 2 + stale_input_count // 2))
        reasons.append(f"stale_inputs_{stale_input_count}")

    return HazardContext(
        hazard_level=level,
        reason_codes=reasons,
        telemetry_blackout=telemetry_blackout,
        subsystem_failure=subsystem_failure,
        correlation_meltdown=correlation_meltdown,
        liquidity_shock=liquidity_shock,
        integrity_failure=integrity_failure,
        stale_input_count=stale_input_count,
        telemetry_integrity="fail" if telemetry_blackout else "ok",
        subsystem_failures_count=1 if subsystem_failure else 0,
        correlation_assumption="worst_case" if correlation_meltdown else "inflated" if level >= 3 else "normal",
    )
