"""
Throttles, caps, and shutdown rules. Explicit and logged.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import AllocationConfig
from .inputs import StrategyInputs


@dataclass
class ThrottleRecord:
    """Reason and magnitude for a throttle (reduce weight)."""

    strategy_id: str
    reason: str
    magnitude: float  # e.g. 0.5 = half weight


@dataclass
class ShutdownRecord:
    """Reason for shutdown (zero allocation). Must be logged."""

    strategy_id: str
    reason: str


@dataclass
class PolicyDecision:
    """Per-strategy: weight before normalize, throttle, shutdown, throttle_records, shutdown_reason."""

    strategy_id: str
    raw_weight: float
    throttle: bool
    throttle_magnitude: float
    shutdown: bool
    shutdown_reason: str = ""
    throttle_reasons: List[ThrottleRecord] = field(default_factory=list)


def apply_shutdown_rules(
    inp: StrategyInputs,
    config: AllocationConfig,
) -> Optional[ShutdownRecord]:
    """
    A strategy is halted if ANY shutdown condition holds.
    Returns ShutdownRecord if shutdown, else None.
    """
    if inp.net_edge_bps < config.min_net_edge_bps:
        return ShutdownRecord(inp.strategy_id, "feasibility_bound_below_zero")
    if inp.regime_confidence < config.min_regime_confidence:
        return ShutdownRecord(inp.strategy_id, "regime_confidence_below_minimum")
    if inp.current_drawdown_pct is not None and inp.current_drawdown_pct >= config.max_drawdown_pct:
        return ShutdownRecord(inp.strategy_id, "drawdown_exceeds_hard_limit")
    if inp.divergence_bps is not None and inp.divergence_bps >= config.divergence_shutdown_bps:
        return ShutdownRecord(inp.strategy_id, "divergence_shutdown")
    return None


def apply_throttles(
    inp: StrategyInputs,
    config: AllocationConfig,
) -> tuple:
    """
    Returns (throttle_magnitude, list of ThrottleRecord).
    magnitude = 1.0 means no reduction; 0.5 = half weight.
    """
    records = []
    mag = 1.0
    if inp.divergence_bps is not None and config.divergence_throttle_bps <= inp.divergence_bps < config.divergence_shutdown_bps:
        records.append(ThrottleRecord(inp.strategy_id, "divergence_throttle", 0.5))
        mag *= 0.5
    if inp.regime_fragile:
        records.append(ThrottleRecord(inp.strategy_id, "regime_fragile", config.regime_penalty))
        mag *= config.regime_penalty
    if inp.regime_confidence < 0.5 and inp.regime_confidence >= config.min_regime_confidence:
        records.append(ThrottleRecord(inp.strategy_id, "low_regime_confidence", 0.7))
        mag *= 0.7
    if inp.current_drawdown_pct is not None and inp.current_drawdown_pct >= config.max_drawdown_pct * 0.75:
        records.append(ThrottleRecord(inp.strategy_id, "drawdown_near_limit", 0.5))
        mag *= 0.5
    return mag, records
