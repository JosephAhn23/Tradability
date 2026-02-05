"""
Allocation policy: rules that map inputs → weight, throttle, shutdown.

"Capital allocation is the problem of deciding how wrong I'm allowed to be."
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import AllocationConfig
from .inputs import StrategyInputs


@dataclass
class AllocationDecision:
    """Per-strategy decision: weight (0-1), throttle, shutdown, reason."""

    strategy_id: str
    weight: float
    throttle: bool  # True = reduce size (e.g. half weight)
    shutdown: bool  # True = zero allocation
    reason: str = ""


def apply_policy(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    portfolio_drawdown_pct: Optional[float] = None,
) -> List[AllocationDecision]:
    """
    Apply policy rules. No optimization — only rules.
    Higher uncertainty / fragility → less capital.
    """

    decisions = []
    for inp in inputs:
        weight = 0.0
        throttle = False
        shutdown = False
        reasons = []

        # --- Shutdown rules ---
        if inp.net_edge_bps < config.min_net_edge_bps:
            shutdown = True
            reasons.append("net_edge_below_zero_alpha")
        if inp.divergence_bps is not None and inp.divergence_bps >= config.divergence_shutdown_bps:
            shutdown = True
            reasons.append("divergence_shutdown")
        if inp.current_drawdown_pct is not None and inp.current_drawdown_pct >= config.drawdown_freeze_pct:
            throttle = True
            reasons.append("drawdown_freeze")

        if shutdown:
            decisions.append(AllocationDecision(
                strategy_id=inp.strategy_id,
                weight=0.0,
                throttle=False,
                shutdown=True,
                reason="; ".join(reasons),
            ))
            continue

        # --- Throttle (reduce, don't zero) ---
        if inp.divergence_bps is not None and inp.divergence_bps >= config.divergence_throttle_bps:
            throttle = True
            reasons.append("divergence_throttle")
        if inp.regime_fragile:
            throttle = True
            reasons.append("regime_fragile")
        if inp.regime_confidence < config.regime_confidence_floor:
            throttle = True
            reasons.append("low_regime_confidence")

        # --- Raw score: net edge (only positive gets capital) ---
        if inp.net_edge_bps > 0:
            # Simple: weight proportional to net_edge_bps, capped
            raw = min(inp.net_edge_bps / 50.0, 1.0)  # 50 bps → 1.0
            if inp.regime_fragile:
                raw *= 0.5
            if throttle:
                raw *= 0.5
            weight = min(raw, config.max_weight_per_strategy)
        else:
            weight = 0.0
            reasons.append("non_positive_net_edge")

        # Portfolio-wide drawdown: reduce all
        if portfolio_drawdown_pct is not None and portfolio_drawdown_pct >= config.portfolio_drawdown_reduce_pct:
            weight *= 0.5
            throttle = True
            reasons.append("portfolio_drawdown_reduce")

        if weight < config.min_weight:
            weight = 0.0
            shutdown = True
            reasons.append("below_min_weight")

        # Capacity cap (handled in allocator by scaling)
        decisions.append(AllocationDecision(
            strategy_id=inp.strategy_id,
            weight=weight,
            throttle=throttle,
            shutdown=shutdown,
            reason="; ".join(reasons) if reasons else "ok",
        ))

    return decisions
