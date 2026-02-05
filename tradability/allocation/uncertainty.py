"""
Model confidence and estimation error handling.

Reduce allocation when:
- parameter instability / uncertainty_score is high
- backtest vs forward divergence is large
- feasibility bound is close to zero
"""

from typing import List

from .config import AllocationConfig
from .inputs import StrategyInputs


def uncertainty_penalty_factor(
    inp: StrategyInputs,
    config: AllocationConfig,
) -> float:
    """
    Multiplier >= 1.0. Higher uncertainty → larger penalty → lower effective weight.
    Used as divisor: weight /= penalty.
    """
    penalty = 1.0
    # Uncertainty score (config: higher = less confidence)
    penalty *= 1.0 + (inp.uncertainty_score * (config.uncertainty_penalty - 1.0))
    # Feasibility bound close to zero: more penalty
    if inp.net_edge_bps is not None and inp.net_edge_bps <= 0:
        penalty *= 2.0
    elif inp.net_edge_bps is not None and inp.net_edge_bps < 10:  # very low edge
        penalty *= 1.0 + (10 - inp.net_edge_bps) / 50.0
    # Divergence (realized vs expected)
    if inp.divergence_bps is not None:
        if inp.divergence_bps >= config.divergence_shutdown_bps:
            penalty *= 10.0
        elif inp.divergence_bps >= config.divergence_throttle_bps:
            penalty *= 1.5
    return max(penalty, 1.0)


def feasible_capacity_share(
    inp: StrategyInputs,
    total_capital: float,
) -> float:
    """
    Share of feasible capacity (0-1) for this strategy.
    If capacity_max_aum is set, share = min(1, capacity_max_aum / total_capital).
    Otherwise 1.0.
    """
    if inp.capacity_max_aum is None or inp.capacity_max_aum <= 0 or total_capital <= 0:
        return 1.0
    return min(1.0, inp.capacity_max_aum / total_capital)
