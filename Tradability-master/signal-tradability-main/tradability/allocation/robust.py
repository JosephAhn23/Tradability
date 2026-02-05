"""
Robust control mode: edge_low, capacity_low, inflated correlation.
Robust allocations must never exceed nominal allocations.
"""

from typing import List, Optional

from .config import AllocationConfig
from .inputs import StrategyInputs


def apply_robust_mode(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    k_sigma: float = 3.0,
    edge_stderr_frac: float = 0.3,
    capacity_shock_factor: float = 0.5,
    inflate_correlation: bool = True,
) -> List[StrategyInputs]:
    """
    Transform inputs to worst-case envelope:
    - edge_low = edge - k * stderr (approx as edge * (1 - k*edge_stderr_frac))
    - capacity_low = capacity * shock_factor
    - All strategies forced to same correlation group if inflate_correlation (worst-case)
    Robust allocations must be <= nominal (component-wise or gross).
    """
    out = []
    for inp in inputs:
        # Conservative edge: treat as lower bound (edge - k*uncertainty)
        edge_low = inp.net_edge_bps - k_sigma * (abs(inp.net_edge_bps) * edge_stderr_frac + 5.0)
        edge_low = max(0.0, edge_low)  # do not go negative for "high short edge" nonsense

        cap_low = None
        if inp.capacity_max_aum is not None and inp.capacity_max_aum > 0:
            cap_low = inp.capacity_max_aum * capacity_shock_factor

        # Lower regime confidence in robust mode
        regime_low = max(0.0, inp.regime_confidence - 0.2)
        uncertainty_high = min(1.0, inp.uncertainty_score + 0.2)

        out.append(StrategyInputs(
            strategy_id=inp.strategy_id,
            net_edge_bps=edge_low,
            capacity_max_aum=cap_low,
            regime_fragile=inp.regime_fragile or True,  # assume fragile in robust
            regime_sensitivity_ratio=inp.regime_sensitivity_ratio,
            regime_confidence=regime_low,
            uncertainty_score=uncertainty_high,
            recent_realized_return_bps=inp.recent_realized_return_bps,
            recent_expected_return_bps=inp.recent_expected_return_bps,
            current_drawdown_pct=inp.current_drawdown_pct,
            turnover=inp.turnover,
            correlation_group="robust_single_group" if inflate_correlation else inp.correlation_group,
            zero_alpha_turnover=inp.zero_alpha_turnover,
        ))
    return out
