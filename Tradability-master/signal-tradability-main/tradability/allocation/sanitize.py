"""
Fail-closed input sanitization. Bad/missing inputs → conservative or HALT.
"""

import math
from typing import List

from .config import AllocationConfig
from .inputs import StrategyInputs


def _safe_float(v, default: float, for_halt: bool = False) -> float:
    """Return default if v is NaN, Inf, or None. If for_halt, default is worst-case (triggers halt)."""
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def sanitize_inputs(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
) -> List[StrategyInputs]:
    """
    Sanitize inputs: NaN/Inf/None → fail-closed (reduce or halt).
    Missing critical data → treat as worst case (e.g. regime_confidence=0 → halt).
    """
    out = []
    for inp in inputs:
        # Fail-closed: bad net_edge → 0 (will trigger no allocation or halt)
        net_edge_bps = _safe_float(inp.net_edge_bps, 0.0)
        if net_edge_bps < 0:
            net_edge_bps = 0.0

        # Bad regime_confidence → min below threshold so strategy halts
        regime_confidence = _safe_float(inp.regime_confidence, config.min_regime_confidence - 0.01)

        # Bad uncertainty → high (penalize)
        uncertainty_score = _safe_float(inp.uncertainty_score, 1.0)
        uncertainty_score = max(0.0, min(1.0, uncertainty_score))

        capacity_max_aum = None
        if inp.capacity_max_aum is not None:
            c = _safe_float(inp.capacity_max_aum, 0.0)
            if c > 0 and not math.isnan(c):
                capacity_max_aum = c

        current_drawdown_pct = None
        if inp.current_drawdown_pct is not None:
            d = _safe_float(inp.current_drawdown_pct, 0.0)
            if 0 <= d <= 1.0:
                current_drawdown_pct = d
            elif d > 1.0:  # e.g. 150% → treat as max
                current_drawdown_pct = 1.0

        recent_realized_return_bps = inp.recent_realized_return_bps
        recent_expected_return_bps = inp.recent_expected_return_bps
        if recent_realized_return_bps is not None:
            recent_realized_return_bps = _safe_float(recent_realized_return_bps, 0.0)
        if recent_expected_return_bps is not None:
            recent_expected_return_bps = _safe_float(recent_expected_return_bps, 0.0)

        out.append(StrategyInputs(
            strategy_id=inp.strategy_id,
            net_edge_bps=net_edge_bps,
            capacity_max_aum=capacity_max_aum,
            regime_fragile=inp.regime_fragile,
            regime_sensitivity_ratio=inp.regime_sensitivity_ratio,
            regime_confidence=regime_confidence,
            uncertainty_score=uncertainty_score,
            recent_realized_return_bps=recent_realized_return_bps,
            recent_expected_return_bps=recent_expected_return_bps,
            current_drawdown_pct=current_drawdown_pct,
            turnover=inp.turnover,
            correlation_group=inp.correlation_group,
            zero_alpha_turnover=inp.zero_alpha_turnover,
        ))
    return out
