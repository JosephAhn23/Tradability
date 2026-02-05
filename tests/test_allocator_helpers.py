"""
Shared helpers for allocator war-level tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from typing import List, Optional

from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs


def make_config(
    total_capital: float = 1_000_000,
    strategies: Optional[List[StrategySpec]] = None,
    max_weight_per_strategy: float = 0.4,
    min_weight_threshold: float = 0.02,
    correlation_penalty: float = 1.5,
    min_regime_confidence: float = 0.2,
    min_net_edge_bps: float = 0.0,
    max_drawdown_pct: float = 0.20,
) -> AllocationConfig:
    if strategies is None:
        strategies = [
            StrategySpec(name="A", correlation_group="g1"),
            StrategySpec(name="B", correlation_group="g1"),
            StrategySpec(name="C", correlation_group="g1"),
        ]
    return AllocationConfig(
        total_capital=total_capital,
        strategies=strategies,
        max_weight_per_strategy=max_weight_per_strategy,
        min_weight_threshold=min_weight_threshold,
        correlation_penalty=correlation_penalty,
        min_regime_confidence=min_regime_confidence,
        min_net_edge_bps=min_net_edge_bps,
        max_drawdown_pct=max_drawdown_pct,
    )


def make_input(
    strategy_id: str,
    net_edge_bps: float = 20.0,
    capacity_max_aum: Optional[float] = None,
    regime_confidence: float = 0.7,
    uncertainty_score: float = 0.4,
    current_drawdown_pct: Optional[float] = None,
    correlation_group: Optional[str] = None,
    regime_fragile: bool = False,
    divergence_bps: Optional[float] = None,
) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=strategy_id,
        net_edge_bps=net_edge_bps,
        capacity_max_aum=capacity_max_aum,
        regime_confidence=regime_confidence,
        uncertainty_score=uncertainty_score,
        current_drawdown_pct=current_drawdown_pct,
        correlation_group=correlation_group,
        regime_fragile=regime_fragile,
        recent_realized_return_bps=divergence_bps + 10 if divergence_bps is not None else None,
        recent_expected_return_bps=10.0 if divergence_bps is not None else None,
    )
