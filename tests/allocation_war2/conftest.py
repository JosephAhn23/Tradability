"""
WW2-level allocator tests: fixtures and conservative baseline allocator.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional

from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation, AllocationResult


@dataclass
class BaselineResult:
    weights: Dict[str, float]
    amounts: Dict[str, float]
    shutdown: Dict[str, bool]


def baseline_allocator(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
) -> BaselineResult:
    """
    Dumb conservative baseline: equal weight across non-halted strategies, strict caps.
    Halt if net_edge <= 0 or regime_confidence < min. No fancy uncertainty.
    """
    halted = set()
    for inp in inputs:
        if inp.net_edge_bps <= config.min_net_edge_bps:
            halted.add(inp.strategy_id)
        if inp.regime_confidence < config.min_regime_confidence:
            halted.add(inp.strategy_id)
        if inp.current_drawdown_pct is not None and inp.current_drawdown_pct >= config.max_drawdown_pct:
            halted.add(inp.strategy_id)

    active = [inp for inp in inputs if inp.strategy_id not in halted]
    if not active:
        return BaselineResult(
            weights={inp.strategy_id: 0.0 for inp in inputs},
            amounts={inp.strategy_id: 0.0 for inp in inputs},
            shutdown={inp.strategy_id: inp.strategy_id in halted for inp in inputs},
        )

    n = len(active)
    raw_w = min(1.0 / n, config.max_weight_per_strategy)
    weights = {}
    amounts = {}
    for inp in inputs:
        if inp.strategy_id in halted:
            weights[inp.strategy_id] = 0.0
            amounts[inp.strategy_id] = 0.0
        else:
            weights[inp.strategy_id] = raw_w
            cap = config.total_capital * raw_w
            if inp.capacity_max_aum is not None:
                cap = min(cap, inp.capacity_max_aum)
            amounts[inp.strategy_id] = cap

    total_a = sum(amounts.values())
    if total_a > 0 and abs(total_a - config.total_capital) > 1:
        scale = config.total_capital / total_a
        for s in amounts:
            amounts[s] *= scale
        for s in weights:
            weights[s] = amounts[s] / config.total_capital

    return BaselineResult(
        weights=weights,
        amounts=amounts,
        shutdown={inp.strategy_id: inp.strategy_id in halted for inp in inputs},
    )


@pytest.fixture
def base_config() -> AllocationConfig:
    return AllocationConfig(
        total_capital=1_000_000,
        strategies=[
            StrategySpec(name="A", correlation_group="g1"),
            StrategySpec(name="B", correlation_group="g1"),
            StrategySpec(name="C", correlation_group="g2"),
        ],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
        correlation_penalty=1.5,
        min_regime_confidence=0.2,
        min_net_edge_bps=0.0,
        max_drawdown_pct=0.20,
    )


def make_inp(
    sid: str,
    net_edge_bps: float = 25.0,
    regime_confidence: float = 0.7,
    uncertainty_score: float = 0.4,
    capacity_max_aum: Optional[float] = None,
    current_drawdown_pct: Optional[float] = None,
    correlation_group: Optional[str] = "g1",
    regime_fragile: bool = False,
) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        uncertainty_score=uncertainty_score,
        capacity_max_aum=capacity_max_aum,
        current_drawdown_pct=current_drawdown_pct,
        correlation_group=correlation_group,
        regime_fragile=regime_fragile,
    )
