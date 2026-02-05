"""
WW3: Trojan strategy ("I'm perfect") => must NOT dominate; caps bind.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation


def test_trojan_strategy_cannot_dominate():
    """One strategy reports massive edge, zero uncertainty, 1.0 regime_confidence => still capped."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="trojan"), StrategySpec(name="normal")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    trojan = StrategyInputs(
        strategy_id="trojan",
        net_edge_bps=500,
        regime_confidence=1.0,
        uncertainty_score=0.0,
        capacity_max_aum=1_000_000,
        regime_fragile=False,
    )
    normal = StrategyInputs(
        strategy_id="normal",
        net_edge_bps=20,
        regime_confidence=0.6,
        uncertainty_score=0.5,
        capacity_max_aum=500_000,
    )
    result = compute_allocation([trojan, normal], config)
    assert result.weights["trojan"] <= config.max_weight_per_strategy + 1e-9
    assert result.weights["trojan"] <= 0.5  # must not get majority


def test_trojan_with_feasibility_near_zero_halts():
    """If feasibility bound <= 0 for trojan, it must be halted."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="trojan")],
        max_weight_per_strategy=0.4,
        min_net_edge_bps=0.0,
    )
    trojan = StrategyInputs(
        strategy_id="trojan",
        net_edge_bps=0.0,
        regime_confidence=1.0,
        uncertainty_score=0.0,
    )
    result = compute_allocation([trojan], config)
    assert result.weights["trojan"] == 0.0
    assert result.shutdown["trojan"]
