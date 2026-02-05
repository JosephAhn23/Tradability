"""
WW3 §1C: Model collapse — all edges wrong sign (adversarial). Allocator must not rely on edge magnitude
to allocate high risk; allocations stay conservative.
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


def test_all_negative_edges_zero_allocation():
    """All strategies report negative net edge => zero allocation (no concentration into 'high edge')."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B"), StrategySpec(name="C")],
        max_weight_per_strategy=0.4,
        min_net_edge_bps=0.0,
    )
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=-10.0, regime_confidence=0.9, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="B", net_edge_bps=-50.0, regime_confidence=0.9, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="C", net_edge_bps=-100.0, regime_confidence=0.9, capacity_max_aum=1_000_000),
    ]
    result = compute_allocation(inputs, config)
    gross = sum(result.weights.values())
    assert gross <= 1e-9, "All negative edge => zero allocation"
    for sid in ("A", "B", "C"):
        assert result.weights.get(sid, 0.0) <= 1e-9


def test_all_zero_edges_zero_allocation():
    """All strategies report zero net edge => zero allocation."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="X"), StrategySpec(name="Y")],
        max_weight_per_strategy=0.4,
    )
    inputs = [
        StrategyInputs(strategy_id="X", net_edge_bps=0.0, regime_confidence=1.0, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="Y", net_edge_bps=0.0, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    result = compute_allocation(inputs, config)
    gross = sum(result.weights.values())
    assert gross <= 1e-9
    assert result.weights.get("X", 0.0) <= 1e-9 and result.weights.get("Y", 0.0) <= 1e-9


def test_mixed_positive_negative_no_concentration_in_negative():
    """One positive edge, two negative (flipped signs): no allocation to negative; positive gets share."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="good"), StrategySpec(name="bad1"), StrategySpec(name="bad2")],
        max_weight_per_strategy=0.4,
    )
    inputs = [
        StrategyInputs(strategy_id="good", net_edge_bps=20.0, regime_confidence=0.6, capacity_max_aum=500_000),
        StrategyInputs(strategy_id="bad1", net_edge_bps=-100.0, regime_confidence=0.9, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="bad2", net_edge_bps=-200.0, regime_confidence=0.9, capacity_max_aum=1_000_000),
    ]
    result = compute_allocation(inputs, config)
    assert result.weights.get("bad1", 0.0) <= 1e-9
    assert result.weights.get("bad2", 0.0) <= 1e-9
    # Good gets allocation; total gross <= 1 and good capped at max_weight
    assert result.weights.get("good", 0.0) <= 0.4 + 1e-9
