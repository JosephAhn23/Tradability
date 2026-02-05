"""
WW3: No survivors - all strategies bad => zero allocation, clean outputs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import math
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation


def test_all_feasibility_zero_allocates_zero():
    """Feasibility <= 0 for all => allocate 0 to all, no crash, no NaN."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        min_net_edge_bps=0.0,
    )
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=0, regime_confidence=0.5),
        StrategyInputs(strategy_id="B", net_edge_bps=-10, regime_confidence=0.5),
    ]
    result = compute_allocation(inputs, config)
    assert sum(result.weights.values()) == 0.0
    assert result.weights["A"] == 0.0 and result.weights["B"] == 0.0
    assert not math.isnan(result.weights["A"]) and not math.isnan(result.weights["B"])
    assert all(result.shutdown[s] for s in ("A", "B"))


def test_all_regime_confidence_below_min_halts():
    """Regime confidence below minimum for all => all halted."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        min_regime_confidence=0.3,
    )
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=0.1)]
    result = compute_allocation(inputs, config)
    assert result.weights["A"] == 0.0
    assert result.shutdown["A"]
