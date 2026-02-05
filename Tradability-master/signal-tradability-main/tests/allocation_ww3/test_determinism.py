"""
WW3: Same inputs + same seed => bitwise-identical outputs.
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


def test_determinism_same_inputs_same_outputs():
    """Run allocator twice on same inputs => identical weights and amounts."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=0.7, capacity_max_aum=300_000),
        StrategyInputs(strategy_id="B", net_edge_bps=18, regime_confidence=0.6, capacity_max_aum=300_000),
    ]
    r1 = compute_allocation(inputs, config)
    r2 = compute_allocation(inputs, config)
    for s in r1.weights:
        assert r1.weights[s] == r2.weights[s]
        assert r1.amounts[s] == r2.amounts[s]
    assert r1.shutdown == r2.shutdown
