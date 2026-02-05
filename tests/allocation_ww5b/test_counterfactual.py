# 4B Counterfactual swap: swap good/bad labels, keep constraints -> allocations driven by constraints not labels
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation


def test_allocation_driven_by_constraints_not_labels(base_config):
    # Same numeric inputs, different "labels" (strategy names only)
    inputs_a_good = [StrategyInputs(strategy_id="good", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=500_000),
                    StrategyInputs(strategy_id="bad", net_edge_bps=5, regime_confidence=0.3, capacity_max_aum=100_000)]
    inputs_swapped = [StrategyInputs(strategy_id="bad", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=500_000),
                     StrategyInputs(strategy_id="good", net_edge_bps=5, regime_confidence=0.3, capacity_max_aum=100_000)]
    r1 = compute_allocation(inputs_a_good, base_config)
    r2 = compute_allocation(inputs_swapped, base_config)
    assert r1.weights.get("good", 0) > 0.1
    assert r2.weights.get("bad", 0) > 0.1
    assert r1.weights.get("bad", 0) <= 0.1
    assert r2.weights.get("good", 0) <= 0.1
    assert abs((r1.weights.get("good", 0) + r1.weights.get("bad", 0)) - (r2.weights.get("bad", 0) + r2.weights.get("good", 0))) < 0.01
