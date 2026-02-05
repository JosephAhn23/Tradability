"""
WW4 §4A/4B: Metamorphic tests. Permutation invariance; scale invariance in SURVIVAL (don't trust edges).
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
from tradability.allocation.ww4 import compute_allocation_ww4, build_ww4_context


def test_permutation_invariance(base_config):
    """Reordering strategies must not change weights (same strategy_id => same weight)."""
    inputs1 = [
        StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=0.6, capacity_max_aum=400_000),
        StrategyInputs(strategy_id="B", net_edge_bps=25, regime_confidence=0.7, capacity_max_aum=400_000),
        StrategyInputs(strategy_id="C", net_edge_bps=15, regime_confidence=0.5, capacity_max_aum=400_000),
    ]
    inputs2 = [inputs1[1], inputs1[2], inputs1[0]]
    r1 = compute_allocation(inputs1, base_config)
    r2 = compute_allocation(inputs2, base_config)
    for sid in ("A", "B", "C"):
        assert abs(r1.weights.get(sid, 0) - r2.weights.get(sid, 0)) < 1e-9, f"Permutation changed weight for {sid}"


def test_scale_invariance_in_survival(base_config):
    """In SURVIVAL, scaling all edges by 10x must not increase exposure (we don't trust edges)."""
    ctx = build_ww4_context(telemetry_fail=True)
    assert ctx.state_machine.state >= 3  # SURVIVAL
    inputs_low = [
        StrategyInputs(strategy_id="A", net_edge_bps=10, regime_confidence=0.6, capacity_max_aum=500_000),
    ]
    inputs_high = [
        StrategyInputs(strategy_id="A", net_edge_bps=100, regime_confidence=0.6, capacity_max_aum=500_000),
    ]
    res_low = compute_allocation_ww4(inputs_low, base_config, ww4_context=ctx)
    res_high = compute_allocation_ww4(inputs_high, base_config, ww4_context=ctx)
    assert res_high.exposure_after <= res_low.exposure_after + 0.05, "Scaling edges up must not increase exposure in SURVIVAL"
