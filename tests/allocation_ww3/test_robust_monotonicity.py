"""
WW3: Robust mode allocations must never exceed nominal (component-wise and gross).
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
from tradability.allocation.robust import apply_robust_mode


def test_robust_allocations_never_exceed_nominal():
    """Robust mode: robust weight per strategy <= nominal weight; robust gross <= nominal gross."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.7, uncertainty_score=0.3, capacity_max_aum=500_000),
        StrategyInputs(strategy_id="B", net_edge_bps=25, regime_confidence=0.6, uncertainty_score=0.4, capacity_max_aum=400_000),
    ]
    nominal = compute_allocation(inputs, config)
    robust_inputs = apply_robust_mode(inputs, config, k_sigma=3.0, capacity_shock_factor=0.5, inflate_correlation=True)
    robust = compute_allocation(robust_inputs, config)
    gross_nom = sum(nominal.weights.values())
    gross_rob = sum(robust.weights.values())
    assert gross_rob <= gross_nom + 1e-9, "Robust gross must not exceed nominal"
    for sid in nominal.weights:
        assert robust.weights.get(sid, 0) <= nominal.weights.get(sid, 0) + 1e-9, f"Robust weight {sid} must not exceed nominal"


def test_robust_increasing_uncertainty_reduces_exposure():
    """Higher uncertainty in robust mode => lower exposure."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    low_unc = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.7, uncertainty_score=0.2, capacity_max_aum=500_000)]
    high_unc = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.7, uncertainty_score=0.9, capacity_max_aum=500_000)]
    r_low = compute_allocation(apply_robust_mode(low_unc, config), config)
    r_high = compute_allocation(apply_robust_mode(high_unc, config), config)
    assert sum(r_high.weights.values()) <= sum(r_low.weights.values()) + 1e-9
