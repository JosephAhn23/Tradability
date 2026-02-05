"""
WW3: Correlation lock + liquidity zero + gap risk => exposure collapses to EMERGENCY_GROSS.
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
from tradability.allocation.hazard import compute_hazard_level, EMERGENCY_GROSS


def make_inp(sid: str, net_edge_bps: float = 20.0, regime_confidence: float = 0.6,
             capacity_max_aum: float = 400_000, correlation_group: str = "g1") -> StrategyInputs:
    return StrategyInputs(strategy_id=sid, net_edge_bps=net_edge_bps, regime_confidence=regime_confidence,
                         capacity_max_aum=capacity_max_aum, correlation_group=correlation_group)


def test_correlation_lock_liquidity_zero_exposure_collapses():
    """All strategies one bucket, capacity near zero, hazard_level >= 4 => gross <= EMERGENCY_GROSS."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B"), StrategySpec(name="C")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
        correlation_penalty=1.5,
    )
    inputs = [
        make_inp("A", net_edge_bps=15, capacity_max_aum=10_000, correlation_group="collapse"),
        make_inp("B", net_edge_bps=15, capacity_max_aum=10_000, correlation_group="collapse"),
        make_inp("C", net_edge_bps=15, capacity_max_aum=10_000, correlation_group="collapse"),
    ]
    hazard = compute_hazard_level(correlation_meltdown=True, liquidity_shock=True)
    assert hazard.hazard_level >= 4
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9, "Exposure must collapse under catastrophe"
    assert result.hazard_level >= 4
    assert "GLOBAL CATASTROPHE" in list(result.reasons.values())[0] or gross == 0


def test_emergency_gross_cap_never_exceeded():
    """With hazard_level=5, gross must never exceed EMERGENCY_GROSS."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="X")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [make_inp("X", net_edge_bps=50, regime_confidence=0.9, capacity_max_aum=1_000_000)]
    hazard = compute_hazard_level(telemetry_blackout=True)
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9
