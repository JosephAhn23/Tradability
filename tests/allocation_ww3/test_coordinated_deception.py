"""
WW3 §2B: Coordinated deception — many strategies collude, appear uncorrelated but share hidden group.
Worst-case correlation mode => group exposure collapses.
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


def make_inp(sid: str, net_edge_bps: float = 30.0, regime_confidence: float = 0.7,
             capacity_max_aum: float = 300_000, correlation_group: str = "hidden") -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        capacity_max_aum=capacity_max_aum,
        correlation_group=correlation_group,
    )


def test_coordinated_deception_group_cap_binds():
    """10 strategies in same correlation_group => per-strategy cap binds; no single strategy dominates."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name=f"S{i}", correlation_group="hidden") for i in range(10)],
        max_weight_per_strategy=0.4,
        correlation_penalty=1.5,
        min_weight_threshold=0.01,
    )
    inputs = [make_inp(f"S{i}", correlation_group="hidden") for i in range(10)]
    result = compute_allocation(inputs, config)
    # Per-strategy cap: no single strategy can dominate (raw group is scaled, then normalized)
    for i in range(10):
        assert result.weights.get(f"S{i}", 0) <= 0.4 + 1e-6, f"S{i} must not exceed max_weight"
    # Gross is bounded
    gross = sum(result.weights.values())
    assert gross <= 1.0 + 1e-6


def test_coordinated_deception_under_stress_collapses():
    """Same 10 strategies; under correlation_meltdown => gross <= EMERGENCY_GROSS."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name=f"S{i}", correlation_group="collapse") for i in range(10)],
        max_weight_per_strategy=0.4,
        correlation_penalty=1.5,
    )
    inputs = [make_inp(f"S{i}", correlation_group="collapse") for i in range(10)]
    hazard = compute_hazard_level(correlation_meltdown=True)
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9, "Under meltdown, group exposure must collapse"
    assert result.hazard_level >= 4
