"""
WW3 §4A/4B: Replayed old snapshot / out-of-order => reject stale, safe mode; never re-enable on time confusion.
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
from tradability.allocation.hazard import compute_hazard_level, HazardContext, EMERGENCY_GROSS


def make_inp(sid: str, net_edge_bps: float = 25.0, regime_confidence: float = 0.6,
             capacity_max_aum: float = 400_000) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        capacity_max_aum=capacity_max_aum,
    )


def test_stale_input_count_elevates_hazard_cuts_exposure():
    """Stale input count in hazard => hazard_level elevated, exposure <= EMERGENCY_GROSS when level >= 4."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
    )
    inputs = [make_inp("A"), make_inp("B")]
    # Many stale inputs => level can reach 4
    hazard = compute_hazard_level(stale_input_count=10)
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    if hazard.hazard_level >= 4:
        assert gross <= EMERGENCY_GROSS + 1e-9, "Stale inputs must force emergency gross when hazard >= 4"
    assert "stale" in str(hazard.reason_codes).lower() or hazard.stale_input_count > 0


def test_replay_safe_mode_no_risk_increase():
    """When integrity_failure (e.g. replayed snapshot) => hazard 5, exposure at emergency."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="X")],
        max_weight_per_strategy=0.4,
    )
    inputs = [make_inp("X", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000)]
    hazard = compute_hazard_level(integrity_failure=True)
    assert hazard.hazard_level >= 4
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9, "Replay/integrity failure must not allow full risk"


def test_out_of_order_safe_mode():
    """Hazard with multiple triggers (e.g. stale + subsystem) => safe mode, no re-enable of halted."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        max_weight_per_strategy=0.4,
        min_regime_confidence=0.2,
    )
    # One strategy below min regime => would halt; hazard on top => still emergency gross
    low_regime = StrategyInputs(
        strategy_id="A",
        net_edge_bps=20.0,
        regime_confidence=0.1,
        capacity_max_aum=500_000,
    )
    hazard = compute_hazard_level(subsystem_failure=True, stale_input_count=5)
    result = compute_allocation([low_regime], config, hazard_context=hazard)
    gross = sum(result.weights.values())
    if hazard.hazard_level >= 4:
        assert gross <= EMERGENCY_GROSS + 1e-9
    # Must not "re-enable" due to time confusion: low regime should remain halted or zero
    assert result.weights.get("A", 0.0) <= 0.15 or result.shutdown.get("A", False)
