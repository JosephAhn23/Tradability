"""
WW4 §3A: In hazard >= DANGER, allocator must not produce worse survival than all-cash.
Simplified: gross exposure must be <= emergency level (no full risk).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww4 import compute_allocation_ww4, build_ww4_context
from tradability.allocation.ww4_state import SurvivalState


def test_danger_state_low_exposure(base_config):
    """In DANGER, exposure must be at or below emergency gross (~5%)."""
    ctx = build_ww4_context(correlation_crisis=True)
    assert ctx.state_machine.state >= SurvivalState.DANGER
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.exposure_after <= 0.15, "DANGER must not allow full risk"


def test_survival_state_near_zero(base_config):
    """In SURVIVAL, exposure near zero (better than or equal to all-cash)."""
    ctx = build_ww4_context(telemetry_fail=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=200, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.exposure_after <= 0.1, "SURVIVAL must be near zero"
    assert res.why_not_taking_risk != ""
