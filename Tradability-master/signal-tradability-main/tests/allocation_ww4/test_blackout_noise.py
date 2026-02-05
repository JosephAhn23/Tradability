"""
WW4 §2B: Blackout + noise — 50% fields missing, rest random. Expected: LOCKDOWN, zero weights, audit reason.
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


def test_blackout_forces_lockdown(base_config):
    """Blackout => state LOCKDOWN, zero or near-zero weights."""
    ctx = build_ww4_context(blackout=True)
    assert ctx.state_machine.state == SurvivalState.LOCKDOWN
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.ww4_state == "LOCKDOWN"
    assert res.exposure_after <= 0.01
    assert "blackout" in res.reason_codes
    assert res.why_not_taking_risk != ""


def test_blackout_zero_weights_explicit(base_config):
    """In LOCKDOWN, weights must be zero (or audit explains)."""
    ctx = build_ww4_context(blackout=True)
    inputs = [
        StrategyInputs(strategy_id="X", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    for sid, w in res.result.weights.items():
        assert w <= 0.01, f"LOCKDOWN must yield zero weight for {sid}"
