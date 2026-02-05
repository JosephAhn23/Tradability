"""
WW4 §1: Non-negotiable invariants. Fail any = instant disqualification.
- Exposure monotonicity: as hazard increases, gross exposure never increases.
- Concentration monotonicity: as hazard increases, max single-strategy weight never increases.
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


def test_exposure_monotonicity_under_hazard(base_config):
    """As state worsens (NORMAL -> CAUTION -> DANGER -> SURVIVAL -> LOCKDOWN), gross must never increase."""
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.8, capacity_max_aum=500_000),
        StrategyInputs(strategy_id="B", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=500_000),
    ]
    prev_gross = 1.0
    for trigger, state_name in [
        (None, "NORMAL"),
        ({"model_drift": True}, "CAUTION"),
        ({"liquidity_crisis": True}, "DANGER"),
        ({"telemetry_fail": True}, "SURVIVAL"),
        ({"blackout": True}, "LOCKDOWN"),
    ]:
        ctx = build_ww4_context(**(trigger or {}))
        res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
        gross = res.exposure_after
        assert gross <= prev_gross + 1e-9, f"Exposure must not increase when moving to {state_name}"
        prev_gross = gross


def test_concentration_monotonicity_under_hazard(base_config):
    """As hazard increases, max single-strategy weight must never increase."""
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.9, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="B", net_edge_bps=10, regime_confidence=0.5, capacity_max_aum=200_000),
    ]
    prev_max_single = 1.0
    for trigger in [None, {"model_drift": True}, {"correlation_crisis": True}, {"blackout": True}]:
        ctx = build_ww4_context(**(trigger or {}))
        res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
        max_w = max(res.result.weights.values()) if res.result.weights else 0.0
        assert max_w <= prev_max_single + 1e-9, "Concentration must not increase as hazard increases"
        prev_max_single = max_w


def test_no_single_point_of_failure(base_config):
    """If module health reports failure, allocation should be more conservative."""
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    ctx_ok = build_ww4_context()
    ctx_fail = build_ww4_context(infra_failure=True)
    res_ok = compute_allocation_ww4(inputs, base_config, ww4_context=ctx_ok)
    res_fail = compute_allocation_ww4(inputs, base_config, ww4_context=ctx_fail)
    assert res_fail.exposure_after <= res_ok.exposure_after + 1e-9, "Infra failure must not increase exposure"
