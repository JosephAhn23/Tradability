"""
WW4 §5: Chaos engineering — randomly kill components. Expected: safe shutdown to cash, no crash, audit trail.
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


def test_infra_failure_safe_shutdown(base_config):
    """Infra failure => conservative allocation, no crash."""
    ctx = build_ww4_context(infra_failure=True)
    ctx.module_health = {"feasibility": "fail", "stress": "ok", "regime": "ok"}
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.exposure_after <= 0.2
    assert "infra_failure" in res.reason_codes or res.ww4_state in ("DANGER", "SURVIVAL", "LOCKDOWN")


def test_missing_context_no_crash(base_config):
    """No WW4 context => normal allocation, no crash."""
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.7, capacity_max_aum=400_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=None)
    assert res.ww4_state == "NORMAL"
    assert res.result.weights is not None
    assert res.exposure_after >= 0


def test_deterministic_handling_same_inputs_same_output(base_config):
    """Same inputs + same context => same output (determinism)."""
    ctx = build_ww4_context(liquidity_crisis=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.6, capacity_max_aum=300_000),
    ]
    r1 = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    r2 = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert abs(r1.exposure_after - r2.exposure_after) < 1e-9
    for sid in r1.result.weights:
        assert abs(r1.result.weights[sid] - r2.result.weights[sid]) < 1e-9
