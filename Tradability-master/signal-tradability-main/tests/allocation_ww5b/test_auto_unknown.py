# UNKNOWN must be auto-triggered from conditions, not only from flag
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context
from tradability.allocation.ww5_unknown import compute_unknown_conditions, UnknownConditions


def test_unknown_from_telemetry_stale(base_config):
    cond = UnknownConditions(telemetry_stale_periods=5)
    ctx = build_ww5_context()
    ctx.unknown_conditions = cond
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=500_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"
    assert res.bluff_audit and res.bluff_audit.unknown_declared


def test_unknown_from_estimator_disagreement(base_config):
    cond = UnknownConditions(estimator_disagreement=0.4)
    ctx = build_ww5_context()
    ctx.unknown_conditions = cond
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=400_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"


def test_unknown_from_subsystem_failure(base_config):
    cond = UnknownConditions(subsystem_failure=True)
    ctx = build_ww5_context()
    ctx.unknown_conditions = cond
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.9, capacity_max_aum=500_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"


def test_compute_unknown_conditions_api():
    should, reasons = compute_unknown_conditions(telemetry_stale_periods=6)
    assert should is True
    assert "telemetry_stale" in reasons
