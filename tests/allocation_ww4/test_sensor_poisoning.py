# WW4 2A: Sensor poisoning => SURVIVAL/LOCKDOWN, risk near zero
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
from tradability.allocation.ww4_consensus import check_sensor_poisoning


def test_sensor_poisoning_detection():
    assert check_sensor_poisoning(
        regime_confidence=0.9,
        uncertainty_score=0.1,
        net_edge_bps=100,
        divergence_bps=150,
        feasibility_ratio=0.5,
        turnover=0.5,
    ) is True


def test_sensor_poisoning_implies_survival_or_lockdown(base_config):
    ctx = build_ww4_context(sensor_poisoning=True)
    assert ctx.state_machine.state >= SurvivalState.SURVIVAL
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=200, regime_confidence=1.0, uncertainty_score=0.0, capacity_max_aum=1_000_000),
        StrategyInputs(strategy_id="B", net_edge_bps=200, regime_confidence=1.0, uncertainty_score=0.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.ww4_state in ("SURVIVAL", "LOCKDOWN")
    assert res.exposure_after <= 0.1


def test_lying_metrics_with_consensus_disagree(base_config):
    ctx = build_ww4_context(sensor_poisoning=True, consensus_disagree=True)
    inputs = [StrategyInputs(strategy_id="X", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000)]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.ww4_state in ("SURVIVAL", "LOCKDOWN")
    assert res.exposure_after <= 0.1
