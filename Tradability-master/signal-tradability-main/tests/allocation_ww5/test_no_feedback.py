# WW5 Scenario 1: No Feedback Universe. Confidence decays, exposure shrinks.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context
from tradability.allocation.ww5_state import NO_FEEDBACK_CONSERVATIVE_PERIODS, NO_FEEDBACK_DORMANT_PERIODS


def test_no_feedback_decay_drifts_toward_conservative(base_config):
    ctx = build_ww5_context(no_feedback_periods=NO_FEEDBACK_CONSERVATIVE_PERIODS)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state in ("CONSERVATIVE", "SURVIVAL", "DORMANT")
    assert res.result.gross_exposure <= 0.3
    assert res.confidence_decay_rate > 0


def test_no_feedback_large_n_dormant(base_config):
    ctx = build_ww5_context(no_feedback_periods=NO_FEEDBACK_DORMANT_PERIODS)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"
    assert res.result.gross_exposure <= 0.01
    assert "DORMANT" in res.reason_for_not_acting


def test_no_feedback_does_not_hold_conviction(base_config):
    ctx_zero = build_ww5_context(no_feedback_periods=0)
    ctx_many = build_ww5_context(no_feedback_periods=5)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.9, capacity_max_aum=1_000_000),
    ]
    r0 = compute_allocation_ww5(inputs, base_config, ww5_context=ctx_zero)
    r5 = compute_allocation_ww5(inputs, base_config, ww5_context=ctx_many)
    assert r5.result.gross_exposure <= r0.result.gross_exposure + 0.05
