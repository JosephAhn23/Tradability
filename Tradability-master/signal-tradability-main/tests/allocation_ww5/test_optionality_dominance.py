# WW5 Optionality dominance: allocator must not reduce optionality more than cash in extreme uncertainty.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context
from tradability.allocation.ww5_confidence_decay import optionality_score


def test_cash_baseline_optionality_one():
    opt = optionality_score({}, 0.0)
    assert opt >= 0.99


def test_dormant_optionality_at_least_cash(base_config):
    ctx = build_ww5_context(radiation_noise=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=100, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"
    assert res.optionality_score >= 0.99


def test_extreme_uncertainty_exposure_near_zero(base_config):
    ctx = build_ww5_context(uncertainty_extreme=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.result.gross_exposure <= 0.02
    assert res.optionality_score >= 0.9
