import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_radiation_noise_dormant(base_config):
    ctx = build_ww5_context(radiation_noise=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=200, regime_confidence=1.0, capacity_max_aum=1_000_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"
    assert res.result.gross_exposure <= 0.01
    assert res.optionality_score >= 0.99


def test_uncertainty_extreme_dormant(base_config):
    ctx = build_ww5_context(uncertainty_extreme=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=100, regime_confidence=0.9, capacity_max_aum=1_000_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state == "DORMANT"
    assert res.result.gross_exposure <= 0.01
