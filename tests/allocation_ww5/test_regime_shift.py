# WW5 Scenario 2: Permanent hidden regime shift. Reduce exposure absent validation.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_hidden_regime_shift_reduces_exposure(base_config):
    ctx = build_ww5_context(hidden_regime_shift=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=60, regime_confidence=0.9, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state in ("CONSERVATIVE", "SURVIVAL", "DORMANT")
    assert res.result.gross_exposure <= 0.3


def test_deceptive_stability_decays(base_config):
    ctx = build_ww5_context(deceptive_stability=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.8, capacity_max_aum=500_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state in ("CONSERVATIVE", "SURVIVAL", "DORMANT")
    assert res.survival_state in ("CONSERVATIVE", "SURVIVAL", "DORMANT")
    assert "deceptive_stability" in res.assumptions_required or "model_drift" in (res.result.reason_codes or [])
