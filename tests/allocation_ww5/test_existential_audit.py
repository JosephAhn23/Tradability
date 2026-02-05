# WW5 Output: existential audit fields must exist.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_audit_fields_present(base_config):
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.6, capacity_max_aum=400_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=None)
    assert hasattr(res, "survival_state")
    assert res.survival_state in ("NORMAL", "CONSERVATIVE", "SURVIVAL", "DORMANT")
    assert hasattr(res, "irreversible_actions_blocked")
    assert hasattr(res, "assumptions_required")
    assert hasattr(res, "assumptions_rejected")
    assert hasattr(res, "optionality_score")
    assert hasattr(res, "confidence_decay_rate")
    assert hasattr(res, "reason_for_not_acting")


def test_dormant_reason_for_not_acting(base_config):
    ctx = build_ww5_context(radiation_noise=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=500_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.reason_for_not_acting != ""
    assert "DORMANT" in res.reason_for_not_acting or "optionality" in res.reason_for_not_acting.lower()
