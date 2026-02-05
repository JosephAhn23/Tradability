# WW5 Scenario 3: One-way door trap. Reject large upside that is irreversible + fragile.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context
from tradability.allocation.ww5_irreversibility import classify_allocation_action, Reversibility


def test_one_way_door_classified():
    inp = StrategyInputs(strategy_id="X", net_edge_bps=100, regime_confidence=0.8, uncertainty_score=0.7)
    cl = classify_allocation_action(inp, proposed_weight_delta=0.3, uncertainty_score=0.7, liquidity_ratio=0.1)
    assert cl.reversibility == Reversibility.IRREVERSIBLE
    assert cl.is_one_way_door is True


def test_irreversible_blocked_under_uncertainty(base_config):
    ctx = build_ww5_context(block_irreversible=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=90, regime_confidence=0.7, uncertainty_score=0.65, capacity_max_aum=10_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx, weights_before={})
    assert res.irreversible_actions_blocked >= 0
    assert res.assumptions_rejected is not None


def test_reversible_not_blocked(base_config):
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=0.6, uncertainty_score=0.3, capacity_max_aum=800_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=None)
    assert res.survival_state == "NORMAL"
    assert res.result.gross_exposure > 0 or res.optionality_score >= 0
