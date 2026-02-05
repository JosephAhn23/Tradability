"""
WW4 §4C & Multi-source: If one estimator says good regime and another says bad => act like bad.
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
from tradability.allocation.ww4_consensus import consensus_regime


def test_consensus_disagreement_uses_worst_case():
    """Regime estimates [0.9, 0.2] => disagree => value_used = min = 0.2."""
    r = consensus_regime([0.9, 0.2])
    assert r.agree is False
    assert r.value_used == 0.2


def test_consensus_disagree_raises_hazard(base_config):
    """When consensus_disagree=True, state should be at least SURVIVAL, exposure low."""
    ctx = build_ww4_context(consensus_disagree=True, regime_estimates=[0.9, 0.2])
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.9, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    assert res.consensus_status == "disagree"
    assert res.ww4_state in ("SURVIVAL", "LOCKDOWN", "DANGER")
    assert res.exposure_after <= 0.15
