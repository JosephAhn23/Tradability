# 2A Unverified narrative -> ignore story, no allocation change from text. 2B Authority -> no change unless data changes.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_narrative_only_no_extra_evidence(base_config):
    ctx = build_ww5_context(narrative_only=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.9, capacity_max_aum=500_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.bluff_audit is not None
    assert res.bluff_audit.evidence_added == [] or "narrative" in str(res.bluff_audit.assumptions_unverified).lower()


def test_authority_endorsement_only_no_evidence_added(base_config):
    ctx = build_ww5_context(authority_endorsement_only=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=400_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.bluff_audit is not None
    assert res.bluff_audit.evidence_added == []
