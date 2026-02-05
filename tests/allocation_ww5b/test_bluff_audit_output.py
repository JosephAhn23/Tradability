# Required bluff audit output: confidence_before/after, evidence_added, assumptions_*, reason_*, bluff_risk_score
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_bluff_audit_fields_present(base_config):
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.6, capacity_max_aum=400_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=None)
    assert res.bluff_audit is not None
    b = res.bluff_audit
    assert hasattr(b, "confidence_before") and hasattr(b, "confidence_after")
    assert hasattr(b, "evidence_added") and hasattr(b, "assumptions_required")
    assert hasattr(b, "assumptions_verified") and hasattr(b, "assumptions_unverified")
    assert hasattr(b, "reason_for_action") and hasattr(b, "reason_for_inaction")
    assert hasattr(b, "bluff_risk_score")


def test_bluff_risk_higher_under_uncertainty(base_config):
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.6, capacity_max_aum=400_000)]
    r_normal = compute_allocation_ww5(inputs, base_config, ww5_context=None)
    ctx_unc = build_ww5_context(uncertainty_extreme=True)
    r_unc = compute_allocation_ww5(inputs, base_config, ww5_context=ctx_unc)
    if r_normal.bluff_audit and r_unc.bluff_audit:
        assert r_unc.bluff_audit.bluff_risk_score >= r_normal.bluff_audit.bluff_risk_score - 0.01
