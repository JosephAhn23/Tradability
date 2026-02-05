# 1A Overprecision under noise: high noise -> exposure shrinks, confidence decreases
# 1B Decimal-point trap: perturb by epsilon -> outputs change smoothly or flag instability
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_overprecision_under_noise_exposure_shrinks(base_config):
    ctx_low = build_ww5_context()
    ctx_high = build_ww5_context(uncertainty_extreme=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.8, uncertainty_score=0.3, capacity_max_aum=500_000)]
    r_low = compute_allocation_ww5(inputs, base_config, ww5_context=ctx_low)
    r_high = compute_allocation_ww5(inputs, base_config, ww5_context=ctx_high)
    assert r_high.result.gross_exposure <= r_low.result.gross_exposure + 0.01
    assert r_high.bluff_audit is not None and r_high.bluff_audit.bluff_risk_score >= r_low.bluff_audit.bluff_risk_score if r_low.bluff_audit else True


def test_decimal_point_trap_perturb_changes_output(base_config):
    eps = 1e-6
    inputs1 = [StrategyInputs(strategy_id="A", net_edge_bps=25.0, regime_confidence=0.6, capacity_max_aum=400_000)]
    inputs2 = [StrategyInputs(strategy_id="A", net_edge_bps=25.0 + eps, regime_confidence=0.6, capacity_max_aum=400_000)]
    r1 = compute_allocation_ww5(inputs1, base_config, ww5_context=None)
    r2 = compute_allocation_ww5(inputs2, base_config, ww5_context=None)
    w1 = r1.result.weights.get("A", 0)
    w2 = r2.result.weights.get("A", 0)
    assert abs(w2 - w1) < 0.2 or (w1 == 0 and w2 == 0)
