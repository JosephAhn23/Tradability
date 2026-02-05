# WW5 Single point of truth attack: one module says all clear, others silent. Distrust -> conservative.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_single_point_of_truth_survival(base_config):
    ctx = build_ww5_context(single_point_of_truth=True)
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=80, regime_confidence=1.0, capacity_max_aum=1_000_000),
    ]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.survival_state in ("SURVIVAL", "DORMANT", "CONSERVATIVE")
    assert res.result.gross_exposure <= 0.15
