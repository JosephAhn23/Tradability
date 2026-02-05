# 5A Last-chance temptation: one action promises survival but single fragile assumption -> refuse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


def test_single_fragile_assumption_blocked(base_config):
    ctx = build_ww5_context(block_irreversible=True)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=150, regime_confidence=0.9, uncertainty_score=0.7, capacity_max_aum=5_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.result.gross_exposure <= 0.2
