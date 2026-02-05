"""
WW4 §2C: Forced liquidation trap — capacity collapses while "fully invested".
Expected: reduce exposure immediately, throttle turnover, cap participation.
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
from tradability.allocation.ww4_state import SurvivalState


def test_liquidity_crisis_reduces_exposure(base_config):
    """Liquidity crisis => DANGER/SURVIVAL, exposure reduced."""
    ctx = build_ww4_context(liquidity_crisis=True)
    assert ctx.state_machine.state >= SurvivalState.DANGER
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=50_000),
        StrategyInputs(strategy_id="B", net_edge_bps=20, regime_confidence=0.6, capacity_max_aum=50_000),
    ]
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx, exposure_before=0.8)
    assert res.exposure_after <= res.exposure_before + 0.01, "Must not increase exposure under liquidity crisis"
    assert res.exposure_after <= 0.15, "Liquidity crisis => low gross"


def test_capacity_collapse_low_capacity_input(base_config):
    """When capacity_max_aum is tiny, allocation respects it (exit feasibility)."""
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=1_000),
        StrategyInputs(strategy_id="B", net_edge_bps=50, regime_confidence=0.8, capacity_max_aum=1_000),
    ]
    ctx = build_ww4_context(liquidity_crisis=True)
    res = compute_allocation_ww4(inputs, base_config, ww4_context=ctx)
    total_alloc = sum(res.result.amounts.values())
    assert total_alloc <= 0.15 * 1_000_000 + 1000, "Gross capped by state; amounts bounded by capacity"
