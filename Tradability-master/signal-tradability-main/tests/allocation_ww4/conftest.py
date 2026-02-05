"""
WW4 survival tests: fixtures, state machine, consensus, risk tokens.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from typing import Optional

from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation
from tradability.allocation.ww4 import (
    compute_allocation_ww4,
    build_ww4_context,
    WW4Context,
)
from tradability.allocation.ww4_state import (
    SurvivalState,
    WW4StateMachine,
    compute_ww4_state,
)
from tradability.allocation.ww4_consensus import consensus_regime, check_sensor_poisoning


def make_inp(
    sid: str,
    net_edge_bps: float = 25.0,
    regime_confidence: float = 0.7,
    uncertainty_score: float = 0.4,
    capacity_max_aum: float = 500_000,
    turnover: Optional[float] = None,
    recent_realized_return_bps: Optional[float] = None,
    recent_expected_return_bps: Optional[float] = None,
) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        uncertainty_score=uncertainty_score,
        capacity_max_aum=capacity_max_aum,
        turnover=turnover,
        recent_realized_return_bps=recent_realized_return_bps,
        recent_expected_return_bps=recent_expected_return_bps,
    )


@pytest.fixture
def base_config() -> AllocationConfig:
    return AllocationConfig(
        total_capital=1_000_000,
        strategies=[
            StrategySpec(name="A"),
            StrategySpec(name="B"),
            StrategySpec(name="C"),
        ],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
        min_regime_confidence=0.2,
    )
