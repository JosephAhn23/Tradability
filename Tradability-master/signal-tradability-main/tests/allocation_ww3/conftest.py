"""
WW3 annihilation tests: fixtures, hazard helper, robust vs nominal.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from dataclasses import dataclass
from typing import Dict, List

from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation, AllocationResult
from tradability.allocation.hazard import HazardContext, compute_hazard_level, EMERGENCY_GROSS
from tradability.allocation.robust import apply_robust_mode


@dataclass
class WW3Scenario:
    name: str
    inputs: List[StrategyInputs]
    config: AllocationConfig
    hazard: HazardContext
    expected_max_gross: float
    expect_all_halted: bool = False


def make_inp(
    sid: str,
    net_edge_bps: float = 25.0,
    regime_confidence: float = 0.7,
    uncertainty_score: float = 0.4,
    capacity_max_aum: float = 500_000,
    correlation_group: str = "g1",
) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        uncertainty_score=uncertainty_score,
        capacity_max_aum=capacity_max_aum,
        correlation_group=correlation_group,
    )


@pytest.fixture
def base_config() -> AllocationConfig:
    return AllocationConfig(
        total_capital=1_000_000,
        strategies=[
            StrategySpec(name="A", correlation_group="g1"),
            StrategySpec(name="B", correlation_group="g1"),
            StrategySpec(name="C", correlation_group="g2"),
        ],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
        correlation_penalty=1.5,
        min_regime_confidence=0.2,
        min_net_edge_bps=0.0,
        max_drawdown_pct=0.20,
    )
