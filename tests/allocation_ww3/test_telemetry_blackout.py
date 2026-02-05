"""
WW3: Telemetry blackout => global kill switch, exposure to emergency gross.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation
from tradability.allocation.hazard import compute_hazard_level, EMERGENCY_GROSS


def make_inp(sid: str, **kw) -> StrategyInputs:
    return StrategyInputs(strategy_id=sid, net_edge_bps=kw.get("net_edge_bps", 25),
                         regime_confidence=kw.get("regime_confidence", 0.7), **{k: v for k, v in kw.items() if k in ("capacity_max_aum", "correlation_group")})


def test_telemetry_blackout_forces_emergency_gross():
    """Critical inputs missing => hazard_level high => gross <= EMERGENCY_GROSS."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [make_inp("A", net_edge_bps=30), make_inp("B", net_edge_bps=30)]
    hazard = compute_hazard_level(telemetry_blackout=True)
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9
    assert result.telemetry_integrity == "fail"


def test_subsystem_failure_forces_emergency():
    """Subsystem failure => hazard_level >= 4 => cut risk."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [make_inp("A", net_edge_bps=40)]
    hazard = compute_hazard_level(subsystem_failure=True)
    result = compute_allocation(inputs, config, hazard_context=hazard)
    gross = sum(result.weights.values())
    assert gross <= EMERGENCY_GROSS + 1e-9
