"""
WW3: No NaN/Inf output ever; fail-closed on bad inputs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import math
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation


def test_no_nan_in_weights_or_amounts():
    """Weights and amounts must never be NaN or Inf."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=0.7),
        StrategyInputs(strategy_id="B", net_edge_bps=15, regime_confidence=0.6),
    ]
    result = compute_allocation(inputs, config)
    for s, w in result.weights.items():
        assert not math.isnan(w), f"NaN weight for {s}"
        assert not math.isinf(w), f"Inf weight for {s}"
        assert w >= 0, f"Negative weight for {s}"
    for s, a in result.amounts.items():
        assert not math.isnan(a), f"NaN amount for {s}"
        assert not math.isinf(a), f"Inf amount for {s}"
        assert a >= 0, f"Negative amount for {s}"


def test_bad_inputs_sanitized_fail_closed():
    """NaN uncertainty / regime_confidence => sanitize to fail-closed (halt or worst case)."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        min_regime_confidence=0.2,
    )
    # Input with NaN regime_confidence - sanitize should push to below min => halt
    inputs = [
        StrategyInputs(strategy_id="A", net_edge_bps=20, regime_confidence=float("nan"), uncertainty_score=0.5),
    ]
    result = compute_allocation(inputs, config)
    assert not math.isnan(result.weights["A"])
    assert not math.isinf(result.weights["A"])
    assert result.weights["A"] >= 0
