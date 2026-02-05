"""
WW3 §3B: Partial write / corrupted state => conservative reset, never "reset to full risk".
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.allocator import compute_allocation
from tradability.allocation.hazard import EMERGENCY_GROSS


def test_empty_inputs_conservative_no_crash():
    """Empty inputs => zero allocation, no crash, no full risk."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        max_weight_per_strategy=0.4,
    )
    result = compute_allocation([], config)
    assert result.weights == {}
    assert result.amounts == {}
    gross = sum(result.weights.values())
    assert gross <= 1e-9
    assert not math.isnan(gross) and not math.isinf(gross)


def test_corrupted_missing_critical_fields_fail_closed():
    """Inputs with None/NaN critical fields => sanitized to worst-case; no full allocation."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="X")],
        max_weight_per_strategy=0.4,
        min_regime_confidence=0.2,
    )
    # regime_confidence effectively bad → halt; net_edge 0 → no allocation
    corrupted = StrategyInputs(
        strategy_id="X",
        net_edge_bps=float("nan"),
        regime_confidence=float("nan"),
        uncertainty_score=float("nan"),
        capacity_max_aum=float("inf"),
    )
    result = compute_allocation([corrupted], config)
    # Fail-closed: bad regime/edge => zero or minimal
    gross = sum(result.weights.values())
    assert gross <= 0.2 + 1e-9, "Corrupted inputs must not yield full risk"
    assert not any(math.isnan(w) or math.isinf(w) for w in result.weights.values())
    assert not any(math.isnan(a) or math.isinf(a) for a in result.amounts.values())


def test_partial_inputs_conservative_reset():
    """One strategy with valid data, one with all None/NaN => conservative, no spike from bad."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="good"), StrategySpec(name="bad")],
        max_weight_per_strategy=0.4,
        min_regime_confidence=0.2,
    )
    good = StrategyInputs(
        strategy_id="good",
        net_edge_bps=30.0,
        regime_confidence=0.7,
        uncertainty_score=0.3,
        capacity_max_aum=500_000,
    )
    bad = StrategyInputs(
        strategy_id="bad",
        net_edge_bps=float("nan"),
        regime_confidence=None,
        uncertainty_score=None,
        capacity_max_aum=None,
    )
    result = compute_allocation([good, bad], config)
    # Bad is sanitized to halt or zero edge; good gets allocation but we must not have full risk from "reset"
    gross = sum(result.weights.values())
    assert gross <= 1.0 + 1e-9
    assert result.weights.get("bad", 0.0) <= 0.01 or result.shutdown.get("bad", False)
    assert not any(math.isnan(w) or math.isinf(w) for w in result.weights.values())
