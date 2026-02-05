"""
WW3 §5B: Continuity under small perturbations — weights change smoothly; no cliffs unless crossing kill threshold.
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

# Max allowed weight change for a small epsilon change in one input (no oscillatory cliffs)
EPSILON = 1e-6
MAX_DELTA_WEIGHT = 0.15  # single small input change should not move weight by > 15%


def make_inp(sid: str, net_edge_bps: float = 25.0, regime_confidence: float = 0.6,
             uncertainty_score: float = 0.4, capacity_max_aum: float = 400_000) -> StrategyInputs:
    return StrategyInputs(
        strategy_id=sid,
        net_edge_bps=net_edge_bps,
        regime_confidence=regime_confidence,
        uncertainty_score=uncertainty_score,
        capacity_max_aum=capacity_max_aum,
    )


def test_continuity_epsilon_perturbation_net_edge():
    """Slightly change net_edge_bps by epsilon => weight change bounded (no cliff)."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
        min_weight_threshold=0.02,
    )
    base = [make_inp("A", net_edge_bps=30.0), make_inp("B", net_edge_bps=20.0)]
    perturbed = [make_inp("A", net_edge_bps=30.0 + EPSILON), make_inp("B", net_edge_bps=20.0)]
    r0 = compute_allocation(base, config)
    r1 = compute_allocation(perturbed, config)
    for sid in r0.weights:
        delta = abs(r1.weights.get(sid, 0) - r0.weights.get(sid, 0))
        assert delta <= MAX_DELTA_WEIGHT + 1e-9, f"Cliff: {sid} delta={delta}"


def test_continuity_epsilon_perturbation_uncertainty():
    """Slightly change uncertainty_score by epsilon => weight change bounded."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A")],
        max_weight_per_strategy=0.4,
    )
    base = [make_inp("A", uncertainty_score=0.4)]
    perturbed = [make_inp("A", uncertainty_score=0.4 + EPSILON)]
    r0 = compute_allocation(base, config)
    r1 = compute_allocation(perturbed, config)
    delta = abs(r1.weights.get("A", 0) - r0.weights.get("A", 0))
    assert delta <= MAX_DELTA_WEIGHT + 1e-9, f"Cliff on uncertainty perturbation: delta={delta}"


def test_continuity_no_nan_under_perturbation():
    """Tiny perturbations must not produce NaN/Inf."""
    config = AllocationConfig(
        total_capital=1_000_000,
        strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
        max_weight_per_strategy=0.4,
    )
    for eps in (1e-9, -1e-9, 1e-7):
        inputs = [
            make_inp("A", net_edge_bps=25.0 + eps, regime_confidence=0.6),
            make_inp("B", net_edge_bps=25.0, regime_confidence=0.6),
        ]
        result = compute_allocation(inputs, config)
        for s, w in result.weights.items():
            assert not (w != w), f"NaN weight for {s}"
            assert abs(w) != float("inf"), f"Inf weight for {s}"
