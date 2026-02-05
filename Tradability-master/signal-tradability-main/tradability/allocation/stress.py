"""
Worst-case and misspecification stress tests.

Simulate: 2× estimation error, correlation → 1, liquidity shock.
Report how allocations change.
"""

from typing import Dict, List, Optional

import pandas as pd

from .config import AllocationConfig
from .inputs import StrategyInputs
from .allocator import compute_allocation, AllocationResult


def stress_2x_estimation_error(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
) -> AllocationResult:
    """Assume all net_edge_bps are wrong by 2× (halved)."""
    from dataclasses import replace
    stressed = [replace(inp, net_edge_bps=inp.net_edge_bps / 2.0) for inp in inputs]
    return compute_allocation(stressed, config)


def stress_correlation_one(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
) -> AllocationResult:
    """Treat all strategies as one correlation group (combined exposure capped)."""
    stressed_config = AllocationConfig(
        total_capital=config.total_capital,
        strategies=config.strategies,
        rebalance_frequency=config.rebalance_frequency,
        max_weight_per_strategy=config.max_weight_per_strategy,
        min_weight_threshold=config.min_weight_threshold,
        correlation_penalty=3.0,
        uncertainty_penalty=config.uncertainty_penalty,
        regime_penalty=config.regime_penalty,
        max_drawdown_pct=config.max_drawdown_pct,
        max_edge_decay_pct=config.max_edge_decay_pct,
        min_regime_confidence=config.min_regime_confidence,
        min_net_edge_bps=config.min_net_edge_bps,
        divergence_throttle_bps=config.divergence_throttle_bps,
        divergence_shutdown_bps=config.divergence_shutdown_bps,
    )
    from dataclasses import replace
    stressed_inputs = [replace(inp, correlation_group="stress_all") for inp in inputs]
    return compute_allocation(stressed_inputs, stressed_config)


def stress_liquidity_shock(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
) -> AllocationResult:
    """Reduce capacity (capacity_max_aum halved) to simulate liquidity shock."""
    from dataclasses import replace
    stressed = [
        replace(inp, capacity_max_aum=inp.capacity_max_aum / 2.0 if inp.capacity_max_aum else None)
        for inp in inputs
    ]
    return compute_allocation(stressed, config)


def run_stress_tests(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    base_result: AllocationResult,
) -> pd.DataFrame:
    """Run all stress scenarios; return DataFrame with scenario, strategy, weight_base, weight_stress."""
    rows = []
    for scenario_name, stress_fn in [
        ("2x_estimation_error", stress_2x_estimation_error),
        ("correlation_to_one", stress_correlation_one),
        ("liquidity_shock", stress_liquidity_shock),
    ]:
        try:
            res = stress_fn(inputs, config)
        except Exception:
            continue
        for sid in base_result.weights:
            rows.append({
                "scenario": scenario_name,
                "strategy": sid,
                "weight_base": base_result.weights.get(sid, 0),
                "weight_stress": res.weights.get(sid, 0),
                "amount_base": base_result.amounts.get(sid, 0),
                "amount_stress": res.amounts.get(sid, 0),
            })
    return pd.DataFrame(rows)
