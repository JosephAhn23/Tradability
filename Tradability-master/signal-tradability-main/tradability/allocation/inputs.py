"""
Strategy metrics and uncertainty inputs for allocation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import pandas as pd

from .config import AllocationConfig, StrategySpec


@dataclass
class StrategyInputs:
    """Per-strategy inputs: feasibility, regime, uncertainty, divergence."""

    strategy_id: str
    net_edge_bps: float
    capacity_max_aum: Optional[float] = None
    regime_fragile: bool = False
    regime_sensitivity_ratio: Optional[float] = None
    regime_confidence: float = 0.5
    uncertainty_score: float = 0.5  # higher = less confidence
    recent_realized_return_bps: Optional[float] = None
    recent_expected_return_bps: Optional[float] = None
    current_drawdown_pct: Optional[float] = None
    turnover: Optional[float] = None
    correlation_group: Optional[str] = None
    zero_alpha_turnover: Optional[float] = None

    @property
    def divergence_bps(self) -> Optional[float]:
        if self.recent_realized_return_bps is None or self.recent_expected_return_bps is None:
            return None
        return abs(self.recent_realized_return_bps - self.recent_expected_return_bps)


def load_inputs(
    config: AllocationConfig,
    feasibility_run_dir: Optional[str] = None,
    regime_fragile_overrides: Optional[Dict[str, bool]] = None,
) -> List[StrategyInputs]:
    """
    Build StrategyInputs from config strategy specs + optional feasibility/shadow outputs.
    """
    overrides = regime_fragile_overrides or {}
    out = []
    for spec in config.strategies:
        name = spec.name
        net_bps = spec.expected_net_edge_bps if spec.expected_net_edge_bps is not None else 0.0
        capacity_aum = spec.max_capacity_aum
        zero_turnover = None
        regime_fragile = overrides.get(name, False)

        if feasibility_run_dir and os.path.isdir(feasibility_run_dir):
            surface_path = os.path.join(feasibility_run_dir, "net_edge_surface.csv")
            boundary_path = os.path.join(feasibility_run_dir, "zero_alpha_boundary.csv")
            if os.path.isfile(surface_path):
                df = pd.read_csv(surface_path)
                if not df.empty:
                    low = df.loc[df["turnover"] <= 0.5]
                    if not low.empty:
                        net_bps = float(low["net_edge_bps"].median())
            if os.path.isfile(boundary_path):
                b = pd.read_csv(boundary_path)
                if not b.empty and "turnover_at_zero" in b.columns:
                    zero_turnover = float(b["turnover_at_zero"].median())

        out.append(StrategyInputs(
            strategy_id=name,
            net_edge_bps=net_bps,
            capacity_max_aum=capacity_aum,
            regime_fragile=regime_fragile,
            regime_confidence=spec.regime_confidence,
            uncertainty_score=spec.uncertainty_score,
            recent_realized_return_bps=None,
            recent_expected_return_bps=spec.expected_net_edge_bps,
            current_drawdown_pct=spec.recent_drawdown_pct,
            turnover=spec.turnover,
            correlation_group=spec.correlation_group,
            zero_alpha_turnover=zero_turnover,
        ))
    return out


def load_inputs_from_feasibility(
    strategies: List[str],
    feasibility_run_dir: Optional[str] = None,
    regime_fragile_overrides: Optional[Dict[str, bool]] = None,
) -> List[StrategyInputs]:
    """Backward compat: build inputs from list of strategy names."""
    specs = [StrategySpec(name=s) for s in strategies]
    cfg = AllocationConfig(total_capital=1e6, strategies=specs)
    return load_inputs(cfg, feasibility_run_dir=feasibility_run_dir, regime_fragile_overrides=regime_fragile_overrides)
