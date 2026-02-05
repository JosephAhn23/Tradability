"""
Allocation run configuration: strategies, allocation rules, shutdown rules.

Policy-based; no return maximization. Prefer under-allocation to overconfidence.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StrategySpec:
    """Per-strategy spec from config (can be overridden by feasibility/shadow inputs)."""

    name: str
    max_capacity_aum: Optional[float] = None
    expected_net_edge_bps: Optional[float] = None
    uncertainty_score: float = 0.5  # higher = less confidence
    regime_confidence: float = 0.5
    recent_drawdown_pct: Optional[float] = None
    turnover: Optional[float] = None
    correlation_group: Optional[str] = None


@dataclass
class AllocationConfig:
    """
    Capital allocation config: strategies + allocation_rules + shutdown_rules.
    Allocation is a POLICY, not a formula.
    """

    total_capital: float
    strategies: List[StrategySpec]
    rebalance_frequency: str = "daily"

    # allocation_rules
    max_weight_per_strategy: float = 0.4
    min_weight_threshold: float = 0.02
    correlation_penalty: float = 1.5  # multiplier when same group (reduce combined)
    uncertainty_penalty: float = 1.5   # divisor for weight when high uncertainty
    regime_penalty: float = 0.5       # multiply weight when low regime_confidence

    # shutdown_rules (any → halt)
    max_drawdown_pct: float = 0.20
    max_edge_decay_pct: Optional[float] = None  # e.g. 0.5 = edge dropped 50%
    min_regime_confidence: float = 0.2
    min_net_edge_bps: float = 0.0

    # Throttle (reduce, not halt)
    divergence_throttle_bps: float = 50
    divergence_shutdown_bps: float = 150

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AllocationConfig":
        strat_raw = d.get("strategies") or []
        strategies = []
        for s in strat_raw:
            if isinstance(s, str):
                strategies.append(StrategySpec(name=s))
            else:
                strategies.append(StrategySpec(
                    name=s.get("name", "unknown"),
                    max_capacity_aum=float(s["max_capacity_aum"]) if s.get("max_capacity_aum") is not None else None,
                    expected_net_edge_bps=float(s["expected_net_edge_bps"]) if s.get("expected_net_edge_bps") is not None else None,
                    uncertainty_score=float(s.get("uncertainty_score", 0.5)),
                    regime_confidence=float(s.get("regime_confidence", 0.5)),
                    recent_drawdown_pct=float(s["recent_drawdown_pct"]) if s.get("recent_drawdown_pct") is not None else None,
                    turnover=float(s["turnover"]) if s.get("turnover") is not None else None,
                    correlation_group=s.get("correlation_group"),
                ))

        ar = d.get("allocation_rules") or {}
        sr = d.get("shutdown_rules") or {}
        return cls(
            total_capital=float(d.get("total_capital", 1_000_000)),
            strategies=strategies,
            rebalance_frequency=d.get("rebalance_frequency", "daily"),
            max_weight_per_strategy=float(ar.get("max_weight_per_strategy", 0.4)),
            min_weight_threshold=float(ar.get("min_weight_threshold", 0.02)),
            correlation_penalty=float(ar.get("correlation_penalty", 1.5)),
            uncertainty_penalty=float(ar.get("uncertainty_penalty", 1.5)),
            regime_penalty=float(ar.get("regime_penalty", 0.5)),
            max_drawdown_pct=float(sr.get("max_drawdown_pct", 0.20)),
            max_edge_decay_pct=float(sr["max_edge_decay_pct"]) if sr.get("max_edge_decay_pct") is not None else None,
            min_regime_confidence=float(sr.get("min_regime_confidence", 0.2)),
            min_net_edge_bps=float(sr.get("min_net_edge_bps", 0.0)),
            divergence_throttle_bps=float(sr.get("divergence_throttle_bps", 50)),
            divergence_shutdown_bps=float(sr.get("divergence_shutdown_bps", 150)),
        )


def load_config(path: str) -> AllocationConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AllocationConfig.from_dict(data)
