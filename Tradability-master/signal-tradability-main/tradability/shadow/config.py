"""
Shadow trading run configuration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ShadowConfig:
    """Config for forward walk-forward shadow execution."""

    tickers: List[str]
    start_date: datetime
    end_date: datetime
    rebalance_frequency: str  # daily, weekly
    fill_rule: str  # next_open

    # Signal
    signal_name: str
    position_sizing: str
    fixed_dollar_per_name: Optional[float]
    max_position_pct: float
    initial_equity: float

    # Cost model (documented assumptions)
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    impact_k: float
    adv_window: int

    # Feasibility halt: stop if net edge bound <= 0 or AUM > capacity
    halt_on_feasibility_violation: bool
    max_aum: Optional[float]
    min_net_edge_bps: Optional[float]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShadowConfig":
        def parse_date(v: Any) -> datetime:
            if isinstance(v, datetime):
                return v
            if isinstance(v, str):
                return datetime.strptime(v[:10], "%Y-%m-%d")
            raise ValueError(f"Cannot parse date: {v}")

        start = d.get("start_date")
        end = d.get("end_date")
        if isinstance(start, str):
            start = parse_date(start)
        if isinstance(end, str):
            end = parse_date(end)
        if start is None:
            start = datetime(2015, 1, 1)
        if end is None:
            end = datetime(2024, 12, 31)

        cost = d.get("cost_model") or {}
        impact = cost.get("impact") or {}
        feasibility = d.get("feasibility_halt") or {}

        return cls(
            tickers=d.get("tickers") or ["SPY"],
            start_date=start,
            end_date=end,
            rebalance_frequency=d.get("rebalance_frequency", "daily"),
            fill_rule=d.get("fill_rule", "next_open"),
            signal_name=d.get("signal_name", "momentum_12_1"),
            position_sizing=d.get("position_sizing", "equal_weight"),
            fixed_dollar_per_name=d.get("fixed_dollar_per_name"),
            max_position_pct=float(d.get("max_position_pct", 0.25)),
            initial_equity=float(d.get("initial_equity", 100_000)),
            fee_bps=float(cost.get("fee_bps", 5)),
            spread_bps=float(cost.get("spread_bps", 10)),
            slippage_bps=float(cost.get("slippage_bps", 5)),
            impact_k=float(impact.get("k", 10)),
            adv_window=int(impact.get("adv_window", cost.get("adv_window", 20))),
            halt_on_feasibility_violation=bool(feasibility.get("enabled", True)),
            max_aum=feasibility.get("max_aum"),
            min_net_edge_bps=feasibility.get("min_net_edge_bps"),
        )


def load_config(path: str) -> ShadowConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ShadowConfig.from_dict(data)
