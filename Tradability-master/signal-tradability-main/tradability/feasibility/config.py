"""
Feasibility run configuration.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FeasibilityConfig:
    """Parsed feasibility config."""

    tickers: List[str]
    start_date: datetime
    end_date: datetime
    rebalance_frequency: str  # daily, weekly
    fill_rule: str  # next_open

    gross_edge_proxy: Dict[str, Any]  # {type: "ic" | "return_prediction", ...}
    cost_model: Dict[str, Any]
    capacity: Dict[str, Any]
    regimes: Dict[str, Any]

    # Optional overrides
    turnover_grid_pct: Optional[List[float]] = None
    aum_grid: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeasibilityConfig":
        def parse_date(v: Any) -> datetime:
            if isinstance(v, datetime):
                return v
            if isinstance(v, str):
                return datetime.strptime(v[:10], "%Y-%m-%d")
            raise ValueError(f"Cannot parse date: {v}")

        tickers = d.get("tickers") or ["SPY"]
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

        cap = d.get("capacity") or {}
        return cls(
            tickers=tickers,
            start_date=start,
            end_date=end,
            rebalance_frequency=d.get("rebalance_frequency", "daily"),
            fill_rule=d.get("fill_rule", "next_open"),
            gross_edge_proxy=d.get("gross_edge_proxy") or {"type": "ic"},
            cost_model=d.get("cost_model") or {},
            capacity=cap,
            regimes=d.get("regimes") or {},
            turnover_grid_pct=d.get("turnover_grid_pct"),
            aum_grid=d.get("aum_grid") or cap.get("aum_grid"),
        )


def load_config(path: str) -> FeasibilityConfig:
    """Load YAML config from path."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return FeasibilityConfig.from_dict(data)
