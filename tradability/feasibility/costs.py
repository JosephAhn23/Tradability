"""
Cost model: spread, fee, slippage, impact, delay.

All in bps. Turnover in annualized fraction (e.g. 2.0 = 200%).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CostModel:
    """Cost parameters in bps where applicable."""

    fee_bps: float = 5.0
    spread_bps: float = 10.0
    slippage_bps_per_turnover: float = 5.0  # per 100% annual turnover
    delay_bps: float = 2.0
    impact_type: str = "sqrt"
    impact_k: float = 10.0
    adv_window: int = 20

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CostModel":
        cost = d.get("cost_model") or d
        impact = cost.get("impact") or {}
        return cls(
            fee_bps=float(cost.get("fee_bps", 5)),
            spread_bps=float(cost.get("spread_bps", 10)),
            slippage_bps_per_turnover=float(cost.get("slippage_bps_per_turnover", 5)),
            delay_bps=float(cost.get("delay_bps", 2)),
            impact_type=(impact.get("type") or "sqrt"),
            impact_k=float(impact.get("k", 10)),
            adv_window=int(impact.get("adv_window", cost.get("adv_window", 20))),
        )


def compute_total_cost_bps(
    turnover_annual: float,
    cost_model: CostModel,
    impact_bps: float = 0.0,
) -> float:
    """
    Total cost in bps (annualized equivalent for edge comparison).
    turnover_annual: e.g. 2.0 = 200% per year.
    """
    spread_cost = cost_model.spread_bps * turnover_annual
    fee_cost = cost_model.fee_bps * turnover_annual
    slippage_cost = cost_model.slippage_bps_per_turnover * turnover_annual
    delay_cost = cost_model.delay_bps * turnover_annual
    return spread_cost + fee_cost + slippage_cost + delay_cost + impact_bps
