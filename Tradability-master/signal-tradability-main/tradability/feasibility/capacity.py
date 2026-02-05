"""
Capacity and impact: sqrt impact, net-edge surface, zero-alpha boundary.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .costs import CostModel, compute_total_cost_bps


def compute_adv_notional(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int,
) -> pd.Series:
    """ADV notional per date: mean over tickers of (close*volume) rolling mean. Index = date."""
    common_idx = prices.index.intersection(volumes.index)
    common_cols = prices.columns.intersection(volumes.columns)
    P = prices.reindex(index=common_idx, columns=common_cols).ffill()
    V = volumes.reindex(index=common_idx, columns=common_cols).fillna(0)
    notional = P * V
    adv = notional.rolling(window, min_periods=1).mean().mean(axis=1)
    return adv


def compute_impact_bps(
    order_notional: float,
    adv_notional: float,
    cost_model: CostModel,
    participation_rate: Optional[float] = None,
) -> float:
    """
    Square-root impact in bps.
    impact_bps ≈ k * sqrt(order_notional / ADV_notional).
    If participation_rate is set, order_notional is interpreted relative to ADV.
    """
    if adv_notional <= 0:
        return 0.0
    if cost_model.impact_type != "sqrt":
        return cost_model.impact_k * (order_notional / adv_notional) * 10_000
    ratio = order_notional / adv_notional
    if participation_rate is not None and participation_rate > 0:
        ratio = min(ratio, participation_rate)
    impact_bps = cost_model.impact_k * np.sqrt(ratio) * 100  # scale to bps
    return float(impact_bps)


def net_edge_surface(
    gross_edge_bps: float,
    cost_model: CostModel,
    aum_grid: List[float],
    turnover_grid: List[float],
    adv_notional_avg: float,
    participation_rate_max: float = 0.01,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """
    Net edge (bps) for each (aum, turnover) pair.
    impact scales with (aum * turnover / periods) per trade notional vs ADV.
    """
    rows = []
    for aum in aum_grid:
        for turnover in turnover_grid:
            # Approximate order notional per period: aum * (turnover / periods_per_year)
            trade_notional_per_period = aum * (turnover / periods_per_year) if periods_per_year else 0
            impact_bps = compute_impact_bps(
                trade_notional_per_period,
                adv_notional_avg,
                cost_model,
                participation_rate=participation_rate_max,
            )
            total_cost_bps = compute_total_cost_bps(turnover, cost_model, impact_bps=impact_bps)
            net_bps = gross_edge_bps - total_cost_bps
            rows.append({"aum": aum, "turnover": turnover, "net_edge_bps": net_bps})
    return pd.DataFrame(rows)


def zero_alpha_boundary(
    surface: pd.DataFrame,
    aum_grid: List[float],
) -> pd.DataFrame:
    """
    For each AUM, find turnover at which net_edge_bps crosses 0 (linear interpolation).
    Returns DataFrame with columns aum, turnover_at_zero.
    """
    rows = []
    for aum in aum_grid:
        sub = surface.loc[surface["aum"] == aum].sort_values("turnover")
        if len(sub) < 2:
            rows.append({"aum": aum, "turnover_at_zero": np.nan})
            continue
        t = sub["turnover"].values
        n = sub["net_edge_bps"].values
        # Crossing where sign changes
        idx = np.where(np.diff(np.sign(n)) != 0)[0]
        if len(idx) == 0:
            if n[-1] >= 0:
                turnover_at_zero = t[-1]
            else:
                turnover_at_zero = t[0]
            rows.append({"aum": aum, "turnover_at_zero": turnover_at_zero})
            continue
        i = idx[0]
        if n[i + 1] != n[i]:
            turnover_at_zero = t[i] - n[i] * (t[i + 1] - t[i]) / (n[i + 1] - n[i])
        else:
            turnover_at_zero = (t[i] + t[i + 1]) / 2
        rows.append({"aum": aum, "turnover_at_zero": turnover_at_zero})
    return pd.DataFrame(rows)
