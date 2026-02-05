"""
Transaction costs - the thing that kills most strategies.

Started with Almgren-Chriss but found it overfit to assumptions.
Simpler model: commission + spread. Market impact added separately.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_turnover(positions: pd.Series) -> pd.Series:
    """Absolute position changes. First position counts as entry."""
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0] if len(positions) > 0 else 0
    return turnover


def compute_annual_turnover(positions: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized turnover. 4x means you flip the whole book 4 times/year."""
    if len(positions) == 0:
        return 0.0
    
    turnover = compute_turnover(positions)
    return turnover.mean() * periods_per_year


def compute_fixed_commission_cost(positions: pd.Series, 
                                  commission_per_trade: float = 0.005,
                                  periods_per_year: int = 252) -> pd.Series:
    """
    Fixed cost per trade. Using 0.5% as conservative default.
    IB is ~$0.005/share but we're modeling % of notional.
    """
    turnover = compute_turnover(positions)
    return turnover * commission_per_trade


def compute_bid_ask_spread_cost(positions: pd.Series,
                                half_spread: float = 0.001,
                                prices: Optional[pd.Series] = None,
                                periods_per_year: int = 252) -> pd.Series:
    """
    Cross the spread on every trade. Half-spread = cost to execute.
    SPY is ~1.5bps, small caps can be 50bps+. Using 10bps default.
    
    prices param unused - was planning dynamic spread but kept it simple.
    """
    turnover = compute_turnover(positions)
    return turnover * half_spread


def compute_total_explicit_costs(positions: pd.Series,
                                 commission_per_trade: float = 0.005,
                                 half_spread: float = 0.001,
                                 prices: Optional[pd.Series] = None,
                                 periods_per_year: int = 252) -> pd.Series:
    """Commission + spread. Impact modeled elsewhere."""
    commission = compute_fixed_commission_cost(positions, commission_per_trade, periods_per_year)
    spread = compute_bid_ask_spread_cost(positions, half_spread, prices, periods_per_year)
    return commission + spread


def compute_net_returns_from_positions(gross_returns: pd.Series,
                                       positions: pd.Series,
                                       commission_per_trade: float = 0.005,
                                       half_spread: float = 0.001,
                                       prices: Optional[pd.Series] = None,
                                       periods_per_year: int = 252) -> pd.Series:
    """
    This is where strategies go to die.
    Gross Sharpe 1.2 -> Net Sharpe 0.3 is common with 4x turnover.
    """
    aligned_positions = positions.reindex(gross_returns.index, method='ffill').fillna(0)
    costs = compute_total_explicit_costs(aligned_positions, commission_per_trade, half_spread, prices, periods_per_year)
    return gross_returns - costs

