"""
Transaction Costs Module

Models explicit trading costs:
- Fixed commissions
- Bid-ask spread (half-spread per trade)
- Simple linear models to avoid overfitting
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_turnover(positions: pd.Series) -> pd.Series:
    """
    Compute position turnover (absolute change in positions).
    
    Args:
        positions: Series of position values (e.g., -1, 0, 1 for short/flat/long)
    
    Returns:
        Series of turnover values (absolute changes)
    """
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0] if len(positions) > 0 else 0
    return turnover


def compute_annual_turnover(positions: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized turnover rate.
    
    Args:
        positions: Series of position values
        periods_per_year: Number of periods per year (252 for daily)
    
    Returns:
        Annualized turnover rate
    """
    if len(positions) == 0:
        return 0.0
    
    turnover = compute_turnover(positions)
    avg_daily_turnover = turnover.mean()
    annual_turnover = avg_daily_turnover * periods_per_year
    
    return annual_turnover


def compute_fixed_commission_cost(positions: pd.Series, 
                                  commission_per_trade: float = 0.005,
                                  periods_per_year: int = 252) -> pd.Series:
    """
    Compute fixed commission costs based on position changes.
    
    Args:
        positions: Series of position values (-1, 0, 1)
        commission_per_trade: Commission cost per trade (as fraction, e.g., 0.005 = 0.5%)
        periods_per_year: Number of periods per year
    
    Returns:
        Series of commission costs (as fraction of notional per period)
    """
    turnover = compute_turnover(positions)
    
    # Commission is paid on each trade (each position change)
    # Scale by periods per year to get per-period cost
    commission_costs = turnover * commission_per_trade
    
    return commission_costs


def compute_bid_ask_spread_cost(positions: pd.Series,
                                half_spread: float = 0.001,
                                prices: Optional[pd.Series] = None,
                                periods_per_year: int = 252) -> pd.Series:
    """
    Compute bid-ask spread costs (half-spread per trade).
    
    Args:
        positions: Series of position values (-1, 0, 1)
        half_spread: Half of bid-ask spread (as fraction, e.g., 0.001 = 0.1%)
        prices: Optional price series for spread scaling (not used in simple model)
        periods_per_year: Number of periods per year
    
    Returns:
        Series of spread costs (as fraction of notional per period)
    """
    turnover = compute_turnover(positions)
    
    # Half-spread is paid on each trade (crossing the spread)
    # When going long: pay ask price (higher)
    # When going short: receive bid price (lower)
    # Cost = half_spread * 2 = full spread per round trip
    # But we count it as half_spread per trade direction change
    spread_costs = turnover * half_spread
    
    return spread_costs


def compute_total_explicit_costs(positions: pd.Series,
                                 commission_per_trade: float = 0.005,
                                 half_spread: float = 0.001,
                                 prices: Optional[pd.Series] = None,
                                 periods_per_year: int = 252) -> pd.Series:
    """
    Compute total explicit transaction costs.
    
    Args:
        positions: Series of position values
        commission_per_trade: Commission cost per trade (as fraction)
        half_spread: Half of bid-ask spread (as fraction)
        prices: Optional price series (for future extensibility)
        periods_per_year: Number of periods per year
    
    Returns:
        Series of total explicit costs per period
    """
    commission = compute_fixed_commission_cost(
        positions, commission_per_trade, periods_per_year
    )
    spread = compute_bid_ask_spread_cost(
        positions, half_spread, prices, periods_per_year
    )
    
    total_costs = commission + spread
    
    return total_costs


def compute_net_returns_from_positions(gross_returns: pd.Series,
                                       positions: pd.Series,
                                       commission_per_trade: float = 0.005,
                                       half_spread: float = 0.001,
                                       prices: Optional[pd.Series] = None,
                                       periods_per_year: int = 252) -> pd.Series:
    """
    Compute net returns after explicit transaction costs.
    
    Args:
        gross_returns: Gross strategy returns (before costs)
        positions: Series of position values (aligned with returns)
        commission_per_trade: Commission cost per trade
        half_spread: Half of bid-ask spread
        prices: Optional price series
        periods_per_year: Number of periods per year
    
    Returns:
        Series of net returns (after explicit costs)
    """
    # Align positions with returns
    aligned_positions = positions.reindex(gross_returns.index, method='ffill').fillna(0)
    
    # Compute costs
    costs = compute_total_explicit_costs(
        aligned_positions, commission_per_trade, half_spread, 
        prices, periods_per_year
    )
    
    # Net returns = gross returns - costs
    net_returns = gross_returns - costs
    
    return net_returns

