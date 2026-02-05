"""
Slippage Module

Models execution costs:
- Volatility-scaled price impact
- Volume-scaled execution cost
Simple linear models to avoid overfitting.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_volatility_slippage(positions: pd.Series,
                                volatility: pd.Series,
                                impact_coefficient: float = 0.1,
                                periods_per_year: int = 252) -> pd.Series:
    """
    Compute slippage based on volatility-scaled price impact.
    
    Higher volatility → higher price impact when trading.
    
    Args:
        positions: Series of position values
        volatility: Series of realized volatility (annualized)
        impact_coefficient: Linear coefficient for impact scaling (default 0.1)
        periods_per_year: Number of periods per year
    
    Returns:
        Series of volatility-scaled slippage costs per period
    """
    # Align volatility with positions
    volatility_aligned = volatility.reindex(positions.index, method='ffill')
    
    # Convert annualized volatility to per-period
    if periods_per_year > 0:
        period_volatility = volatility_aligned / np.sqrt(periods_per_year)
    else:
        period_volatility = volatility_aligned
    
    # Compute turnover
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0] if len(positions) > 0 else 0
    
    # Price impact = coefficient * volatility * turnover
    # Higher volatility and higher turnover → higher impact
    slippage = impact_coefficient * period_volatility * turnover
    
    return slippage


def compute_volume_slippage(positions: pd.Series,
                           volumes: pd.Series,
                           prices: Optional[pd.Series] = None,
                           impact_coefficient: float = 0.0001,
                           base_volume: Optional[float] = None) -> pd.Series:
    """
    Compute slippage based on volume-scaled execution cost.
    
    Lower volume → higher execution cost (less liquidity).
    
    Args:
        positions: Series of position values
        volumes: Series of trading volumes
        prices: Optional price series (for dollar volume calculation)
        impact_coefficient: Linear coefficient for impact scaling
        base_volume: Base volume for normalization (default: median volume)
    
    Returns:
        Series of volume-scaled slippage costs per period
    """
    # Align volumes with positions
    volumes_aligned = volumes.reindex(positions.index, method='ffill')
    
    # Normalize volumes (inverse: lower volume = higher cost)
    if base_volume is None:
        base_volume = volumes_aligned.median()
    
    if base_volume > 0:
        # Normalized volume impact: higher volume = lower impact
        volume_factor = base_volume / volumes_aligned
        volume_factor = volume_factor.clip(lower=0.1, upper=10.0)  # Cap extreme values
    else:
        volume_factor = pd.Series(1.0, index=positions.index)
    
    # Compute turnover
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0] if len(positions) > 0 else 0
    
    # Execution cost = coefficient * volume_factor * turnover
    # Lower volume → higher volume_factor → higher cost
    slippage = impact_coefficient * volume_factor * turnover
    
    return slippage


def compute_total_slippage(positions: pd.Series,
                          volatility: Optional[pd.Series] = None,
                          volumes: Optional[pd.Series] = None,
                          prices: Optional[pd.Series] = None,
                          vol_impact_coefficient: float = 0.1,
                          vol_impact_coefficient2: float = 0.0001,
                          periods_per_year: int = 252,
                          base_volume: Optional[float] = None) -> pd.Series:
    """
    Compute total slippage from volatility and volume components.
    
    Args:
        positions: Series of position values
        volatility: Optional volatility series
        volumes: Optional volume series
        prices: Optional price series
        vol_impact_coefficient: Coefficient for volatility-scaled impact
        vol_impact_coefficient2: Coefficient for volume-scaled impact
        periods_per_year: Number of periods per year
        base_volume: Base volume for normalization
    
    Returns:
        Series of total slippage costs per period
    """
    slippage_components = []
    
    # Volatility-based slippage
    if volatility is not None:
        vol_slippage = compute_volatility_slippage(
            positions, volatility, vol_impact_coefficient, periods_per_year
        )
        slippage_components.append(vol_slippage)
    
    # Volume-based slippage
    if volumes is not None:
        vol_slippage2 = compute_volume_slippage(
            positions, volumes, prices, vol_impact_coefficient2, base_volume
        )
        slippage_components.append(vol_slippage2)
    
    if len(slippage_components) == 0:
        # No slippage data available
        return pd.Series(0.0, index=positions.index)
    
    # Total slippage is sum of components
    total_slippage = pd.concat(slippage_components, axis=1).sum(axis=1)
    
    return total_slippage


def compute_net_returns_with_slippage(gross_returns: pd.Series,
                                      positions: pd.Series,
                                      volatility: Optional[pd.Series] = None,
                                      volumes: Optional[pd.Series] = None,
                                      prices: Optional[pd.Series] = None,
                                      vol_impact_coefficient: float = 0.1,
                                      vol_impact_coefficient2: float = 0.0001,
                                      periods_per_year: int = 252,
                                      base_volume: Optional[float] = None) -> pd.Series:
    """
    Compute net returns after slippage costs.
    
    Args:
        gross_returns: Gross strategy returns (before costs)
        positions: Series of position values (aligned with returns)
        volatility: Optional volatility series
        volumes: Optional volume series
        prices: Optional price series
        vol_impact_coefficient: Coefficient for volatility-scaled impact
        vol_impact_coefficient2: Coefficient for volume-scaled impact
        periods_per_year: Number of periods per year
        base_volume: Base volume for normalization
    
    Returns:
        Series of net returns (after slippage)
    """
    # Align positions with returns
    aligned_positions = positions.reindex(gross_returns.index, method='ffill').fillna(0)
    
    # Compute slippage
    slippage = compute_total_slippage(
        aligned_positions, volatility, volumes, prices,
        vol_impact_coefficient, vol_impact_coefficient2,
        periods_per_year, base_volume
    )
    
    # Align slippage with returns index
    slippage_aligned = slippage.reindex(gross_returns.index, method='ffill').fillna(0)
    
    # Net returns = gross returns - slippage
    net_returns = gross_returns - slippage_aligned
    
    return net_returns

