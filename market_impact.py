"""
Market Impact Models

Implements proper microstructure models with citations:
- Almgren-Chriss (2000): Temporary and permanent impact
- Kyle (1985): Lambda model
- Goyenko et al. (2009): Realized spreads

References:
- Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
  Journal of Risk, 3(2), 5-39.
- Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica, 53(6), 1315-1335.
- Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009). Do liquidity measures measure liquidity?
  Journal of Financial Economics, 92(2), 153-181.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy import stats


def compute_almgren_chriss_impact(
    positions: pd.Series,
    volatility: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    participation_rate: float = 0.01,
    lambda_temp: float = 0.5,
    mu_perm: float = 0.1,
    periods_per_year: int = 252
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute market impact using Almgren-Chriss (2000) model.
    
    Model decomposes impact into:
    1. Temporary impact (reverts): lambda * volatility * sqrt(participation) * sqrt(order_size/ADV)
    2. Permanent impact (doesn't revert): mu * volatility * participation
    
    Args:
        positions: Series of position values (-1, 0, 1)
        volatility: Annualized volatility series
        volumes: Trading volume series (shares)
        prices: Price series
        participation_rate: Participation rate (default 0.01 = 1%)
        lambda_temp: Temporary impact coefficient (default 0.5, from Almgren-Chriss)
        mu_perm: Permanent impact coefficient (default 0.1, from Almgren-Chriss)
        periods_per_year: Number of periods per year
    
    Returns:
        Tuple of (temporary_impact, permanent_impact) as Series
    """
    # Align all series
    aligned_data = pd.DataFrame({
        'position': positions,
        'volatility': volatility,
        'volume': volumes,
        'price': prices
    }).dropna()
    
    if len(aligned_data) == 0:
        return pd.Series(dtype=float, index=positions.index), pd.Series(dtype=float, index=positions.index)
    
    # Compute turnover (position changes)
    turnover = aligned_data['position'].diff().abs()
    turnover.iloc[0] = aligned_data['position'].iloc[0] if len(aligned_data) > 0 else 0
    
    # Convert volatility to per-period
    period_vol = aligned_data['volatility'] / np.sqrt(periods_per_year)
    
    # Compute average daily volume (ADV) - rolling 20-day
    adv = aligned_data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Order size (in shares) - approximate from position change
    # For simplicity, assume position change of 1 = trading 1% of ADV
    order_size = turnover * adv * participation_rate
    
    # Temporary impact: lambda * vol * sqrt(participation) * sqrt(order_size/ADV)
    # This impact reverts (price moves back)
    sqrt_participation = np.sqrt(participation_rate)
    sqrt_order_ratio = np.sqrt(order_size / adv.clip(lower=1e-6))
    temp_impact = lambda_temp * period_vol * sqrt_participation * sqrt_order_ratio
    
    # Permanent impact: mu * vol * participation
    # This impact doesn't revert (permanent price change)
    perm_impact = mu_perm * period_vol * participation_rate
    
    # Align back to original index
    temp_impact_series = pd.Series(temp_impact.values, index=aligned_data.index)
    perm_impact_series = pd.Series(perm_impact.values, index=aligned_data.index)
    
    return temp_impact_series, perm_impact_series


def compute_kyle_lambda(
    volatility: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    participation_rate: float = 0.01,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Compute Kyle (1985) lambda (price impact coefficient).
    
    Lambda = volatility / (2 * sqrt(ADV * participation_rate))
    
    This measures the price impact per unit of order flow.
    
    Args:
        volatility: Annualized volatility series
        volumes: Trading volume series (shares)
        prices: Price series
        participation_rate: Participation rate (default 0.01 = 1%)
        periods_per_year: Number of periods per year
    
    Returns:
        Series of Kyle lambda values
    """
    # Align all series
    aligned_data = pd.DataFrame({
        'volatility': volatility,
        'volume': volumes,
        'price': prices
    }).dropna()
    
    if len(aligned_data) == 0:
        return pd.Series(dtype=float, index=volatility.index)
    
    # Convert volatility to per-period
    period_vol = aligned_data['volatility'] / np.sqrt(periods_per_year)
    
    # Average daily volume (ADV)
    adv = aligned_data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Kyle lambda: vol / (2 * sqrt(ADV * participation))
    sqrt_adv_participation = np.sqrt(adv * participation_rate).clip(lower=1e-6)
    kyle_lambda = period_vol / (2 * sqrt_adv_participation)
    
    return pd.Series(kyle_lambda.values, index=aligned_data.index)


def compute_kyle_impact(
    positions: pd.Series,
    kyle_lambda: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    participation_rate: float = 0.01
) -> pd.Series:
    """
    Compute market impact using Kyle (1985) model.
    
    Impact = lambda * order_size
    
    Args:
        positions: Series of position values
        kyle_lambda: Series of Kyle lambda values (from compute_kyle_lambda)
        volumes: Trading volume series
        prices: Price series
        participation_rate: Participation rate
    
    Returns:
        Series of market impact costs
    """
    # Align all series
    aligned_data = pd.DataFrame({
        'position': positions,
        'lambda': kyle_lambda,
        'volume': volumes,
        'price': prices
    }).dropna()
    
    if len(aligned_data) == 0:
        return pd.Series(dtype=float, index=positions.index)
    
    # Compute turnover
    turnover = aligned_data['position'].diff().abs()
    turnover.iloc[0] = aligned_data['position'].iloc[0] if len(aligned_data) > 0 else 0
    
    # Order size
    adv = aligned_data['volume'].rolling(window=20, min_periods=1).mean()
    order_size = turnover * adv * participation_rate
    
    # Kyle impact: lambda * order_size
    impact = aligned_data['lambda'] * order_size
    
    return pd.Series(impact.values, index=aligned_data.index)


def compute_realized_spread(
    quoted_spread: float,
    realized_multiplier: float = 0.65
) -> float:
    """
    Compute realized spread from quoted spread (Goyenko et al. 2009).
    
    Realized spreads are typically 30-50% tighter than quoted spreads
    because market makers don't always capture the full spread.
    
    Args:
        quoted_spread: Quoted bid-ask spread (as fraction)
        realized_multiplier: Multiplier for realized spread (default 0.65 = 35% tighter)
    
    Returns:
        Realized spread (as fraction)
    """
    return quoted_spread * realized_multiplier


def compute_regime_adjusted_spread(
    base_spread: float,
    vix: Optional[pd.Series] = None,
    time_of_day: Optional[pd.Series] = None,
    vix_threshold: float = 30.0,
    open_multiplier: float = 1.5,
    stress_multiplier: float = 5.0
) -> pd.Series:
    """
    Compute regime-adjusted spread (time-of-day and stress effects).
    
    Spreads are:
    - 50% wider at market open (Goyenko et al. 2009)
    - 5x wider in stress periods (VIX > 30)
    
    Args:
        base_spread: Base spread (as fraction)
        vix: Optional VIX series for stress detection
        time_of_day: Optional time-of-day indicator (1 = market open, 0 = other)
        vix_threshold: VIX threshold for stress (default 30)
        open_multiplier: Multiplier at market open (default 1.5 = 50% wider)
        stress_multiplier: Multiplier in stress (default 5.0 = 5x wider)
    
    Returns:
        Series of regime-adjusted spreads
    """
    # Start with base spread
    adjusted = pd.Series(base_spread, index=time_of_day.index if time_of_day is not None else vix.index if vix is not None else pd.RangeIndex(1))
    
    # Time-of-day adjustment
    if time_of_day is not None:
        adjusted = adjusted * (1.0 + (open_multiplier - 1.0) * time_of_day)
    
    # Stress adjustment (VIX > threshold)
    if vix is not None:
        stress_indicator = (vix > vix_threshold).astype(float)
        adjusted = adjusted * (1.0 + (stress_multiplier - 1.0) * stress_indicator)
    
    return adjusted


def compute_total_market_impact(
    positions: pd.Series,
    volatility: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    model: str = 'almgren_chriss',
    participation_rate: float = 0.01,
    **kwargs
) -> pd.Series:
    """
    Compute total market impact using specified model.
    
    Args:
        positions: Series of position values
        volatility: Annualized volatility series
        volumes: Trading volume series
        prices: Price series
        model: Impact model ('almgren_chriss' or 'kyle')
        participation_rate: Participation rate
        **kwargs: Additional model parameters
    
    Returns:
        Series of total market impact costs
    """
    if model == 'almgren_chriss':
        temp_impact, perm_impact = compute_almgren_chriss_impact(
            positions, volatility, volumes, prices,
            participation_rate,
            lambda_temp=kwargs.get('lambda_temp', 0.5),
            mu_perm=kwargs.get('mu_perm', 0.1),
            periods_per_year=kwargs.get('periods_per_year', 252)
        )
        # Total impact = temporary + permanent
        # For cost purposes, we care about temporary (reverts) + permanent (doesn't revert)
        # Both are costs we pay
        total_impact = temp_impact + perm_impact
        return total_impact
    
    elif model == 'kyle':
        kyle_lambda = compute_kyle_lambda(
            volatility, volumes, prices,
            participation_rate,
            periods_per_year=kwargs.get('periods_per_year', 252)
        )
        impact = compute_kyle_impact(
            positions, kyle_lambda, volumes, prices,
            participation_rate
        )
        return impact
    
    else:
        raise ValueError(f"Unknown model: {model}. Use 'almgren_chriss' or 'kyle'")


# Example usage and validation
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    prices = 100 * (1 + np.random.randn(252).cumsum() * 0.01)
    volumes = 1e6 * (1 + np.random.randn(252) * 0.1).clip(lower=0.1)
    volatility = pd.Series(0.15, index=dates)  # 15% annualized
    positions = pd.Series(np.random.choice([-1, 0, 1], 252), index=dates)
    
    # Test Almgren-Chriss
    temp, perm = compute_almgren_chriss_impact(positions, volatility, pd.Series(volumes, index=dates), pd.Series(prices, index=dates))
    print(f"Almgren-Chriss: temp impact mean = {temp.mean():.6f}, perm impact mean = {perm.mean():.6f}")
    
    # Test Kyle
    kyle_lambda = compute_kyle_lambda(volatility, pd.Series(volumes, index=dates), pd.Series(prices, index=dates))
    print(f"Kyle lambda mean = {kyle_lambda.mean():.6f}")
    
    kyle_impact = compute_kyle_impact(positions, kyle_lambda, pd.Series(volumes, index=dates), pd.Series(prices, index=dates))
    print(f"Kyle impact mean = {kyle_impact.mean():.6f}")

