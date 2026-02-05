"""
Capacity Constraints Module

Estimates capital scalability:
- Maximum viable capital (AUM)
- Performance as function of AUM
- Breakpoints where alpha collapses

Even rough capacity estimates provide meaningful insight.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple


def estimate_capacity_from_volume(volumes: pd.Series,
                                  prices: Optional[pd.Series] = None,
                                  participation_rate: float = 0.01,
                                  lookback_days: int = 20) -> float:
    """
    Estimate capacity based on average trading volume.
    
    Mathematical derivation:
    
    Capacity constraint: We cannot trade more than X% of daily volume without excessive impact.
    
    Basic formula:
        Capacity = Participation_Rate × Average_Daily_Dollar_Volume
    
    Assumptions:
    1. Participation rate: 1% of daily volume (conservative, based on academic literature)
       - Almgren & Chriss (2000) suggest < 5% for minimal impact
       - We use 1% for conservative estimate
    2. Average volume: 20-day rolling average (smooths daily variation)
    3. No liquidity crisis: Assumes normal market conditions
    
    Why not higher participation?
    - Market impact scales nonlinearly with participation
    - Kyle (1985): Impact ≈ participation^α where α ≈ 0.5-1.0
    - At 1% participation, impact ~ 10 bps (acceptable)
    - At 5% participation, impact ~ 50-100 bps (destroys edge)
    
    Why not $500k or $5M?
    - Depends on underlying volume and participation assumption
    - If daily volume = $100M, 1% = $1M capacity (our default)
    - If daily volume = $50M, 1% = $500k capacity
    - If daily volume = $500M, 1% = $5M capacity
    - Our estimate scales with actual volume data
    
    Args:
        volumes: Series of trading volumes (shares)
        prices: Optional price series (for dollar volume)
        participation_rate: Max participation rate (default 1% = 0.01)
        lookback_days: Days to average over for capacity estimate
    
    Returns:
        Estimated capacity in dollars (or shares if no prices)
    """
    if len(volumes) == 0:
        return 0.0
    
    # Compute average volume over lookback period
    avg_volume = volumes.rolling(window=lookback_days, min_periods=1).mean().iloc[-1]
    
    if prices is not None:
        # Dollar volume
        avg_price = prices.rolling(window=lookback_days, min_periods=1).mean().iloc[-1]
        avg_dollar_volume = avg_volume * avg_price
        capacity = participation_rate * avg_dollar_volume
        
        # Store assumptions for transparency
        capacity_info = {
            'participation_rate': participation_rate,
            'avg_daily_volume_shares': avg_volume,
            'avg_price': avg_price,
            'avg_daily_dollar_volume': avg_dollar_volume,
            'estimated_capacity': capacity
        }
    else:
        # Share volume only
        capacity = participation_rate * avg_volume
    
    return capacity


def compute_turnover_impact(annual_turnover: float,
                           capacity: float,
                           avg_volume: float,
                           avg_price: float = 1.0) -> float:
    """
    CAVEAT: This impact model is simplified and context-dependent.
    The linear approximation holds for small participation rates but may
    underestimate impact at larger scales. Impact coefficients vary by:
    - Asset class (equities vs futures vs FX)
    - Market microstructure
    - Time of day / market regime
    - Signal characteristics
    
    This is a stylized model for estimation, not a universal law.
    """
    """
    Estimate performance impact from capacity constraints.
    
    Mathematical derivation:
    
    Market impact scales with participation rate and turnover.
    
    Formula:
        Participation_Rate = (Capacity × Daily_Turnover) / Daily_Dollar_Volume
        Impact = Participation_Rate × Impact_Coefficient
    
    Assumptions:
    1. Linear impact model (simplified; actual is sublinear)
       - Almgren & Chriss (2000): Impact ≈ participation^0.5 (square root)
       - We use linear for simplicity: Impact ≈ participation × 0.001 (10 bps per 1%)
    2. Daily turnover = Annual turnover / 252
    3. Impact coefficient: 0.001 = 10 basis points per 1% participation
       - Based on empirical market impact studies
       - Kyle (1985): Impact coefficient typically 0.0005-0.002 depending on asset
    
    Why linear approximation?
    - Square root model: Impact = k × √(participation)
    - Linear approximation: Impact ≈ k × participation (for small participation)
    - At 1% participation, error is small (~5%)
    - At 5% participation, square root model gives lower impact (more optimistic)
    - We use linear as conservative estimate
    
    Example:
        Capacity = $1M, Turnover = 3x/year, Volume = $100M/day
        Daily turnover = 3/252 = 0.0119
        Participation = ($1M × 0.0119) / $100M = 0.000119 = 0.0119%
        Impact = 0.0119% × 0.001 = 0.000012% per day ≈ 0.003% annualized
    
    Args:
        annual_turnover: Annual turnover rate
        capacity: Strategy capacity (AUM)
        avg_volume: Average daily trading volume (shares)
        avg_price: Average price
    
    Returns:
        Estimated impact on returns (as fraction, negative = drag)
    """
    if avg_volume <= 0:
        return 0.0
    
    # Daily turnover
    daily_turnover = annual_turnover / 252.0
    
    # Dollar volume
    dollar_volume = avg_volume * avg_price
    
    # Participation rate: how much of daily volume we trade
    participation = (capacity * daily_turnover) / dollar_volume
    
    # Market impact: linear model (conservative)
    # Impact coefficient: 0.001 = 10 bps per 1% participation
    # This is a simplified linear approximation of the square-root model
    impact_coefficient = 0.001  # 10 bps per 1% participation
    impact = participation * impact_coefficient
    
    return -abs(impact)  # Negative impact (drag on returns)


def simulate_capacity_decay(gross_returns: pd.Series,
                            positions: pd.Series,
                            volumes: Optional[pd.Series] = None,
                            prices: Optional[pd.Series] = None,
                            capacity_levels: Optional[np.ndarray] = None,
                            participation_rate: float = 0.01,
                            periods_per_year: int = 252) -> pd.DataFrame:
    """
    Simulate performance decay as function of capacity (AUM).
    
    Args:
        gross_returns: Gross strategy returns
        positions: Series of position values
        volumes: Optional volume series
        prices: Optional price series
        capacity_levels: Array of capacity levels to test (default: log scale)
        participation_rate: Max participation rate
        periods_per_year: Number of periods per year
    
    Returns:
        DataFrame with columns: capacity, net_return, sharpe_ratio, max_drawdown
    """
    if capacity_levels is None:
        # Default: test capacity from $1M to $1B (log scale)
        capacity_levels = np.logspace(6, 9, 20)  # 1e6 to 1e9
    
    # Compute annual turnover
    annual_turnover = positions.diff().abs().sum() / len(positions) * periods_per_year
    
    # Estimate base capacity from volume
    if volumes is not None:
        base_capacity = estimate_capacity_from_volume(
            volumes, prices, participation_rate
        )
    else:
        base_capacity = 1e6  # Default $1M if no volume data
    
    results = []
    
    for capacity in capacity_levels:
        # Compute capacity impact
        if volumes is not None and prices is not None:
            avg_volume = volumes.mean()
            avg_price = prices.mean()
        else:
            avg_volume = 1e6
            avg_price = 1.0
        
        impact = compute_turnover_impact(
            annual_turnover, capacity, avg_volume, avg_price
        )
        
        # Apply capacity drag to returns
        net_returns = gross_returns + impact / periods_per_year
        
        # Compute metrics
        net_return_mean = net_returns.mean()
        net_return_std = net_returns.std()
        
        if net_return_std > 0:
            sharpe = net_return_mean / net_return_std * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + net_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        results.append({
            'capacity': capacity,
            'net_return': net_return_mean,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'capacity_impact': impact,
        })
    
    return pd.DataFrame(results)


def find_capacity_breakpoint(capacity_results: pd.DataFrame,
                             sharpe_threshold: float = 0.5) -> float:
    """
    Find capacity level where Sharpe ratio falls below threshold.
    
    Args:
        capacity_results: DataFrame from simulate_capacity_decay
        sharpe_threshold: Sharpe threshold (default 0.5)
    
    Returns:
        Capacity level where Sharpe < threshold (or max capacity if never below, or 0 if dead at min)
    """
    if len(capacity_results) == 0:
        return 0.0
    
    # If dead at minimum capacity, capacity is 0
    if capacity_results.iloc[0]['sharpe_ratio'] < sharpe_threshold:
        return 0.0
    
    below_threshold = capacity_results[capacity_results['sharpe_ratio'] < sharpe_threshold]
    
    if len(below_threshold) == 0:
        # Never below threshold
        return capacity_results['capacity'].max()
    
    # Return first capacity level below threshold
    return below_threshold.iloc[0]['capacity']


def estimate_maximum_viable_capital(gross_returns: pd.Series,
                                    positions: pd.Series,
                                    volumes: Optional[pd.Series] = None,
                                    prices: Optional[pd.Series] = None,
                                    participation_rate: float = 0.01,
                                    sharpe_threshold: float = 0.5,
                                    periods_per_year: int = 252,
                                    use_sharpe_vs_aum: bool = True,
                                    volatility: Optional[pd.Series] = None,
                                    fixed_costs_annual: float = 500_000) -> Dict:
    """
    Estimate maximum viable capital for a strategy.
    
    Args:
        gross_returns: Gross strategy returns
        positions: Series of position values
        volumes: Optional volume series
        prices: Optional price series
        participation_rate: Max participation rate
        sharpe_threshold: Sharpe threshold for viability
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary with capacity metrics
    """
    # Optionally use proper Sharpe vs AUM analysis
    if use_sharpe_vs_aum and volatility is not None and volumes is not None and prices is not None:
        try:
            from sharpe_vs_aum import compute_sharpe_vs_aum_curve, find_optimal_aum, find_capacity_breakpoint as find_aum_breakpoint
            
            # Compute Sharpe vs AUM curve
            base_aum = 1_000_000  # Base for scaling
            sharpe_curve = compute_sharpe_vs_aum_curve(
                gross_returns, positions, volatility, volumes, prices,
                base_aum=base_aum,
                fixed_costs_annual=fixed_costs_annual,
                participation_rate=participation_rate,
                periods_per_year=periods_per_year
            )
            
            # Find optimal AUM
            optimal = find_optimal_aum(sharpe_curve, min_sharpe=sharpe_threshold)
            
            # Find capacity breakpoint
            breakpoint_aum = find_aum_breakpoint(sharpe_curve, sharpe_threshold=sharpe_threshold)
            
            # Estimate base capacity from volume
            base_capacity = estimate_capacity_from_volume(
                volumes, prices, participation_rate
            )
            
            # Final capacity = min of all constraints
            final_capacity = min(
                optimal['optimal_aum'] if optimal['viable'] else 0,
                breakpoint_aum if breakpoint_aum else float('inf'),
                base_capacity if base_capacity > 0 else float('inf')
            )
            
            if final_capacity == float('inf'):
                final_capacity = base_capacity if base_capacity > 0 else 0
            
            return {
                'max_viable_capacity': final_capacity,
                'base_capacity_estimate': base_capacity,
                'breakpoint_capacity': breakpoint_aum,
                'optimal_aum': optimal['optimal_aum'] if optimal['viable'] else None,
                'sharpe_curve': sharpe_curve,
                'capacity_results': sharpe_curve,  # For backward compatibility
            }
        except Exception:
            # Fall back to original method if Sharpe vs AUM fails
            pass
    
    # Original method (backward compatible)
    capacity_results = simulate_capacity_decay(
        gross_returns, positions, volumes, prices,
        participation_rate=participation_rate,
        periods_per_year=periods_per_year
    )
    
    # Find breakpoint
    breakpoint_capacity = find_capacity_breakpoint(
        capacity_results, sharpe_threshold
    )
    
    # Estimate base capacity from volume
    base_capacity = estimate_capacity_from_volume(
        volumes if volumes is not None else pd.Series([1e6]),
        prices,
        participation_rate
    )
    
    # Final capacity = min of liquidity constraint and performance constraint
    final_capacity = min(breakpoint_capacity, base_capacity) if base_capacity > 0 else breakpoint_capacity
    
    return {
        'max_viable_capacity': final_capacity,
        'base_capacity_estimate': base_capacity,
        'breakpoint_capacity': breakpoint_capacity,
        'capacity_results': capacity_results,
        'sharpe_at_base_capacity': capacity_results[
            capacity_results['capacity'] <= base_capacity
        ]['sharpe_ratio'].iloc[-1] if len(capacity_results) > 0 else None,
    }

