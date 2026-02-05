"""
Sharpe vs AUM Analysis

Computes Sharpe ratio as a function of AUM to identify:
- Optimal AUM (maximizes Sharpe)
- Capacity breakpoint (where Sharpe collapses)
- Fixed cost impact at different AUM levels
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from decay_analysis import compute_performance_metrics
from statistical_rigor import compute_sharpe_with_ci
from transaction_costs import compute_net_returns_from_positions, compute_annual_turnover
from market_impact import compute_total_market_impact


def compute_sharpe_at_aum(
    gross_returns: pd.Series,
    positions: pd.Series,
    aum: float,
    base_aum: float,
    volatility: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    commission_per_trade: float = 0.005,
    half_spread: float = 0.001,
    fixed_costs_annual: float = 500_000,  # $500K infrastructure
    participation_rate: float = 0.01,
    use_proper_impact: bool = True,
    periods_per_year: int = 252
) -> Tuple[float, float, Dict]:
    """
    Compute Sharpe ratio at a specific AUM level.
    
    Accounts for:
    - Market impact increases with AUM
    - Fixed costs become negligible at high AUM
    - Optimal AUM maximizes Sharpe
    
    Args:
        gross_returns: Gross strategy returns
        positions: Position series
        aum: AUM level to test
        base_aum: Base AUM for scaling positions
        volatility: Volatility series
        volumes: Volume series
        prices: Price series
        commission_per_trade: Commission per trade
        half_spread: Half spread
        fixed_costs_annual: Annual fixed costs (infrastructure)
        participation_rate: Participation rate
        use_proper_impact: Use Almgren-Chriss model
        periods_per_year: Periods per year
    
    Returns:
        Tuple of (sharpe, net_return_mean, metrics_dict)
    """
    # Scale positions to AUM (if AUM changes, positions scale proportionally)
    # For simplicity, assume position size scales with AUM
    position_scale = aum / base_aum if base_aum > 0 else 1.0
    scaled_positions = positions * position_scale
    
    # Compute explicit costs (commission + spread)
    explicit_costs = compute_net_returns_from_positions(
        gross_returns, scaled_positions,
        commission_per_trade=commission_per_trade,
        half_spread=half_spread,
        periods_per_year=periods_per_year
    )
    # Net after explicit costs
    net_after_explicit = gross_returns - (gross_returns - explicit_costs)
    
    # Market impact (increases with AUM)
    if use_proper_impact and volatility is not None and volumes is not None and prices is not None:
        market_impact = compute_total_market_impact(
            scaled_positions, volatility, volumes, prices,
            model='almgren_chriss',
            participation_rate=participation_rate,
            periods_per_year=periods_per_year
        )
        market_impact_aligned = market_impact.reindex(gross_returns.index, method='ffill').fillna(0)
    else:
        market_impact_aligned = pd.Series(0.0, index=gross_returns.index)
    
    # Fixed costs (decreases as % of AUM)
    fixed_cost_per_period = (fixed_costs_annual / aum) / periods_per_year if aum > 0 else 0
    
    # Total costs
    total_costs = (gross_returns - explicit_costs) + market_impact_aligned + fixed_cost_per_period
    
    # Net returns
    net_returns = gross_returns - total_costs
    
    # Compute Sharpe
    metrics = compute_performance_metrics(net_returns)
    sharpe = metrics.sharpe_ratio if metrics.sharpe_ratio else 0.0
    
    return sharpe, metrics.return_mean, {
        'sharpe': sharpe,
        'net_return': metrics.return_mean,
        'cost_drag': total_costs.mean() * periods_per_year,
        'fixed_cost_pct': (fixed_costs_annual / aum) * 100 if aum > 0 else None,
        'market_impact_mean': market_impact_aligned.mean() * periods_per_year if use_proper_impact else 0
    }


def compute_sharpe_vs_aum_curve(
    gross_returns: pd.Series,
    positions: pd.Series,
    volatility: pd.Series,
    volumes: pd.Series,
    prices: pd.Series,
    aum_levels: Optional[List[float]] = None,
    base_aum: float = 1_000_000,
    fixed_costs_annual: float = 500_000,
    commission_per_trade: float = 0.005,
    half_spread: float = 0.001,
    participation_rate: float = 0.01,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Compute Sharpe ratio across a range of AUM levels.
    
    Args:
        gross_returns: Gross strategy returns
        positions: Position series
        volatility: Volatility series
        volumes: Volume series
        prices: Price series
        aum_levels: List of AUM levels to test (default: $1M to $500M)
        base_aum: Base AUM for scaling
        fixed_costs_annual: Annual fixed costs
        commission_per_trade: Commission per trade
        half_spread: Half spread
        participation_rate: Participation rate
        periods_per_year: Periods per year
    
    Returns:
        DataFrame with AUM, Sharpe, Net Return, Cost Drag for each level
    """
    if aum_levels is None:
        # Default: $1M to $500M in log space
        aum_levels = np.logspace(6, 8.7, 20).tolist()  # $1M to $500M
    
    results = []
    
    for aum in aum_levels:
        sharpe, net_return, metrics = compute_sharpe_at_aum(
            gross_returns, positions, aum, base_aum,
            volatility, volumes, prices,
            commission_per_trade, half_spread,
            fixed_costs_annual, participation_rate,
            use_proper_impact=True,
            periods_per_year=periods_per_year
        )
        
        results.append({
            'aum': aum,
            'aum_millions': aum / 1_000_000,
            'sharpe': sharpe,
            'net_return': net_return,
            'cost_drag': metrics['cost_drag'],
            'fixed_cost_pct': metrics['fixed_cost_pct'],
            'market_impact_annual': metrics['market_impact_mean']
        })
    
    return pd.DataFrame(results)


def find_optimal_aum(
    sharpe_curve: pd.DataFrame,
    min_sharpe: float = 0.5
) -> Dict:
    """
    Find optimal AUM that maximizes Sharpe.
    
    Args:
        sharpe_curve: Output from compute_sharpe_vs_aum_curve
        min_sharpe: Minimum acceptable Sharpe
    
    Returns:
        Dict with optimal AUM and metrics
    """
    # Filter to viable AUMs (Sharpe >= min_sharpe)
    viable = sharpe_curve[sharpe_curve['sharpe'] >= min_sharpe].copy()
    
    if len(viable) == 0:
        return {
            'optimal_aum': None,
            'optimal_sharpe': None,
            'max_viable_aum': None,
            'viable': False
        }
    
    # Find AUM with maximum Sharpe
    optimal_idx = viable['sharpe'].idxmax()
    optimal_row = viable.loc[optimal_idx]
    
    # Find maximum viable AUM (where Sharpe drops below threshold)
    # Start from highest AUM and find where it crosses threshold
    sorted_curve = sharpe_curve.sort_values('aum', ascending=False)
    max_viable = None
    for _, row in sorted_curve.iterrows():
        if row['sharpe'] >= min_sharpe:
            max_viable = row['aum']
            break
    
    return {
        'optimal_aum': optimal_row['aum'],
        'optimal_aum_millions': optimal_row['aum_millions'],
        'optimal_sharpe': optimal_row['sharpe'],
        'max_viable_aum': max_viable,
        'max_viable_aum_millions': max_viable / 1_000_000 if max_viable else None,
        'viable': True
    }


def find_capacity_breakpoint(
    sharpe_curve: pd.DataFrame,
    sharpe_threshold: float = 0.5
) -> float:
    """
    Find AUM where Sharpe drops below threshold (capacity breakpoint).
    
    Args:
        sharpe_curve: Output from compute_sharpe_vs_aum_curve
        sharpe_threshold: Sharpe threshold
    
    Returns:
        AUM where Sharpe crosses threshold (or None if never crosses)
    """
    # Sort by AUM ascending
    sorted_curve = sharpe_curve.sort_values('aum').copy()
    
    # Find where Sharpe drops below threshold
    for i in range(len(sorted_curve)):
        if sorted_curve.iloc[i]['sharpe'] < sharpe_threshold:
            # Interpolate between this point and previous
            if i > 0:
                prev_row = sorted_curve.iloc[i-1]
                curr_row = sorted_curve.iloc[i]
                
                # Linear interpolation
                sharpe_diff = curr_row['sharpe'] - prev_row['sharpe']
                if sharpe_diff != 0:
                    aum_diff = curr_row['aum'] - prev_row['aum']
                    threshold_diff = sharpe_threshold - prev_row['sharpe']
                    breakpoint_aum = prev_row['aum'] + (threshold_diff / sharpe_diff) * aum_diff
                    return breakpoint_aum
                else:
                    return curr_row['aum']
            else:
                return sorted_curve.iloc[i]['aum']
    
    # Never crosses threshold
    return None


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    gross_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005, index=dates)
    positions = pd.Series(np.random.choice([-1, 0, 1], 252), index=dates)
    volatility = pd.Series(0.15, index=dates)
    volumes = pd.Series(1e6, index=dates)
    prices = pd.Series(100, index=dates)
    
    # Compute Sharpe vs AUM curve
    curve = compute_sharpe_vs_aum_curve(
        gross_returns, positions, volatility, volumes, prices
    )
    
    print("Sharpe vs AUM Curve (first 5):")
    print(curve.head().to_string())
    
    # Find optimal AUM
    optimal = find_optimal_aum(curve)
    print(f"\nOptimal AUM: ${optimal['optimal_aum_millions']:.2f}M")
    print(f"Optimal Sharpe: {optimal['optimal_sharpe']:.4f}")
    
    # Find capacity breakpoint
    breakpoint = find_capacity_breakpoint(curve)
    print(f"\nCapacity Breakpoint: ${breakpoint / 1_000_000:.2f}M" if breakpoint else "No breakpoint found")

