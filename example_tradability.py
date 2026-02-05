"""
Example: Tradability Analysis

Demonstrates how to use the tradability analysis modules to measure
economic decay vs statistical decay.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import (
    analyze_tradability, 
    compute_drawdown_sensitivity_to_costs,
    compute_positions_from_returns
)


def example_basic_tradability():
    """
    Basic example: Compute gross vs net performance for a single signal.
    """
    print("=" * 70)
    print("Example: Basic Tradability Analysis")
    print("=" * 70)
    
    # Load data
    ticker = 'SPY'
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    print(f"\nLoading data for {ticker}...")
    prices, volumes = load_price_data(ticker, start_date, end_date)
    
    # Compute signal
    signal_name = 'momentum_12_1'
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    
    # Compute forward returns
    forward_returns = compute_forward_returns(prices)
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    
    # Compute gross returns
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    
    print(f"\nSignal: {signal_name}")
    print(f"Period: {gross_returns.index[0]} to {gross_returns.index[-1]}")
    print(f"Observations: {len(gross_returns)}")
    
    # Compute gross performance
    gross_metrics = compute_performance_metrics(gross_returns)
    print(f"\nGross Performance:")
    print(f"  Sharpe Ratio: {gross_metrics.sharpe_ratio:.3f}")
    print(f"  Mean Return: {gross_metrics.return_mean:.4f} ({gross_metrics.return_mean*252*100:.2f}% annualized)")
    print(f"  Hit Rate: {gross_metrics.hit_rate:.1%}")
    print(f"  Max Drawdown: {gross_metrics.max_drawdown:.1%}")
    
    # Compute tradability metrics (with costs)
    print(f"\nComputing tradability metrics (with costs)...")
    tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=prices.pct_change().rolling(20).std() * np.sqrt(252),  # Annualized vol
        volumes=volumes,
        prices=prices,
        commission_per_trade=0.005,  # 0.5% per trade
        half_spread=0.001,  # 0.1% half-spread
        periods_per_year=252
    )
    
    # Net performance
    if tradability.net_metrics:
        print(f"\nNet Performance (after costs):")
        print(f"  Sharpe Ratio: {tradability.net_metrics.sharpe_ratio:.3f}")
        print(f"  Mean Return: {tradability.net_metrics.return_mean:.4f} ({tradability.net_metrics.return_mean*252*100:.2f}% annualized)")
        print(f"  Hit Rate: {tradability.net_metrics.hit_rate:.1%}")
        print(f"  Max Drawdown: {tradability.net_metrics.max_drawdown:.1%}")
    
    # Economic metrics
    print(f"\nEconomic Metrics:")
    print(f"  Annual Turnover: {tradability.annual_turnover:.1f}x")
    print(f"  Cost Drag: {tradability.cost_drag*100:.2f}% annualized")
    print(f"  Break-even Cost: {tradability.break_even_cost*100:.3f}% per trade")
    if tradability.max_viable_capacity:
        print(f"  Max Viable Capacity: ${tradability.max_viable_capacity/1e6:.1f}M")
    
    # Performance degradation
    if tradability.gross_metrics and tradability.net_metrics:
        sharpe_decay = (tradability.net_metrics.sharpe_ratio - tradability.gross_metrics.sharpe_ratio) / abs(tradability.gross_metrics.sharpe_ratio) * 100
        return_decay = (tradability.net_metrics.return_mean - tradability.gross_metrics.return_mean) / abs(tradability.gross_metrics.return_mean) * 100
        print(f"\nPerformance Decay (due to costs):")
        print(f"  Sharpe Ratio Decay: {sharpe_decay:.1f}%")
        print(f"  Return Decay: {return_decay:.1f}%")
        print(f"  Hit Rate Survival: {tradability.hit_rate_survival:.1%}")
        print(f"  PnL Collapse: {tradability.pnl_collapse:.1%}")
    
    print("\n" + "=" * 70)


def example_cost_sensitivity():
    """
    Example: Sensitivity analysis of drawdown to transaction costs.
    """
    print("\n" + "=" * 70)
    print("Example: Cost Sensitivity Analysis")
    print("=" * 70)
    
    # Load data
    ticker = 'SPY'
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    prices, volumes = load_price_data(ticker, start_date, end_date)
    
    # Compute signal and returns
    signal_name = 'mean_reversion'
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    forward_returns = compute_forward_returns(prices)
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    
    # Reconstruct positions
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    print(f"\nSignal: {signal_name}")
    print(f"Computing drawdown sensitivity to transaction costs...")
    
    # Compute sensitivity
    sensitivity = compute_drawdown_sensitivity_to_costs(
        gross_returns, positions, periods_per_year=252
    )
    
    print(f"\nCost Sensitivity Results:")
    print(f"Cost Level | Sharpe Ratio | Max Drawdown")
    print("-" * 50)
    for idx in [0, 10, 20, 30, 40, 49]:  # Sample points
        if idx < len(sensitivity):
            row = sensitivity.iloc[idx]
            print(f"{row['cost_level']*100:6.3f}%  | {row['sharpe_ratio']:11.3f}  | {row['max_drawdown']:11.1%}")
    
    # Find break-even cost (where Sharpe ≈ 0)
    break_even_idx = (sensitivity['sharpe_ratio'].abs()).idxmin()
    break_even_cost = sensitivity.loc[break_even_idx, 'cost_level']
    print(f"\nBreak-even cost (Sharpe ~ 0): {break_even_cost*100:.3f}% per trade")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run examples
    example_basic_tradability()
    example_cost_sensitivity()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)

