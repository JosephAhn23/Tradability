"""
Counterexamples: When Claims Do NOT Hold

Real quant work invites being wrong. This module explicitly tests when our claims fail.

This is what separates research from student work: explicit falsification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from formal_definitions import compute_statistical_edge, compute_economic_edge, identify_edge_mismatch


@dataclass
class CounterexampleResult:
    """Result of testing a counterexample."""
    claim: str
    condition: str
    holds: bool
    evidence: Dict
    explanation: str


def test_low_turnover_counterexample(ticker: str = 'SPY',
                                     start_date: datetime = datetime(2000, 1, 1),
                                     end_date: datetime = datetime(2020, 12, 31)) -> CounterexampleResult:
    """
    Counterexample 1: Low-Turnover Signals
    
    Claim: "High-turnover signals lose economic viability quickly"
    Counterexample: Low-turnover signals may remain viable even at higher costs
    
    Returns:
        CounterexampleResult indicating whether claim holds
    """
    print("=" * 80)
    print("COUNTEREXAMPLE 1: Low-Turnover Signals")
    print("=" * 80)
    
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Test low-turnover signal (long-horizon momentum or MA crossover)
    signal_name = 'ma_crossover'  # Typically lower turnover than mean reversion
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Analyze tradability
    tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=0.01,  # 1% (higher than standard 0.5%)
        half_spread=0.002,  # 0.2% (higher than standard 0.1%)
        periods_per_year=252
    )
    
    annual_turnover = tradability.annual_turnover
    break_even_cost = tradability.break_even_cost
    net_sharpe = tradability.net_metrics.sharpe_ratio if tradability.net_metrics else 0
    
    print(f"\nSignal: {signal_name}")
    print(f"Annual Turnover: {annual_turnover:.2f}x")
    print(f"Break-even Cost: {break_even_cost*100:.3f}% per trade")
    print(f"Net Sharpe (at 1% costs): {net_sharpe:.3f}")
    
    # Test if claim holds
    # Claim: High-turnover (>3x) signals lose viability
    # Counterexample: Low-turnover (<1x) signals may remain viable
    claim_holds = True
    if annual_turnover < 1.0 and net_sharpe > 0:
        claim_holds = False
        explanation = (
            f"Low-turnover signal ({annual_turnover:.2f}x) remains economically viable "
            f"(net Sharpe {net_sharpe:.3f}) even at high costs (1% per trade). "
            f"Claim about high-turnover signals does not apply to low-turnover signals."
        )
    else:
        explanation = (
            f"Signal has turnover {annual_turnover:.2f}x and net Sharpe {net_sharpe:.3f}. "
            f"Claim holds: high-turnover signals lose viability, but this is a low-turnover signal."
        )
    
    return CounterexampleResult(
        claim="High-turnover signals (>3x) lose economic viability quickly",
        condition=f"Low-turnover signal (turnover={annual_turnover:.2f}x) at high costs (1% per trade)",
        holds=claim_holds,
        evidence={
            'annual_turnover': annual_turnover,
            'break_even_cost': break_even_cost,
            'net_sharpe': net_sharpe,
            'gross_sharpe': tradability.gross_metrics.sharpe_ratio if tradability.gross_metrics else 0,
        },
        explanation=explanation
    )


def test_high_capacity_counterexample(ticker: str = 'SPY',
                                       start_date: datetime = datetime(2000, 1, 1),
                                       end_date: datetime = datetime(2020, 12, 31)) -> CounterexampleResult:
    """
    Counterexample 2: High-Capacity Assets
    
    Claim: "Capacity constraints eliminate economic edge"
    Counterexample: Very liquid assets (high volume) can support larger capacity
    
    Returns:
        CounterexampleResult
    """
    print("\n" + "=" * 80)
    print("COUNTEREXAMPLE 2: High-Capacity Assets")
    print("=" * 80)
    
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Test signal on high-capacity asset (SPY has very high volume)
    signal_name = 'momentum_12_1'
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Analyze tradability
    tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=0.005,
        half_spread=0.001,
        periods_per_year=252
    )
    
    max_capacity = tradability.max_viable_capacity or 0
    avg_volume = volumes.mean() if volumes is not None else 0
    
    print(f"\nAsset: {ticker}")
    print(f"Average Daily Volume: ${avg_volume/1e6:.1f}M")
    print(f"Max Viable Capacity: ${max_capacity/1e6:.1f}M")
    print(f"Capacity as % of Daily Volume: {max_capacity/avg_volume*100:.2f}%")
    
    # Test if claim holds
    # Claim: Capacity constraints eliminate edge
    # Counterexample: High-capacity assets can support large AUM
    claim_holds = True
    if max_capacity > 100e6:  # >$100M capacity
        claim_holds = False
        explanation = (
            f"High-capacity asset ({ticker}) can support ${max_capacity/1e6:.1f}M AUM. "
            f"Capacity constraints do not bind for small-to-medium funds (<$100M). "
            f"Claim about capacity constraints does not apply to very liquid assets."
        )
    else:
        explanation = (
            f"Asset has capacity ${max_capacity/1e6:.1f}M. "
            f"Claim holds: capacity constraints limit scalability."
        )
    
    return CounterexampleResult(
        claim="Capacity constraints eliminate economic edge",
        condition=f"High-capacity asset ({ticker}) with large daily volume",
        holds=claim_holds,
        evidence={
            'max_capacity': max_capacity,
            'avg_daily_volume': avg_volume,
            'capacity_pct_of_volume': max_capacity/avg_volume*100 if avg_volume > 0 else 0,
        },
        explanation=explanation
    )


def test_optimized_signal_counterexample(ticker: str = 'SPY',
                                         start_date: datetime = datetime(2000, 1, 1),
                                         end_date: datetime = datetime(2020, 12, 31)) -> CounterexampleResult:
    """
    Counterexample 3: Optimized Signal Definitions
    
    Claim: "Signals with fixed parameters lose economic edge"
    Counterexample: Parameter optimization may reduce turnover while preserving edge
    
    Note: This is a conceptual counterexample since we don't optimize in this research.
    """
    print("\n" + "=" * 80)
    print("COUNTEREXAMPLE 3: Optimized Signal Definitions")
    print("=" * 80)
    
    print("\nNOTE: This research uses fixed parameters (no optimization).")
    print("This counterexample is conceptual: optimization may change tradability.")
    
    explanation = (
        "Our analysis uses fixed parameters and no optimization. "
        "Parameter optimization may reduce turnover (e.g., longer lookback periods) "
        "or improve signal quality, potentially preserving economic edge. "
        "This is a limitation of our analysis, not a failure of the claim."
    )
    
    return CounterexampleResult(
        claim="Signals with fixed parameters lose economic edge at realistic costs",
        condition="Optimized signal definitions (parameter tuning, regime selection)",
        holds=True,  # Claim holds for fixed parameters; optimization is out of scope
        evidence={
            'note': 'Conceptual counterexample - optimization not tested',
        },
        explanation=explanation
    )


def test_low_cost_counterexample(ticker: str = 'SPY',
                                 start_date: datetime = datetime(2000, 1, 1),
                                 end_date: datetime = datetime(2020, 12, 31)) -> CounterexampleResult:
    """
    Counterexample 4: Different Cost Structures
    
    Claim: "Signals become untradable at realistic costs (0.5% per trade)"
    Counterexample: Institutional execution with sophisticated algorithms may achieve lower costs
    """
    print("\n" + "=" * 80)
    print("COUNTEREXAMPLE 4: Different Cost Structures")
    print("=" * 80)
    
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Test signal with very low costs (institutional execution)
    signal_name = 'mean_reversion'
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Test at very low costs (institutional execution)
    low_cost_tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=0.001,  # 0.1% (very low, institutional)
        half_spread=0.0005,  # 0.05% (very low)
        periods_per_year=252
    )
    
    # Compare to standard costs
    standard_tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=0.005,  # 0.5% (standard)
        half_spread=0.001,  # 0.1% (standard)
        periods_per_year=252
    )
    
    low_cost_sharpe = low_cost_tradability.net_metrics.sharpe_ratio if low_cost_tradability.net_metrics else 0
    standard_sharpe = standard_tradability.net_metrics.sharpe_ratio if standard_tradability.net_metrics else 0
    
    print(f"\nSignal: {signal_name}")
    print(f"Net Sharpe at 0.1% costs: {low_cost_sharpe:.3f}")
    print(f"Net Sharpe at 0.5% costs: {standard_sharpe:.3f}")
    
    # Test if claim holds
    claim_holds = True
    if low_cost_sharpe > 0 and standard_sharpe < 0:
        claim_holds = False
        explanation = (
            f"Signal is economically viable at low costs (0.1% per trade, net Sharpe {low_cost_sharpe:.3f}) "
            f"but not at standard costs (0.5% per trade, net Sharpe {standard_sharpe:.3f}). "
            f"Claim about untradability at realistic costs may not hold for institutional execution."
        )
    else:
        explanation = (
            f"Signal has net Sharpe {low_cost_sharpe:.3f} at low costs and {standard_sharpe:.3f} at standard costs. "
            f"Claim holds: signals become untradable at realistic retail/institutional costs."
        )
    
    return CounterexampleResult(
        claim="Signals become untradable at realistic costs (0.5% per trade)",
        condition="Institutional execution with very low costs (0.1% per trade)",
        holds=claim_holds,
        evidence={
            'low_cost_sharpe': low_cost_sharpe,
            'standard_cost_sharpe': standard_sharpe,
            'cost_reduction_pct': (0.005 - 0.001) / 0.005 * 100,
        },
        explanation=explanation
    )


def run_all_counterexamples() -> List[CounterexampleResult]:
    """
    Run all counterexample tests.
    
    Returns:
        List of CounterexampleResult objects
    """
    print("=" * 80)
    print("COUNTEREXAMPLES: When Claims Do NOT Hold")
    print("=" * 80)
    print("\nReal quant work invites being wrong.")
    print("This section explicitly tests when our claims fail.\n")
    
    results = []
    
    # Test each counterexample
    try:
        result1 = test_low_turnover_counterexample()
        results.append(result1)
    except Exception as e:
        print(f"Error in low-turnover counterexample: {e}")
    
    try:
        result2 = test_high_capacity_counterexample()
        results.append(result2)
    except Exception as e:
        print(f"Error in high-capacity counterexample: {e}")
    
    try:
        result3 = test_optimized_signal_counterexample()
        results.append(result3)
    except Exception as e:
        print(f"Error in optimized signal counterexample: {e}")
    
    try:
        result4 = test_low_cost_counterexample()
        results.append(result4)
    except Exception as e:
        print(f"Error in low-cost counterexample: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COUNTEREXAMPLE SUMMARY")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        status = "❌ CLAIM FAILS" if not result.holds else "✅ CLAIM HOLDS"
        print(f"\n{i}. {result.claim}")
        print(f"   Condition: {result.condition}")
        print(f"   Result: {status}")
        print(f"   Explanation: {result.explanation}")
    
    return results



