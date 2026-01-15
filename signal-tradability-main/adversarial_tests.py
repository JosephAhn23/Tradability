"""
Adversarial Tests: Break the Framework Properly

These tests attack external reality, not just my assumptions.

They will break things. That's the point.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from transaction_costs import compute_net_returns_from_positions
from decay_analysis import compute_returns


@dataclass
class AdversarialScenario:
    """A scenario designed to break the framework."""
    name: str
    description: str
    breaks_what: str


def create_2008_regime_break(prices: pd.Series, volumes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Adversarial Test 1: 2008-style regime break
    
    What breaks:
    - Volume collapses when you need it most
    - Spreads widen catastrophically
    - Liquidity disappears
    - Capacity model assumes normal markets
    """
    # Find a period that looks like 2008 (high volatility, declining prices)
    returns = prices.pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    
    # Simulate 2008: volume collapse + spread widening
    crisis_start = prices.index[len(prices) // 3]  # Roughly 1/3 through
    crisis_end = prices.index[len(prices) * 2 // 3]  # Roughly 2/3 through
    
    volumes_adversarial = volumes.copy()
    volumes_adversarial.loc[crisis_start:crisis_end] *= 0.1  # 90% volume collapse
    
    # Prices: sharp decline then recovery
    prices_adversarial = prices.copy()
    crisis_period = (prices.index >= crisis_start) & (prices.index <= crisis_end)
    prices_adversarial.loc[crisis_period] *= 0.7  # 30% drawdown
    
    return prices_adversarial, volumes_adversarial


def create_limit_down_days(prices: pd.Series) -> pd.Series:
    """
    Adversarial Test 2: Limit-down days
    
    What breaks:
    - Can't exit positions
    - Cost models assume you can always trade
    - Framework doesn't model execution failure
    """
    prices_adversarial = prices.copy()
    
    # Add limit-down days (can't trade, price gaps down)
    limit_down_days = prices.index[::50]  # Every 50th day
    for day in limit_down_days:
        if day in prices_adversarial.index:
            # Price gaps down 5% (limit down)
            prices_adversarial.loc[day] = prices_adversarial.loc[day] * 0.95
    
    return prices_adversarial


def create_volume_collapse_after_signal(prices: pd.Series, volumes: pd.Series, signals: pd.Series) -> pd.Series:
    """
    Adversarial Test 3: Volume collapses right after signal triggers
    
    What breaks:
    - Signal says "trade now"
    - But liquidity disappears
    - Cost models assume you can execute
    """
    volumes_adversarial = volumes.copy()
    
    # Find when signals trigger (large position changes)
    signal_changes = signals.diff().abs()
    trigger_days = signal_changes[signal_changes > signal_changes.quantile(0.9)].index
    
    # Collapse volume on days after triggers
    for trigger_day in trigger_days:
        next_day = trigger_day + timedelta(days=1)
        if next_day in volumes_adversarial.index:
            volumes_adversarial.loc[next_day] *= 0.05  # 95% collapse
    
    return volumes_adversarial


def create_slippage_nonlinearity(prices: pd.Series, volumes: pd.Series, positions: pd.Series) -> pd.Series:
    """
    Adversarial Test 4: Slippage is nonlinear (breaks at scale)
    
    What breaks:
    - Linear slippage models assume constant impact
    - Reality: impact scales superlinearly
    - Framework assumes linear, so capacity is overestimated
    """
    # This would require modifying the slippage function itself
    # For now, we'll create a scenario where large positions face nonlinear costs
    # by artificially reducing volumes when positions are large
    
    volumes_adversarial = volumes.copy()
    large_position_days = positions[positions.abs() > 0.8].index
    
    for day in large_position_days:
        if day in volumes_adversarial.index:
            # Nonlinear: large positions face much worse liquidity
            volumes_adversarial.loc[day] *= 0.2  # 80% reduction for large positions
    
    return volumes_adversarial


def create_pathological_price_path(prices: pd.Series) -> pd.Series:
    """
    Adversarial Test 5: Pathological price path
    
    What breaks:
    - Framework assumes continuous trading
    - Reality: gaps, halts, circuit breakers
    - Cost models don't account for execution failure
    """
    prices_adversarial = prices.copy()
    
    # Add extreme volatility clusters (flash crash style)
    for i in range(0, len(prices), 100):
        if i + 5 < len(prices):
            # 5-day extreme volatility cluster
            cluster = prices_adversarial.iloc[i:i+5]
            # Random walk with extreme moves
            for j in range(1, len(cluster)):
                move = np.random.choice([-0.10, -0.05, 0.05, 0.10])  # ±5-10% moves
                prices_adversarial.iloc[i+j] = prices_adversarial.iloc[i+j-1] * (1 + move)
    
    return prices_adversarial


def test_adversarial_scenario(scenario_name: str, 
                              signal_name: str = 'momentum_12_1',
                              ticker: str = 'SPY',
                              start_date: datetime = datetime(2000, 1, 1),
                              end_date: datetime = datetime(2020, 12, 31) -> Dict:
    """
    Run a signal through an adversarial scenario.
    
    Returns:
        Dict with results showing how framework breaks
    """
    print(f"=" * 80)
    print(f"ADVERSARIAL TEST: {scenario_name}")
    print("=" * 80)
    print()
    
    # Load baseline data
    prices_baseline, volumes_baseline = load_price_data(ticker, start_date, end_date)
    forward_returns_baseline = compute_forward_returns(prices_baseline)
    
    # Create adversarial scenario
    if scenario_name == "2008_regime_break":
        prices_adv, volumes_adv = create_2008_regime_break(prices_baseline, volumes_baseline)
    elif scenario_name == "limit_down_days":
        prices_adv = create_limit_down_days(prices_baseline)
        volumes_adv = volumes_baseline
    elif scenario_name == "pathological_price_path":
        prices_adv = create_pathological_price_path(prices_baseline)
        volumes_adv = volumes_baseline
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    forward_returns_adv = compute_forward_returns(prices_adv)
    volatility_adv = prices_adv.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Compute signal on adversarial data
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices_adv, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns_adv)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Test with volume collapse after signal (if applicable)
    if scenario_name == "volume_collapse_after_signal":
        volumes_adv = create_volume_collapse_after_signal(prices_adv, volumes_adv, aligned_signals)
    
    # Run framework on adversarial data
    tradability_adv = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility_adv,
        volumes=volumes_adv,
        prices=prices_adv,
        commission_per_trade=0.005,
        half_spread=0.001,
        periods_per_year=252
    )
    
    # Compare to baseline
    tradability_baseline = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=prices_baseline.pct_change().rolling(20).std() * np.sqrt(252),
        volumes=volumes_baseline,
        prices=prices_baseline,
        commission_per_trade=0.005,
        half_spread=0.001,
        periods_per_year=252
    )
    
    # Extract what breaks
    baseline_capacity = tradability_baseline.max_viable_capacity or 0
    adv_capacity = tradability_adv.max_viable_capacity or 0
    
    baseline_break_even = tradability_baseline.break_even_cost
    adv_break_even = tradability_adv.break_even_cost
    
    baseline_cost_drag = tradability_baseline.cost_drag
    adv_cost_drag = tradability_adv.cost_drag
    
    # Compute net returns (what actually happens)
    net_returns_baseline = compute_net_returns_from_positions(
        gross_returns, positions, commission_per_trade=0.005, half_spread=0.001, periods_per_year=252
    )
    
    # For adversarial, use wider spreads (crisis conditions)
    net_returns_adv = compute_net_returns_from_positions(
        gross_returns, positions, commission_per_trade=0.005, half_spread=0.005, periods_per_year=252  # 5x wider spread
    )
    
    results = {
        'scenario': scenario_name,
        'signal': signal_name,
        'baseline': {
            'capacity': baseline_capacity,
            'break_even_cost': baseline_break_even,
            'cost_drag': baseline_cost_drag,
            'net_sharpe': net_returns_baseline.mean() / net_returns_baseline.std() * np.sqrt(252) if net_returns_baseline.std() > 0 else 0
        },
        'adversarial': {
            'capacity': adv_capacity,
            'break_even_cost': adv_break_even,
            'cost_drag': adv_cost_drag,
            'net_sharpe': net_returns_adv.mean() / net_returns_adv.std() * np.sqrt(252) if net_returns_adv.std() > 0 else 0
        },
        'breaks': {
            'capacity_overestimate_pct': ((baseline_capacity - adv_capacity) / baseline_capacity * 100) if baseline_capacity > 0 else 0,
            'break_even_worse_pct': ((adv_break_even - baseline_break_even) / baseline_break_even * 100) if baseline_break_even > 0 else 0,
            'cost_drag_increase_pct': ((adv_cost_drag - baseline_cost_drag) / baseline_cost_drag * 100) if baseline_cost_drag > 0 else 0,
            'sharpe_deterioration': results['baseline']['net_sharpe'] - results['adversarial']['net_sharpe']
        }
    }
    
    print(f"Baseline capacity: ${baseline_capacity/1e6:.1f}M")
    print(f"Adversarial capacity: ${adv_capacity/1e6:.1f}M")
    print(f"Capacity overestimate: {results['breaks']['capacity_overestimate_pct']:.1f}%")
    print()
    print(f"Baseline break-even: {baseline_break_even*100:.2f}%")
    print(f"Adversarial break-even: {adv_break_even*100:.2f}%")
    print()
    print(f"Baseline net Sharpe: {results['baseline']['net_sharpe']:.3f}")
    print(f"Adversarial net Sharpe: {results['adversarial']['net_sharpe']:.3f}")
    print(f"Sharpe deterioration: {results['breaks']['sharpe_deterioration']:.3f}")
    print()
    
    return results


def run_all_adversarial_tests():
    """Run all adversarial tests and show where framework breaks."""
    print("=" * 80)
    print("ADVERSARIAL TEST SUITE")
    print("Breaking the Framework Properly")
    print("=" * 80)
    print()
    
    scenarios = [
        "2008_regime_break",
        "limit_down_days",
        "pathological_price_path",
    ]
    
    all_results = []
    
    for scenario in scenarios:
        try:
            results = test_adversarial_scenario(scenario)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in {scenario}: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY: Where Framework Breaks")
    print("=" * 80)
    print()
    
    for result in all_results:
        print(f"{result['scenario']}:")
        print(f"  Capacity overestimate: {result['breaks']['capacity_overestimate_pct']:.1f}%")
        print(f"  Sharpe deterioration: {result['breaks']['sharpe_deterioration']:.3f}")
        print()
    
    return all_results


if __name__ == "__main__":
    run_all_adversarial_tests()

