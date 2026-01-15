"""
Find Viable Signal for Sensitivity Analysis

Tests all signals to find one that's DEPLOY in base case,
then runs comprehensive sensitivity analysis on it.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from signals import list_signals
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from signals import get_signal
from decay_analysis import compute_returns
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from sensitivity_analysis import (
    compute_sensitivity_matrix,
    create_standard_sensitivity_scenarios,
    identify_critical_assumptions
)

print("=" * 80)
print("FINDING VIABLE SIGNAL FOR SENSITIVITY ANALYSIS")
print("=" * 80)

# Get all signals
all_signals = [s for s in list_signals() if s != 'mean_reversion']
print(f"\nTesting {len(all_signals)} signals to find viable candidate...")

# Test each signal
viable_signals = []

for signal_name in all_signals:
    try:
        print(f"\nTesting {signal_name}...")
        
        # Load data
        prices, volumes = load_price_data('SPY', datetime(2010, 1, 1), datetime(2020, 12, 31))
        forward_returns = compute_forward_returns(prices)
        volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Compute signal
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
        
        # Check if viable (break-even > 1%, capacity > $25M)
        break_even = tradability.break_even_cost or 0
        capacity = tradability.max_viable_capacity or 0
        
        print(f"  Break-Even: {break_even:.4f}")
        print(f"  Capacity: ${capacity/1e6:.2f}M")
        print(f"  Gross Sharpe: {tradability.gross_metrics.sharpe_ratio:.4f}" if tradability.gross_metrics else "  Gross Sharpe: None")
        
        if break_even > 0.01 and capacity > 25_000_000:
            print(f"  [VIABLE] Break-even > 1%, Capacity > $25M")
            viable_signals.append({
                'signal': signal_name,
                'break_even': break_even,
                'capacity': capacity,
                'gross_sharpe': tradability.gross_metrics.sharpe_ratio if tradability.gross_metrics else None
            })
        else:
            print(f"  [NOT VIABLE] Break-even = {break_even:.4f}, Capacity = ${capacity/1e6:.2f}M")
    
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("VIABLE SIGNALS FOUND")
print("=" * 80)

if len(viable_signals) > 0:
    print(f"\nFound {len(viable_signals)} viable signal(s):")
    for vs in viable_signals:
        print(f"  - {vs['signal']}: Break-Even = {vs['break_even']:.4f}, Capacity = ${vs['capacity']/1e6:.2f}M")
    
    # Run sensitivity analysis on first viable signal
    if len(viable_signals) > 0:
        test_signal = viable_signals[0]['signal']
        print(f"\n" + "=" * 80)
        print(f"RUNNING SENSITIVITY ANALYSIS ON: {test_signal}")
        print("=" * 80)
        
        base, scenarios = create_standard_sensitivity_scenarios()
        matrix = compute_sensitivity_matrix(
            signal_name=test_signal,
            ticker='SPY',
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2020, 12, 31),
            base_scenario=base,
            scenarios=scenarios
        )
        
        print("\nSensitivity Matrix:")
        print(matrix[['scenario', 'break_even_cost', 'decision', 'decision_flip']].to_string())
        
        # Identify critical assumptions
        critical = identify_critical_assumptions(matrix)
        if len(critical) > 0:
            print("\nCritical Assumptions:")
            print(critical.to_string())
        else:
            print("\nNo critical assumptions identified (signal is robust)")
else:
    print("\n[FAIL] NO VIABLE SIGNALS FOUND")
    print("All signals are rejected in base case.")
    print("Cannot test sensitivity on dead signals.")
    print("\nRecommendation: Lower thresholds or test on different time period.")

print("\n" + "=" * 80)
print("VIABLE SIGNAL SEARCH COMPLETE")
print("=" * 80)

