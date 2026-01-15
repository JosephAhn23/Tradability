"""
Comprehensive Validation Script

Runs all validation tests with detailed output:
1. Out-of-sample validation with confidence intervals
2. Regime analysis
3. Sensitivity analysis on viable signals
4. Stress tests
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from signals import list_signals
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from signals import get_signal
from decay_analysis import compute_returns
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from statistical_rigor import compute_sharpe_with_ci, compute_information_coefficient
from out_of_sample_validation import run_out_of_sample_validation, compute_validation_metrics
from sensitivity_analysis import compute_sensitivity_matrix, create_standard_sensitivity_scenarios, identify_critical_assumptions
from regime_analysis import partition_regimes, compute_regime_sharpe, compute_regime_sensitivity

print("=" * 80)
print("COMPREHENSIVE VALIDATION REPORT")
print("=" * 80)

# Get all signals
all_signals = [s for s in list_signals() if s != 'mean_reversion']
print(f"\nTesting {len(all_signals)} signals: {all_signals}")

# ============================================================================
# 1. OUT-OF-SAMPLE VALIDATION WITH CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "=" * 80)
print("1. OUT-OF-SAMPLE VALIDATION (2015-2020)")
print("=" * 80)

validation_results = run_out_of_sample_validation(
    signal_names=all_signals[:4],  # Test first 4
    ticker='SPY',
    train_start=datetime(2000, 1, 1),
    train_end=datetime(2014, 12, 31),
    test_start=datetime(2015, 1, 1),
    test_end=datetime(2020, 12, 31),
    rejection_thresholds={'break_even': 0.01, 'capacity': 25_000_000}
)

print("\nDetailed Results:")
for idx, row in validation_results.iterrows():
    print(f"\n{row['signal']}:")
    print(f"  Train Decision: {row['train_decision']}")
    print(f"  Train Break-Even: {row['train_break_even']:.4f}")
    print(f"  Test Actual Sharpe: {row['test_actual_sharpe']:.4f}")
    print(f"  Test Actual Break-Even: {row['test_actual_break_even']:.4f}")
    print(f"  Correct: {row['correct']}")
    
    # Compute test period Sharpe with CI
    try:
        prices, _ = load_price_data('SPY', datetime(2015, 1, 1), datetime(2020, 12, 31))
        forward_returns = compute_forward_returns(prices)
        signal_def = get_signal(row['signal'])
        signal_values = signal_def.compute(prices, **signal_def.default_params())
        aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
        test_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
        
        test_sharpe, test_ci, test_se = compute_sharpe_with_ci(test_returns)
        if test_sharpe is not None:
            print(f"  Test Sharpe with CI: {test_sharpe:.4f} [{test_ci[0]:.4f}, {test_ci[1]:.4f}], SE: {test_se:.4f}")
    except Exception as e:
        print(f"  Error computing CI: {e}")

metrics = compute_validation_metrics(validation_results)
print(f"\nValidation Metrics:")
print(f"  Accuracy: {metrics['accuracy']:.2%}")
print(f"  False Positive Rate: {metrics['false_positive_rate']:.2%}")
print(f"  False Negative Rate: {metrics['false_negative_rate']:.2%}")

# ============================================================================
# 2. REGIME ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. REGIME ANALYSIS")
print("=" * 80)

for signal_name in all_signals[:2]:  # Test first 2
    print(f"\n{signal_name}:")
    try:
        # Load full period data
        prices, volumes = load_price_data('SPY', datetime(2010, 1, 1), datetime(2020, 12, 31))
        forward_returns = compute_forward_returns(prices)
        volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Compute signal
        signal_def = get_signal(signal_name)
        signal_values = signal_def.compute(prices, **signal_def.default_params())
        aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
        strategy_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
        
        # Market returns (use SPY returns as proxy)
        market_returns = prices.pct_change()
        
        # Partition regimes
        regimes = partition_regimes(
            strategy_returns,
            market_returns=market_returns,
            volatility=volatility
        )
        
        # Compute Sharpe in each regime
        regime_results = compute_regime_sharpe(strategy_returns, regimes)
        print("  Regime Sharpe Results:")
        print(regime_results[['regime', 'sharpe', 'ci_lower', 'ci_upper', 'n_observations']].to_string())
        
        # Compute sensitivity
        sensitivity = compute_regime_sensitivity(regime_results)
        print(f"\n  Regime Sensitivity:")
        print(f"    Max Sharpe: {sensitivity['max_sharpe']:.4f}")
        print(f"    Min Sharpe: {sensitivity['min_sharpe']:.4f}")
        print(f"    Sensitivity Ratio: {sensitivity['sensitivity_ratio']:.2f}" if sensitivity['sensitivity_ratio'] else "    Sensitivity Ratio: N/A")
        print(f"    Sign Flip: {sensitivity['sign_flip']}")
        print(f"    Regime Fragile: {sensitivity['regime_fragile']}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 3. SENSITIVITY ANALYSIS ON VIABLE SIGNALS
# ============================================================================
print("\n" + "=" * 80)
print("3. SENSITIVITY ANALYSIS")
print("=" * 80)

base, scenarios = create_standard_sensitivity_scenarios()

# Find a signal that might be viable (or test all)
for signal_name in all_signals[:3]:
    print(f"\n{signal_name}:")
    try:
        matrix = compute_sensitivity_matrix(
            signal_name=signal_name,
            ticker='SPY',
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2020, 12, 31),
            base_scenario=base,
            scenarios=scenarios[:6]  # Test first 6 scenarios
        )
        
        # Show base case
        base_break_even = matrix.iloc[0]['break_even_cost'] if len(matrix) > 0 else 0
        base_decision = matrix.iloc[0]['decision'] if len(matrix) > 0 else 'REJECT'
        
        print(f"  Base Case: Break-Even = {base_break_even:.4f}, Decision = {base_decision}")
        
        # Count flips
        flips = matrix['decision_flip'].sum()
        print(f"  Decision Flips: {flips} / {len(matrix)} scenarios")
        
        if flips > 0:
            print("  Scenarios that cause flips:")
            flip_scenarios = matrix[matrix['decision_flip'] == True]
            for _, row in flip_scenarios.iterrows():
                print(f"    - {row['scenario']}: Break-Even = {row['break_even_cost']:.4f}")
            
            # Identify critical assumptions
            critical = identify_critical_assumptions(matrix)
            if len(critical) > 0:
                print("  Critical Assumptions:")
                for _, row in critical.iterrows():
                    print(f"    - {row['assumption']}: {row['change_pct']:.1f}% change causes flip")
        
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# 4. INFORMATION COEFFICIENT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. INFORMATION COEFFICIENT (IC) ANALYSIS")
print("=" * 80)

for signal_name in all_signals[:3]:
    print(f"\n{signal_name}:")
    try:
        prices, _ = load_price_data('SPY', datetime(2010, 1, 1), datetime(2020, 12, 31))
        forward_returns = compute_forward_returns(prices)
        signal_def = get_signal(signal_name)
        signal_values = signal_def.compute(prices, **signal_def.default_params())
        aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
        
        ic, p_value = compute_information_coefficient(aligned_signals, aligned_returns)
        print(f"  IC: {ic:.4f}, p-value: {p_value:.4f}")
        
        if ic is not None:
            if abs(ic) < 0.01:
                print(f"  [WARN] IC < 0.01: Economically meaningless")
            elif abs(ic) > 0.05:
                print(f"  [WARN] IC > 0.05: Suspiciously high (check for overfitting)")
            else:
                print(f"  [OK] IC in reasonable range")
        
        if p_value is not None:
            if p_value < 0.05:
                print(f"  [PASS] IC is statistically significant (p < 0.05)")
            else:
                print(f"  [FAIL] IC is not statistically significant (p >= 0.05)")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

