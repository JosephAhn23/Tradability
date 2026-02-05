"""
Threshold Sensitivity Analysis: Prove thresholds aren't arbitrary.

Why 0.5 Sharpe? Why $25M capacity? Prove these aren't arbitrary.
"""

import numpy as np
import pandas as pd
from prove_selectivity import create_signal_spectrum, simulate_signal_verdict


def sweep_sharpe_threshold() -> pd.DataFrame:
    """
    Test what happens as you vary Sharpe threshold from 0.1 to 2.0.
    
    Expected: S-curve
    - At 0.1: Almost everything passes (useless)
    - At 2.0: Almost everything fails (too strict)
    - At ~0.5: Inflection point where quality signals separate
    """
    thresholds = np.arange(0.1, 2.05, 0.1)
    signals = create_signal_spectrum()
    
    # Use conservative costs (10 bps)
    cost_bps = 10.0
    vol = 0.15
    
    results = []
    for threshold in thresholds:
        pass_count = 0
        for signal in signals:
            annual_cost = signal.turnover * (cost_bps / 10000) * 2
            net_sharpe = signal.gross_sharpe - (annual_cost / vol)
            
            # Check against threshold
            passes = net_sharpe >= threshold and signal.turnover <= 3.0
            if passes:
                pass_count += 1
        
        results.append({
            'threshold': threshold,
            'pass_count': pass_count,
            'pass_rate': pass_count / len(signals)
        })
    
    return pd.DataFrame(results)


def find_optimal_sharpe_threshold(df: pd.DataFrame) -> float:
    """
    Find inflection point (maximum curvature) in the pass rate curve.
    This is where the threshold is most "informative" - separating good from bad.
    """
    # Calculate second derivative (curvature)
    df['first_deriv'] = df['pass_rate'].diff()
    df['second_deriv'] = df['first_deriv'].diff().abs()
    
    # Find threshold with maximum curvature (excluding edges)
    middle = df.iloc[2:-2]
    if len(middle) > 0:
        optimal_idx = middle['second_deriv'].idxmax()
        optimal_threshold = df.loc[optimal_idx, 'threshold']
    else:
        optimal_threshold = 0.5
    
    return optimal_threshold


def sweep_turnover_threshold() -> pd.DataFrame:
    """
    Test what happens as you vary turnover threshold from 0.5x to 10x.
    """
    thresholds = np.arange(0.5, 10.5, 0.5)
    signals = create_signal_spectrum()
    
    cost_bps = 10.0
    vol = 0.15
    sharpe_threshold = 0.5
    
    results = []
    for max_turnover in thresholds:
        pass_count = 0
        for signal in signals:
            annual_cost = signal.turnover * (cost_bps / 10000) * 2
            net_sharpe = signal.gross_sharpe - (annual_cost / vol)
            
            passes = net_sharpe >= sharpe_threshold and signal.turnover <= max_turnover
            if passes:
                pass_count += 1
        
        results.append({
            'max_turnover': max_turnover,
            'pass_count': pass_count,
            'pass_rate': pass_count / len(signals)
        })
    
    return pd.DataFrame(results)


def sweep_cost_assumption() -> pd.DataFrame:
    """
    Test what happens as you vary cost assumption from 1 bps to 100 bps.
    """
    cost_levels = np.concatenate([
        np.arange(1, 10, 1),
        np.arange(10, 50, 5),
        np.arange(50, 105, 10)
    ])
    
    signals = create_signal_spectrum()
    vol = 0.15
    sharpe_threshold = 0.5
    
    results = []
    for cost_bps in cost_levels:
        pass_count = 0
        for signal in signals:
            annual_cost = signal.turnover * (cost_bps / 10000) * 2
            net_sharpe = signal.gross_sharpe - (annual_cost / vol)
            
            passes = net_sharpe >= sharpe_threshold and signal.turnover <= 3.0
            if passes:
                pass_count += 1
        
        results.append({
            'cost_bps': cost_bps,
            'pass_count': pass_count,
            'pass_rate': pass_count / len(signals)
        })
    
    return pd.DataFrame(results)


def justify_thresholds():
    """
    Run all sensitivity analyses and determine if thresholds are data-driven.
    """
    print("=" * 70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # 1. Sharpe threshold sweep
    print("\n### 1. Sharpe Threshold Sensitivity ###\n")
    sharpe_df = sweep_sharpe_threshold()
    optimal_sharpe = find_optimal_sharpe_threshold(sharpe_df.copy())
    
    print(f"{'Threshold':<12} {'Pass Rate':<12}")
    print("-" * 24)
    for _, row in sharpe_df.iterrows():
        marker = " <-- optimal" if abs(row['threshold'] - optimal_sharpe) < 0.05 else ""
        marker2 = " <-- CHOSEN" if abs(row['threshold'] - 0.5) < 0.05 else ""
        print(f"{row['threshold']:<12.1f} {row['pass_rate']:<12.1%}{marker}{marker2}")
    
    sharpe_deviation = abs(optimal_sharpe - 0.5)
    sharpe_justified = "DATA_DRIVEN" if sharpe_deviation < 0.2 else "ARBITRARY"
    
    print(f"\nOptimal threshold from data: {optimal_sharpe:.2f}")
    print(f"Your chosen threshold: 0.50")
    print(f"Deviation: {sharpe_deviation:.2f}")
    print(f"Justification: {sharpe_justified}")
    
    # 2. Turnover threshold sweep
    print("\n### 2. Turnover Threshold Sensitivity ###\n")
    turnover_df = sweep_turnover_threshold()
    
    print(f"{'Max Turnover':<15} {'Pass Rate':<12}")
    print("-" * 27)
    for _, row in turnover_df.iterrows():
        marker = " <-- CHOSEN" if abs(row['max_turnover'] - 3.0) < 0.25 else ""
        print(f"{row['max_turnover']:<15.1f} {row['pass_rate']:<12.1%}{marker}")
    
    # Find where pass rate stabilizes (diminishing returns)
    turnover_df['marginal_gain'] = turnover_df['pass_rate'].diff()
    stabilization_idx = turnover_df[turnover_df['marginal_gain'] < 0.02].index
    if len(stabilization_idx) > 0:
        natural_turnover = turnover_df.loc[stabilization_idx[0], 'max_turnover']
    else:
        natural_turnover = 5.0
    
    print(f"\nNatural turnover threshold (marginal gain < 2%): {natural_turnover:.1f}x")
    print(f"Your chosen threshold: 3.0x")
    
    # 3. Cost assumption sensitivity
    print("\n### 3. Cost Assumption Sensitivity ###\n")
    cost_df = sweep_cost_assumption()
    
    print(f"{'Cost (bps)':<12} {'Pass Rate':<12}")
    print("-" * 24)
    for _, row in cost_df.iterrows():
        marker = ""
        if row['cost_bps'] == 10:
            marker = " <-- CONSERVATIVE"
        elif row['cost_bps'] == 2:
            marker = " <-- REALISTIC (SPY)"
        elif row['cost_bps'] == 50:
            marker = " <-- SABOTAGE"
        print(f"{row['cost_bps']:<12.0f} {row['pass_rate']:<12.1%}{marker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("THRESHOLD JUSTIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"""
    Sharpe threshold (0.5):
        Status: {sharpe_justified}
        Optimal from data: {optimal_sharpe:.2f}
        Industry standard: 0.5 for "investable", 0.3 for "interesting"
    
    Turnover threshold (3.0x):
        Status: {'CONSERVATIVE' if natural_turnover > 3.0 else 'DATA_DRIVEN'}
        Natural cutoff: {natural_turnover:.1f}x
        Rationale: Above 3x, costs dominate for most strategies
    
    Cost assumption (10 bps):
        Status: CONSERVATIVE
        SPY reality: 1.5 bps (6.5x lower)
        Rationale: Safety margin for volatility spikes, illiquid names
    """)
    
    # Save results
    sharpe_df.to_csv('threshold_sharpe_sensitivity.csv', index=False)
    turnover_df.to_csv('threshold_turnover_sensitivity.csv', index=False)
    cost_df.to_csv('threshold_cost_sensitivity.csv', index=False)
    
    return {
        'sharpe': {'chosen': 0.5, 'optimal': optimal_sharpe, 'status': sharpe_justified},
        'turnover': {'chosen': 3.0, 'natural': natural_turnover},
        'cost': {'chosen': 10.0, 'realistic': 1.5}
    }


if __name__ == "__main__":
    justify_thresholds()
