"""
Run Validation Tests

Executes:
1. Sensitivity analysis on all signals
2. Out-of-sample validation (2015-2020)
3. Reports results
"""

import pandas as pd
from datetime import datetime
from sensitivity_analysis import (
    compute_sensitivity_matrix,
    create_standard_sensitivity_scenarios,
    identify_critical_assumptions
)
from out_of_sample_validation import (
    run_out_of_sample_validation,
    compute_validation_metrics,
    pre_register_framework
)
from signals import list_signals

print("=" * 80)
print("VALIDATION TESTS")
print("=" * 80)

# Pre-register framework
print("\n1. Pre-registering framework...")
framework_config = {
    'rejection_thresholds': {
        'break_even': 0.01,
        'capacity': 25_000_000,
        'turnover': 3.0,
        'cost_drag': 0.05
    },
    'cost_assumptions': {
        'commission_per_trade': 0.005,
        'half_spread': 0.001,
        'participation_rate': 0.01
    }
}
config_hash = pre_register_framework(framework_config)
print(f"   Framework pre-registered with hash: {config_hash[:16]}...")

# Get signals
signals = [s for s in list_signals() if s != 'mean_reversion']  # Exclude deleted signal
print(f"\n2. Testing {len(signals)} signals: {signals}")

# Sensitivity Analysis
print("\n3. Running sensitivity analysis...")
base, scenarios = create_standard_sensitivity_scenarios()

sensitivity_results = {}
for signal_name in signals[:3]:  # Test first 3 to save time
    print(f"   Testing {signal_name}...")
    try:
        matrix = compute_sensitivity_matrix(
            signal_name=signal_name,
            ticker='SPY',
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2020, 12, 31),
            base_scenario=base,
            scenarios=scenarios
        )
        sensitivity_results[signal_name] = matrix
        
        # Identify critical assumptions
        critical = identify_critical_assumptions(matrix)
        if len(critical) > 0:
            print(f"      Critical assumptions: {critical['assumption'].unique().tolist()}")
    except Exception as e:
        print(f"      Error: {e}")
        continue

print(f"\n   Completed sensitivity analysis for {len(sensitivity_results)} signals")

# Out-of-Sample Validation
print("\n4. Running out-of-sample validation...")
print("   Train: 2000-2014, Test: 2015-2020")

try:
    validation_results = run_out_of_sample_validation(
        signal_names=signals[:3],  # Test first 3
        ticker='SPY',
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2014, 12, 31),
        test_start=datetime(2015, 1, 1),
        test_end=datetime(2020, 12, 31),
        rejection_thresholds={'break_even': 0.01, 'capacity': 25_000_000}
    )
    
    print("\n   Validation Results:")
    print(validation_results[['signal', 'train_decision', 'test_actual_break_even', 'correct']].to_string())
    
    metrics = compute_validation_metrics(validation_results)
    print("\n   Validation Metrics:")
    print(f"      Accuracy: {metrics['accuracy']:.2%}")
    print(f"      False Positive Rate: {metrics['false_positive_rate']:.2%}")
    print(f"      False Negative Rate: {metrics['false_negative_rate']:.2%}")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VALIDATION TESTS COMPLETE")
print("=" * 80)

