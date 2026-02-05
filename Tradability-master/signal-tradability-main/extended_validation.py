"""
Extended Out-of-Sample Validation (2015-2024)

10-year test period for more robust validation.
"""

import pandas as pd
from datetime import datetime
from out_of_sample_validation import run_out_of_sample_validation, compute_validation_metrics
from statistical_rigor import compute_sharpe_with_ci
from signals import list_signals
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from signals import get_signal
from decay_analysis import compute_returns

print("=" * 80)
print("EXTENDED OUT-OF-SAMPLE VALIDATION (2015-2024)")
print("=" * 80)

# Get all signals
all_signals = [s for s in list_signals() if s != 'mean_reversion']
print(f"\nTesting {len(all_signals)} signals over 10-year test period")

# Extended validation
validation_results = run_out_of_sample_validation(
    signal_names=all_signals,
    ticker='SPY',
    train_start=datetime(2000, 1, 1),
    train_end=datetime(2014, 12, 31),
    test_start=datetime(2015, 1, 1),
    test_end=datetime(2024, 12, 31),  # Extended to 2024
    rejection_thresholds={'break_even': 0.01, 'capacity': 25_000_000}
)

print("\n" + "=" * 80)
print("DETAILED RESULTS WITH CONFIDENCE INTERVALS")
print("=" * 80)

for idx, row in validation_results.iterrows():
    print(f"\n{row['signal']}:")
    print(f"  Train Decision: {row['train_decision']}")
    print(f"  Train Break-Even: {row['train_break_even']:.4f}")
    print(f"  Test Actual Sharpe: {row['test_actual_sharpe']:.4f}")
    print(f"  Test Actual Break-Even: {row['test_actual_break_even']:.4f}")
    print(f"  Correct: {row['correct']}")
    
    # Compute test period Sharpe with CI
    try:
        prices, _ = load_price_data('SPY', datetime(2015, 1, 1), datetime(2024, 12, 31))
        forward_returns = compute_forward_returns(prices)
        signal_def = get_signal(row['signal'])
        signal_values = signal_def.compute(prices, **signal_def.default_params())
        aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
        test_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
        
        test_sharpe, test_ci, test_se = compute_sharpe_with_ci(test_returns)
        if test_sharpe is not None:
            print(f"  Test Sharpe (10-year): {test_sharpe:.4f} [{test_ci[0]:.4f}, {test_ci[1]:.4f}], SE: {test_se:.4f}")
            print(f"  Test Period: 2015-2024 ({len(test_returns)} observations)")
    except Exception as e:
        print(f"  Error computing CI: {e}")

metrics = compute_validation_metrics(validation_results)
print(f"\n" + "=" * 80)
print("VALIDATION METRICS (10-Year Test Period)")
print("=" * 80)
print(f"  Total Signals: {metrics['total_signals']}")
print(f"  Correct: {metrics['correct']}")
print(f"  Accuracy: {metrics['accuracy']:.2%}")
print(f"  False Positive Rate: {metrics['false_positive_rate']:.2%}")
print(f"  False Negative Rate: {metrics['false_negative_rate']:.2%}")

print("\n" + "=" * 80)
print("EXTENDED VALIDATION COMPLETE")
print("=" * 80)

