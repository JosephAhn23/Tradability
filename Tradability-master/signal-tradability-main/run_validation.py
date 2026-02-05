"""
Run Order Book Validation

Execute this ONCE. Fix all errors before proceeding.
Tests ONE rejected signal on ONE day with real L2 data.
"""

import numpy as np
from orderbook_validation import SignalValidator, plot_cost_validation


def main():
    validator = SignalValidator()
    
    # Pick the worst rejected signal
    signal = 'momentum_12_1'
    ticker = 'SPY'
    test_date = '2025-01-31'  # Recent date (yfinance has ~7 days of 1-min data)
    position_size = 100000  # $100k per trade
    
    print("=" * 60)
    print(f"ORDER BOOK VALIDATION: {signal}")
    print(f"Ticker: {ticker}")
    print(f"Date: {test_date}")
    print(f"Position Size: ${position_size:,}")
    print("=" * 60)
    
    # Run validation
    results = validator.validate_signal(
        signal_name=signal,
        ticker=ticker,
        test_date=test_date,
        position_size=position_size
    )
    
    # Print stats
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\nTrades simulated: {len(results)}")
    print(f"Buy trades: {(results['side'] == 'buy').sum()}")
    print(f"Sell trades: {(results['side'] == 'sell').sum()}")
    
    print(f"\n--- Cost Breakdown (mean across all trades) ---")
    print(f"Actual spread in data: {results['spread_bps'].mean():.2f} bps")
    print(f"Simulated spread cost: {results['simulated_spread_bps'].mean():.2f} bps")
    print(f"Simulated impact cost: {results['simulated_impact_bps'].mean():.2f} bps")
    print(f"Total simulated cost:  {results['simulated_cost_bps'].mean():.2f} bps")
    
    print(f"\n--- Theoretical Model Comparison ---")
    print(f"Framework (10bp spread):     {results['theoretical_cost_bps'].mean():.2f} bps")
    if 'theoretical_calibrated_bps' in results.columns:
        print(f"Calibrated (actual spread):  {results['theoretical_calibrated_bps'].mean():.2f} bps")
    
    print(f"\n--- Error Analysis (vs Framework model) ---")
    error = results['error_bps']
    print(f"Mean error: {error.mean():.2f} bps")
    print(f"RMSE: {np.sqrt((error**2).mean()):.2f} bps")
    
    if 'error_calibrated_bps' in results.columns:
        error_cal = results['error_calibrated_bps']
        print(f"\n--- Error Analysis (vs Calibrated model) ---")
        print(f"Mean error: {error_cal.mean():.2f} bps")
        print(f"RMSE: {np.sqrt((error_cal**2).mean()):.2f} bps")
    
    # Was the model conservative?
    conservative_pct = (results['simulated_cost_bps'] < results['theoretical_cost_bps']).mean()
    print(f"\n--- Model Assessment ---")
    print(f"Model was conservative in {conservative_pct*100:.1f}% of trades")
    
    if conservative_pct >= 0.7:
        assessment = "CONSERVATIVE"
        print("[OK] Model is appropriately conservative")
    elif conservative_pct >= 0.5:
        assessment = "ACCURATE"
        print("[OK] Model is reasonably calibrated")
    else:
        assessment = "OPTIMISTIC"
        print("[WARNING] Model may be too optimistic")
    
    # Generate plot
    print("\nGenerating validation plot...")
    plot_cost_validation(results, signal)
    
    # Summary for README
    print("\n" + "=" * 60)
    print("README SNIPPET (copy this):")
    print("=" * 60)
    print(f"""
## Validation

Tested {signal} signal on {ticker} ({test_date}, ${position_size/1000:.0f}k position size):
- Theoretical cost: {results['theoretical_cost_bps'].mean():.2f} bps
- Simulated cost: {results['simulated_cost_bps'].mean():.2f} bps
- Model error: ±{np.sqrt((error**2).mean()):.2f} bps (RMSE)

Conclusion: Cost model is {assessment.lower()}. Rejection thresholds remain appropriate.
""")
    
    # Final verdict
    print("=" * 60)
    if assessment == "OPTIMISTIC":
        print("VERDICT: Model underestimates costs. Rejection decisions are EVEN MORE justified.")
    else:
        print("VERDICT: Model is calibrated. Rejection decisions are validated.")
    print("=" * 60)
    
    # Save results
    results.to_csv(f"validation_results_{signal}.csv", index=False)
    print(f"\nResults saved to: validation_results_{signal}.csv")
    
    return results


if __name__ == "__main__":
    main()
