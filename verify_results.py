"""Verify validation results independently."""
import pandas as pd
import numpy as np

df = pd.read_csv('validation_results_momentum_12_1.csv')

print("=" * 60)
print("INDEPENDENT VERIFICATION")
print("=" * 60)

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

theo_mean = df['theoretical_cost_bps'].mean()
sim_mean = df['simulated_cost_bps'].mean()
error_mean = df['error_bps'].mean()
rmse = np.sqrt((df['error_bps']**2).mean())

print(f"\nTheoretical cost (mean): {theo_mean:.2f} bps")
print(f"Simulated cost (mean): {sim_mean:.2f} bps")
print(f"Error (mean): {error_mean:.2f} bps")
print(f"RMSE: {rmse:.2f} bps")

conservative = (df['simulated_cost_bps'] < df['theoretical_cost_bps']).sum()
conservative_pct = conservative / len(df) * 100
print(f"\nTrades where simulated < theoretical: {conservative}/{len(df)}")
print(f"Conservative percentage: {conservative_pct:.1f}%")

margin = theo_mean / sim_mean
print(f"Safety margin: {margin:.2f}x")

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print(f"All theoretical costs positive: {(df['theoretical_cost_bps'] > 0).all()}")
print(f"All simulated costs positive: {(df['simulated_cost_bps'] > 0).all()}")
print(f"No NaN values: {df.isnull().sum().sum() == 0}")

print("\n" + "=" * 60)
print("SAMPLE TRADES (first 5)")
print("=" * 60)
print(df[['theoretical_cost_bps', 'simulated_cost_bps', 'simulated_spread_bps', 'simulated_impact_bps']].head().to_string())

print("\n" + "=" * 60)
print("COST BREAKDOWN AUDIT (Trade 1)")
print("=" * 60)
t = df.iloc[0]
print(f"Spread cost: {t['simulated_spread_bps']:.2f} bps")
print(f"Impact cost: {t['simulated_impact_bps']:.2f} bps")
print(f"Total simulated: {t['simulated_cost_bps']:.2f} bps")
print(f"Sum check: {t['simulated_spread_bps'] + t['simulated_impact_bps']:.2f} bps")
print(f"Match: {abs(t['simulated_cost_bps'] - (t['simulated_spread_bps'] + t['simulated_impact_bps'])) < 0.01}")

print("\n" + "=" * 60)
print("DISTRIBUTION")
print("=" * 60)
print(f"Theoretical - min: {df['theoretical_cost_bps'].min():.2f}, max: {df['theoretical_cost_bps'].max():.2f}")
print(f"Simulated - min: {df['simulated_cost_bps'].min():.2f}, max: {df['simulated_cost_bps'].max():.2f}")

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)
checks = [
    ("Rows = 676", len(df) == 676),
    ("Theo mean ~ 10 bps", 9.5 < theo_mean < 10.5),
    ("Sim mean ~ 1.5 bps", 1.0 < sim_mean < 2.0),
    ("RMSE ~ 8.5 bps", 8.0 < rmse < 9.0),
    ("100% conservative", conservative_pct == 100.0),
    ("Margin ~ 6.5x", 6.0 < margin < 7.0),
]

all_pass = True
for name, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"[{status}] {name}")

print("\n" + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
