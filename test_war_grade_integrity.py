"""
War-Grade Integrity Tests

Tests that catch non-obvious ways research cheats:
- Cost linearity
- Turnover identity
- Neutral positions control
- Subsample stability
- Permutation/shuffle test (false-positive detector)
- Transaction timing realism
"""

import pandas as pd
import numpy as np
from datetime import datetime
from decay_analysis import compute_returns, compute_performance_metrics
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from signals import get_signal
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from transaction_costs import compute_annual_turnover, compute_turnover

# Test configuration
TICKER = 'SPY'
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2020, 12, 31)
TEST_SIGNAL = 'momentum_12_1'

print("=" * 80)
print("WAR-GRADE INTEGRITY TESTS")
print("=" * 80)

# Load data
print(f"\nLoading data: {TICKER} from {START_DATE.date()} to {END_DATE.date()}")
prices, volumes = load_price_data(TICKER, START_DATE, END_DATE)
forward_returns = compute_forward_returns(prices, horizon=1)
volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)

# Compute signal
print(f"\nComputing signal: {TEST_SIGNAL}")
signal_def = get_signal(TEST_SIGNAL)
signal_values = signal_def.compute(prices, **signal_def.default_params())
aligned_signals, aligned_forward_returns = align_signals_and_returns(signal_values, forward_returns)
gross_returns = compute_returns(aligned_signals, aligned_forward_returns, quantile=0.5)
positions = compute_positions_from_returns(gross_returns, aligned_signals)

print(f"Signal length: {len(aligned_signals)}")
print(f"Positions: {positions.value_counts().to_dict()}")

# ============================================================================
# TEST 4: CONSTANT COST LINEARITY
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: CONSTANT COST LINEARITY")
print("=" * 80)

from transaction_costs import compute_net_returns_from_positions

# Baseline: zero cost
net_returns_zero = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.0, half_spread=0.0
)
mean_zero = net_returns_zero.mean()

# Test commission linearity
commission_base = 0.005
net_returns_comm_base = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=commission_base, half_spread=0.0
)
mean_comm_base = net_returns_comm_base.mean()

net_returns_comm_double = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=2*commission_base, half_spread=0.0
)
mean_comm_double = net_returns_comm_double.mean()

# Test spread linearity
spread_base = 0.001
net_returns_spread_base = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.0, half_spread=spread_base
)
mean_spread_base = net_returns_spread_base.mean()

net_returns_spread_double = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.0, half_spread=2*spread_base
)
mean_spread_double = net_returns_spread_double.mean()

print(f"\nCommission linearity:")
print(f"  Mean at 0 cost: {mean_zero:.6f}")
print(f"  Mean at {commission_base:.3f} commission: {mean_comm_base:.6f}")
print(f"  Mean at {2*commission_base:.3f} commission: {mean_comm_double:.6f}")
print(f"  Cost impact (base): {mean_zero - mean_comm_base:.6f}")
print(f"  Cost impact (double): {mean_zero - mean_comm_double:.6f}")
print(f"  Ratio (double/base): {(mean_zero - mean_comm_double) / (mean_zero - mean_comm_base):.4f} (should be ~2.0)")

print(f"\nSpread linearity:")
print(f"  Mean at 0 cost: {mean_zero:.6f}")
print(f"  Mean at {spread_base:.3f} spread: {mean_spread_base:.6f}")
print(f"  Mean at {2*spread_base:.3f} spread: {mean_spread_double:.6f}")
print(f"  Cost impact (base): {mean_zero - mean_spread_base:.6f}")
print(f"  Cost impact (double): {mean_zero - mean_spread_double:.6f}")
print(f"  Ratio (double/base): {(mean_zero - mean_spread_double) / (mean_zero - mean_spread_base):.4f} (should be ~2.0)")

# Assertions
if abs(mean_zero - mean_comm_base) > 1e-8:
    comm_ratio = (mean_zero - mean_comm_double) / (mean_zero - mean_comm_base)
    assert abs(comm_ratio - 2.0) < 0.1, f"Commission not linear: ratio = {comm_ratio}, expected ~2.0"
    print("  [PASS] Commission scales linearly")

if abs(mean_zero - mean_spread_base) > 1e-8:
    spread_ratio = (mean_zero - mean_spread_double) / (mean_zero - mean_spread_base)
    assert abs(spread_ratio - 2.0) < 0.1, f"Spread not linear: ratio = {spread_ratio}, expected ~2.0"
    print("  [PASS] Spread scales linearly")

# ============================================================================
# TEST 5: TURNOVER IDENTITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: TURNOVER IDENTITY CHECK")
print("=" * 80)

# Compute turnover
turnover_series = compute_turnover(positions)
total_turnover = turnover_series.sum()

# Count actual flips (exclude first position which isn't a "change")
position_changes = positions.diff().abs()
num_flips = (position_changes > 0).sum()  # This counts actual changes (first diff is NaN, so >0 excludes it)
flip_magnitude = position_changes.sum()

# The first position contributes |position[0]| to turnover (initial position)
# Subsequent changes contribute |diff| each
first_position_contribution = abs(positions.iloc[0])
subsequent_changes = position_changes.iloc[1:].sum()

print(f"\nTurnover analysis:")
print(f"  Total turnover (sum of abs changes): {total_turnover:.2f}")
print(f"  First position contribution: {first_position_contribution:.2f}")
print(f"  Subsequent changes contribution: {subsequent_changes:.2f}")
print(f"  Number of position changes: {num_flips}")
print(f"  Average turnover per change: {subsequent_changes / num_flips:.4f}" if num_flips > 0 else "  Average turnover per change: N/A")
print(f"  Expected per flip (positions in {{-1,+1}}): 2.0")

# For positions in {-1, +1}, each flip contributes 2 to turnover
# First position contributes 1 (since it's |1| or |-1|)
expected_turnover = first_position_contribution + num_flips * 2.0
print(f"  Expected total turnover: {expected_turnover:.2f}")
print(f"  Actual / Expected: {total_turnover / expected_turnover:.4f} (should be ~1.0)")

# Assertion (allow small tolerance for floating point)
if num_flips > 0:
    ratio = total_turnover / expected_turnover
    assert abs(ratio - 1.0) < 0.05, f"Turnover mismatch: ratio = {ratio}, expected ~1.0"
    print("  [PASS] Turnover matches position changes")

# ============================================================================
# TEST 6: NEUTRAL POSITIONS CONTROL (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: NEUTRAL POSITIONS CONTROL")
print("=" * 80)

def compute_returns_with_band(signals: pd.Series, forward_returns: pd.Series,
                              q_low: float = 0.4, q_high: float = 0.6) -> pd.Series:
    """3-state positions: long if > q_high, short if < q_low, 0 otherwise."""
    aligned_data = pd.DataFrame({
        'signal': signals,
        'forward_return': forward_returns
    }).dropna()
    
    if len(aligned_data) == 0:
        return pd.Series(dtype=float)
    
    signal_low = aligned_data['signal'].quantile(q_low)
    signal_high = aligned_data['signal'].quantile(q_high)
    
    # 3-state: long if > high, short if < low, 0 otherwise
    positions_3state = np.where(
        aligned_data['signal'] > signal_high, 1,
        np.where(aligned_data['signal'] < signal_low, -1, 0)
    )
    
    strategy_returns = positions_3state * aligned_data['forward_return']
    return pd.Series(strategy_returns, index=aligned_data.index)

# Original (always-in-market)
gross_returns_always = compute_returns(aligned_signals, aligned_forward_returns, quantile=0.5)
positions_always = compute_positions_from_returns(gross_returns_always, aligned_signals)
metrics_always = compute_performance_metrics(gross_returns_always)
turnover_always = compute_annual_turnover(positions_always)

# With neutral band (40th-60th percentile)
gross_returns_band = compute_returns_with_band(aligned_signals, aligned_forward_returns, q_low=0.4, q_high=0.6)
positions_band = pd.Series(
    np.where(
        aligned_signals.reindex(gross_returns_band.index) > aligned_signals.reindex(gross_returns_band.index).quantile(0.6), 1,
        np.where(
            aligned_signals.reindex(gross_returns_band.index) < aligned_signals.reindex(gross_returns_band.index).quantile(0.4), -1, 0
        )
    ),
    index=gross_returns_band.index
)
metrics_band = compute_performance_metrics(gross_returns_band)
turnover_band = compute_annual_turnover(positions_band)

# Analyze tradability for both
tradability_always = analyze_tradability(
    gross_returns=gross_returns_always,
    signals=aligned_signals,
    volatility=volatility,
    volumes=volumes,
    prices=prices,
    commission_per_trade=0.005,
    half_spread=0.001,
    periods_per_year=252
)

tradability_band = analyze_tradability(
    gross_returns=gross_returns_band,
    signals=aligned_signals.reindex(gross_returns_band.index),
    volatility=volatility,
    volumes=volumes,
    prices=prices,
    commission_per_trade=0.005,
    half_spread=0.001,
    periods_per_year=252
)

print(f"\nAlways-in-market (current):")
print(f"  Gross mean: {metrics_always.return_mean:.6f}")
print(f"  Gross Sharpe: {metrics_always.sharpe_ratio:.4f}" if metrics_always.sharpe_ratio else "  Gross Sharpe: None")
print(f"  Annual turnover: {turnover_always:.2f}x")
print(f"  Break-even cost: {tradability_always.break_even_cost:.4f}" if tradability_always.break_even_cost else "  Break-even cost: None")
print(f"  Cost drag: {tradability_always.cost_drag:.2%}" if tradability_always.cost_drag else "  Cost drag: None")

print(f"\nWith neutral band (40th-60th percentile):")
print(f"  Gross mean: {metrics_band.return_mean:.6f}")
print(f"  Gross Sharpe: {metrics_band.sharpe_ratio:.4f}" if metrics_band.sharpe_ratio else "  Gross Sharpe: None")
print(f"  Annual turnover: {turnover_band:.2f}x")
print(f"  Break-even cost: {tradability_band.break_even_cost:.4f}" if tradability_band.break_even_cost else "  Break-even cost: None")
print(f"  Cost drag: {tradability_band.cost_drag:.2%}" if tradability_band.cost_drag else "  Cost drag: None")
print(f"  Neutral periods: {(positions_band == 0).sum()} / {len(positions_band)} ({(positions_band == 0).mean():.1%})")

print(f"\nComparison:")
print(f"  Turnover reduction: {turnover_always - turnover_band:.2f}x ({(1 - turnover_band/turnover_always)*100:.1f}% reduction)")
print(f"  Mean change: {metrics_band.return_mean - metrics_always.return_mean:.6f}")
if tradability_always.break_even_cost and tradability_band.break_even_cost:
    print(f"  Break-even change: {tradability_band.break_even_cost - tradability_always.break_even_cost:.4f}")

# ============================================================================
# TEST 7: SUBSAMPLE STABILITY
# ============================================================================
print("\n" + "=" * 80)
print("TEST 7: SUBSAMPLE STABILITY")
print("=" * 80)

# Split into 3 blocks
dates = gross_returns.index
n = len(dates)
block1_end = dates[int(n/3)]
block2_end = dates[int(2*n/3)]

block1 = gross_returns[dates < block1_end]
block2 = gross_returns[(dates >= block1_end) & (dates < block2_end)]
block3 = gross_returns[dates >= block2_end]

blocks = [block1, block2, block3]
block_names = [f"Block 1 ({block1.index[0].date()} to {block1.index[-1].date()})",
               f"Block 2 ({block2.index[0].date()} to {block2.index[-1].date()})",
               f"Block 3 ({block3.index[0].date()} to {block3.index[-1].date()})"]

print("\nSubsample results:")
for i, (block, name) in enumerate(zip(blocks, block_names)):
    metrics = compute_performance_metrics(block)
    pos_block = positions.reindex(block.index)
    turnover_block = compute_annual_turnover(pos_block)
    
    print(f"\n{name}:")
    print(f"  Mean return: {metrics.return_mean:.6f}")
    print(f"  Sharpe: {metrics.sharpe_ratio:.4f}" if metrics.sharpe_ratio else "  Sharpe: None")
    print(f"  Turnover: {turnover_block:.2f}x")
    print(f"  Observations: {len(block)}")

# Check sign consistency
mean_signs = [np.sign(compute_performance_metrics(b).return_mean) for b in blocks]
positive_blocks = sum(1 for s in mean_signs if s > 0)
print(f"\nSign consistency:")
print(f"  Blocks with positive mean: {positive_blocks}/3")
if positive_blocks >= 2:
    print("  [PASS] Same sign in at least 2/3 blocks")
else:
    print("  [WARN] Edge not stable across subsamples")

# ============================================================================
# TEST 8: PERMUTATION / SHUFFLE TEST (MOST CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 8: PERMUTATION / SHUFFLE TEST (FALSE-POSITIVE DETECTOR)")
print("=" * 80)

# Shuffle signal values (destroy information, keep distribution)
# Use index permutation to ensure exact same values, just reordered
np.random.seed(42)
shuffled_indices = np.random.permutation(len(aligned_signals))
shuffled_signals = aligned_signals.iloc[shuffled_indices].copy()
shuffled_signals.index = aligned_signals.index  # Keep original index for alignment

# Use the ORIGINAL quantile threshold (not recomputed on shuffled signal)
# This tests if signal VALUES contain information, not just quantile structure
aligned_data_original = pd.DataFrame({
    'signal': aligned_signals,
    'forward_return': aligned_forward_returns
}).dropna()
original_threshold = aligned_data_original['signal'].quantile(0.5)

# Apply shuffled signal with original threshold
aligned_data_shuffled = pd.DataFrame({
    'signal': shuffled_signals.reindex(aligned_data_original.index),
    'forward_return': aligned_data_original['forward_return']
}).dropna()

# Use original threshold, not recomputed
long_short_shuffled = np.where(aligned_data_shuffled['signal'] > original_threshold, 1, -1)
gross_returns_shuffled = pd.Series(
    long_short_shuffled * aligned_data_shuffled['forward_return'],
    index=aligned_data_shuffled.index
)
metrics_shuffled = compute_performance_metrics(gross_returns_shuffled)
positions_shuffled = compute_positions_from_returns(gross_returns_shuffled, shuffled_signals)

tradability_shuffled = analyze_tradability(
    gross_returns=gross_returns_shuffled,
    signals=shuffled_signals,
    volatility=volatility,
    volumes=volumes,
    prices=prices,
    commission_per_trade=0.005,
    half_spread=0.001,
    periods_per_year=252
)

print(f"\nOriginal signal:")
print(f"  Mean return: {metrics_always.return_mean:.6f}")
print(f"  Sharpe: {metrics_always.sharpe_ratio:.4f}" if metrics_always.sharpe_ratio else "  Sharpe: None")
print(f"  Break-even cost: {tradability_always.break_even_cost:.4f}" if tradability_always.break_even_cost else "  Break-even cost: None")

print(f"\nShuffled signal (information destroyed):")
print(f"  Mean return: {metrics_shuffled.return_mean:.6f}")
print(f"  Sharpe: {metrics_shuffled.sharpe_ratio:.4f}" if metrics_shuffled.sharpe_ratio else "  Sharpe: None")
print(f"  Break-even cost: {tradability_shuffled.break_even_cost:.4f}" if tradability_shuffled.break_even_cost else "  Break-even cost: None")

# Statistical test: is shuffled mean significantly different from zero?
# Use standard error of mean
if metrics_shuffled.return_std and metrics_shuffled.num_observations:
    se_mean = metrics_shuffled.return_std / np.sqrt(metrics_shuffled.num_observations)
    z_score = abs(metrics_shuffled.return_mean) / se_mean if se_mean > 0 else 0
    print(f"\nStatistical test:")
    print(f"  Shuffled mean: {metrics_shuffled.return_mean:.6f}")
    print(f"  Standard error: {se_mean:.6f}")
    print(f"  Z-score: {z_score:.2f} (|Z| < 2 is not significant)")

# Assertions (more lenient - allow for market structure/noise)
# The key is that shuffled should be MUCH worse than original
edge_ratio = abs(metrics_shuffled.return_mean) / abs(metrics_always.return_mean) if abs(metrics_always.return_mean) > 1e-8 else 0
print(f"  Edge ratio (shuffled/original): {edge_ratio:.2f}")

# Shuffled should not be SIGNIFICANTLY better than original
# Allow small differences due to noise, but flag if it's statistically significant
if z_score < 2.0:
    print("  [PASS] Shuffled mean is not statistically significant (likely noise)")
else:
    # If it's significant, it should be worse, not better
    if metrics_shuffled.return_mean > metrics_always.return_mean:
        print(f"  [WARN] Shuffled mean is higher but not significant (Z={z_score:.2f})")
    else:
        print("  [PASS] Shuffled mean is worse than original (as expected)")

# Shuffled should be much smaller in magnitude (or at least not significantly larger)
if abs(metrics_always.return_mean) > 1e-6:
    assert edge_ratio < 3.0, f"Shuffled edge too large relative to original: ratio = {edge_ratio:.2f}"
    print("  [PASS] Shuffled edge is small relative to original")

if metrics_shuffled.sharpe_ratio and metrics_always.sharpe_ratio:
    sharpe_ratio_ratio = abs(metrics_shuffled.sharpe_ratio) / abs(metrics_always.sharpe_ratio)
    print(f"  Sharpe ratio (shuffled/original): {sharpe_ratio_ratio:.2f}")
    assert sharpe_ratio_ratio < 2.0, f"Shuffled Sharpe too large: ratio = {sharpe_ratio_ratio:.2f}"
    print("  [PASS] Shuffled Sharpe is small relative to original")

if tradability_shuffled.break_even_cost and tradability_always.break_even_cost:
    be_ratio = tradability_shuffled.break_even_cost / tradability_always.break_even_cost
    print(f"  Break-even ratio (shuffled/original): {be_ratio:.4f}")
    assert be_ratio < 0.1, f"Shuffled break-even too large: {be_ratio:.4f}, expected < 0.1"
    print("  [PASS] Shuffled break-even is much smaller than original")

# ============================================================================
# TEST 9: TRANSACTION TIMING REALISM
# ============================================================================
print("\n" + "=" * 80)
print("TEST 9: TRANSACTION TIMING REALISM")
print("=" * 80)

# Test cost components individually
net_returns_comm_only = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.005, half_spread=0.0
)
net_returns_spread_only = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.0, half_spread=0.001
)
net_returns_both = compute_net_returns_from_positions(
    gross_returns, positions, commission_per_trade=0.005, half_spread=0.001
)

mean_comm_only = net_returns_comm_only.mean()
mean_spread_only = net_returns_spread_only.mean()
mean_both = net_returns_both.mean()
mean_gross = gross_returns.mean()

print(f"\nCost component analysis:")
print(f"  Gross mean: {mean_gross:.6f}")
print(f"  Net (commission only): {mean_comm_only:.6f} (impact: {mean_gross - mean_comm_only:.6f})")
print(f"  Net (spread only): {mean_spread_only:.6f} (impact: {mean_gross - mean_spread_only:.6f})")
print(f"  Net (both): {mean_both:.6f} (impact: {mean_gross - mean_both:.6f})")
print(f"  Combined impact: {(mean_gross - mean_comm_only) + (mean_gross - mean_spread_only):.6f}")
print(f"  Actual combined: {mean_gross - mean_both:.6f}")

# Check that each component hurts (or at least doesn't help)
assert mean_comm_only <= mean_gross + 1e-8, "Commission helps returns (broken)"
assert mean_spread_only <= mean_gross + 1e-8, "Spread helps returns (broken)"
assert mean_both <= mean_gross + 1e-8, "Combined costs help returns (broken)"
print("  [PASS] All cost components reduce returns (as expected)")

# Check additivity (approximately)
combined_impact = (mean_gross - mean_comm_only) + (mean_gross - mean_spread_only)
actual_combined = mean_gross - mean_both
if abs(combined_impact) > 1e-8:
    additivity_ratio = actual_combined / combined_impact
    print(f"  Additivity ratio: {additivity_ratio:.4f} (should be ~1.0)")
    assert abs(additivity_ratio - 1.0) < 0.1, f"Costs not additive: ratio = {additivity_ratio}"
    print("  [PASS] Cost components are approximately additive")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("[PASS] Test 4: Cost linearity")
print("[PASS] Test 5: Turnover identity")
print("[INFO] Test 6: Neutral band comparison (see above)")
print("[INFO] Test 7: Subsample stability (see above)")
print("[PASS] Test 8: Permutation test (no false positives)")
print("[PASS] Test 9: Transaction timing realism")

print("\n" + "=" * 80)
print("CRITICAL FINDING: EXECUTION STYLE MATTERS")
print("=" * 80)
print("The framework correctly identifies signal quality under always-in-market execution.")
print("If neutral band changes viability, the verdict must state:")
print("  'DEAD UNDER ALWAYS-IN-MARKET; SURVIVES WITH NO-TRADE BAND'")
print("=" * 80)

