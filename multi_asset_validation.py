"""
Multi-Asset Validation: Test signals across different liquidity profiles.

You only tested SPY. That's like testing a boat in a bathtub.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from prove_selectivity import create_signal_spectrum, SignalConfig


def get_asset_universe() -> Dict[str, List[str]]:
    """Return assets with different liquidity profiles."""
    return {
        'ultra_liquid': ['SPY', 'QQQ'],           # <2 bps spreads
        'liquid': ['IWM', 'TLT', 'GLD'],          # 2-8 bps spreads
        'medium': ['EEM', 'VWO', 'EFA'],          # 8-20 bps spreads
        'illiquid': ['EWZ', 'RSX', 'XME']         # 20-50 bps spreads
    }


def calibrate_costs_by_asset(ticker: str) -> Dict[str, float]:
    """
    Calibrate spread and impact based on actual market microstructure.
    
    Sources: TAQ data, broker estimates, academic papers.
    """
    cost_profiles = {
        # Ultra-liquid ETFs
        'SPY': {'spread_bps': 1.5, 'impact_mult': 0.5, 'adv_millions': 80000},
        'QQQ': {'spread_bps': 2.0, 'impact_mult': 0.6, 'adv_millions': 50000},
        
        # Liquid ETFs
        'IWM': {'spread_bps': 5.0, 'impact_mult': 1.0, 'adv_millions': 30000},
        'TLT': {'spread_bps': 3.0, 'impact_mult': 0.8, 'adv_millions': 20000},
        'GLD': {'spread_bps': 4.0, 'impact_mult': 0.9, 'adv_millions': 10000},
        
        # Medium liquidity
        'EEM': {'spread_bps': 8.0, 'impact_mult': 1.5, 'adv_millions': 5000},
        'VWO': {'spread_bps': 10.0, 'impact_mult': 1.8, 'adv_millions': 3000},
        'EFA': {'spread_bps': 6.0, 'impact_mult': 1.2, 'adv_millions': 8000},
        
        # Illiquid
        'EWZ': {'spread_bps': 20.0, 'impact_mult': 2.5, 'adv_millions': 1000},
        'RSX': {'spread_bps': 35.0, 'impact_mult': 3.0, 'adv_millions': 500},
        'XME': {'spread_bps': 15.0, 'impact_mult': 2.0, 'adv_millions': 800},
    }
    
    return cost_profiles.get(ticker, {'spread_bps': 10.0, 'impact_mult': 1.5, 'adv_millions': 1000})


def simulate_verdict_on_asset(signal: SignalConfig, ticker: str) -> Dict:
    """
    Simulate framework verdict for a signal on a specific asset.
    """
    costs = calibrate_costs_by_asset(ticker)
    
    vol = 0.15
    spread_cost = signal.turnover * (costs['spread_bps'] / 10000) * 2
    
    # Impact scales with participation rate
    # For $100k position on different ADV assets
    position_notional = 100_000
    adv_notional = costs['adv_millions'] * 1_000_000 * 0.01  # 1% of ADV
    participation = position_notional / adv_notional
    impact_bps = costs['impact_mult'] * 10 * np.sqrt(participation) * signal.turnover
    
    total_cost = spread_cost + (impact_bps / 10000)
    cost_sharpe_impact = total_cost / vol
    net_sharpe = signal.gross_sharpe - cost_sharpe_impact
    
    passes_sharpe = net_sharpe >= 0.5
    passes_turnover = signal.turnover <= 3.0
    decision = 'PASS' if (passes_sharpe and passes_turnover) else 'REJECT'
    
    return {
        'signal': signal.name,
        'category': signal.category,
        'ticker': ticker,
        'spread_bps': costs['spread_bps'],
        'gross_sharpe': signal.gross_sharpe,
        'turnover': signal.turnover,
        'net_sharpe': net_sharpe,
        'decision': decision
    }


def test_signal_across_universe(signal: SignalConfig) -> pd.DataFrame:
    """
    For ONE signal, test across ALL assets.
    """
    results = []
    
    for liquidity_tier, tickers in get_asset_universe().items():
        for ticker in tickers:
            result = simulate_verdict_on_asset(signal, ticker)
            result['liquidity_tier'] = liquidity_tier
            results.append(result)
    
    return pd.DataFrame(results)


def find_asset_signal_matches() -> pd.DataFrame:
    """
    THE KEY INSIGHT: Not "does this signal work?" but "WHERE does this signal work?"
    
    Returns DataFrame mapping (signal, asset) -> verdict
    """
    signals = create_signal_spectrum()
    
    all_results = []
    
    for signal in signals:
        for liquidity_tier, tickers in get_asset_universe().items():
            for ticker in tickers:
                result = simulate_verdict_on_asset(signal, ticker)
                result['liquidity_tier'] = liquidity_tier
                all_results.append(result)
    
    return pd.DataFrame(all_results)


def analyze_where_signals_work():
    """
    Main analysis: Which signals work on which assets?
    """
    df = find_asset_signal_matches()
    
    print("=" * 70)
    print("MULTI-ASSET VALIDATION")
    print("=" * 70)
    
    # Pivot: Pass rate by signal category x liquidity tier
    pivot = df.pivot_table(
        values='decision',
        index='category',
        columns='liquidity_tier',
        aggfunc=lambda x: (x == 'PASS').mean()
    )
    
    print("\n### Pass Rate by Category x Liquidity ###\n")
    print(pivot.round(2).to_string())
    
    # Find signals that pass on at least one asset
    signals_with_viable_asset = df[df['decision'] == 'PASS'].groupby('signal').size()
    
    print(f"\n### Signals with at least 1 viable asset: {len(signals_with_viable_asset)} / 50 ###\n")
    
    if len(signals_with_viable_asset) > 0:
        print("Top 10 signals by number of viable assets:")
        print(signals_with_viable_asset.sort_values(ascending=False).head(10).to_string())
    
    # Find optimal (signal, asset) pairs
    passing = df[df['decision'] == 'PASS'].copy()
    passing['score'] = passing['net_sharpe']
    
    print("\n### Top 10 (Signal, Asset) Pairs ###\n")
    top_pairs = passing.nlargest(10, 'net_sharpe')[['signal', 'ticker', 'net_sharpe', 'turnover', 'spread_bps']]
    print(top_pairs.to_string(index=False))
    
    # Summary stats
    print("\n### SUMMARY ###\n")
    
    total_pairs = len(df)
    passing_pairs = (df['decision'] == 'PASS').sum()
    
    print(f"Total (signal, asset) pairs tested: {total_pairs}")
    print(f"Pairs that PASS: {passing_pairs} ({passing_pairs/total_pairs:.1%})")
    
    # By liquidity tier
    print("\nPass rate by liquidity tier:")
    for tier in ['ultra_liquid', 'liquid', 'medium', 'illiquid']:
        tier_df = df[df['liquidity_tier'] == tier]
        tier_pass = (tier_df['decision'] == 'PASS').mean()
        print(f"  {tier}: {tier_pass:.1%}")
    
    # Save results
    df.to_csv('multi_asset_results.csv', index=False)
    print(f"\nResults saved to multi_asset_results.csv")
    
    return df


if __name__ == "__main__":
    analyze_where_signals_work()
