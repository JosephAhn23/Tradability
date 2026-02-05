"""
THE CRITICAL TEST: Does the framework have a "sweet spot" where it's selective but not nihilistic?

Build a spectrum of signals from obviously-bad to obviously-good and prove the framework can distinguish.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SignalConfig:
    name: str
    category: str
    gross_sharpe: float
    turnover: float
    description: str


def create_signal_spectrum() -> List[SignalConfig]:
    """
    Generate 50 signals across quality spectrum:
    - 10 "garbage" signals (random noise, should fail)
    - 20 "mediocre" signals (weak edge, high turnover, should fail)
    - 15 "borderline" signals (some edge, moderate costs, sensitivity test)
    - 5 "robust" signals (strong edge, low turnover, should pass even at high costs)
    """
    signals = []
    
    # GARBAGE TIER: Should fail even at zero costs (negative/zero Sharpe)
    garbage = [
        SignalConfig('random_noise_1', 'garbage', 0.0, 2.0, 'Pure random'),
        SignalConfig('random_noise_2', 'garbage', 0.0, 3.0, 'Pure random high turnover'),
        SignalConfig('reverse_momentum', 'garbage', -0.3, 2.5, 'Buy losers'),
        SignalConfig('buy_high_vol', 'garbage', -0.2, 3.0, 'Lottery stocks'),
        SignalConfig('chase_news', 'garbage', -0.1, 4.0, 'News chasing'),
        SignalConfig('moon_phase', 'garbage', 0.0, 1.5, 'Astrology trading'),
        SignalConfig('friday_effect', 'garbage', 0.05, 2.0, 'Calendar anomaly'),
        SignalConfig('twitter_sentiment', 'garbage', 0.1, 5.0, 'Social media'),
        SignalConfig('round_number', 'garbage', 0.0, 3.0, 'Round price levels'),
        SignalConfig('contrarian_extreme', 'garbage', -0.15, 4.0, 'Buy crashes'),
    ]
    signals.extend(garbage)
    
    # MEDIOCRE TIER: Positive Sharpe but dies from costs
    mediocre = [
        SignalConfig('momentum_12_1', 'mediocre', 0.4, 4.2, 'Classic momentum'),
        SignalConfig('momentum_6_1', 'mediocre', 0.35, 5.0, 'Short momentum'),
        SignalConfig('momentum_3_1', 'mediocre', 0.3, 6.0, 'Very short momentum'),
        SignalConfig('ma_crossover_20_50', 'mediocre', 0.3, 3.5, 'Moving average'),
        SignalConfig('ma_crossover_10_30', 'mediocre', 0.25, 4.5, 'Fast MA'),
        SignalConfig('rsi_reversal', 'mediocre', 0.35, 4.0, 'RSI mean reversion'),
        SignalConfig('bollinger_breakout', 'mediocre', 0.3, 3.8, 'Volatility breakout'),
        SignalConfig('volume_spike', 'mediocre', 0.25, 5.5, 'Volume trading'),
        SignalConfig('gap_fade', 'mediocre', 0.4, 6.0, 'Fade gaps'),
        SignalConfig('mean_reversion_5d', 'mediocre', 0.45, 4.5, 'Short MR'),
        SignalConfig('mean_reversion_10d', 'mediocre', 0.4, 3.5, 'Medium MR'),
        SignalConfig('breakout_20d', 'mediocre', 0.35, 3.0, 'Breakout'),
        SignalConfig('earnings_drift', 'mediocre', 0.5, 4.0, 'PEAD'),
        SignalConfig('sector_rotation', 'mediocre', 0.3, 2.5, 'Sector momentum'),
        SignalConfig('pairs_trading', 'mediocre', 0.4, 5.0, 'Stat arb'),
        SignalConfig('beta_timing', 'mediocre', 0.25, 3.0, 'Market timing'),
        SignalConfig('vix_timing', 'mediocre', 0.3, 2.8, 'Vol timing'),
        SignalConfig('yield_curve', 'mediocre', 0.35, 2.0, 'Macro signal'),
        SignalConfig('credit_spread', 'mediocre', 0.3, 1.8, 'Credit signal'),
        SignalConfig('momentum_factor', 'mediocre', 0.45, 3.5, 'Factor momentum'),
    ]
    signals.extend(mediocre)
    
    # BORDERLINE TIER: Should pass at realistic costs, fail at adversarial costs
    borderline = [
        SignalConfig('quality_quarterly', 'borderline', 0.55, 1.5, 'Quality rebal quarterly'),
        SignalConfig('value_pe_quarterly', 'borderline', 0.5, 1.2, 'Value quarterly'),
        SignalConfig('value_pb_quarterly', 'borderline', 0.45, 1.3, 'Book value'),
        SignalConfig('momentum_12_3', 'borderline', 0.5, 2.0, 'Momentum longer hold'),
        SignalConfig('momentum_12_6', 'borderline', 0.55, 1.5, 'Momentum 6mo hold'),
        SignalConfig('low_vol_quarterly', 'borderline', 0.5, 1.0, 'Low vol'),
        SignalConfig('dividend_yield', 'borderline', 0.45, 0.8, 'Dividend'),
        SignalConfig('profitability', 'borderline', 0.55, 1.2, 'Gross profit'),
        SignalConfig('investment', 'borderline', 0.5, 1.0, 'Asset growth'),
        SignalConfig('size_small', 'borderline', 0.4, 0.9, 'Size factor'),
        SignalConfig('quality_roe', 'borderline', 0.55, 1.4, 'ROE'),
        SignalConfig('quality_fcf', 'borderline', 0.6, 1.3, 'FCF yield'),
        SignalConfig('combo_value_mom', 'borderline', 0.55, 1.8, 'Value+momentum'),
        SignalConfig('combo_quality_value', 'borderline', 0.6, 1.2, 'Quality+value'),
        SignalConfig('carry_fx', 'borderline', 0.5, 1.5, 'FX carry'),
    ]
    signals.extend(borderline)
    
    # ROBUST TIER: Should pass even at SABOTAGE costs
    robust = [
        SignalConfig('value_annual', 'robust', 0.7, 0.4, 'Value annual rebal'),
        SignalConfig('quality_annual', 'robust', 0.75, 0.3, 'Quality annual'),
        SignalConfig('buyhold_value', 'robust', 0.65, 0.2, 'Buy and hold value'),
        SignalConfig('dividend_annual', 'robust', 0.6, 0.25, 'Dividend annual'),
        SignalConfig('multifactor_annual', 'robust', 0.8, 0.35, 'Multi-factor annual'),
    ]
    signals.extend(robust)
    
    return signals


def simulate_signal_verdict(signal: SignalConfig, cost_bps: float) -> Dict:
    """
    Simulate framework verdict for a signal at given cost level.
    
    Net Sharpe = Gross Sharpe - (turnover * cost_bps * 2 / 10000) / volatility
    Assuming 15% annual volatility.
    """
    vol = 0.15
    annual_cost_drag = signal.turnover * (cost_bps / 10000) * 2
    cost_sharpe_impact = annual_cost_drag / vol
    net_sharpe = signal.gross_sharpe - cost_sharpe_impact
    
    # Thresholds
    passes_sharpe = net_sharpe >= 0.5
    passes_turnover = signal.turnover <= 3.0
    
    decision = 'PASS' if (passes_sharpe and passes_turnover) else 'REJECT'
    
    return {
        'signal': signal.name,
        'category': signal.category,
        'gross_sharpe': signal.gross_sharpe,
        'turnover': signal.turnover,
        'cost_bps': cost_bps,
        'net_sharpe': net_sharpe,
        'decision': decision
    }


def test_framework_selectivity():
    """
    THE KEY TEST: Run all 50 signals at 3 cost levels.
    
    Expected outcome:
    - REALISTIC: ~25 signals pass (50%)
    - CONSERVATIVE: ~10 signals pass (20%)
    - SABOTAGE: ~5 signals pass (10%)
    """
    cost_scenarios = {
        'realistic': 1.5,      # SPY actual spread
        'conservative': 10.0,  # Framework default
        'sabotage': 50.0       # Adversarial
    }
    
    signals = create_signal_spectrum()
    
    all_results = []
    summary = {}
    
    for scenario_name, cost_bps in cost_scenarios.items():
        scenario_results = []
        for signal in signals:
            result = simulate_signal_verdict(signal, cost_bps)
            result['scenario'] = scenario_name
            scenario_results.append(result)
            all_results.append(result)
        
        df = pd.DataFrame(scenario_results)
        pass_count = (df['decision'] == 'PASS').sum()
        
        summary[scenario_name] = {
            'passed': pass_count,
            'total': len(signals),
            'pass_rate': pass_count / len(signals),
            'by_category': df.groupby('category')['decision'].apply(lambda x: (x == 'PASS').sum()).to_dict()
        }
    
    return summary, pd.DataFrame(all_results)


def run_selectivity_test():
    """Main test runner."""
    print("=" * 70)
    print("FRAMEWORK SELECTIVITY TEST")
    print("=" * 70)
    
    summary, results_df = test_framework_selectivity()
    
    print("\n### Pass Rates by Cost Scenario ###\n")
    print(f"{'Scenario':<15} {'Passed':<10} {'Total':<10} {'Pass Rate':<10}")
    print("-" * 45)
    
    for scenario, data in summary.items():
        print(f"{scenario:<15} {data['passed']:<10} {data['total']:<10} {data['pass_rate']:.1%}")
    
    print("\n### Pass Rates by Category (at Conservative costs) ###\n")
    conservative = summary['conservative']
    print(f"{'Category':<15} {'Passed':<10}")
    print("-" * 25)
    for cat, count in conservative['by_category'].items():
        print(f"{cat:<15} {count}")
    
    # VALIDATION
    print("\n### VALIDATION CHECKS ###\n")
    
    checks = []
    
    # Check 1: Gradient exists
    gradient_ok = summary['realistic']['pass_rate'] > summary['conservative']['pass_rate'] > summary['sabotage']['pass_rate']
    checks.append(('Cost sensitivity gradient', gradient_ok))
    
    # Check 2: Not too conservative
    not_too_strict = summary['realistic']['pass_rate'] > 0.3
    checks.append(('Pass rate > 30% at realistic costs', not_too_strict))
    
    # Check 3: Not too lenient
    not_too_lenient = summary['sabotage']['pass_rate'] < 0.2
    checks.append(('Pass rate < 20% at sabotage costs', not_too_lenient))
    
    # Check 4: Garbage rejected
    garbage_rejected = summary['realistic']['by_category'].get('garbage', 0) <= 2
    checks.append(('Garbage signals mostly rejected', garbage_rejected))
    
    # Check 5: Robust signals pass
    robust_pass = summary['sabotage']['by_category'].get('robust', 0) >= 3
    checks.append(('Robust signals survive sabotage', robust_pass))
    
    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"[{status}] {check_name}")
    
    print("\n" + "=" * 70)
    if all_pass:
        print("VERDICT: Framework is SELECTIVE (not nihilistic)")
    else:
        print("VERDICT: Framework needs calibration")
    print("=" * 70)
    
    # Save results
    results_df.to_csv('selectivity_results.csv', index=False)
    print(f"\nDetailed results saved to selectivity_results.csv")
    
    return summary, results_df


if __name__ == "__main__":
    run_selectivity_test()
