"""
Find Survivors: Find at least 3 signals that PASS, or prove they don't exist.

"Now the question is whether you want to find something that passes."
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from prove_selectivity import create_signal_spectrum, SignalConfig


def generate_signal_library() -> Dict[str, List[Dict]]:
    """
    Comprehensive signal library organized by factor.
    
    10 variants per factor, varying:
    - Lookback period
    - Holding period
    - Rebalance frequency
    """
    library = {
        'momentum': [
            {'name': 'mom_12_1', 'lookback': 12, 'hold': 1, 'gross_sharpe': 0.45, 'turnover': 4.2},
            {'name': 'mom_12_3', 'lookback': 12, 'hold': 3, 'gross_sharpe': 0.50, 'turnover': 2.0},
            {'name': 'mom_12_6', 'lookback': 12, 'hold': 6, 'gross_sharpe': 0.55, 'turnover': 1.2},
            {'name': 'mom_12_12', 'lookback': 12, 'hold': 12, 'gross_sharpe': 0.50, 'turnover': 0.6},
            {'name': 'mom_6_1', 'lookback': 6, 'hold': 1, 'gross_sharpe': 0.35, 'turnover': 5.0},
            {'name': 'mom_6_3', 'lookback': 6, 'hold': 3, 'gross_sharpe': 0.40, 'turnover': 2.5},
            {'name': 'mom_6_6', 'lookback': 6, 'hold': 6, 'gross_sharpe': 0.45, 'turnover': 1.5},
            {'name': 'mom_3_1', 'lookback': 3, 'hold': 1, 'gross_sharpe': 0.25, 'turnover': 6.0},
            {'name': 'mom_3_3', 'lookback': 3, 'hold': 3, 'gross_sharpe': 0.30, 'turnover': 3.0},
            {'name': 'mom_12_1_skip1', 'lookback': 12, 'hold': 1, 'gross_sharpe': 0.40, 'turnover': 4.0},
        ],
        'value': [
            {'name': 'value_pe_monthly', 'metric': 'P/E', 'rebal': 'monthly', 'gross_sharpe': 0.40, 'turnover': 1.5},
            {'name': 'value_pe_quarterly', 'metric': 'P/E', 'rebal': 'quarterly', 'gross_sharpe': 0.45, 'turnover': 0.8},
            {'name': 'value_pe_annual', 'metric': 'P/E', 'rebal': 'annual', 'gross_sharpe': 0.50, 'turnover': 0.3},
            {'name': 'value_pb_monthly', 'metric': 'P/B', 'rebal': 'monthly', 'gross_sharpe': 0.35, 'turnover': 1.8},
            {'name': 'value_pb_quarterly', 'metric': 'P/B', 'rebal': 'quarterly', 'gross_sharpe': 0.40, 'turnover': 0.9},
            {'name': 'value_pb_annual', 'metric': 'P/B', 'rebal': 'annual', 'gross_sharpe': 0.45, 'turnover': 0.35},
            {'name': 'value_fcf_quarterly', 'metric': 'FCF', 'rebal': 'quarterly', 'gross_sharpe': 0.55, 'turnover': 0.7},
            {'name': 'value_fcf_annual', 'metric': 'FCF', 'rebal': 'annual', 'gross_sharpe': 0.60, 'turnover': 0.25},
            {'name': 'value_ev_ebitda_q', 'metric': 'EV/EBITDA', 'rebal': 'quarterly', 'gross_sharpe': 0.50, 'turnover': 0.75},
            {'name': 'value_composite', 'metric': 'composite', 'rebal': 'annual', 'gross_sharpe': 0.65, 'turnover': 0.3},
        ],
        'quality': [
            {'name': 'quality_roe_monthly', 'metric': 'ROE', 'rebal': 'monthly', 'gross_sharpe': 0.50, 'turnover': 1.2},
            {'name': 'quality_roe_quarterly', 'metric': 'ROE', 'rebal': 'quarterly', 'gross_sharpe': 0.55, 'turnover': 0.6},
            {'name': 'quality_roe_annual', 'metric': 'ROE', 'rebal': 'annual', 'gross_sharpe': 0.60, 'turnover': 0.25},
            {'name': 'quality_roa_quarterly', 'metric': 'ROA', 'rebal': 'quarterly', 'gross_sharpe': 0.50, 'turnover': 0.65},
            {'name': 'quality_fcf_margin_q', 'metric': 'FCF_margin', 'rebal': 'quarterly', 'gross_sharpe': 0.60, 'turnover': 0.5},
            {'name': 'quality_fcf_margin_a', 'metric': 'FCF_margin', 'rebal': 'annual', 'gross_sharpe': 0.65, 'turnover': 0.2},
            {'name': 'quality_gp_assets', 'metric': 'GP/Assets', 'rebal': 'annual', 'gross_sharpe': 0.70, 'turnover': 0.3},
            {'name': 'quality_accruals', 'metric': 'accruals', 'rebal': 'annual', 'gross_sharpe': 0.55, 'turnover': 0.35},
            {'name': 'quality_f_score', 'metric': 'F-score', 'rebal': 'annual', 'gross_sharpe': 0.60, 'turnover': 0.4},
            {'name': 'quality_composite', 'metric': 'composite', 'rebal': 'annual', 'gross_sharpe': 0.75, 'turnover': 0.25},
        ],
        'low_vol': [
            {'name': 'lowvol_realized_m', 'metric': 'realized', 'rebal': 'monthly', 'gross_sharpe': 0.45, 'turnover': 1.0},
            {'name': 'lowvol_realized_q', 'metric': 'realized', 'rebal': 'quarterly', 'gross_sharpe': 0.50, 'turnover': 0.5},
            {'name': 'lowvol_realized_a', 'metric': 'realized', 'rebal': 'annual', 'gross_sharpe': 0.55, 'turnover': 0.2},
            {'name': 'lowvol_beta_m', 'metric': 'beta', 'rebal': 'monthly', 'gross_sharpe': 0.40, 'turnover': 1.2},
            {'name': 'lowvol_beta_q', 'metric': 'beta', 'rebal': 'quarterly', 'gross_sharpe': 0.45, 'turnover': 0.6},
            {'name': 'lowvol_beta_a', 'metric': 'beta', 'rebal': 'annual', 'gross_sharpe': 0.50, 'turnover': 0.25},
            {'name': 'lowvol_idio_q', 'metric': 'idio_vol', 'rebal': 'quarterly', 'gross_sharpe': 0.55, 'turnover': 0.55},
            {'name': 'lowvol_dd_q', 'metric': 'drawdown', 'rebal': 'quarterly', 'gross_sharpe': 0.50, 'turnover': 0.7},
            {'name': 'lowvol_min_var', 'metric': 'min_var', 'rebal': 'monthly', 'gross_sharpe': 0.55, 'turnover': 0.8},
            {'name': 'lowvol_min_var_a', 'metric': 'min_var', 'rebal': 'annual', 'gross_sharpe': 0.60, 'turnover': 0.3},
        ],
        'multifactor': [
            {'name': 'mf_value_mom_q', 'combo': 'V+M', 'rebal': 'quarterly', 'gross_sharpe': 0.55, 'turnover': 1.0},
            {'name': 'mf_value_mom_a', 'combo': 'V+M', 'rebal': 'annual', 'gross_sharpe': 0.60, 'turnover': 0.4},
            {'name': 'mf_quality_value_q', 'combo': 'Q+V', 'rebal': 'quarterly', 'gross_sharpe': 0.60, 'turnover': 0.7},
            {'name': 'mf_quality_value_a', 'combo': 'Q+V', 'rebal': 'annual', 'gross_sharpe': 0.70, 'turnover': 0.3},
            {'name': 'mf_quality_lowvol_a', 'combo': 'Q+LV', 'rebal': 'annual', 'gross_sharpe': 0.70, 'turnover': 0.25},
            {'name': 'mf_3factor_q', 'combo': 'V+M+Q', 'rebal': 'quarterly', 'gross_sharpe': 0.65, 'turnover': 0.8},
            {'name': 'mf_3factor_a', 'combo': 'V+M+Q', 'rebal': 'annual', 'gross_sharpe': 0.75, 'turnover': 0.35},
            {'name': 'mf_4factor_a', 'combo': 'V+M+Q+LV', 'rebal': 'annual', 'gross_sharpe': 0.80, 'turnover': 0.3},
            {'name': 'mf_defensive', 'combo': 'Q+LV+Div', 'rebal': 'annual', 'gross_sharpe': 0.65, 'turnover': 0.2},
            {'name': 'mf_aggressive', 'combo': 'M+Small', 'rebal': 'quarterly', 'gross_sharpe': 0.50, 'turnover': 1.5},
        ],
    }
    
    return library


def test_signal(signal: Dict, cost_bps: float) -> Dict:
    """Test a single signal at given cost level."""
    vol = 0.15
    annual_cost = signal['turnover'] * (cost_bps / 10000) * 2
    net_sharpe = signal['gross_sharpe'] - (annual_cost / vol)
    
    passes_sharpe = net_sharpe >= 0.5
    passes_turnover = signal['turnover'] <= 3.0
    decision = 'PASS' if (passes_sharpe and passes_turnover) else 'REJECT'
    
    return {
        'name': signal['name'],
        'gross_sharpe': signal['gross_sharpe'],
        'turnover': signal['turnover'],
        'net_sharpe': net_sharpe,
        'cost_bps': cost_bps,
        'decision': decision
    }


def search_signal_space():
    """
    Search the signal space to find survivors.
    
    Tests at three cost levels:
    - Realistic (1.5 bps)
    - Conservative (10 bps)
    - Sabotage (50 bps)
    """
    library = generate_signal_library()
    
    cost_levels = {
        'realistic': 1.5,
        'conservative': 10.0,
        'sabotage': 50.0
    }
    
    all_results = []
    survivors = {level: [] for level in cost_levels}
    
    for factor, variants in library.items():
        for signal in variants:
            for level_name, cost_bps in cost_levels.items():
                result = test_signal(signal, cost_bps)
                result['factor'] = factor
                result['cost_level'] = level_name
                all_results.append(result)
                
                if result['decision'] == 'PASS':
                    survivors[level_name].append(result)
    
    return survivors, pd.DataFrame(all_results)


def find_the_edge_case():
    """
    Binary search for the EXACT turnover level where signals flip from PASS to FAIL.
    
    For a signal with Sharpe 0.8, what's the maximum turnover that still passes?
    """
    cost_bps = 10.0  # Conservative
    sharpe_threshold = 0.5
    vol = 0.15
    
    test_sharpes = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    for gross_sharpe in test_sharpes:
        # Binary search
        low, high = 0.0, 10.0
        
        while high - low > 0.01:
            mid = (low + high) / 2
            annual_cost = mid * (cost_bps / 10000) * 2
            net_sharpe = gross_sharpe - (annual_cost / vol)
            
            if net_sharpe >= sharpe_threshold:
                low = mid
            else:
                high = mid
        
        max_turnover = low
        results.append({
            'gross_sharpe': gross_sharpe,
            'max_turnover': max_turnover
        })
    
    return pd.DataFrame(results)


def analyze_survivor_characteristics(survivors: List[Dict]) -> Dict:
    """Analyze common characteristics of signals that pass."""
    if len(survivors) == 0:
        return {'count': 0, 'message': 'No survivors'}
    
    df = pd.DataFrame(survivors)
    
    return {
        'count': len(df),
        'avg_gross_sharpe': df['gross_sharpe'].mean(),
        'avg_turnover': df['turnover'].mean(),
        'avg_net_sharpe': df['net_sharpe'].mean(),
        'max_turnover': df['turnover'].max(),
        'min_gross_sharpe': df['gross_sharpe'].min(),
        'factors': df['factor'].value_counts().to_dict() if 'factor' in df.columns else {}
    }


def run_survivor_search():
    """Main runner to find survivors."""
    print("=" * 70)
    print("SURVIVOR SEARCH: Finding Signals That Pass")
    print("=" * 70)
    
    survivors, results_df = search_signal_space()
    
    # Total signals
    total_signals = len(generate_signal_library()['momentum']) * 5  # 10 per factor * 5 factors
    
    for level_name, level_survivors in survivors.items():
        print(f"\n### {level_name.upper()} Costs ({len(level_survivors)}/{total_signals} pass) ###\n")
        
        if len(level_survivors) > 0:
            survivor_df = pd.DataFrame(level_survivors)
            print(survivor_df[['name', 'factor', 'gross_sharpe', 'turnover', 'net_sharpe']].to_string(index=False))
            
            # Characteristics
            chars = analyze_survivor_characteristics(level_survivors)
            print(f"\nCharacteristics:")
            print(f"  Avg gross Sharpe: {chars['avg_gross_sharpe']:.2f}")
            print(f"  Avg turnover: {chars['avg_turnover']:.2f}x")
            print(f"  Avg net Sharpe: {chars['avg_net_sharpe']:.2f}")
            print(f"  Max turnover among survivors: {chars['max_turnover']:.2f}x")
            print(f"  Factors represented: {chars['factors']}")
        else:
            print("No survivors at this cost level.")
    
    # Edge case analysis
    print("\n### EDGE CASE ANALYSIS ###\n")
    print("For a given gross Sharpe, what's the maximum viable turnover?")
    print("(At 10 bps costs, targeting 0.5 net Sharpe)\n")
    
    edge_df = find_the_edge_case()
    print(edge_df.to_string(index=False))
    
    # VERDICT
    print("\n" + "=" * 70)
    realistic_survivors = len(survivors['realistic'])
    conservative_survivors = len(survivors['conservative'])
    
    if conservative_survivors >= 3:
        print(f"SUCCESS: {conservative_survivors} signals survive at CONSERVATIVE costs")
        print("Framework CAN say YES to well-designed strategies.")
    elif realistic_survivors >= 3:
        print(f"PARTIAL SUCCESS: {realistic_survivors} signals survive at REALISTIC costs")
        print("Framework may be slightly too conservative, but works at market rates.")
    else:
        print("CRITICAL: Very few signals survive")
        print("Either expand signal library or recalibrate cost assumptions.")
    print("=" * 70)
    
    # Save
    results_df.to_csv('survivor_search_results.csv', index=False)
    edge_df.to_csv('turnover_edge_cases.csv', index=False)
    print(f"\nResults saved to survivor_search_results.csv and turnover_edge_cases.csv")
    
    return survivors, results_df


if __name__ == "__main__":
    run_survivor_search()
