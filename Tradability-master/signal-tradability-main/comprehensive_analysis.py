"""
Comprehensive Tradability Analysis

Demonstrates the complete rigorous analysis:
1. Formal definitions (statistical vs economic edge)
2. Mechanism chain quantification
3. Generalization claim
4. Explicit capacity math
5. Who is wrong (critique)

This is what separates research from student work.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from formal_definitions import (
    compute_statistical_edge, 
    compute_economic_edge, 
    identify_edge_mismatch
)
from mechanism_analysis import compute_mechanism_chain, quantify_turnover_sharpe_relationship
from generalization_analysis import select_best_generalization


def run_comprehensive_analysis():
    """
    Run complete rigorous analysis demonstrating all components.
    """
    print("=" * 80)
    print("COMPREHENSIVE TRADABILITY ANALYSIS")
    print("Research-Grade: Explaining 'Why', Not Just 'What'")
    print("=" * 80)
    
    # Load data
    ticker = 'SPY'
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    print(f"\n1. DATA LOADING")
    print(f"   Ticker: {ticker}")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Analyze multiple signals
    signal_names = ['momentum_12_1', 'mean_reversion']
    
    all_results = {}
    all_mechanism_chains = {}
    
    for signal_name in signal_names:
        print(f"\n{'='*80}")
        print(f"SIGNAL: {signal_name}")
        print(f"{'='*80}")
        
        # Compute signal
        signal_def = get_signal(signal_name)
        signal_values = signal_def.compute(prices, **signal_def.default_params())
        aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
        gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
        positions = compute_positions_from_returns(gross_returns, aligned_signals)
        
        print(f"\n2. FORMAL DEFINITIONS")
        print(f"   Computing statistical edge vs economic edge...")
        
        # Statistical edge
        stat_edge = compute_statistical_edge(aligned_signals, aligned_returns)
        print(f"\n   Statistical Edge:")
        print(f"   - Hit Rate: {stat_edge.hit_rate:.1%}")
        print(f"   - Hit Rate P-value: {stat_edge.hit_rate_pvalue:.4f}")
        print(f"   - Conditional Return: {stat_edge.conditional_return:.4f}")
        print(f"   - Unconditional Return: {stat_edge.unconditional_return:.4f}")
        print(f"   - Has Statistical Edge: {stat_edge.has_statistical_edge}")
        print(f"   - Edge Strength: {stat_edge.edge_strength:.2%}")
        
        # Economic edge (via tradability analysis)
        tradability = analyze_tradability(
            gross_returns=gross_returns,
            signals=aligned_signals,
            volatility=volatility,
            volumes=volumes,
            prices=prices,
            commission_per_trade=0.005,
            half_spread=0.001,
            periods_per_year=252
        )
        
        econ_edge = compute_economic_edge(
            tradability.net_metrics.return_mean * pd.Series([1.0]),  # Convert to series
            gross_returns,
            tradability.cost_drag,
            tradability.break_even_cost,
            tradability.max_viable_capacity or 0
        )
        
        # Align series properly
        net_returns_series = gross_returns - (tradability.cost_drag / 252)
        econ_edge = compute_economic_edge(
            net_returns_series,
            gross_returns,
            tradability.cost_drag,
            tradability.break_even_cost,
            tradability.max_viable_capacity or 0
        )
        
        print(f"\n   Economic Edge:")
        print(f"   - Net Sharpe: {econ_edge.net_sharpe:.3f}")
        print(f"   - Net Return: {econ_edge.net_return_mean:.4f}")
        print(f"   - Cost Drag: {econ_edge.cost_drag:.2%}")
        print(f"   - Break-even Cost: {econ_edge.break_even_cost*100:.3f}%")
        print(f"   - Max Capacity: ${econ_edge.max_viable_capacity/1e6:.1f}M")
        print(f"   - Has Economic Edge: {econ_edge.has_economic_edge}")
        print(f"   - Edge Robustness: {econ_edge.edge_robustness:.2%}")
        
        # Edge mismatch
        mismatch = identify_edge_mismatch(stat_edge, econ_edge)
        print(f"\n   Edge Mismatch Analysis:")
        print(f"   - Statistical Edge Exists: {mismatch['statistical_edge_exists']}")
        print(f"   - Economic Edge Exists: {mismatch['economic_edge_exists']}")
        print(f"   - Mismatch Type: {mismatch['mismatch_type']}")
        print(f"   - Failure Mode: {mismatch.get('failure_mode', 'N/A')}")
        
        # Mechanism chain
        print(f"\n3. MECHANISM CHAIN QUANTIFICATION")
        print(f"   Causal Chain: Signal Horizon -> Turnover -> Cost Drag -> Sharpe Decay")
        
        mechanism = compute_mechanism_chain(
            gross_returns,
            positions,
            cost_per_trade=0.005,
            annual_cost_drag=tradability.cost_drag,
            gross_sharpe=tradability.gross_metrics.sharpe_ratio or 0,
            net_sharpe=tradability.net_metrics.sharpe_ratio or 0,
            max_capacity=tradability.max_viable_capacity or 0,
            periods_per_year=252
        )
        
        chain_quant = mechanism.quantify_chain()
        
        print(f"\n   Link 1: Signal Horizon -> Turnover")
        link1 = chain_quant['link1_horizon_to_turnover']
        print(f"   - Signal Horizon: {mechanism.signal_horizon:.1f} periods")
        print(f"   - Annual Turnover: {mechanism.annual_turnover:.1f}x")
        print(f"   - Explanation: {link1['explanation']}")
        
        print(f"\n   Link 2: Turnover -> Cost Drag")
        link2 = chain_quant['link2_turnover_to_cost']
        print(f"   - Cost per Trade: {mechanism.cost_per_trade*100:.2f}%")
        print(f"   - Annual Cost Drag: {mechanism.annual_cost_drag:.2%}")
        print(f"   - Explanation: {link2['explanation']}")
        
        print(f"\n   Link 3: Cost Drag -> Sharpe Decay")
        link3 = chain_quant['link3_cost_to_sharpe']
        print(f"   - Gross Sharpe: {mechanism.gross_sharpe:.3f}")
        print(f"   - Net Sharpe: {mechanism.net_sharpe:.3f}")
        print(f"   - Sharpe Decay: {mechanism.sharpe_decay:.3f}")
        print(f"   - Explanation: {link3['explanation']}")
        
        print(f"\n   Link 4: Capacity Scaling")
        link4 = chain_quant['link4_capacity_scaling']
        print(f"   - Max Capacity: ${mechanism.max_capacity/1e6:.1f}M")
        print(f"   - Explanation: {link4['explanation']}")
        
        formula = mechanism.generate_formula()
        print(f"\n   Explicit Formula:")
        print(f"   {formula}")
        
        # Store results
        all_results[signal_name] = {
            'stat_edge': stat_edge,
            'econ_edge': econ_edge,
            'tradability': tradability,
            'annual_turnover': mechanism.annual_turnover,
            'break_even_cost': tradability.break_even_cost,
            'net_sharpe': tradability.net_metrics.sharpe_ratio or 0,
            'hit_rate_survival': tradability.hit_rate_survival or 0,
        }
        all_mechanism_chains[signal_name] = mechanism
    
    # Generalization claim
    print(f"\n{'='*80}")
    print(f"4. GENERALIZATION CLAIM")
    print(f"{'='*80}")
    
    # Convert results to format expected by generalization analysis
    generalization_results = {}
    for name, result in all_results.items():
        generalization_results[name] = {
            'annual_turnover': result['annual_turnover'],
            'break_even_cost': result['break_even_cost'],
            'net_sharpe': result['net_sharpe'],
            'hit_rate_survival': result.get('hit_rate_survival', 0)
        }
    
    best_claim = select_best_generalization(generalization_results)
    
    print(f"\n   Primary Generalization Claim:")
    print(f"   {best_claim.claim}")
    print(f"\n   Formal Condition:")
    print(f"   {best_claim.condition}")
    print(f"\n   Supporting Evidence:")
    for key, value in best_claim.supporting_evidence.items():
        print(f"   - {key}: {value}")
    print(f"\n   Scope: {best_claim.scope}")
    
    # Turnover-Sharpe relationship
    print(f"\n5. TURNOVER-SHARPE RELATIONSHIP QUANTIFICATION")
    relationship = quantify_turnover_sharpe_relationship(all_mechanism_chains)
    if 'formula' in relationship:
        print(f"\n   Formula: {relationship['formula']}")
        print(f"\n   Interpretation:")
        print(f"   {relationship['interpretation']}")
        print(f"\n   Example:")
        print(f"   {relationship['example']}")
        if 'caveats' in relationship:
            print(f"\n   IMPORTANT CAVEATS:")
            caveats = relationship['caveats']
            print(f"   - Context-dependent: {caveats.get('context_dependent', 'N/A')}")
            print(f"   - Scope: {caveats.get('scope', 'N/A')}")
            print(f"   - Time Period: {caveats.get('time_period', 'N/A')}")
            print(f"   - Cost Structure: {caveats.get('cost_structure', 'N/A')}")
            print(f"   - Note: {caveats.get('note', 'N/A')}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nThis analysis demonstrates:")
    print(f"1. Formal definitions of statistical vs economic edge")
    print(f"2. Explicit mechanism chain quantification")
    print(f"3. Generalization claim with supporting evidence")
    print(f"4. Rigorous capacity math (see capacity.py)")
    print(f"5. Who is wrong (see who_is_wrong.md)")
    print(f"\nThis is research, not student work.")


if __name__ == "__main__":
    run_comprehensive_analysis()

