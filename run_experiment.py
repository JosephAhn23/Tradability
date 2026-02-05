"""
Experiment Runner: Config-Driven Analysis

Separates engine (reusable code) from analysis (notebooks).
Configs drive experiments; notebooks only interpret results.

This is the industry pattern: engine → experiment → analysis
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from experiment_config import (
    ExperimentConfig, get_default_config, get_sensitivity_config,
    get_uncertainty_config, get_counterexample_config
)
from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from formal_definitions import (
    compute_statistical_edge, compute_economic_edge, identify_edge_mismatch
)
from uncertainty_analysis import quantify_uncertainty_comprehensive
from counterexamples import run_all_counterexamples
from pm_decisions import generate_pm_summary_report


class ExperimentRunner:
    """
    Runs experiments based on configuration.
    
    This separates the engine (reusable code) from the analysis (notebooks).
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary with all results
        """
        print("=" * 80)
        print(f"EXPERIMENT: {self.config.experiment_name}")
        print("=" * 80)
        if self.config.description:
            print(f"Description: {self.config.description}")
        if self.config.notes:
            print(f"Notes: {self.config.notes}")
        print()
        
        # Load data
        print(f"Loading data for {self.config.data.ticker}...")
        prices, volumes = load_price_data(
            self.config.data.ticker,
            self.config.data.start_date,
            self.config.data.end_date
        )
        forward_returns = compute_forward_returns(prices)
        volatility = prices.pct_change().rolling(20).std() * np.sqrt(self.config.data.periods_per_year)
        
        # Run analysis for each signal
        signal_results = {}
        
        for signal_config in self.config.signals:
            print(f"\n{'='*80}")
            print(f"Signal: {signal_config.name}")
            print(f"{'='*80}")
            
            # Compute signal
            signal_def = get_signal(signal_config.name)
            signal_params = signal_config.params if signal_config.params else signal_def.default_params()
            signal_values = signal_def.compute(prices, **signal_params)
            aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
            gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=signal_config.quantile)
            positions = compute_positions_from_returns(gross_returns, aligned_signals)
            
            # Statistical edge
            stat_edge = compute_statistical_edge(aligned_signals, aligned_returns, quantile=signal_config.quantile)
            
            # Tradability analysis
            tradability = analyze_tradability(
                gross_returns=gross_returns,
                signals=aligned_signals,
                volatility=volatility,
                volumes=volumes,
                prices=prices,
                commission_per_trade=self.config.costs.commission_per_trade,
                half_spread=self.config.costs.half_spread,
                vol_impact_coefficient=self.config.costs.vol_impact_coefficient,
                vol_impact_coefficient2=self.config.costs.vol_impact_coefficient2,
                periods_per_year=self.config.data.periods_per_year,
                sharpe_threshold=self.config.capacity.sharpe_threshold
            )
            
            # Economic edge
            net_returns_series = gross_returns - (tradability.cost_drag / self.config.data.periods_per_year)
            econ_edge = compute_economic_edge(
                net_returns_series,
                gross_returns,
                tradability.cost_drag,
                tradability.break_even_cost,
                tradability.max_viable_capacity or 0
            )
            
            # Edge mismatch
            mismatch = identify_edge_mismatch(stat_edge, econ_edge)
            
            # Uncertainty quantification (if enabled)
            uncertainty_results = None
            if self.config.experiment_name == "uncertainty_quantification":
                print("\nComputing uncertainty quantification...")
                uncertainty_results = quantify_uncertainty_comprehensive(
                    gross_returns,
                    net_returns_series,
                    positions,
                    volatility,
                    self.config.data.periods_per_year
                )
            
            # Store results
            signal_results[signal_config.name] = {
                'statistical_edge': {
                    'hit_rate': stat_edge.hit_rate,
                    'has_statistical_edge': stat_edge.has_statistical_edge,
                    'edge_strength': stat_edge.edge_strength,
                },
                'economic_edge': {
                    'net_sharpe': econ_edge.net_sharpe,
                    'has_economic_edge': econ_edge.has_economic_edge,
                    'edge_robustness': econ_edge.edge_robustness,
                },
                'tradability': {
                    'gross_sharpe': tradability.gross_metrics.sharpe_ratio if tradability.gross_metrics else None,
                    'net_sharpe': tradability.net_metrics.sharpe_ratio if tradability.net_metrics else None,
                    'annual_turnover': tradability.annual_turnover,
                    'cost_drag': tradability.cost_drag,
                    'break_even_cost': tradability.break_even_cost,
                    'max_viable_capacity': tradability.max_viable_capacity,
                },
                'mismatch': mismatch,
                'uncertainty': uncertainty_results,
            }
        
        self.results = {
            'experiment_name': self.config.experiment_name,
            'config': self.config.to_dict(),
            'signals': signal_results,
        }
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Convert to JSON-serializable format
        results_json = json.dumps(self.results, default=str, indent=2)
        with open(filepath, 'w') as f:
            f.write(results_json)
        print(f"\nResults saved to {filepath}")


def run_default_experiment():
    """Run the default tradability analysis experiment."""
    config = get_default_config()
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results('results_default.json')
    return results


def run_sensitivity_experiment():
    """Run cost sensitivity analysis."""
    config = get_sensitivity_config()
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results('results_sensitivity.json')
    return results


def run_uncertainty_experiment():
    """Run uncertainty quantification experiment."""
    config = get_uncertainty_config()
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results('results_uncertainty.json')
    return results


def run_full_analysis():
    """
    Run complete analysis including:
    1. Default tradability analysis
    2. Counterexamples
    3. PM decision framework
    """
    print("=" * 80)
    print("COMPLETE RESEARCH ANALYSIS")
    print("=" * 80)
    
    # 1. Default experiment
    print("\n1. Running default tradability analysis...")
    default_results = run_default_experiment()
    
    # 2. Counterexamples
    print("\n2. Running counterexample tests...")
    counterexample_results = run_all_counterexamples()
    
    # 3. PM decisions
    print("\n3. Generating PM decision framework...")
    signal_names = [s.name for s in get_default_config().signals]
    pm_report = generate_pm_summary_report(signal_names)
    
    # Save PM report
    with open('pm_decisions_report.txt', 'w') as f:
        f.write(pm_report)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nFiles generated:")
    print("  - results_default.json")
    print("  - pm_decisions_report.txt")
    print("\nThis demonstrates:")
    print("  1. Config-driven experiments (engine separated from analysis)")
    print("  2. Explicit counterexamples (when claims don't hold)")
    print("  3. PM decision framework (what to do differently with capital)")
    print("  4. Uncertainty quantification (not just averages)")
    print("  5. Explicit claims and assumptions (see RESEARCH_SUMMARY.md)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
        if experiment_type == "default":
            run_default_experiment()
        elif experiment_type == "sensitivity":
            run_sensitivity_experiment()
        elif experiment_type == "uncertainty":
            run_uncertainty_experiment()
        elif experiment_type == "counterexamples":
            run_all_counterexamples()
        elif experiment_type == "pm":
            signal_names = [s.name for s in get_default_config().signals]
            generate_pm_summary_report(signal_names)
        elif experiment_type == "full":
            run_full_analysis()
        else:
            print(f"Unknown experiment type: {experiment_type}")
            print("Available: default, sensitivity, uncertainty, counterexamples, pm, full")
    else:
        # Run full analysis by default
        run_full_analysis()



