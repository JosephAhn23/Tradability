"""
Sensitivity Analysis Module

Creates sensitivity matrices showing how framework decisions change
under different assumption scenarios.

This addresses the critical review demand for:
- Sensitivity matrix for assumptions
- Identification of critical assumptions
- Threshold justification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns
from tradability_analysis import analyze_tradability, compute_positions_from_returns


@dataclass
class AssumptionScenario:
    """Represents a scenario with different assumption values."""
    name: str
    commission_per_trade: float
    half_spread: float
    participation_rate: float
    adv_multiplier: float = 1.0  # Multiplier for ADV (1.0 = base, 0.5 = halved)


def compute_sensitivity_matrix(
    signal_name: str,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    base_scenario: AssumptionScenario,
    scenarios: List[AssumptionScenario]
) -> pd.DataFrame:
    """
    Compute sensitivity matrix showing how decisions change under different scenarios.
    
    Args:
        signal_name: Signal to test
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        base_scenario: Base case scenario
        scenarios: List of alternative scenarios
    
    Returns:
        DataFrame with sensitivity results
    """
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Compute signal
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Base case
    base_tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes * base_scenario.adv_multiplier,
        prices=prices,
        commission_per_trade=base_scenario.commission_per_trade,
        half_spread=base_scenario.half_spread,
        periods_per_year=252
    )
    
    results = []
    
    # Test each scenario
    for scenario in scenarios:
        # Apply scenario
        scenario_volumes = volumes * scenario.adv_multiplier
        
        tradability = analyze_tradability(
            gross_returns=gross_returns,
            signals=aligned_signals,
            volatility=volatility,
            volumes=scenario_volumes,
            prices=prices,
            commission_per_trade=scenario.commission_per_trade,
            half_spread=scenario.half_spread,
            periods_per_year=252
        )
        
        # Compare to base
        break_even_change = (tradability.break_even_cost or 0) - (base_tradability.break_even_cost or 0)
        capacity_change = (tradability.max_viable_capacity or 0) - (base_tradability.max_viable_capacity or 0)
        cost_drag_change = (tradability.cost_drag or 0) - (base_tradability.cost_drag or 0)
        
        # Decision (simplified: REJECT if break-even < 1%)
        base_decision = "REJECT" if (base_tradability.break_even_cost or 0) < 0.01 else "DEPLOY"
        scenario_decision = "REJECT" if (tradability.break_even_cost or 0) < 0.01 else "DEPLOY"
        decision_flip = base_decision != scenario_decision
        
        results.append({
            'scenario': scenario.name,
            'commission': scenario.commission_per_trade,
            'spread': scenario.half_spread,
            'adv_multiplier': scenario.adv_multiplier,
            'break_even_cost': tradability.break_even_cost,
            'break_even_change': break_even_change,
            'max_capacity': tradability.max_viable_capacity,
            'capacity_change': capacity_change,
            'cost_drag': tradability.cost_drag,
            'cost_drag_change': cost_drag_change,
            'decision': scenario_decision,
            'decision_flip': decision_flip
        })
    
    return pd.DataFrame(results)


def create_standard_sensitivity_scenarios() -> Tuple[AssumptionScenario, List[AssumptionScenario]]:
    """
    Create standard sensitivity scenarios for testing.
    
    Returns:
        Tuple of (base_scenario, list_of_scenarios)
    """
    base = AssumptionScenario(
        name="Base Case",
        commission_per_trade=0.005,  # 0.5%
        half_spread=0.001,  # 0.1%
        participation_rate=0.01,  # 1%
        adv_multiplier=1.0
    )
    
    scenarios = [
        # Cost variations
        AssumptionScenario("Costs +50%", 0.0075, 0.001, 0.01, 1.0),
        AssumptionScenario("Costs +100%", 0.010, 0.001, 0.01, 1.0),
        AssumptionScenario("Costs -50%", 0.0025, 0.001, 0.01, 1.0),
        
        # Spread variations
        AssumptionScenario("Spread +100%", 0.005, 0.002, 0.01, 1.0),
        AssumptionScenario("Spread +500%", 0.005, 0.006, 0.01, 1.0),
        AssumptionScenario("Spread -50%", 0.005, 0.0005, 0.01, 1.0),
        
        # Liquidity variations
        AssumptionScenario("ADV -50%", 0.005, 0.001, 0.01, 0.5),
        AssumptionScenario("ADV -75%", 0.005, 0.001, 0.01, 0.25),
        AssumptionScenario("ADV +50%", 0.005, 0.001, 0.01, 1.5),
        
        # Combined stress
        AssumptionScenario("Stress (2x costs, 0.5x ADV)", 0.010, 0.002, 0.01, 0.5),
        AssumptionScenario("Crisis (2x costs, 5x spread, 0.5x ADV)", 0.010, 0.010, 0.01, 0.5),
    ]
    
    return base, scenarios


def identify_critical_assumptions(
    sensitivity_matrix: pd.DataFrame,
    threshold_pct: float = 20.0
) -> pd.DataFrame:
    """
    Identify critical assumptions that cause decision flips.
    
    Args:
        sensitivity_matrix: Output from compute_sensitivity_matrix
        threshold_pct: Percentage change threshold for "critical"
    
    Returns:
        DataFrame with critical assumptions ranked by impact
    """
    # Filter scenarios where decision flipped
    flips = sensitivity_matrix[sensitivity_matrix['decision_flip'] == True].copy()
    
    if len(flips) == 0:
        return pd.DataFrame(columns=['assumption', 'change_pct', 'impact', 'critical'])
    
    # Compute percentage changes
    flips['commission_change_pct'] = ((flips['commission'] - 0.005) / 0.005) * 100
    flips['spread_change_pct'] = ((flips['spread'] - 0.001) / 0.001) * 100
    flips['adv_change_pct'] = ((flips['adv_multiplier'] - 1.0) / 1.0) * 100
    
    # Rank by impact (break-even change)
    flips['impact'] = abs(flips['break_even_change'])
    flips = flips.sort_values('impact', ascending=False)
    
    # Identify critical assumptions
    critical = []
    
    for _, row in flips.iterrows():
        if abs(row['commission_change_pct']) >= threshold_pct:
            critical.append({
                'assumption': 'commission',
                'change_pct': row['commission_change_pct'],
                'impact': row['impact'],
                'critical': True,
                'scenario': row['scenario']
            })
        
        if abs(row['spread_change_pct']) >= threshold_pct:
            critical.append({
                'assumption': 'spread',
                'change_pct': row['spread_change_pct'],
                'impact': row['impact'],
                'critical': True,
                'scenario': row['scenario']
            })
        
        if abs(row['adv_change_pct']) >= threshold_pct:
            critical.append({
                'assumption': 'adv',
                'change_pct': row['adv_change_pct'],
                'impact': row['impact'],
                'critical': True,
                'scenario': row['scenario']
            })
    
    if len(critical) == 0:
        return pd.DataFrame(columns=['assumption', 'change_pct', 'impact', 'critical'])
    
    critical_df = pd.DataFrame(critical)
    critical_df = critical_df.sort_values('impact', ascending=False)
    
    return critical_df


def justify_thresholds(
    sensitivity_matrix: pd.DataFrame,
    thresholds: Dict[str, float]
) -> pd.DataFrame:
    """
    Justify thresholds by showing sensitivity analysis.
    
    Args:
        sensitivity_matrix: Output from compute_sensitivity_matrix
        thresholds: Dict of threshold names and values
    
    Returns:
        DataFrame showing threshold justification
    """
    justification = []
    
    # For each threshold, show how many signals flip if threshold changes
    for threshold_name, threshold_value in thresholds.items():
        # Simulate threshold variations
        variations = [threshold_value * 0.5, threshold_value * 0.75, 
                      threshold_value, threshold_value * 1.25, threshold_value * 1.5]
        
        for var_value in variations:
            # Count how many scenarios would flip at this threshold
            if threshold_name == 'break_even':
                flips = len(sensitivity_matrix[
                    (sensitivity_matrix['break_even_cost'] < threshold_value) != 
                    (sensitivity_matrix['break_even_cost'] < var_value)
                ])
            else:
                flips = 0  # Placeholder for other thresholds
            
            justification.append({
                'threshold': threshold_name,
                'base_value': threshold_value,
                'variation': var_value,
                'variation_pct': ((var_value - threshold_value) / threshold_value) * 100,
                'scenarios_affected': flips
            })
    
    return pd.DataFrame(justification)


# Example usage
if __name__ == "__main__":
    # Test sensitivity
    base, scenarios = create_standard_sensitivity_scenarios()
    
    matrix = compute_sensitivity_matrix(
        signal_name='momentum_12_1',
        ticker='SPY',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2020, 12, 31),
        base_scenario=base,
        scenarios=scenarios
    )
    
    print("Sensitivity Matrix:")
    print(matrix[['scenario', 'break_even_cost', 'decision', 'decision_flip']])
    
    # Identify critical assumptions
    critical = identify_critical_assumptions(matrix)
    print("\nCritical Assumptions:")
    print(critical)
    
    # Justify thresholds
    thresholds = {'break_even': 0.01, 'capacity': 25_000_000, 'turnover': 3.0}
    justification = justify_thresholds(matrix, thresholds)
    print("\nThreshold Justification:")
    print(justification)

