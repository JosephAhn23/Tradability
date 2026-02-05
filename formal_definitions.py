"""
Formal Definitions: Statistical Edge vs Economic Edge

Explicit definitions and conditions for tradability analysis.
This formalizes the distinction that the research depends on.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class StatisticalEdge:
    """
    Statistical Edge: Evidence that a signal contains predictive information.
    
    Definition:
    A signal has statistical edge if it produces a hit rate significantly 
    different from 50% (for binary predictions) or if forward returns 
    conditional on signal values are statistically different from unconditional returns.
    
    Note: Statistical edge does NOT imply tradability.
    """
    hit_rate: float  # Percentage of correct predictions
    hit_rate_pvalue: float  # Statistical significance of hit rate deviation from 50%
    conditional_return: float  # Mean return when signal is "on"
    unconditional_return: float  # Mean return overall
    return_difference_pvalue: float  # Significance of conditional vs unconditional
    
    @property
    def has_statistical_edge(self) -> bool:
        """Statistical edge exists if hit rate > 50% with significance OR conditional returns differ."""
        return (
            (self.hit_rate > 0.50 and self.hit_rate_pvalue < 0.05) or
            (self.return_difference_pvalue < 0.05)
        )
    
    @property
    def edge_strength(self) -> float:
        """Measure of statistical edge strength (0 to 1)."""
        hit_rate_edge = max(0, (self.hit_rate - 0.50) / 0.50)  # Normalize to [0, 1]
        return_edge = abs(self.conditional_return - self.unconditional_return) / (abs(self.unconditional_return) + 1e-6)
        return_edge = min(1.0, return_edge)  # Cap at 1
        
        # Combine (weighted average)
        return 0.6 * hit_rate_edge + 0.4 * return_edge


@dataclass
class EconomicEdge:
    """
    Economic Edge: Evidence that a signal can be profitably traded.
    
    Definition:
    A signal has economic edge if the expected net return (after all frictions)
    is positive, with net Sharpe ratio > 0, and sufficient capacity exists
    to deploy meaningful capital.
    
    Key frictions:
    - Explicit costs (commissions, bid-ask spread)
    - Slippage (volatility and volume impact)
    - Capacity constraints (market impact at scale)
    """
    net_return_mean: float  # Expected return after all costs
    net_sharpe: float  # Sharpe ratio after all costs
    cost_drag: float  # Annual cost drag (as fraction)
    break_even_cost: float  # Maximum cost level that preserves profitability
    max_viable_capacity: float  # Maximum AUM before impact overwhelms edge
    
    @property
    def has_economic_edge(self) -> bool:
        """Economic edge exists if net Sharpe > 0 and sufficient capacity exists."""
        return (
            self.net_sharpe > 0 and
            self.net_return_mean > 0 and
            self.max_viable_capacity > 1e4  # At least $10k capacity
        )
    
    @property
    def edge_robustness(self) -> float:
        """Measure of economic edge robustness (0 to 1)."""
        # Robustness = how much costs can increase before edge disappears
        # Normalize break_even_cost to [0, 1] (assume >1% is very robust)
        robustness = min(1.0, self.break_even_cost / 0.01)
        
        # Adjust for capacity constraints
        # If capacity < $1M, reduce robustness
        capacity_factor = min(1.0, self.max_viable_capacity / 1e6)
        
        return robustness * capacity_factor


def compute_statistical_edge(signals: pd.Series, 
                             forward_returns: pd.Series,
                             quantile: float = 0.5) -> StatisticalEdge:
    """
    Compute statistical edge metrics.
    
    Args:
        signals: Signal values
        forward_returns: Forward-looking returns
        quantile: Threshold for binary classification
    
    Returns:
        StatisticalEdge object
    """
    # Align data
    aligned = pd.DataFrame({
        'signal': signals,
        'forward_return': forward_returns
    }).dropna()
    
    if len(aligned) == 0:
        raise ValueError("No aligned data for statistical edge computation")
    
    # Binary predictions: long if signal > threshold, short if < threshold
    threshold = aligned['signal'].quantile(quantile)
    predictions = (aligned['signal'] > threshold).astype(int)
    
    # Returns when long (prediction = 1) vs short (prediction = 0)
    long_returns = aligned[predictions == 1]['forward_return']
    short_returns = aligned[predictions == 0]['forward_return']
    
    # Hit rate: correct when (prediction=1 and return>0) or (prediction=0 and return<0)
    # For simplicity: long when signal high, so correct if return > 0 when signal > threshold
    correct_predictions = (
        ((predictions == 1) & (aligned['forward_return'] > 0)) |
        ((predictions == 0) & (aligned['forward_return'] < 0))
    )
    hit_rate = correct_predictions.mean()
    
    # Statistical significance of hit rate (binomial test)
    n = len(correct_predictions)
    k = correct_predictions.sum()
    hit_rate_pvalue = 1 - stats.binom.cdf(k - 1, n, 0.50)  # One-sided test
    
    # Conditional returns
    conditional_return = long_returns.mean() if len(long_returns) > 0 else 0.0
    unconditional_return = aligned['forward_return'].mean()
    
    # Test if conditional return differs from unconditional
    if len(long_returns) > 10 and len(short_returns) > 10:
        # Two-sample t-test
        t_stat, return_diff_pvalue = stats.ttest_ind(long_returns, short_returns)
        # Convert to one-sided p-value
        if t_stat > 0:
            return_difference_pvalue = return_diff_pvalue / 2
        else:
            return_difference_pvalue = 1 - return_diff_pvalue / 2
    else:
        return_difference_pvalue = 1.0
    
    return StatisticalEdge(
        hit_rate=hit_rate,
        hit_rate_pvalue=hit_rate_pvalue,
        conditional_return=conditional_return,
        unconditional_return=unconditional_return,
        return_difference_pvalue=return_difference_pvalue
    )


def compute_economic_edge(net_returns: pd.Series,
                         gross_returns: pd.Series,
                         cost_drag: float,
                         break_even_cost: float,
                         max_viable_capacity: float) -> EconomicEdge:
    """
    Compute economic edge metrics.
    
    Args:
        net_returns: Returns after all costs
        gross_returns: Returns before costs
        cost_drag: Annual cost drag (fraction)
        break_even_cost: Maximum viable cost per trade
        max_viable_capacity: Maximum viable AUM
    
    Returns:
        EconomicEdge object
    """
    net_return_mean = net_returns.mean()
    net_return_std = net_returns.std()
    
    # Annualized Sharpe
    periods_per_year = 252 if len(net_returns) > 252 else len(net_returns)
    if net_return_std > 0:
        net_sharpe = net_return_mean / net_return_std * np.sqrt(periods_per_year)
    else:
        net_sharpe = 0.0
    
    return EconomicEdge(
        net_return_mean=net_return_mean,
        net_sharpe=net_sharpe,
        cost_drag=cost_drag,
        break_even_cost=break_even_cost,
        max_viable_capacity=max_viable_capacity
    )


def identify_edge_mismatch(stat_edge: StatisticalEdge, 
                          econ_edge: EconomicEdge) -> Dict:
    """
    Identify cases where statistical edge exists but economic edge does not.
    
    This is the core phenomenon: signals that work statistically but fail economically.
    
    Returns:
        Dictionary with mismatch analysis
    """
    has_stat = stat_edge.has_statistical_edge
    has_econ = econ_edge.has_economic_edge
    
    mismatch = {
        'statistical_edge_exists': has_stat,
        'economic_edge_exists': has_econ,
        'mismatch_type': None,
        'edge_strength': stat_edge.edge_strength if has_stat else 0.0,
        'edge_robustness': econ_edge.edge_robustness if has_econ else 0.0,
    }
    
    if has_stat and not has_econ:
        mismatch['mismatch_type'] = 'statistical_without_economic'
        mismatch['failure_mode'] = 'costs_overwhelm_edge'
    elif not has_stat and has_econ:
        mismatch['mismatch_type'] = 'economic_without_statistical'
        mismatch['failure_mode'] = 'spurious_economic_signal'
    elif has_stat and has_econ:
        mismatch['mismatch_type'] = 'both_edges_exist'
        mismatch['failure_mode'] = 'tradable_signal'
    else:
        mismatch['mismatch_type'] = 'neither_edge_exists'
        mismatch['failure_mode'] = 'no_edge'
    
    return mismatch


def formalize_generalization_claim(results: Dict[str, Dict]) -> str:
    """
    Generate a formal generalization claim based on empirical results.
    
    Args:
        results: Dictionary mapping signal names to edge analysis results
    
    Returns:
        Formal claim string
    """
    # Analyze patterns across signals
    turnover_thresholds = []
    cost_sensitivities = []
    
    for signal_name, result in results.items():
        if 'annual_turnover' in result:
            turnover_thresholds.append(result['annual_turnover'])
        if 'break_even_cost' in result:
            cost_sensitivities.append(result['break_even_cost'])
    
    if len(turnover_thresholds) > 0 and len(cost_sensitivities) > 0:
        avg_turnover = np.mean(turnover_thresholds)
        avg_break_even = np.mean(cost_sensitivities)
        
        # Generate claim based on patterns
        claim = (
            f"Signals with annual turnover above {avg_turnover:.1f}x "
            f"require transaction costs below {avg_break_even*100:.2f}% per trade "
            f"to preserve economic edge. Signals exceeding this threshold exhibit "
            f"statistical edge (hit rate > 50%) but fail economically due to "
            f"cost drag exceeding residual alpha."
        )
        return claim
    else:
        return "Insufficient data for generalization claim."

