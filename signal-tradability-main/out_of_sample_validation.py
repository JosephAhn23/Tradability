"""
Out-of-Sample Validation Framework

Implements proper train/test split and validates framework decisions
against actual signal performance.

Addresses critical review demand for:
- Pre-registration of framework
- Out-of-sample validation (2015-2026 unseen)
- False positive/negative rate calculation
- Accuracy measurement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from signals import get_signal, list_signals
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from statistical_rigor import compute_sharpe_with_ci


@dataclass
class ValidationResult:
    """Results from out-of-sample validation."""
    signal_name: str
    train_decision: str  # DEPLOY or REJECT from training period
    test_actual_performance: float  # Actual Sharpe in test period
    test_actual_break_even: float  # Actual break-even in test period
    test_decision: str  # What framework would decide in test period
    correct: bool  # Was training decision correct?
    false_positive: bool  # Rejected but actually worked?
    false_negative: bool  # Deployed but actually failed?


def run_out_of_sample_validation(
    signal_names: List[str],
    ticker: str,
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    rejection_thresholds: Dict[str, float],
    use_regime_costs: bool = False
) -> pd.DataFrame:
    """
    Run out-of-sample validation: train on one period, test on another.
    
    Args:
        signal_names: List of signals to test
        ticker: Ticker symbol
        train_start: Training period start
        train_end: Training period end
        test_start: Test period start
        test_end: Test period end
        rejection_thresholds: Dict of thresholds (break_even, capacity, etc.)
        use_regime_costs: Whether to use regime-dependent costs in test
    
    Returns:
        DataFrame with validation results
    """
    results = []
    
    for signal_name in signal_names:
        # TRAIN: Run framework on training period
        train_decision, train_metrics = run_framework_on_period(
            signal_name, ticker, train_start, train_end, rejection_thresholds
        )
        
        # TEST: Run framework on test period (unseen)
        test_decision, test_metrics = run_framework_on_period(
            signal_name, ticker, test_start, test_end, rejection_thresholds,
            use_regime_costs=use_regime_costs
        )
        
        # TEST: Actual performance in test period
        test_actual_sharpe, test_actual_break_even = compute_actual_performance(
            signal_name, ticker, test_start, test_end
        )
        
        # Determine if training decision was correct
        # Correct if: (DEPLOY and test Sharpe > 0) or (REJECT and test Sharpe <= 0)
        # Or use break-even: (DEPLOY and break-even > threshold) or (REJECT and break-even < threshold)
        correct = (
            (train_decision == "DEPLOY" and test_actual_break_even >= rejection_thresholds['break_even']) or
            (train_decision == "REJECT" and test_actual_break_even < rejection_thresholds['break_even'])
        )
        
        false_positive = (train_decision == "REJECT" and test_actual_break_even >= rejection_thresholds['break_even'])
        false_negative = (train_decision == "DEPLOY" and test_actual_break_even < rejection_thresholds['break_even'])
        
        results.append({
            'signal': signal_name,
            'train_decision': train_decision,
            'train_break_even': train_metrics.get('break_even_cost', 0),
            'test_decision': test_decision,
            'test_break_even': test_metrics.get('break_even_cost', 0),
            'test_actual_sharpe': test_actual_sharpe,
            'test_actual_break_even': test_actual_break_even,
            'correct': correct,
            'false_positive': false_positive,
            'false_negative': false_negative
        })
    
    return pd.DataFrame(results)


def run_framework_on_period(
    signal_name: str,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    rejection_thresholds: Dict[str, float],
    use_regime_costs: bool = False
) -> Tuple[str, Dict]:
    """
    Run framework on a specific time period.
    
    Returns:
        Tuple of (decision, metrics_dict)
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
    
    # Analyze tradability
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
    
    # Apply regime costs if requested
    if use_regime_costs:
        # In stress periods, costs are higher
        # For now, simple: if VIX > 30, double costs
        # TODO: Implement proper VIX-based cost adjustment
        pass
    
    # Make decision based on thresholds
    break_even = tradability.break_even_cost or 0
    max_capacity = tradability.max_viable_capacity or 0
    
    decision = "DEPLOY"
    if break_even < rejection_thresholds.get('break_even', 0.01):
        decision = "REJECT"
    elif max_capacity < rejection_thresholds.get('capacity', 25_000_000):
        decision = "REJECT"
    
    metrics = {
        'break_even_cost': break_even,
        'max_viable_capacity': max_capacity,
        'cost_drag': tradability.cost_drag,
        'annual_turnover': tradability.annual_turnover
    }
    
    return decision, metrics


def compute_actual_performance(
    signal_name: str,
    ticker: str,
    start_date: datetime,
    end_date: datetime
) -> Tuple[float, float]:
    """
    Compute actual signal performance in test period.
    
    Returns:
        Tuple of (actual_sharpe, actual_break_even)
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
    
    # Compute actual performance with costs
    from tradability_analysis import analyze_tradability
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
    
    actual_sharpe = tradability.net_metrics.sharpe_ratio if tradability.net_metrics else None
    actual_break_even = tradability.break_even_cost or 0
    
    return actual_sharpe or 0, actual_break_even


def compute_validation_metrics(validation_results: pd.DataFrame) -> Dict:
    """
    Compute validation metrics from results.
    
    Returns:
        Dict with accuracy, false positive rate, false negative rate
    """
    total = len(validation_results)
    correct = validation_results['correct'].sum()
    false_positives = validation_results['false_positive'].sum()
    false_negatives = validation_results['false_negative'].sum()
    
    accuracy = correct / total if total > 0 else 0
    fpr = false_positives / total if total > 0 else 0
    fnr = false_negatives / total if total > 0 else 0
    
    return {
        'total_signals': total,
        'correct': correct,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }


def pre_register_framework(
    framework_config: Dict,
    output_file: str = 'framework_preregistration.json'
) -> str:
    """
    Pre-register framework before analyzing data.
    
    Creates a timestamped, hashable record of framework configuration.
    
    Args:
        framework_config: Dict with framework parameters
        output_file: Output file path
    
    Returns:
        SHA-256 hash of the configuration
    """
    import hashlib
    from datetime import datetime
    
    # Add timestamp
    framework_config['preregistration_timestamp'] = datetime.utcnow().isoformat()
    framework_config['preregistration_commit'] = "TBD"  # Should be git commit hash
    
    # Serialize and hash
    config_json = json.dumps(framework_config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    
    # Save
    framework_config['config_hash'] = config_hash
    with open(output_file, 'w') as f:
        json.dump(framework_config, f, indent=2)
    
    return config_hash


# Example usage
if __name__ == "__main__":
    # Pre-register framework
    framework_config = {
        'rejection_thresholds': {
            'break_even': 0.01,
            'capacity': 25_000_000,
            'turnover': 3.0,
            'cost_drag': 0.05
        },
        'cost_assumptions': {
            'commission_per_trade': 0.005,
            'half_spread': 0.001,
            'participation_rate': 0.01
        },
        'signals': ['momentum_12_1', 'volatility_breakout', 'ma_crossover']
    }
    
    config_hash = pre_register_framework(framework_config)
    print(f"Framework pre-registered with hash: {config_hash}")
    
    # Run validation
    signals = ['momentum_12_1', 'volatility_breakout', 'ma_crossover']
    results = run_out_of_sample_validation(
        signal_names=signals,
        ticker='SPY',
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2014, 12, 31),
        test_start=datetime(2015, 1, 1),
        test_end=datetime(2020, 12, 31),
        rejection_thresholds={'break_even': 0.01, 'capacity': 25_000_000}
    )
    
    print("\nValidation Results:")
    print(results[['signal', 'train_decision', 'test_actual_break_even', 'correct']])
    
    metrics = compute_validation_metrics(results)
    print("\nValidation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.2%}")

