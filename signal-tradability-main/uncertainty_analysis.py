"""
Uncertainty Quantification Module

Real quants show:
- Stability across subsamples
- Sensitivity to costs
- Confidence bands
- Regime dependence

Not just averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from decay_analysis import compute_performance_metrics, PerformanceMetrics
from tradability_analysis import compute_drawdown_sensitivity_to_costs, TradabilityMetrics


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty quantification results."""
    mean_value: float
    std_value: float
    confidence_interval: Tuple[float, float]
    percentile_5: float
    percentile_95: float
    min_value: float
    max_value: float
    n_samples: int


def bootstrap_metric(returns: pd.Series,
                     metric_func,
                     n_samples: int = 1000,
                     confidence_level: float = 0.95,
                     random_seed: Optional[int] = None) -> UncertaintyMetrics:
    """
    Bootstrap confidence intervals for any metric.
    
    Args:
        returns: Returns series
        metric_func: Function that takes returns and returns a scalar metric
        n_samples: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        UncertaintyMetrics with confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(returns)
    bootstrap_values = []
    
    for _ in range(n_samples):
        # Resample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample_returns = returns.iloc[sample_indices]
        
        try:
            metric_value = metric_func(sample_returns)
            if metric_value is not None and not np.isnan(metric_value):
                bootstrap_values.append(metric_value)
        except:
            continue
    
    if len(bootstrap_values) == 0:
        return UncertaintyMetrics(
            mean_value=0.0,
            std_value=0.0,
            confidence_interval=(0.0, 0.0),
            percentile_5=0.0,
            percentile_95=0.0,
            min_value=0.0,
            max_value=0.0,
            n_samples=0
        )
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Compute statistics
    mean_val = np.mean(bootstrap_values)
    std_val = np.std(bootstrap_values)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return UncertaintyMetrics(
        mean_value=mean_val,
        std_value=std_val,
        confidence_interval=(lower, upper),
        percentile_5=np.percentile(bootstrap_values, 5),
        percentile_95=np.percentile(bootstrap_values, 95),
        min_value=np.min(bootstrap_values),
        max_value=np.max(bootstrap_values),
        n_samples=len(bootstrap_values)
    )


def bootstrap_sharpe(returns: pd.Series,
                     n_samples: int = 1000,
                     confidence_level: float = 0.95,
                     periods_per_year: int = 252) -> UncertaintyMetrics:
    """Bootstrap confidence intervals for Sharpe ratio."""
    def compute_sharpe(rs):
        metrics = compute_performance_metrics(rs)
        if metrics.sharpe_ratio is None:
            return 0.0
        return metrics.sharpe_ratio
    
    return bootstrap_metric(returns, compute_sharpe, n_samples, confidence_level)


def bootstrap_return(returns: pd.Series,
                     n_samples: int = 1000,
                     confidence_level: float = 0.95) -> UncertaintyMetrics:
    """Bootstrap confidence intervals for mean return."""
    def compute_return(rs):
        return rs.mean()
    
    return bootstrap_metric(returns, compute_return, n_samples, confidence_level)


def compute_regime_dependence(returns: pd.Series,
                              regime_indicator: pd.Series,
                              n_regimes: int = 3) -> Dict[str, Dict]:
    """
    Compute performance across different regimes.
    
    Args:
        returns: Returns series
        regime_indicator: Series indicating regime (e.g., volatility)
        n_regimes: Number of regimes to split into
    
    Returns:
        Dictionary mapping regime names to performance metrics
    """
    # Align data
    aligned = pd.DataFrame({
        'returns': returns,
        'regime': regime_indicator
    }).dropna()
    
    if len(aligned) == 0:
        return {}
    
    # Split into regimes
    regime_quantiles = np.linspace(0, 1, n_regimes + 1)
    regime_thresholds = aligned['regime'].quantile(regime_quantiles)
    
    results = {}
    
    for i in range(n_regimes):
        lower = regime_thresholds.iloc[i]
        upper = regime_thresholds.iloc[i + 1]
        
        regime_returns = aligned[
            (aligned['regime'] >= lower) & (aligned['regime'] < upper)
        ]['returns']
        
        if len(regime_returns) < 50:
            continue
        
        metrics = compute_performance_metrics(regime_returns)
        
        # Bootstrap uncertainty
        sharpe_uncertainty = bootstrap_sharpe(regime_returns, n_samples=500)
        
        results[f'regime_{i+1}'] = {
            'n_observations': len(regime_returns),
            'mean_return': metrics.return_mean,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sharpe_ci_lower': sharpe_uncertainty.confidence_interval[0],
            'sharpe_ci_upper': sharpe_uncertainty.confidence_interval[1],
            'sharpe_std': sharpe_uncertainty.std_value,
            'regime_range': (lower, upper),
        }
    
    return results


def compute_time_varying_stability(returns: pd.Series,
                                    window_size: int = 252,
                                    step_size: int = 63) -> pd.DataFrame:
    """
    Compute rolling performance metrics to assess stability.
    
    Args:
        returns: Returns series
        window_size: Rolling window size (e.g., 252 for 1 year)
        step_size: Step size for rolling windows (e.g., 63 for quarterly)
    
    Returns:
        DataFrame with rolling metrics
    """
    results = []
    
    for start_idx in range(0, len(returns) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_returns = returns.iloc[start_idx:end_idx]
        
        if len(window_returns) < window_size:
            continue
        
        metrics = compute_performance_metrics(window_returns)
        
        results.append({
            'start_date': window_returns.index[0],
            'end_date': window_returns.index[-1],
            'sharpe_ratio': metrics.sharpe_ratio,
            'mean_return': metrics.return_mean,
            'volatility': metrics.return_std,
            'max_drawdown': metrics.max_drawdown,
            'hit_rate': metrics.hit_rate,
        })
    
    return pd.DataFrame(results)


def compute_cost_sensitivity_with_uncertainty(gross_returns: pd.Series,
                                               positions: pd.Series,
                                               cost_levels: Optional[np.ndarray] = None,
                                               n_bootstrap: int = 100,
                                               periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute cost sensitivity with confidence bands.
    
    Args:
        gross_returns: Gross strategy returns
        positions: Position series
        cost_levels: Array of cost levels to test
        n_bootstrap: Number of bootstrap samples for uncertainty
        periods_per_year: Periods per year
    
    Returns:
        DataFrame with cost levels, mean Sharpe, and confidence bands
    """
    if cost_levels is None:
        cost_levels = np.linspace(0, 0.01, 50)
    
    from transaction_costs import compute_net_returns_from_positions
    
    results = []
    
    for cost in cost_levels:
        # Compute net returns at this cost level
        net_returns = compute_net_returns_from_positions(
            gross_returns, positions,
            commission_per_trade=cost,
            half_spread=0.0,
            periods_per_year=periods_per_year
        )
        
        # Bootstrap Sharpe ratio
        sharpe_uncertainty = bootstrap_sharpe(net_returns, n_samples=n_bootstrap)
        
        results.append({
            'cost_level': cost,
            'sharpe_mean': sharpe_uncertainty.mean_value,
            'sharpe_std': sharpe_uncertainty.std_value,
            'sharpe_ci_lower': sharpe_uncertainty.confidence_interval[0],
            'sharpe_ci_upper': sharpe_uncertainty.confidence_interval[1],
            'sharpe_percentile_5': sharpe_uncertainty.percentile_5,
            'sharpe_percentile_95': sharpe_uncertainty.percentile_95,
        })
    
    return pd.DataFrame(results)


def quantify_uncertainty_comprehensive(gross_returns: pd.Series,
                                        net_returns: pd.Series,
                                        positions: pd.Series,
                                        volatility: Optional[pd.Series] = None,
                                        periods_per_year: int = 252) -> Dict:
    """
    Comprehensive uncertainty quantification.
    
    Returns:
        Dictionary with all uncertainty metrics
    """
    results = {}
    
    # 1. Bootstrap Sharpe ratios
    print("Computing bootstrap confidence intervals for Sharpe ratios...")
    gross_sharpe_uncertainty = bootstrap_sharpe(gross_returns, n_samples=1000)
    net_sharpe_uncertainty = bootstrap_sharpe(net_returns, n_samples=1000)
    
    results['gross_sharpe'] = {
        'mean': gross_sharpe_uncertainty.mean_value,
        'std': gross_sharpe_uncertainty.std_value,
        'ci_95_lower': gross_sharpe_uncertainty.confidence_interval[0],
        'ci_95_upper': gross_sharpe_uncertainty.confidence_interval[1],
    }
    
    results['net_sharpe'] = {
        'mean': net_sharpe_uncertainty.mean_value,
        'std': net_sharpe_uncertainty.std_value,
        'ci_95_lower': net_sharpe_uncertainty.confidence_interval[0],
        'ci_95_upper': net_sharpe_uncertainty.confidence_interval[1],
    }
    
    # 2. Time-varying stability
    print("Computing time-varying stability...")
    time_varying = compute_time_varying_stability(net_returns, window_size=252, step_size=63)
    results['time_varying_stability'] = {
        'sharpe_mean': time_varying['sharpe_ratio'].mean(),
        'sharpe_std': time_varying['sharpe_ratio'].std(),
        'sharpe_min': time_varying['sharpe_ratio'].min(),
        'sharpe_max': time_varying['sharpe_ratio'].max(),
        'n_windows': len(time_varying),
    }
    
    # 3. Regime dependence (if volatility provided)
    if volatility is not None:
        print("Computing regime dependence...")
        regime_results = compute_regime_dependence(net_returns, volatility, n_regimes=3)
        results['regime_dependence'] = regime_results
    
    # 4. Cost sensitivity with uncertainty
    print("Computing cost sensitivity with confidence bands...")
    cost_sensitivity = compute_cost_sensitivity_with_uncertainty(
        gross_returns, positions, n_bootstrap=100, periods_per_year=periods_per_year
    )
    results['cost_sensitivity'] = cost_sensitivity.to_dict('records')
    
    return results



