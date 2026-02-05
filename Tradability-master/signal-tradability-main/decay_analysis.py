"""
Decay Analysis Module

Measures signal performance pre- vs post-discovery.
Key metrics: Sharpe, hit rate, drawdown, stability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.sharpe_ratio: Optional[float] = None
        self.hit_rate: Optional[float] = None
        self.max_drawdown: Optional[float] = None
        self.return_mean: Optional[float] = None
        self.return_std: Optional[float] = None
        self.total_return: Optional[float] = None
        self.num_observations: int = 0
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy export."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'hit_rate': self.hit_rate,
            'max_drawdown': self.max_drawdown,
            'return_mean': self.return_mean,
            'return_std': self.return_std,
            'total_return': self.total_return,
            'num_observations': self.num_observations,
            'start_date': self.start_date,
            'end_date': self.end_date,
        }
    
    def __repr__(self):
        sharpe_str = f"{self.sharpe_ratio:.3f}" if self.sharpe_ratio else 'N/A'
        hit_rate_str = f"{self.hit_rate:.1%}" if self.hit_rate else 'N/A'
        max_dd_str = f"{self.max_drawdown:.1%}" if self.max_drawdown else 'N/A'
        return_str = f"{self.return_mean:.3f}" if self.return_mean else 'N/A'
        return f"""PerformanceMetrics(
    Sharpe: {sharpe_str},
    Hit Rate: {hit_rate_str},
    Max DD: {max_dd_str},
    Mean Return: {return_str},
    Observations: {self.num_observations}
)"""


def compute_returns(signals: pd.Series, forward_returns: pd.Series, 
                   quantile: float = 0.5) -> pd.Series:
    """
    Convert signal to returns using quantile-based portfolios.
    
    Args:
        signals: Signal values
        forward_returns: Forward-looking returns (aligned with signal timing)
        quantile: Threshold for long/short (e.g., 0.5 = median split)
    
    Returns:
        pd.Series: Strategy returns (with timezone-naive index)
    """
    # Ensure both inputs have timezone-naive indices
    if signals.index.tz is not None:
        signals = signals.copy()
        signals.index = signals.index.tz_convert('UTC').tz_localize(None)
    if forward_returns.index.tz is not None:
        forward_returns = forward_returns.copy()
        forward_returns.index = forward_returns.index.tz_convert('UTC').tz_localize(None)
    
    # Align indices
    aligned_data = pd.DataFrame({
        'signal': signals,
        'forward_return': forward_returns
    }).dropna()
    
    if len(aligned_data) == 0:
        return pd.Series(dtype=float)
    
    # Long top quantile, short bottom quantile
    signal_quantile = aligned_data['signal'].quantile(quantile)
    
    # Simple binary signal: 1 if above median, -1 if below
    long_short = np.where(aligned_data['signal'] > signal_quantile, 1, -1)
    
    # Strategy returns
    strategy_returns = long_short * aligned_data['forward_return']
    
    # Ensure the returned index is timezone-naive
    result_index = aligned_data.index
    if result_index.tz is not None:
        result_index = result_index.tz_convert('UTC').tz_localize(None)
    
    return pd.Series(strategy_returns, index=result_index)


def compute_performance_metrics(returns: pd.Series) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.
    
    Args:
        returns: Strategy returns series
    
    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()
    
    if len(returns) == 0 or returns.isna().all():
        return metrics
    
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return metrics
    
    metrics.num_observations = len(returns_clean)
    metrics.start_date = returns_clean.index[0]
    metrics.end_date = returns_clean.index[-1]
    
    # Basic statistics
    metrics.return_mean = returns_clean.mean()
    metrics.return_std = returns_clean.std()
    metrics.total_return = (1 + returns_clean).prod() - 1
    
    # Sharpe ratio (annualized if daily data)
    if metrics.return_std > 0:
        # Assume daily returns, annualize
        if len(returns_clean) > 252:
            periods_per_year = 252
        else:
            periods_per_year = len(returns_clean)  # Use actual if less than year
        
        sharpe = metrics.return_mean / metrics.return_std * np.sqrt(periods_per_year)
        metrics.sharpe_ratio = sharpe
    
    # Hit rate (percentage of positive returns)
    metrics.hit_rate = (returns_clean > 0).mean()
    
    # Maximum drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics.max_drawdown = drawdown.min()
    
    return metrics


def split_pre_post(returns: pd.Series, discovery_date: datetime) -> Tuple[pd.Series, pd.Series]:
    """
    Split returns into pre- and post-discovery periods.
    
    Args:
        returns: Strategy returns series
        discovery_date: Discovery date (signals before this are "pre")
    
    Returns:
        Tuple of (pre_discovery_returns, post_discovery_returns)
    """
    # Defensive: Remove timezone from returns index if present
    # Create a timezone-naive copy of the index
    if returns.index.tz is not None:
        returns = returns.copy()
        # Convert timezone-aware to naive by converting to UTC first, then removing timezone
        returns.index = returns.index.tz_convert('UTC').tz_localize(None)
    
    # Convert discovery_date to pandas Timestamp (ensure it's naive)
    discovery_date = pd.Timestamp(discovery_date)
    if discovery_date.tz is not None:
        discovery_date = discovery_date.tz_convert('UTC').tz_localize(None)
    else:
        discovery_date = pd.Timestamp(discovery_date, tz=None)
    
    pre_returns = returns[returns.index < discovery_date]
    post_returns = returns[returns.index >= discovery_date]
    
    return pre_returns, post_returns


def compute_decay_stats(pre_metrics: PerformanceMetrics, 
                       post_metrics: PerformanceMetrics) -> Dict:
    """
    Compute decay statistics comparing pre vs post.
    
    Returns:
        Dictionary with decay metrics
    """
    decay_stats = {
        'sharpe_decay': None,
        'sharpe_decay_pct': None,
        'hit_rate_decay': None,
        'hit_rate_decay_pct': None,
        'max_dd_deterioration': None,
        'return_decay': None,
        'return_decay_pct': None,
        'statistically_significant': None,
    }
    
    # Sharpe decay
    if pre_metrics.sharpe_ratio is not None and post_metrics.sharpe_ratio is not None:
        decay_stats['sharpe_decay'] = post_metrics.sharpe_ratio - pre_metrics.sharpe_ratio
        if abs(pre_metrics.sharpe_ratio) > 1e-6:
            decay_stats['sharpe_decay_pct'] = decay_stats['sharpe_decay'] / abs(pre_metrics.sharpe_ratio)
    
    # Hit rate decay
    if pre_metrics.hit_rate is not None and post_metrics.hit_rate is not None:
        decay_stats['hit_rate_decay'] = post_metrics.hit_rate - pre_metrics.hit_rate
        if pre_metrics.hit_rate > 0:
            decay_stats['hit_rate_decay_pct'] = decay_stats['hit_rate_decay'] / pre_metrics.hit_rate
    
    # Drawdown deterioration (more negative is worse)
    if pre_metrics.max_drawdown is not None and post_metrics.max_drawdown is not None:
        decay_stats['max_dd_deterioration'] = post_metrics.max_drawdown - pre_metrics.max_drawdown
    
    # Return decay
    if pre_metrics.return_mean is not None and post_metrics.return_mean is not None:
        decay_stats['return_decay'] = post_metrics.return_mean - pre_metrics.return_mean
        if abs(pre_metrics.return_mean) > 1e-6:
            decay_stats['return_decay_pct'] = decay_stats['return_decay'] / abs(pre_metrics.return_mean)
    
    return decay_stats


def test_decay_significance(pre_returns: pd.Series, post_returns: pd.Series) -> Dict:
    """
    Test statistical significance of decay.
    
    Uses Mann-Whitney U test (non-parametric) to test if post returns
    are significantly lower than pre returns.
    
    Returns:
        Dictionary with test results
    """
    pre_clean = pre_returns.dropna()
    post_clean = post_returns.dropna()
    
    if len(pre_clean) < 10 or len(post_clean) < 10:
        return {
            'statistic': None,
            'pvalue': None,
            'significant_at_5pct': None,
            'test_name': 'Mann-Whitney U'
        }
    
    # Mann-Whitney U test (one-sided: H0: post >= pre, H1: post < pre)
    statistic, pvalue = stats.mannwhitneyu(
        post_clean, pre_clean, 
        alternative='less'  # Testing if post is less than pre
    )
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant_at_5pct': pvalue < 0.05,
        'significant_at_1pct': pvalue < 0.01,
        'test_name': 'Mann-Whitney U (one-sided)'
    }

