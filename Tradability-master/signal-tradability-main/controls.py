"""
Control Variables Module

Tests whether decay correlates with alternative explanations:
- Market efficiency (liquidity, volume)
- Volatility regime
- Transaction costs
- Crowding proxies

This is what makes this Level-7 - we're not just saying "it died",
we're asking WHY.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from datetime import datetime


class MarketControls:
    """Container for market control variables."""
    
    def __init__(self):
        self.liquidity_proxy: Optional[pd.Series] = None  # Volume, bid-ask spread
        self.volatility: Optional[pd.Series] = None       # Realized volatility
        self.transaction_costs: Optional[pd.Series] = None  # Turnover-based costs
        self.crowding_proxy: Optional[pd.Series] = None   # Correlations, factor loadings
    
    def compute_from_prices(self, prices: pd.Series, volumes: Optional[pd.Series] = None):
        """
        Compute control variables from price (and optionally volume) data.
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
        """
        returns = prices.pct_change().dropna()
        
        # Volatility: rolling realized volatility
        self.volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Liquidity proxy: if volumes available, use volume; otherwise use return volatility (inverse)
        if volumes is not None:
            # Higher volume = more liquid
            self.liquidity_proxy = volumes.rolling(window=20).mean()
        else:
            # Lower volatility = more liquid (crude proxy)
            self.liquidity_proxy = 1 / (1 + self.volatility)
        
        # Transaction costs proxy: based on turnover and volatility
        # Higher volatility and higher turnover = higher costs
        turnover_proxy = returns.abs().rolling(window=20).mean()
        self.transaction_costs = turnover_proxy * self.volatility
        
        # Crowding proxy: autocorrelation of returns (high autocorr = crowded)
        # When many people trade same signal, returns become more correlated
        self.crowding_proxy = returns.rolling(window=60).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 30 else np.nan
        )


def compute_control_regimes(data: pd.Series, n_regimes: int = 3) -> pd.Series:
    """
    Split data into regimes (e.g., low/med/high volatility).
    
    Args:
        data: Control variable series
        n_regimes: Number of regimes (default 3: low/med/high)
    
    Returns:
        pd.Series with regime labels (0, 1, 2, ...)
    """
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return pd.Series(dtype=int, index=data.index)
    
    # Use quantiles to define regimes
    quantiles = np.linspace(0, 1, n_regimes + 1)
    thresholds = clean_data.quantile(quantiles[1:-1])
    
    regimes = pd.Series(index=data.index, dtype=int)
    for i, threshold in enumerate(thresholds):
        regimes[data >= threshold] = i + 1
    
    return regimes


def test_decay_by_regime(pre_returns: pd.Series, post_returns: pd.Series,
                         pre_control: pd.Series, post_control: pd.Series,
                         n_regimes: int = 3) -> Dict:
    """
    Test whether decay varies by market regime.
    
    For example: Does signal decay more in high-volatility periods?
    
    Args:
        pre_returns: Pre-discovery returns
        post_returns: Post-discovery returns
        pre_control: Pre-discovery control variable
        post_control: Post-discovery control variable
        n_regimes: Number of regimes to split into
    
    Returns:
        Dictionary with results by regime
    """
    # Define regimes
    all_control = pd.concat([pre_control, post_control])
    regimes_all = compute_control_regimes(all_control, n_regimes)
    
    pre_regimes = regimes_all[pre_returns.index]
    post_regimes = regimes_all[post_returns.index]
    
    results = {}
    
    for regime in range(n_regimes):
        pre_regime_returns = pre_returns[pre_regimes == regime]
        post_regime_returns = post_returns[post_regimes == regime]
        
        if len(pre_regime_returns) < 10 or len(post_regime_returns) < 10:
            results[f'regime_{regime}'] = {
                'pre_mean': None,
                'post_mean': None,
                'decay': None,
                'sufficient_data': False
            }
            continue
        
        pre_mean = pre_regime_returns.mean()
        post_mean = post_regime_returns.mean()
        decay = post_mean - pre_mean
        
        results[f'regime_{regime}'] = {
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'decay': decay,
            'decay_pct': decay / abs(pre_mean) if abs(pre_mean) > 1e-6 else None,
            'sufficient_data': True,
            'n_pre': len(pre_regime_returns),
            'n_post': len(post_regime_returns)
        }
    
    return results


def correlate_decay_with_controls(decay_metrics: pd.Series,
                                  control_variables: Dict[str, pd.Series]) -> Dict:
    """
    Correlate decay with control variables across multiple signals.
    
    Args:
        decay_metrics: Series of decay values (index = signal names)
        control_variables: Dict mapping control names to time series
    
    Returns:
        Dictionary with correlation results
    """
    correlations = {}
    
    # For each control, compute average value and correlate with decay
    for control_name, control_series in control_variables.items():
        control_mean = control_series.mean()
        
        # Simple correlation: if we had multiple signals with different control values
        # For now, this is a placeholder structure
        correlations[control_name] = {
            'mean_value': control_mean,
            'note': 'Multi-signal correlation analysis requires multiple signals with varying control values'
        }
    
    return correlations


def compute_time_varying_decay(returns: pd.Series, discovery_date: datetime,
                               window_size: int = 252) -> pd.DataFrame:
    """
    Compute rolling decay metrics to see how decay evolves over time.
    
    Args:
        returns: Strategy returns
        discovery_date: Discovery date
        window_size: Rolling window size (default 252 = 1 year)
    
    Returns:
        DataFrame with rolling metrics
    """
    pre_returns = returns[returns.index < discovery_date]
    post_returns = returns[returns.index >= discovery_date]
    
    if len(post_returns) < window_size:
        return pd.DataFrame()
    
    # Rolling window analysis on post-discovery period
    rolling_metrics = []
    
    for i in range(window_size, len(post_returns) + 1):
        window_returns = post_returns.iloc[i - window_size:i]
        
        window_mean = window_returns.mean()
        window_std = window_returns.std()
        window_sharpe = window_mean / window_std * np.sqrt(252) if window_std > 0 else 0
        
        # Compare to pre-discovery baseline
        if len(pre_returns) >= window_size:
            baseline_returns = pre_returns.iloc[-window_size:]
            baseline_mean = baseline_returns.mean()
            baseline_sharpe = baseline_mean / baseline_returns.std() * np.sqrt(252) if baseline_returns.std() > 0 else 0
            
            decay = window_sharpe - baseline_sharpe
        else:
            baseline_mean = pre_returns.mean()
            baseline_sharpe = pre_returns.mean() / pre_returns.std() * np.sqrt(252) if pre_returns.std() > 0 else 0
            decay = window_sharpe - baseline_sharpe
        
        rolling_metrics.append({
            'date': window_returns.index[-1],
            'rolling_sharpe': window_sharpe,
            'decay_vs_baseline': decay,
            'rolling_mean': window_mean
        })
    
    return pd.DataFrame(rolling_metrics).set_index('date')

