"""
Regime Analysis Module

Partitions data into market regimes and computes Sharpe in each regime.
Addresses critical review demand for regime dependence analysis.

Regimes:
- Bull/Bear (based on market returns)
- High Vol/Low Vol (based on VIX or realized volatility)
- 4 total regimes: Bull/High Vol, Bull/Low Vol, Bear/High Vol, Bear/Low Vol
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
from scipy import stats

from statistical_rigor import compute_sharpe_with_ci
from decay_analysis import compute_performance_metrics


def partition_regimes(
    returns: pd.Series,
    market_returns: Optional[pd.Series] = None,
    vix: Optional[pd.Series] = None,
    volatility: Optional[pd.Series] = None,
    bull_threshold: float = 0.10,
    vol_threshold: float = 25.0
) -> pd.Series:
    """
    Partition data into 4 regimes:
    1. Bull/High Vol
    2. Bull/Low Vol
    3. Bear/High Vol
    4. Bear/Low Vol
    
    Args:
        returns: Strategy returns
        market_returns: Market returns (for bull/bear classification)
        vix: VIX series (for volatility classification)
        volatility: Realized volatility (alternative to VIX)
        bull_threshold: Annual return threshold for bull market (default 10%)
        vol_threshold: VIX threshold for high vol (default 25)
    
    Returns:
        Series with regime labels: 'Bull_HighVol', 'Bull_LowVol', 'Bear_HighVol', 'Bear_LowVol'
    """
    # Align all series
    aligned = pd.DataFrame({'returns': returns})
    
    # Bull/Bear classification
    if market_returns is not None:
        aligned['market'] = market_returns.reindex(returns.index, method='ffill')
        # Annualized market return (rolling 252-day)
        market_annual = aligned['market'].rolling(252, min_periods=60).mean() * 252
        aligned['is_bull'] = market_annual > bull_threshold
    else:
        # Use strategy returns as proxy
        strategy_annual = aligned['returns'].rolling(252, min_periods=60).mean() * 252
        aligned['is_bull'] = strategy_annual > bull_threshold
    
    # Volatility classification
    if vix is not None:
        aligned['vix'] = vix.reindex(returns.index, method='ffill')
        aligned['is_high_vol'] = aligned['vix'] > vol_threshold
    elif volatility is not None:
        aligned['vol'] = volatility.reindex(returns.index, method='ffill')
        # Use median as threshold
        vol_threshold_actual = aligned['vol'].median()
        aligned['is_high_vol'] = aligned['vol'] > vol_threshold_actual
    else:
        # Use realized volatility from returns
        aligned['vol'] = aligned['returns'].rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_threshold_actual = aligned['vol'].median()
        aligned['is_high_vol'] = aligned['vol'] > vol_threshold_actual
    
    # Create regime labels
    aligned['regime'] = 'Unknown'
    aligned.loc[aligned['is_bull'] & aligned['is_high_vol'], 'regime'] = 'Bull_HighVol'
    aligned.loc[aligned['is_bull'] & ~aligned['is_high_vol'], 'regime'] = 'Bull_LowVol'
    aligned.loc[~aligned['is_bull'] & aligned['is_high_vol'], 'regime'] = 'Bear_HighVol'
    aligned.loc[~aligned['is_bull'] & ~aligned['is_high_vol'], 'regime'] = 'Bear_LowVol'
    
    return aligned['regime']


def compute_regime_sharpe(
    returns: pd.Series,
    regimes: pd.Series,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Compute Sharpe ratio in each regime with confidence intervals.
    
    Args:
        returns: Strategy returns
        regimes: Regime labels
        periods_per_year: Number of periods per year
    
    Returns:
        DataFrame with Sharpe, CI, and sample size for each regime
    """
    results = []
    
    for regime in ['Bull_HighVol', 'Bull_LowVol', 'Bear_HighVol', 'Bear_LowVol']:
        regime_returns = returns[regimes == regime].dropna()
        
        if len(regime_returns) < 20:  # Need minimum observations
            results.append({
                'regime': regime,
                'sharpe': None,
                'ci_lower': None,
                'ci_upper': None,
                'se': None,
                'n_observations': len(regime_returns),
                'mean_return': None,
                'std_return': None
            })
            continue
        
        sharpe, ci, se = compute_sharpe_with_ci(regime_returns, periods_per_year)
        metrics = compute_performance_metrics(regime_returns)
        
        results.append({
            'regime': regime,
            'sharpe': sharpe,
            'ci_lower': ci[0] if ci else None,
            'ci_upper': ci[1] if ci else None,
            'se': se,
            'n_observations': len(regime_returns),
            'mean_return': metrics.return_mean,
            'std_return': metrics.return_std
        })
    
    return pd.DataFrame(results)


def compute_regime_sensitivity(
    regime_results: pd.DataFrame
) -> Dict:
    """
    Compute regime sensitivity metrics.
    
    Returns:
        Dict with:
        - max_sharpe: Maximum Sharpe across regimes
        - min_sharpe: Minimum Sharpe across regimes
        - sensitivity_ratio: max / min (if both positive) or max - min (if sign flips)
        - sign_flip: Whether Sharpe flips sign across regimes
    """
    sharpe_values = regime_results['sharpe'].dropna()
    
    if len(sharpe_values) == 0:
        return {
            'max_sharpe': None,
            'min_sharpe': None,
            'sensitivity_ratio': None,
            'sign_flip': None,
            'regime_fragile': None
        }
    
    max_sharpe = sharpe_values.max()
    min_sharpe = sharpe_values.min()
    
    # Check for sign flip
    sign_flip = (max_sharpe > 0 and min_sharpe < 0) or (max_sharpe < 0 and min_sharpe > 0)
    
    # Compute sensitivity ratio
    if sign_flip:
        sensitivity_ratio = abs(max_sharpe - min_sharpe)
    elif min_sharpe > 0:
        sensitivity_ratio = max_sharpe / min_sharpe
    elif max_sharpe < 0:
        sensitivity_ratio = min_sharpe / max_sharpe  # Both negative
    else:
        sensitivity_ratio = None
    
    # Framework is fragile if ratio > 2.0 or sign flips
    regime_fragile = (sensitivity_ratio is not None and sensitivity_ratio > 2.0) or sign_flip
    
    return {
        'max_sharpe': max_sharpe,
        'min_sharpe': min_sharpe,
        'sensitivity_ratio': sensitivity_ratio,
        'sign_flip': sign_flip,
        'regime_fragile': regime_fragile
    }


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2010-01-01', periods=2520, freq='D')
    np.random.seed(42)
    
    # Simulate returns with regime dependence
    returns = pd.Series(np.random.randn(2520) * 0.01, index=dates)
    market_returns = pd.Series(np.random.randn(2520) * 0.008, index=dates)
    vix = pd.Series(15 + np.random.randn(2520) * 5, index=dates).clip(lower=10)
    
    # Partition regimes
    regimes = partition_regimes(returns, market_returns, vix)
    
    # Compute Sharpe in each regime
    regime_results = compute_regime_sharpe(returns, regimes)
    print("Regime Analysis Results:")
    print(regime_results.to_string())
    
    # Compute sensitivity
    sensitivity = compute_regime_sensitivity(regime_results)
    print("\nRegime Sensitivity:")
    print(f"Max Sharpe: {sensitivity['max_sharpe']:.4f}")
    print(f"Min Sharpe: {sensitivity['min_sharpe']:.4f}")
    print(f"Sensitivity Ratio: {sensitivity['sensitivity_ratio']:.2f}")
    print(f"Sign Flip: {sensitivity['sign_flip']}")
    print(f"Regime Fragile: {sensitivity['regime_fragile']}")

