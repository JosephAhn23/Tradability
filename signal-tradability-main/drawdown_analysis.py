"""
Drawdown Analysis Module

Computes drawdown metrics including:
- Maximum drawdown (gross and net)
- Cost-adjusted drawdown increase
- Recovery time
- Conditional Value at Risk (CVaR)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats


def compute_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
    """
    Compute maximum drawdown and drawdown series.
    
    Args:
        returns: Series of returns
    
    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0.0, pd.Series(dtype=float)
    
    # Cumulative returns
    cumulative = (1 + returns_clean).cumprod()
    
    # Running maximum
    running_max = cumulative.expanding().max()
    
    # Drawdown
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    
    return max_dd, drawdown


def compute_recovery_time(
    returns: pd.Series,
    drawdown_series: Optional[pd.Series] = None
) -> Dict:
    """
    Compute recovery time from maximum drawdown.
    
    Args:
        returns: Series of returns
        drawdown_series: Optional pre-computed drawdown series
    
    Returns:
        Dict with recovery metrics
    """
    if drawdown_series is None:
        _, drawdown_series = compute_max_drawdown(returns)
    
    if len(drawdown_series) == 0:
        return {
            'recovery_time_days': None,
            'recovery_time_years': None,
            'max_dd_date': None,
            'recovery_date': None
        }
    
    # Find maximum drawdown date
    max_dd_idx = drawdown_series.idxmin()
    max_dd_date = max_dd_idx if hasattr(max_dd_idx, 'date') else None
    
    # Find recovery (when drawdown returns to 0 or above)
    recovery_mask = drawdown_series.loc[drawdown_series.index >= max_dd_idx] >= 0
    recovery_dates = recovery_mask[recovery_mask].index
    
    if len(recovery_dates) > 0:
        recovery_date = recovery_dates[0]
        recovery_time = (recovery_date - max_dd_idx).days if hasattr(recovery_date, '__sub__') else None
        recovery_time_years = recovery_time / 252.0 if recovery_time else None
    else:
        recovery_date = None
        recovery_time = None
        recovery_time_years = None
    
    return {
        'recovery_time_days': recovery_time,
        'recovery_time_years': recovery_time_years,
        'max_dd_date': max_dd_date,
        'recovery_date': recovery_date,
        'recovered': recovery_date is not None
    }


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> float:
    """
    Compute Conditional Value at Risk (CVaR) at specified confidence level.
    
    CVaR = Expected loss given that loss exceeds VaR threshold.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (default 0.05 = 5% tail)
    
    Returns:
        CVaR (negative value, represents expected loss)
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return None
    
    # VaR threshold
    var_threshold = np.percentile(returns_clean, confidence_level * 100)
    
    # CVaR: mean of returns below VaR threshold
    tail_returns = returns_clean[returns_clean <= var_threshold]
    
    if len(tail_returns) == 0:
        return None
    
    cvar = tail_returns.mean()
    
    return cvar


def compute_drawdown_metrics(
    gross_returns: pd.Series,
    net_returns: pd.Series
) -> Dict:
    """
    Compute comprehensive drawdown metrics comparing gross vs net.
    
    Args:
        gross_returns: Gross strategy returns
        net_returns: Net returns after costs
    
    Returns:
        Dict with all drawdown metrics
    """
    # Gross drawdown
    gross_max_dd, gross_dd_series = compute_max_drawdown(gross_returns)
    
    # Net drawdown
    net_max_dd, net_dd_series = compute_max_drawdown(net_returns)
    
    # Drawdown increase
    if gross_max_dd != 0:
        dd_increase_pct = ((net_max_dd - gross_max_dd) / abs(gross_max_dd)) * 100
    else:
        dd_increase_pct = None
    
    # Recovery time
    gross_recovery = compute_recovery_time(gross_returns, gross_dd_series)
    net_recovery = compute_recovery_time(net_returns, net_dd_series)
    
    # CVaR
    gross_cvar_5pct = compute_cvar(gross_returns, 0.05)
    gross_cvar_1pct = compute_cvar(gross_returns, 0.01)
    net_cvar_5pct = compute_cvar(net_returns, 0.05)
    net_cvar_1pct = compute_cvar(net_returns, 0.01)
    
    return {
        'gross_max_drawdown': gross_max_dd,
        'net_max_drawdown': net_max_dd,
        'drawdown_increase': net_max_dd - gross_max_dd,
        'drawdown_increase_pct': dd_increase_pct,
        'gross_recovery_time_days': gross_recovery['recovery_time_days'],
        'gross_recovery_time_years': gross_recovery['recovery_time_years'],
        'net_recovery_time_days': net_recovery['recovery_time_days'],
        'net_recovery_time_years': net_recovery['recovery_time_years'],
        'gross_cvar_5pct': gross_cvar_5pct,
        'gross_cvar_1pct': gross_cvar_1pct,
        'net_cvar_5pct': net_cvar_5pct,
        'net_cvar_1pct': net_cvar_1pct,
        'gross_recovered': gross_recovery['recovered'],
        'net_recovered': net_recovery['recovered']
    }


# Example usage
if __name__ == "__main__":
    # Sample returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    gross_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005, index=dates)
    net_returns = gross_returns - 0.0001  # Simple cost drag
    
    metrics = compute_drawdown_metrics(gross_returns, net_returns)
    print("Drawdown Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

