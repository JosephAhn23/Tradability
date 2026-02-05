"""
Statistical Rigor Module

Implements proper statistical measures with confidence intervals and corrections:
- Sharpe ratio with confidence intervals (Lo 2002)
- Multiple testing correction (Bonferroni, FDR)
- Information Coefficient (IC) calculations
- Statistical tests for decay significance

References:
- Lo, A. W. (2002). The statistics of Sharpe ratios. Financial Analysts Journal, 58(4), 36-52.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society, 57(1), 289-300.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy import stats
from scipy.stats import spearmanr


def compute_sharpe_with_ci(
    returns: pd.Series,
    periods_per_year: int = 252,
    confidence: float = 0.95
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute Sharpe ratio with confidence interval (Lo 2002).
    
    Standard error formula from Lo (2002):
    SE(Sharpe) = sqrt((1 + 0.5 * Sharpe^2) / n)
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (for annualization)
        confidence: Confidence level (default 0.95 = 95%)
    
    Returns:
        Tuple of (sharpe_ratio, (ci_lower, ci_upper), standard_error)
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0 or returns_clean.std() == 0:
        return None, (None, None), None
    
    n = len(returns_clean)
    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    
    # Annualized Sharpe
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    # Standard error (Lo 2002)
    se = np.sqrt((1 + 0.5 * sharpe**2) / n)
    
    # Confidence interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = sharpe - z_score * se
    ci_upper = sharpe + z_score * se
    
    return sharpe, (ci_lower, ci_upper), se


def compute_sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Compute Sortino ratio (downside deviation only).
    
    Sortino = (Mean Return - Target) / Downside Deviation
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
        target_return: Target return (default 0.0)
    
    Returns:
        Sortino ratio (annualized)
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return None
    
    mean_return = returns_clean.mean()
    
    # Downside deviation (only negative returns)
    downside_returns = returns_clean[returns_clean < target_return]
    if len(downside_returns) == 0:
        return np.inf if mean_return > target_return else None
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return None
    
    # Annualized Sortino
    sortino = (mean_return - target_return) / downside_std * np.sqrt(periods_per_year)
    
    return sortino


def compute_information_coefficient(
    signals: pd.Series,
    forward_returns: pd.Series,
    method: str = 'spearman'
) -> Tuple[float, float]:
    """
    Compute Information Coefficient (IC) - correlation between signal and forward returns.
    
    IC measures signal strength. IC > 0.05 is considered meaningful.
    
    Args:
        signals: Series of signal values
        forward_returns: Series of forward returns
        method: Correlation method ('spearman' or 'pearson')
    
    Returns:
        Tuple of (ic, p_value)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({
        'signal': signals,
        'return': forward_returns
    }).dropna()
    
    if len(aligned) < 10:
        return None, None
    
    if method == 'spearman':
        ic, p_value = spearmanr(aligned['signal'], aligned['return'])
    else:
        ic, p_value = stats.pearsonr(aligned['signal'], aligned['return'])
    
    return ic, p_value


def compute_ic_with_confidence(
    signals: pd.Series,
    forward_returns: pd.Series,
    confidence: float = 0.95,
    method: str = 'spearman'
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute IC with confidence interval using bootstrap.
    
    Args:
        signals: Series of signal values
        forward_returns: Series of forward returns
        confidence: Confidence level
        method: Correlation method
    
    Returns:
        Tuple of (ic, (ci_lower, ci_upper), standard_error)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({
        'signal': signals,
        'return': forward_returns
    }).dropna()
    
    if len(aligned) < 10:
        return None, (None, None), None
    
    # Bootstrap IC
    n_bootstrap = 1000
    ic_samples = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(aligned), size=len(aligned), replace=True)
        sample = aligned.iloc[sample_idx]
        
        if method == 'spearman':
            ic_sample, _ = spearmanr(sample['signal'], sample['return'])
        else:
            ic_sample, _ = stats.pearsonr(sample['signal'], sample['return'])
        
        if not np.isnan(ic_sample):
            ic_samples.append(ic_sample)
    
    if len(ic_samples) == 0:
        return None, (None, None), None
    
    ic_samples = np.array(ic_samples)
    ic_mean = np.mean(ic_samples)
    ic_std = np.std(ic_samples)
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(ic_samples, 100 * alpha / 2)
    ci_upper = np.percentile(ic_samples, 100 * (1 - alpha / 2))
    
    return ic_mean, (ci_lower, ci_upper), ic_std


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple testing.
    
    Adjusted threshold: alpha / n_tests
    
    Args:
        p_values: List of p-values
        alpha: Original significance level (default 0.05)
    
    Returns:
        Tuple of (significant_flags, adjusted_threshold)
    """
    n_tests = len(p_values)
    adjusted_threshold = alpha / n_tests
    
    significant = [p < adjusted_threshold for p in p_values]
    
    return significant, adjusted_threshold


def benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """
    Apply Benjamini-Hochberg procedure for False Discovery Rate control.
    
    Args:
        p_values: List of p-values
        alpha: FDR level (default 0.05)
    
    Returns:
        Tuple of (significant_flags, adjusted_p_values)
    """
    n = len(p_values)
    p_array = np.array(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_array)
    sorted_p = p_array[sorted_idx]
    
    # Compute adjusted thresholds
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    
    # Find significant tests
    significant = adjusted_p < alpha
    
    return significant.tolist(), adjusted_p.tolist()


def test_decay_significance(
    pre_returns: pd.Series,
    post_returns: pd.Series,
    test_type: str = 'mann_whitney'
) -> dict:
    """
    Test if performance decay is statistically significant.
    
    Args:
        pre_returns: Returns before discovery/change
        post_returns: Returns after discovery/change
        test_type: Test type ('mann_whitney' or 't_test')
    
    Returns:
        Dictionary with test results
    """
    pre_clean = pre_returns.dropna()
    post_clean = post_returns.dropna()
    
    if len(pre_clean) < 10 or len(post_clean) < 10:
        return {
            'statistic': None,
            'p_value': None,
            'significant': None,
            'test_name': test_type
        }
    
    if test_type == 'mann_whitney':
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            post_clean, pre_clean,
            alternative='less'  # Testing if post < pre
        )
        test_name = 'Mann-Whitney U (one-sided)'
    
    elif test_type == 't_test':
        # T-test (parametric)
        statistic, p_value = stats.ttest_ind(post_clean, pre_clean, alternative='less')
        test_name = 'T-test (one-sided)'
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'test_name': test_name
    }


def compute_theoretical_return_from_ic(
    ic: float,
    volatility: float,
    periods_per_year: int = 252
) -> float:
    """
    Compute theoretical expected return from Information Coefficient.
    
    Formula: E[R] = IC * Volatility * sqrt(periods_per_year)
    
    This gives the expected annualized return if signal has IC correlation.
    
    Args:
        ic: Information Coefficient
        volatility: Annualized volatility
        periods_per_year: Number of periods per year
    
    Returns:
        Theoretical annualized return
    """
    return ic * volatility * np.sqrt(periods_per_year)


# Example usage
if __name__ == "__main__":
    # Sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
    
    # Sharpe with CI
    sharpe, ci, se = compute_sharpe_with_ci(returns)
    print(f"Sharpe: {sharpe:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], SE: {se:.4f}")
    
    # Sortino
    sortino = compute_sortino_ratio(returns)
    print(f"Sortino: {sortino:.4f}")
    
    # IC
    signals = pd.Series(np.random.randn(252))
    forward_returns = pd.Series(np.random.randn(252) * 0.01)
    ic, p = compute_information_coefficient(signals, forward_returns)
    print(f"IC: {ic:.4f}, p-value: {p:.4f}")
    
    # Multiple testing
    p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
    significant_bonf, threshold = bonferroni_correction(p_values)
    significant_fdr, adjusted_p = benjamini_hochberg_fdr(p_values)
    print(f"Bonferroni: {significant_bonf}, threshold: {threshold:.4f}")
    print(f"FDR: {significant_fdr}")

