"""
Tradability Analysis Module

Main integration module for economic decay analysis.
Computes gross vs net performance, break-even costs, and economic viability.

Now includes proper market impact models (Almgren-Chriss, Kyle) and statistical rigor.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime

from decay_analysis import compute_performance_metrics, PerformanceMetrics
from transaction_costs import (
    compute_total_explicit_costs,
    compute_net_returns_from_positions,
    compute_annual_turnover
)
from slippage import (
    compute_total_slippage,
    compute_net_returns_with_slippage
)
from capacity import estimate_maximum_viable_capital
from market_impact import compute_total_market_impact
from statistical_rigor import compute_sharpe_with_ci, compute_information_coefficient


class TradabilityMetrics:
    """Container for tradability metrics."""
    
    def __init__(self):
        self.gross_metrics: Optional[PerformanceMetrics] = None
        self.net_metrics: Optional[PerformanceMetrics] = None
        self.break_even_cost: Optional[float] = None
        self.annual_turnover: Optional[float] = None
        self.cost_drag: Optional[float] = None
        self.max_viable_capacity: Optional[float] = None
        self.hit_rate_survival: Optional[float] = None  # Hit rate before/after costs
        self.pnl_collapse: Optional[float] = None  # PnL before/after costs
        # New: Statistical rigor metrics
        self.gross_sharpe_ci: Optional[Tuple[float, float]] = None  # (lower, upper)
        self.net_sharpe_ci: Optional[Tuple[float, float]] = None
        self.information_coefficient: Optional[float] = None
        self.ic_ci: Optional[Tuple[float, float]] = None
        # Drawdown metrics
        self.gross_max_drawdown: Optional[float] = None
        self.net_max_drawdown: Optional[float] = None
        self.drawdown_increase_pct: Optional[float] = None
        self.gross_recovery_time_years: Optional[float] = None
        self.net_recovery_time_years: Optional[float] = None
        self.gross_cvar_5pct: Optional[float] = None
        self.net_cvar_5pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy export."""
        return {
            'gross_sharpe': self.gross_metrics.sharpe_ratio if self.gross_metrics else None,
            'net_sharpe': self.net_metrics.sharpe_ratio if self.net_metrics else None,
            'gross_return': self.gross_metrics.return_mean if self.gross_metrics else None,
            'net_return': self.net_metrics.return_mean if self.net_metrics else None,
            'break_even_cost': self.break_even_cost,
            'annual_turnover': self.annual_turnover,
            'cost_drag': self.cost_drag,
            'max_viable_capacity': self.max_viable_capacity,
            'hit_rate_survival': self.hit_rate_survival,
            'pnl_collapse': self.pnl_collapse,
        }


def compute_positions_from_returns(returns: pd.Series, 
                                   signals: pd.Series,
                                   quantile: float = 0.5) -> pd.Series:
    """
    Reconstruct positions from signals (used for cost computation).
    
    Args:
        returns: Strategy returns series (for alignment)
        signals: Signal values series
        quantile: Threshold for long/short positions
    
    Returns:
        Series of positions (-1, 0, 1)
    """
    # Align signals with returns
    aligned_signals = signals.reindex(returns.index, method='ffill')
    
    # Compute threshold
    signal_threshold = aligned_signals.quantile(quantile)
    
    # Binary positions: 1 if above threshold, -1 if below
    positions = np.where(aligned_signals > signal_threshold, 1, -1)
    
    return pd.Series(positions, index=returns.index)


def compute_gross_vs_net_performance(gross_returns: pd.Series,
                                     positions: pd.Series,
                                     commission_per_trade: float = 0.005,
                                     half_spread: float = 0.001,
                                     volatility: Optional[pd.Series] = None,
                                     volumes: Optional[pd.Series] = None,
                                     prices: Optional[pd.Series] = None,
                                     vol_impact_coefficient: float = 0.1,
                                     vol_impact_coefficient2: float = 0.0001,
                                     periods_per_year: int = 252) -> TradabilityMetrics:
    """
    Compute gross vs net performance after all costs.
    
    Args:
        gross_returns: Gross strategy returns (before costs)
        positions: Series of position values
        commission_per_trade: Commission cost per trade
        half_spread: Half of bid-ask spread
        volatility: Optional volatility series
        volumes: Optional volume series
        prices: Optional price series
        vol_impact_coefficient: Volatility impact coefficient
        vol_impact_coefficient2: Volume impact coefficient
        periods_per_year: Number of periods per year
    
    Returns:
        TradabilityMetrics object with gross and net performance
    """
    metrics = TradabilityMetrics()
    
    # Compute gross metrics
    metrics.gross_metrics = compute_performance_metrics(gross_returns)
    
    # Compute Sharpe with confidence intervals (Lo 2002)
    if metrics.gross_metrics.sharpe_ratio is not None:
        _, gross_ci, _ = compute_sharpe_with_ci(gross_returns, periods_per_year)
        metrics.gross_sharpe_ci = gross_ci
    
    # Compute explicit costs
    explicit_costs = compute_total_explicit_costs(
        positions, commission_per_trade, half_spread, prices, periods_per_year
    )
    explicit_costs_aligned = explicit_costs.reindex(gross_returns.index, method='ffill').fillna(0)
    
    # Compute slippage
    slippage = compute_total_slippage(
        positions, volatility, volumes, prices,
        vol_impact_coefficient, vol_impact_coefficient2,
        periods_per_year
    )
    slippage_aligned = slippage.reindex(gross_returns.index, method='ffill').fillna(0)
    
    # Total costs
    total_costs = explicit_costs_aligned + slippage_aligned
    
    # Net returns
    net_returns = gross_returns - total_costs
    
    # Compute net metrics
    metrics.net_metrics = compute_performance_metrics(net_returns)
    
    # Compute Sharpe with confidence intervals (Lo 2002)
    if metrics.net_metrics.sharpe_ratio is not None:
        _, net_ci, _ = compute_sharpe_with_ci(net_returns, periods_per_year)
        metrics.net_sharpe_ci = net_ci
    
    # Annual turnover
    metrics.annual_turnover = compute_annual_turnover(positions, periods_per_year)
    
    # Cost drag (average cost per period, annualized)
    avg_cost_per_period = total_costs.mean()
    metrics.cost_drag = avg_cost_per_period * periods_per_year
    
    # Hit rate survival (hit rate before vs after)
    if metrics.gross_metrics.hit_rate is not None and metrics.net_metrics.hit_rate is not None:
        metrics.hit_rate_survival = metrics.net_metrics.hit_rate / metrics.gross_metrics.hit_rate
    
    # PnL collapse (return before vs after)
    if metrics.gross_metrics.return_mean is not None and metrics.net_metrics.return_mean is not None:
        if abs(metrics.gross_metrics.return_mean) > 1e-6:
            metrics.pnl_collapse = metrics.net_metrics.return_mean / metrics.gross_metrics.return_mean
    
    return metrics


def compute_break_even_cost(gross_returns: pd.Series,
                            positions: pd.Series,
                            periods_per_year: int = 252,
                            tolerance: float = 1e-6) -> Optional[float]:
    """
    Compute break-even transaction cost (cost level where net edge ≈ 0).
    
    Args:
        gross_returns: Gross strategy returns
        positions: Series of position values
        periods_per_year: Number of periods per year
        tolerance: Tolerance for break-even (default 1e-6)
    
    Returns:
        Break-even cost per trade (as fraction), or None if unprofitable even at zero cost (model failure)
    """
    import numpy as np
    from typing import Optional
    
    # First check: if unprofitable at zero cost, break-even is undefined (model failure)
    net_returns_zero_cost = compute_net_returns_from_positions(
        gross_returns, positions,
        commission_per_trade=0.0,
        half_spread=0.0,
        periods_per_year=periods_per_year
    )
    
    if net_returns_zero_cost.isna().mean() > 0.05 or net_returns_zero_cost.std() == 0:
        # Model failure - cannot compute break-even
        return None
    
    edge_at_zero = net_returns_zero_cost.mean()
    
    # If edge is negative/zero even at zero cost, break-even is 0.0 (can't pay any cost)
    # This is a valid result, not a model failure
    if edge_at_zero <= 0:
        return 0.0
    
    # Binary search for break-even cost
    low_cost = 0.0
    high_cost = 0.1  # 10% per trade (very high upper bound)
    
    for _ in range(50):  # Max 50 iterations
        mid_cost = (low_cost + high_cost) / 2.0
        
        # Compute net returns at this cost level
        net_returns = compute_net_returns_from_positions(
            gross_returns, positions,
            commission_per_trade=mid_cost,
            half_spread=0.0,  # Only test commission for simplicity
            periods_per_year=periods_per_year
        )
        
        # Check for model failure
        if net_returns.isna().mean() > 0.05 or net_returns.std() == 0:
            # Model failure - force search away from "success"
            edge = -1e9
        else:
            # Break-even is when edge (mean return) goes to zero
            edge = net_returns.mean()
        
        # Check if break-even (edge ≈ 0)
        if abs(edge) < tolerance:
            return mid_cost
        
        if edge > 0:
            # Still profitable, try higher cost
            low_cost = mid_cost
        else:
            # Unprofitable, try lower cost
            high_cost = mid_cost
    
    # Return midpoint as approximation
    return (low_cost + high_cost) / 2.0


def analyze_tradability(gross_returns: pd.Series,
                        signals: pd.Series,
                        positions: Optional[pd.Series] = None,
                        commission_per_trade: float = 0.005,
                        half_spread: float = 0.001,
                        volatility: Optional[pd.Series] = None,
                        volumes: Optional[pd.Series] = None,
                        prices: Optional[pd.Series] = None,
                        vol_impact_coefficient: float = 0.1,
                        vol_impact_coefficient2: float = 0.0001,
                        periods_per_year: int = 252,
                        sharpe_threshold: float = 0.5,
                        use_proper_impact_model: bool = False,
                        impact_model: str = 'almgren_chriss',
                        forward_returns: Optional[pd.Series] = None) -> TradabilityMetrics:
    """
    Comprehensive tradability analysis.
    
    Args:
        gross_returns: Gross strategy returns
        signals: Signal values (for position reconstruction if needed)
        positions: Optional pre-computed positions (if None, reconstructed from signals)
        commission_per_trade: Commission cost per trade
        half_spread: Half of bid-ask spread
        volatility: Optional volatility series
        volumes: Optional volume series
        prices: Optional price series
        vol_impact_coefficient: Volatility impact coefficient
        vol_impact_coefficient2: Volume impact coefficient
        periods_per_year: Number of periods per year
        sharpe_threshold: Sharpe threshold for capacity analysis
    
    Returns:
        TradabilityMetrics object with all metrics
    """
    # Reconstruct positions if not provided
    if positions is None:
        positions = compute_positions_from_returns(gross_returns, signals)
    
    # Compute Information Coefficient if forward returns provided
    if forward_returns is not None:
        ic, ic_p = compute_information_coefficient(signals, forward_returns)
        metrics = TradabilityMetrics()
        metrics.information_coefficient = ic
        if ic is not None:
            from statistical_rigor import compute_ic_with_confidence
            _, ic_ci, _ = compute_ic_with_confidence(signals, forward_returns)
            metrics.ic_ci = ic_ci
    else:
        metrics = TradabilityMetrics()
    
    # Compute gross vs net performance
    # Optionally use proper market impact models
    if use_proper_impact_model and volatility is not None and volumes is not None and prices is not None:
        # Use Almgren-Chriss or Kyle model
        from market_impact import compute_total_market_impact
        market_impact = compute_total_market_impact(
            positions, volatility, volumes, prices,
            model=impact_model,
            participation_rate=0.01,
            periods_per_year=periods_per_year
        )
        market_impact_aligned = market_impact.reindex(gross_returns.index, method='ffill').fillna(0)
        
        # Explicit costs (commission + spread)
        explicit_costs = compute_total_explicit_costs(
            positions, commission_per_trade, half_spread, prices, periods_per_year
        )
        explicit_costs_aligned = explicit_costs.reindex(gross_returns.index, method='ffill').fillna(0)
        
        # Total costs = explicit + market impact
        total_costs = explicit_costs_aligned + market_impact_aligned
        net_returns = gross_returns - total_costs
        
        # Compute metrics
        metrics.gross_metrics = compute_performance_metrics(gross_returns)
        metrics.net_metrics = compute_performance_metrics(net_returns)
        
        # Sharpe with CI
        if metrics.gross_metrics.sharpe_ratio is not None:
            _, gross_ci, _ = compute_sharpe_with_ci(gross_returns, periods_per_year)
            metrics.gross_sharpe_ci = gross_ci
        if metrics.net_metrics.sharpe_ratio is not None:
            _, net_ci, _ = compute_sharpe_with_ci(net_returns, periods_per_year)
            metrics.net_sharpe_ci = net_ci
        
        # Other metrics
        metrics.annual_turnover = compute_annual_turnover(positions, periods_per_year)
        metrics.cost_drag = total_costs.mean() * periods_per_year
        
        if metrics.gross_metrics.hit_rate is not None and metrics.net_metrics.hit_rate is not None:
            metrics.hit_rate_survival = metrics.net_metrics.hit_rate / metrics.gross_metrics.hit_rate
        if metrics.gross_metrics.return_mean is not None and metrics.net_metrics.return_mean is not None:
            if abs(metrics.gross_metrics.return_mean) > 1e-6:
                metrics.pnl_collapse = metrics.net_metrics.return_mean / metrics.gross_metrics.return_mean
        
        # Drawdown analysis
        try:
            from drawdown_analysis import compute_drawdown_metrics
            dd_metrics = compute_drawdown_metrics(gross_returns, net_returns)
            metrics.gross_max_drawdown = dd_metrics['gross_max_drawdown']
            metrics.net_max_drawdown = dd_metrics['net_max_drawdown']
            metrics.drawdown_increase_pct = dd_metrics['drawdown_increase_pct']
            metrics.gross_recovery_time_years = dd_metrics['gross_recovery_time_years']
            metrics.net_recovery_time_years = dd_metrics['net_recovery_time_years']
            metrics.gross_cvar_5pct = dd_metrics['gross_cvar_5pct']
            metrics.net_cvar_5pct = dd_metrics['net_cvar_5pct']
        except Exception:
            pass
    else:
        # Use original method (backward compatible)
        metrics_temp = compute_gross_vs_net_performance(
            gross_returns, positions,
            commission_per_trade, half_spread,
            volatility, volumes, prices,
            vol_impact_coefficient, vol_impact_coefficient2,
            periods_per_year
        )
        # Copy metrics
        metrics.gross_metrics = metrics_temp.gross_metrics
        metrics.net_metrics = metrics_temp.net_metrics
        metrics.annual_turnover = metrics_temp.annual_turnover
        metrics.cost_drag = metrics_temp.cost_drag
        metrics.hit_rate_survival = metrics_temp.hit_rate_survival
        metrics.pnl_collapse = metrics_temp.pnl_collapse
        metrics.gross_sharpe_ci = metrics_temp.gross_sharpe_ci
        metrics.net_sharpe_ci = metrics_temp.net_sharpe_ci
    
    # Break-even cost
    metrics.break_even_cost = compute_break_even_cost(
        gross_returns, positions, periods_per_year
    )
    
    # Capacity analysis
    if volumes is not None:
        capacity_result = estimate_maximum_viable_capital(
            gross_returns, positions, volumes, prices,
            sharpe_threshold=sharpe_threshold,
            periods_per_year=periods_per_year,
            use_sharpe_vs_aum=True,
            volatility=volatility,
            fixed_costs_annual=500_000
        )
        if isinstance(capacity_result, dict):
            metrics.max_viable_capacity = capacity_result.get('max_viable_capacity', 0)
        else:
            metrics.max_viable_capacity = capacity_result if capacity_result else 0
    
    return metrics


def compute_drawdown_sensitivity_to_costs(gross_returns: pd.Series,
                                          positions: pd.Series,
                                          cost_levels: Optional[np.ndarray] = None,
                                          periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute drawdown sensitivity as function of transaction costs.
    
    Args:
        gross_returns: Gross strategy returns
        positions: Series of position values
        cost_levels: Array of cost levels to test (default: 0 to 1%)
        periods_per_year: Number of periods per year
    
    Returns:
        DataFrame with columns: cost_level, max_drawdown, sharpe_ratio
    """
    if cost_levels is None:
        cost_levels = np.linspace(0, 0.01, 50)  # 0 to 1%
    
    results = []
    
    for cost in cost_levels:
        # Compute net returns at this cost level
        net_returns = compute_net_returns_from_positions(
            gross_returns, positions,
            commission_per_trade=cost,
            half_spread=0.0,
            periods_per_year=periods_per_year
        )
        
        # Compute metrics
        metrics = compute_performance_metrics(net_returns)
        
        results.append({
            'cost_level': cost,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'return_mean': metrics.return_mean,
        })
    
    return pd.DataFrame(results)

