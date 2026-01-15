"""
Data Utilities Module

Handles data loading, cleaning, and preparation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_price_data(ticker: str, start_date: datetime, end_date: datetime,
                   freq: str = 'D') -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Load price and volume data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, 'M' for monthly)
    
    Returns:
        Tuple of (prices, volumes) - both with timezone-naive index
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) == 0:
            print(f"Warning: No data for {ticker}")
            return pd.Series(dtype=float), None
        
        prices = data['Close']
        volumes = data['Volume']
        
        # Remove timezone info to avoid comparison issues
        # Convert timezone-aware to naive by converting to UTC first, then removing timezone
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert('UTC').tz_localize(None)
        if volumes is not None and volumes.index.tz is not None:
            volumes.index = volumes.index.tz_convert('UTC').tz_localize(None)
        
        # Resample if needed
        if freq == 'M':
            prices = prices.resample('M').last()
            volumes = volumes.resample('M').last()
        
        return prices, volumes
    
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.Series(dtype=float), None


def compute_forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Compute forward-looking returns for signal evaluation.
    
    Args:
        prices: Price series
        horizon: Forward horizon (default 1 period)
    
    Returns:
        Forward returns aligned with original index (timezone-naive)
    """
    forward_prices = prices.shift(-horizon)
    forward_returns = (forward_prices / prices) - 1
    
    # Ensure index is timezone-naive (in case prices had timezone)
    if forward_returns.index.tz is not None:
        forward_returns.index = forward_returns.index.tz_convert('UTC').tz_localize(None)
    
    return forward_returns


def align_signals_and_returns(signals: pd.Series, forward_returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align signals with forward returns, handling timing properly.
    
    Args:
        signals: Signal values
        forward_returns: Forward-looking returns
    
    Returns:
        Tuple of (aligned_signals, aligned_returns) - both with timezone-naive indices
    """
    # Ensure both indices are timezone-naive
    if signals.index.tz is not None:
        signals = signals.copy()
        signals.index = signals.index.tz_convert('UTC').tz_localize(None)
    if forward_returns.index.tz is not None:
        forward_returns = forward_returns.copy()
        forward_returns.index = forward_returns.index.tz_convert('UTC').tz_localize(None)
    
    # Find common index
    common_idx = signals.index.intersection(forward_returns.index)
    
    aligned_signals = signals.loc[common_idx]
    aligned_returns = forward_returns.loc[common_idx]
    
    # Drop any remaining NaNs
    valid_mask = aligned_signals.notna() & aligned_returns.notna()
    
    return aligned_signals[valid_mask], aligned_returns[valid_mask]


def get_sp500_constituents(date: datetime) -> list:
    """
    Get S&P 500 constituents as of a given date.
    
    Note: This is a simplified version. In practice, you'd want to use
    actual historical constituent lists.
    
    Returns:
        List of ticker symbols
    """
    # For demonstration, return a sample of large-cap stocks
    # In production, use actual S&P 500 historical data
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX',
        'BAC', 'XOM', 'JPM', 'CVX', 'ABBV', 'PFE', 'KO', 'AVGO', 'COST'
    ]
    
    return sample_tickers


def create_portfolio_returns(prices_dict: dict, weights: Optional[dict] = None) -> pd.Series:
    """
    Create equal-weighted or custom-weighted portfolio returns.
    
    Args:
        prices_dict: Dictionary mapping ticker to price series
        weights: Optional dictionary of weights (default: equal-weighted)
    
    Returns:
        Portfolio returns series
    """
    returns_dict = {}
    for ticker, prices in prices_dict.items():
        returns_dict[ticker] = prices.pct_change()
    
    returns_df = pd.DataFrame(returns_dict)
    
    if weights is None:
        # Equal-weighted
        portfolio_returns = returns_df.mean(axis=1)
    else:
        # Weighted
        weights_series = pd.Series(weights)
        portfolio_returns = (returns_df * weights_series).sum(axis=1)
    
    return portfolio_returns

