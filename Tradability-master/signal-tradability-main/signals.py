"""
Signal Definitions Module

Simple, well-known signals for decay analysis.
No fancy ML - that's the point.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class SignalDefinition:
    """Base class for signal definitions."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        """
        Compute signal values.
        
        Returns:
            pd.Series: Signal values (long/short signals, typically normalized)
        """
        raise NotImplementedError
    
    def default_params(self) -> Dict:
        """Return default parameters for the signal."""
        raise NotImplementedError


class Momentum12_1(SignalDefinition):
    """
    12-1 Month Momentum Signal
    
    Classic momentum: buy winners, sell losers.
    Returns = (Price(t-1) / Price(t-13)) - 1
    """
    
    def __init__(self):
        super().__init__(
            name="12-1 Momentum",
            description="12-month return lagged 1 month (Jegadeesh-Titman style)"
        )
    
    def default_params(self) -> Dict:
        return {
            'lookback_months': 12,
            'skip_months': 1
        }
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        lookback = params.get('lookback_months', 12)
        skip = params.get('skip_months', 1)
        
        # Convert to monthly if needed (assuming daily prices)
        if prices.index.freq is None or prices.index.freq.name.startswith('D'):
            monthly_prices = prices.resample('M').last()
        else:
            monthly_prices = prices
        
        # Compute momentum: return from (t-lookback-skip) to (t-skip)
        lookback_periods = lookback + skip
        momentum = (monthly_prices.shift(skip) / monthly_prices.shift(lookback_periods)) - 1
        
        # Expand back to original frequency
        if prices.index.freq is None or prices.index.freq.name.startswith('D'):
            momentum = momentum.reindex(prices.index, method='ffill')
        
        return momentum


class ShortTermMeanReversion(SignalDefinition):
    """
    Short-Term Mean Reversion Signal
    
    Reversion to recent mean - contrarian strategy.
    Signal = (Price - MA(short)) / MA(long)
    """
    
    def __init__(self):
        super().__init__(
            name="Short-Term Mean Reversion",
            description="Deviation from short-term moving average"
        )
    
    def default_params(self) -> Dict:
        return {
            'short_window': 5,  # days
            'long_window': 20   # days
        }
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        short_window = params.get('short_window', 5)
        long_window = params.get('long_window', 20)
        
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        
        # Negative signal: price below short MA relative to long MA = buy signal
        signal = -(prices - ma_short) / ma_long
        
        return signal


class VolatilityBreakout(SignalDefinition):
    """
    Volatility Breakout Signal
    
    Buy when price breaks above recent volatility band.
    Band = MA ± k * rolling_std
    """
    
    def __init__(self):
        super().__init__(
            name="Volatility Breakout",
            description="Price breakout beyond volatility-adjusted bands"
        )
    
    def default_params(self) -> Dict:
        return {
            'ma_window': 20,  # days
            'volatility_window': 20,  # days
            'std_multiplier': 2.0
        }
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        ma_window = params.get('ma_window', 20)
        vol_window = params.get('volatility_window', 20)
        k = params.get('std_multiplier', 2.0)
        
        ma = prices.rolling(window=ma_window).mean()
        rolling_std = prices.rolling(window=vol_window).std()
        
        upper_band = ma + k * rolling_std
        lower_band = ma - k * rolling_std
        
        # Signal: positive when above upper band, negative when below lower
        signal = np.where(prices > upper_band, (prices - upper_band) / rolling_std,
                 np.where(prices < lower_band, (prices - lower_band) / rolling_std, 0))
        
        return pd.Series(signal, index=prices.index)


class MovingAverageCrossover(SignalDefinition):
    """
    Moving Average Crossover Signal
    
    Golden/Death cross: when short MA crosses above/below long MA.
    """
    
    def __init__(self):
        super().__init__(
            name="MA Crossover",
            description="Short-term MA crossing long-term MA"
        )
    
    def default_params(self) -> Dict:
        return {
            'short_window': 50,  # days
            'long_window': 200   # days
        }
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        short_window = params.get('short_window', 50)
        long_window = params.get('long_window', 200)
        
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        
        # Signal: difference between MAs (positive when short > long)
        signal = (ma_short - ma_long) / ma_long
        
        return signal


class ValueFactor(SignalDefinition):
    """
    Value Factor (Book-to-Market Proxy)
    
    Note: Requires fundamental data. This is a placeholder structure.
    For price-only analysis, we use price-to-book proxies or earnings yield.
    """
    
    def __init__(self):
        super().__init__(
            name="Value Factor",
            description="Book-to-market ratio (requires fundamental data)"
        )
    
    def default_params(self) -> Dict:
        return {
            'earnings_window': 252  # Use trailing earnings as proxy
        }
    
    def compute(self, prices: pd.Series, **params) -> pd.Series:
        """
        Simplified value signal using price-to-earnings proxy.
        In real implementation, would use actual book value data.
        """
        # This is a simplified version - real implementation needs fundamental data
        # For now, we'll use inverse price momentum as a proxy
        # (falling prices = potentially undervalued)
        
        lookback = params.get('earnings_window', 252)
        price_ratio = prices / prices.shift(lookback)
        
        # Value signal: lower price ratio = higher value signal
        signal = -price_ratio
        
        return signal


# Signal registry
# TEST 1: mean_reversion DELETED (not rejected, deleted as if it never existed)
SIGNAL_REGISTRY = {
    'momentum_12_1': Momentum12_1(),
    # 'mean_reversion': ShortTermMeanReversion(),  # DELETED per WAR TEST II
    'volatility_breakout': VolatilityBreakout(),
    'ma_crossover': MovingAverageCrossover(),
    'value': ValueFactor()
}


def get_signal(name: str) -> SignalDefinition:
    """Get signal definition by name."""
    if name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal: {name}. Available: {list(SIGNAL_REGISTRY.keys())}")
    return SIGNAL_REGISTRY[name]


def list_signals() -> Dict[str, str]:
    """List all available signals with descriptions."""
    return {name: sig.description for name, sig in SIGNAL_REGISTRY.items()}

