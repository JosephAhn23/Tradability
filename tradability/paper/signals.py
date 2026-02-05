"""
Adapter: Tradability signals → target positions.

Uses only data available up to as_of_date (no lookahead).
Output: target dollar value per symbol (signed: long +, short -).
"""

from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# Import from repo root (run from signal-tradability-main)
from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns
from tradability_analysis import compute_positions_from_returns


def _load_prices_multi(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Load close prices for all tickers into a DataFrame. Columns = tickers."""
    from data_utils import load_price_data as load_one

    out = {}
    for t in tickers:
        prices, _ = load_one(t, start_date, end_date)
        if prices is not None and len(prices) > 0:
            out[t] = prices
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def generate_target_positions(
    tickers: List[str],
    signal_name: str,
    as_of_date: datetime,
    lookback_start: Optional[datetime] = None,
    position_sizing: str = "equal_weight",
    fixed_dollar_per_name: Optional[float] = None,
    max_position_pct: float = 0.25,
    equity: float = 100_000.0,
    quantile: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Generate target positions from Tradability signal logic (no lookahead).

    Uses data only up to and including as_of_date. Returns list of (symbol, target_dollar_value).
    target_dollar_value: positive = long, negative = short, 0 = flat.

    Args:
        tickers: Symbols to trade.
        signal_name: Name from signal registry (e.g. 'momentum_12_1').
        as_of_date: Signal and targets are computed as of this date (no future data).
        lookback_start: Start date for history; default ~2 years before as_of_date.
        position_sizing: 'equal_weight' or 'fixed_dollar'.
        fixed_dollar_per_name: Used when position_sizing == 'fixed_dollar'.
        max_position_pct: Cap per-name weight (e.g. 0.25 = 25% of equity per name).
        equity: Portfolio equity for sizing.
        quantile: Signal quantile for long/short split (0.5 = median).

    Returns:
        List of (symbol, target_dollar_value). Not included tickers = flat (0).
    """
    if lookback_start is None:
        from datetime import timedelta
        lookback_start = datetime(as_of_date.year - 2, as_of_date.month, 1)

    # Single-ticker path: existing framework is one series
    if len(tickers) == 1:
        return _targets_single_ticker(
            tickers[0], signal_name, as_of_date, lookback_start,
            position_sizing, fixed_dollar_per_name, max_position_pct, equity, quantile,
        )

    # Multi-ticker: load all, compute signal per ticker, cross-sectional long/short
    prices_df = _load_prices_multi(tickers, lookback_start, as_of_date)
    if prices_df.empty or len(prices_df) < 20:
        return []

    # Latest date in data (must be <= as_of_date)
    last_date = prices_df.index.max()
    if hasattr(last_date, "tz") and last_date.tz is not None:
        last_date = last_date.tz_localize(None)
    if last_date > pd.Timestamp(as_of_date):
        prices_df = prices_df.loc[prices_df.index <= pd.Timestamp(as_of_date)]
        if prices_df.empty:
            return []
        last_date = prices_df.index.max()

    signal_def = get_signal(signal_name)
    # Signal value per ticker on last_date
    signal_values = {}
    for sym in prices_df.columns:
        try:
            p = prices_df[sym].dropna()
            if len(p) < 20:
                continue
            sig = signal_def.compute(p, **signal_def.default_params())
            if sig is not None and last_date in sig.index:
                signal_values[sym] = sig.loc[last_date]
            elif sig is not None and len(sig) > 0:
                # ffill to last date
                sig = sig.reindex(p.index, method="ffill")
                if last_date in sig.index and pd.notna(sig.loc[last_date]):
                    signal_values[sym] = sig.loc[last_date]
        except Exception:
            continue

    if len(signal_values) < 2:
        return _targets_single_ticker(
            tickers[0], signal_name, as_of_date, lookback_start,
            position_sizing, fixed_dollar_per_name, max_position_pct, equity, quantile,
        )

    # Cross-sectional: long top half, short bottom half
    ser = pd.Series(signal_values)
    thresh = ser.quantile(quantile)
    # We want +1 above threshold, -1 below
    weights = np.where(ser > thresh, 1.0, -1.0)
    n = len(weights)
    if position_sizing == "fixed_dollar" and fixed_dollar_per_name is not None:
        dollar_per_side = abs(fixed_dollar_per_name)
    else:
        dollar_per_side = (equity * max_position_pct) / max(n, 1)
    targets = []
    for sym, w in zip(ser.index, weights):
        if w == 0:
            continue
        targets.append((sym, w * dollar_per_side))
    return targets


def _targets_single_ticker(
    ticker: str,
    signal_name: str,
    as_of_date: datetime,
    lookback_start: datetime,
    position_sizing: str,
    fixed_dollar_per_name: Optional[float],
    max_position_pct: float,
    equity: float,
    quantile: float,
) -> List[Tuple[str, float]]:
    """Targets for single ticker: long/short/flat from signal."""
    prices, _ = load_price_data(ticker, lookback_start, as_of_date)
    if prices is None or len(prices) < 20:
        return []

    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    forward_returns = compute_forward_returns(prices)
    aligned_s, aligned_r = align_signals_and_returns(signal_values, forward_returns)
    if aligned_s.empty:
        return []

    # Last date
    last_date = aligned_s.index.max()
    if last_date > pd.Timestamp(as_of_date):
        aligned_s = aligned_s.loc[aligned_s.index <= pd.Timestamp(as_of_date)]
        aligned_r = aligned_r.reindex(aligned_s.index, method="ffill")
    if aligned_s.empty:
        return []
    last_date = aligned_s.index.max()

    gross_returns = compute_returns(aligned_s, aligned_r, quantile=quantile)
    positions = compute_positions_from_returns(gross_returns, aligned_s, quantile=quantile)
    if last_date not in positions.index:
        return []
    pos = positions.loc[last_date]
    if pos == 0:
        return [(ticker, 0.0)]

    dollar = equity * max_position_pct
    if position_sizing == "fixed_dollar" and fixed_dollar_per_name is not None:
        dollar = abs(fixed_dollar_per_name)
    return [(ticker, pos * dollar)]
