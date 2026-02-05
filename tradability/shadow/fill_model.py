"""
Deterministic fill model for shadow trading.

Assumption (documented): orders generated at close(D) are filled at open(D+1).
Fill price = open(D+1) * (1 + spread_side + slippage_side) in bps.
No partial fills; no rejections unless feasibility halt.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_ohlc(
    tickers: List[str],
    start: datetime,
    end: datetime,
) -> tuple:
    """Load OHLCV; return (prices_df close, opens_df, volumes_df)."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from data_utils import load_price_data

    closes = {}
    opens = {}
    volumes = {}
    for t in tickers:
        try:
            import yfinance as yf
            hist = yf.Ticker(t).history(start=start, end=end)
            if hist is None or len(hist) < 2:
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            closes[t] = hist["Close"]
            opens[t] = hist["Open"]
            volumes[t] = hist["Volume"]
        except Exception:
            continue
    if not closes:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(closes), pd.DataFrame(opens), pd.DataFrame(volumes)


def fill_orders_deterministic(
    orders: List[Dict[str, Any]],
    fill_date: datetime,
    opens: pd.DataFrame,
    spread_bps: float = 10,
    slippage_bps: float = 5,
    impact_bps_per_order: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic fill at fill_date open.
    Fill price = open * (1 + (spread_bps + slippage_bps + impact_bps)/10000) for buy,
                 open * (1 - (...)) for sell.
    impact_bps_per_order: optional dict order_key -> impact_bps (e.g. from sqrt(notional/ADV)).
    Returns list of {symbol, side, qty, fill_price, fill_date, spread_bps, slippage_bps, impact_bps}.
    """
    impact_bps_per_order = impact_bps_per_order or {}
    fills = []
    for o in orders:
        sym = o["symbol"]
        side = o["side"]
        qty = o["qty"]
        key = (sym, side)
        impact_bps = impact_bps_per_order.get(key) or impact_bps_per_order.get(sym) or 0
        if sym not in opens.columns or fill_date not in opens.index:
            continue
        open_px = float(opens.loc[fill_date, sym])
        if np.isnan(open_px) or open_px <= 0:
            continue
        total_bps = (spread_bps + slippage_bps + impact_bps) / 10_000
        if side == "buy":
            fill_price = open_px * (1 + total_bps)
        else:
            fill_price = open_px * (1 - total_bps)
        fills.append({
            "symbol": sym,
            "side": side,
            "qty": qty,
            "fill_price": fill_price,
            "fill_date": fill_date,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "impact_bps": impact_bps,
        })
    return fills
