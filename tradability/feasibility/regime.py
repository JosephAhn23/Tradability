"""
Regime conditioning: vol and liquidity bins, net edge by regime.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .costs import CostModel, compute_total_cost_bps
from .capacity import compute_impact_bps


def volatility_proxy(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling annualized vol (std of returns) per date, averaged across tickers."""
    if returns.empty:
        return pd.Series(dtype=float)
    vol = returns.rolling(window, min_periods=1).std() * np.sqrt(252)
    return vol.mean(axis=1)


def liquidity_proxy(adv_notional: pd.Series) -> pd.Series:
    """Liquidity = ADV notional (already a series)."""
    return adv_notional


def compute_regimes(
    returns: pd.DataFrame,
    adv_notional: pd.Series,
    vol_bins: int = 3,
    liquidity_bins: int = 3,
    vol_window: int = 20,
) -> pd.Series:
    """
    Assign each date to a regime: (vol_bin, liq_bin) -> regime_id.
    Returns Series index=date, value=regime_id (string e.g. "vol0_liq1").
    """
    vol = volatility_proxy(returns, window=vol_window)
    liq = liquidity_proxy(adv_notional)
    idx = vol.index.intersection(liq.index)
    vol = vol.reindex(idx).ffill().dropna()
    liq = liq.reindex(idx).ffill().dropna()
    idx = vol.index.intersection(liq.index)
    vol = vol.loc[idx]
    liq = liq.loc[idx]
    try:
        vol_q = pd.qcut(vol.rank(method="first"), vol_bins, labels=False, duplicates="drop")
        liq_q = pd.qcut(liq.rank(method="first"), liquidity_bins, labels=False, duplicates="drop")
    except Exception:
        vol_q = pd.Series(0, index=vol.index)
        liq_q = pd.Series(0, index=liq.index)
    regime = vol_q.astype(str) + "_vol_" + liq_q.astype(str) + "_liq"
    return regime


def net_edge_by_regime(
    gross_edge_bps: float,
    cost_model: CostModel,
    regimes: pd.Series,
    turnover: float = 1.0,
    impact_bps_per_regime: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Net edge (bps) per regime. Same gross edge; cost can vary by regime if impact_bps_per_regime given.
    """
    if impact_bps_per_regime is None:
        impact_bps_per_regime = {}
    rows = []
    for regime_id in regimes.dropna().unique():
        imp = impact_bps_per_regime.get(regime_id, 0.0)
        total_cost = compute_total_cost_bps(turnover, cost_model, impact_bps=imp)
        net_bps = gross_edge_bps - total_cost
        n_days = (regimes == regime_id).sum()
        rows.append({"regime_id": regime_id, "net_edge_bps": net_bps, "sample_days": int(n_days)})
    return pd.DataFrame(rows)
