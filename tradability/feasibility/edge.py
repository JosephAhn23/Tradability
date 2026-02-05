"""
Gross edge proxy: conservative expected return bound from IC or return prediction.

No lookahead: signal at date D uses only data up to and including D;
forward return is from close(D) to close(D+1).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Conservative scaling: expected return ≈ IC * vol * scale. Scale < 1 to bound upward bias.
# See report: "IC-to-return scaling assumption".
IC_TO_RETURN_SCALE = 0.5


def _ensure_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def compute_signal_scores(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame],
    signal_name: str,
) -> pd.DataFrame:
    """
    Compute signal scores per ticker per date using repo's signal definitions.
    No lookahead: at each date only past/current data is used.
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from signals import get_signal

    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    sig_def = get_signal(signal_name)
    params = sig_def.default_params()
    for col in prices.columns:
        ser = prices[col].dropna()
        if len(ser) < 20:
            continue
        try:
            s = sig_def.compute(ser, **params)
            if s is not None:
                out[col] = s.reindex(out.index).values
        except Exception:
            continue
    return out


def compute_forward_returns_panel(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Forward returns from close(t) to close(t+horizon). No lookahead at t."""
    fwd = (prices.shift(-horizon) / prices) - 1
    return fwd


def compute_ic_panel(
    signal_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> Tuple[float, int]:
    """
    Information coefficient over the panel: (signal at D, return D->D+1).
    Aligns on index; drops NaN. Returns (IC, sample_count).
    """
    from scipy.stats import spearmanr, pearsonr

    idx = signal_scores.index.intersection(forward_returns.index)
    s = signal_scores.loc[idx].stack()
    r = forward_returns.loc[idx].stack()
    df = pd.DataFrame({"signal": s, "return": r}).dropna()
    if len(df) < 10:
        return 0.0, 0
    if method == "spearman":
        ic, _ = spearmanr(df["signal"], df["return"])
    else:
        ic, _ = pearsonr(df["signal"], df["return"])
    ic = 0.0 if np.isnan(ic) else float(ic)
    return ic, len(df)


def gross_edge_proxy_ic(
    signal_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    next_return_vol: Optional[float] = None,
    scale: float = IC_TO_RETURN_SCALE,
    method: str = "spearman",
) -> Tuple[float, float, int]:
    """
    Conservative gross edge proxy from IC.
    expected_return_bps ≈ IC * next_return_vol_bps * scale.
    If next_return_vol not provided, use empirical vol of forward_returns (annualized to daily bps).
    Returns (gross_edge_bps, ic, sample_count).
    """
    ic, n = compute_ic_panel(signal_scores, forward_returns, method=method)
    if n == 0:
        return 0.0, 0.0, 0
    if next_return_vol is None:
        ret_ser = forward_returns.stack().dropna()
        if len(ret_ser) < 2:
            return 0.0, ic, n
        # Daily vol in bps; we use as "typical next-period vol"
        next_return_vol = ret_ser.std() * 10_000  # bps
    else:
        next_return_vol = float(next_return_vol)
    # Conservative: expected return in bps
    gross_edge_bps = ic * next_return_vol * scale
    return gross_edge_bps, ic, n


def compute_gross_edge_proxy(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame],
    config: Dict[str, Any],
    signal_scores: Optional[pd.DataFrame] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute gross edge proxy from config.
    Returns (gross_edge_bps, info_dict) with keys: ic, sample_count, next_return_vol, scale.
    """
    proxy = config.get("gross_edge_proxy") or {}
    ptype = (proxy.get("type") or "ic").lower()
    scale = float(proxy.get("scale", IC_TO_RETURN_SCALE))

    if signal_scores is None:
        signal_name = proxy.get("signal_name", "momentum_12_1")
        signal_scores = compute_signal_scores(prices, volumes, signal_name)
    signal_scores = _ensure_tz_naive(signal_scores)
    prices = _ensure_tz_naive(prices)
    forward_returns = compute_forward_returns_panel(prices, horizon=1)
    forward_returns = _ensure_tz_naive(forward_returns)

    if ptype == "ic":
        next_vol = proxy.get("next_return_vol_bps")
        gross_bps, ic, n = gross_edge_proxy_ic(
            signal_scores, forward_returns,
            next_return_vol=next_vol,
            scale=scale,
            method=proxy.get("method", "spearman"),
        )
        return gross_bps, {
            "ic": ic,
            "sample_count": n,
            "scale": scale,
            "type": "ic",
        }

    # Placeholder for return_prediction (same as IC for now if no model)
    if ptype == "return_prediction":
        gross_bps, ic, n = gross_edge_proxy_ic(
            signal_scores, forward_returns, scale=scale,
        )
        return gross_bps, {"ic": ic, "sample_count": n, "scale": scale, "type": "return_prediction"}
    gross_bps, ic, n = gross_edge_proxy_ic(signal_scores, forward_returns, scale=scale)
    return gross_bps, {"ic": ic, "sample_count": n, "scale": scale, "type": "ic"}
