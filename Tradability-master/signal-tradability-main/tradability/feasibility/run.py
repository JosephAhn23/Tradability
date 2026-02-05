"""
Alpha Feasibility Bounds — main entry and CLI.

Usage:
  python -m tradability.feasibility.run --config configs/feasibility_example.yaml
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from .config import load_config, FeasibilityConfig
from .edge import compute_gross_edge_proxy, compute_signal_scores
from .costs import CostModel, compute_total_cost_bps
from .capacity import (
    compute_adv_notional,
    net_edge_surface,
    zero_alpha_boundary,
)
from .regime import compute_regimes, net_edge_by_regime
from .report import run_report


def load_prices_volumes(
    tickers: list,
    start_date: datetime,
    end_date: datetime,
) -> tuple:
    """Load OHLCV into (prices DataFrame, volumes DataFrame)."""
    from data_utils import load_price_data

    prices = {}
    volumes = {}
    for t in tickers:
        p, v = load_price_data(t, start_date, end_date)
        if p is not None and len(p) > 0:
            prices[t] = p
            if v is not None:
                volumes[t] = v
    if not prices:
        return pd.DataFrame(), pd.DataFrame()
    prices_df = pd.DataFrame(prices)
    vol_df = pd.DataFrame(volumes) if volumes else pd.DataFrame(index=prices_df.index)
    if not vol_df.empty and vol_df.index.intersection(prices_df.index).empty:
        vol_df = vol_df.reindex(prices_df.index).fillna(0)
    return prices_df, vol_df


def run_feasibility(config: FeasibilityConfig, run_dir: str) -> None:
    """Run full feasibility analysis and write outputs to run_dir."""
    prices, volumes = load_prices_volumes(
        config.tickers,
        config.start_date,
        config.end_date,
    )
    if prices.empty or len(prices) < 50:
        raise ValueError("Insufficient price data for feasibility run.")

    # Timezone-naive
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if not volumes.empty and volumes.index.tz is not None:
        volumes.index = volumes.index.tz_localize(None)

    # Gross edge proxy (no lookahead): signal at D, return D->D+1
    proxy = config.gross_edge_proxy or {}
    signal_name = proxy.get("signal_name", "momentum_12_1")
    signal_scores = compute_signal_scores(prices, volumes, signal_name)
    config_dict = {"gross_edge_proxy": config.gross_edge_proxy}
    gross_edge_bps, info = compute_gross_edge_proxy(
        prices, volumes, config_dict, signal_scores=signal_scores
    )

    cost_model = CostModel.from_dict(config.cost_model)
    cap = config.capacity
    aum_grid = config.aum_grid or cap.get("aum_grid") or [
        100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000
    ]
    turnover_grid = config.turnover_grid_pct or np.linspace(0, 4, 21).tolist()  # 0 to 400%
    adv_window = cap.get("adv_window", 20)
    participation_max = cap.get("participation_rate_max", 0.01)

    adv_series = compute_adv_notional(prices, volumes, adv_window)
    adv_avg = float(adv_series.mean()) if not adv_series.empty else 1e9

    surface = net_edge_surface(
        gross_edge_bps,
        cost_model,
        aum_grid,
        turnover_grid,
        adv_avg,
        participation_rate_max=participation_max,
    )
    zero_bd = zero_alpha_boundary(surface, aum_grid)

    # Regimes
    returns = prices.pct_change().dropna(how="all")
    regimes_ser = compute_regimes(
        returns,
        adv_series.reindex(returns.index).ffill(),
        vol_bins=config.regimes.get("vol_bins", 3),
        liquidity_bins=config.regimes.get("liquidity_bins", 3),
        vol_window=adv_window,
    )
    regime_table = net_edge_by_regime(gross_edge_bps, cost_model, regimes_ser, turnover=1.0)

    run_report(
        run_dir,
        surface,
        zero_bd,
        regime_table,
        gross_edge_bps,
        info,
        cost_model,
        aum_grid,
        turnover_grid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Feasibility Bounds")
    parser.add_argument("--config", default="configs/feasibility_example.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(_REPO_ROOT, "runs", "feasibility", run_id)

    run_feasibility(config, run_dir)
    print(f"Feasibility run written to: {run_dir}")


if __name__ == "__main__":
    main()
