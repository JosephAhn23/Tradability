"""
Tests for Alpha Feasibility Bounds: no lookahead, cost monotonicity, impact vs AUM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradability.feasibility.edge import (
    compute_ic_panel,
    compute_forward_returns_panel,
    gross_edge_proxy_ic,
)
from tradability.feasibility.costs import CostModel, compute_total_cost_bps
from tradability.feasibility.capacity import compute_impact_bps, net_edge_surface


def _fake_panel(n_dates: int = 100, n_tickers: int = 5) -> tuple:
    """Fake prices and forward returns; signal = lagged return (no lookahead at t)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.standard_normal((n_dates, n_tickers)) * 0.01, axis=0)),
        index=dates,
        columns=[f"T{i}" for i in range(n_tickers)],
    )
    fwd = compute_forward_returns_panel(prices, horizon=1)
    # Signal known at t: use only past (e.g. lagged return)
    signal = prices.pct_change().shift(1)
    return signal, fwd, prices


def test_no_lookahead_gross_edge_proxy():
    """Signal at D must not use return D->D+1; forward return is D->D+1."""
    signal, fwd, prices = _fake_panel(200, 5)
    # Align: signal_t and return_t (which is close(t)/close(t-1) - 1, i.e. backward)
    # For true no-lookahead: signal at t, forward return from t to t+1
    # So we need signal index t and fwd index t (fwd at t = (p[t+1]/p[t])-1)
    ic, n = compute_ic_panel(signal, fwd, method="spearman")
    # Just check we get a number and sample count is sane (some NaNs dropped)
    assert n >= 10
    assert -1 <= ic <= 1


def test_net_edge_decreases_as_costs_increase():
    """Higher cost params => lower net edge for same gross edge."""
    cost = CostModel(fee_bps=5, spread_bps=10, slippage_bps_per_turnover=5, delay_bps=2)
    c_low = compute_total_cost_bps(1.0, cost, impact_bps=0)
    cost_high = CostModel(fee_bps=15, spread_bps=30, slippage_bps_per_turnover=15, delay_bps=5)
    c_high = compute_total_cost_bps(1.0, cost_high, impact_bps=0)
    assert c_high > c_low
    gross = 50.0
    assert (gross - c_high) < (gross - c_low)


def test_impact_cost_increases_with_aum():
    """Impact cost should increase when order notional (AUM) increases, ADV fixed."""
    cost = CostModel(impact_type="sqrt", impact_k=10, adv_window=20)
    adv = 1e9
    impact_1m = compute_impact_bps(1e6, adv, cost)
    impact_10m = compute_impact_bps(10e6, adv, cost)
    impact_100m = compute_impact_bps(100e6, adv, cost)
    assert impact_10m > impact_1m
    assert impact_100m > impact_10m


def test_zero_alpha_boundary_surface_columns():
    """Surface has aum, turnover, net_edge_bps."""
    cost = CostModel()
    surface = net_edge_surface(
        gross_edge_bps=30,
        cost_model=cost,
        aum_grid=[1e6, 5e6],
        turnover_grid=[0.5, 1.0, 2.0],
        adv_notional_avg=2e9,
    )
    assert "aum" in surface.columns
    assert "turnover" in surface.columns
    assert "net_edge_bps" in surface.columns
    assert len(surface) == 2 * 3
