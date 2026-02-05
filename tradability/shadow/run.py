"""
Shadow trading: forward walk-forward execution simulation.

Usage:
  python -m tradability.shadow.run --config configs/shadow.yaml

No broker. Deterministic fills; full logging; feasibility halt when bounds violated.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from .config import load_config, ShadowConfig
from .signals import generate_target_positions
from .execution import orders_from_targets, current_positions_to_dollars
from .fill_model import load_ohlc, fill_orders_deterministic
from .ledger import ShadowLedger
from .feasibility_check import check_feasibility_halt


def _trading_dates(start: datetime, end: datetime, freq: str) -> pd.DatetimeIndex:
    """Business days between start and end; optionally weekly."""
    idx = pd.date_range(start=start, end=end, freq="B")
    if "week" in freq.lower():
        idx = idx[idx.dayofweek == 0]
    return idx


def run_shadow(config: ShadowConfig, run_dir: str) -> None:
    closes, opens, volumes = load_ohlc(config.tickers, config.start_date, config.end_date)
    if closes.empty or len(closes) < 30:
        raise ValueError("Insufficient OHLC data for shadow run.")

    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
        opens.index = opens.index.tz_localize(None)

    dates = _trading_dates(config.start_date, config.end_date, config.rebalance_frequency)
    dates = dates[dates.isin(closes.index)].intersection(closes.index)
    if len(dates) < 2:
        raise ValueError("Too few trading dates.")

    ledger = ShadowLedger(run_dir)
    positions_shares: Dict[str, float] = {}
    cash = config.initial_equity
    realized_pl = 0.0

    for i, as_of in enumerate(dates):
        as_of = pd.Timestamp(as_of)
        # Prices as of as_of (close for valuation and order sizing)
        row = closes.loc[closes.index <= as_of].iloc[-1] if as_of in closes.index else None
        if row is None:
            continue
        prices_today = row.to_dict()
        equity = cash + sum(positions_shares.get(s, 0) * prices_today.get(s, 0) for s in positions_shares)

        # Target positions (no lookahead)
        targets = generate_target_positions(
            config.tickers,
            config.signal_name,
            as_of.to_pydatetime(),
            equity=equity,
            position_sizing=config.position_sizing,
            fixed_dollar_per_name=config.fixed_dollar_per_name,
            max_position_pct=config.max_position_pct,
        )
        if not targets:
            ledger.snapshot_equity(as_of, equity, cash, realized_pl)
            ledger.snapshot_positions(as_of, dict(positions_shares), prices_today)
            continue

        current_dollars = current_positions_to_dollars(positions_shares, prices_today)
        orders = orders_from_targets(targets, current_dollars, prices_today)
        ledger.log_orders(as_of, orders)

        # Feasibility: optional net_edge from feasibility module; here we use simple halt rules
        halt, reason = check_feasibility_halt(
            config, equity, annual_turnover=0, net_edge_bps=config.min_net_edge_bps
        )
        if halt:
            ledger.log_diagnostic(as_of, "halt", 1, reason)
            break

        # Fill at next open (deterministic)
        next_idx = dates.get_indexer([as_of], method="ffill")[0] + 1
        if next_idx >= len(dates):
            ledger.snapshot_equity(as_of, equity, cash, realized_pl)
            ledger.snapshot_positions(as_of, dict(positions_shares), prices_today)
            continue
        fill_date = dates[next_idx]
        if fill_date not in opens.index:
            ledger.snapshot_equity(as_of, equity, cash, realized_pl)
            ledger.snapshot_positions(as_of, dict(positions_shares), prices_today)
            continue

        opens_day = opens.loc[fill_date].to_dict()
        # Impact: k * sqrt(notional / ADV) in bps (simplified: use recent ADV from closes*volumes)
        impact_per_order = {}
        if not volumes.empty and fill_date in volumes.index and config.impact_k > 0:
            window = min(config.adv_window, len(closes))
            adv = (closes * volumes).rolling(window, min_periods=1).mean()
            if fill_date in adv.index:
                adv_row = adv.loc[fill_date]
                for o in orders:
                    sym = o["symbol"]
                    notional = o.get("notional", 0) or (o.get("qty", 0) * opens_day.get(sym, 0))
                    adv_sym = float(adv_row.get(sym, 1))
                    if adv_sym > 0 and notional > 0:
                        import math
                        impact_bps = config.impact_k * 100 * math.sqrt(notional / adv_sym)
                        impact_per_order[(sym, o["side"])] = impact_bps
        fills = fill_orders_deterministic(
            orders, fill_date, opens,
            spread_bps=config.spread_bps,
            slippage_bps=config.slippage_bps,
            impact_bps_per_order=impact_per_order,
        )
        ledger.log_fills(fills)

        # Update state
        for f in fills:
            sym = f["symbol"]
            side = f["side"]
            qty = f["qty"]
            fill_price = f["fill_price"]
            if side == "buy":
                positions_shares[sym] = positions_shares.get(sym, 0) + qty
                cash -= qty * fill_price
            else:
                positions_shares[sym] = positions_shares.get(sym, 0) - qty
                cash += qty * fill_price
            if positions_shares.get(sym, 0) == 0:
                del positions_shares[sym]

        # Equity at fill date (use close of fill_date for mark)
        if fill_date in closes.index:
            row_fill = closes.loc[fill_date]
            prices_fill = row_fill.to_dict()
            equity = cash + sum(positions_shares.get(s, 0) * prices_fill.get(s, 0) for s in positions_shares)
            ledger.snapshot_equity(fill_date, equity, cash, realized_pl)
            ledger.snapshot_positions(fill_date, dict(positions_shares), prices_fill)

    ledger.write_all()

    # Summary
    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Shadow trading run summary\n\n")
        f.write("**Forward walk-forward execution simulation.** No broker; deterministic fills.\n\n")
        f.write("## Fill assumption\n\n")
        f.write("Orders generated at close(D) are filled at open(D+1). ")
        f.write("Fill price = open(D+1) × (1 ± (spread_bps + slippage_bps)/10000).\n\n")
        f.write("## Feasibility\n\n")
        f.write("Halt when feasibility bounds are violated (config: halt_on_feasibility_violation).\n")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow trading (forward execution simulation)")
    parser.add_argument("--config", default="configs/shadow.yaml")
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
    run_dir = args.run_dir or os.path.join(_REPO_ROOT, "runs", "shadow", run_id)

    run_shadow(config, run_dir)
    print(f"Shadow run written to: {run_dir}")


if __name__ == "__main__":
    main()
